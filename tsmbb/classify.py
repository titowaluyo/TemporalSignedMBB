"""Subject-level networks, features and classification benchmarks (Sections 2.8-2.10)."""
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from . import preprocess as PP
from . import te as TE
from . import topology as TP
from . import evaluate as EV

LAM, GAM = 0.1, 0.3


# ---------------------------------------------------------------- partitions
def partitions(sim, seed=42):
    subj = sim["subjects"]
    ids = np.array([s["subject"] for s in subj])
    held = np.array([s["cohort"] == "SC2" for s in subj])
    rest = ids[~held]
    strat = np.array([f'{s["group"]}_{s["cohort"]}_{s["traj"]}' for s in subj])[~held]
    tr, te_ = train_test_split(rest, test_size=0.2, stratify=strat, random_state=seed)
    return np.sort(tr), np.sort(te_), ids[held]


def xgb_fit(X, y, seed=42):
    """Gradient-boosted trees with depth tuned by five-fold CV inside the training data."""
    best, best_d = -1, 2
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for d in (2, 3, 4):
        aucs = []
        for a, b in skf.split(X, y):
            m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=d, subsample=0.8,
                                  colsample_bytree=0.5, random_state=seed, n_jobs=2, verbosity=0)
            m.fit(X[a], y[a]); aucs.append(roc_auc_score(y[b], m.predict_proba(X[b])[:, 1]))
        if np.mean(aucs) > best:
            best, best_d = np.mean(aucs), d
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=best_d, subsample=0.8,
                          colsample_bytree=0.5, random_state=seed, n_jobs=2, verbosity=0)
    m.fit(X, y)
    return m


def rf_fit(X, y, seed=42):
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2).fit(X, y)


# ---------------------------------------------------------------- subject statistics
class SubjectData:
    def __init__(self, sim, pp, windows=None):
        self.sim, self.pp = sim, pp
        self.Zd = PP.demean_by_subject(pp, windows)
        self.N = self.Zd.shape[1]
        self.subj = {s["subject"]: s for s in sim["subjects"]}
        self.gram, self.corr = {}, {}
        for w in (windows or ("T1", "T2", "T3")):
            X, Y, own = PP.transitions(pp, self.Zd, list(self.subj), w)
            g = {}
            for q in np.unique(own):
                m = own == q
                g[int(q)] = TE.Gram(X[m], Y[m])
            self.gram[w] = g
            sel = pp["window"] == w
            Zw, sw = self.Zd[sel], pp["subject"][sel]
            c = {}
            for q in np.unique(sw):
                z = Zw[sw == q]
                c[int(q)] = (z.T @ z, z.sum(0), len(z))
            self.corr[w] = c

    def group_gram(self, ids, w):
        g = TE.Gram(N=self.N)
        for q in ids:
            if q in self.gram[w]:
                g = g + self.gram[w][q]
        return g

    def mean_feature(self, ids, w, M):
        sel = self.pp["window"] == w
        out = []
        for q in ids:
            m = sel & (self.pp["subject"] == q)
            out.append(M[m].mean(0) if m.any() else np.full(M.shape[1], np.nan))
        return np.array(out)


def _corr_from(S, s, n):
    mu = s / n
    C = S / n - np.outer(mu, mu)
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


def lioness_subjects(D, ids_ref, ids, w):
    """LIONESS lagged-coefficient matrices and lag-0 correlation matrices for subjects `ids`."""
    ids_ref = [q for q in ids_ref if int(q) in D.corr[w]]   # (revision) skip subjects without visits
    gR = D.group_gram(ids_ref, w)
    bR = gR.coef()
    nR = len(ids_ref)
    ref_set = set(int(x) for x in ids_ref)
    Ssum = sum(D.corr[w][q][0] for q in ids_ref); ssum = sum(D.corr[w][q][1] for q in ids_ref)
    nsum = sum(D.corr[w][q][2] for q in ids_ref)
    rR = _corr_from(Ssum, ssum, nsum)
    coefs, cors = [], []
    for q in ids:
        q = int(q)
        gq = D.gram[w].get(q, TE.Gram(N=D.N))
        coefs.append(TE.lioness_coef(gR, bR, nR, gq, q in ref_set))
        S, s, n = D.corr[w].get(q, (np.zeros((D.N, D.N)), np.zeros(D.N), 0))
        if q in ref_set:
            r_wo = _corr_from(Ssum - S, ssum - s, nsum - n)
            cors.append(nR * (rR - r_wo) + r_wo)
        else:
            r_w = _corr_from(Ssum + S, ssum + s, nsum + n)
            cors.append((nR + 1) * (r_w - rR) + rR)
    return np.array(coefs), np.array(cors)


def network_block(C, E, h1_ref, motif_inst):
    """Feature blocks for one subject and window from its LIONESS coefficients C."""
    A = np.where(E, C, 0.0)
    te_f = C[E]
    ps = TP.perturbation_scores(A)
    h1, tot, phi = TP.persistence(A, cocycles=True)
    w2 = TP.wasserstein2(h1, h1_ref)
    mot = np.array([np.prod([A[a, b] for a, b in arcs]) for arcs in motif_inst]) if motif_inst else np.zeros(0)
    return dict(TE=te_f, SPTS=ps, TDA=np.concatenate([phi, [tot, w2]]), MOTIF=mot, A=A, phi=phi)


def disease_motifs(A_dis, A_hc, rng, n_null=100, q=0.10):
    """Arcs of instances of signed classes enriched in A_dis but not in A_hc (training networks)."""
    S_d, S_h = np.sign(A_dis).astype(int), np.sign(A_hc).astype(int)
    ed = TP.motif_enrichment(S_d, n_null, rng); eh = TP.motif_enrichment(S_h, n_null, rng)
    keys = list(ed["p"]); rej = TE.bh([ed["p"][k] for k in keys], q)[0] if keys else []
    hk = list(eh["p"]); hrej = TE.bh([eh["p"][k] for k in hk], q)[0] if hk else []
    hc_enr = {k for k, r in zip(hk, hrej) if r}
    inst = []
    for k, r in zip(keys, rej):
        if r and k not in hc_enr:
            for t in ed["instances"][k]:
                inst.append([(a, b) for a in t for b in t if a != b and S_d[a, b] != 0])
    return inst


def stack(blocks, keys):
    return np.array([np.concatenate([b[k] for k in keys]) for b in blocks])


# ---------------------------------------------------------------- main benchmark
def run(sim, pp, seed=42, full=True):
    rng = np.random.default_rng(seed)
    D = SubjectData(sim, pp, ("T1",))
    tr, te_int, held = partitions(sim, seed)
    subj = D.subj
    lab = lambda ids: np.array([1 if subj[q]["group"] == "MDD" else 0 for q in ids])
    traj = lambda q: subj[q]["traj"]
    out = dict(n_train=len(tr), n_test=len(te_int), n_heldout=len(held))
    allids = np.concatenate([tr, te_int, held])

    def gnet(ids, w):
        return TE.cte_from_gram(D.group_gram(ids, w))

    tr_hc = [q for q in tr if traj(q) == "HC"]; tr_mdd = [q for q in tr if traj(q) != "HC"]
    tr_R = [q for q in tr if traj(q) == "R"]; tr_NR = [q for q in tr if traj(q) == "NR"]
    n_hc1, n_md1 = gnet(tr_hc, "T1"), gnet(tr_mdd, "T1")
    E_cc = n_hc1["R"] | n_md1["R"]
    h1_ref = TP.persistence(n_hc1["A"])[0]
    inst = disease_motifs(n_md1["A"], n_hc1["A"], rng) if full else []
    out["n_motif_instances_cc"] = len(inst)

    # ---------------- case-control (T1)
    C1, R1 = lioness_subjects(D, tr, allids, "T1")
    blocks = [network_block(C, E_cc, h1_ref, inst) for C in C1]
    idx = {q: i for i, q in enumerate(allids)}
    iu = np.where(np.triu(E_cc | E_cc.T, 1))
    static_s = np.array([R[iu] for R in R1]); static_u = np.abs(static_s)
    M = dict(ALL=stack(blocks, ["TE", "SPTS", "TDA", "MOTIF"]))
    for drop in ("TE", "SPTS", "TDA", "MOTIF"):
        M["no_" + drop] = stack(blocks, [k for k in ("TE", "SPTS", "TDA", "MOTIF") if k != drop])
    genus = D.mean_feature(allids, "T1", pp["Z"][:, :46])
    asv = D.mean_feature(allids, "T1", pp["asv_clr"])
    shan = D.mean_feature(allids, "T1", pp["shannon"][:, None])[:, 0]
    fpra = genus[:, 0]
    sel = lambda ids: np.array([idx[q] for q in ids])
    itr, ite, iho = sel(tr), sel(te_int), sel(held)
    ytr, yte, yho = lab(tr), lab(te_int), lab(held)
    scores = {}

    def eval_model(name, Xall, learner="xgb"):
        m = xgb_fit(Xall[itr], ytr, seed) if learner == "xgb" else rf_fit(Xall[itr], ytr, seed)
        scores[name] = (m.predict_proba(Xall[ite])[:, 1], m.predict_proba(Xall[iho])[:, 1])
        return m

    full_model = eval_model("network", M["ALL"])
    if full:
        for k in ("no_TE", "no_SPTS", "no_TDA", "no_MOTIF"):
            if M[k].shape[1]:
                eval_model(k, M[k])
        eval_model("static_unsigned", static_u)
        eval_model("static_signed", static_s)
        eval_model("network+asv", np.hstack([M["ALL"], asv]))
        scores["shannon"] = (-shan[ite], -shan[iho])
        scores["fprau"] = (-fpra[ite], -fpra[iho])
    eval_model("rf_asv", asv, "rf")
    if not full:
        eval_model("network+asv", np.hstack([M["ALL"], asv]))
    cc = {}
    for k, (s_te, s_ho) in scores.items():
        cc[k] = dict(test=EV.summary(yte, s_te), heldout=EV.auc_ci(yho, s_ho))
        if k != "network":
            cc[k]["diff_vs_network"] = EV.delong_paired(yte, scores["network"][0], s_te)
    if full:
        contrib = full_model.get_booster().predict(xgb.DMatrix(M["ALL"][ite]), pred_contribs=True)[:, :-1]
        sizes = [len(blocks[0][k]) for k in ("TE", "SPTS", "TDA", "MOTIF")]
        imp = np.abs(contrib).sum(0); cuts = np.cumsum([0] + sizes)
        tot = imp.sum()
        cc["shap_share"] = {k: float(imp[cuts[i]:cuts[i + 1]].sum() / tot)
                            for i, k in enumerate(("TE", "SPTS", "TDA", "MOTIF"))}
        cc["roc_scores"] = {k: dict(y=yte.tolist(), s=v[0].tolist()) for k, v in scores.items()}
    out["case_control"] = cc

    # ---------------- response (MDD only; T1 and T2; subject means from T1 and T2 only)
    D = SubjectData(sim, pp, ("T1", "T2"))
    mdd = lambda ids: [q for q in ids if traj(q) != "HC"]
    rtr, rte, rho = mdd(tr), mdd(te_int), mdd(held)
    pooled = rte + rho
    yr = lambda ids: np.array([1 if traj(q) == "R" else 0 for q in ids])
    E_rp = (gnet(tr_R, "T1")["R"] | gnet(tr_NR, "T1")["R"] | gnet(tr_R, "T2")["R"] | gnet(tr_NR, "T2")["R"])
    h1_ref2 = TP.persistence(gnet(tr_hc, "T2")["A"])[0]
    ids_r = rtr + pooled
    C1r, _ = lioness_subjects(D, tr, ids_r, "T1")
    C2r, R2r = lioness_subjects(D, tr, ids_r, "T2")
    b1 = [network_block(C, E_rp, h1_ref, []) for C in C1r]
    b2 = [network_block(C, E_rp, h1_ref2, []) for C in C2r]

    def resp_features(lam=LAM, gam=GAM):
        F, parts = [], None
        for x1, x2, c1, c2 in zip(b1, b2, C1r, C2r):
            A1, A2 = x1["A"], x2["A"]
            flip = ((np.sign(c1) != np.sign(c2)) & E_rp).astype(float)
            r = TP.reversal_rates([np.where(E_rp, c1, 0), np.where(E_rp, c2, 0)])
            sp = x1["SPTS"] + x2["SPTS"] + lam * r + gam * (x1["phi"] + x2["phi"])
            blk = dict(TE=np.concatenate([x1["TE"], x2["TE"], x2["TE"] - x1["TE"]]),
                       SPTS=np.concatenate([x1["SPTS"], x2["SPTS"], x2["SPTS"] - x1["SPTS"], sp]),
                       TDA=np.concatenate([x1["TDA"], x2["TDA"], x2["TDA"] - x1["TDA"]]),
                       TESTM=np.concatenate([flip[E_rp], r]))
            F.append(blk)
        return F
    Fr = resp_features()
    n_tr = len(rtr)
    jt, jte, jpool = np.arange(n_tr), np.arange(n_tr, n_tr + len(rte)), np.arange(n_tr, len(ids_r))
    y_tr, y_te, y_pool = yr(rtr), yr(rte), yr(pooled)
    rscores = {}

    def eval_r(name, X, learner="xgb"):
        if learner == "xgb":
            m = xgb_fit(X[jt], y_tr, seed)
        elif learner == "rf":
            m = rf_fit(X[jt], y_tr, seed)
        else:
            m = LogisticRegression().fit(X[jt], y_tr)
        s = m.predict_proba(X)[:, 1]
        rscores[name] = s
    KEYS = ["TE", "SPTS", "TDA", "TESTM"]
    eval_r("network", stack(Fr, KEYS))
    asv1 = D.mean_feature(ids_r, "T1", pp["asv_clr"])
    eval_r("rf_asv_T1", asv1, "rf")
    if full:
        for k in KEYS:
            eval_r("no_" + k, stack(Fr, [x for x in KEYS if x != k]))
        eval_r("network_T1", np.array([np.concatenate([x["TE"], x["SPTS"], x["TDA"]]) for x in b1]))
        asv2 = D.mean_feature(ids_r, "T2", pp["asv_clr"])
        eval_r("rf_asv_T1T2", np.hstack([asv1, asv2]), "rf")
        iu2 = np.where(np.triu(E_rp | E_rp.T, 1))
        _, R1r = lioness_subjects(D, tr, ids_r, "T1")
        eval_r("static_signed", np.array([np.concatenate([a[iu2], b[iu2]]) for a, b in zip(R1r, R2r)]))
        sh = np.column_stack([D.mean_feature(ids_r, w, pp["shannon"][:, None])[:, 0] for w in ("T1", "T2")])
        eval_r("shannon", sh, "lr")
        ham = np.array([[subj[q]["hamd0"]] for q in ids_r])
        eval_r("hamd", ham, "lr")
    rp = {}
    for k, s in rscores.items():
        rp[k] = dict(pooled=EV.summary(y_pool, s[jpool]), test=EV.auc_ci(y_te, s[jte]))
        if k != "network":
            rp[k]["diff_vs_network"] = EV.delong_paired(y_pool, rscores["network"][jpool], s[jpool])
    rp["n_pooled"] = len(pooled); rp["n_pooled_responders"] = int(y_pool.sum())
    if full:
        rp["roc_scores"] = {k: dict(y=y_pool.tolist(), s=v[jpool].tolist()) for k, v in rscores.items()}
        grid = {}
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for lam in (0, 0.1, 0.3, 1.0):
            for gam in (0, 0.1, 0.3, 1.0):
                F = resp_features(lam, gam)
                X = np.array([f["SPTS"][-D.N:] for f in F])[jt]
                aucs = []
                for a, b in skf.split(X, y_tr):
                    m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=2,
                                          random_state=seed, n_jobs=2, verbosity=0).fit(X[a], y_tr[a])
                    aucs.append(roc_auc_score(y_tr[b], m.predict_proba(X[b])[:, 1]))
                grid[f"{lam}_{gam}"] = float(np.mean(aucs))
        rp["spts_grid_cv"] = grid
    out["response"] = rp
    return out
