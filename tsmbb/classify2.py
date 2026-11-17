"""Revised subject-level benchmark (revision; Reviewers 1, 3, 4, 5 and editor).

Changes relative to classify.py (original submission):
  * every feature family is evaluated with the SAME learners and the same hyper-parameter search:
    an L2-regularised logistic regression (linear) and gradient-boosted trees (nonlinear), with a
    random forest as a further nonlinear learner;
  * label-dependent construction steps (group networks that define the edge set, the healthy
    reference diagram, the disease-specific motif classes and the LIONESS reference set) are rebuilt
    inside every inner cross-validation fold and applied, frozen, to that fold's validation subjects;
  * operating thresholds (Youden) are chosen on training-only out-of-fold predictions and frozen
    before the test data are touched;
  * preprocessing (ASV prevalence filter, cohort correction) is fitted on the training subjects
    only (inductive); the unseen cohort SC2 is handled either by site-level recentring on its own
    unlabelled samples or by no correction at all;
  * edges are selected with the cluster-robust test;
  * the true generating response probabilities give an oracle AUROC for response prediction;
  * response performance is reported for the internal test partition, SC2 and pooled.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import xgboost as xgb

from . import preprocess as PP, te as TE, topology as TP, evaluate as EV
from .classify import partitions, SubjectData, lioness_subjects, network_block, stack

LEARNERS = {
    "lr": [dict(C=c) for c in (0.01, 0.1, 1.0)],
    "xgb": [dict(max_depth=d) for d in (2, 3)],
    "rf": [dict(max_features="sqrt")],
}


def make(learner, hp, seed):
    if learner == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=2000))
    if learner == "xgb":
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                                 random_state=seed, n_jobs=1, verbosity=0, **hp)
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1, **hp)


def impute(X, ntr):
    """Replace non-finite feature values (e.g. a subject without usable visits in a window) by the
    median of the training rows; medians use training rows only."""
    X = np.array(X, float)
    bad = ~np.isfinite(X)
    if bad.any():
        med = np.nanmedian(np.where(np.isfinite(X[:ntr]), X[:ntr], np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X[bad] = np.take(med, np.where(bad)[1])
    return X


def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    k = np.argmax(tpr - fpr)
    return float(thr[k])


def at_threshold(y, s, thr):
    y = np.asarray(y).astype(bool); pred = np.asarray(s) >= thr
    return float((pred & y).sum() / max(y.sum(), 1)), float((~pred & ~y).sum() / max((~y).sum(), 1))


def cluster_net(pp, Zd, ids, w):
    X, Y, own = PP.transitions(pp, Zd, ids, w)
    o = TE.cte_tests(X, Y, own)
    return o["cluster"]


def disease_motifs_signed(A_dis, A_hc, rng, n_null=100, q=0.10):
    S_d, S_h = np.sign(A_dis).astype(int), np.sign(A_hc).astype(int)
    ed = TP.motif_enrichment_signed(S_d, n_null, rng); eh = TP.motif_enrichment_signed(S_h, n_null, rng)
    keys = list(ed["p"]); rej = TE.bh([ed["p"][k] for k in keys], q)[0] if keys else []
    hk = list(eh["p"]); hrej = TE.bh([eh["p"][k] for k in hk], q)[0] if hk else []
    hc_enr = {k for k, r in zip(hk, hrej) if r}
    inst = []
    for k, r in zip(keys, rej):
        if r and k not in hc_enr:
            for t in ed["instances"][k]:
                inst.append([(a, b) for a in t for b in t if a != b and S_d[a, b] != 0])
    return inst


# ---------------------------------------------------------------- feature construction
def build_cc(sim, pp, D, train, apply_ids, rng, motifs=True):
    """Case-control features for `apply_ids`; every label-dependent step uses `train` only."""
    traj = {s["subject"]: s["traj"] for s in sim["subjects"]}
    hc = [q for q in train if traj[q] == "HC"]; md = [q for q in train if traj[q] != "HC"]
    n_hc, n_md = cluster_net(pp, D.Zd, hc, "T1"), cluster_net(pp, D.Zd, md, "T1")
    E = n_hc["R"] | n_md["R"]
    h1_ref = TP.persistence(n_hc["A"])[0]
    inst = disease_motifs_signed(n_md["A"], n_hc["A"], rng) if motifs else []
    C1, R1 = lioness_subjects(D, train, apply_ids, "T1")
    blocks = [network_block(C, E, h1_ref, inst) for C in C1]
    iu = np.where(np.triu(E | E.T, 1))
    st_s = np.array([R[iu] for R in R1])
    F = dict(network=stack(blocks, ["TE", "SPTS", "TDA", "MOTIF"]),
             static_signed=st_s, static_unsigned=np.abs(st_s))
    for drop in ("TE", "SPTS", "TDA", "MOTIF"):
        keys = [k for k in ("TE", "SPTS", "TDA", "MOTIF") if k != drop]
        F["network_no_" + drop] = stack(blocks, keys)
    F["_sizes"] = [len(blocks[0][k]) for k in ("TE", "SPTS", "TDA", "MOTIF")] if blocks else []
    F["_n_edges"] = int(E.sum()); F["_n_motif_inst"] = len(inst)
    return F


def build_rp(sim, pp, D, train, apply_ids, hc_train):
    """Response features. `train` are MDD training subjects (inner fold); `hc_train` are the
    training controls, which carry no response label and provide the healthy reference diagrams
    and, together with `train`, the unlabeled LIONESS reference set."""
    traj = {s["subject"]: s["traj"] for s in sim["subjects"]}
    tR = [q for q in train if traj[q] == "R"]; tN = [q for q in train if traj[q] == "NR"]
    thc = list(hc_train)
    ref = np.concatenate([np.asarray(train), np.asarray(hc_train)])
    E = np.zeros((D.N, D.N), bool)
    for ids in (tR, tN):
        for w in ("T1", "T2"):
            E |= cluster_net(pp, D.Zd, ids, w)["R"]
    h1_1 = TP.persistence(cluster_net(pp, D.Zd, thc, "T1")["A"])[0]
    h1_2 = TP.persistence(cluster_net(pp, D.Zd, thc, "T2")["A"])[0]
    C1, R1 = lioness_subjects(D, ref, apply_ids, "T1")
    C2, R2 = lioness_subjects(D, ref, apply_ids, "T2")
    b1 = [network_block(C, E, h1_1, []) for C in C1]
    b2 = [network_block(C, E, h1_2, []) for C in C2]
    F = []
    for x1, x2, c1, c2 in zip(b1, b2, C1, C2):
        flip = ((np.sign(c1) != np.sign(c2)) & E).astype(float)
        r = TP.reversal_rates([np.where(E, c1, 0), np.where(E, c2, 0)])
        sp = x1["SPTS"] + x2["SPTS"] + 0.1 * r + 0.3 * (x1["phi"] + x2["phi"])
        F.append(dict(TE=np.concatenate([x1["TE"], x2["TE"], x2["TE"] - x1["TE"]]),
                      SPTS=np.concatenate([x1["SPTS"], x2["SPTS"], x2["SPTS"] - x1["SPTS"], sp]),
                      TDA=np.concatenate([x1["TDA"], x2["TDA"], x2["TDA"] - x1["TDA"]]),
                      TESTM=np.concatenate([flip[E], r])))
    iu = np.where(np.triu(E | E.T, 1))
    out = dict(network=stack(F, ["TE", "SPTS", "TDA", "TESTM"]),
               network_T1=np.array([np.concatenate([x["TE"], x["SPTS"], x["TDA"]]) for x in b1]),
               static_signed=np.array([np.concatenate([a[iu], b[iu]]) for a, b in zip(R1, R2)]))
    for drop in ("TE", "SPTS", "TDA", "TESTM"):
        out["network_no_" + drop] = stack(F, [k for k in ("TE", "SPTS", "TDA", "TESTM") if k != drop])
    out["_n_edges"] = int(E.sum())
    return out


# ---------------------------------------------------------------- nested evaluation
def nested_eval(train, y_train, eval_sets, builder, fixed, families, learners, seed, n_inner=5):
    """builder(train_ids, apply_ids) -> dict family -> matrix (label-dependent families);
    fixed: dict family -> callable(ids) -> matrix (label-free families, e.g. abundance).
    eval_sets: dict name -> (ids, y). Returns per family/learner: chosen hp, OOF AUROC,
    frozen threshold, and AUROC/sens/spec/AUPRC on each evaluation set, plus raw scores."""
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    train = np.asarray(train)
    oof = {}
    folds = list(skf.split(train, y_train))
    for k, (a, b) in enumerate(folds):
        Ftr = builder(train[a], np.concatenate([train[a], train[b]]))
        na = len(a)
        for fam in families:
            Xall = Ftr[fam] if fam in Ftr else fixed[fam](np.concatenate([train[a], train[b]]))
            if Xall.shape[1] == 0:
                continue
            Xall = impute(Xall, len(a))
            for L in learners.get(fam, ("lr", "xgb", "rf")):
                for h, hp in enumerate(LEARNERS[L]):
                    m = make(L, hp, seed).fit(Xall[:na], y_train[a])
                    s = m.predict_proba(Xall[na:])[:, 1]
                    oof.setdefault((fam, L, h), np.full(len(train), np.nan))[b] = s
    allids = np.concatenate([train] + [v[0] for v in eval_sets.values()])
    Ffull = builder(train, allids)
    res, scores = {}, {}
    ntr = len(train)
    for fam in families:
        Xall = Ffull[fam] if fam in Ffull else fixed[fam](allids)
        if Xall.shape[1] == 0:
            continue
        Xall = impute(Xall, ntr)
        for L in learners.get(fam, ("lr", "xgb", "rf")):
            cand = [(h, roc_auc_score(y_train, oof[(fam, L, h)])) for h in range(len(LEARNERS[L]))
                    if (fam, L, h) in oof]
            h, cv_auc = max(cand, key=lambda x: x[1])
            thr = youden_threshold(y_train, oof[(fam, L, h)])
            m = make(L, LEARNERS[L][h], seed).fit(Xall[:ntr], y_train)
            s_all = m.predict_proba(Xall[ntr:])[:, 1]
            r = dict(hp=LEARNERS[L][h], cv_auc=float(cv_auc), threshold=thr)
            pos = 0
            for name, (ids, y) in eval_sets.items():
                s = s_all[pos:pos + len(ids)]; pos += len(ids)
                ci = EV.auc_ci(y, s)
                se, sp = at_threshold(y, s, thr)
                se_d, sp_d = EV.operating_point(y, s)
                r[name] = dict(ci, sens=se, spec=sp, sens_test_optimised=se_d, spec_test_optimised=sp_d,
                               auprc=float(average_precision_score(y, s)))
                scores[(fam, L, name)] = s
            if L == "xgb" and fam == "network":
                r["_model"] = m
            res[f"{fam}|{L}"] = r
    return res, scores, Ffull


def run(sim, seed=42, cc=True, rp=True, full=True, n_inner=5, inductive=True, light=False):
    """light=True (design experiment): case-control only with the network, abundance and combined
    families, logistic regression and gradient-boosted trees, no motif block."""
    rng = np.random.default_rng(seed)
    tr, te_int, held = partitions(sim, seed)
    subj = {s["subject"]: s for s in sim["subjects"]}
    pp = PP.preprocess(sim, fit_subjects=tr if inductive else None, unseen_policy="recenter")
    pp_none = PP.preprocess(sim, fit_subjects=tr, unseen_policy="none") if inductive else None
    out = dict(n_train=len(tr), n_test=len(te_int), n_heldout=len(held), inductive=inductive,
               n_asv_kept=pp["n_asv_kept"])
    lab = lambda ids: np.array([1 if subj[q]["group"] == "MDD" else 0 for q in ids])
    if cc:
        D = SubjectData(sim, pp, ("T1",))
        Dn = SubjectData(sim, pp_none, ("T1",)) if inductive else D
        fams = ["network", "static_signed", "static_unsigned", "abundance", "network+abundance"]
        if light:
            fams = ["network", "abundance", "network+abundance"]
        if full:
            fams += ["network_no_TE", "network_no_SPTS", "network_no_TDA", "network_no_MOTIF"]
        cache = {}

        def builder(train, apply_ids):
            key = (tuple(train), len(apply_ids))
            if key not in cache:
                F = build_cc(sim, pp, D, train, apply_ids, np.random.default_rng(seed), motifs=not light)
                ab = D.mean_feature(apply_ids, "T1", pp["asv_clr"])
                F["network+abundance"] = np.hstack([F["network"], ab])
                cache.clear(); cache[key] = F
            return cache[key]
        fixed = dict(abundance=lambda ids: D.mean_feature(ids, "T1", pp["asv_clr"]))
        learners = {f: ("lr", "xgb") for f in fams if f.startswith("network_no_") or light}
        evs = dict(test=(te_int, lab(te_int)), heldout=(held, lab(held)))
        res, scores, Ffull = nested_eval(tr, lab(tr), evs, builder, fixed, fams, learners, seed, n_inner)
        # single-score comparators
        allids = np.concatenate([te_int, held])
        genus = D.mean_feature(allids, "T1", pp["Z"][:, :46])
        shan = D.mean_feature(allids, "T1", pp["shannon"][:, None])[:, 0]
        for name, s in (("fprau", -genus[:, 0]), ("shannon", -shan)):
            s = np.where(np.isfinite(s), s, np.nanmedian(s))
            res[name] = dict(test=EV.auc_ci(lab(te_int), s[:len(te_int)]),
                             heldout=EV.auc_ci(lab(held), s[len(te_int):]))
        # strictly prospective handling of SC2 (no cohort correction) for abundance
        if inductive and not light:
            ab_n = impute(Dn.mean_feature(np.concatenate([tr, held]), "T1", pp_none["asv_clr"]), len(tr))
            for L in ("lr", "xgb", "rf"):
                hp = res[f"abundance|{L}"]["hp"]
                m = make(L, hp, seed).fit(ab_n[:len(tr)], lab(tr))
                res[f"abundance|{L}"]["heldout_nocorrection"] = EV.auc_ci(lab(held), m.predict_proba(ab_n[len(tr):])[:, 1])
        # paired DeLong: matched learners, abundance vs network, and combined vs abundance
        yte = lab(te_int)
        for L in (("lr", "xgb") if light else ("lr", "xgb", "rf")):
            res[f"diff_abundance_minus_network|{L}"] = EV.delong_paired(
                yte, scores[("abundance", L, "test")], scores[("network", L, "test")])
            res[f"diff_combined_minus_abundance|{L}"] = EV.delong_paired(
                yte, scores[("network+abundance", L, "test")], scores[("abundance", L, "test")])
        if full:
            for L in ("lr", "xgb"):
                for f in fams:
                    if f.startswith("network_no_"):
                        res[f"diff_{f}|{L}"] = EV.delong_paired(yte, scores[(f, L, "test")],
                                                                scores[("network", L, "test")])
            m = res["network|xgb"].pop("_model")
            X = Ffull["network"][len(tr):len(tr) + len(te_int)]
            contrib = m.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
            imp = np.abs(contrib).sum(0); cuts = np.cumsum([0] + Ffull["_sizes"])
            res["shap_share"] = {k: float(imp[cuts[i]:cuts[i + 1]].sum() / imp.sum())
                                 for i, k in enumerate(("TE", "SPTS", "TDA", "MOTIF"))}
        for k in list(res):
            if isinstance(res[k], dict):
                res[k].pop("_model", None)
        res["_n_edges"] = Ffull["_n_edges"]; res["_n_motif_inst"] = Ffull["_n_motif_inst"]
        out["case_control"] = res
        out["cc_scores"] = {f"{a}|{b}|{c}": v.tolist() for (a, b, c), v in scores.items()
                            if a in ("network", "abundance", "static_signed", "network+abundance")}
        out["cc_y"] = dict(test=lab(te_int).tolist(), heldout=lab(held).tolist())
    if rp:
        D2 = SubjectData(sim, pp, ("T1", "T2"))
        mdd = lambda ids: [q for q in ids if subj[q]["traj"] != "HC"]
        rtr, rte, rho = np.array(mdd(tr)), np.array(mdd(te_int)), np.array(mdd(held))
        hc_tr = np.array([q for q in tr if subj[q]["traj"] == "HC"])
        yr = lambda ids: np.array([1 if subj[q]["traj"] == "R" else 0 for q in ids])
        fams = ["network", "network_T1", "static_signed", "abundance_T1", "abundance_T1T2", "hamd", "shannon"]
        if full:
            fams += ["network_no_TE", "network_no_SPTS", "network_no_TDA", "network_no_TESTM"]
        cache = {}

        def builder(train, apply_ids):
            key = (tuple(train), len(apply_ids))
            if key not in cache:
                cache.clear(); cache[key] = build_rp(sim, pp, D2, train, apply_ids, hc_tr)
            return cache[key]
        fixed = dict(
            abundance_T1=lambda ids: D2.mean_feature(ids, "T1", pp["asv_clr"]),
            abundance_T1T2=lambda ids: np.hstack([D2.mean_feature(ids, "T1", pp["asv_clr"]),
                                                  D2.mean_feature(ids, "T2", pp["asv_clr"])]),
            hamd=lambda ids: np.array([[subj[q]["hamd0"]] for q in ids]),
            shannon=lambda ids: np.column_stack([D2.mean_feature(ids, w, pp["shannon"][:, None])[:, 0]
                                                 for w in ("T1", "T2")]))
        learners = dict(hamd=("lr",), shannon=("lr",))
        learners.update({f: ("lr", "xgb") for f in fams if f.startswith("network_no_")})
        pooled = np.concatenate([rte, rho])
        evs = dict(test=(rte, yr(rte)), heldout=(rho, yr(rho)), pooled=(pooled, yr(pooled)))
        res, scores, _ = nested_eval(rtr, yr(rtr), evs, builder, fixed, fams, learners, seed, n_inner)
        # oracle: true generating response probabilities
        for name, ids in (("test", rte), ("heldout", rho), ("pooled", pooled)):
            pr = np.array([subj[q]["p_resp"] for q in ids])
            res.setdefault("oracle", {})[name] = EV.auc_ci(yr(ids), pr)
        ypool = yr(pooled)
        for L in ("lr", "xgb", "rf"):
            for f in ("abundance_T1", "abundance_T1T2", "network_T1", "static_signed"):
                if (f, L, "pooled") in scores:
                    res[f"diff_{f}_minus_network|{L}"] = EV.delong_paired(
                        ypool, scores[(f, L, "pooled")], scores[("network", L, "pooled")])
        pr = np.array([subj[q]["p_resp"] for q in pooled])
        for L in ("lr", "xgb"):
            res[f"diff_oracle_minus_network|{L}"] = EV.delong_paired(ypool, pr, scores[("network", L, "pooled")])
        for k in list(res):
            if isinstance(res[k], dict):
                res[k].pop("_model", None)
        out["response"] = res
    return out
