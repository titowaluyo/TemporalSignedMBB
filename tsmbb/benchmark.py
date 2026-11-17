"""Group-level Monte Carlo benchmark (revision).

One call of `replicate(seed, scenario)` generates an independent dataset and returns every
group-level performance measure used in the revised manuscript. Independent replications replace
the overlapping subsamples of a single dataset used in the original submission.
"""
import numpy as np
from scipy import stats

from . import generator as G, preprocess as PP, te as TE, topology as TP

GROUPS = ("HC", "MDD", "R", "NR")
NETS = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T1", "NR-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
TRAJ = {"HC": ["HC-T1", "HC-T2", "HC-T3"], "R": ["R-T1", "R-T2", "R-T3"], "NR": ["NR-T1", "NR-T2", "NR-T3"]}
DESC_NETS = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
LOOPS = [["IL-6", "TNF-alpha", "Cortisol", "LPS"],
         ["IDO1/Kyn", "Quinolinic acid", "Glutamate", "GABA", "Tryptophan"],
         ["R. gnavus", "Sec. bile acids", "BCAA", "Dopamine pathway"]]


def group_ids(sim):
    """Correct group membership (fixes the substring test of the original run_all.py, in which
    'R' in 'NR' made every responder a member of the non-responder group)."""
    subj = sim["subjects"]
    ids = {t: np.array([s["subject"] for s in subj if s["traj"] == t]) for t in ("HC", "R", "NR")}
    ids["MDD"] = np.concatenate([ids["R"], ids["NR"]])
    return ids


def true_B(sim, net):
    g, w = net.split("-")
    return sim["B"][("MDD", w) if g == "MDD" else (g, w)]


def edge_metrics(R, Bhat, T, curated_mask=None):
    """Precision, recall, conditional sign accuracy, signed precision/recall, FDP."""
    true = T != 0
    tp = R & true
    n_r, n_t = int(R.sum()), int(true.sum())
    sgn_ok = tp & (np.sign(Bhat) == T)
    out = dict(edges=n_r,
               precision=tp.sum() / n_r if n_r else np.nan,
               recall=tp.sum() / n_t,
               sign=(sgn_ok.sum() / tp.sum()) if tp.sum() else np.nan,
               signed_precision=sgn_ok.sum() / n_r if n_r else np.nan,
               signed_recall=sgn_ok.sum() / n_t,
               fdp=(1 - tp.sum() / n_r) if n_r else 0.0,
               false_edges=int((R & ~true).sum()))
    if curated_mask is not None:
        cm = curated_mask & true
        bm = ~curated_mask & true
        out["recall_curated"] = (R & cm).sum() / max(cm.sum(), 1)
        out["recall_background"] = (R & bm).sum() / max(bm.sum(), 1)
    return {k: float(v) for k, v in out.items()}


def curated_mask(keep):
    M = np.zeros((G.N_LATENT, G.N_LATENT), bool)
    for s, t, _ in G.CURATED:
        M[G.IDX[s], G.IDX[t]] = True
    for s, t in [("F. prausnitzii", "R. gnavus"), ("R. gnavus", "F. prausnitzii"),
                 ("A. putredinis", "IL-6"), ("IL-6", "IDO1/Kyn")]:
        M[G.IDX[s], G.IDX[t]] = True
    for loop in LOOPS:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            M[G.IDX[a], G.IDX[b]] = True
    return M[np.ix_(keep, keep)]


# ---------------------------------------------------------------- observation stages
def stage_matrices(sim, pp):
    """Network input at three stages (Reviewer 3 comment 2): latent trajectories, the CLR
    transform of noiseless latent abundances, and the final noisy observations (pp['Z'])."""
    rows, keep = pp["rows"], pp["obs_index"]
    lat = sim["latent"][rows]
    mu = sim["mu"][rows]
    nt = G.N_TAXA
    clr = lat.copy(); clr[:, :nt] = lat[:, :nt] - lat[:, :nt].mean(1, keepdims=True)
    clr_mu = mu.copy(); clr_mu[:, :nt] = mu[:, :nt] - mu[:, :nt].mean(1, keepdims=True)
    return dict(latent=(lat[:, keep], mu[:, keep]), clr=(clr[:, keep], clr_mu[:, keep]),
                observed=(pp["Z"], None))


def demean(Z, pp, how="full", true_mu=None):
    s = pp["subject"]
    if how == "true":
        return Z - true_mu
    if how == "window":
        out = Z.copy()
        key = np.char.add(s.astype(str), pp["window"].astype(str))
        for k in np.unique(key):
            m = key == k
            out[m] = Z[m] - Z[m].mean(0)
        return out
    out = Z.copy()
    for q in np.unique(s):
        m = s == q
        out[m] = Z[m] - Z[m].mean(0)
    return out


def observable_estimand(B, keep, host_noise=0.10, sd=0.35):
    """Population lag-1 regression coefficients on the observed coordinates (CLR without the
    omitted coordinate, host nodes with measurement noise) implied by a stationary latent VAR.
    beta = Var(y)^-1 Cov(y(t-1), y(t)) with y = M x (+ noise on host nodes)."""
    nt = G.N_TAXA
    N = B.shape[0]
    M = np.eye(N)
    M[:nt, :nt] = np.eye(nt) - 1.0 / nt
    M = M[keep]
    S0 = G.stationary_cov(B, sd)
    V = M @ S0 @ M.T
    hn = np.r_[np.zeros(len(keep) - G.N_HOST), np.full(G.N_HOST, host_noise ** 2)]
    V = V + np.diag(hn)
    C1 = M @ S0 @ B @ M.T
    return np.linalg.solve(V, C1)


# ---------------------------------------------------------------- reversal truth
def reversal_scoring(nets, sim, keep):
    """TESTM against the complete edge-state truth of every trajectory/window pair."""
    out = {}
    planted = set((G.IDX[s], G.IDX[t]) for s, t in G.REVERSAL)
    kidx = {k: i for i, k in enumerate(keep)}
    planted_obs = set((kidx[i], kidx[j]) for i, j in planted if i in kidx and j in kidx)
    for tr, seq in TRAJ.items():
        for a, b in zip(seq[:-1], seq[1:]):
            tt = G.transition_truth(true_B(sim, a), true_B(sim, b), keep)
            A1, A2 = nets[a]["A"], nets[b]["A"]
            both = (A1 != 0) & (A2 != 0)
            det = both & (np.sign(A1) != np.sign(A2))
            trev = tt["reversal"]
            key = f"{tr}:{a.split('-')[1]}{b.split('-')[1]}"
            missed = trev & ~det
            out[key] = dict(
                true_reversals=int(trev.sum()), detected=int(det.sum()),
                tp=int((det & trev).sum()), fp=int((det & ~trev).sum()),
                miss_absent_first=int((missed & (A1 == 0) & (A2 != 0)).sum()),
                miss_absent_second=int((missed & (A1 != 0) & (A2 == 0)).sum()),
                miss_absent_both=int((missed & (A1 == 0) & (A2 == 0)).sum()),
                miss_same_sign=int((missed & both & (np.sign(A1) == np.sign(A2))).sum()),
                true_appear=int(tt["appear"].sum()), true_disappear=int(tt["disappear"].sum()),
                det_appear_correct=int(((A1 == 0) & (A2 != 0) & tt["appear"]).sum()),
                det_disappear_correct=int(((A1 != 0) & (A2 == 0) & tt["disappear"]).sum()),
                planted_detected=[bool(det[i, j]) for i, j in sorted(planted_obs)],
                true_list=[(int(i), int(j)) for i, j in np.argwhere(trev)])
    return out


# ---------------------------------------------------------------- topology oracle
def loop_oracle(sim, keep, names):
    """Apply the topological construction (Eq. 6) to the ground-truth matrices (Reviewer 3 c. 7):
    with and without the planted loops, with and without background edges."""
    I = {n: i for i, n in enumerate(names)}
    B = true_B(sim, "NR-T3")
    W = np.abs(B[np.ix_(keep, keep)]); np.fill_diagonal(W, 0)
    loop_arcs = [[(I[a], I[b]) for a, b in zip(L, L[1:] + L[:1])] for L in LOOPS]
    added = []
    Bt2 = np.abs(true_B(sim, "NR-T2")[np.ix_(keep, keep)])
    for arcs in loop_arcs:
        added.append([(i, j) for i, j in arcs if Bt2[i, j] == 0])
    cur = curated_mask(keep)

    def h1(Wm):
        h, tot = TP.persistence_D(TP.distance(Wm))
        return len(h), tot

    res = {}
    for bg in (True, False):
        Wb = W.copy()
        if not bg:
            lm = np.zeros_like(cur)
            for arcs in loop_arcs:
                for i, j in arcs:
                    lm[i, j] = True
            Wb = np.where(cur | lm, Wb, 0)
        base = h1(Wb)
        res[f"bg={bg}"] = dict(full=base)
        for k, arcs in enumerate(added):
            Wn = Wb.copy()
            for i, j in arcs:
                Wn[i, j] = 0
            res[f"bg={bg}"][f"without_loop{k+1}"] = h1(Wn)
        Wn = Wb.copy()
        for arcs in added:
            for i, j in arcs:
                Wn[i, j] = 0
        res[f"bg={bg}"]["without_all_loops"] = h1(Wn)
        # each loop alone (sanity: one H1 class)
        for k, arcs in enumerate(loop_arcs):
            Wl = np.zeros_like(W)
            for i, j in arcs:
                Wl[i, j] = W[i, j]
            res[f"loop{k+1}_alone"] = h1(Wl)
    return res, loop_arcs


def loop_recovery(A, loop_arcs):
    rec = [np.mean([A[i, j] != 0 for i, j in arcs]) for arcs in loop_arcs]
    full = [all(A[i, j] > 0 for i, j in arcs) for arcs in loop_arcs]
    return dict(arc_recall=float(np.mean([A[i, j] != 0 for arcs in loop_arcs for i, j in arcs])),
                loops_complete=int(sum(full)), per_loop_recall=[float(x) for x in rec])


# ---------------------------------------------------------------- replicate
def replicate(seed, scenario=None, size=200, n_density=200, topo=True, stages=True,
              demean_variants=True, spts=True, visits=5):
    rng = np.random.default_rng(10_000 + seed)
    sim = G.simulate(seed, visits_per_window=visits, scenario=scenario)
    pp = PP.preprocess(sim)
    keep, names = pp["obs_index"], pp["names"]
    ids = group_ids(sim)
    cmask = curated_mask(keep)
    out = dict(seed=seed, n=dict({k: int(len(v)) for k, v in ids.items()}))
    Zd = demean(pp["Z"], pp, "full")

    def nets_from(Zdm, sids_of, tests=("naive", "dfcorr", "cluster")):
        res = {}
        for net in NETS:
            g, w = net.split("-")
            X, Y, own = PP.transitions(pp, Zdm, sids_of(g), w)
            res[net] = TE.cte_tests(X, Y, own)
        return res

    # ---- full-size networks, observed data, three tests
    full = nets_from(Zd, lambda g: ids[g])
    rec = {}
    for net in NETS:
        T = G.truth_matrix(true_B(sim, net), keep)
        rec[net] = {m: edge_metrics(full[net][m]["R"], full[net]["B"], T, cmask)
                    for m in ("naive", "dfcorr", "cluster")}
        nullpairs = (T == 0) & ~np.eye(len(keep), dtype=bool)
        for m in ("naive", "dfcorr", "cluster"):
            rec[net][m]["p05_rate_nulls"] = float((full[net][m]["P"][nullpairs] < 0.05).mean())
            rec[net][m]["any_false"] = float(rec[net][m]["false_edges"] > 0)
        rec[net]["n_transitions"] = int(full[net]["n"]); rec[net]["n_subjects"] = int(full[net]["G"])
    out["full"] = rec
    # observable-scale estimand: sign agreement on planted edges, magnitude of implied non-edges
    obs = {}
    for net in ("HC-T1", "MDD-T1", "NR-T3"):
        Bt = true_B(sim, net)
        beta = observable_estimand(Bt, keep)
        T = G.truth_matrix(Bt, keep)
        off = ~np.eye(len(keep), dtype=bool)
        te_ = T != 0
        obs[net] = dict(sign_agree=float((np.sign(beta[te_]) == T[te_]).mean()),
                        n_sign_flip=int((np.sign(beta[te_]) != T[te_]).sum()),
                        nonedge_abs_max=float(np.abs(beta[off & ~te_]).max()),
                        nonedge_abs_p99=float(np.percentile(np.abs(beta[off & ~te_]), 99)),
                        nonedge_n_gt_0p05=int((np.abs(beta[off & ~te_]) > 0.05).sum()),
                        edge_abs_median=float(np.median(np.abs(beta[te_]))),
                        est_sign_vs_beta=float((np.sign(full[net]["B"][full[net]["cluster"]["R"]]) ==
                                                np.sign(beta[full[net]["cluster"]["R"]])).mean()))
    out["observable_estimand"] = obs

    # ---- observation stages and demeaning (cluster-robust test)
    if stages:
        st = stage_matrices(sim, pp)
        res = {}
        for stage, (Zs, mus) in st.items():
            variants = ["full"] + (["window"] if demean_variants else [])
            if mus is not None and demean_variants:
                variants.append("true")
            for how in variants:
                Zdm = demean(Zs, pp, how, mus)
                nn = {}
                for net in ("HC-T1", "MDD-T1", "R-T3", "NR-T3", "R-T2", "NR-T2"):
                    g, w = net.split("-")
                    X, Y, own = PP.transitions(pp, Zdm, ids[g], w)
                    o = TE.cte_tests(X, Y, own)
                    T = G.truth_matrix(true_B(sim, net), keep)
                    nn[net] = edge_metrics(o["cluster"]["R"], o["B"], T, cmask)
                    # self-coefficient bias (dynamic-panel bias)
                    nn[net]["self_coef_mean"] = float(np.mean(np.diag(o["B"])))
                res[f"{stage}|{how}"] = nn
        out["stages"] = res
        # window-specific demeaning, observed stage: effect on reversals
        Zw = demean(pp["Z"], pp, "window")
        fw = nets_from(Zw, lambda g: ids[g], tests=("cluster",))
        out["reversal_window_demean"] = reversal_scoring({k: v["cluster"] for k, v in fw.items()}, sim, keep)

    # ---- complete reversal truth (full-size, cluster test and naive test)
    out["reversal"] = {m: reversal_scoring({k: v[m] for k, v in full.items()}, sim, keep)
                       for m in ("naive", "cluster")}

    if not topo:
        return out
    # ---- size-matched networks for descriptors (one independent draw per replicate)
    hc = rng.permutation(ids["HC"])
    smp = dict(HC=hc[:size], MDD=rng.choice(ids["MDD"], size, replace=False),
               R=rng.choice(ids["R"], min(size, len(ids["R"])), replace=False),
               NR=rng.choice(ids["NR"], min(size, len(ids["NR"])), replace=False))
    hc_ref = hc[size:2 * size]
    sm = nets_from(Zd, lambda g: smp[g], tests=("cluster",))
    ref = {}
    for w in ("T1", "T2", "T3"):
        X, Y, own = PP.transitions(pp, Zd, hc_ref, w)
        ref[w] = TE.cte_tests(X, Y, own)["cluster"]
    oracle, loop_arcs = loop_oracle(sim, keep, names)
    out["loop_oracle"] = oracle
    desc = {}
    for net in DESC_NETS:
        o = sm[net]; A = o["cluster"]["A"]; g, w = net.split("-")
        T = G.truth_matrix(true_B(sim, net), keep)
        em = edge_metrics(o["cluster"]["R"], o["B"], T, cmask)
        h1, tot, phi = TP.persistence(A, cocycles=True)
        dn = TP.density_null(A, n_density, rng)
        nz = A[A != 0]
        fr_sum = TP.frustration_exact(A, "sum"); fr_multi = TP.frustration_exact(A, "multi")
        fr_greedy = TP.frustration(A, 20, rng)
        seq = TRAJ.get(g)
        if g == "MDD":
            Aseq, phis = [A], [phi]
        else:
            upto = seq[:["T1", "T2", "T3"].index(w) + 1]
            Aseq = [sm[x]["cluster"]["A"] for x in upto]
            phis = [TP.persistence(sm[x]["cluster"]["A"], cocycles=True)[2] for x in upto]
        sp = TP.spts(Aseq, phis)
        rs = TP.resilience(A, np.argsort(-sp), 20, rng)
        cyc = TP.directed_cycles(A, 5)
        alt = {k: TP.persistence_D(TP.distance_alt(A, k))[1] for k in ("log", "rank")}
        d = dict(em, pos=100 * float((nz > 0).mean()) if len(nz) else np.nan,
                 mean_te=100 * float(np.abs(nz).mean()) if len(nz) else np.nan,
                 h1=tot, h1_null_mean=dn["null_mean"], h1_null_sd=dn["null_sd"], h1_z=dn["z"],
                 h1_log=alt["log"], h1_rank=alt["rank"],
                 w2=TP.wasserstein2(h1, TP.persistence(ref[w]["A"])[0]),
                 frustration_greedy=fr_greedy, frustration_exact=fr_sum[0], frustration_multi=fr_multi[0],
                 cancelled_pairs=fr_sum[2], auc=rs["auc"], auc_random=rs["auc_random"], dfrag=rs["dfrag"],
                 cyc3=cyc[3], cyc4=cyc[4], cyc5=cyc[5])
        d.update({f"loop_{k}": v for k, v in loop_recovery(A, loop_arcs).items()})
        desc[net] = d
    out["sizematched"] = desc
    # full-size loop recovery
    out["loops_full"] = {net: loop_recovery(full[net]["cluster"]["A"], loop_arcs)
                         for net in ("NR-T3", "R-T3", "NR-T2")}

    # ---- SPTS / perturbation score against perturbations of the generative model
    if spts:
        sres = {}
        for net in ("HC-T1", "MDD-T1", "R-T3", "NR-T3"):
            Bt = true_B(sim, net)
            lr, ie = TP.dynamic_response(Bt, keep)
            o = full[net]; A = o["cluster"]["A"]
            Bh = np.where(o["cluster"]["R"], o["B"], 0.0)
            Bfull = o["B"].copy()
            Ttrue = Bt[np.ix_(keep, keep)].copy(); np.fill_diagonal(Ttrue, 0)
            r = {}
            for lab, W in (("TE", A), ("coef", Bh), ("TE_maxnorm", A / max(np.abs(A).max(), 1e-12)),
                           ("oracle_B", Ttrue)):
                for lap in ("out", "in", "sym"):
                    for rc in (1e-15, 1e-8, 1e-4):
                        ps, rank, cond = TP.perturbation_scores_var(W, rc, lap)
                        r[f"{lab}|{lap}|{rc:g}"] = dict(
                            rho_longrun=float(stats.spearmanr(ps, lr)[0]),
                            rho_impulse=float(stats.spearmanr(ps, ie)[0]), rank=rank, cond=cond)
            # direct VAR-based response from the estimated coefficient matrix
            try:
                Mh = np.linalg.inv(np.eye(len(keep)) - Bfull.T)
                vr = np.linalg.norm(Mh, axis=0)
                r["VAR_estimated"] = dict(rho_longrun=float(stats.spearmanr(vr, lr)[0]),
                                          rho_impulse=float(stats.spearmanr(vr, ie)[0]))
            except np.linalg.LinAlgError:
                pass
            # scale check: multiplying weights by 2 halves the pseudoinverse score
            ps1 = TP.perturbation_scores(A); ps2 = TP.perturbation_scores(2 * A)
            r["scale_ratio"] = float(np.median(ps2 / np.where(ps1 > 0, ps1, np.nan)))
            sres[net] = r
        out["spts_validation"] = sres
    return out
