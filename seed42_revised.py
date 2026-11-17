"""Revision: the illustrative dataset (seed 42) regenerated with the revised generator (stationary
initialisation with burn-in) and analysed with corrected group membership and the cluster-robust
test. Provides Table 1, Fig. 1-3 inputs and the worked examples of Section 3.
Writes results/main_rev.json and results/main_rev_arrays.npz."""
import json, time
import numpy as np
from scipy import stats
from tsmbb import generator as G, preprocess as PP, te as TE, topology as TP, benchmark as BM
from mc_group import j, SCEN

SEED = 42
STAT = SCEN["main"][0]


def main():
    t0 = time.time()
    sim = G.simulate(SEED, scenario=STAT)
    pp = PP.preprocess(sim)
    keep, names = pp["obs_index"], pp["names"]
    subj = sim["subjects"]
    ids = BM.group_ids(sim)
    res = dict(seed=SEED, scenario=STAT, nodes=names)
    tab = []
    for c in ("SC1", "SC2", "SC3", "Combined"):
        S = [s for s in subj if c == "Combined" or s["cohort"] == c]
        M = [s for s in S if s["group"] == "MDD"]
        ham = np.array([s["hamd0"] for s in M]); age = np.array([s["age"] for s in S])
        nres = sum(s["responder"] == 1 for s in M)
        nq = int(np.sum(np.isin(pp["subject"], [s["subject"] for s in S])))
        tab.append(dict(cohort=c, mdd=len(M), hc=len(S) - len(M), female=100 * np.mean([s["female"] for s in S]),
                        age_mean=age.mean(), age_sd=age.std(ddof=1), hamd_mean=ham.mean(), hamd_sd=ham.std(ddof=1),
                        responders=nres, responders_pct=100 * nres / len(M), qc_samples=nq))
    res["cohorts"] = tab
    res["preprocess"] = {k: pp[k] for k in ("n_scheduled", "n_missed", "n_generated", "n_qc", "n_asv_kept",
                                            "r2_cohort_before", "r2_cohort_after")}
    sel = pp["window"] == "T1"
    sh = {g: np.array([pp["shannon"][sel & (pp["subject"] == q)].mean() for q in ids[g]]) for g in ("HC", "MDD")}
    t = stats.ttest_ind(sh["MDD"], sh["HC"], nan_policy="omit")
    res["shannon_T1"] = dict(mdd_mean=float(np.nanmean(sh["MDD"])), mdd_sd=float(np.nanstd(sh["MDD"], ddof=1)),
                             hc_mean=float(np.nanmean(sh["HC"])), hc_sd=float(np.nanstd(sh["HC"], ddof=1)),
                             t=float(t.statistic), p=float(t.pvalue))
    grp = np.array([1.0 if subj[q]["group"] == "MDD" else 0.0 for q in pp["subject"][sel]])
    Xg = np.column_stack([np.ones(sel.sum()), grp]); Zt = pp["Z"][sel]
    fit = Xg @ np.linalg.lstsq(Xg, Zt, rcond=None)[0]
    res["preprocess"]["r2_case_control_T1"] = float(1 - ((Zt - fit) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum())
    # effect size of the planted F. prausnitzii shift on subject-mean CLR at T1 (Cohen's d)
    fp = names.index("F. prausnitzii")
    m = {g: np.array([pp["Z"][sel & (pp["subject"] == q), fp].mean() for q in ids[g]
                      if (sel & (pp["subject"] == q)).any()]) for g in ("HC", "MDD")}
    sp = np.sqrt((m["HC"].var(ddof=1) + m["MDD"].var(ddof=1)) / 2)
    res["fprau_cohen_d"] = float((m["HC"].mean() - m["MDD"].mean()) / sp)
    Zd = BM.demean(pp["Z"], pp, "full")
    full = {}
    for net in BM.NETS:
        g, w = net.split("-")
        X, Y, own = PP.transitions(pp, Zd, ids[g], w)
        full[net] = TE.cte_tests(X, Y, own)
    res["full"] = {}
    for net in BM.NETS:
        T = G.truth_matrix(BM.true_B(sim, net), keep)
        res["full"][net] = {m_: BM.edge_metrics(full[net][m_]["R"], full[net]["B"], T, BM.curated_mask(keep))
                            for m_ in ("naive", "dfcorr", "cluster")}
        res["full"][net]["n_transitions"] = int(full[net]["n"])
    I = {n: i for i, n in enumerate(names)}

    def edge(net, s, t_):
        o = full[net]; c = o["cluster"]
        return dict(te=float(o["TE"][I[s], I[t_]]), q=float(c["Q"][I[s], I[t_]]),
                    sign=int(np.sign(o["B"][I[s], I[t_]])), retained=bool(c["R"][I[s], I[t_]]),
                    coef=float(o["B"][I[s], I[t_]]))
    res["fig3_edges"] = {f"{net}:{s}->{t_}": edge(net, s, t_) for net in ("HC-T1", "MDD-T1")
                         for s, t_ in [("F. prausnitzii", "5-HT synthesis"), ("F. prausnitzii", "Butyrate"),
                                       ("R. gnavus", "IDO1/Kyn"), ("F. prausnitzii", "R. gnavus"),
                                       ("R. gnavus", "F. prausnitzii"), ("A. putredinis", "IDO1/Kyn")]}
    # KSG check on retained MDD-T1 edges
    X, Y, _ = PP.transitions(pp, Zd, ids["MDD"], "T1")
    o = full["MDD-T1"]; E = np.argwhere(o["cluster"]["R"])
    ksg, gb, gc = [], [], []
    for i, jj in E:
        ksg.append(TE.ksg_cmi(Y[:, jj], X[:, i], X[:, jj], k=4))
        A = X[:, [jj]]; b0 = np.linalg.lstsq(A, Y[:, jj], rcond=None)[0]; r0 = ((Y[:, jj] - A @ b0) ** 2).sum()
        A2 = X[:, [jj, i]]; b1 = np.linalg.lstsq(A2, Y[:, jj], rcond=None)[0]; r1 = ((Y[:, jj] - A2 @ b1) ** 2).sum()
        gb.append(0.5 * np.log2(r0 / r1)); gc.append(o["TE"][i, jj])
    res["ksg_check"] = dict(n=len(E), rho_bivariate=float(stats.spearmanr(ksg, gb)[0]),
                            p_bivariate=float(stats.spearmanr(ksg, gb)[1]),
                            rho_conditional=float(stats.spearmanr(ksg, gc)[0]),
                            p_conditional=float(stats.spearmanr(ksg, gc)[1]))
    rv = BM.reversal_scoring({k: v["cluster"] for k, v in full.items()}, sim, keep)
    for k in rv:
        rv[k]["true_names"] = [f"{names[a]} -> {names[b]}" for a, b in rv[k]["true_list"]]
    res["reversal_truth"] = rv
    # coefficient of the extra responder reversal in the seed-42 matrices
    B3 = BM.true_B(sim, "R-T3"); B2 = BM.true_B(sim, "R-T2")
    res["rgnavus_fprau"] = dict(T2=float(B2[G.IDX["R. gnavus"], G.IDX["F. prausnitzii"]]),
                                T3=float(B3[G.IDX["R. gnavus"], G.IDX["F. prausnitzii"]]))
    # reciprocal conflicting arcs in the full-size networks (frustration projection)
    res["cancelled_pairs_full"] = {n: TP.frustration_exact(full[n]["cluster"]["A"])[2] for n in BM.DESC_NETS}
    res["runtime_s"] = time.time() - t0
    json.dump(j(res), open("results/main_rev.json", "w"), indent=1)
    np.savez_compressed("results/main_rev_arrays.npz", A_HC_T1=full["HC-T1"]["cluster"]["A"],
                        A_MDD_T1=full["MDD-T1"]["cluster"]["A"], qc_week=pp["week"], qc_subject=pp["subject"],
                        qc_cohort=pp["cohort"], qc_group=np.array([subj[q]["group"] for q in pp["subject"]]))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
