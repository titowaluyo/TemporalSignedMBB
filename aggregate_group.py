"""Revision: summarise the group-level Monte Carlo replications (results/mc/*) into
results/summary_group.json. Every summary reports the mean over independent replications, the
between-replication SD and the Monte Carlo standard error (MCSE = SD / sqrt(R))."""
import glob, json, os
import numpy as np
from scipy import stats
from tsmbb import generator as G

NETS = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T1", "NR-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
DESC = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
TESTS = ("naive", "dfcorr", "cluster")


def load(label):
    return [json.load(open(f)) for f in sorted(glob.glob(f"results/mc/{label}/rep_*.json"))]


def ms(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return dict(mean=None, sd=None, mcse=None, R=0)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return dict(mean=float(v.mean()), sd=sd, mcse=sd / np.sqrt(len(v)), R=int(len(v)),
                p2_5=float(np.percentile(v, 2.5)), p97_5=float(np.percentile(v, 97.5)))


def paired(a, b):
    d = np.array(a, float) - np.array(b, float)
    d = d[np.isfinite(d)]
    t = stats.ttest_1samp(d, 0.0)
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    return dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), mcse=float(d.std(ddof=1) / np.sqrt(len(d))),
                ci_lo=float(d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
                ci_hi=float(d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))),
                t=float(t.statistic), p=float(t.pvalue), p_wilcoxon=float(w.pvalue) if w else 1.0, R=int(len(d)))


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1)
    return adj.tolist()


def recovery(reps, nets=NETS, tests=TESTS):
    keys = ("precision", "recall", "sign", "signed_precision", "signed_recall", "fdp", "edges",
            "false_edges", "recall_curated", "recall_background", "p05_rate_nulls", "any_false")
    out = {}
    for net in nets:
        out[net] = {t: {k: ms([r["full"][net][t].get(k) for r in reps]) for k in keys} for t in tests}
        out[net]["n_transitions"] = ms([r["full"][net]["n_transitions"] for r in reps])
    # pooled across the 10 networks
    out["ALL"] = {t: {k: ms([np.nanmean([r["full"][n][t][k] for n in nets]) for r in reps])
                      for k in ("precision", "recall", "sign", "signed_precision", "signed_recall", "fdp",
                                "p05_rate_nulls", "any_false", "false_edges")} for t in tests}
    return out


def reversal_summary(reps, key="reversal", test="cluster"):
    pairs = list(reps[0][key][test].keys()) if test else list(reps[0][key].keys())
    out = {}
    for p in pairs:
        rr = [(r[key][test] if test else r[key])[p] for r in reps]
        tot = lambda k: int(sum(x[k] for x in rr))
        out[p] = dict(true_reversals_total=tot("true_reversals"), tp_total=tot("tp"), fp_total=tot("fp"),
                      sensitivity=tot("tp") / tot("true_reversals") if tot("true_reversals") else None,
                      sens_by_rep=ms([x["tp"] / x["true_reversals"] for x in rr if x["true_reversals"]]),
                      fp_per_pair=ms([x["fp"] for x in rr]),
                      true_per_rep=ms([x["true_reversals"] for x in rr]),
                      miss_absent_first=tot("miss_absent_first"), miss_absent_second=tot("miss_absent_second"),
                      miss_absent_both=tot("miss_absent_both"), miss_same_sign=tot("miss_same_sign"),
                      appear_true=tot("true_appear"), appear_correct=tot("det_appear_correct"),
                      disappear_true=tot("true_disappear"), disappear_correct=tot("det_disappear_correct"),
                      planted_rate=np.mean([x["planted_detected"] for x in rr], 0).tolist())
    # additional (non-prespecified) true reversals: which edges, how often
    extra = {}
    keep_names = None
    for r in reps:
        pl = None
    allp = [x for p in out for x in [out[p]]]
    tp = sum(o["tp_total"] for o in out.values()); tr = sum(o["true_reversals_total"] for o in out.values())
    fp = sum(o["fp_total"] for o in out.values())
    out["ALL"] = dict(sensitivity=tp / tr if tr else None, tp=tp, true=tr, fp=fp,
                      n_pair_evaluations=len(reps) * len(pairs),
                      fp_per_pair_evaluation=fp / (len(reps) * len(pairs)))
    return out


def main():
    S = {}
    # ---------------- main scenario
    reps = load("main")
    S["main_R"] = len(reps)
    S["main_recovery"] = recovery(reps)
    S["main_reversal"] = {t: reversal_summary(reps, "reversal", t) for t in ("naive", "cluster")}
    S["main_reversal_window_demean"] = reversal_summary(reps, "reversal_window_demean", None)
    # planted-edge names in the order used by planted_detected
    from tsmbb import preprocess as PP
    keep = list(G.TAXA.index(t) for t in G.TAXA if t != G.REF_TAXON) + list(range(G.N_TAXA, G.N_LATENT))
    kidx = {k: i for i, k in enumerate(keep)}
    planted = sorted((kidx[G.IDX[s]], kidx[G.IDX[t]]) for s, t in G.REVERSAL)
    names = [n for n in G.NODES if n != G.REF_TAXON]
    S["planted_order"] = [f"{names[i]} -> {names[j]}" for i, j in planted]
    # complete truth: which unplanted reversals exist
    extra = {}
    for r in reps:
        for p, v in r["reversal"]["cluster"].items():
            for i, j in v["true_list"]:
                if (i, j) not in planted:
                    extra.setdefault(f"{p}: {names[i]} -> {names[j]}", 0)
                    extra[f"{p}: {names[i]} -> {names[j]}"] += 1
    S["unplanted_true_reversals"] = extra
    # stages
    st = {}
    for key in reps[0]["stages"]:
        st[key] = {net: {k: ms([r["stages"][key][net][k] for r in reps])
                         for k in ("precision", "recall", "sign", "signed_recall", "self_coef_mean")}
                   for net in reps[0]["stages"][key]}
    S["stages"] = st
    S["observable_estimand"] = {net: {k: ms([r["observable_estimand"][net][k] for r in reps])
                                      for k in reps[0]["observable_estimand"][net]}
                                for net in reps[0]["observable_estimand"]}
    # size-matched descriptors
    dk = [k for k in reps[0]["sizematched"]["HC-T1"] if not isinstance(reps[0]["sizematched"]["HC-T1"][k], list)]
    S["sizematched"] = {net: {k: ms([r["sizematched"][net][k] for r in reps]) for k in dk} for net in DESC}
    sm = lambda net, k: [r["sizematched"][net][k] for r in reps]
    for r in reps:
        for net in DESC:
            d = r["sizematched"][net]; d["cyc45"] = d["cyc4"] + d["cyc5"]
            d["h1_resid"] = d["h1"] - d["h1_null_mean"]
    # pre-specified tests of planted differences (Holm within the family)
    fam = [("NR-T3", "R-T3", "h1_z"), ("NR-T3", "R-T3", "h1"), ("NR-T3", "R-T3", "cyc45"),
           ("NR-T3", "R-T3", "loop_arc_recall"), ("NR-T3", "R-T3", "pos"), ("NR-T3", "NR-T2", "h1_z"),
           ("NR-T3", "HC-T3", "h1_z"), ("NR-T3", "HC-T3", "w2"), ("MDD-T1", "HC-T1", "frustration_exact"),
           ("NR-T3", "R-T3", "frustration_exact"), ("MDD-T1", "HC-T1", "dfrag"), ("NR-T3", "R-T3", "dfrag"),
           ("NR-T3", "R-T3", "auc"), ("MDD-T1", "HC-T1", "h1_z")]
    tests = {f"{a} vs {b}: {k}": paired(sm(a, k), sm(b, k)) for a, b, k in fam}
    adj = holm([v["p"] for v in tests.values()])
    for (k, v), a in zip(tests.items(), adj):
        v["p_holm"] = a
    S["group_tests"] = tests
    S["dfrag_one_sample"] = {net: paired(sm(net, "dfrag"), [0] * len(reps)) for net in DESC}
    # density association
    allh = [(r["sizematched"][n]["h1"], r["sizematched"][n]["edges"], r["sizematched"][n]["h1_log"],
             r["sizematched"][n]["h1_rank"], r["sizematched"][n]["h1_z"]) for r in reps for n in DESC]
    a = np.array(allh)
    S["h1_edges_spearman"] = dict(linear=float(stats.spearmanr(a[:, 0], a[:, 1])[0]),
                                  log=float(stats.spearmanr(a[:, 2], a[:, 1])[0]),
                                  rank=float(stats.spearmanr(a[:, 3], a[:, 1])[0]),
                                  z_vs_edges=float(stats.spearmanr(a[:, 4], a[:, 1])[0]),
                                  n=len(a))
    for kind in ("h1_log", "h1_rank"):
        S["group_tests"][f"NR-T3 vs R-T3: {kind}"] = paired(sm("NR-T3", kind), sm("R-T3", kind))
    # frustration: exact vs greedy
    S["frustration_greedy_minus_exact"] = ms([r["sizematched"][n]["frustration_greedy"] - r["sizematched"][n]["frustration_exact"]
                                             for r in reps for n in DESC])
    S["frustration_greedy_equal_exact"] = float(np.mean([abs(r["sizematched"][n]["frustration_greedy"] - r["sizematched"][n]["frustration_exact"]) < 1e-12
                                                       for r in reps for n in DESC]))
    S["cancelled_pairs"] = {n: ms(sm(n, "cancelled_pairs")) for n in DESC}
    # loop oracle
    lo = {}
    for bg in ("bg=True", "bg=False"):
        lo[bg] = {k: dict(n_classes=ms([r["loop_oracle"][bg][k][0] for r in reps]),
                          total=ms([r["loop_oracle"][bg][k][1] for r in reps]))
                  for k in reps[0]["loop_oracle"][bg]}
    for k in ("loop1_alone", "loop2_alone", "loop3_alone"):
        lo[k] = dict(n_classes=ms([r["loop_oracle"][k][0] for r in reps]))
    # per loop: does removing the loop's added arcs reduce the number of H1 classes?
    for bg in ("bg=True", "bg=False"):
        for L in (1, 2, 3):
            lo[bg][f"loop{L}_represented_frac"] = float(np.mean(
                [r["loop_oracle"][bg]["full"][0] > r["loop_oracle"][bg][f"without_loop{L}"][0] for r in reps]))
    S["loop_oracle"] = lo
    S["loops_full"] = {n: {k: ms([r["loops_full"][n][k] for r in reps]) for k in ("arc_recall", "loops_complete")}
                       for n in reps[0]["loops_full"]}
    # SPTS validation
    sp = {}
    for net in reps[0]["spts_validation"]:
        sp[net] = {}
        for var in reps[0]["spts_validation"][net]:
            if var == "scale_ratio":
                sp[net][var] = ms([r["spts_validation"][net][var] for r in reps]); continue
            sp[net][var] = {k: ms([r["spts_validation"][net][var].get(k) for r in reps])
                            for k in reps[0]["spts_validation"][net][var]}
    S["spts_validation"] = sp
    # ---------------- other scenarios
    S["scenarios"] = {}
    for lab in ("correct", "initdiag", "null", "tanh", "lag2", "glv", "t3", "driver", "hetero", "nonadd", "weak", "combined"):
        rr = load(lab)
        if not rr:
            continue
        d = dict(R=len(rr), recovery=recovery(rr))
        d["reversal"] = {t: reversal_summary(rr, "reversal", t) for t in ("naive", "cluster")}
        S["scenarios"][lab] = d
    # paired comparison with the correctly specified main scenario is not possible (different seeds);
    # scenario runs share seeds 3001.. with each other (common random numbers).
    json.dump(S, open("results/summary_group.json", "w"), indent=1)
    print("written", {k: (v["R"]) for k, v in S["scenarios"].items()}, S["main_R"])


if __name__ == "__main__":
    main()
