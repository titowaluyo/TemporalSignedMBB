"""Revision: summarise subject-level replications (results/mc_subject/*), the LIONESS check,
the neural-mass benchmark, the motif replications and the CLR/phi checks into
results/summary_subject.json."""
import glob, json, os
import numpy as np
from aggregate_group import ms


def load(label):
    return [json.load(open(f)) for f in sorted(glob.glob(f"results/mc_subject/{label}/rep_*.json"))]


def summ_task(reps, task, sets):
    out = {}
    keys = [k for k in reps[0][task] if isinstance(reps[0][task][k], dict) and any(s in reps[0][task][k] for s in sets)]
    for k in keys:
        out[k] = {}
        for s in sets:
            if s in reps[0][task][k]:
                out[k][s] = {m: ms([r[task][k][s].get(m) for r in reps if k in r[task]])
                             for m in ("auc", "sens", "spec", "sens_test_optimised", "spec_test_optimised", "auprc")
                             if m in reps[0][task][k][s]}
        if "heldout_nocorrection" in reps[0][task][k]:
            out[k]["heldout_nocorrection"] = {"auc": ms([r[task][k]["heldout_nocorrection"]["auc"] for r in reps])}
        if "cv_auc" in reps[0][task][k]:
            out[k]["cv_auc"] = ms([r[task][k]["cv_auc"] for r in reps])
    diffs = [k for k in reps[0][task] if k.startswith("diff_")]
    for k in diffs:
        out[k] = dict(diff=ms([r[task][k]["diff"] for r in reps]),
                      n_sig=int(sum(r[task][k]["p"] < 0.05 for r in reps)),
                      n_ci_excl0=int(sum((r[task][k]["lo"] > 0) or (r[task][k]["hi"] < 0) for r in reps)),
                      R=len(reps))
    if "shap_share" in reps[0][task]:
        out["shap_share"] = {k: ms([r[task]["shap_share"][k] for r in reps]) for k in reps[0][task]["shap_share"]}
    return out


def main():
    S = {}
    for lab, tasks in (("cc_main", ("case_control", "response")), ("cc_coupling", ("case_control",)),
                       ("cc_calib", ("case_control",)), ("rp_early", ("response",))):
        reps = load(lab)
        if not reps:
            continue
        d = dict(R=len(reps))
        for t in tasks:
            sets = ("test", "heldout") if t == "case_control" else ("test", "heldout", "pooled")
            d[t] = summ_task(reps, t, sets)
        S[lab] = d
    # design experiment
    des = {}
    for f in sorted(glob.glob("results/mc_subject/design_*")):
        lab = os.path.basename(f); reps = load(lab)
        if reps:
            des[lab] = dict(R=len(reps), **summ_task(reps, "case_control", ("test", "heldout")))
    S["design"] = des
    # LIONESS check
    if os.path.exists("results/lioness_check.json"):
        L = json.load(open("results/lioness_check.json"))
        lio = {}
        for v in (5, 10, 20):
            rr = [r for r in L if r["visits"] == v]
            lio[v] = {}
            for key in ("observed|HC", "observed|MDD", "latent|HC", "latent|MDD"):
                lio[v][key] = {}
                for est in ("lioness", "shrink_cv", "population"):
                    lio[v][key][est] = {m: ms([r[key][est][m] for r in rr])
                                        for m in ("rmse", "sign", "dev_corr", "dev_corr_median", "subj_profile_corr")}
                lio[v][key]["lam_mult"] = [r[key]["shrink_cv"]["lam_mult"] for r in rr]
                lio[v][key]["mean_transitions"] = ms([r[key]["mean_transitions"] for r in rr])
        S["lioness"] = lio
    # neural mass
    if os.path.exists("results/neuralmass.json"):
        N = json.load(open("results/neuralmass.json"))
        nm = {}
        for v in ("direct", "strong", "mixing", "bold_TR0.72", "bold_TR2.0"):
            nm[v] = dict(k_bic=[r[v]["k_bic"] for r in N])
            for m in ("k1", "kBIC", "kBIC_lBIC"):
                if m not in N[0][v]:
                    continue
                d = {}
                for c in ("A", "B"):
                    for k in ("precision", "recall", "f1", "sign", "signed_recall", "signed_precision",
                              "inh_recall", "exc_recall"):
                        d[f"{c}_{k}"] = ms([r[v][m][c][k] for r in N])
                for k in ("precision", "recall", "f1", "sign", "signed_recall", "inh_recall", "exc_recall"):
                    d[k] = ms([np.nanmean([r[v][m]["A"][k], r[v][m]["B"][k]]) if r[v][m]["A"][k] is not None
                               and r[v][m]["B"][k] is not None else None for r in N])
                rv = [r[v][m]["reversal"] for r in N]
                d["rev_true"] = int(sum(x["true"] for x in rv)); d["rev_tp"] = int(sum(x["tp"] for x in rv))
                d["rev_fp"] = int(sum(x["fp"] for x in rv)); d["rev_sens"] = d["rev_tp"] / max(d["rev_true"], 1)
                if "undirected_A" in N[0][v][m]:
                    d["undirected_precision"] = ms([r[v][m]["undirected_A"]["precision"] for r in N])
                    d["undirected_recall"] = ms([r[v][m]["undirected_A"]["recall"] for r in N])
                nm[v][m] = d
        nm["R"] = len(N)
        S["neuralmass"] = nm
    # motifs
    if os.path.exists("results/motif_mc.json"):
        M = json.load(open("results/motif_mc.json"))["reps"]
        R = len(M)
        hc = ("HC-T1", "HC-T2", "HC-T3")
        dis_nets = ("MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3")
        n_dis = []
        cls = {}
        for r in M:
            hce = set(k for n in hc for k in r[n]["enriched"])
            nd = 0
            for n in dis_nets:
                for k in r[n]["enriched"]:
                    if k not in hce:
                        nd += 1; cls.setdefault((n, k), 0); cls[(n, k)] += 1
            n_dis.append(nd)
        top = sorted(cls.items(), key=lambda x: -x[1])[:8]
        # direct test MDD-T1 vs HC-T1 on z scores, classes present in both in >= half of reps
        from scipy import stats
        keys = set(k for r in M for k in r["MDD-T1"]["z"]) & set(k for r in M for k in r["HC-T1"]["z"])
        direct = []
        for k in keys:
            pairs = [(r["MDD-T1"]["z"][k], r["HC-T1"]["z"][k]) for r in M if k in r["MDD-T1"]["z"] and k in r["HC-T1"]["z"]]
            if len(pairs) >= R // 2:
                d = np.array([a - b for a, b in pairs])
                t = stats.ttest_1samp(d, 0)
                direct.append(dict(cls=k, n=len(d), mean=float(d.mean()), p=float(t.pvalue)))
        from aggregate_group import holm
        if direct:
            adj = holm([x["p"] for x in direct])
            for x, a in zip(direct, adj):
                x["p_holm"] = a
        S["motifs"] = dict(R=R, n_disease_specific=ms(n_dis), frac_reps_any=float(np.mean(np.array(n_dis) > 0)),
                           most_reproducible=[dict(net=n, cls=k, frac=c / R) for (n, k), c in top],
                           n_enriched={n: ms([len(r[n]["enriched"]) for r in M]) for n in M[0]},
                           n_classes={n: ms([r[n]["n_classes"] for r in M]) for n in M[0]},
                           direct_mdd_vs_hc=sorted(direct, key=lambda x: x["p"])[:10],
                           n_direct_tested=len(direct),
                           n_direct_sig_holm=int(sum(x["p_holm"] < 0.05 for x in direct)))
    # CLR omitted coordinate and phi checks
    if os.path.exists("results/clr_phi_checks.json"):
        C = json.load(open("results/clr_phi_checks.json"))
        refs = ["Flavonifractor", "Holdemania", "Turicibacter", "Christensenella"]
        cl = {}
        for ref in refs:
            for net in ("HC-T1", "MDD-T1", "NR-T3"):
                cl[f"{ref}|{net}"] = {m: ms([r[f"{ref}|{net}"][m] for r in C]) for m in ("precision", "recall", "sign")}
        ag = {}
        for k in C[0]["agreement_vs_Flavonifractor"]:
            ag[k] = dict(jaccard=ms([r["agreement_vs_Flavonifractor"][k]["jaccard"] for r in C]),
                         sign_disagree=int(sum(r["agreement_vs_Flavonifractor"][k]["sign_disagree_shared"] for r in C)))
        ph = {net: {m: ms([r["phi"][net][m] for r in C]) for m in C[0]["phi"][net]} for net in C[0]["phi"]}
        maxchg = max(abs(cl[f"{ref}|{net}"][m]["mean"] - cl[f"Flavonifractor|{net}"][m]["mean"])
                     for ref in refs[1:] for net in ("HC-T1", "MDD-T1", "NR-T3") for m in ("precision", "recall"))
        S["clr"] = dict(metrics=cl, agreement=ag, max_change=maxchg,
                        jaccard_min=min(v["jaccard"]["mean"] for v in ag.values()),
                        sign_disagree_total=int(sum(v["sign_disagree"] for v in ag.values())), R=len(C))
        S["phi"] = ph
    json.dump(S, open("results/summary_subject.json", "w"), indent=1)
    print("written", list(S))


if __name__ == "__main__":
    main()
