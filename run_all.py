"""Main MGB analysis (Sections 2.1-2.9, 3.1-3.6). Writes results/main.json and results/main_arrays.npz.

    python run_all.py            # seed 42
"""
import json, time, sys
import numpy as np
from scipy import stats

from tsmbb import generator as G, preprocess as PP, te as TE, topology as TP, evaluate as EV
from tsmbb import classify as CL

SEED, N_SUB, SUB_SIZE, N_NULL_MOTIF = 42, 100, 200, 300
NETS = G.GROUP_WINDOWS                       # 8 networks of Table 2
TRAJ = {"HC": ["HC-T1", "HC-T2", "HC-T3"], "R": ["R-T1", "R-T2", "R-T3"], "NR": ["NR-T1", "NR-T2", "NR-T3"]}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    global N_SUB, N_NULL_MOTIF
    quick = "--quick" in sys.argv
    if quick:
        N_SUB, N_NULL_MOTIF = 2, 20
    t0 = time.time()
    if "--subject-only" in sys.argv:            # recompute the subject-level part only
        sim = G.simulate(SEED); pp = PP.preprocess(sim)
        res = json.load(open("results/main.json"))
        res["subject_level"] = jsonable(CL.run(sim, pp, SEED, full=True))
        json.dump(jsonable(res), open("results/main.json", "w"), indent=1)
        print(f"subject level done in {time.time()-t0:.0f}s"); return
    sim = G.simulate(SEED)
    pp = PP.preprocess(sim)
    Zd = PP.demean_by_subject(pp)
    keep = pp["obs_index"]
    names = pp["names"]
    subj = sim["subjects"]
    ids = {t: np.array([s["subject"] for s in subj if s["traj"] in t]) for t in ("HC", "R", "NR")}
    ids["MDD"] = np.concatenate([ids["R"], ids["NR"]])
    res = dict(seed=SEED, nodes=names)

    # ---------------- cohort table and preprocessing summary
    tab = []
    for c in ("SC1", "SC2", "SC3", "Combined"):
        S = [s for s in subj if c == "Combined" or s["cohort"] == c]
        M = [s for s in S if s["group"] == "MDD"]
        ham = np.array([s["hamd0"] for s in M]); age = np.array([s["age"] for s in S])
        nres = sum(s["responder"] == 1 for s in M)
        nq = int(np.sum(np.isin(pp["subject"], [s["subject"] for s in S])))
        tab.append(dict(cohort=c, mdd=len(M), hc=len(S) - len(M),
                        female=100 * np.mean([s["female"] for s in S]),
                        age_mean=age.mean(), age_sd=age.std(ddof=1), hamd_mean=ham.mean(),
                        hamd_sd=ham.std(ddof=1), responders=nres, responders_pct=100 * nres / len(M),
                        qc_samples=nq))
    res["cohorts"] = tab
    res["preprocess"] = {k: pp[k] for k in ("n_scheduled", "n_missed", "n_generated", "n_qc",
                                            "n_asv_kept", "r2_cohort_before", "r2_cohort_after")}
    sel = pp["window"] == "T1"
    sh = {}
    for g in ("HC", "MDD"):
        sh[g] = np.array([pp["shannon"][sel & (pp["subject"] == q)].mean() for q in ids[g]])
    t = stats.ttest_ind(sh["MDD"], sh["HC"], nan_policy="omit")
    res["shannon_T1"] = dict(mdd_mean=float(np.nanmean(sh["MDD"])), mdd_sd=float(np.nanstd(sh["MDD"], ddof=1)),
                             hc_mean=float(np.nanmean(sh["HC"])), hc_sd=float(np.nanstd(sh["HC"], ddof=1)),
                             t=float(t.statistic), p=float(t.pvalue))
    cz = np.corrcoef
    # case-control R2 at T1 (share of variance explained by group, after correction)
    grp = np.array([1.0 if s == "MDD" else 0.0 for s in
                    [next(x["group"] for x in [subj[q]]) for q in pp["subject"][sel]]])
    Xg = np.column_stack([np.ones(sel.sum()), grp])
    Zt = pp["Z"][sel]
    fit = Xg @ np.linalg.lstsq(Xg, Zt, rcond=None)[0]
    res["preprocess"]["r2_case_control_T1"] = float(1 - ((Zt - fit) ** 2).sum() / ((Zt - Zt.mean(0)) ** 2).sum())
    print(f"[{time.time()-t0:.0f}s] simulated and preprocessed", flush=True)

    def truth(net):
        g, w = net.split("-")
        gk = ("MDD", w) if g == "MDD" else (g, w)
        return G.truth_matrix(sim["B"][gk], keep)

    def build(net, sids):
        g, w = net.split("-")
        X, Y, _ = PP.transitions(pp, Zd, sids, w)
        return TE.cte_from_gram(TE.Gram(X, Y))

    def members(net):
        return ids[net.split("-")[0]]

    # ---------------- full-size networks
    full = {}
    allnets = [f"{g}-{w}" for g, w in NETS] + ["R-T1", "NR-T1"]
    for net in allnets:
        full[net] = build(net, members(net))
    res["full"] = {}
    for net in allnets:
        r = full[net]
        p_, rc, sa = EV.edge_recovery(r["R"], r["B"], truth(net))
        res["full"][net] = dict(n_transitions=r["n"], edges=int(r["R"].sum()), precision=p_, recall=rc, sign=sa)
    # Fig. 3 quantities
    I = {n: i for i, n in enumerate(names)}
    def edge(net, s, t):
        r = full[net]; return dict(te=float(r["TE"][I[s], I[t]]), q=float(r["Q"][I[s], I[t]]),
                                   sign=int(np.sign(r["B"][I[s], I[t]])), retained=bool(r["R"][I[s], I[t]]))
    res["fig3_edges"] = {f"{net}:{s}->{t}": edge(net, s, t) for net in ("HC-T1", "MDD-T1")
                         for s, t in [("F. prausnitzii", "5-HT synthesis"), ("F. prausnitzii", "Butyrate"),
                                      ("R. gnavus", "IDO1/Kyn"), ("F. prausnitzii", "R. gnavus"),
                                      ("R. gnavus", "F. prausnitzii"), ("A. putredinis", "IDO1/Kyn")]}
    # KSG check on MDD-T1 retained edges
    X, Y, _ = PP.transitions(pp, Zd, ids["MDD"], "T1")
    r = full["MDD-T1"]
    E = np.argwhere(r["R"])
    ksg, gb, gc = [], [], []
    for i, j in E:
        ksg.append(TE.ksg_cmi(Y[:, j], X[:, i], X[:, j], k=4))
        A = np.column_stack([X[:, j]]); b0 = np.linalg.lstsq(A, Y[:, j], rcond=None)[0]
        r0 = ((Y[:, j] - A @ b0) ** 2).sum()
        A2 = np.column_stack([X[:, j], X[:, i]]); b1 = np.linalg.lstsq(A2, Y[:, j], rcond=None)[0]
        r1 = ((Y[:, j] - A2 @ b1) ** 2).sum()
        gb.append(0.5 * np.log2(r0 / r1)); gc.append(r["TE"][i, j])
    rb, pb = stats.spearmanr(ksg, gb); rcd, pc = stats.spearmanr(ksg, gc)
    res["ksg_check"] = dict(n=len(E), rho_bivariate=float(rb), p_bivariate=float(pb),
                            rho_conditional=float(rcd), p_conditional=float(pc))
    # full-size TESTM Fisher test
    def pairs(seq):
        flips = tot = 0
        for a, b in zip(seq[:-1], seq[1:]):
            A1, A2 = full[a]["A"], full[b]["A"]; both = (A1 != 0) & (A2 != 0)
            tot += both.sum(); flips += (both & (np.sign(A1) != np.sign(A2))).sum()
        return int(flips), int(tot)
    fm = [pairs(TRAJ["R"]), pairs(TRAJ["NR"])]; fh = pairs(TRAJ["HC"])
    fl_m, tot_m = sum(f for f, _ in fm), sum(t for _, t in fm)
    odds, pf = stats.fisher_exact([[fl_m, tot_m - fl_m], [fh[0], fh[1] - fh[0]]])
    res["full_testm"] = dict(mdd_flips=fl_m, mdd_pairs=tot_m, hc_flips=fh[0], hc_pairs=fh[1], fisher_p=float(pf))
    print(f"[{time.time()-t0:.0f}s] full-size networks", flush=True)

    # ---------------- motifs on full-size networks
    rngm = np.random.default_rng(SEED)
    mot = {}
    for net, _ in [(f"{g}-{w}", 0) for g, w in NETS]:
        S = np.sign(full[net]["A"]).astype(int)
        e = TP.motif_enrichment(S, N_NULL_MOTIF, rngm)
        keys = list(e["p"])
        rej, qv = TE.bh([e["p"][k] for k in keys], 0.10)
        mot[net] = dict(n_classes=e["n_classes"], keys=keys, q=dict(zip(keys, qv)),
                        enriched=[k for k, r_ in zip(keys, rej) if r_], instances=e["instances"])
    hc_enr = set(k for n in ("HC-T1", "HC-T2", "HC-T3") for k in mot[n]["enriched"])
    disease = {}
    for net in ("MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"):
        for k in mot[net]["enriched"]:
            if k not in hc_enr:
                disease.setdefault(str(k), dict(code=k, nets=[], q=[], instances=None))
                disease[str(k)]["nets"].append(net); disease[str(k)]["q"].append(mot[net]["q"][k])
                if disease[str(k)]["instances"] is None:
                    insts = mot[net]["instances"][k]
                    disease[str(k)]["instances"] = [[names[a] for a in t] for t in insts]
                    disease[str(k)]["arcs"] = [[f"{names[a]} {'->' if full[net]['A'][a, b] > 0 else '-|'} {names[b]}"
                                                for a in t for b in t if a != b and full[net]["A"][a, b] != 0]
                                               for t in insts]
                    disease[str(k)]["arcs_hc"] = [[f"{names[a]} {'->' if full['HC-T1']['A'][a, b] > 0 else '-|'} {names[b]}"
                                                   for a in t for b in t if a != b and full["HC-T1"]["A"][a, b] != 0]
                                                  for t in insts]
                    disease[str(k)]["first_net"] = net
    res["motifs"] = dict(n_classes={n: mot[n]["n_classes"] for n in mot},
                         n_enriched={n: len(mot[n]["enriched"]) for n in mot},
                         hc_enriched=[str(k) for k in hc_enr], disease_specific=disease)
    print(f"[{time.time()-t0:.0f}s] motifs", flush=True)

    # ---------------- subsamples
    rng = np.random.default_rng(SEED + 1)
    labels = [f"{g}-{w}" for g, w in NETS]
    metr = {n: {k: [] for k in ("edges", "pos", "mean_te", "frustration", "auc", "dfrag", "h1",
                                 "w2", "precision", "recall", "sign")} for n in labels}
    planted = [(I[s], I[t]) for s, t in G.REVERSAL]
    rev_freq = {k: np.zeros((len(planted),)) for k in
                ("HC:T1T2", "HC:T2T3", "R:T1T2", "R:T2T3", "NR:T1T2", "NR:T2T3")}
    rev_counts = {k: [] for k in rev_freq}
    unplanted_rev = 0
    other_flips = []
    share = {k: [] for k in rev_freq}
    curves = {n: [] for n in labels}
    example = None
    for it in range(N_SUB):
        hc = rng.permutation(ids["HC"]); hc_s, hc_ref = hc[:SUB_SIZE], hc[SUB_SIZE:2 * SUB_SIZE]
        smp = dict(HC=hc_s, MDD=rng.choice(ids["MDD"], SUB_SIZE, replace=False),
                   R=rng.choice(ids["R"], SUB_SIZE, replace=False),
                   NR=rng.choice(ids["NR"], SUB_SIZE, replace=False))
        nets = {}
        for net in labels + ["R-T1", "NR-T1"]:
            nets[net] = build(net, smp[net.split("-")[0]])
        ref = {w: build(f"HC-{w}", hc_ref) for w in ("T1", "T2", "T3")}
        ref_h1 = {w: TP.persistence(ref[w]["A"])[0] for w in ref}
        pers = {n: TP.persistence(nets[n]["A"], cocycles=True) for n in nets}
        for tr, seq in TRAJ.items():
            for a, b, key in [(seq[0], seq[1], f"{tr}:T1T2"), (seq[1], seq[2], f"{tr}:T2T3")]:
                A1, A2 = nets[a]["A"], nets[b]["A"]
                both = (A1 != 0) & (A2 != 0); flip = both & (np.sign(A1) != np.sign(A2))
                rev_counts[key].append(int(flip.sum()))
                share[key].append([float((both & (A1 > 0) & (A2 > 0)).sum() / max(both.sum(), 1)),
                                   float((both & (A1 < 0) & (A2 < 0)).sum() / max(both.sum(), 1)),
                                   float(flip.sum() / max(both.sum(), 1))])
                for e, (i, j) in enumerate(planted):
                    rev_freq[key][e] += flip[i, j]
                pm = np.zeros_like(flip); pm[tuple(np.array(planted).T)] = True
                unplanted_rev += int((flip & ~pm).sum())
                for i, j in np.argwhere(flip & ~pm):
                    other_flips.append(f"{key}: {names[i]} -> {names[j]}")
        for net in labels:
            r = nets[net]; A = r["A"]; g, w = net.split("-")
            nz = A[A != 0]
            p_, rc, sa = EV.edge_recovery(r["R"], r["B"], truth(net))
            seq = TRAJ.get(g, None)
            if g == "MDD":
                Aseq, phis = [A], [pers[net][2]]
            else:
                upto = seq[:["T1", "T2", "T3"].index(w) + 1]
                Aseq, phis = [nets[x]["A"] for x in upto], [pers[x][2] for x in upto]
            order = np.argsort(-TP.spts(Aseq, phis))
            rs = TP.resilience(A, order, 20, rng)
            curves[net].append(rs["St"])
            m = metr[net]
            m["edges"].append(int((A != 0).sum())); m["pos"].append(100 * float((nz > 0).mean()))
            m["mean_te"].append(100 * float(np.abs(nz).mean())); m["frustration"].append(TP.frustration(A, 20, rng))
            m["auc"].append(rs["auc"]); m["dfrag"].append(rs["dfrag"]); m["h1"].append(pers[net][1])
            m["w2"].append(TP.wasserstein2(pers[net][0], ref_h1[w]))
            m["precision"].append(p_); m["recall"].append(rc); m["sign"].append(sa)
            if it == 0 and net in ("HC-T1", "NR-T3"):
                example = example or {}
                example[net] = pers[net][0].tolist()
                example[net + "_random_curve"] = rs["Sr"].tolist()
        if it % 10 == 0:
            print(f"[{time.time()-t0:.0f}s] subsample {it}", flush=True)
    summ = {n: {k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)),
                        p2_5=float(np.nanpercentile(v, 2.5)), p97_5=float(np.nanpercentile(v, 97.5)))
                for k, v in metr[n].items()} for n in labels}
    res["subsamples"] = summ
    def pdiff(a, b, key):
        d = np.array(metr[a][key]) - np.array(metr[b][key])
        return dict(mean=float(d.mean()), p2_5=float(np.percentile(d, 2.5)), p97_5=float(np.percentile(d, 97.5)))
    res["paired"] = {"h1_NR3_R3": pdiff("NR-T3", "R-T3", "h1"), "h1_NR3_MDD1": pdiff("NR-T3", "MDD-T1", "h1"),
                     "h1_NR3_NR2": pdiff("NR-T3", "NR-T2", "h1"), "h1_NR3_HC3": pdiff("NR-T3", "HC-T3", "h1"),
                     "w2_NR3_HC3": pdiff("NR-T3", "HC-T3", "w2"), "pos_NR3_R3": pdiff("NR-T3", "R-T3", "pos"),
                     "auc_NR3_HC1": pdiff("NR-T3", "HC-T1", "auc")}
    mh = [summ[n]["h1"]["mean"] for n in labels]; me = [summ[n]["edges"]["mean"] for n in labels]
    res["rho_h1_edges"] = float(stats.spearmanr(mh, me)[0])
    res["reversals"] = dict(
        counts={k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in rev_counts.items()},
        freq={k: dict(zip([f"{s} -> {t}" for s, t in G.REVERSAL], (v / N_SUB).tolist())) for k, v in rev_freq.items()},
        unplanted_total=unplanted_rev, other_flips=other_flips,
        share={k: np.mean(v, 0).tolist() for k, v in share.items()})
    print(f"[{time.time()-t0:.0f}s] group-level done", flush=True)

    # ---------------- subject level
    res["subject_level"] = {} if quick else CL.run(sim, pp, SEED, full=True)
    print(f"[{time.time()-t0:.0f}s] subject level done", flush=True)
    res["runtime_s"] = time.time() - t0
    json.dump(jsonable(res), open("results/main_quick.json" if quick else "results/main.json", "w"), indent=1)
    np.savez_compressed("results/main_arrays.npz",
                        A_HC_T1=full["HC-T1"]["A"], A_MDD_T1=full["MDD-T1"]["A"],
                        Q_HC_T1=full["HC-T1"]["Q"], Q_MDD_T1=full["MDD-T1"]["Q"],
                        qc_week=pp["week"], qc_subject=pp["subject"],
                        qc_cohort=pp["cohort"],
                        qc_group=np.array([subj[q]["group"] for q in pp["subject"]]),
                        **{f"curve_{n}": np.array(curves[n]) for n in labels},
                        **{f"h1_{n}": np.array(metr[n]["h1"]) for n in labels},
                        **{f"prec_{n}": np.array(metr[n]["precision"]) for n in labels},
                        **{f"rec_{n}": np.array(metr[n]["recall"]) for n in labels},
                        **{f"auc_{n}": np.array(metr[n]["auc"]) for n in labels})
    json.dump(jsonable(example), open("results/example_diagrams.json", "w"))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
