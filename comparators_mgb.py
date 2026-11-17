"""Revision: external directed-connectivity comparators on the microbiome benchmark
(Reviewers 1 and 3): l1-regularised VAR (LassoLarsIC, BIC), bivariate lag-1 Granger causality and
PCMCI (ParCorr, maximum lag 1, multiple-dataset mode), scored against the latent coupling graph on
the same pooled, demeaned transitions as Layer 1. Writes results/comparators_mgb.json."""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from tsmbb import generator as G, preprocess as PP, te as TE, benchmark as BM
from mc_group import j, SCEN

NETS = ("HC-T1", "MDD-T1", "R-T3", "NR-T3")


def pcmci_panel(pp, Zd, sids, w):
    from tigramite import data_processing as tpp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    data = {}
    for q in sids:
        sel = np.where((pp["window"] == w) & (pp["subject"] == q))[0]
        if len(sel) < 3:
            continue
        o = np.argsort(pp["week"][sel]); sel = sel[o]
        wk = pp["week"][sel]
        if not np.all(np.diff(wk) == 1):      # keep the longest run of consecutive weeks
            runs = np.split(np.arange(len(wk)), np.where(np.diff(wk) != 1)[0] + 1)
            sel = sel[max(runs, key=len)]
            if len(sel) < 3:
                continue
        data[len(data)] = Zd[sel]
    df = tpp.DataFrame(data, analysis_mode="multiple")
    m = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    r = m.run_pcmci(tau_min=1, tau_max=1, pc_alpha=0.05)
    q = m.get_corrected_pvalues(p_matrix=r["p_matrix"], fdr_method="fdr_bh", exclude_contemporaneous=True)
    R = q[:, :, 1] < 0.05
    np.fill_diagonal(R, False)
    return R, r["val_matrix"][:, :, 1]


def one(seed):
    sim = G.simulate(seed, scenario=SCEN["main"][0]); pp = PP.preprocess(sim)
    keep = pp["obs_index"]; ids = BM.group_ids(sim); Zd = BM.demean(pp["Z"], pp, "full")
    cm = BM.curated_mask(keep)
    out = dict(seed=seed)
    for net in NETS:
        g, w = net.split("-")
        X, Y, own = PP.transitions(pp, Zd, ids[g], w)
        T = G.truth_matrix(BM.true_B(sim, net), keep)
        res = {}
        t = time.time(); o = TE.cte_tests(X, Y, own)
        res["cTE_cluster"] = dict(BM.edge_metrics(o["cluster"]["R"], o["B"], T, cm), time=time.time() - t)
        t = time.time(); lv = TE.lasso_var(X, Y)
        res["lasso_VAR"] = dict(BM.edge_metrics(lv["R"], lv["B"], T, cm), time=time.time() - t)
        t = time.time(); bv = TE.bivariate_pooled(X, Y, len(np.unique(own)))
        res["bivariate_GC"] = dict(BM.edge_metrics(bv["R"], bv["B"], T, cm), time=time.time() - t)
        try:
            t = time.time(); Rp, V = pcmci_panel(pp, Zd, ids[g], w)
            res["PCMCI"] = dict(BM.edge_metrics(Rp, V, T, cm), time=time.time() - t)
        except Exception as e:
            res["PCMCI"] = dict(error=str(e)[:200])
        out[net] = res
    print(seed, flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 1) as p:
        res = p.map(one, range(1001, 1001 + n), chunksize=1)
    json.dump(j(res), open("results/comparators_mgb.json", "w"))
