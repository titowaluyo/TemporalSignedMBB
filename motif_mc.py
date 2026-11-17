"""Revision: signed-motif enrichment across independent replications with a null model that
preserves each node's positive and negative in- and out-degrees (Reviewer 1 minor 8), and a direct
disease-versus-control comparison of class z-scores across replications (Reviewer 3 add. 1).
Writes results/motif_mc.json."""
import json, sys
import numpy as np
from multiprocessing import Pool
from tsmbb import generator as G, preprocess as PP, te as TE, topology as TP, benchmark as BM
from mc_group import j, SCEN

NETS = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]


def one(seed, n_null=100):
    sim = G.simulate(seed, scenario=SCEN["main"][0]); pp = PP.preprocess(sim)
    ids = BM.group_ids(sim); Zd = BM.demean(pp["Z"], pp, "full")
    rng = np.random.default_rng(seed)
    out = {}
    for net in NETS:
        g, w = net.split("-")
        X, Y, own = PP.transitions(pp, Zd, ids[g], w)
        A = TE.cte_tests(X, Y, own)["cluster"]["A"]
        S = np.sign(A).astype(int)
        e = TP.motif_enrichment_signed(S, n_null, rng)
        keys = list(e["p"])
        rej, qv = TE.bh([e["p"][k] for k in keys], 0.10) if keys else ([], [])
        z = {}
        for k in keys:
            nl = np.array(e["null"][k], float)
            z[str(k)] = float((e["counts"][k] - nl.mean()) / (nl.std(ddof=1) + 1e-9))
        out[net] = dict(n_classes=e["n_classes"], enriched=[str(k) for k, r in zip(keys, rej) if r],
                        z=z, counts={str(k): int(e["counts"][k]) for k in keys})
    print(seed, flush=True)
    return out


if __name__ == "__main__":
    seeds = range(1001, 1031)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        res = p.map(one, seeds, chunksize=1)
    json.dump(j(dict(seeds=list(seeds), reps=res)), open("results/motif_mc.json", "w"))
