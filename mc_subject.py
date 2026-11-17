"""Revision: subject-level Monte Carlo (classification, response, design experiment).
    python mc_subject.py <label> <first_seed> <n_reps> [workers]"""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from tsmbb import generator as G, classify2 as C2
from mc_group import j

STAT = dict(init="stationary", burn_in=20)
SCEN = {
    "cc_main":     (STAT, 5, dict(full=True)),
    "cc_coupling": (dict(STAT, shift_scale=0.0), 5, dict(rp=False, full=False, n_inner=3)),
    "cc_calib":    (dict(STAT, shift_scale=0.2), 5, dict(rp=False, full=False, n_inner=3)),
    "rp_early":    (dict(STAT, early_prog=0.3), 5, dict(cc=False, full=False, n_inner=3)),
}
for v in (5, 10, 20):
    for c in (1.0, 0.7):
        SCEN[f"design_v{v}_c{c}"] = (dict(STAT, coupling_scale=c), v, dict(rp=False, full=False, light=True, n_inner=3))


def one(args):
    label, seed = args
    path = f"results/mc_subject/{label}/rep_{seed}.json"
    if os.path.exists(path):
        return path
    sc, v, kw = SCEN[label]
    t = time.time()
    sim = G.simulate(seed, visits_per_window=v, scenario=sc)
    r = C2.run(sim, seed, **kw)
    r["runtime_s"] = time.time() - t; r["scenario"] = sc; r["visits"] = v
    json.dump(j(r), open(path, "w"))
    print(f"{label} {seed} {time.time()-t:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    label, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    os.makedirs(f"results/mc_subject/{label}", exist_ok=True)
    with Pool(w) as p:
        p.map(one, [(label, s) for s in range(s0, s0 + n)], chunksize=1)
