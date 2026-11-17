"""Revision: independent Monte Carlo replications of the group-level benchmark.
    python mc_group.py <label> <first_seed> <n_reps> [workers]
Labels define the scenario (see SCEN). One JSON file per replicate in results/mc/<label>/."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from tsmbb import benchmark as BM

STAT = dict(init="stationary", burn_in=20)
SCEN = {
    "main":      (STAT, dict()),
    "correct":   (STAT, dict(topo=False, stages=False, spts=False)),
    "initdiag":  (dict(), dict(topo=False, stages=False, spts=False)),
    "null":      (dict(STAT, null=True), dict(topo=False, stages=False, spts=False)),
    "tanh":      (dict(STAT, dynamics="tanh"), dict(topo=False, stages=False, spts=False)),
    "lag2":      (dict(STAT, dynamics="lag2"), dict(topo=False, stages=False, spts=False)),
    "glv":       (dict(STAT, dynamics="glv"), dict(topo=False, stages=False, spts=False)),
    "t3":        (dict(STAT, noise="t3"), dict(topo=False, stages=False, spts=False)),
    "driver":    (dict(STAT, driver=0.4), dict(topo=False, stages=False, spts=False)),
    "hetero":    (dict(STAT, hetero=0.3, hetero_drop=0.25), dict(topo=False, stages=False, spts=False)),
    "nonadd":    (dict(STAT, batch="nonadditive"), dict(topo=False, stages=False, spts=False)),
    "weak":      (dict(STAT, coupling_scale=0.7), dict(topo=False, stages=False, spts=False)),
    "combined":  (dict(STAT, dynamics="glv", noise="t3", driver=0.4, batch="nonadditive",
                       hetero=0.3, hetero_drop=0.25), dict(topo=False, stages=False, spts=False)),
}


def j(o):
    if isinstance(o, dict):
        return {str(k): j(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [j(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    if isinstance(o, np.ndarray):
        return j(o.tolist())
    return o


def one(args):
    label, seed = args
    path = f"results/mc/{label}/rep_{seed}.json"
    if os.path.exists(path):
        return path
    sc, kw = SCEN[label]
    t = time.time()
    r = BM.replicate(seed, sc, **kw)
    r["runtime_s"] = time.time() - t
    r["scenario"] = sc
    json.dump(j(r), open(path, "w"))
    print(f"{label} {seed} {time.time()-t:.0f}s", flush=True)
    return path


if __name__ == "__main__":
    label, s0, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    os.makedirs(f"results/mc/{label}", exist_ok=True)
    with Pool(w) as p:
        p.map(one, [(label, s) for s in range(s0, s0 + n)], chunksize=1)
