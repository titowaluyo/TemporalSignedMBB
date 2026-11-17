"""Revision: Wilson-Cowan neural-mass benchmark (planted excitatory/inhibitory directed couplings,
sign switches between two conditions). Writes results/neuralmass.json."""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from tsmbb import neuralmass as NM, te as TE, evaluate as EV
from neural_demo import bic_k
from mc_group import j

FS = 250.0


def pooled_bic(segs, kmax):
    """Target history chosen by BIC on transitions pooled over segments, per channel; median."""
    ks = []
    for jn in range(segs[0].shape[1]):
        best, bk = np.inf, 1
        for p in range(1, kmax + 1):
            rs, n = 0.0, 0
            for X in segs:
                x = X[:, jn] - X[:, jn].mean()
                H = np.column_stack([x[kmax - h:len(x) - h] for h in range(1, p + 1)]); y = x[kmax:]
                b = np.linalg.lstsq(H, y, rcond=None)[0]; rs += ((y - H @ b) ** 2).sum(); n += len(y)
            bic = n * np.log(rs / n) + p * np.log(n)
            if bic < best:
                best, bk = bic, p
        ks.append(bk)
    return ks


def score(R, B, W):
    G = W != 0
    d = EV.edge_scores(R, G, True)
    tp = R & G
    d["sign"] = float((np.sign(B[tp]) == W[tp]).mean()) if tp.sum() else np.nan
    d["signed_recall"] = float((tp & (np.sign(B) == W)).sum() / G.sum())
    d["signed_precision"] = float((tp & (np.sign(B) == W)).sum() / max(R.sum(), 1))
    d["inh_recall"] = float((tp & (W < 0) & (np.sign(B) < 0)).sum() / max((W < 0).sum(), 1))
    d["exc_recall"] = float((tp & (W > 0) & (np.sign(B) > 0)).sum() / max((W > 0).sum(), 1))
    return {k: (float(v) if v is not None else None) for k, v in d.items()}


def reversals(rA, rB, WA, WB):
    both = rA["R"] & rB["R"]
    det = both & (np.sign(rA["B"]) != np.sign(rB["B"]))
    true = (WA != 0) & (WB != 0) & (WA != WB)
    return dict(true=int(true.sum()), detected=int(det.sum()), tp=int((det & true).sum()),
                fp=int((det & ~true).sum()))


def one(seed):
    rng = np.random.default_rng(seed)
    WA = NM.random_graph(rng); WB = NM.switch(WA, rng)
    out = dict(seed=seed)
    variants = {"direct": dict(), "strong": dict(w=1.0)}
    sims = {}
    for v, pv in variants.items():
        p = dict(NM.WC, **pv)
        sims[v] = [NM.simulate(W, 120, np.random.default_rng([seed, i, len(v)]), FS, p=p) for i, W in enumerate((WA, WB))]
    # sensor noise and volume conduction
    M = NM.mixing_matrix(WA.shape[0], 0.3)
    obs = {}
    for v in ("direct", "strong"):
        obs[v] = [X + 0.1 * X.std(0) * rng.normal(size=X.shape) for X in sims[v]]
    obs["mixing"] = [(X @ M.T) + 0.1 * X.std(0) * rng.normal(size=X.shape) for X in sims["direct"]]
    for v, (XA, XB) in obs.items():
        k = int(np.median(pooled_bic([XA[:5000], XB[:5000]], 30)))
        res = dict(k_bic=k)
        for name, kk, ll in (("k1", 1, 1), ("kBIC", k, 1), ("kBIC_lBIC", k, min(k, 8))):
            rA, rB = TE.cte_segments([XA], kk, l=ll), TE.cte_segments([XB], kk, l=ll)
            res[name] = dict(A=score(rA["R"], rA["B"], WA), B=score(rB["R"], rB["B"], WB),
                             reversal=reversals(rA, rB, WA, WB))
        out[v] = res
    # hemodynamic variant: 600 s per condition, BOLD sampled at TR = 0.72 s and 2.0 s
    p = dict(NM.WC)
    long = [NM.simulate(W, 600, np.random.default_rng([seed, i, 99]), 50.0, p=p) for i, W in enumerate((WA, WB))]
    for tr in (0.72, 2.0):
        step = int(round(tr * 50))
        bold = [NM.balloon(X, 1 / 50.0)[::step] for X in long]
        bold = [b + 0.1 * b.std(0) * rng.normal(size=b.shape) for b in bold]
        k = int(np.median(pooled_bic(bold, 5)))
        res = dict(k_bic=k, n_vol=len(bold[0]))
        for name, kk in (("k1", 1), ("kBIC", k)):
            rA, rB = TE.cte_segments([bold[0]], kk), TE.cte_segments([bold[1]], kk)
            res[name] = dict(A=score(rA["R"], rA["B"], WA), B=score(rB["R"], rB["B"], WB),
                             reversal=reversals(rA, rB, WA, WB),
                             undirected_A=EV.edge_scores(rA["R"], WA != 0, False))
        out[f"bold_TR{tr}"] = res
    out["n_edges"] = int((WA != 0).sum()); out["n_inh"] = int((WA < 0).sum())
    print(seed, flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as pool:
        res = pool.map(one, range(7001, 7001 + n), chunksize=1)
    json.dump(j(res), open("results/neuralmass.json", "w"))
