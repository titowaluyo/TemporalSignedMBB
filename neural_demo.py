"""Application of Layers 1-3 to public neural data (Section 2.12).

Data (no registration needed):
  NetSim simulation 3, NumPy format: https://github.com/sakhanna/SRU_for_GCI (data/netsim)
      -> data/netsim/sim3_subject_*.npz
  EEGLAB tutorial recording: https://github.com/sccn/eeglab (sample_data)
      -> data/eeglab/eeglab_data.set and eeglab_data.fdt
Writes results/neural.json and results/neural_arrays.npz.
"""
import glob, json, os, sys, time
import numpy as np
import scipy.io as sio
from scipy import signal, stats

from tsmbb import te as TE, topology as TP, evaluate as EV

NETSIM = os.environ.get("NETSIM_DIR", "data/netsim")
EEGDIR = os.environ.get("EEGLAB_DIR", "data/eeglab")


def bic_k(X, kmax):
    ks = []
    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean(); best, bk = np.inf, 1
        for p in range(1, kmax + 1):
            H = np.column_stack([x[kmax - h:len(x) - h] for h in range(1, p + 1)])
            y = x[kmax:]
            b = np.linalg.lstsq(H, y, rcond=None)[0]
            v = np.mean((y - H @ b) ** 2)
            bic = len(y) * np.log(v) + p * np.log(len(y))
            if bic < best:
                best, bk = bic, p
        ks.append(bk)
    return ks


def pcmci(Xs):
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    if len(Xs) > 1:
        df = pp.DataFrame({i: X for i, X in enumerate(Xs)}, analysis_mode="multiple")
    else:
        df = pp.DataFrame(Xs[0])
    m = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    r = m.run_pcmci(tau_min=1, tau_max=1, pc_alpha=0.05)
    q = m.get_corrected_pvalues(p_matrix=r["p_matrix"], fdr_method="fdr_bh", exclude_contemporaneous=True)
    R = q[:, :, 1] < 0.05
    np.fill_diagonal(R, False)
    return R


def parcorr(Xs, q=0.05):
    Z = np.vstack([X - X.mean(0) for X in Xs]); N = Z.shape[1]
    P = np.linalg.inv(np.cov(Z.T)); pc = -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
    df = Z.shape[0] - len(Xs) - (N - 2) - 3
    z = np.arctanh(np.clip(pc, -0.999999, 0.999999)) * np.sqrt(df)
    p = 2 * stats.norm.sf(np.abs(z)); iu = np.triu_indices(N, 1)
    R = np.zeros((N, N), bool); R[iu] = TE.bh(p[iu], q)[0]
    return R | R.T


def agg(v):
    v = np.array(v, float)
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), mcse=float(np.std(v, ddof=1) / np.sqrt(len(v))))


def run_netsim():
    files = sorted(glob.glob(os.path.join(NETSIM, "sim3_subject_*.npz")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    Xs, G = [], None
    for f in files:
        d = np.load(f)
        X = d["X_np"].T.astype(float); Xs.append((X - X.mean(0)) / X.std(0))
        if G is None:
            G = d["Gref"].astype(bool).T.copy(); np.fill_diagonal(G, False)
    ks = [k for X in Xs for k in bic_k(X, 5)]
    kb = int(np.median(ks))
    out = dict(n_subjects=len(Xs), T=Xs[0].shape[0], N=G.shape[0], true_edges=int(G.sum()), k_bic=kb)
    methods = {"cTE_k1": lambda S: TE.cte_segments(S, 1)["R"],
               "cTE_kBIC": lambda S: TE.cte_segments(S, kb)["R"],
               "bTE_k1": lambda S: TE.bte_segments(S, 1)["R"],
               "PCMCI": pcmci}
    pooled_R = {}
    for name, fn in methods.items():
        R = fn(Xs); pooled_R[name] = R
        d, u = EV.edge_scores(R, G, True), EV.edge_scores(R, G, False)
        tr = np.argwhere(G)
        corr_only = sum(R[i, j] and not R[j, i] for i, j in tr)
        both = sum(R[i, j] and R[j, i] for i, j in tr)
        rev_only = sum((not R[i, j]) and R[j, i] for i, j in tr)
        per = [(EV.edge_scores(fn([X]), G, True), EV.edge_scores(fn([X]), G, False)) for X in Xs] \
            if name != "PCMCI" else None
        if name == "PCMCI":
            per = []
            for X in Xs:
                Rq = pcmci([X]); per.append((EV.edge_scores(Rq, G, True), EV.edge_scores(Rq, G, False)))
        out[name] = dict(pooled=dict(directed=d, undirected=u, correct_only=int(corr_only), both=int(both),
                                     reversed_only=int(rev_only)),
                         single=dict(f1=agg([a["f1"] for a, _ in per]), und_f1=agg([b["f1"] for _, b in per]),
                                     recall=agg([a["recall"] for a, _ in per])))
        print(" ", name, round(d["precision"], 2), round(u["precision"], 2), out[name]["single"]["f1"]["mean"], flush=True)
    R = parcorr(Xs); u = EV.edge_scores(R, G, False)
    per = [EV.edge_scores(parcorr([X]), G, False) for X in Xs]
    out["parcorr"] = dict(pooled=dict(undirected=u), single=dict(und_f1=agg([p["f1"] for p in per]),
                          precision=agg([p["precision"] for p in per]), recall=agg([p["recall"] for p in per])))
    signs = TE.cte_segments(Xs, 1)
    tp = G & signs["R"]
    out["cTE_k1"]["pooled"]["positive_sign_of_detected_true"] = float((signs["B"][tp] > 0).mean())
    return out, G, pooled_R["cTE_k1"]


def run_eeg(rng):
    m = sio.loadmat(os.path.join(EEGDIR, "eeglab_data.set"), squeeze_me=True, struct_as_record=False)
    E = m["EEG"]
    fs, nb, pnts = float(E.srate), int(E.nbchan), int(E.pnts)
    X = np.fromfile(os.path.join(EEGDIR, "eeglab_data.fdt"), dtype="<f4").reshape(pnts, nb).astype(float)
    labels = [c.labels for c in E.chanlocs]
    keep = [i for i, l in enumerate(labels) if not l.upper().startswith("EOG")]
    X = X[:, keep]; labels = [labels[i] for i in keep]
    X = X - X.mean(1, keepdims=True)                         # common average reference
    b, a = signal.butter(4, [1 / (fs / 2), 30 / (fs / 2)], btype="band")
    X = signal.filtfilt(b, a, X, axis=0)
    cz = labels.index("Cz"); X = np.delete(X, cz, 1); labels.pop(cz)
    lat = np.round(np.array([e.latency for e in E.event if str(e.type) == "square"]) - 1).astype(int)
    w = int(0.5 * fs)
    ok = []
    for t, tn in enumerate(lat):
        prev = lat[t - 1] if t > 0 else -10 ** 9
        if tn - w >= prev + w and tn - w >= 0 and tn + w <= pnts and np.abs(X[tn - w:tn + w]).max() < 100:
            ok.append(tn)
    pre = [X[t - w:t] for t in ok]; post = [X[t:t + w] for t in ok]
    mpre, mpost = np.mean(pre, 0), np.mean(post, 0)
    pre = [p - mpre for p in pre]; post = [p - mpost for p in post]
    k = int(np.median([kk for s in pre for kk in bic_k(s, 12)]))
    f = lambda S: TE.cte_segments(S, k)
    rp, rq = f(pre), f(post)

    def rev(a, b):
        both = a["R"] & b["R"]
        return int((both & (np.sign(a["B"]) != np.sign(b["B"]))).sum()), int(both.sum()), int((a["R"] ^ b["R"]).sum())
    r_obs, shared, chg = rev(rp, rq)
    rn, cn = [], []
    for _ in range(200):
        flip = rng.random(len(ok)) < 0.5
        s1 = [q if fl else p for p, q, fl in zip(pre, post, flip)]
        s2 = [p if fl else q for p, q, fl in zip(pre, post, flip)]
        a_, _, c_ = rev(f(s1), f(s2)); rn.append(a_); cn.append(c_)
    rn, cn = np.array(rn), np.array(cn)
    odd, even = f(pre[0::2]), f(pre[1::2])
    hp = TP.density_null(rp["A"], 500, rng); hq = TP.density_null(rq["A"], 500, rng)
    out = dict(fs=fs, channels=len(labels), trials=len(ok), k_bic=k, edges_pre=int(rp["R"].sum()),
               edges_post=int(rq["R"].sum()), shared=shared, reversals=r_obs,
               rev_null_mean=float(rn.mean()), rev_null_95=float(np.percentile(rn, 95)),
               rev_p=float((1 + (rn >= r_obs).sum()) / 201), changed=chg,
               chg_null_mean=float(cn.mean()), chg_null_95=float(np.percentile(cn, 95)),
               chg_p=float((1 + (cn >= chg).sum()) / 201),
               split_half=[int(odd["R"].sum()), int(even["R"].sum()), int((odd["R"] & even["R"]).sum())],
               H1_pre=hp, H1_post=hq)
    return out, rn, cn


def main():
    t0 = time.time(); rng = np.random.default_rng(42)
    # consistency check against a known VAR(1) (seed 1)
    r1 = np.random.default_rng(1); N = 6; B = np.zeros((N, N)); B[0, 1] = 0.4; B[2, 3] = -0.4; B[1, 4] = 0.3
    np.fill_diagonal(B, 0.4)
    segs = []
    for _ in range(40):
        x = np.zeros(N); S = []
        for t in range(60):
            x = B.T @ x + r1.normal(0, 1, N); S.append(x.copy())
        segs.append(np.array(S))
    chk = TE.cte_segments(segs, 1)
    g = TE.Gram(N=N)
    for S in segs:
        S = S - S.mean(0); g = g + TE.Gram(S[:-1], S[1:])
    chk2 = TE.cte_from_gram(g, df_deduct=len(segs))
    off = ~np.eye(N, dtype=bool)
    res = dict(consistency=dict(max_abs_te_diff=float(np.abs(chk["TE"] - chk2["TE"])[off].max()),
                                same_edges=bool((chk["R"] == chk2["R"]).all())))
    ns, G, Rc = run_netsim(); res["netsim"] = ns
    print(f"[{time.time()-t0:.0f}s] netsim", flush=True)
    eeg, rn, cn = run_eeg(rng); res["eeg"] = eeg
    print(f"[{time.time()-t0:.0f}s] eeg", flush=True)
    json.dump(res, open("results/neural.json", "w"), indent=1, default=float)
    np.savez_compressed("results/neural_arrays.npz", G=G, R_cte=Rc, rev_null=rn, chg_null=cn)


if __name__ == "__main__":
    main()
