"""Revision analyses of the public neural data (Reviewers 2, 3, 4, 5, 6).
NetSim sim3: additional comparators (PCMCI+, Patel's tau, DirectLiNGAM) and direction accuracy
on the true connections. EEGLAB recording: extended, label-invariant lag selection, residual
whiteness, filter/reference sensitivity, and permutations that re-estimate the evoked means.
Writes results/neural_revision.json."""
import glob, hashlib, json, os, sys, time
import numpy as np
import scipy.io as sio
from scipy import signal, stats
from tsmbb import te as TE, topology as TP, evaluate as EV
from neural_demo import NETSIM, EEGDIR, parcorr, pcmci
from neural_mass_bench import pooled_bic
from mc_group import j


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_netsim():
    files = sorted(glob.glob(os.path.join(NETSIM, "sim3_subject_*.npz")),
                   key=lambda f: int(f.split("_")[-1].split(".")[0]))
    Xs, G = [], None
    for f in files:
        d = np.load(f)
        X = d["X_np"].T.astype(float); Xs.append((X - X.mean(0)) / X.std(0))
        if G is None:
            G = d["Gref"].astype(bool).T.copy(); np.fill_diagonal(G, False)
    return Xs, G, files


def patel(Xs):
    """Patel et al. (2006) kappa and tau on [0,1]-scaled data (pooled over subjects).
    tau[i, j] > 0 means i -> j."""
    Z = np.vstack([(X - np.percentile(X, 10, 0)) / (np.percentile(X, 90, 0) - np.percentile(X, 10, 0))
                   for X in Xs])
    Z = np.clip(Z, 0, 1)
    N = Z.shape[1]
    tau = np.zeros((N, N)); kap = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            A, B = Z[:, a], Z[:, b]
            t1 = np.mean(A * B); t2 = np.mean(A * (1 - B)); t3 = np.mean((1 - A) * B)
            t4 = np.mean((1 - A) * (1 - B))
            E = (t1 + t2) * (t1 + t3)
            D = min(t1 + t2, t1 + t3) - E if t1 > E else E - max(0, 2 * t1 + t2 + t3 - 1)
            kap[a, b] = (t1 - E) / D if D else 0
            tau[a, b] = (1 - (t1 + t3) / (t1 + t2)) if t2 > t3 else ((t1 + t2) / (t1 + t3) - 1)
    return kap, tau


def direction_accuracy(score, G):
    """Fraction of true connections i -> j with score[i, j] > score[j, i]."""
    tr = np.argwhere(G)
    return float(np.mean([score[i, j] > score[j, i] for i, j in tr]))


def run_netsim():
    Xs, G, files = load_netsim()
    out = dict(files_sha256={os.path.basename(f): sha256(f) for f in files})
    # direction accuracy on the true connections, pooled data
    cte = TE.cte_segments(Xs, 1)
    out["dir_acc_cTE_k1"] = direction_accuracy(cte["TE"], G)
    cte2 = TE.cte_segments(Xs, 2)
    out["dir_acc_cTE_k2"] = direction_accuracy(cte2["TE"], G)
    bte = TE.bte_segments(Xs, 1)
    out["dir_acc_bTE"] = direction_accuracy(bte["TE"], G)
    kap, tau = patel(Xs)
    out["dir_acc_patel_tau"] = direction_accuracy(tau, G)
    per = [direction_accuracy(patel([X])[1], G) for X in Xs]
    out["dir_acc_patel_tau_single"] = dict(mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)),
                                           mcse=float(np.std(per, ddof=1) / np.sqrt(len(per))))
    per = [direction_accuracy(TE.cte_segments([X], 1)["TE"], G) for X in Xs]
    out["dir_acc_cTE_single"] = dict(mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)),
                                     mcse=float(np.std(per, ddof=1) / np.sqrt(len(per))))
    # DirectLiNGAM on pooled data (instantaneous model)
    from lingam import DirectLiNGAM
    Z = np.vstack(Xs)
    m = DirectLiNGAM(random_state=0).fit(Z)
    Bl = m.adjacency_matrix_.T            # B[i, j]: i -> j
    out["dir_acc_lingam"] = direction_accuracy(np.abs(Bl), G)
    Rl = (np.abs(Bl) > 0.05)
    out["lingam"] = dict(directed=EV.edge_scores(Rl, G, True), undirected=EV.edge_scores(Rl, G, False))
    # PCMCI+ (contemporaneous and lagged links)
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    df = pp.DataFrame({i: X for i, X in enumerate(Xs)}, analysis_mode="multiple")
    pc = PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    r = pc.run_pcmciplus(tau_min=0, tau_max=1, pc_alpha=0.01)
    gr = r["graph"]
    N = G.shape[0]
    R = np.zeros((N, N), bool); U = np.zeros((N, N), bool)
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            if gr[a, b, 1] == "-->" or gr[a, b, 0] == "-->":
                R[a, b] = True
            if gr[a, b, 0] in ("o-o", "x-x", "-->", "<--") or gr[a, b, 1] == "-->":
                U[a, b] = True
    out["pcmciplus"] = dict(directed=EV.edge_scores(R, G, True), undirected=EV.edge_scores(U | U.T | R | R.T, G, False),
                            n_unoriented_contemp=int(((gr[:, :, 0] == "o-o") | (gr[:, :, 0] == "x-x")).sum() // 2))
    sc = np.where(R, 1.0, 0.0)
    out["dir_acc_pcmciplus_oriented_only"] = float(np.mean([R[i, jj] and not R[jj, i] for i, jj in np.argwhere(G)]))
    return out


def load_eeg(band=(1, 30), reference="average"):
    m = sio.loadmat(os.path.join(EEGDIR, "eeglab_data.set"), squeeze_me=True, struct_as_record=False)
    E = m["EEG"]
    fs, nb, pnts = float(E.srate), int(E.nbchan), int(E.pnts)
    X = np.fromfile(os.path.join(EEGDIR, "eeglab_data.fdt"), dtype="<f4").reshape(pnts, nb).astype(float)
    labels = [c.labels for c in E.chanlocs]
    keep = [i for i, l in enumerate(labels) if not l.upper().startswith("EOG")]
    X = X[:, keep]; labels = [labels[i] for i in keep]
    if reference == "average":
        X = X - X.mean(1, keepdims=True)
    b, a = signal.butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    X = signal.filtfilt(b, a, X, axis=0)
    if reference == "average":
        cz = labels.index("Cz"); X = np.delete(X, cz, 1); labels.pop(cz)
    lat = np.round(np.array([e.latency for e in E.event if str(e.type) == "square"]) - 1).astype(int)
    w = int(0.5 * fs)
    ok = []
    for t, tn in enumerate(lat):
        prev = lat[t - 1] if t > 0 else -10 ** 9
        if tn - w >= prev + w and tn - w >= 0 and tn + w <= pnts and np.abs(X[tn - w:tn + w]).max() < 100:
            ok.append(tn)
    pre = [X[t - w:t] for t in ok]; post = [X[t:t + w] for t in ok]
    return fs, labels, pre, post


def ljung_box_resid(segs, k, lags=10):
    """Residual whiteness of the conditional model (target history k, sources at lag 1)."""
    N = segs[0].shape[1]
    pvals = []
    for jn in range(N):
        Ys, Xd = [], []
        for X in segs:
            X = X - X.mean(0); T = X.shape[0]
            H = np.column_stack([X[k - h:T - h, jn] for h in range(1, k + 1)] +
                                [X[k - 1:T - 1, i] for i in range(N) if i != jn])
            Ys.append(X[k:, jn]); Xd.append(H)
        b = np.linalg.lstsq(np.vstack(Xd), np.concatenate(Ys), rcond=None)[0]
        # pooled within-segment autocorrelations of residuals
        num = np.zeros(lags); den = 0.0; n = 0
        for H, y in zip(Xd, Ys):
            r = y - H @ b
            den += r @ r; n += len(r)
            for L in range(1, lags + 1):
                num[L - 1] += r[L:] @ r[:-L]
        rho = num / den
        Q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
        pvals.append(float(stats.chi2.sf(Q, lags)))
    return pvals


def eeg_analysis(band, reference, kmax, n_perm, rng, l=1):
    fs, labels, pre, post = load_eeg(band, reference)
    ks = pooled_bic(pre + post, kmax)             # label-invariant selection (pre and post pooled)
    k = int(np.median(ks))
    f = lambda S: TE.cte_segments(S, k, l=l)

    def residualize(trials):
        m = np.mean(trials, 0)
        return [t - m for t in trials]

    def net_pair(a, b):
        return f(residualize(a)), f(residualize(b))

    def rev(a, b):
        both = a["R"] & b["R"]
        return int((both & (np.sign(a["B"]) != np.sign(b["B"]))).sum()), int(both.sum()), int((a["R"] ^ b["R"]).sum())
    rp, rq = net_pair(pre, post)
    r_obs, shared, chg = rev(rp, rq)
    rn, cn = [], []
    for _ in range(n_perm):
        flip = rng.random(len(pre)) < 0.5
        s1 = [q if fl else p for p, q, fl in zip(pre, post, flip)]
        s2 = [p if fl else q for p, q, fl in zip(pre, post, flip)]
        a_, b_ = net_pair(s1, s2)                 # evoked means re-estimated inside each permutation
        x, _, c = rev(a_, b_); rn.append(x); cn.append(c)
    rn, cn = np.array(rn), np.array(cn)
    rpre = residualize(pre)
    odd, even = f(rpre[0::2]), f(rpre[1::2])
    inter = int((odd["R"] & even["R"]).sum()); uni = int((odd["R"] | even["R"]).sum())
    lb_k = ljung_box_resid(rpre, k); lb_1 = ljung_box_resid(rpre, 1)
    return dict(band=list(band), reference=reference, kmax=kmax, k_bic=k, k_dist=ks, l=l,
                k_at_boundary=bool(k >= kmax), n_trials=len(pre), channels=len(labels),
                edges_pre=int(rp["R"].sum()), edges_post=int(rq["R"].sum()), shared=shared,
                reversals=r_obs, rev_p=float((1 + (rn >= r_obs).sum()) / (n_perm + 1)),
                rev_null_mean=float(rn.mean()), changed=chg, chg_null_mean=float(cn.mean()),
                chg_null_95=float(np.percentile(cn, 95)),
                chg_p=float((1 + (cn >= chg).sum()) / (n_perm + 1)),
                split_half=[int(odd["R"].sum()), int(even["R"].sum()), inter],
                split_half_jaccard=float(inter / uni) if uni else np.nan,
                ljungbox_k_frac_p05=float(np.mean(np.array(lb_k) < 0.05)),
                ljungbox_k_median_p=float(np.median(lb_k)),
                ljungbox_1_frac_p05=float(np.mean(np.array(lb_1) < 0.05)))


def main():
    t0 = time.time()
    res = dict(eeg_files_sha256={f: sha256(os.path.join(EEGDIR, f)) for f in ("eeglab_data.set", "eeglab_data.fdt")})
    res["netsim"] = run_netsim()
    print(f"[{time.time()-t0:.0f}s] netsim", flush=True)
    rng = np.random.default_rng(42)
    res["eeg"] = []
    for band, ref, kmax, nperm in (((1, 30), "average", 12, 200), ((1, 30), "average", 30, 200),
                                   ((0.5, 45), "average", 30, 100), ((1, 30), "recorded", 30, 100)):
        r = eeg_analysis(band, ref, kmax, nperm, rng)
        res["eeg"].append(r)
        print(f"[{time.time()-t0:.0f}s] eeg {band} {ref} kmax={kmax}: k={r['k_bic']} edges {r['edges_pre']}/{r['edges_post']} "
              f"rev p={r['rev_p']:.3f} chg p={r['chg_p']:.3f} LB={r['ljungbox_k_frac_p05']:.2f}", flush=True)
        json.dump(j(res), open("results/neural_revision.json", "w"), indent=1)


if __name__ == "__main__":
    main()
