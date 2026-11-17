"""Layer 1: Gaussian conditional transfer entropy, LIONESS and the KSG check (Section 2.3, 2.8)."""
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma


def bh(p, q=0.05):
    """Benjamini-Hochberg: boolean rejection mask and q values."""
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    ranked = p[o] * n / np.arange(1, n + 1)
    qv = np.minimum.accumulate(ranked[::-1])[::-1]
    qvals = np.empty(n)
    qvals[o] = np.minimum(qv, 1)
    return qvals < q, qvals


class Gram:
    """Sufficient statistics of the lag-1 regression Y ~ X (no intercept, data demeaned)."""

    def __init__(self, X=None, Y=None, N=None):
        if X is None:
            self.XX = np.zeros((N, N)); self.XY = np.zeros((N, N)); self.YY = np.zeros(N); self.n = 0
        else:
            self.XX = X.T @ X; self.XY = X.T @ Y; self.YY = (Y ** 2).sum(0); self.n = len(X)

    def __add__(self, o):
        g = Gram(N=self.XX.shape[0])
        g.XX, g.XY, g.YY, g.n = self.XX + o.XX, self.XY + o.XY, self.YY + o.YY, self.n + o.n
        return g

    def __sub__(self, o):
        g = Gram(N=self.XX.shape[0])
        g.XX, g.XY, g.YY, g.n = self.XX - o.XX, self.XY - o.XY, self.YY - o.YY, self.n - o.n
        return g

    def coef(self):
        return np.linalg.solve(self.XX, self.XY)          # B[i, j]: effect of i on j


def cte_from_gram(g, q=0.05, df_deduct=0):
    """Conditional TE for every ordered pair from Gram statistics.
    Returns dict with TE (bits), coefficient B, p, q values, retained mask R and signed A."""
    N = g.XX.shape[0]
    C = np.linalg.inv(g.XX)
    B = C @ g.XY
    rss_f = g.YY - np.einsum("ij,ij->j", g.XY, B)
    df = g.n - N - df_deduct
    drss = B ** 2 / np.diag(C)[:, None]
    TE = 0.5 * np.log2((rss_f[None, :] + drss) / rss_f[None, :])
    F = drss / (rss_f[None, :] / df)
    P = stats.f.sf(F, 1, df)
    off = ~np.eye(N, dtype=bool)
    R = np.zeros((N, N), bool)
    rej, qv = bh(P[off], q)
    R[off] = rej
    Q = np.ones((N, N)); Q[off] = qv
    A = np.where(R, np.sign(B) * TE, 0.0)
    return dict(TE=TE, B=B, P=P, Q=Q, R=R, A=A, n=g.n, df=df)


def cte_segments(segments, k=1, q=0.05, deduct_means=True, l=1):
    """Conditional TE on raw segments (list of T x N arrays) with target history k and
    source history l (l = 1 in the original analyses; the revision also reports l > 1, in which
    case the edge sign is the sign of the summed source-lag coefficients and the F test has l
    numerator df). Each segment is demeaned."""
    if l > 1:
        return _cte_segments_l(segments, k, q, deduct_means, l)
    Ys, Hs = [], []
    for X in segments:
        X = X - X.mean(0)
        T = X.shape[0]
        if T <= k + 1:
            continue
        Ys.append(X[k:])
        Hs.append(np.stack([X[k - h:T - h] for h in range(1, k + 1)], 2))  # (T-k, N, k)
    Y = np.vstack(Ys); H = np.concatenate(Hs, 0)
    n, N = Y.shape
    nseg = len(Ys) if deduct_means else 0
    TE = np.zeros((N, N)); P = np.ones((N, N)); B = np.zeros((N, N))
    lag1 = H[:, :, 0]
    for j in range(N):
        others = [i for i in range(N) if i != j]
        Xd = np.column_stack([H[:, j, :], lag1[:, others]])
        XtX = Xd.T @ Xd
        C = np.linalg.inv(XtX)
        b = C @ (Xd.T @ Y[:, j])
        r = Y[:, j] - Xd @ b
        rss = float(r @ r)
        df = n - Xd.shape[1] - nseg
        for c, i in enumerate(others):
            bi = b[k + c]; d = bi ** 2 / C[k + c, k + c]
            P[i, j] = stats.f.sf(d / (rss / df), 1, df)
            TE[i, j] = 0.5 * np.log2((rss + d) / rss)
            B[i, j] = bi
    off = ~np.eye(N, dtype=bool)
    R = np.zeros((N, N), bool)
    R[off] = bh(P[off], q)[0]
    return dict(TE=TE, B=B, P=P, R=R, A=np.where(R, np.sign(B) * TE, 0.0))


def bte_segments(segments, k=1, q=0.05):
    """Bivariate Gaussian TE (pairwise Granger causality) with target history k."""
    Ys, Hs = [], []
    for X in segments:
        X = X - X.mean(0); T = X.shape[0]
        Ys.append(X[k:]); Hs.append(np.stack([X[k - h:T - h] for h in range(1, k + 1)], 2))
    Y = np.vstack(Ys); H = np.concatenate(Hs, 0)
    n, N = Y.shape; nseg = len(Ys)
    P = np.ones((N, N)); B = np.zeros((N, N)); TE = np.zeros((N, N))
    for j in range(N):
        own = H[:, j, :]
        br = np.linalg.lstsq(own, Y[:, j], rcond=None)[0]
        rss_r = float(((Y[:, j] - own @ br) ** 2).sum())
        for i in range(N):
            if i == j:
                continue
            Xd = np.column_stack([own, H[:, i, 0]])
            b = np.linalg.lstsq(Xd, Y[:, j], rcond=None)[0]
            rss = float(((Y[:, j] - Xd @ b) ** 2).sum())
            df = n - Xd.shape[1] - nseg
            P[i, j] = stats.f.sf((rss_r - rss) / (rss / df), 1, df)
            TE[i, j] = 0.5 * np.log2(rss_r / rss); B[i, j] = b[-1]
    off = ~np.eye(N, dtype=bool)
    R = np.zeros((N, N), bool); R[off] = bh(P[off], q)[0]
    return dict(TE=TE, B=B, P=P, R=R, A=np.where(R, np.sign(B) * TE, 0.0))


def lioness_coef(g_ref, b_R, n_ref, g_subj, in_ref):
    """LIONESS coefficient matrix for one subject (Kuijjer et al., 2019).
    g_ref: Gram of the reference set (n_ref subjects), b_R its coefficient matrix."""
    if in_ref:
        b_wo = (g_ref - g_subj).coef()
        return n_ref * (b_R - b_wo) + b_wo
    b_w = (g_ref + g_subj).coef()
    return (n_ref + 1) * (b_w - b_R) + b_R


def ksg_cmi(x, y, z, k=4):
    """Kraskov-Stoegbauer-Grassberger estimate of I(x; y | z) in bits (Frenzel-Pompe form)."""
    x, y, z = [np.asarray(a, float).reshape(len(a), -1) for a in (x, y, z)]
    xyz = np.hstack([x, y, z])
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k + 1, p=np.inf)
    eps = d[:, -1] - 1e-12

    def count(m):
        t = cKDTree(m)
        return np.array([len(t.query_ball_point(m[i], eps[i], p=np.inf)) - 1 for i in range(len(m))])
    nxz = count(np.hstack([x, z])); nyz = count(np.hstack([y, z])); nz = count(z)
    val = digamma(k) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))
    return float(val / np.log(2))


# ---------------------------------------------------------------- revision: calibrated inference
def cte_tests(X, Y, owner, q=0.05):
    """Conditional TE / lag-1 Granger tests on pooled, subject-demeaned transitions with three
    inferential procedures (revision; Reviewers 1, 3, 5, editor):
      'naive'   : F test with the residual df of the pooled regression (original submission);
      'dfcorr'  : F test whose residual df also subtracts the number of subject means removed;
      'cluster' : cluster-robust (CR1, clustered by subject) Wald t test with G - 1 df,
                  which allows arbitrary within-subject dependence of the scores.
    Returns dict method -> dict(P, Q, R, A) plus shared TE, B, n, G."""
    N = X.shape[1]
    g = Gram(X, Y)
    C = np.linalg.inv(g.XX)
    B = C @ g.XY
    U = Y - X @ B
    rss = (U ** 2).sum(0)
    drss = B ** 2 / np.diag(C)[:, None]
    TE = 0.5 * np.log2((rss[None, :] + drss) / rss[None, :])
    subj = np.unique(owner)
    G = len(subj)
    n = len(X)
    out = dict(TE=TE, B=B, n=n, G=G)
    off = ~np.eye(N, dtype=bool)
    # score contributions per subject: S_g = X_g' U_g ; W_g = C S_g ; var(B_ij) = sum_g W_g[i, j]^2
    var = np.zeros((N, N))
    order = np.argsort(owner, kind="stable")
    o_sorted = owner[order]
    cuts = np.flatnonzero(np.r_[True, o_sorted[1:] != o_sorted[:-1], True])
    Xs, Us = X[order], U[order]
    for a, b in zip(cuts[:-1], cuts[1:]):
        W = C @ (Xs[a:b].T @ Us[a:b])
        var += W ** 2
    var *= G / (G - 1) * (n - 1) / (n - N)
    tstat = B / np.sqrt(var)
    P_cl = 2 * stats.t.sf(np.abs(tstat), G - 1)
    for name, df_ in (("naive", n - N), ("dfcorr", n - N - G)):
        F = drss / (rss[None, :] / df_)
        out[name] = dict(P=stats.f.sf(F, 1, df_))
    out["cluster"] = dict(P=P_cl)
    for name in ("naive", "dfcorr", "cluster"):
        P = out[name]["P"]
        R = np.zeros((N, N), bool); Q = np.ones((N, N))
        rej, qv = bh(P[off], q)
        R[off] = rej; Q[off] = qv
        out[name].update(Q=Q, R=R, A=np.where(R, np.sign(B) * TE, 0.0))
    return out


def lasso_var(X, Y):
    """Comparator: l1-regularised VAR(1) (LassoLarsIC with BIC, one target at a time)."""
    from sklearn.linear_model import LassoLarsIC
    N = X.shape[1]
    B = np.zeros((N, N))
    Xs = X / X.std(0)
    for j in range(N):
        m = LassoLarsIC(criterion="bic", fit_intercept=False).fit(Xs, Y[:, j])
        B[:, j] = m.coef_ / X.std(0)
    R = (B != 0) & ~np.eye(N, dtype=bool)
    return dict(B=B, R=R, A=np.where(R, B, 0.0))


def bivariate_pooled(X, Y, G, q=0.05):
    """Comparator: bivariate lag-1 Granger causality (target history 1) on pooled transitions."""
    N = X.shape[1]
    n = len(X)
    P = np.ones((N, N)); Bc = np.zeros((N, N))
    for j in range(N):
        x0 = X[:, j]; y = Y[:, j]
        r0 = y - x0 * (x0 @ y) / (x0 @ x0); rss0 = r0 @ r0
        for i in range(N):
            if i == j:
                continue
            A2 = np.column_stack([x0, X[:, i]])
            b = np.linalg.lstsq(A2, y, rcond=None)[0]
            r = y - A2 @ b; rss = r @ r
            df = n - 2 - G
            P[i, j] = stats.f.sf((rss0 - rss) / (rss / df), 1, df); Bc[i, j] = b[1]
    off = ~np.eye(N, dtype=bool)
    R = np.zeros((N, N), bool); R[off] = bh(P[off], q)[0]
    return dict(B=Bc, R=R, A=np.where(R, Bc, 0.0))


def _cte_segments_l(segments, k, q, deduct_means, l):
    h = max(k, l)
    Ys, Hs = [], []
    for X in segments:
        X = X - X.mean(0); T = X.shape[0]
        if T <= h + 1:
            continue
        Ys.append(X[h:]); Hs.append(np.stack([X[h - s:T - s] for s in range(1, h + 1)], 2))
    Y = np.vstack(Ys); H = np.concatenate(Hs, 0)
    n, N = Y.shape
    nseg = len(Ys) if deduct_means else 0
    TE = np.zeros((N, N)); P = np.ones((N, N)); B = np.zeros((N, N))
    for j in range(N):
        others = [i for i in range(N) if i != j]
        cols = [H[:, j, :k]] + [H[:, i, :l] for i in others]
        Xd = np.column_stack(cols)
        b, *_ = np.linalg.lstsq(Xd, Y[:, j], rcond=None)
        rss = float(((Y[:, j] - Xd @ b) ** 2).sum())
        df = n - Xd.shape[1] - nseg
        for c, i in enumerate(others):
            sl = slice(k + c * l, k + (c + 1) * l)
            Xr = np.delete(Xd, np.r_[sl], 1)
            br, *_ = np.linalg.lstsq(Xr, Y[:, j], rcond=None)
            rr = float(((Y[:, j] - Xr @ br) ** 2).sum())
            P[i, j] = stats.f.sf(((rr - rss) / l) / (rss / df), l, df)
            TE[i, j] = 0.5 * np.log2(rr / rss); B[i, j] = b[sl].sum()
    off = ~np.eye(N, dtype=bool)
    R = np.zeros((N, N), bool)
    R[off] = bh(P[off], q)[0]
    return dict(TE=TE, B=B, P=P, R=R, A=np.where(R, np.sign(B) * TE, 0.0))
