"""Performance measures: DeLong AUROC (Sun & Xu, 2014), AUPRC, sensitivity/specificity, edge recovery."""
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_curve


def _midrank(x):
    o = np.argsort(x); xs = x[o]; n = len(x)
    T = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n); out[o] = T
    return out


def _delong_components(y, scores):
    """scores: (k, n). Returns aucs (k,) and covariance (k, k)."""
    y = np.asarray(y).astype(bool)
    pos, neg = scores[:, y], scores[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def auc_ci(y, s):
    a, S = _delong_components(y, np.atleast_2d(s))
    se = np.sqrt(S[0, 0])
    return dict(auc=float(a[0]), lo=float(a[0] - 1.96 * se), hi=float(a[0] + 1.96 * se))


def delong_paired(y, s_ref, s_other):
    """Difference AUC(ref) - AUC(other) with 95% CI and two-sided p."""
    a, S = _delong_components(y, np.vstack([s_ref, s_other]))
    d = a[0] - a[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = np.sqrt(max(var, 1e-300))
    z = d / se
    return dict(diff=float(d), lo=float(d - 1.96 * se), hi=float(d + 1.96 * se),
                p=float(2 * stats.norm.sf(abs(z))))


def operating_point(y, s):
    """Sensitivity and specificity at the Youden-optimal threshold."""
    fpr, tpr, thr = roc_curve(y, s)
    k = np.argmax(tpr - fpr)
    return float(tpr[k]), float(1 - fpr[k])


def summary(y, s):
    out = auc_ci(y, s)
    out["sens"], out["spec"] = operating_point(y, s)
    out["auprc"] = float(average_precision_score(y, s))
    return out


def edge_recovery(R, sign_est, T):
    """Precision, recall and sign accuracy of retained edges R against planted signed truth T."""
    true = T != 0
    tp = R & true
    prec = tp.sum() / R.sum() if R.sum() else np.nan
    rec = tp.sum() / true.sum()
    sacc = (np.sign(sign_est)[tp] == T[tp]).mean() if tp.sum() else np.nan
    return float(prec), float(rec), float(sacc)


def edge_scores(R, G, directed=True):
    """Directed or undirected precision/recall/F1 against a boolean truth G[source, target]."""
    N = G.shape[0]
    if directed:
        m = ~np.eye(N, dtype=bool); e, t = R[m], G[m]
    else:
        iu = np.triu_indices(N, 1)
        Ru, Gu = R | R.T, G | G.T
        e, t = Ru[iu], Gu[iu]
    tp = int((e & t).sum()); fp = int((e & ~t).sum()); fn = int((~e & t).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)
