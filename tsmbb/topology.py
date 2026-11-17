"""Layer 3 (persistent homology, Section 2.5) and network descriptors (Sections 2.4, 2.6, 2.7)."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from ripser import ripser


# ---------------------------------------------------------------- persistence
def distance(A):
    """Eq. (6): d_ij = 1 - max(|A_ij|, |A_ji|) / max|A|; pairs without an edge at 1."""
    M = np.maximum(np.abs(A), np.abs(A).T)
    mx = M.max()
    D = np.ones_like(M) if mx == 0 else 1 - M / mx
    np.fill_diagonal(D, 0)
    return D


def persistence(A, cocycles=False):
    """H1 diagram (finite points), total H1 persistence and node contributions phi_i (Eq. 8)."""
    D = distance(A)
    r = ripser(D, distance_matrix=True, maxdim=1, do_cocycles=cocycles)
    h1 = r["dgms"][1]
    fin = np.isfinite(h1).all(1)
    h1 = h1[fin]
    total = float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) else 0.0
    phi = np.zeros(A.shape[0])
    if cocycles and len(h1):
        cocs = [c for c, f in zip(r["cocycles"][1], fin) if f]
        for (b, d_), c in zip(h1, cocs):
            nodes = np.unique(c[:, :2])
            phi[nodes] += d_ - b
    return h1, total, phi


def wasserstein2(D1, D2):
    """Exact 2-Wasserstein distance with L-infinity ground metric (Eq. 7)."""
    n1, n2 = len(D1), len(D2)
    if n1 + n2 == 0:
        return 0.0
    C = np.zeros((n1 + n2, n1 + n2))
    if n1 and n2:
        C[:n1, :n2] = np.max(np.abs(D1[:, None, :] - D2[None, :, :]), 2) ** 2
    d1 = ((D1[:, 1] - D1[:, 0]) / 2) ** 2 if n1 else np.zeros(0)
    d2 = ((D2[:, 1] - D2[:, 0]) / 2) ** 2 if n2 else np.zeros(0)
    C[:n1, n2:] = np.inf; C[n1:, :n2] = np.inf
    for i in range(n1):
        C[i, n2 + i] = d1[i]
    for j in range(n2):
        C[n1 + j, j] = d2[j]
    C[n1:, n2:] = 0
    C[np.isinf(C)] = 1e9
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r, c].sum()))


def density_null(A, n=500, rng=None):
    """Total H1 persistence of A against networks with the same edge weights on random pairs."""
    rng = rng or np.random.default_rng(0)
    N = A.shape[0]
    M = np.maximum(np.abs(A), np.abs(A).T)
    iu = np.triu_indices(N, 1)
    w = M[iu]; w = w[w > 0]
    obs = persistence(A)[1]
    null = []
    for _ in range(n):
        R = np.zeros((N, N))
        sel = rng.choice(len(iu[0]), len(w), replace=False)
        R[iu[0][sel], iu[1][sel]] = rng.permutation(w)
        null.append(persistence(R + R.T)[1])
    null = np.array(null)
    return dict(obs=obs, null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                z=float((obs - null.mean()) / null.std(ddof=1)))


# ---------------------------------------------------------------- SPTS (Eqs. 9-10)
def perturbation_scores(A):
    D = np.diag(np.abs(A).sum(1))
    Lp = np.linalg.pinv(D - A)
    return np.linalg.norm(Lp, axis=0)          # || L+ e_i ||_2 for every i


def reversal_rates(A_seq):
    """Eq. (5): node reversal rate over a sequence of signed matrices (consecutive windows)."""
    N = A_seq[0].shape[0]
    flips = np.zeros((N, N)); pairs = np.zeros((N, N))
    for A1, A2 in zip(A_seq[:-1], A_seq[1:]):
        both = (A1 != 0) & (A2 != 0)
        pairs += both
        flips += both & (np.sign(A1) != np.sign(A2))
    multi = pairs > 0
    edge_rate = np.where(multi, flips / np.maximum(pairs, 1), 0)
    r = np.zeros(N)
    for i in range(N):
        inc = multi[i, :] | multi[:, i]
        e = np.concatenate([edge_rate[i, multi[i, :]], edge_rate[multi[:, i], i]])
        r[i] = e.mean() if len(e) else 0.0
    return r


def spts(A_seq, phi_seq, lam=0.1, gam=0.3):
    ps = sum(perturbation_scores(A) for A in A_seq)
    r = reversal_rates(A_seq) if len(A_seq) > 1 else np.zeros(A_seq[0].shape[0])
    return ps + lam * r + gam * sum(phi_seq)


# ---------------------------------------------------------------- balance and resilience
def frustration(A, restarts=20, rng=None):
    """Normalized frustration index (Eq. 11) by greedy local search."""
    rng = rng or np.random.default_rng(0)
    S = np.sign(A) + np.sign(A.T)
    S = np.sign(S)
    np.fill_diagonal(S, 0)
    m = (np.abs(S) > 0).sum() / 2
    if m == 0:
        return 0.0
    best = np.inf
    for _ in range(restarts):
        x = rng.choice([-1, 1], S.shape[0])
        improved = True
        while improved:
            improved = False
            for i in rng.permutation(S.shape[0]):
                gain = x[i] * (S[i] @ x)       # >0 means i is mostly satisfied
                if gain < 0:
                    x[i] = -x[i]; improved = True
        viol = ((S * np.outer(x, x)) < 0).sum() / 2
        best = min(best, viol)
    return float(best / m)


def lcc_fraction(adj, keep, N):
    if keep.sum() == 0:
        return 0.0
    sub = adj[np.ix_(keep, keep)]
    _, lab = connected_components(sub, directed=False)
    return np.bincount(lab).max() / N


def resilience(A, order, n_random=20, rng=None):
    """AUC^R for removal in `order` and relative excess fragmentation vs random removal (Eq. 12)."""
    rng = rng or np.random.default_rng(0)
    N = A.shape[0]
    adj = ((A != 0) | (A.T != 0)).astype(int)
    f = np.arange(N + 1) / N

    def curve(o):
        keep = np.ones(N, bool); S = [lcc_fraction(adj, keep, N)]
        for k in range(N):
            keep[o[k]] = False; S.append(lcc_fraction(adj, keep, N))
        return np.array(S)
    St = curve(order)
    Sr = np.mean([curve(rng.permutation(N)) for _ in range(n_random)], 0)
    auc_t, auc_r = np.trapezoid(St, f), np.trapezoid(Sr, f)
    return dict(auc=float(auc_t), auc_random=float(auc_r),
                dfrag=float((auc_r - auc_t) / auc_r) if auc_r > 0 else 0.0, f=f, St=St, Sr=Sr)


# ---------------------------------------------------------------- signed motifs
PERMS = list(itertools.permutations(range(3)))


def triad_code(S, a, b, c):
    nodes = (a, b, c)
    best = None
    for p in PERMS:
        n = [nodes[i] for i in p]
        code = tuple(int(S[n[i], n[j]]) for i in range(3) for j in range(3) if i != j)
        if best is None or code < best:
            best = code
    return best


def motif_census(S):
    """Count signed three-node classes over weakly connected triples. S: signed {-1,0,1} matrix."""
    N = S.shape[0]
    und = (S != 0) | (S.T != 0)
    nb = [set(np.where(und[i])[0]) for i in range(N)]
    seen, counts, inst = set(), {}, {}
    for a in range(N):
        for b in nb[a]:
            for c in (nb[a] | nb[b]) - {a, b}:
                t = tuple(sorted((a, b, c)))
                if t in seen:
                    continue
                seen.add(t)
                code = triad_code(S, *t)
                counts[code] = counts.get(code, 0) + 1
                inst.setdefault(code, []).append(t)
    return counts, inst


def rewire(S, rng, n_swaps=None):
    """Directed degree-preserving edge swaps; signs reshuffled keeping the positive/negative ratio."""
    src, dst = np.nonzero(S)
    signs = S[src, dst].copy()
    E = list(zip(src.tolist(), dst.tolist()))
    Es = set(E)
    m = len(E)
    for _ in range(n_swaps or 10 * m):
        i, j = rng.integers(0, m, 2)
        (a, b), (c, d) = E[i], E[j]
        if len({a, b, c, d}) < 4 or (a, d) in Es or (c, b) in Es:
            continue
        Es -= {(a, b), (c, d)}; Es |= {(a, d), (c, b)}
        E[i], E[j] = (a, d), (c, b)
    R = np.zeros_like(S)
    sg = rng.permutation(signs)
    for (a, b), s in zip(E, sg):
        R[a, b] = s
    return R


def motif_enrichment(S, n_null=300, rng=None, min_arcs=3):
    rng = rng or np.random.default_rng(0)
    obs, inst = motif_census(S)
    keys = [k for k in obs if sum(v != 0 for v in k) >= min_arcs]
    null = {k: [] for k in keys}
    for _ in range(n_null):
        c, _ = motif_census(rewire(S, rng))
        for k in keys:
            null[k].append(c.get(k, 0))
    p = {k: (1 + np.sum(np.array(null[k]) >= obs[k])) / (1 + n_null) for k in keys}
    return dict(n_classes=len(obs), counts=obs, instances=inst, p=p)


# ================================================================ revision additions
def frustration_exact(A, projection="sum"):
    """Exact normalised frustration index (Eq. 11) by mixed-integer linear programming
    (HiGHS via scipy.optimize.milp). projection='sum' reproduces the original projection
    sign(sign(A) + sign(A^T)), in which reciprocal arcs of opposite sign cancel; projection='multi'
    keeps every directed arc as its own undirected signed edge, so a conflicting reciprocal pair
    always contributes one violation. Returns (F, n_edges, n_cancelled_pairs)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    N = A.shape[0]
    Sg = np.sign(A); np.fill_diagonal(Sg, 0)
    cancelled = int((((Sg != 0) & (Sg.T != 0) & (Sg != Sg.T)).sum()) // 2)
    if projection == "sum":
        S = np.sign(Sg + Sg.T)
        iu = np.triu_indices(N, 1)
        edges = [(i, j, S[i, j]) for i, j in zip(*iu) if S[i, j] != 0]
    else:
        edges = [(i, j, Sg[i, j]) for i, j in np.argwhere(Sg != 0)]
    m = len(edges)
    if m == 0:
        return 0.0, 0, cancelled
    nv = N + m
    c = np.r_[np.zeros(N), np.ones(m)]
    rows, lo, hi = [], [], []
    for e, (i, j, s) in enumerate(edges):
        if s > 0:   # y >= xi - xj ; y >= xj - xi
            for a, b in ((i, j), (j, i)):
                r = np.zeros(nv); r[a] = 1; r[b] = -1; r[N + e] = -1; rows.append(r); lo.append(-np.inf); hi.append(0)
        else:       # y >= 1 - xi - xj ; y >= xi + xj - 1
            r = np.zeros(nv); r[i] = -1; r[j] = -1; r[N + e] = -1; rows.append(r); lo.append(-np.inf); hi.append(-1)
            r = np.zeros(nv); r[i] = 1; r[j] = 1; r[N + e] = -1; rows.append(r); lo.append(-np.inf); hi.append(1)
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options=dict(time_limit=60))
    return float(round(res.fun) / m), m, cancelled


def directed_cycles(A, max_len=5):
    """Number of simple directed cycles of length 3..max_len in the retained-edge digraph."""
    import networkx as nx
    Gd = nx.DiGraph()
    Gd.add_nodes_from(range(A.shape[0]))
    Gd.add_edges_from([tuple(e) for e in np.argwhere((A != 0) & ~np.eye(A.shape[0], dtype=bool))])
    cnt = {L: 0 for L in range(3, max_len + 1)}
    for cyc in nx.simple_cycles(Gd, length_bound=max_len):
        if len(cyc) >= 3:
            cnt[len(cyc)] += 1
    return cnt


def distance_alt(A, kind="linear"):
    """Alternative weight-to-distance transforms (Reviewer 1 minor 10)."""
    M = np.maximum(np.abs(A), np.abs(A).T)
    mx = M.max() if M.max() > 0 else 1.0
    w = M / mx
    if kind == "linear":
        D = 1 - w
    elif kind == "log":                 # -log w, absent pairs at +inf -> large finite value
        D = np.where(w > 0, -np.log(np.clip(w, 1e-12, 1)), 50.0)
    elif kind == "rank":                # rank-based: strongest edge 0, weakest retained ~1, absent 2
        D = np.full_like(w, 2.0)
        nz = w > 0
        r = w[nz].argsort().argsort()
        D[nz] = 1 - r / max(len(r) - 1, 1)
    np.fill_diagonal(D, 0)
    return D


def persistence_D(D):
    r = ripser(D, distance_matrix=True, maxdim=1)
    h1 = r["dgms"][1]
    h1 = h1[np.isfinite(h1).all(1)]
    return h1, float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) else 0.0


def density_null_z(A, n=200, rng=None):
    """Density- and weight-matched null for total H1 persistence (as in the EEG analysis)."""
    rng = rng or np.random.default_rng(0)
    d = density_null(A, n, rng)
    return d


def rewire_signed(S, rng, n_swaps=None):
    """Null for signed motifs that preserves the in- and out-degree of every node separately in
    the positive and in the negative subgraph (Reviewer 1 minor 8)."""
    R = np.zeros_like(S)
    for s in (1, -1):
        Ss = (S == s).astype(int)
        if Ss.sum() < 2:
            R[Ss > 0] = s; continue
        Rs = rewire(Ss, rng, n_swaps)
        R[Rs != 0] = s
    return R


def motif_enrichment_signed(S, n_null=300, rng=None, min_arcs=3):
    rng = rng or np.random.default_rng(0)
    obs, inst = motif_census(S)
    keys = [k for k in obs if sum(v != 0 for v in k) >= min_arcs]
    null = {k: [] for k in keys}
    for _ in range(n_null):
        c, _ = motif_census(rewire_signed(S, rng))
        for k in keys:
            null[k].append(c.get(k, 0))
    p = {k: (1 + np.sum(np.array(null[k]) >= obs[k])) / (1 + n_null) for k in keys}
    return dict(n_classes=len(obs), counts=obs, instances=inst, p=p, null=null)


def perturbation_scores_var(A, rcond=1e-15, laplacian="out"):
    """Variants of Eq. (9) for the sensitivity analysis (Reviewer 1)."""
    if laplacian == "out":
        L = np.diag(np.abs(A).sum(1)) - A
    elif laplacian == "in":
        L = np.diag(np.abs(A).sum(0)) - A
    else:
        As = (A + A.T) / 2
        L = np.diag(np.abs(As).sum(1)) - As
    Lp = np.linalg.pinv(L, rcond=rcond)
    s = np.linalg.svd(L, compute_uv=False)
    rank = int((s > rcond * s.max()).sum()) if s.max() > 0 else 0
    cond = float(s.max() / s[rank - 1]) if rank else np.inf
    return np.linalg.norm(Lp, axis=0), rank, cond


def dynamic_response(B, keep):
    """Response of the generative VAR to a perturbation of node i (Reviewer 1, Reviewer 3 c. 8):
    long-run response to a sustained unit input, ||(I - B^T)^-1 e_i||_2, and impulse-response energy
    (sum_h ||(B^T)^h e_i||^2)^1/2, both over the latent nodes; returned for the observed nodes `keep`."""
    from scipy.linalg import solve_discrete_lyapunov
    N = B.shape[0]
    M = np.linalg.inv(np.eye(N) - B.T)
    lr = np.linalg.norm(M, axis=0)
    # impulse energy: column norms of sum_h (B^T)^h e_i e_i^T B^h -> diag of P with P = B P B^T + I
    P = solve_discrete_lyapunov(B, np.eye(N))      # P = sum_h B^h B^hT ; ||(B^T)^h e_i||^2 = (B^h B^hT)_ii
    ie = np.sqrt(np.diag(P))
    return lr[keep], ie[keep]
