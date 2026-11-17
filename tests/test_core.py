"""Seven unit tests for TemporalSignedMBB (run: python -m pytest -q tests)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsmbb import te as TE, topology as TP, evaluate as EV, generator as G


def _var_segments(seed=0, n=60, T=40):
    rng = np.random.default_rng(seed)
    N = 5; B = np.zeros((N, N)); np.fill_diagonal(B, 0.4); B[0, 1] = 0.5; B[2, 3] = -0.5
    segs = []
    for _ in range(n):
        x = np.zeros(N); S = []
        for _ in range(T):
            x = B.T @ x + rng.normal(0, 1, N); S.append(x.copy())
        segs.append(np.array(S))
    return segs, B


def test_estimator_recovers_known_var():
    segs, B = _var_segments()
    r = TE.cte_segments(segs, 1)
    truth = (B != 0) & ~np.eye(5, dtype=bool)
    assert (r["R"] == truth).all()
    assert np.sign(r["B"][0, 1]) == 1 and np.sign(r["B"][2, 3]) == -1


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    rej, q = TE.bh(p, 0.05)
    assert rej.tolist() == [True, True] + [False] * 8
    assert np.isclose(q[0], 0.01)


def test_wasserstein_identity_and_symmetry():
    a = np.array([[0.2, 1.0], [0.5, 1.0]]); b = np.array([[0.3, 1.0]])
    assert TP.wasserstein2(a, a) == 0
    assert np.isclose(TP.wasserstein2(a, b), TP.wasserstein2(b, a))


def test_h1_detects_known_cycle():
    A = np.zeros((4, 4))
    for i in range(4):
        A[i, (i + 1) % 4] = 1.0
    h1, total, _ = TP.persistence(A * 0.5 + 0.5 * (A > 0))
    assert len(h1) == 1


def test_frustration_triangles():
    bal = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], float)
    unb = np.array([[0, 1, 1], [0, 0, -1], [0, 0, 0]], float)
    assert TP.frustration(bal) == 0
    assert np.isclose(TP.frustration(unb), 1 / 3)


def test_delong_matches_mann_whitney():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(50), np.zeros(70)]; s = rng.normal(y * 0.8, 1)
    from sklearn.metrics import roc_auc_score
    assert np.isclose(EV.auc_ci(y, s)["auc"], roc_auc_score(y, s))


def test_simulation_is_reproducible():
    a = G.simulate(7, visits_per_window=5)
    b = G.simulate(7, visits_per_window=5)
    assert np.array_equal(a["counts"], b["counts"])


# ---------------------------------------------------------------- revision checks
def test_transition_truth_complete():
    """Every sign change between two coupling matrices is found, including unplanted ones."""
    from tsmbb import generator as G
    B1 = np.zeros((4, 4)); B2 = np.zeros((4, 4))
    B1[0, 1] = 0.3; B2[0, 1] = -0.3          # reversal
    B1[1, 2] = -0.35; B2[1, 2] = 0.11        # small reversal (as R. gnavus -> F. prausnitzii)
    B2[2, 3] = 0.2                           # appearance
    B1[3, 0] = 0.2                           # disappearance
    tt = G.transition_truth(B1, B2, np.arange(4))
    assert tt["reversal"].sum() == 2 and tt["appear"].sum() == 1 and tt["disappear"].sum() == 1


def test_stationary_initialisation():
    """The stationary covariance solves Sigma = B^T Sigma B + s^2 I."""
    from tsmbb import generator as G
    rng = np.random.default_rng(0)
    B = np.eye(5) * 0.45 + 0.2 * rng.normal(size=(5, 5)) * (rng.random((5, 5)) < 0.3)
    S = G.stationary_cov(B, 0.35)
    assert np.allclose(S, B.T @ S @ B + 0.35 ** 2 * np.eye(5))


def test_reciprocal_conflicting_signs():
    """The original frustration projection cancels opposite reciprocal arcs; the multigraph
    projection keeps both and counts one violation."""
    from tsmbb import topology as TP
    A = np.zeros((3, 3)); A[0, 1] = 0.1; A[1, 0] = -0.1
    F_sum, m_sum, canc = TP.frustration_exact(A, "sum")
    F_multi, m_multi, _ = TP.frustration_exact(A, "multi")
    assert canc == 1 and m_sum == 0 and m_multi == 2 and F_multi == 0.5


def test_cluster_test_calibration():
    """Under a panel null with subject means removed, the cluster-robust test is close to nominal
    while the naive pooled F test is anti-conservative."""
    from tsmbb import te as TE
    rng = np.random.default_rng(1)
    N, rates = 6, {"naive": [], "cluster": []}
    off = ~np.eye(N, dtype=bool)
    for _ in range(60):
        Xs, Ys, own = [], [], []
        for s in range(150):
            x = rng.normal(0, 1, N) / np.sqrt(1 - 0.45 ** 2); seq = []
            for t in range(5):
                if t:
                    x = 0.45 * x + rng.normal(0, 1, N)
                seq.append(x.copy())
            seq = np.array(seq); seq -= seq.mean(0)
            Xs.append(seq[:-1]); Ys.append(seq[1:]); own += [s] * 4
        o = TE.cte_tests(np.vstack(Xs), np.vstack(Ys), np.array(own))
        for k in rates:
            rates[k].append((o[k]["P"][off] < 0.05).mean())
    assert abs(np.mean(rates["cluster"]) - 0.05) < 0.015
    assert np.mean(rates["naive"]) > 0.065


def test_group_membership():
    """Responders are not members of the non-responder group (fix of the original substring test)."""
    from tsmbb import generator as G, benchmark as BM
    sim = G.simulate(42)
    ids = BM.group_ids(sim)
    assert len(set(ids["R"]) & set(ids["NR"])) == 0
    assert len(ids["R"]) + len(ids["NR"]) == len(ids["MDD"])


def test_default_generator_unchanged():
    """With no scenario options the revised generator reproduces the original seed-42 data."""
    from tsmbb import generator as G
    s = G.simulate(42)
    assert int(s["counts"].sum()) == 754_185_526 and s["counts"].shape == (21000, 300)
