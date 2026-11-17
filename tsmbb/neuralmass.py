"""Revision: stochastic Wilson-Cowan network benchmark with planted excitatory and inhibitory
directed couplings and sign switches (Reviewers 1, 4; Reviewer 6 on volume conduction and
hemodynamics). Each region has an excitatory (E) and an inhibitory (I) population. An excitatory
inter-regional connection j -> i projects onto E_i; an inhibitory connection projects onto I_i, so
that its net (disynaptic) effect on E_i is negative. The recorded signal is E (plus sensor noise),
optionally mixed instantaneously across neighbouring sensors (volume conduction) or passed through
a Balloon-Windkessel hemodynamic model and sampled at a slow repetition time (BOLD)."""
import numpy as np

WC = dict(cEE=10.0, cIE=12.0, cEI=10.0, cII=2.0, aE=1.2, thE=2.8, aI=1.0, thI=4.0,
          tauE=0.010, tauI=0.020, P=0.5, sigma=0.05, w=0.5, delay=0.008)


def S(x, a, th):
    return 1 / (1 + np.exp(-a * (x - th))) - 1 / (1 + np.exp(a * th))


def random_graph(rng, n=10, n_edges=14, frac_inh=0.35):
    W = np.zeros((n, n))     # W[j, i] = sign of j -> i
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    sel = rng.choice(len(pairs), n_edges, replace=False)
    n_inh = int(round(frac_inh * n_edges))
    for k, s in enumerate(sel):
        a, b = pairs[s]
        W[a, b] = -1 if k < n_inh else 1
    return W


def switch(W, rng, n_switch=3, n_appear=1, n_disappear=1):
    """Condition B: sign switches on existing edges, one new edge, one removed edge."""
    W2 = W.copy()
    E = np.argwhere(W != 0)
    idx = rng.permutation(len(E))
    for k in idx[:n_switch]:
        W2[tuple(E[k])] *= -1
    for k in idx[n_switch:n_switch + n_disappear]:
        W2[tuple(E[k])] = 0
    free = [(a, b) for a in range(W.shape[0]) for b in range(W.shape[0]) if a != b and W[a, b] == 0]
    for k in rng.choice(len(free), n_appear, replace=False):
        W2[free[k]] = rng.choice([-1, 1])
    return W2


def simulate(W, T, rng, fs=250.0, dt=0.001, p=WC, burn=2.0):
    """Euler-Maruyama integration; returns E sampled at fs (block averages)."""
    n = W.shape[0]
    d = int(round(p["delay"] / dt))
    nsteps = int((T + burn) / dt)
    Wexc = np.where(W > 0, p["w"], 0.0); Winh = np.where(W < 0, p["w"], 0.0)
    E = np.zeros(n) + 0.1; I = np.zeros(n) + 0.05
    histE = np.zeros((d + 1, n)) + 0.1
    sub = int(round(1 / (fs * dt)))
    out = np.zeros((int(T * fs), n)); acc = np.zeros(n); k = 0; m = 0
    nb = int(burn / dt)
    sq = np.sqrt(dt)
    for t in range(nsteps):
        Ed = histE[t % (d + 1)]
        inE = p["cEE"] * E - p["cIE"] * I + p["P"] + Wexc.T @ Ed
        inI = p["cEI"] * E - p["cII"] * I + Winh.T @ Ed
        E = E + dt / p["tauE"] * (-E + S(inE, p["aE"], p["thE"])) + p["sigma"] * sq * rng.normal(size=n)
        I = I + dt / p["tauI"] * (-I + S(inI, p["aI"], p["thI"]))
        histE[t % (d + 1)] = E
        if t >= nb:
            acc += E; m += 1
            if m == sub:
                out[k] = acc / sub; k += 1; acc[:] = 0; m = 0
                if k == len(out):
                    break
    return out


def balloon(x, dt, rng=None):
    """Balloon-Windkessel model (Friston et al. 2003 parameters) driven by neural signal x (T x n),
    returns BOLD at the same sampling interval dt."""
    kappa, gamma, tau, alpha, rho, V0 = 0.65, 0.41, 0.98, 0.32, 0.34, 0.02
    k1, k2, k3 = 7 * rho, 2.0, 2 * rho - 0.2
    n = x.shape[1]
    s = np.zeros(n); f = np.ones(n); v = np.ones(n); q = np.ones(n)
    y = np.zeros_like(x)
    xs = 0.3 * (x - x.mean(0)) / x.std(0)        # neuronal input of moderate amplitude
    for t in range(len(x)):
        ds = xs[t] - kappa * s - gamma * (f - 1)
        df = s
        dv = (f - v ** (1 / alpha)) / tau
        dq = (f * (1 - (1 - rho) ** (1 / f)) / rho - v ** (1 / alpha) * q / v) / tau
        s, f, v, q = s + dt * ds, f + dt * df, v + dt * dv, q + dt * dq
        f = np.maximum(f, 0.1); v = np.maximum(v, 0.1); q = np.maximum(q, 0.05)
        y[t] = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    return y


def mixing_matrix(n, alpha=0.3):
    """Instantaneous volume-conduction mixing on a ring of sensors."""
    M = np.eye(n) * (1 - alpha)
    for i in range(n):
        M[i, (i - 1) % n] += alpha / 2; M[i, (i + 1) % n] += alpha / 2
    return M
