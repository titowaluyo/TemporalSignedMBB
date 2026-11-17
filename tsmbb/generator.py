"""Simulation model (Sections 2.1-2.2).

Latent VAR(1) dynamics on 63 nodes (47 genus-level taxa + 16 host-pathway nodes) with
signed coupling matrices that depend on group and window, observed through an amplicon
sequence variant (ASV) layer, cohort batch effects, variable sequencing depth and a
Dirichlet-multinomial count model.

Convention: B[i, j] is the effect of node i (source) on node j (target), so
x(t) - mu = B^T (x(t-1) - mu) + eps.
"""
import numpy as np

TAXA = [
    "F. prausnitzii", "R. gnavus", "A. putredinis", "E. lenta", "Lachno. NK4A136", "C. comes",
    "L. reuteri", "Bifidobacterium", "Roseburia", "Bacteroides", "Prevotella", "Akkermansia",
    "Blautia", "Escherichia/Shigella", "Parabacteroides", "Dialister", "Streptococcus",
    "Eubacterium", "Ruminococcus", "Alistipes", "Oscillibacter", "Coprococcus", "Dorea",
    "Subdoligranulum", "Butyricicoccus", "Veillonella", "Sutterella", "Collinsella",
    "Lactobacillus", "Enterococcus", "Odoribacter", "Barnesiella", "Phascolarctobacterium",
    "Megamonas", "Holdemania", "Anaerostipes", "Fusicatenibacter", "Agathobacter",
    "Christensenella", "Desulfovibrio", "Bilophila", "Haemophilus", "Klebsiella",
    "Methanobrevibacter", "Turicibacter", "Erysipelatoclostridium", "Flavonifractor",
]
HOST = ["Butyrate", "Propionate", "Acetate", "Tryptophan", "5-HT synthesis", "IDO1/Kyn",
        "Quinolinic acid", "GABA", "Glutamate", "Dopamine pathway", "Sec. bile acids", "BCAA",
        "LPS", "IL-6", "TNF-alpha", "Cortisol"]
NODES = TAXA + HOST
N_TAXA, N_HOST = len(TAXA), len(HOST)
N_LATENT = N_TAXA + N_HOST           # 63
REF_TAXON = "Flavonifractor"         # background taxon used as CLR reference (dropped)
IDX = {n: i for i, n in enumerate(NODES)}

CURATED = [  # (source, target, sign)  29 curated MGB edges
    ("F. prausnitzii", "Butyrate", +1), ("F. prausnitzii", "5-HT synthesis", +1),
    ("Roseburia", "Butyrate", +1), ("Lachno. NK4A136", "Butyrate", +1),
    ("C. comes", "Butyrate", +1), ("Bacteroides", "Propionate", +1),
    ("Bifidobacterium", "Acetate", +1), ("Bifidobacterium", "Tryptophan", +1),
    ("L. reuteri", "GABA", +1), ("Butyrate", "IL-6", -1), ("Butyrate", "IDO1/Kyn", -1),
    ("Tryptophan", "IDO1/Kyn", +1), ("Tryptophan", "5-HT synthesis", +1),
    ("IDO1/Kyn", "Quinolinic acid", +1), ("R. gnavus", "IDO1/Kyn", +1),
    ("R. gnavus", "IL-6", +1), ("A. putredinis", "IDO1/Kyn", +1),
    ("E. lenta", "Dopamine pathway", -1), ("Akkermansia", "LPS", -1),
    ("Escherichia/Shigella", "LPS", +1), ("LPS", "IL-6", +1), ("LPS", "TNF-alpha", +1),
    ("IL-6", "IDO1/Kyn", +1), ("TNF-alpha", "IDO1/Kyn", +1), ("GABA", "Cortisol", -1),
    ("Sec. bile acids", "TNF-alpha", -1), ("Streptococcus", "BCAA", -1),
    ("F. prausnitzii", "IDO1/Kyn", -1), ("Parabacteroides", "Glutamate", +1),
]
REVERSAL = [("Butyrate", "IL-6"), ("Butyrate", "IDO1/Kyn"), ("GABA", "Cortisol"),
            ("Lachno. NK4A136", "Butyrate"), ("C. comes", "Butyrate"),
            ("E. lenta", "Dopamine pathway"), ("Bacteroides", "Propionate"),
            ("Akkermansia", "LPS"), ("F. prausnitzii", "IDO1/Kyn"),
            ("Sec. bile acids", "TNF-alpha")]
GROUP_WINDOWS = [("HC", "T1"), ("HC", "T2"), ("HC", "T3"), ("MDD", "T1"),
                 ("R", "T2"), ("NR", "T2"), ("R", "T3"), ("NR", "T3")]

PARAMS = dict(self_persistence=0.45, rho_max=0.92, curated_mag=(0.30, 0.42),
              background_mag=(0.18, 0.30), n_background=70, noise_sd=0.35,
              host_noise_sd=0.10, n_asv=300, batch_sd=0.50, depth_median=38420,
              depth_sdlog=0.35, shallow_frac=0.08, shallow_range=(800, 4900),
              dm_precision=1500, missing_frac=0.05, relax_R=0.85,
              cohorts=[("SC1", 300, 0.62, 34.8, 8.9), ("SC2", 100, 0.50, 37.3, 9.2),
                       ("SC3", 300, 0.62, 37.3, 9.5)])


def _edge(B, s, t, val):
    B[IDX[s], IDX[t]] = val


def build_couplings(rng, p=PARAMS):
    """Return dict {(group, window): B} and the healthy reference."""
    N = N_LATENT
    Bh = np.zeros((N, N))
    lo, hi = p["curated_mag"]
    for s, t, sg in CURATED:
        _edge(Bh, s, t, sg * rng.uniform(lo, hi))
    # background edges: distinct pairs that do not coincide with curated ones
    lo, hi = p["background_mag"]
    n = 0
    while n < p["n_background"]:
        i, j = rng.integers(0, N, 2)
        if i == j or Bh[i, j] != 0 or NODES[i] == REF_TAXON or NODES[j] == REF_TAXON:
            continue
        Bh[i, j] = rng.choice([-1, 1]) * rng.uniform(lo, hi)
        n += 1

    def finish(B):
        B = B.copy()
        np.fill_diagonal(B, 0)
        off = B.copy()
        D = np.eye(N) * p["self_persistence"]
        M = D + off
        r = max(abs(np.linalg.eigvals(M)))
        if r >= p["rho_max"]:
            scale = (p["rho_max"] - 0.02 - p["self_persistence"]) / (r - p["self_persistence"])
            M = D + off * scale
        return M

    Bmdd = Bh.copy()
    Bmdd[IDX["F. prausnitzii"], IDX["5-HT synthesis"]] *= 0.15
    Bmdd[IDX["F. prausnitzii"], IDX["Butyrate"]] *= 0.5
    Bmdd[IDX["R. gnavus"], IDX["IDO1/Kyn"]] *= 2.2
    Bmdd[IDX["A. putredinis"], IDX["IDO1/Kyn"]] *= 2.0
    _edge(Bmdd, "F. prausnitzii", "R. gnavus", -0.35)
    _edge(Bmdd, "R. gnavus", "F. prausnitzii", -0.35)
    Bt2 = Bmdd.copy()
    for s, t in REVERSAL:
        Bt2[IDX[s], IDX[t]] *= -1
    BR3 = p["relax_R"] * Bh + (1 - p["relax_R"]) * Bmdd     # signs revert, 85% relaxed
    BNR3 = Bt2.copy()
    # incoherent feed-forward structure around A. putredinis and IDO1
    _edge(BNR3, "A. putredinis", "IL-6", 0.35)
    _edge(BNR3, "IL-6", "IDO1/Kyn", -0.35)
    # three chordless feedback loops of length 4-5
    loops = [["IL-6", "TNF-alpha", "Cortisol", "LPS"],
             ["IDO1/Kyn", "Quinolinic acid", "Glutamate", "GABA", "Tryptophan"],
             ["R. gnavus", "Sec. bile acids", "BCAA", "Dopamine pathway"]]
    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1]):
            if BNR3[IDX[a], IDX[b]] == 0:
                _edge(BNR3, a, b, 0.32)
    B = {("HC", w): finish(Bh) for w in ("T1", "T2", "T3")}
    B[("MDD", "T1")] = finish(Bmdd)
    B[("R", "T1")] = B[("NR", "T1")] = B[("MDD", "T1")]
    B[("R", "T2")] = B[("NR", "T2")] = finish(Bt2)
    B[("R", "T3")] = finish(BR3)
    B[("NR", "T3")] = finish(BNR3)
    return B


def window_of_week(week, visits_per_window):
    """Weeks are numbered so that T1 is the first block; one unsampled week between windows."""
    v = visits_per_window
    if week < v:
        return "T1", True
    if week == v:
        return "T2", False
    if week < 2 * v + 1:
        return "T2", True
    if week == 2 * v + 1:
        return "T3", False
    return "T3", True


# ---------------------------------------------------------------- scenarios (revision)
# Every option defaults to the configuration of the original submission, so that
# simulate(42) reproduces the original dataset bit for bit.
DEFAULT_SCENARIO = dict(
    init="diag",            # "diag": original independent initialisation; "stationary": draw from the
                            # stationary covariance of the full T1 regime and run `burn_in` steps first
    burn_in=0,
    dynamics="linear",      # "linear" | "tanh" (saturating coupling) | "lag2" (half of the off-diagonal
                            # couplings act at lag 2) | "glv" (gLV/Ricker-type abundance-scale coupling)
    tanh_c=0.15,
    noise="gauss",          # "gauss" | "t3" (Student t, 3 df, rescaled to the same SD)
    driver=0.0,             # loading of a hidden AR(1) common driver on 12 random nodes (0 = none)
    hetero=0.0,             # SD of log-normal subject-specific multipliers of every off-diagonal coupling
    hetero_drop=0.0,        # probability that a background edge is absent in a given subject
    coupling_scale=1.0,     # multiplies every off-diagonal coupling (design dimension)
    shift_scale=1.0,        # multiplies the MDD baseline mean shifts (0 = coupling-only differences)
    null=False,             # True: no off-diagonal coupling in any group or window (inferential null)
    batch="additive",       # "additive" | "nonadditive" (cohort-specific scaling of latent deviations
                            # and cohort-specific structural zeros)
    zero_frac=0.25, zero_p=(0.2, 0.6),
    early_prog=0.0,         # fraction of the T3 responder change already present in responders at T2
)


def _stable(M, rho_max=0.92, d=0.45):
    r = max(abs(np.linalg.eigvals(M)))
    if r >= rho_max:
        off = M - np.diag(np.diag(M))
        M = np.diag(np.diag(M)) + off * (rho_max - 0.02 - d) / (r - d)
    return M


def stationary_cov(B, sd):
    """Stationary covariance of x(t) = B^T x(t-1) + e, e ~ N(0, sd^2 I)."""
    from scipy.linalg import solve_discrete_lyapunov
    return solve_discrete_lyapunov(B.T, sd ** 2 * np.eye(B.shape[0]))


def simulate(seed=42, visits_per_window=5, p=PARAMS, scenario=None, keep_subject_B=False):
    """Simulate all cohorts. Returns dict with latent states, counts and metadata.
    `scenario` overrides entries of DEFAULT_SCENARIO (revision analyses)."""
    sc = dict(DEFAULT_SCENARIO, **(scenario or {}))
    rng = np.random.default_rng(seed)
    B = build_couplings(rng, p)
    D0 = np.eye(N_LATENT) * p["self_persistence"]
    if sc["null"] or sc["coupling_scale"] != 1.0 or sc["early_prog"]:
        if sc["early_prog"]:
            B[("R", "T2")] = B[("R", "T2")] + sc["early_prog"] * (B[("R", "T3")] - B[("R", "T2")])
        for k in list(B):
            off = B[k] - D0
            off = np.zeros_like(off) if sc["null"] else off * sc["coupling_scale"]
            B[k] = _stable(D0 + off)
        B[("R", "T1")] = B[("NR", "T1")] = B[("MDD", "T1")]
    n_weeks = 3 * visits_per_window + 2
    # ASV structure: each genus gets >=1 ASV; Dirichlet split of genus abundance
    asv_genus = np.concatenate([np.arange(N_TAXA), rng.integers(0, N_TAXA, p["n_asv"] - N_TAXA)])
    asv_genus.sort()
    asv_logprop = np.zeros(p["n_asv"])
    for g in range(N_TAXA):
        idx = np.where(asv_genus == g)[0]
        asv_logprop[idx] = np.log(rng.dirichlet(np.full(len(idx), 2.0)))
    taxa_base = rng.normal(0, 1.0, N_TAXA)         # population log-abundance profile
    mdd_shift = np.zeros(N_LATENT)
    for name, d in [("F. prausnitzii", -0.9), ("Roseburia", -0.5), ("C. comes", -0.5),
                    ("Lachno. NK4A136", -0.4), ("R. gnavus", 0.7), ("A. putredinis", 0.5),
                    ("E. lenta", 0.4), ("Butyrate", -0.6), ("IDO1/Kyn", 0.5), ("IL-6", 0.4)]:
        mdd_shift[IDX[name]] = d * sc["shift_scale"]
    # scenario-specific structures (drawn from a separate stream so the main stream is untouched)
    rs = np.random.default_rng([seed, 7919])
    lag2_mask = rs.random((N_LATENT, N_LATENT)) < 0.5
    drv_nodes = rs.choice(N_LATENT, 12, replace=False)
    drv_load = np.zeros(N_LATENT); drv_load[drv_nodes] = rs.choice([-1, 1], 12) * sc["driver"]
    sd = p["noise_sd"]
    cur = np.zeros((N_LATENT, N_LATENT), bool)
    for s_, t_, _ in CURATED:
        cur[IDX[s_], IDX[t_]] = True
    bg_mask = (np.abs(B[("HC", "T1")] - D0) > 0) & ~cur     # background edges of the reference

    def innov(r):
        if sc["noise"] == "t3":
            return r.standard_t(3, N_LATENT) * sd / np.sqrt(3.0)
        return r.normal(0, 1, N_LATENT) * sd

    def step(Bm, dev, dev2):
        off = Bm - np.diag(np.diag(Bm))
        lin = np.diag(Bm) * dev
        if sc["dynamics"] == "tanh":
            c = sc["tanh_c"]
            return lin + c * np.tanh(off.T @ dev / c)
        if sc["dynamics"] == "lag2":
            return lin + (off * ~lag2_mask).T @ dev + (off * lag2_mask).T @ dev2
        if sc["dynamics"] == "glv":
            g = np.expm1(np.clip(dev, -3, 1.5)) / 1.13      # abundance-scale deviation, unit slope ~
            return lin + off.T @ g
        return Bm.T @ dev

    rows, subj = [], []
    sid = 0
    batch = {}
    subjB = []
    for cohort, n_per_group, female, age_m, age_sd in p["cohorts"]:
        batch[cohort] = (rng.normal(0, p["batch_sd"], p["n_asv"]),
                         rng.normal(0, p["batch_sd"] * 0.5, N_HOST))
        for group in ("MDD", "HC"):
            for _ in range(n_per_group):
                mu = np.concatenate([taxa_base, np.zeros(N_HOST)]) + rng.normal(0, 0.35, N_LATENT)
                p_resp = np.nan
                if group == "MDD":
                    mu = mu + mdd_shift
                    eta = -0.28 + 0.45 * (mu[IDX["F. prausnitzii"]] - taxa_base[0] + 0.9) \
                        + 0.45 * (mu[IDX["Butyrate"]] + 0.6)
                    p_resp = 1 / (1 + np.exp(-eta))
                    responder = int(rng.random() < p_resp)
                    traj = "R" if responder else "NR"
                    hamd = rng.normal(23.2, 3.9)
                else:
                    responder, traj, hamd = -1, "HC", rng.normal(3.0, 2.0)
                subj.append(dict(subject=sid, cohort=cohort, group=group, traj=traj,
                                 responder=responder, female=int(rng.random() < female),
                                 age=rng.normal(age_m, age_sd), hamd0=hamd, p_resp=p_resp))
                sdv = np.where(np.arange(N_LATENT) < N_TAXA, p["noise_sd"], p["noise_sd"])
                # subject-specific coupling matrices (heterogeneity scenario)
                Bq = {}
                for w in ("T1", "T2", "T3"):
                    gk = ("HC", w) if traj == "HC" else (traj, w)
                    Bq[w] = B[gk]
                if sc["hetero"] > 0 or sc["hetero_drop"] > 0:
                    mult = np.exp(rs.normal(0, sc["hetero"], (N_LATENT, N_LATENT)))
                    drop = rs.random((N_LATENT, N_LATENT)) < sc["hetero_drop"]
                    for w in Bq:
                        off = (Bq[w] - D0) * mult
                        off = np.where(drop & bg_mask, 0.0, off)
                        Bq[w] = _stable(D0 + off)
                if keep_subject_B:
                    subjB.append(np.stack([Bq[w] for w in ("T1", "T2", "T3")]).astype(np.float32))
                if sc["init"] == "stationary":
                    S0 = stationary_cov(Bq["T1"], p["noise_sd"])
                    dev = rng.multivariate_normal(np.zeros(N_LATENT), S0)
                    x = mu + dev
                else:
                    x = mu + rng.normal(0, 1, N_LATENT) * sdv / np.sqrt(1 - p["self_persistence"] ** 2)
                x_prev = x.copy()
                f = 0.0
                for _b in range(sc["burn_in"]):
                    f = 0.6 * f + rs.normal(0, 0.35)
                    xn = mu + step(Bq["T1"], x - mu, x_prev - mu) + innov(rng) + drv_load * f
                    x_prev, x = x, xn
                for week in range(n_weeks):
                    w, sampled = window_of_week(week, visits_per_window)
                    if week > 0:
                        if sc["dynamics"] == "linear" and sc["noise"] == "gauss" and sc["driver"] == 0:
                            xn = mu + Bq[w].T @ (x - mu) + rng.normal(0, 1, N_LATENT) * sdv
                        else:
                            f = 0.6 * f + rs.normal(0, 0.35)
                            xn = mu + step(Bq[w], x - mu, x_prev - mu) + innov(rng) + drv_load * f
                        x_prev, x = x, xn
                    if not sampled:
                        continue
                    missed = rng.random() < p["missing_frac"]
                    rows.append(dict(subject=sid, week=week, window=w, missed=missed,
                                     latent=x.copy(), mu=mu))
                sid += 1
    # observation model
    n = len(rows)
    counts = np.zeros((n, p["n_asv"]), np.int64)
    host = np.zeros((n, N_HOST))
    depth = np.zeros(n, np.int64)
    coh_of = {s["subject"]: s["cohort"] for s in subj}
    nonadd = {}
    if sc["batch"] == "nonadditive":
        for c in batch:
            zmask = rs.random(p["n_asv"]) < sc["zero_frac"]
            nonadd[c] = (rs.uniform(0.7, 1.3, p["n_asv"]), rs.uniform(0.7, 1.3, N_HOST),
                         np.where(zmask, rs.uniform(*sc["zero_p"], p["n_asv"]), 0.0))
    popmean = np.concatenate([taxa_base, np.zeros(N_HOST)])
    for k, r in enumerate(rows):
        if r["missed"]:
            continue
        b_asv, b_host = batch[coh_of[r["subject"]]]
        lat = r["latent"]
        if nonadd:
            s_asv, s_host, pz = nonadd[coh_of[r["subject"]]]
            latg = popmean[asv_genus] + s_asv * (lat[asv_genus] - popmean[asv_genus])
            logit = latg + asv_logprop + b_asv
        else:
            logit = lat[asv_genus] + asv_logprop + b_asv
        prop = np.exp(logit - logit.max()); prop /= prop.sum()
        if rng.random() < p["shallow_frac"]:
            dep = int(rng.integers(*p["shallow_range"]))
        else:
            dep = int(np.exp(rng.normal(np.log(p["depth_median"]), p["depth_sdlog"])))
        alpha = prop * p["dm_precision"]
        q = rng.dirichlet(alpha + 1e-8)
        if nonadd:
            q = q * (rs.random(p["n_asv"]) >= pz); q = q / q.sum()
        counts[k] = rng.multinomial(dep, q)
        depth[k] = dep
        if nonadd:
            host[k] = s_host * lat[N_TAXA:] + b_host + rng.normal(0, p["host_noise_sd"], N_HOST)
        else:
            host[k] = lat[N_TAXA:] + b_host + rng.normal(0, p["host_noise_sd"], N_HOST)
    meta = dict(subject=np.array([r["subject"] for r in rows]),
                week=np.array([r["week"] for r in rows]),
                window=np.array([r["window"] for r in rows]),
                missed=np.array([r["missed"] for r in rows]))
    out = dict(B=B, subjects=subj, meta=meta, counts=counts, host=host, depth=depth,
               asv_genus=asv_genus, latent=np.array([r["latent"] for r in rows]),
               mu=np.array([r["mu"] for r in rows]), taxa_base=taxa_base,
               visits_per_window=visits_per_window, seed=seed, scenario=sc)
    if keep_subject_B:
        out["subject_B"] = np.array(subjB)
    return out


def truth_matrix(B, keep):
    """Planted signed edge matrix on the observed node set `keep` (off-diagonal)."""
    T = np.sign(B[np.ix_(keep, keep)])
    np.fill_diagonal(T, 0)
    return T


def transition_truth(B1, B2, keep):
    """Complete edge-state truth between two windows (revision, Reviewer 3 comment 4):
    reversal (present in both, opposite sign), appearance, disappearance, retained same sign."""
    T1, T2 = truth_matrix(B1, keep), truth_matrix(B2, keep)
    both = (T1 != 0) & (T2 != 0)
    return dict(reversal=both & (T1 != T2), same=both & (T1 == T2),
                appear=(T1 == 0) & (T2 != 0), disappear=(T1 != 0) & (T2 == 0))
