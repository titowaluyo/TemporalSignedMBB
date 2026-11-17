"""Quality control, CLR transform and cohort correction (Section 2.2)."""
import numpy as np
from .generator import N_TAXA, TAXA, HOST, REF_TAXON

DEPTH_MIN = 5000
PREV_MIN, PREV_COHORTS = 0.30, 2


def preprocess(sim, ref_taxon=REF_TAXON, fit_subjects=None, unseen_policy="recenter"):
    """Return dict with the QC-passed sample table and the 62-node network matrix Z.

    Revision options:
      ref_taxon     taxon whose CLR coordinate is omitted (sensitivity analysis, Reviewer 1 minor 6).
      fit_subjects  if given, the ASV prevalence filter and the cohort correction are fitted on the
                    samples of these subjects only (inductive preprocessing, Reviewers 1 and 3).
                    Cohorts without training samples are handled by `unseen_policy`:
                    'none'     - no correction (strictly prospective);
                    'recenter' - the cohort is centred on the training grand mean using its own
                                 unlabelled samples (site-level calibration)."""
    meta, counts, depth = sim["meta"], sim["counts"], sim["depth"]
    subj = {s["subject"]: s for s in sim["subjects"]}
    cohort = np.array([subj[s]["cohort"] for s in meta["subject"]])
    n_sched = len(meta["subject"])
    n_missed = int(meta["missed"].sum())
    ok = (~meta["missed"]) & (depth >= DEPTH_MIN)
    n_generated = n_sched - n_missed
    C = counts[ok]
    coh = cohort[ok]
    subj_ok = meta["subject"][ok]
    fit = np.ones(len(coh), bool) if fit_subjects is None else np.isin(subj_ok, np.asarray(fit_subjects))
    # ASV prevalence filter: present in >= 30% of samples in >= 2 cohorts (fitted on `fit` samples)
    prev_ok = np.zeros(C.shape[1], int)
    for c in np.unique(coh[fit]):
        prev_ok += ((C[fit & (coh == c)] > 0).mean(0) >= PREV_MIN).astype(int)
    keep_asv = prev_ok >= min(PREV_COHORTS, len(np.unique(coh[fit])))
    Ck = C[:, keep_asv]
    g = sim["asv_genus"][keep_asv]
    genus = np.zeros((Ck.shape[0], N_TAXA))
    for k in range(N_TAXA):
        genus[:, k] = Ck[:, g == k].sum(1)
    lg = np.log(genus + 0.5)
    clr = lg - lg.mean(1, keepdims=True)
    ref = TAXA.index(ref_taxon)
    taxa_keep = [i for i in range(N_TAXA) if i != ref]
    host = sim["host"][ok]
    Z = np.column_stack([clr[:, taxa_keep], host])
    names = [TAXA[i] for i in taxa_keep] + HOST
    # ASV-level CLR (for abundance comparators) and Shannon diversity (ASV level)
    la = np.log(Ck + 0.5)
    asv_clr = la - la.mean(1, keepdims=True)
    rel = Ck / Ck.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon = -np.nansum(np.where(rel > 0, rel * np.log(rel), 0), 1)
    # cohort correction: OLS on cohort indicators only, subtract cohort effects
    levels = sorted(np.unique(coh))
    D = np.column_stack([(coh == c).astype(float) for c in levels[1:]])
    X = np.column_stack([np.ones(len(coh)), D])

    def r2_cohort(M):
        beta = np.linalg.lstsq(X, M, rcond=None)[0]
        fit_ = X @ beta
        ss_res = ((M - fit_) ** 2).sum()
        ss_tot = ((M - M.mean(0)) ** 2).sum()
        return 1 - ss_res / ss_tot, beta

    def correct(M):
        if fit_subjects is None:
            _, beta = r2_cohort(M)
            return M - D @ beta[1:]
        out = M.copy()
        grand = M[fit].mean(0)
        for c in levels:
            sc_ = coh == c
            if (fit & sc_).any():          # cohort offset estimated from training samples only
                out[sc_] = M[sc_] - (M[fit & sc_].mean(0) - grand)
            elif unseen_policy == "recenter":
                out[sc_] = M[sc_] - (M[sc_].mean(0) - grand)
        return out

    r2_before, _ = r2_cohort(Z)
    Zc = correct(Z)
    r2_after, _ = r2_cohort(Zc)
    asv_clr_c = correct(asv_clr)
    return dict(Z=Zc, names=names, taxa_keep=taxa_keep, subject=meta["subject"][ok],
                week=meta["week"][ok], window=meta["window"][ok], cohort=coh,
                asv_clr=asv_clr_c, shannon=shannon, n_scheduled=n_sched, n_missed=n_missed,
                n_generated=n_generated, n_qc=int(ok.sum()), n_asv_kept=int(keep_asv.sum()),
                r2_cohort_before=float(r2_before), r2_cohort_after=float(r2_after), rows=np.where(ok)[0],
                obs_index=np.array(taxa_keep + list(range(N_TAXA, N_TAXA + len(HOST)))))


def demean_by_subject(pp, windows=None):
    """Subtract each subject's mean over its QC-passed visits. By default the mean is taken over
    the whole analysis horizon (group-level networks). For subject-level prediction, `windows`
    restricts the mean to the windows a model may use, so that no later visit enters a feature."""
    Z = pp["Z"].copy()
    s = pp["subject"]
    use = np.ones(len(s), bool) if windows is None else np.isin(pp["window"], list(windows))
    m = {}
    for q in np.unique(s):
        sel = (s == q) & use
        m[q] = Z[sel].mean(0) if sel.any() else Z[s == q].mean(0)
    return Z - np.array([m[x] for x in s])


def transitions(pp, Zd, subjects, window):
    """Within-window transitions between consecutive weekly visits for a subject set.
    Returns X (prev), Y (next), owner subject ids."""
    sset = set(int(x) for x in subjects)
    sel = np.where((pp["window"] == window) & np.isin(pp["subject"], list(sset)))[0]
    sub, wk = pp["subject"][sel], pp["week"][sel]
    o = np.lexsort((wk, sub))
    sel, sub, wk = sel[o], sub[o], wk[o]
    nxt = (sub[1:] == sub[:-1]) & (wk[1:] == wk[:-1] + 1)
    i0, i1 = sel[:-1][nxt], sel[1:][nxt]
    return Zd[i0], Zd[i1], sub[1:][nxt]
