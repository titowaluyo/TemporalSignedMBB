"""Revision: individual-level recovery against heterogeneous subject-specific ground truth
(Reviewer 1 comment on LIONESS; Reviewer 3 comment 3). Writes results/lioness_check.json."""
import json, sys, time
import numpy as np
from multiprocessing import Pool
from tsmbb import generator as G, preprocess as PP, te as TE, classify as CL, benchmark as BM
from mc_group import j

SC = dict(init="stationary", burn_in=20, hetero=0.3, hetero_drop=0.25)


def shrink(gq, b_pop, lam):
    return np.linalg.solve(gq.XX + lam * np.eye(gq.XX.shape[0]), gq.XY + lam * b_pop)


def score(est, truth_q, mask):
    """est, truth_q: (n_subj, N, N); mask: edges evaluated (true edges of the group matrix)."""
    E = est[:, mask]; T = truth_q[:, mask]
    dE = E - E.mean(0); dT = T - T.mean(0)
    cors = [np.corrcoef(dE[:, k], dT[:, k])[0, 1] for k in range(E.shape[1]) if dT[:, k].std() > 0]
    return dict(rmse=float(np.sqrt(np.mean((E - T) ** 2))),
                sign=float(np.mean(np.sign(E) == np.sign(T))),
                dev_corr=float(np.nanmean(cors)),
                dev_corr_median=float(np.nanmedian(cors)),
                subj_profile_corr=float(np.mean([np.corrcoef(E[i], T[i])[0, 1] for i in range(len(E))])))


def one(args):
    seed, v = args
    sim = G.simulate(seed, visits_per_window=v, scenario=SC, keep_subject_B=True)
    pp = PP.preprocess(sim)
    keep = pp["obs_index"]
    ids = BM.group_ids(sim)
    out = dict(seed=seed, visits=v)
    for stage in ("observed", "latent"):
        if stage == "latent":
            Zs, _ = BM.stage_matrices(sim, pp)["latent"]
            pp2 = dict(pp, Z=Zs)
        else:
            pp2 = pp
        D = CL.SubjectData(sim, pp2, ("T1",))
        for grp in ("HC", "MDD"):
            sids = [q for q in ids[grp] if q in D.gram["T1"]]
            gR = D.group_gram(sids, "T1"); bR = gR.coef()
            truth = sim["subject_B"][sids][:, 0][:, keep][:, :, keep].astype(float)
            gk = ("HC", "T1") if grp == "HC" else ("MDD", "T1")
            mask = G.truth_matrix(sim["B"][gk], keep) != 0
            lio = np.array([TE.lioness_coef(gR, bR, len(sids), D.gram["T1"][q], True) for q in sids])
            res = dict(lioness=score(lio, truth, mask))
            res["population"] = score(np.repeat(bR[None], len(sids), 0) + 1e-12 *
                                      np.random.default_rng(0).normal(size=(len(sids),) + bR.shape), truth, mask)
            n_tr = np.mean([D.gram["T1"][q].n for q in sids])
            scale = np.mean(np.diag(gR.XX)) / gR.n
            best = None
            for lam_mult in (0.3, 1, 3, 10, 30, 100):
                lam = lam_mult * scale
                est = np.array([shrink(D.gram["T1"][q], bR, lam) for q in sids])
                s = score(est, truth, mask)
                res[f"shrink_{lam_mult}"] = s
            # data-driven lambda: leave-last-transition-out prediction error
            X, Y, own = PP.transitions(pp2, D.Zd, sids, "T1")
            errs = {}
            for lam_mult in (0.3, 1, 3, 10, 30, 100):
                lam = lam_mult * scale; e = 0.0
                for q in sids:
                    m = np.where(own == q)[0]
                    if len(m) < 2:
                        continue
                    g = TE.Gram(X[m[:-1]], Y[m[:-1]])
                    b = shrink(g, bR, lam)
                    e += float(((Y[m[-1]] - X[m[-1]] @ b) ** 2).sum())
                errs[lam_mult] = e
            lb = min(errs, key=errs.get)
            res["shrink_cv"] = dict(res[f"shrink_{lb}"], lam_mult=lb)
            res["mean_transitions"] = float(n_tr)
            out[f"{stage}|{grp}"] = res
    print(seed, v, flush=True)
    return out


if __name__ == "__main__":
    args = [(s, v) for v in (5, 10, 20) for s in range(6001, 6009)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        res = p.map(one, args, chunksize=1)
    json.dump(j(res), open("results/lioness_check.json", "w"))
