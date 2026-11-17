"""Revision: (1) sensitivity of signed edges to the omitted CLR coordinate (Reviewer 1 minor 6,
Reviewer 3 comment 2); (2) stability of the node topological feature phi_i (Eq. 8) under node
relabelling and small weight perturbations (Reviewer 3 comment 7); (3) scales of the SPTS
components (Reviewer 1 minor 11, Reviewer 3 comment 8). Writes results/clr_phi_checks.json."""
import json, sys
import numpy as np
from scipy import stats
from multiprocessing import Pool
from tsmbb import generator as G, preprocess as PP, te as TE, topology as TP, benchmark as BM
from mc_group import j, SCEN

REFS = ["Flavonifractor", "Holdemania", "Turicibacter", "Christensenella"]


def one(seed):
    sim = G.simulate(seed, scenario=SCEN["main"][0])
    ids = BM.group_ids(sim)
    out = dict(seed=seed)
    common = [n for n in G.NODES if n not in REFS]
    nets = {}
    for ref in REFS:
        pp = PP.preprocess(sim, ref_taxon=ref)
        Zd = BM.demean(pp["Z"], pp, "full")
        names = pp["names"]
        idx = [names.index(n) for n in common]
        lat = [G.IDX[n] for n in common]
        for net in ("HC-T1", "MDD-T1", "NR-T3"):
            g, w = net.split("-")
            X, Y, own = PP.transitions(pp, Zd, ids[g], w)
            o = TE.cte_tests(X, Y, own)
            R = o["cluster"]["R"][np.ix_(idx, idx)]; Bh = o["B"][np.ix_(idx, idx)]
            T = G.truth_matrix(BM.true_B(sim, net), lat)
            m = BM.edge_metrics(R, Bh, T)
            nets[(ref, net)] = np.where(R, np.sign(Bh), 0)
            out[f"{ref}|{net}"] = m
    agree = {}
    for net in ("HC-T1", "MDD-T1", "NR-T3"):
        base = nets[(REFS[0], net)]
        for ref in REFS[1:]:
            o = nets[(ref, net)]
            inter = (base != 0) & (o != 0); uni = (base != 0) | (o != 0)
            agree[f"{ref}|{net}"] = dict(jaccard=float(inter.sum() / uni.sum()),
                                         sign_disagree_shared=int((inter & (base != o)).sum()))
    out["agreement_vs_Flavonifractor"] = agree
    # phi stability and SPTS component scales (main preprocessing)
    pp = PP.preprocess(sim); Zd = BM.demean(pp["Z"], pp, "full")
    rng = np.random.default_rng(seed)
    phi_res = {}
    for net in ("HC-T1", "NR-T3"):
        g, w = net.split("-")
        X, Y, own = PP.transitions(pp, Zd, ids[g], w)
        A = TE.cte_tests(X, Y, own)["cluster"]["A"]
        _, tot, phi = TP.persistence(A, cocycles=True)
        rp, rw = [], []
        for _ in range(10):
            p = rng.permutation(A.shape[0])
            _, _, ph2 = TP.persistence(A[np.ix_(p, p)], cocycles=True)
            back = np.empty_like(ph2); back[p] = ph2
            rp.append(stats.spearmanr(phi, back)[0] if phi.std() and back.std() else np.nan)
            Ap = A * np.exp(rng.normal(0, 0.01, A.shape))
            _, _, ph3 = TP.persistence(Ap, cocycles=True)
            rw.append(stats.spearmanr(phi, ph3)[0] if phi.std() and ph3.std() else np.nan)
        ps = TP.perturbation_scores(A)
        r = np.zeros(A.shape[0])
        phi_res[net] = dict(phi_perm_rho=float(np.nanmean(rp)), phi_perm_rho_min=float(np.nanmin(rp)),
                            phi_weight_rho=float(np.nanmean(rw)), phi_weight_rho_min=float(np.nanmin(rw)),
                            nonzero_phi=int((phi > 0).sum()), total_h1=tot,
                            ps_median=float(np.median(ps)), ps_iqr=float(np.subtract(*np.percentile(ps, [75, 25]))),
                            phi_median_nonzero=float(np.median(phi[phi > 0])) if (phi > 0).any() else 0.0,
                            phi_max=float(phi.max()))
    out["phi"] = phi_res
    print(seed, flush=True)
    return out


if __name__ == "__main__":
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        res = p.map(one, range(1001, 1011), chunksize=1)
    json.dump(j(res), open("results/clr_phi_checks.json", "w"))
