"""Design-factor experiment (Section 2.10): 5, 10 and 20 visits per window, two seeds.
Writes results/design_factor.json."""
import json, time, sys
from tsmbb import generator as G, preprocess as PP, classify as CL

SEEDS, VISITS = (42, 43), (5, 10, 20)


def main():
    t0 = time.time()
    out = []
    for seed in SEEDS:
        for v in VISITS:
            sim = G.simulate(seed, visits_per_window=v)
            pp = PP.preprocess(sim)
            r = CL.run(sim, pp, seed, full=False)
            cc, rp = r["case_control"], r["response"]
            out.append(dict(seed=seed, visits=v,
                            cc_network=cc["network"]["test"], cc_abundance=cc["rf_asv"]["test"],
                            cc_combined=cc["network+asv"]["test"],
                            rp_network=rp["network"]["pooled"], rp_abundance=rp["rf_asv_T1"]["pooled"],
                            rp_diff=rp["rf_asv_T1"]["diff_vs_network"]))
            print(f"[{time.time()-t0:.0f}s] seed {seed} visits {v}: cc {cc['network']['test']['auc']:.3f} "
                  f"vs {cc['rf_asv']['test']['auc']:.3f}; resp {rp['network']['pooled']['auc']:.3f} "
                  f"vs {rp['rf_asv_T1']['pooled']['auc']:.3f}", flush=True)
            json.dump(out, open("results/design_factor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
