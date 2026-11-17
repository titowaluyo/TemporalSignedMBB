"""Online Resource 2: supplementary tables generated from the result files."""
import json, os
import numpy as np

S = json.load(open("results/summary_group.json"))
U = json.load(open("results/summary_subject.json"))
L = []
f3 = lambda m: "--" if m is None or m.get("mean") is None else f"{m['mean']:.3f}"
L.append("# Online Resource 2\n\n**Ground-truth benchmarking of signed, directed lagged connectivity: TemporalSignedMBB from a simulated gut-brain model to neural data**\n\nTito Waluyo Purboyo\n\nDepartment of Biomedical Engineering, School of Electrical Engineering, Telkom University, Bandung 40257, Indonesia; titowaluyo@telkomuniversity.ac.id\n\nSupplementary tables. Every value is read from the result files of Online Resource 1. Means are over independent replications; MCSE, Monte Carlo standard error.\n")

# S1 CLR
c = U["clr"]
L.append(f"\n**Table S1** Sensitivity of Layer 1 to the omitted CLR coordinate ({c['R']} replications of the main scenario, cluster-robust test). Precision, recall and conditional sign accuracy on the nodes common to all four choices; Jaccard index of the signed edge sets relative to the main analysis (Flavonifractor omitted) and number of shared edges with opposite signs, summed over replications. In the generator, Flavonifractor carries no coupling, whereas the other three taxa carry one or two background edges in most replications\n")
L.append("| Omitted coordinate | Network | Precision | Recall | Sign accuracy | Jaccard vs main | Sign changes |\n|:--|:--|:--:|:--:|:--:|:--:|:--:|")
for ref in ("Flavonifractor", "Holdemania", "Turicibacter", "Christensenella"):
    for net in ("HC-T1", "MDD-T1", "NR-T3"):
        m = c["metrics"][f"{ref}|{net}"]
        ag = c["agreement"].get(f"{ref}|{net}")
        L.append(f"| {ref} | {net} | {f3(m['precision'])} | {f3(m['recall'])} | {f3(m['sign'])} | "
                 f"{'--' if ag is None else f'{ag['jaccard']['mean']:.3f}'} | {'--' if ag is None else ag['sign_disagree']} |")

# S2 null calibration per network
nul = S["scenarios"]["null"]["recovery"]
L.append(f"\n**Table S2** Null scenario ({S['scenarios']['null']['R']} replications): per-pair rejection rate at α = 0.05 (MCSE) and proportion of networks with any false edge after Benjamini-Hochberg correction at q = 0.05, by network and test\n")
L.append("| Network | Naive: rate | Naive: any false | df-corrected: rate | df-corrected: any false | Cluster: rate | Cluster: any false |\n|:--|:--:|:--:|:--:|:--:|:--:|:--:|")
for net in [k for k in nul if k != "ALL"]:
    row = [net]
    for t in ("naive", "dfcorr", "cluster"):
        a = nul[net][t]
        row += [f"{a['p05_rate_nulls']['mean']:.4f} ({a['p05_rate_nulls']['mcse']:.4f})", f"{a['any_false']['mean']:.2f}"]
    L.append("| " + " | ".join(row) + " |")

# S3 stages
st = S["stages"]
L.append("\n**Table S3** Observation stages and demeaning (main scenario, 30 replications, cluster-robust test): precision, recall and mean estimated self-coefficient (true value 0.45) by stage, subject-mean treatment and network\n")
L.append("| Stage and subject mean | Network | Precision | Recall | Self-coefficient |\n|:--|:--|:--:|:--:|:--:|")
for key in st:
    for net in st[key]:
        d = st[key][net]
        L.append(f"| {key.replace('|', ', ')} | {net} | {f3(d['precision'])} | {f3(d['recall'])} | {f3(d['self_coef_mean'])} |")

# S4 scenario x network
L.append("\n**Table S4** Precision / recall of the cluster-robust Layer 1 by scenario and network (20 replications per scenario)\n")
nets = ["HC-T1", "MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
L.append("| Scenario | " + " | ".join(nets) + " |\n|:--|" + ":--:|" * len(nets))
for sc, d in S["scenarios"].items():
    if sc == "null":
        continue
    L.append(f"| {sc} | " + " | ".join(f"{d['recovery'][n]['cluster']['precision']['mean']:.3f} / {d['recovery'][n]['cluster']['recall']['mean']:.3f}" for n in nets) + " |")

# S5 SPTS variants
sp = S["spts_validation"]
L.append("\n**Table S5** Perturbation score against the long-run response of the generating VAR: mean Spearman correlation across nodes over 30 replications, for every weighting, Laplacian and pseudoinverse tolerance, with mean rank and condition number of the Laplacian\n")
L.append("| Variant (weights, Laplacian, tolerance) | " + " | ".join(sp) + " |\n|:--|" + ":--:|" * len(sp))
for v in sp[list(sp)[0]]:
    if v == "scale_ratio":
        continue
    L.append(f"| {v.replace('|', ', ')} | " + " | ".join(f"{sp[n][v]['rho_longrun']['mean']:.2f}" for n in sp) + " |")
L.append("| Laplacian rank (signed TE, out) | " + " | ".join(f"{sp[n]['TE|out|1e-15']['rank']['mean']:.1f}" for n in sp) + " |")
L.append("| Condition number (signed TE, out) | " + " | ".join(f"{sp[n]['TE|out|1e-15']['cond']['mean']:.0f}" for n in sp) + " |")

# S6 size-matched descriptors
sm = S["sizematched"]
keys = [("edges", "Edges"), ("precision", "Precision"), ("recall", "Recall"), ("pos", "Positive (%)"), ("h1", "Total H1"),
        ("h1_z", "H1 z vs null"), ("h1_log", "H1 (log distance)"), ("w2", "W2 to reference"), ("frustration_exact", "Frustration (exact)"),
        ("frustration_multi", "Frustration (multigraph)"), ("frustration_greedy", "Frustration (greedy)"), ("auc", "AUC^R"),
        ("dfrag", "Δfrag"), ("cyc3", "Directed 3-cycles"), ("cyc4", "Directed 4-cycles"), ("cyc5", "Directed 5-cycles"),
        ("loop_arc_recall", "Loop arcs recovered")]
L.append("\n**Table S6** Descriptors of the size-matched networks (200 subjects per group), mean (SD) over 30 replications\n")
nets8 = list(sm)
L.append("| Descriptor | " + " | ".join(nets8) + " |\n|:--|" + ":--:|" * len(nets8))
for k, lab in keys:
    L.append(f"| {lab} | " + " | ".join(f"{sm[n][k]['mean']:.3f} ({sm[n][k]['sd']:.3f})" for n in nets8) + " |")

# S7 motifs
m = U["motifs"]
L.append(f"\n**Table S7** Signed-motif enrichment over {m['R']} replications (null preserving positive and negative node degrees, 100 null networks, BH q < 0.10). Most reproducible disease-specific classes (fraction of replications in which the class was enriched in the network and in no control network), and direct comparison of class z scores between MDD and controls at T1\n")
L.append("| Network | Class code (six arcs, signed) | Fraction of replications |\n|:--|:--|:--:|")
for x in m["most_reproducible"]:
    L.append(f"| {x['net']} | {x['cls']} | {x['frac']:.2f} |")
L.append("\n| Class code | Replications compared | Mean z difference (MDD − HC) | p | p (Holm) |\n|:--|:--:|:--:|:--:|:--:|")
for x in m["direct_mdd_vs_hc"]:
    L.append(f"| {x['cls']} | {x['n']} | {x['mean']:.2f} | {x['p']:.3f} | {x['p_holm']:.3f} |")

# S8 comparators (if available)
if os.path.exists("results/comparators_mgb.json"):
    C = json.load(open("results/comparators_mgb.json"))
    L.append(f"\n**Table S8** External directed-connectivity comparators on the microbiome benchmark ({len(C)} replications of the main scenario, full-size networks): precision, recall and conditional sign accuracy against the latent coupling graph\n")
    meths = ["cTE_cluster", "lasso_VAR", "bivariate_GC", "PCMCI"]
    L.append("| Network | " + " | ".join(meths) + " |\n|:--|" + ":--:|" * len(meths))
    for net in ("HC-T1", "MDD-T1", "R-T3", "NR-T3"):
        row = []
        for me in meths:
            v = [r[net][me] for r in C if "precision" in r[net][me]]
            row.append("--" if not v else f"{np.nanmean([x['precision'] for x in v]):.3f} / {np.nanmean([x['recall'] for x in v]):.3f} / {np.nanmean([x['sign'] for x in v]):.3f}")
        L.append(f"| {net} | " + " | ".join(row) + " |")

# S9 classification detail (if available)
if "cc_main" in U:
    cc = U["cc_main"]["case_control"]
    L.append(f"\n**Table S9** Case-control classification, main scenario ({U['cc_main']['R']} replications): every feature family and learner. AUROC on the internal test partition and on SC2 (site-level recentering; for abundance also without correction), sensitivity and specificity at the frozen training-derived threshold and at the test-optimized threshold (descriptive only)\n")
    L.append("| Family, learner | AUROC test | AUROC SC2 | AUROC SC2, no correction | Sens / spec, frozen | Sens / spec, test-optimized |\n|:--|:--:|:--:|:--:|:--:|:--:|")
    for k, v in cc.items():
        if not isinstance(v, dict) or "test" not in v or k.startswith("diff") or k == "shap_share":
            continue
        t = v["test"]; h = v.get("heldout", {})
        nc = v.get("heldout_nocorrection", {}).get("auc")
        L.append(f"| {k.replace('|', ', ')} | {f3(t.get('auc'))} | {f3(h.get('auc'))} | {f3(nc) if nc else '--'} | "
                 f"{f3(t.get('sens'))} / {f3(t.get('spec'))} | {f3(t.get('sens_test_optimised'))} / {f3(t.get('spec_test_optimised'))} |")
open("results/OnlineResource2.md", "w").write("\n".join(L) + "\n")
print("written", len(L))
