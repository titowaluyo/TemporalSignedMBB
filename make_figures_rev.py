"""Revision figures. Reads results/summary_group.json, results/main_rev.json, results/main_rev_arrays.npz,
results/summary_subject.json, results/neuralmass.json, results/neural_revision.json, results/neural.json.
Writes figures_rev/Fig1.png ... (600 dpi) and TIFF copies.
    python make_figures_rev.py [fig2 fig3 ...]"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures as MF
from make_figures import MM, W, OI, POS, NEG, letter

MF.OUT = "figures_rev"
os.makedirs(MF.OUT, exist_ok=True)
save = MF.save
NETS = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T1", "NR-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
DESC = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]
GCOL = {"HC": "#0072B2", "MDD": "#7F7F7F", "R": "#009E73", "NR": "#D55E00"}


def S():
    return json.load(open("results/summary_group.json"))


def ci(m):
    return 1.96 * (m["mcse"] or 0)


# ------------------------------------------------------------------ Fig. 2 recovery and calibration
def fig2():
    s = S(); R = json.load(open("results/main_rev.json")); A = np.load("results/main_rev_arrays.npz")
    fig = plt.figure(figsize=(W, 125 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], width_ratios=[1.2, 1], hspace=0.75, wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    wk, coh, grp = A["qc_week"], A["qc_cohort"], A["qc_group"]
    rows = [(c, g) for c in ("SC1", "SC2", "SC3") for g in ("MDD", "HC")]
    weeks = np.arange(wk.max() + 1)
    M = np.array([[np.sum((coh == c) & (grp == g) & (wk == w)) for w in weeks] for c, g in rows], float)
    M[:, [5, 11]] = np.nan
    im = ax.imshow(M, aspect="auto", cmap="Greys", vmin=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{c} {g}" for c, g in rows])
    ax.set_xticks(weeks); ax.set_xticklabels([str(w - 4) for w in weeks], fontsize=5.5)
    ax.set_xlabel("Week (treatment starts after week 0)")
    for x0, lab in [(-0.5, "T1"), (5.5, "T2"), (11.5, "T3")]:
        ax.add_patch(plt.Rectangle((x0, -0.5), 5, len(rows), fill=False, ec=NEG, lw=1))
        ax.text(x0 + 2.5, -0.9, lab, ha="center", va="bottom", color=NEG, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("QC-passed samples")
    letter(ax, "a", -0.2, 1.1)
    # (b) precision / recall / signed recall over replications (cluster test)
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(NETS)); w = 0.27
    rec = s["main_recovery"]
    for k, (key, col, lab, mk) in enumerate([("precision", "#1F3B5A", "Precision", "o"), ("recall", "#9FC3E7", "Recall", "s"),
                                              ("signed_recall", OI["orange"], "Signed recall", "^")]):
        m = np.array([rec[n]["cluster"][key]["mean"] for n in NETS]); sd = np.array([rec[n]["cluster"][key]["sd"] for n in NETS])
        ax.errorbar(x + (k - 1) * w, m, yerr=sd, fmt=mk, ms=3, color=col, ecolor=col, lw=0.7, capsize=0, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(NETS, rotation=60, ha="right", fontsize=6)
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("Against latent coupling graph")
    ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=5.8, bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=0.8)
    letter(ax, "b", -0.2, 1.1)
    # (c) calibration under the null
    ax = fig.add_subplot(gs[1, 0])
    nul = s["scenarios"]["null"]["recovery"]["ALL"]; mainA = rec["ALL"]
    tests = [("naive", "Naive F"), ("dfcorr", "df-corrected F"), ("cluster", "Cluster-robust")]
    x = np.arange(3)
    m1 = [nul[t]["p05_rate_nulls"]["mean"] for t in tests[:3] for t in [t[0]]]
    e1 = [1.96 * nul[t]["p05_rate_nulls"]["mcse"] for t in [t[0] for t in tests]]
    m2 = [nul[t]["any_false"]["mean"] for t in [t[0] for t in tests]]
    m3 = [mainA[t]["fdp"]["mean"] for t in [t[0] for t in tests]]
    e3 = [1.96 * mainA[t]["fdp"]["mcse"] for t in [t[0] for t in tests]]
    ax.bar(x - 0.27, m1, 0.25, yerr=e1, color="#BBBBBB", label="Null: rejection rate per pair (α = 0.05)", ecolor="k", error_kw=dict(lw=0.6))
    ax.bar(x, m2, 0.25, color="#666666", label="Null: networks with any false edge (FDR)")
    ax.bar(x + 0.27, m3, 0.25, yerr=e3, color=OI["verm"], label="Main: realized FDR vs latent graph", ecolor="k", error_kw=dict(lw=0.6))
    ax.axhline(0.05, color="k", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([t[1] for t in tests])
    ax.set_ylabel("Rate"); ax.set_ylim(0, 0.16)
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")
    letter(ax, "c", -0.2, 1.06)
    # (d) stages and demeaning: FDP (1 - precision) and self-coefficient
    ax = fig.add_subplot(gs[1, 1])
    st = s["stages"]
    stages = [("latent", "Latent"), ("clr", "CLR, noiseless"), ("observed", "Observed")]
    hows = [("true", "true mean", "#1F3B5A"), ("full", "full-horizon mean", OI["sky"]), ("window", "window mean", OI["verm"])]
    for k, (h, lab, col) in enumerate(hows):
        xs, ys, es = [], [], []
        for i, (stg, _) in enumerate(stages):
            key = f"{stg}|{h}"
            if key not in st:
                continue
            v = [1 - st[key][n]["precision"]["mean"] for n in st[key]]
            e = [st[key][n]["precision"]["mcse"] for n in st[key]]
            xs.append(i + (k - 1) * 0.25); ys.append(np.mean(v)); es.append(1.96 * np.mean(e))
        ax.bar(xs, ys, 0.23, yerr=es, color=col, label=lab, ecolor="k", error_kw=dict(lw=0.6))
    ax.axhline(0.05, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(3)); ax.set_xticklabels([s_[1] for s_ in stages])
    ax.set_ylabel("Realized FDR (cluster-robust test)")
    ax.set_ylim(0, 0.36)
    ax.legend(frameon=False, fontsize=5.6, title="Subject mean removed", title_fontsize=5.6, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handletextpad=0.3)
    letter(ax, "d", -0.2, 1.06)
    save(fig, "Fig2")


# ------------------------------------------------------------------ Fig. 3 (illustrative heatmap, cluster test)
def fig3():
    R = json.load(open("results/main_rev.json")); A = np.load("results/main_rev_arrays.npz")
    MF.R = R; MF.A = {"A_HC_T1": A["A_HC_T1"], "A_MDD_T1": A["A_MDD_T1"]}
    return MF.fig3()


# ------------------------------------------------------------------ Fig. 4 misspecification
SCN = [("correct", "Correct (paired baseline)"), ("initdiag", "Original initialization"), ("tanh", "Saturating coupling"),
       ("lag2", "Delayed coupling (lag 2)"), ("glv", "gLV-type coupling"), ("t3", "Heavy-tailed noise (t$_3$)"),
       ("driver", "Latent common driver"), ("hetero", "Heterogeneous coupling"), ("nonadd", "Non-additive batch, zeros"),
       ("weak", "Weak coupling (×0.7)"), ("combined", "Combined violations")]


def fig4():
    s = S(); sc = s["scenarios"]
    labs = [l for k, l in SCN if k in sc]; keys = [k for k, l in SCN if k in sc]
    fig, axs = plt.subplots(1, 4, figsize=(W, 70 * MM), sharey=True)
    y = np.arange(len(keys))
    panels = [("precision", "Precision"), ("recall", "Recall"), ("signed_recall", "Signed recall")]
    for ax, (key, title) in zip(axs[:3], panels):
        m = [sc[k]["recovery"]["ALL"]["cluster"][key]["mean"] for k in keys]
        e = [1.96 * sc[k]["recovery"]["ALL"]["cluster"][key]["mcse"] for k in keys]
        ax.errorbar(m, y, xerr=e, fmt="o", ms=3, color="#1F3B5A", lw=0.7)
        ax.axvline(m[0], color="#999999", lw=0.5, ls="--")
        ax.set_title(title); ax.set_xlim(0, 1.02)
    ax = axs[3]
    m = [sc[k]["reversal"]["cluster"]["ALL"]["sensitivity"] for k in keys]
    fp = [sc[k]["reversal"]["cluster"]["ALL"]["fp_per_pair_evaluation"] for k in keys]
    ax.plot(m, y, "o", ms=3, color=OI["verm"], label="Reversal sensitivity")
    ax.plot(np.minimum(fp, 1), y, "x", ms=3.5, color="k", label="False reversals per pair evaluation")
    ax.set_title("Sign reversals"); ax.set_xlim(0, 1.02)
    ax.legend(frameon=False, fontsize=5.2, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    axs[0].set_yticks(y); axs[0].set_yticklabels(labs); axs[0].invert_yaxis()
    for ax, l in zip(axs, "abcd"):
        letter(ax, l, -0.12 if l != "a" else -0.9, 1.04)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.25, top=0.9, wspace=0.12)
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig. 5 reversals and topology
def fig5():
    s = S()
    fig = plt.figure(figsize=(W, 128 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.8, wspace=0.6, width_ratios=[1.35, 1, 1], left=0.2, right=0.97, top=0.95, bottom=0.12)
    ax = fig.add_subplot(gs[0, :2])
    rv = s["main_reversal"]["cluster"]; names = s["planted_order"]
    y = np.arange(len(names)); h = 0.26
    for k, (key, lab, col, hatch) in enumerate([("R:T1T2", "Responders T1→T2", "#009E73", ""),
                                                ("R:T2T3", "Responders T2→T3", "#7FCDB5", "////"),
                                                ("NR:T1T2", "Non-responders T1→T2", "#D55E00", "....")]):
        ax.barh(y + (k - 1) * h, rv[key]["planted_rate"], h, color=col, label=lab, hatch=hatch, ec="white", lw=0)
    ax.set_yticks(y); ax.set_yticklabels([n.replace(" -> ", " → ") for n in names], fontsize=5.8); ax.invert_yaxis()
    ax.set_xlabel(f"Detection rate over {s['main_R']} independent replications"); ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    letter(ax, "a", -0.36, 1.03)
    # (b) miss decomposition
    ax = fig.add_subplot(gs[0, 2])
    keys = ["R:T1T2", "R:T2T3", "NR:T1T2", "NR:T2T3"]
    comp = [("tp_total", "detected", OI["green"]), ("miss_absent_first", "absent in 1st", "#BBBBBB"),
            ("miss_absent_second", "absent in 2nd", "#888888"), ("miss_absent_both", "absent in both", "#444444"),
            ("miss_same_sign", "same sign", OI["purple"])]
    bottom = np.zeros(len(keys))
    for k, lab, col in comp:
        v = np.array([rv[x][k] / max(rv[x]["true_reversals_total"], 1) for x in keys])
        ax.bar(range(len(keys)), v, bottom=bottom, color=col, label=lab, width=0.7); bottom += v
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace(":", "\n") for k in keys], fontsize=5.6)
    ax.set_ylabel("Share of true reversals"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5.2, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    letter(ax, "b", -0.35, 1.03)
    # (c) H1 persistence z against density-matched null
    sm = s["sizematched"]
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(DESC))
    ax.bar(x, [sm[n]["h1_z"]["mean"] for n in DESC], yerr=[sm[n]["h1_z"]["sd"] for n in DESC],
           color=[GCOL[n.split("-")[0]] for n in DESC], ecolor="k", error_kw=dict(lw=0.6))
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1.96, color="k", lw=0.5, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(DESC, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Total H$_1$ persistence, z vs density null")
    letter(ax, "c", -0.3, 1.03)
    # (d) directed cycles and loop-arc recall
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x, [sm[n]["cyc4"]["mean"] + sm[n]["cyc5"]["mean"] for n in DESC],
           color=[GCOL[n.split("-")[0]] for n in DESC])
    ax.set_xticks(x); ax.set_xticklabels(DESC, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Directed 4- and 5-cycles (mean)")
    ax2 = ax.twinx(); ax2.plot(x, [sm[n]["loop_arc_recall"]["mean"] for n in DESC], "kD", ms=2.5)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Planted loop arcs recovered", fontsize=6); ax2.spines["right"].set_visible(True)
    letter(ax, "d", -0.3, 1.03)
    # (e) oracle topology
    ax = fig.add_subplot(gs[1, 2])
    lo = s["loop_oracle"]
    labs = ["full", "without_loop1", "without_loop2", "without_loop3", "without_all_loops"]
    tl = ["NR-T3", "− loop 1", "− loop 2", "− loop 3", "− all"]
    for k, (bg, col, lab) in enumerate([("bg=True", "#1F3B5A", "with background"), ("bg=False", OI["sky"], "curated + loops only")]):
        ax.bar(np.arange(5) + (k - 0.5) * 0.38, [lo[bg][l]["n_classes"]["mean"] for l in labs], 0.36, color=col, label=lab)
    ax.set_xticks(range(5)); ax.set_xticklabels(tl, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("H$_1$ classes, oracle input"); ax.set_ylim(0, 52)
    ax.legend(frameon=False, fontsize=5.2, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    letter(ax, "e", -0.3, 1.03)
    save(fig, "Fig5")


# ------------------------------------------------------------------ Fig. 6 perturbation score and resilience
def fig6():
    s = S(); sp = s["spts_validation"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 62 * MM), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]
    nets = list(sp)
    variants = [("TE|out|1e-15", "PS, signed TE (Eq. 9)"), ("coef|out|1e-15", "PS, coefficients"),
                ("TE|sym|1e-15", "PS, symmetrized"), ("oracle_B|out|1e-15", "PS, planted matrix"),
                ("VAR_estimated", "Estimated VAR response")]
    x = np.arange(len(nets)); w = 0.16
    cols = ["#1F3B5A", OI["sky"], OI["purple"], "#999999", OI["green"]]
    for k, (v, lab) in enumerate(variants):
        m = [sp[n][v]["rho_longrun"]["mean"] for n in nets]; e = [sp[n][v]["rho_longrun"]["sd"] for n in nets]
        ax.bar(x + (k - 2) * w, m, w, yerr=e, color=cols[k], label=lab, ecolor="k", error_kw=dict(lw=0.5))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(nets)
    ax.set_ylabel("Spearman ρ with long-run response")
    ax.legend(frameon=False, fontsize=5.4, ncol=2, loc="lower left")
    letter(ax, "a", -0.14, 1.03)
    ax = axs[1]
    sm = s["sizematched"]
    xx = np.arange(len(DESC))
    ax.bar(xx, [sm[n]["dfrag"]["mean"] for n in DESC], yerr=[1.96 * sm[n]["dfrag"]["mcse"] for n in DESC],
           color=[GCOL[n.split("-")[0]] for n in DESC], ecolor="k", error_kw=dict(lw=0.6))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(xx); ax.set_xticklabels(DESC, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Excess fragmentation Δfrag")
    letter(ax, "b", -0.3, 1.03)
    fig.tight_layout(); save(fig, "Fig6")




# ------------------------------------------------------------------ Fig. 7 subject level
def fig7():
    U = json.load(open("results/summary_subject.json"))
    fig = plt.figure(figsize=(W, 125 * MM))
    gs = fig.add_gridspec(2, 3, hspace=0.85, wspace=0.55, left=0.08, right=0.98, top=0.95, bottom=0.19)
    LC = {"lr": "#1F3B5A", "xgb": OI["sky"], "rf": OI["orange"]}
    LN = {"lr": "Logistic (L2)", "xgb": "Boosted trees", "rf": "Random forest"}
    # (a) case-control, main scenario, matched learners
    ax = fig.add_subplot(gs[0, 0:2])
    cc = U["cc_main"]["case_control"]
    fams = [("network", "Network"), ("static_signed", "Static signed"), ("static_unsigned", "Static unsigned"),
            ("abundance", "Abundance"), ("network+abundance", "Network +\nabundance")]
    x = np.arange(len(fams)); w = 0.26
    for k, L in enumerate(("lr", "xgb", "rf")):
        m = [cc[f"{f}|{L}"]["test"]["auc"]["mean"] for f, _ in fams]
        e = [cc[f"{f}|{L}"]["test"]["auc"]["sd"] for f, _ in fams]
        ax.bar(x + (k - 1) * w, m, w, yerr=e, color=LC[L], label=LN[L], ecolor="k", error_kw=dict(lw=0.5))
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in fams], fontsize=6); ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Case-control AUROC (test)"); ax.legend(frameon=False, ncol=3, fontsize=5.8, loc="upper left")
    letter(ax, "a", -0.08, 1.03)
    # (b) signal allocation scenarios
    ax = fig.add_subplot(gs[0, 2])
    scen = [("cc_main", "Mean +\ncoupling"), ("cc_calib", "Calibrated\nmean"), ("cc_coupling", "Coupling\nonly")]
    xx = np.arange(len(scen))
    for k, (f, col, lab) in enumerate([("network", "#1F3B5A", "Network"), ("abundance", OI["orange"], "Abundance")]):
        m = [U[s]["case_control"][f"{f}|lr"]["test"]["auc"]["mean"] if s in U else np.nan for s, _ in scen]
        e = [U[s]["case_control"][f"{f}|lr"]["test"]["auc"]["sd"] if s in U else 0 for s, _ in scen]
        ax.bar(xx + (k - 0.5) * 0.38, m, 0.36, yerr=e, color=col, label=lab, ecolor="k", error_kw=dict(lw=0.5))
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in scen], fontsize=5.8); ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUROC (logistic, test)"); ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    letter(ax, "b", -0.3, 1.03)
    # (c) response with oracle
    ax = fig.add_subplot(gs[1, 0])
    rows = [("cc_main", "Main"), ("rp_early", "Early\nprognostic")]
    for k, (f, col, lab) in enumerate([("network", "#1F3B5A", "Network (T1+T2)"), ("abundance_T1T2", OI["orange"], "Abundance"),
                                       ("oracle", "#999999", "Oracle")]):
        m, e = [], []
        for s, _ in rows:
            rp = U.get(s, {}).get("response", {})
            key = f if f == "oracle" else f"{f}|lr"
            v = rp.get(key, {}).get("pooled", {}).get("auc") if key in rp else None
            m.append(v["mean"] if v else np.nan); e.append(v["sd"] if v else 0)
        ax.bar(np.arange(2) + (k - 1) * 0.27, m, 0.25, yerr=e, color=col, label=lab, ecolor="k", error_kw=dict(lw=0.5))
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xticks(range(2)); ax.set_xticklabels([l for _, l in rows], fontsize=6); ax.set_ylim(0.3, 0.9)
    ax.set_ylabel("Response AUROC (pooled held-out)"); ax.legend(frameon=False, fontsize=5.4, loc="upper left")
    letter(ax, "c", -0.32, 1.03)
    # (d) design experiment
    ax = fig.add_subplot(gs[1, 1])
    D = U["design"]
    for c, ls in (("1.0", "-"), ("0.7", "--")):
        for f, col, mk in (("network", "#1F3B5A", "o"), ("abundance", OI["orange"], "s"), ("network+abundance", OI["green"], "^")):
            vv = [5, 10, 20]
            m = [D.get(f"design_v{v}_c{c}", {}).get(f"{f}|lr", {}).get("test", {}).get("auc", {}).get("mean", np.nan) for v in vv]
            e = [1.96 * (D.get(f"design_v{v}_c{c}", {}).get(f"{f}|lr", {}).get("test", {}).get("auc", {}).get("mcse") or 0) for v in vv]
            ax.errorbar(vv, m, yerr=e, color=col, ls=ls, marker=mk, ms=3, lw=0.8,
                        label=f"{f.replace('network+abundance', 'combined')}, coupling ×{c}")
    ax.set_xticks([5, 10, 20]); ax.set_xlabel("Visits per window"); ax.set_ylabel("Case-control AUROC (logistic, test)")
    ax.set_ylim(0.45, 1.02); ax.legend(frameon=False, fontsize=4.6, loc="upper center", ncol=2, bbox_to_anchor=(0.45, -0.28))
    letter(ax, "d", -0.3, 1.03)
    # (e) LIONESS recovery
    ax = fig.add_subplot(gs[1, 2])
    L = U["lioness"]
    for est, col in (("lioness", "#1F3B5A"), ("shrink_cv", OI["green"])):
        for st, ls in (("observed", "-"), ("latent", "--")):
            m = [L[str(v)][f"{st}|HC"][est]["dev_corr"]["mean"] for v in (5, 10, 20)]
            ax.plot([5, 10, 20], m, color=col, ls=ls, marker="o", ms=3, lw=0.8,
                    label=f"{'LIONESS' if est == 'lioness' else 'Shrinkage'}, {st}")
    ax.set_xticks([5, 10, 20]); ax.set_xlabel("Visits per window"); ax.set_ylim(0, 0.6)
    ax.set_ylabel("Recovery of individual differences (r)"); ax.legend(frameon=False, fontsize=5.0, loc="upper left")
    letter(ax, "e", -0.3, 1.03)
    save(fig, "Fig7")


# ------------------------------------------------------------------ Fig. 8 neural mass
def fig8():
    U = json.load(open("results/summary_subject.json")); nm = U["neuralmass"]
    fig = plt.figure(figsize=(W, 64 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.08, right=0.98, top=0.88, bottom=0.3, width_ratios=[1.5, 1, 1])
    variants = [("direct", "Weak coupling"), ("strong", "Strong coupling"), ("mixing", "Volume conduction"),
                ("bold_TR0.72", "BOLD, TR 0.72 s"), ("bold_TR2.0", "BOLD, TR 2 s")]
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(variants)); w = 0.25
    for k, (m_, col, lab) in enumerate([("k1", "#BBBBBB", "k = 1"), ("kBIC", "#1F3B5A", "k by BIC")]):
        for j, (key, mk) in enumerate([("precision", "o"), ("recall", "s")]):
            pass
        p = [nm[v][m_]["precision"]["mean"] if m_ in nm[v] and nm[v][m_]["precision"]["mean"] is not None else np.nan for v, _ in variants]
        r = [nm[v][m_]["recall"]["mean"] if m_ in nm[v] else np.nan for v, _ in variants]
        ax.bar(x + (k - 0.5) * 0.4, p, 0.38, color=col, label=f"Precision, {lab}")
        ax.plot(x + (k - 0.5) * 0.4, r, "D", color=OI["orange"] if k else "#E7B868", ms=3, label=f"Recall, {lab}")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in variants], rotation=35, ha="right", fontsize=6)
    ax.set_ylim(0, 1.32); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("Directed edge recovery")
    ax.legend(frameon=False, fontsize=5.0, ncol=2, loc="upper center")
    letter(ax, "a", -0.14, 1.04)
    ax = fig.add_subplot(gs[0, 1])
    vv = variants[:3]
    xx = np.arange(len(vv))
    for k, (key, col, lab) in enumerate([("exc_recall", OI["sky"], "Excitatory"), ("inh_recall", OI["verm"], "Inhibitory")]):
        ax.bar(xx + (k - 0.5) * 0.38, [nm[v]["kBIC"][key]["mean"] for v, _ in vv], 0.36, color=col, label=lab,
               yerr=[1.96 * nm[v]["kBIC"][key]["mcse"] for v, _ in vv], ecolor="k", error_kw=dict(lw=0.5))
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in vv], rotation=35, ha="right", fontsize=6)
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel("Recall with correct sign (k by BIC)"); ax.legend(frameon=False, fontsize=5.4, ncol=2, loc="upper center")
    letter(ax, "b", -0.3, 1.04)
    ax = fig.add_subplot(gs[0, 2])
    for k, (m_, col, lab) in enumerate([("k1", "#BBBBBB", "k = 1"), ("kBIC", "#1F3B5A", "k by BIC")]):
        ax.bar(xx + (k - 0.5) * 0.38, [nm[v][m_]["rev_sens"] for v, _ in vv], 0.36, color=col, label=lab)
        for i, (v, _) in enumerate(vv):
            ax.text(xx[i] + (k - 0.5) * 0.38, nm[v][m_]["rev_sens"] + 0.02, str(nm[v][m_]["rev_fp"]), ha="center", fontsize=5)
    ax.set_xticks(xx); ax.set_xticklabels([l for _, l in vv], rotation=35, ha="right", fontsize=6)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Reversal sensitivity (false reversals)"); ax.legend(frameon=False, fontsize=5.4, loc="upper right")
    letter(ax, "c", -0.3, 1.04)
    save(fig, "Fig8")


# ------------------------------------------------------------------ Fig. 9 public neural data
def fig9():
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    R = json.load(open("results/neural.json")); A = np.load("results/neural_arrays.npz")
    NR = json.load(open("results/neural_revision.json"))
    ns = R["netsim"]; G, Rc = A["G"], A["R_cte"]; N = G.shape[0]
    fig = plt.figure(figsize=(W, 120 * MM))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.3, 1], hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.95, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    cat = np.zeros((N, N), int)
    for i in range(N):
        for jx in range(N):
            if i == jx:
                continue
            if G[i, jx] and Rc[i, jx] and not Rc[jx, i]:
                cat[i, jx] = 1
            elif G[i, jx] and Rc[i, jx] and Rc[jx, i]:
                cat[i, jx] = 2
            elif Rc[i, jx] and not (G[i, jx] or G[jx, i]):
                cat[i, jx] = 4
    cmap = ListedColormap(["#F2F2F2", "#1F5FA8", "#92C5DE", "#E69F00", NEG])
    ax.imshow(cat, cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(range(N)); ax.set_xticklabels(range(1, N + 1), fontsize=4.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(range(1, N + 1), fontsize=4.5)
    ax.set_xlabel("Target node"); ax.set_ylabel("Source node"); ax.set_title("NetSim: pooled conditional TE")
    ax.legend(handles=[Patch(color="#1F5FA8", label="correct direction"), Patch(color="#92C5DE", label="both directions"),
                       Patch(color=NEG, label="false positive")], frameon=False, fontsize=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.25), ncol=1)
    letter(ax, "a", -0.3, 1.03)
    ax = fig.add_subplot(gs[0, 1])
    labs = [("cTE_k1", "cTE\nk = 1"), ("cTE_kBIC", "cTE\nBIC"), ("bTE_k1", "Bivar.\nTE"),
            ("PCMCI", "PCMCI"), ("parcorr", "Partial\ncorr.")]
    x = np.arange(len(labs)); w = 0.36
    for off, key, col, lab in [(-w / 2, "f1", "#1F5FA8", "Directed F1"), (w / 2, "und_f1", "#92C5DE", "Undirected F1")]:
        v = [ns[k]["single"].get(key, {}).get("mean", np.nan) if ns[k]["single"].get(key) else np.nan for k, _ in labs]
        e = [ns[k]["single"].get(key, {}).get("mcse", 0) if ns[k]["single"].get(key) else 0 for k, _ in labs]
        ax.bar(x + off, v, w, yerr=e, color=col, label=lab, error_kw=dict(lw=0.6))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in labs], fontsize=5.8); ax.set_ylim(0, 1)
    ax.set_ylabel("F1, single subjects (mean of 50)"); ax.legend(frameon=False, fontsize=5.6)
    letter(ax, "b", -0.18, 1.03)
    ax = fig.add_subplot(gs[0, 2])
    n2 = NR["netsim"]
    meth = [("dir_acc_cTE_k1", "Cond. TE"), ("dir_acc_bTE", "Bivar. TE"), ("dir_acc_pcmciplus_oriented_only", "PCMCI+"),
            ("dir_acc_patel_tau", "Patel's τ"), ("dir_acc_lingam", "DirectLiNGAM")]
    ax.barh(range(len(meth)), [n2[k] for k, _ in meth], color=["#1F5FA8", "#92C5DE", OI["purple"], "#999999", OI["green"]])
    ax.axvline(0.5, color="k", lw=0.5, ls=":")
    ax.set_yticks(range(len(meth))); ax.set_yticklabels([l for _, l in meth], fontsize=6); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Direction accuracy (pooled)")
    letter(ax, "c", -0.5, 1.03)
    ax = fig.add_subplot(gs[1, :])
    E = NR["eeg"]
    labs = [f"{e['band'][0]}–{e['band'][1]} Hz, {e['reference']}, lags 1–{e['kmax']}\n(k = {e['k_bic']})" for e in E]
    x = np.arange(len(E))
    ax.bar(x - 0.2, [e["edges_pre"] for e in E], 0.38, color="#999999", label="pre-stimulus edges")
    ax.bar(x + 0.2, [e["edges_post"] for e in E], 0.38, color="#1F3B5A", label="post-stimulus edges")
    for i, e in enumerate(E):
        ax.text(x[i], max(e["edges_pre"], e["edges_post"]) + 6, f"one-window edges p = {e['chg_p']:.2f}\nsplit-half shared = {e['split_half'][2]}",
                ha="center", fontsize=5.4)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5.8); ax.set_ylabel("Retained edges (EEG)")
    ax.set_ylim(0, 270); ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    letter(ax, "d", -0.05, 1.03)
    save(fig, "Fig9")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig2", "fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9"]
    for f in todo:
        globals()[f]()
