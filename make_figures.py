"""Figures 2-6 (MGB analyses; Fig. 1 is drawn by fig1_design.py) at 600 dpi, 174 mm wide, sans-serif (Arial or its metric clone
Liberation Sans), lowercase panel letters. Reads results/main.json, results/main_arrays.npz,
results/design_factor.json. Writes figures/Fig1.tif ... Fig6.tif (LZW) and PNG copies."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from sklearn.metrics import roc_curve
from PIL import Image

from tsmbb import generator as G

MM = 1 / 25.4
W = 174 * MM
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
fams = [f.name for f in font_manager.fontManager.ttflist]
FONT = "Arial" if "Arial" in fams else "Liberation Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
                     "savefig.dpi": 600, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.it": f"{FONT}:italic", "mathtext.rm": FONT,
                     "mathtext.bf": f"{FONT}:bold"})
# Okabe-Ito
OI = dict(blue="#0072B2", sky="#56B4E9", green="#009E73", orange="#E69F00", verm="#D55E00",
          purple="#CC79A7", yellow="#F0E442", grey="#7F7F7F", black="#000000")
GCOL = {"HC-T1": "#0072B2", "HC-T2": "#3E8FC4", "HC-T3": "#86BEDF", "MDD-T1": "#7F7F7F",
        "R-T2": "#009E73", "R-T3": "#7FCDB5", "NR-T2": "#D55E00", "NR-T3": "#F0A070"}
POS, NEG = "#2166AC", "#B2182B"
NETS = ["HC-T1", "HC-T2", "HC-T3", "MDD-T1", "R-T2", "NR-T2", "R-T3", "NR-T3"]


def letter(ax, s, x=-0.14, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, name):
    png = os.path.join(OUT, name + ".png")
    fig.savefig(png, dpi=600)
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    wpx = im.size[0]
    dpi = wpx / (174 / 25.4)
    im.save(os.path.join(OUT, name + ".tif"), compression="tiff_lzw", dpi=(600, 600))
    print(name, im.size, f"{im.size[0] / 600 * 25.4:.0f} x {im.size[1] / 600 * 25.4:.0f} mm")


R = json.load(open("results/main.json"))
A = np.load("results/main_arrays.npz", allow_pickle=True)


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(W, 62 * MM), gridspec_kw=dict(width_ratios=[1.25, 1]))
    wk, coh, grp = A["qc_week"], A["qc_cohort"], A["qc_group"]
    rows = [(c, g) for c in ("SC1", "SC2", "SC3") for g in ("MDD", "HC")]
    weeks = np.arange(wk.max() + 1)
    M = np.array([[np.sum((coh == c) & (grp == g) & (wk == w)) for w in weeks] for c, g in rows], float)
    M[:, [5, 11]] = np.nan
    ax = axs[0]
    im = ax.imshow(M, aspect="auto", cmap="Greys", vmin=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{c} {g}" for c, g in rows])
    ax.set_xticks(weeks); ax.set_xticklabels([str(w - 4) for w in weeks])
    ax.set_xlabel("Week (treatment starts after week 0)")
    for x0, lab in [(-0.5, "T1"), (5.5, "T2"), (11.5, "T3")]:
        ax.add_patch(plt.Rectangle((x0, -0.5), 5, len(rows), fill=False, ec=NEG, lw=1))
        ax.text(x0 + 2.5, -0.9, lab, ha="center", va="bottom", color=NEG, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("QC-passed samples")
    letter(ax, "a", -0.2, 1.08)
    ax = axs[1]
    x = np.arange(len(NETS)); w = 0.38
    for off, key, col, lab in [(-w / 2, "precision", "#1F3B5A", "Precision"), (w / 2, "recall", "#9FC3E7", "Recall")]:
        m = [R["subsamples"][n][key]["mean"] for n in NETS]
        lo = [R["subsamples"][n][key]["p2_5"] for n in NETS]; hi = [R["subsamples"][n][key]["p97_5"] for n in NETS]
        ax.bar(x + off, m, w, color=col, label=lab, hatch="" if key == "precision" else "////", ec="white", lw=0)
        ax.errorbar(x + off, m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="none", ecolor="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(NETS, rotation=45, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Edge recovery against planted truth")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    letter(ax, "b", -0.18, 1.08)
    fig.tight_layout(); save(fig, "Fig2")


# ------------------------------------------------------------------ Fig. 3
def fig3():
    names = R["nodes"]; nt = 46
    Ah, Am = A["A_HC_T1"], A["A_MDD_T1"]
    sub_h, sub_m = Ah[:nt, nt:], Am[:nt, nt:]
    rows = np.where((sub_h != 0).any(1) | (sub_m != 0).any(1))[0]
    vmax = np.abs(np.r_[sub_h[rows], sub_m[rows]]).max() * 100
    H = (42 + 3.6 * len(rows)) * MM
    fig = plt.figure(figsize=(W, H))
    bot = 30 * MM / H; top = 1 - 7 * MM / H
    axs = [fig.add_axes([0.205, bot, 0.33, top - bot]), fig.add_axes([0.55, bot, 0.33, top - bot])]
    cax = fig.add_axes([0.905, bot + 0.15 * (top - bot), 0.015, 0.7 * (top - bot)])
    for ax, S, title, l in [(axs[0], sub_h, "Healthy controls, T1", "a"), (axs[1], sub_m, "MDD, T1", "b")]:
        V = S[rows] * 100
        Vm = np.ma.masked_where(V == 0, V)
        ax.imshow(np.zeros_like(V), cmap="Greys", vmin=0, vmax=10, aspect="auto")
        im = ax.imshow(Vm, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i, j in zip(*np.nonzero(V)):
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=0.5))
            ax.text(j, i, f"{V[i, j]:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=4.4,
                    color="white" if abs(V[i, j]) > 0.6 * vmax else "black")
        ax.set_xticks(range(len(names) - nt)); ax.set_xticklabels(names[nt:], rotation=60, ha="right", fontsize=5.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([names[r] for r in rows] if l == "a" else [], style="italic", fontsize=5.8)
        ax.set_title(title); letter(ax, l, -0.62 if l == "a" else -0.04, 1.0)
        ax.tick_params(length=1.5)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Signed conditional TE (10$^{-2}$ bits)")
    save(fig, "Fig3")
    return len(rows)


# ------------------------------------------------------------------ Fig. 4
def fig4():
    rv = R["reversals"]; edges = list(rv["freq"]["R:T1T2"].keys())
    fig = plt.figure(figsize=(W, 120 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1.15], hspace=0.6, wspace=0.6)
    ax = fig.add_subplot(gs[0, :2])
    y = np.arange(len(edges)); h = 0.26
    for k, (key, lab, col, hatch) in enumerate([("HC:T1T2", "HC", "#0072B2", ""), ("R:T1T2", "Responders", "#009E73", "////"),
                                                ("NR:T1T2", "Non-responders", "#D55E00", "....")]):
        v = [rv["freq"][key][e] for e in edges]
        ax.barh(y + (k - 1) * h, v, h, color=col, label=lab, hatch=hatch, ec="white", lw=0)
    ITAL = ("C. comes", "E. lenta", "F. prausnitzii", "Akkermansia", "Bacteroides")

    def elab(e):
        s_, t_ = e.split(" -> ")
        it = lambda x: r"$\it{" + x.replace(" ", r"\ ") + "}$" if x in ITAL else x
        return f"{it(s_)} → {it(t_)} *"
    ax.set_yticks(y); ax.set_yticklabels([elab(e) for e in edges], fontsize=6); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel("Sign-reversal frequency T1→T2 across subsamples")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    letter(ax, "a", -0.47, 1.02)
    ax = fig.add_subplot(gs[0, 2])
    keys = ["HC:T1T2", "HC:T2T3", "R:T1T2", "R:T2T3", "NR:T1T2", "NR:T2T3"]
    sh = np.array([rv["share"][k] for k in keys])
    labs = [k.replace(":T1T2", "\n1→2").replace(":T2T3", "\n2→3") for k in keys]
    ax.bar(range(6), sh[:, 0], color="#9FC3E7", label="stays +")
    ax.bar(range(6), sh[:, 1], bottom=sh[:, 0], color="#1F3B5A", label="stays −")
    ax.bar(range(6), sh[:, 2], bottom=sh[:, 0] + sh[:, 1], color=NEG, label="reverses", hatch="xxxx", ec="white", lw=0)
    ax.set_xticks(range(6)); ax.set_xticklabels(labs, fontsize=5.5); ax.set_ylim(0, 1.2)
    ax.set_ylabel("Fraction of shared edges"); ax.legend(frameon=False, ncol=3, fontsize=5, loc="upper center",
                                                        columnspacing=0.6, handlelength=1)
    letter(ax, "b", -0.32, 1.02)
    ex = json.load(open("results/example_diagrams.json"))
    for k, (net, l) in enumerate([("HC-T1", "c"), ("NR-T3", "d")]):
        ax = fig.add_subplot(gs[1, k]); d = np.array(ex[net]).reshape(-1, 2)
        ax.plot([0, 1.05], [0, 1.05], "k--", lw=0.5)
        ax.scatter(d[:, 0], d[:, 1], s=10, c=GCOL[net], ec="black", lw=0.3, marker="o" if k == 0 else "s")
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.08); ax.set_xlabel("Birth"); ax.set_ylabel("Death")
        ax.set_title(f"H₁ diagram, {net}")
        ax.text(0.04, 0.9, f"{len(d)} H₁ cycles\ntotal persistence {np.sum(d[:, 1] - d[:, 0]):.2f}",
                transform=ax.transAxes, fontsize=5.8, va="top")
        letter(ax, l, -0.3, 1.02)
    ax = fig.add_subplot(gs[1, 2])
    m = [R["subsamples"][n]["h1"]["mean"] for n in NETS]
    lo = [R["subsamples"][n]["h1"]["p2_5"] for n in NETS]; hi = [R["subsamples"][n]["h1"]["p97_5"] for n in NETS]
    ax.bar(range(8), m, color=[GCOL[n] for n in NETS])
    ax.errorbar(range(8), m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="none", ecolor="black", lw=0.6)
    ax.set_xticks(range(8)); ax.set_xticklabels(NETS, rotation=60, ha="right"); ax.set_ylabel("Total H₁ persistence")
    letter(ax, "e", -0.32, 1.02)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.93, bottom=0.12)
    save(fig, "Fig4")


# ------------------------------------------------------------------ Fig. 5
def fig5():
    fig, axs = plt.subplots(1, 3, figsize=(W, 58 * MM))
    f = np.arange(63) / 62
    ls = {"HC": "-", "MDD": "-.", "R": "--", "NR": ":"}
    for n in NETS:
        c = A[f"curve_{n}"].mean(0)
        axs[0].plot(f, c, color=GCOL[n], ls=ls[n.split("-")[0]], label=n)
    axs[0].set_xlabel("Fraction of nodes removed (SPTS order)"); axs[0].set_ylabel("Relative LCC size")
    axs[0].legend(frameon=False, ncol=2, fontsize=5.3); letter(axs[0], "a", -0.25, 1.03)
    ex = json.load(open("results/example_diagrams.json"))
    for n in ("HC-T1", "NR-T3"):
        axs[1].plot(f, A[f"curve_{n}"][0], color=GCOL[n], label=f"{n} targeted")
        axs[1].plot(f, ex[n + "_random_curve"], color=GCOL[n], ls="--", label=f"{n} random")
    axs[1].set_xlabel("Fraction of nodes removed"); axs[1].legend(frameon=False); letter(axs[1], "b", -0.2, 1.03)
    m = [R["subsamples"][n]["auc"]["mean"] for n in NETS]
    lo = [R["subsamples"][n]["auc"]["p2_5"] for n in NETS]; hi = [R["subsamples"][n]["auc"]["p97_5"] for n in NETS]
    axs[2].bar(range(8), m, color=[GCOL[n] for n in NETS])
    axs[2].errorbar(range(8), m, yerr=[np.array(m) - lo, np.array(hi) - m], fmt="none", ecolor="black", lw=0.6)
    axs[2].set_xticks(range(8)); axs[2].set_xticklabels(NETS, rotation=60, ha="right")
    axs[2].set_ylabel("AUCᴿ (targeted removal)"); letter(axs[2], "c", -0.25, 1.03)
    fig.tight_layout(); save(fig, "Fig5")


# ------------------------------------------------------------------ Fig. 6
def fig6():
    cc = R["subject_level"]["case_control"]["roc_scores"]; rp = R["subject_level"]["response"]["roc_scores"]
    fig, axs = plt.subplots(1, 3, figsize=(W, 62 * MM))
    spec_cc = [("network", "TemporalSignedMBB layers", NEG, "-"), ("rf_asv", "RF, ASV CLR", "#1F3B5A", "-"),
               ("fprau", r"$\it{F.\ prausnitzii}$ CLR", "#56B4E9", "--"), ("shannon", "Shannon diversity", OI["grey"], "-."),
               ("static_signed", "Static signed network", OI["green"], ":")]
    spec_rp = [("network", "TemporalSignedMBB T1+T2", NEG, "-"), ("network_T1", "TemporalSignedMBB T1", OI["orange"], "--"),
               ("rf_asv_T1T2", "RF, ASV CLR T1+T2", "#1F3B5A", "-."), ("hamd", "Baseline HAM-D", OI["grey"], ":")]
    cc_auc = R["subject_level"]["case_control"]; rp_auc = R["subject_level"]["response"]
    for ax, spec, src, aucs, title, l in [(axs[0], spec_cc, cc, cc_auc, "Case-control, internal test", "a"),
                                          (axs[1], spec_rp, rp, rp_auc, "Response, pooled held-out", "b")]:
        for k, lab, col, ls_ in spec:
            fpr, tpr, _ = roc_curve(src[k]["y"], src[k]["s"])
            a = aucs[k]["test"]["auc"] if l == "a" else aucs[k]["pooled"]["auc"]
            ax.plot(fpr, tpr, color=col, ls=ls_, label=f"{lab} ({a:.2f})")
        ax.plot([0, 1], [0, 1], color="black", lw=0.4, ls=":")
        ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity"); ax.set_title(title)
        ax.legend(frameon=False, fontsize=5, loc="lower right"); letter(ax, l, -0.22, 1.03)
    df = json.load(open("results/design_factor.json"))
    ax = axs[2]
    for key, lab, col, mk in [("cc_network", "Network, case-control", NEG, "o"),
                              ("cc_abundance", "Abundance, case-control", "#1F3B5A", "o"),
                              ("cc_combined", "Combined, case-control", OI["green"], "o"),
                              ("rp_network", "Network, response", NEG, "s"),
                              ("rp_abundance", "Abundance, response", "#1F3B5A", "s")]:
        vs = sorted(set(d["visits"] for d in df))
        vals = np.array([[d[key]["auc"] for d in df if d["visits"] == v] for v in vs])
        ax.errorbar(vs, vals.mean(1), yerr=[vals.mean(1) - vals.min(1), vals.max(1) - vals.mean(1)],
                    color=col, marker=mk, ms=3, capsize=2, ls="-" if mk == "o" else "--", label=lab, lw=0.8)
    ax.axhline(0.5, color="black", lw=0.4, ls=":")
    ax.set_xticks([5, 10, 20]); ax.set_xlabel("Visits per window"); ax.set_ylabel("AUROC")
    ax.set_title("Design-factor experiment"); ax.legend(frameon=False, fontsize=4.8, loc="center right", bbox_to_anchor=(1.0, 0.72))
    letter(ax, "c", -0.22, 1.03)
    fig.tight_layout(); save(fig, "Fig6")


if __name__ == "__main__":
    import fig1_design            # Fig. 1 is drawn from the simulation model itself
    fig1_design.main(); fig2(); n = fig3(); fig4(); fig5()
    if os.path.exists("results/design_factor.json"):
        fig6()
    json.dump(dict(fig3_taxa_shown=int(n), font=FONT), open("results/figure_info.json", "w"))
