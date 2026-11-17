"""Fig. 1: study design and analysis pipeline, drawn from the actual simulation model (seed 42).
(a) sampling timeline and coupling regime per trajectory; (b) planted signed couplings on the nodes
carrying the planted structure; (c) observation model and preprocessing with the realized counts;
(d) analysis layers with their defining equations and the evaluation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm
import make_figures as MF
from make_figures import W, MM, NEG, POS, letter
MF.OUT = "figures_rev"
save = MF.save
from tsmbb import generator as G

NAVY, GREY, LIGHT = "#1F3B5A", "#6E6E6E", "#F4F6F9"
REG = {"HC": "#BFD7EA", "MDD": "#9E9E9E", "REV": "#E3A15B", "R3": "#7FCDB5", "NR3": "#D55E00"}


def rbox(ax, x, y, w, h, fc="white", ec=NAVY, lw=0.6, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color="#404040", lw=0.7):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=6, shrinkA=0, shrinkB=0))


def main():
    R = json.load(open("results/main_rev.json"))
    pre = R["preprocess"]
    sim_B = G.build_couplings(np.random.default_rng(42))
    H_mm = 192
    fig = plt.figure(figsize=(W, H_mm * MM))

    # ---------------------------------------------------------------- (a) timeline
    ax = fig.add_axes([0.12, 0.835, 0.70, 0.125])
    weeks = np.arange(-4, 13)
    rows = [("HC", ["HC", "HC", "HC"]), ("R", ["MDD", "REV", "R3"]), ("NR", ["MDD", "REV", "NR3"])]
    spans = [(-4.5, 0.5), (1.5, 6.5), (7.5, 12.5)]
    for k, (lab, regs) in enumerate(rows):
        y = 2 - k
        for (a, b), rg in zip(spans, regs):
            ax.add_patch(Rectangle((a, y - 0.32), b - a, 0.64, fc=REG[rg], ec="none"))
        for gap in (1, 7):
            ax.add_patch(Rectangle((gap - 0.5, y - 0.32), 1, 0.64, fc="white", ec="#BBBBBB", lw=0.4, hatch="////"))
        for w_ in weeks:
            if w_ in (1, 7):
                continue
            ax.plot(w_, y, "o", ms=2.4, mfc="black", mec="black", mew=0)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Controls", "Responders", "Non-responders"])
    ax.set_xticks(weeks); ax.set_xticklabels([str(w) for w in weeks], fontsize=5.5)
    ax.set_xlim(-4.6, 12.6); ax.set_ylim(-0.6, 2.95)
    ax.set_xlabel("Week (treatment starts after week 0)")
    for (a, b), lab in zip(spans, ("T1", "T2", "T3")):
        ax.text((a + b) / 2, 2.62, lab, ha="center", va="center", fontweight="bold", fontsize=7)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    leg = [(REG["HC"], r"$\mathbf{B}_{\mathrm{HC}}$ healthy"), (REG["MDD"], r"$\mathbf{B}_{\mathrm{MDD}}$ baseline MDD"),
           (REG["REV"], r"$\mathbf{B}_{\mathrm{rev}}$ 10 signs reversed"), (REG["R3"], r"$\mathbf{B}_{\mathrm{R}}$ 85% recovered"),
           (REG["NR3"], r"$\mathbf{B}_{\mathrm{NR}}$ reversed + loops")]
    lax = fig.add_axes([0.84, 0.835, 0.15, 0.125]); lax.axis("off"); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    for k, (c, t) in enumerate(leg):
        yy = 0.9 - 0.19 * k
        lax.add_patch(Rectangle((0.0, yy - 0.06), 0.14, 0.12, fc=c, ec="none"))
        lax.text(0.2, yy, t, va="center", fontsize=5.6)
    letter(ax, "a", -0.16, 1.05)

    # ---------------------------------------------------------------- (b) planted couplings
    sub = ["F. prausnitzii", "R. gnavus", "A. putredinis", "E. lenta", "Lachno. NK4A136", "C. comes",
           "Bacteroides", "Akkermansia", "Butyrate", "Propionate", "Tryptophan", "5-HT synthesis",
           "IDO1/Kyn", "Quinolinic acid", "Glutamate", "GABA", "Dopamine pathway", "Sec. bile acids",
           "BCAA", "LPS", "IL-6", "TNF-alpha", "Cortisol"]
    idx = [G.IDX[n] for n in sub]
    mats = [("HC", sim_B[("HC", "T1")], None), ("MDD", sim_B[("MDD", "T1")], ("HC", "T1")),
            ("rev", sim_B[("R", "T2")], ("MDD", "T1")), ("R", sim_B[("R", "T3")], ("R", "T2")),
            ("NR", sim_B[("NR", "T3")], ("NR", "T2"))]
    vmax = max(np.abs(np.where(np.eye(G.N_LATENT, dtype=bool), 0, M)).max() for _, M, _ in mats)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    x0, wpan, gap = 0.075, 0.158, 0.012
    for k, (lab, M, prev) in enumerate(mats):
        a2 = fig.add_axes([x0 + k * (wpan + gap), 0.555, wpan, 0.205])
        S = M[np.ix_(idx, idx)].copy(); np.fill_diagonal(S, 0)
        Sm = np.ma.masked_where(S == 0, S)
        a2.imshow(np.zeros_like(S), cmap="Greys", vmin=0, vmax=20)
        im = a2.imshow(Sm, cmap="RdBu", norm=norm, interpolation="nearest")
        if prev is not None:
            P = sim_B[prev][np.ix_(idx, idx)].copy(); np.fill_diagonal(P, 0)
            chg = (np.sign(P) != np.sign(S)) | (np.abs(S - P) > 0.05)
            for i, j in zip(*np.nonzero(chg)):
                a2.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=0.45))
        a2.axhline(7.5, color="#888888", lw=0.3); a2.axvline(7.5, color="#888888", lw=0.3)
        a2.set_xticks([]); a2.set_yticks([])
        for s_ in a2.spines.values():
            s_.set_linewidth(0.4)
        a2.set_title(r"$\mathbf{B}_{\mathrm{" + lab + "}}$", fontsize=7.5, pad=3)
        if k == 0:
            a2.set_yticks([3.5, 15]); a2.set_yticklabels(["taxa", "host"], fontsize=6, rotation=90, va="center")
            a2.tick_params(length=0, pad=1)
            a2.set_ylabel("source", fontsize=6, labelpad=1)
            letter(a2, "b", -0.42, 1.12)
        a2.set_xticks([3.5, 15]); a2.set_xticklabels(["taxa", "host"], fontsize=6); a2.tick_params(length=0, pad=1)
        a2.set_xlabel("target", fontsize=6, labelpad=1)
    cax = fig.add_axes([x0 + 5 * (wpan + gap) - 0.004, 0.575, 0.01, 0.16])
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=5, length=1.5, width=0.4)
    cb.outline.set_linewidth(0.4); cb.set_label("$B_{ij}$", fontsize=6, labelpad=1)
    # ---------------------------------------------------------------- (c) observation and preprocessing
    ax = fig.add_axes([0.0, 0.358, 1.0, 0.153]); ax.set_xlim(0, 174); ax.set_ylim(0, 30); ax.axis("off")
    letter(ax, "c", 0.012, 0.93)
    steps = [("Latent state", [r"$\mathbf{x}_q(t)$, 63 nodes", "VAR(1), Eq. (1)", "stationary start, 20 burn-in"]),
             ("Observation", ["300 ASVs (Dirichlet split)", r"batch $\sigma = 0.50$", "Dirichlet-multinomial"]),
             ("Sampling", [f"{pre['n_scheduled']:,} scheduled visits", f"{pre['n_missed']:,} missed (5%)",
                           "depth: median 38,420"]),
             ("Quality control", [f"{pre['n_qc']:,} samples ≥ 5,000 reads", f"{pre['n_asv_kept']} ASVs kept",
                                  "prevalence ≥ 30%, 2 cohorts"]),
             ("Network nodes", ["genus CLR, 1 coordinate omitted", "cohort correction",
                                "62 nodes: 46 taxa + 16 host"])]
    bw, bh, xs = 30.5, 21, 5
    for k, (t, lines) in enumerate(steps):
        x = xs + k * (bw + 3.4)
        rbox(ax, x, 3, bw, bh, fc=LIGHT)
        ax.text(x + bw / 2, 3 + bh - 3.2, t, ha="center", va="center", fontsize=6.6, fontweight="bold", color=NAVY)
        for j, l in enumerate(lines):
            ax.text(x + bw / 2, 3 + bh - 8.6 - 4.4 * j, l, ha="center", va="center", fontsize=5.7)
        if k < len(steps) - 1:
            arrow(ax, x + bw + 0.3, 13.5, x + bw + 3.1, 13.5)

    # ---------------------------------------------------------------- (d) layers and evaluation
    ax = fig.add_axes([0.0, 0.022, 1.0, 0.33]); ax.set_xlim(0, 174); ax.set_ylim(0, 64.5); ax.axis("off")
    letter(ax, "d", 0.012, 0.965)
    layers = [("Layer 1  Conditional TE", [r"$\mathrm{TE}_{i\to j\mid\mathrm{rest}}=\frac{1}{2}\log_2"
                                            r"\frac{\mathrm{RSS}_j^{\,\setminus i}}{\mathrm{RSS}_j}$",
                                            r"cluster-robust test, BH $q<0.05$",
                                            r"$A_{ij}=\mathrm{sign}(b_{ij})\,\mathrm{TE}_{i\to j}$"], "Eqs. (2) to (4)"),
              ("Layer 2  TESTM", [r"$s_{ij}^{(t)}\in\{-1,0,+1\}$",
                                  r"reversal: $s_{ij}^{(t)}s_{ij}^{(t')}=-1$",
                                  r"node rate $r_i$"], "Eq. (5)"),
              ("Layer 3  Persistent homology", [r"$d_{ij}=1-\max(|A_{ij}|,|A_{ji}|)/\max|A|$",
                                                r"Vietoris-Rips, H$_0$ and H$_1$",
                                                r"$W_2$; node load $\varphi_i$"], "Eqs. (6) to (8)"),
              ("Layer 4  SPTS", [r"$\mathbf{L}_s=\overline{\mathbf{D}}-\mathbf{A}$",
                                 r"$\mathrm{PS}_i=\|\mathbf{L}_s^{+}\mathbf{e}_i\|_2$",
                                 r"$\sum_t\mathrm{PS}_i^{(t)}+\lambda r_i+\gamma\sum_t\varphi_i^{(t)}$"],
               "Eqs. (9) and (10)")]
    lw_, lh, ly = 38.5, 27, 33
    for k, (t, lines, eq) in enumerate(layers):
        x = 5 + k * (lw_ + 3.0)
        rbox(ax, x, ly, lw_, lh, fc="white")
        ax.add_patch(FancyBboxPatch((x, ly + lh - 6), lw_, 6, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=NAVY, ec=NAVY, lw=0.6))
        ax.add_patch(Rectangle((x, ly + lh - 6), lw_, 2, fc=NAVY, ec="none"))
        ax.text(x + lw_ / 2, ly + lh - 3, t, ha="center", va="center", fontsize=6.4, color="white", fontweight="bold")
        for j, l in enumerate(lines):
            ax.text(x + lw_ / 2, ly + lh - 10.8 - 5.0 * j, l, ha="center", va="center", fontsize=6.0)
        ax.text(x + lw_ - 1.2, ly + 1.4, eq, ha="right", va="bottom", fontsize=5.2, color=GREY)
        if k < 3:
            arrow(ax, x + lw_ + 0.2, ly + lh / 2, x + lw_ + 2.8, ly + lh / 2)
    ax.text(5, 61.7, "applied to every group-by-window network (within-subject transitions, subject means removed)",
            fontsize=5.6, color=GREY, va="bottom")
    # evaluation boxes
    ev = [("Group-level evaluation against the planted truth",
           ["30 independent replications; null and 10 further scenarios",
            "edges, signs, calibration; complete transition truth",
            "loops (oracle, density null, directed cycles); SPTS vs perturbations"], 5, 80.5),
          ("Subject-level evaluation",
           ["LIONESS vs heterogeneous subject-specific truth",
            "matched learners; nested feature construction; oracle",
            "6 to 8 replications per scenario; 5, 10, 20 visits × 2 couplings"], 88.5, 80.5)]
    for t, lines, x, w_ in ev:
        rbox(ax, x, 2, w_, 24, fc=LIGHT)
        ax.text(x + w_ / 2, 22.6, t, ha="center", va="center", fontsize=6.6, fontweight="bold", color=NAVY)
        for j, l in enumerate(lines):
            ax.text(x + w_ / 2, 16.6 - 5.0 * j, l, ha="center", va="center", fontsize=5.8)
    for k in range(4):
        xc = 5 + k * (lw_ + 3.0) + lw_ / 2
        ax.plot([xc, xc], [ly, ly - 2.5], color="#404040", lw=0.7)
    ax.plot([5 + lw_ / 2, 5 + 3 * (lw_ + 3) + lw_ / 2], [ly - 2.5, ly - 2.5], color="#404040", lw=0.7)
    arrow(ax, 45, ly - 2.5, 45, 26.2); arrow(ax, 129, ly - 2.5, 129, 26.2)
    save(fig, "Fig1")


if __name__ == "__main__":
    main()
