"""Fig. 7 (neural-data application) at 600 dpi from results/neural.json and neural_arrays.npz."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from make_figures import W, MM, NEG, letter, save

R = json.load(open("results/neural.json")); A = np.load("results/neural_arrays.npz")
ns, ee = R["netsim"], R["eeg"]
G, Rc = A["G"], A["R_cte"]
N = G.shape[0]

fig = plt.figure(figsize=(W, 128 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.3, 1], hspace=0.6, wspace=0.5)
ax = fig.add_subplot(gs[0, 0])
cat = np.zeros((N, N), int)
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if G[i, j] and Rc[i, j] and not Rc[j, i]:
            cat[i, j] = 1
        elif G[i, j] and Rc[i, j] and Rc[j, i]:
            cat[i, j] = 2
        elif G[j, i] and Rc[i, j] and not G[i, j]:
            cat[i, j] = 0 if Rc[j, i] else 3       # reversed-only would be 3
        elif Rc[i, j] and not (G[i, j] or G[j, i]):
            cat[i, j] = 4
cmap = ListedColormap(["#F2F2F2", "#1F5FA8", "#92C5DE", "#E69F00", NEG])
ax.imshow(cat, cmap=cmap, vmin=0, vmax=4)
ax.set_xticks(range(N)); ax.set_xticklabels(range(1, N + 1), fontsize=4.5)
ax.set_yticks(range(N)); ax.set_yticklabels(range(1, N + 1), fontsize=4.5)
ax.set_xlabel("Target node"); ax.set_ylabel("Source node"); ax.set_title("NetSim: pooled conditional TE")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#1F5FA8", label="correct direction"), Patch(color="#92C5DE", label="both directions"),
                   Patch(color=NEG, label="false positive")], frameon=False, fontsize=5, loc="upper center",
          bbox_to_anchor=(0.5, -0.25), ncol=1)
letter(ax, "a", -0.3, 1.03)
ax = fig.add_subplot(gs[0, 1:])
labs = [("cTE_k1", "Conditional TE\n(k = 1)"), ("cTE_kBIC", f"Conditional TE\n(k = {ns['k_bic']}, BIC)"),
        ("bTE_k1", "Bivariate TE"), ("PCMCI", "PCMCI\n(ParCorr)"), ("parcorr", "Partial corr.\n(lag 0)")]
x = np.arange(len(labs)); w = 0.36
for off, key, col, lab, hatch in [(-w / 2, "f1", "#1F5FA8", "Directed F1", ""), (w / 2, "und_f1", "#92C5DE", "Undirected F1", "////")]:
    v, e = [], []
    for k, _ in labs:
        s = ns[k]["single"].get(key)
        v.append(s["mean"] if s else np.nan); e.append(s["mcse"] if s else 0)
    ax.bar(x + off, v, w, yerr=e, color=col, label=lab, capsize=2, hatch=hatch, ec="white", lw=0,
           error_kw=dict(lw=0.6))
ax.set_xticks(x); ax.set_xticklabels([l for _, l in labs], fontsize=5.8); ax.set_ylim(0, 1)
ax.set_ylabel("F1 (mean over 50 subjects)"); ax.set_title("NetSim: single-subject recovery")
ax.legend(frameon=False); letter(ax, "b", -0.12, 1.03)
ax = fig.add_subplot(gs[1, 0])
rn = A["rev_null"]; ax.hist(rn, bins=np.arange(-0.5, max(rn.max(), ee["reversals"]) + 1.5), color="#BBBBBB", ec="white")
ax.axvline(ee["reversals"], color=NEG, lw=1.2); ax.set_xlabel("Sign reversals, pre → post")
ax.set_ylabel("Permutations"); ax.set_title("EEG: sign reversals"); letter(ax, "c", -0.3, 1.03)
ax = fig.add_subplot(gs[1, 1])
ax.hist(A["chg_null"], bins=25, color="#BBBBBB", ec="white"); ax.axvline(ee["changed"], color=NEG, lw=1.2)
ax.set_xlabel("Edges present in only one window"); ax.set_ylabel("Permutations"); ax.set_title("EEG: edge-set change")
letter(ax, "d", -0.2, 1.03)
ax = fig.add_subplot(gs[1, 2])
for i, k in enumerate(("H1_pre", "H1_post")):
    h = ee[k]
    ax.errorbar(i, h["null_mean"], yerr=1.96 * h["null_sd"], fmt="o", color="grey", capsize=3, ms=4,
                label="Density-matched null" if i == 0 else None)
    ax.plot(i, h["obs"], "D", color=NEG, ms=4, label="Observed" if i == 0 else None)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Pre", "Post"]); ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Total H₁ persistence"); ax.set_title("EEG: H₁ vs density null")
ax.legend(frameon=False, fontsize=5.5); letter(ax, "e", -0.3, 1.03)
fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.1)
save(fig, "Fig7")
