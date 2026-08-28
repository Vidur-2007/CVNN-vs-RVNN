import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_gen import CLASSES

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(HERE, "results", "all_results.json")) as f:
    R = json.load(f)
with open(os.path.join(HERE, "results", "analysis.json")) as f:
    A = json.load(f)

MODELS = ["RVNN_mag", "RVNN_ri", "CVNN"]
COLORS = {"RVNN_mag": "#888888", "RVNN_ri": "#4C78A8", "CVNN": "#E45756"}
SNR_ORDER = ["clean", "15dB", "5dB", "-5dB"]
SNR_X = [20, 15, 5, -5]  # "clean" placed at x=20 for plotting

# --- Fig 1: accuracy vs SNR ---
plt.figure(figsize=(6, 4.2))
for m in MODELS:
    means = [A["summary"][s][m]["mean"] * 100 for s in SNR_ORDER]
    stds = [A["summary"][s][m]["std"] * 100 for s in SNR_ORDER]
    plt.errorbar(SNR_X, means, yerr=stds, marker="o", label=m, color=COLORS[m], capsize=3)
plt.xlabel("SNR (dB)  [20 = clean/no noise]")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs. noise level (mean ± std over 5 seeds)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "acc_vs_snr.png"), dpi=150)
plt.close()

# --- Fig 2: training curves at -5dB (seed 0) ---
plt.figure(figsize=(6, 4.2))
for m in MODELS:
    rs = [r for r in R if r["model"] == m and r["snr_db"] == -5 and r["seed"] == 0]
    curve = rs[0]["curve"]
    xs = [c["epoch"] for c in curve]
    ys = [c["test_acc"] * 100 for c in curve]
    plt.plot(xs, ys, label=m, color=COLORS[m])
plt.xlabel("Epoch")
plt.ylabel("Test accuracy (%)")
plt.title("Training curves at SNR = -5 dB (seed 0)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "train_curves.png"), dpi=150)
plt.close()

# --- Fig 3: confusion matrices at -5dB ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, m in zip(axes, MODELS):
    cm = np.array(A["confusion_minus5dB"][m], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6)); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_title(m)
    for i in range(6):
        for j in range(6):
            if cmn[i, j] > 0.01:
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                         fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
fig.suptitle("Confusion matrices at SNR = -5 dB (row-normalized, summed over 5 seeds)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "confusion_-5dB.png"), dpi=150)
plt.close()

# --- Fig 4: constant vs phase_step subproblem accuracy vs SNR ---
plt.figure(figsize=(6, 4.2))
for m in MODELS:
    ys = [A["constant_vs_phasestep_subacc"][s][m] * 100 for s in SNR_ORDER]
    plt.plot(SNR_X, ys, marker="o", label=m, color=COLORS[m])
plt.axhline(50, color="gray", linestyle="--", linewidth=1, label="chance (2-class)")
plt.xlabel("SNR (dB)  [20 = clean/no noise]")
plt.ylabel("Accuracy on constant-vs-phase_step subproblem (%)")
plt.title("The phase-only distinction: constant vs. phase_step")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "phase_subproblem.png"), dpi=150)
plt.close()

print("figures written to", FIG)
print(os.listdir(FIG))
