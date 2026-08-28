import json
import os
import numpy as np
from scipy import stats

from data_gen import CLASSES

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "results", "all_results.json")) as f:
    R = json.load(f)

MODELS = ["RVNN_mag", "RVNN_ri", "CVNN"]
SNRS = [None, 15, 5, -5]


def snr_label(s):
    return "clean" if s is None else f"{s}dB"


def get(model, snr):
    return [r for r in R if r["model"] == model and r["snr_db"] == snr]


summary = {}
print(f"{'SNR':>8} | {'RVNN_mag':>18} | {'RVNN_ri':>18} | {'CVNN':>18}")
print("-" * 72)
for snr in SNRS:
    row = {}
    accs = {}
    for m in MODELS:
        rs = get(m, snr)
        a = np.array([r["test_acc"] for r in rs])
        accs[m] = a
        row[m] = (a.mean(), a.std())
    print(f"{snr_label(snr):>8} | {row['RVNN_mag'][0]*100:6.2f} +/- {row['RVNN_mag'][1]*100:4.2f}%   "
          f"| {row['RVNN_ri'][0]*100:6.2f} +/- {row['RVNN_ri'][1]*100:4.2f}%   "
          f"| {row['CVNN'][0]*100:6.2f} +/- {row['CVNN'][1]*100:4.2f}%")
    summary[snr_label(snr)] = {m: {"mean": float(row[m][0]), "std": float(row[m][1]),
                                    "vals": accs[m].tolist()} for m in MODELS}

print("\nPaired t-tests (CVNN vs baseline), per SNR, across matched seeds:")
sig = {}
for snr in SNRS:
    cvnn = np.array([r["test_acc"] for r in get("CVNN", snr)])
    sig[snr_label(snr)] = {}
    for base in ["RVNN_mag", "RVNN_ri"]:
        b = np.array([r["test_acc"] for r in get(base, snr)])
        t, p = stats.ttest_rel(cvnn, b)
        diff = (cvnn.mean() - b.mean()) * 100
        print(f"  {snr_label(snr):>6}  CVNN - {base:>8}: Δ={diff:+.2f}pp  t={t:.2f}  p={p:.4f}")
        sig[snr_label(snr)][base] = {"delta_pp": float(diff), "t": float(t), "p": float(p)}

# aggregate confusion matrices (sum over seeds) per model at the hardest SNR (-5dB)
print("\nConfusion matrices at SNR=-5dB (summed over 5 seeds):")
conf = {}
for m in MODELS:
    rs = get(m, -5)
    cm = np.sum([np.array(r["confusion"]) for r in rs], axis=0)
    conf[m] = cm.tolist()
    print(f"\n{m}:")
    print("        " + " ".join(f"{c[:6]:>7}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>8}" + " ".join(f"{v:7d}" for v in row))

# specifically the constant vs phase_step pair (classes 0 and 5) -- the
# pair engineered to be indistinguishable from magnitude alone
print("\nconstant<->phase_step confusion rate (classes 0 & 5), all SNRs:")
cvnn_advantage_04 = {}
for snr in SNRS:
    row = {}
    for m in MODELS:
        rs = get(m, snr)
        cm = np.sum([np.array(r["confusion"]) for r in rs], axis=0)
        # error rate within the 2-class subproblem {0,5}
        sub = cm[np.ix_([0, 5], [0, 5])]
        acc_sub = np.trace(sub) / sub.sum()
        row[m] = float(acc_sub)
    cvnn_advantage_04[snr_label(snr)] = row
    print(f"  {snr_label(snr):>6}: " + "  ".join(f"{m}={row[m]*100:.1f}%" for m in MODELS))

out = {
    "summary": summary,
    "significance": sig,
    "confusion_minus5dB": conf,
    "constant_vs_phasestep_subacc": cvnn_advantage_04,
    "classes": CLASSES,
}
with open(os.path.join(HERE, "results", "analysis.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote results/analysis.json")
