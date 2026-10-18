# -*- coding: utf-8 -*-
"""Figure for the EEG check (bci2a_check_results.json), English labels, same style as the other figures:
(a) accuracy of the four methods for each subject with the full set of channels (five-fold cross-validation within the subject);
(b) error of the covariance estimate from the first T' samples of the window, the quantity sqrt(h(h+1)/T') and the bound
    of the theorem at alpha = 0.05, for the full set of channels and for the subset of central channels;
    means over the nine subjects, error bars are standard deviations across subjects.
Run: python plot_eeg_en.py [results.json] [out.pdf]; defaults: results next to the script, output ../../springer/figs/eeg_check.pdf"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import synthetic_experiments as se

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bci2a_check_results.json")
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "..", "springer", "figs", "eeg_check.pdf")
with open(src, encoding="utf-8") as f:
    res = json.load(f)
subjects = sorted(res, key=int)

se.plot_style()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.4))

# (a) accuracy per subject, full set of channels: bars, since close values of the methods would merge as points
names = [("tsm", "TSM"), ("mdm", "MDM"), ("logeuc", "log-Euclidean"), ("euc", "Euclidean")]
x = np.arange(1, len(subjects) + 1); w = 0.2
for k, (key, lab) in enumerate(names):
    y = [res[s]["h22"]["acc"][key] for s in subjects]
    a1.bar(x + (k - 1.5) * w, y, width=w, color="C%d" % k, label=lab)
a1.set_xticks(x); a1.set_xticklabels([str(int(s)) for s in subjects]); a1.grid(axis="x", visible=False)
a1.set_xlabel("subject"); a1.set_ylabel("accuracy"); a1.set_title("(a)")
a1.set_ylim(0.5, 1.0)
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)

# (b) noise of the covariance estimate, means over subjects: colour encodes h, line style encodes the quantity
for k, tag in enumerate(("h22", "h9")):
    col = "C%d" % k
    T = np.array([r["T"] for r in res[subjects[0]][tag]["noise"]])
    err = np.array([[r["err"] for r in res[s][tag]["noise"]] for s in subjects])
    iid = np.array([r["iid"] for r in res[subjects[0]][tag]["noise"]])
    bound = np.array([r["bound"] for r in res[subjects[0]][tag]["noise"]], dtype=float)
    a2.errorbar(T, err.mean(0), yerr=err.std(0), color=col, marker="o", capsize=2)
    a2.plot(T, iid, ":", color=col)
    ok = np.isfinite(bound)
    a2.plot(T[ok], bound[ok], "--", color=col)
a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel(r"number of samples $\mathcal{T}'$"); a2.set_title("(b)")
a2.set_ylabel(r"$\delta_{\mathcal{R}}(\hat{\mathbf{R}}_i, \mathbf{R}_i)$")
hs2 = [Line2D([0], [0], color="C%d" % k, marker="o") for k in range(2)]
ls2 = [r"$h = %d$" % res[subjects[0]][tag]["h"] for tag in ("h22", "h9")]
hs2 += [Line2D([0], [0], color="gray", marker="o"), Line2D([0], [0], ls=":", color="gray"), Line2D([0], [0], ls="--", color="gray")]
ls2 += ["error", r"$\sqrt{h(h+1)/\mathcal{T}'}$", r"theorem bound, $\alpha = 0.05$"]
a2.legend(hs2, ls2, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3, frameon=False, columnspacing=1.0, handlelength=1.6)
fig.tight_layout(w_pad=1.5)
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
se.savefig(fig, dst); plt.close(fig)
print("written", os.path.abspath(dst))
