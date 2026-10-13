"""Redraw the figures from results.json without rerunning the experiments."""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import synthetic_experiments as se

se.plot_style()
with open(os.path.join(here, "results.json"), encoding="utf-8") as f:
    res = json.load(f)
se.plot_radius(res["radius"], os.path.join(se.OUT, "exp_radius.pdf"))
se.plot_sharpness({float(k): v for k, v in res["sharpness"].items()}, os.path.join(se.OUT, "exp_sharp.pdf"))
out = {int(k): v for k, v in res["noise"].items()}
acc_true = {int(k): v for k, v in res["acc_true"].items()}
se.plot_noise(out, acc_true, os.path.join(se.OUT, "exp_noise.pdf"))
print("figures updated")
