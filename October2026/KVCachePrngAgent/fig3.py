import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
d = json.load(open("explore2.json")); c = d["curve"]; r = d["read_error"]
NAVY, ORANGE, GREEN, PLUM = "#34507a", "#b5652a", "#3f8a6a", "#7a4f7a"
fig, ax = plt.subplots(1, 2, figsize=(9.7, 3.9), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
x = np.array(c["delta"])
a = ax[0]
a.plot(x, 1-np.array(c["p16"]), color=GREEN, lw=2, label="One visitor, 16 bins")
a.plot(x, 1-np.array(c["p48"]), color=ORANGE, lw=2, label="One visitor, 48 bins")
a.plot(x, 1-np.array(c["p128"]), color=PLUM, lw=2, ls="--", label="One visitor, 128 bins")
a.plot(x, 1-np.array(c["colony3"]), color=NAVY, lw=2.4, label="Colony, 3 seeds x 16 bins")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("Key separation  ||x - y||  (keys ~ N(0, I$_{16}$))")
a.set_ylabel("Probability the pair is separated")
a.set_title("(a) Separation: diversity wins", fontsize=10)
a.grid(alpha=.3); a.legend(frameon=False, fontsize=8.5, loc="lower right")
b = ax[1]
labels = ["1 x 16", "mean of\n3 x 16", "1 x 48", "learned\n1 x 16"]
keys = ["1x16", "mean3x16", "1x48", "learned16"]
w = 0.38; idx = np.arange(4)
for j, (kind, col, nm) in enumerate([("isotropic", NAVY, "Isotropic keys"), ("subspace", ORANGE, "Keys on a 4-D subspace")]):
    m = [r[f"{kind}_beta1.0"][k][0] for k in keys]; se = [r[f"{kind}_beta1.0"][k][1] for k in keys]
    b.bar(idx + (j-.5)*w, m, w, yerr=se, color=col, label=nm, capsize=2)
b.set_xticks(idx); b.set_xticklabels(labels, fontsize=8.5)
b.set_ylabel("Relative read error vs. token attention")
b.set_title("(b) Read error at matched state: detail wins", fontsize=10)
b.grid(alpha=.3, axis="y"); b.legend(frameon=False, fontsize=8.5); b.set_ylim(0, 1)
fig.tight_layout(); fig.savefig("image3.png")
