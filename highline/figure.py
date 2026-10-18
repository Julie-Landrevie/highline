"""Figure 1 : bilan (gain - cout) d'un bloc plus haut selon la situation, et contre-pressing contre repli."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import decision as dec
from . import i18n
from .stats import pct


def make(cell_list, lev, path, lang="fr"):
    F = i18n.FIG[lang]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    cs = dec.one_dim(cell_list)
    for i, c in enumerate(cs):
        for j, (w, col, lab) in enumerate(((1, "tab:blue", F["w1"]), (2, "tab:orange", F["w2"]))):
            lo, hi = pct(c["draws"][:, 1] - w * c["draws"][:, 0])
            y0 = i + (j - 0.5) * 0.28
            ax[0].plot([lo, hi], [y0, y0], color=col, lw=2)
            ax[0].plot(c["point"][1] - w * c["point"][0], y0, "o", color=col, label=lab if i == 0 else None)
    ax[0].axvline(0, color="grey", lw=0.8)
    ax[0].set_yticks(range(len(cs)))
    ax[0].set_yticklabels([i18n.label(c["key"], dec.DIM_ORDER, lang) for c in cs], fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlabel(F["x0"], fontsize=8)
    ax[0].set_title(F["t0"], fontsize=10)
    ax[0].legend(fontsize=8)
    if lev:
        kk = [k for k in lev["strata"] if lev["point"].get(k) is not None]
        for i, k in enumerate(kk):
            lo, hi = pct(lev["boots"][k])
            col = {"pressing": "tab:green", "repli": "tab:purple", None: "grey"}[lev["pref"][k]]
            ax[1].plot([lo, hi], [i, i], color=col, lw=2)
            ax[1].plot(lev["point"][k]["D"], i, "o", color=col)
        ax[1].axvline(0, color="grey", lw=0.8)
        ax[1].set_yticks(range(len(kk)))
        ax[1].set_yticklabels([i18n.DIMN[lang]["ensemble"] if k == "ensemble" else f"{i18n.DIMN[lang][k[0]]} : {i18n.DIMN[lang][k[1]]}" for k in kk], fontsize=8)
        ax[1].invert_yaxis()
        ax[1].set_xlabel(F["x1"], fontsize=8)
        ax[1].set_title(F["t1"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
