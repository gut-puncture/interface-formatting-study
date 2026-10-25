"""Paper figures, from results/results.json.

    python -m keyhole.figures --results results --out paper/figures

Every figure is drawn at its printed size (one column = 3.3 in, full page = 6.95 in), in the paper's
serif type, with axes placed explicitly so that titles, labels and legends never collide.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from keyhole import config as C
from keyhole import style

NAMES = {"q4": "Qwen3-4B", "olmo": "OLMo-3-7B", "q8": "Qwen3-8B", "g12": "Gemma-4-12B"}
FAM = {"json": "Tool result", "html": "Web page", "log": "Server log", "email": "Email thread",
       "prose": "Plain prose"}
# wrappers (Figure 2)
FCOL = {"json": "#2471a3", "html": "#1e8449", "log": "#c0392b", "email": "#7d3c98", "prose": "#d68910"}
# what replaces a lost answer (Figure 3): a palette not used for wrappers or patches
TYPES = {"wrong": ("Wrong answer", "#8c510a"), "scope": ("Not there", "#35978f"),
         "task": ("About the wrapper", "#c51b7d")}
# patches (Figures 4 and 5)
PATCH = {"A": ("Own question restored", "#b2182b", "-"),
         "D": ("Other question swapped in", "#222222", "-"),
         "B": ("Last position restored", "#2166ac", "--")}
REF = "#8a8a8a"
DEPTH = "Depth"


def axes_grid(fig, rows, cols, left, right, bottom, top, wgap, hgap, **kw):
    """Axes at exact positions (fractions of the figure), so nothing depends on automatic layout."""
    w = (right - left - wgap * (cols - 1)) / cols
    h = (top - bottom - hgap * (rows - 1)) / rows
    return [[fig.add_axes([left + c * (w + wgap), top - (r + 1) * h - r * hgap, w, h], **kw) for c in range(cols)]
            for r in range(rows)]


def band(ax, x, rows, color, lw=1.6, ls="-", alpha=0.15, marker=None):
    x = np.asarray(x, float)
    m = np.array([r["mean"] if r and r.get("mean") is not None else np.nan for r in rows])
    lo = np.array([r["lo"] if r else np.nan for r in rows])
    hi = np.array([r["hi"] if r else np.nan for r in rows])
    if alpha:
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, marker=marker, ms=2.8)


def models_in(res, need=None):
    return [k for k in C.MODEL_ORDER if k in res["models"] and (need is None or need in res["models"][k])]


def panel_row(n, height, bottom, top=0.9):
    fig = plt.figure(figsize=(style.PAGEW, height))
    return fig, axes_grid(fig, 1, n, 0.07, 0.995, bottom, top, 0.035, 0)[0]


def behaviour(res, out):
    """Share of known answers still given, by prompt length, per wrapper and model."""
    keys = models_in(res)
    fig, axs = panel_row(len(keys), 2.05, 0.33)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        b = res["models"][k]["behaviour"]
        ax.axhline(b["floor"]["mean"], color=REF, lw=0.9, ls=":")
        for f in C.FAMILIES:
            rows = [b["conditions"][f"{f}_{d}"] for d in C.DOSES]
            band(ax, [r["tokens"] for r in rows], [r["retained"] for r in rows], FCOL[f], lw=1.5, marker="o")
        ax.set_xscale("log", base=2)
        med = [np.median([b["conditions"][f"{f}_{d}"]["tokens"] for f in C.FAMILIES]) for d in C.DOSES]
        ax.set_xticks(med, [f"{round(med[0], -2):.0f}", f"{med[1] / 1000:.0f}k", f"{med[2] / 1000:.0f}k"])
        ax.minorticks_off()
        ax.set_xlim(med[0] / 1.5, med[2] * 1.5)
        ax.set_ylim(0.25, 1.02)
        ax.set_xlabel("Prompt tokens")
        if i == 0:
            ax.set_ylabel("Known answers kept")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), NAMES[k])
    handles = [Line2D([], [], color=FCOL[f], marker="o", ms=2.8, lw=1.5, label=FAM[f]) for f in C.FAMILIES]
    handles.append(Line2D([], [], color=REF, lw=0.9, ls=":", label="Noise floor"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=6, columnspacing=1.2,
               handlelength=1.8, handletextpad=0.4)
    style.save(fig, out / "fig_behaviour.pdf")


def confidence(res, out):
    """What wrapped prompts produce, by how sure the model is of the answer without a wrapper."""
    p = res["pooled"]
    labels = ["0.5–0.8", "0.8–0.95", "0.95–0.99", "0.99–0.999", "> 0.999"]
    fig = plt.figure(figsize=(style.PAGEW, 2.15))
    axs = axes_grid(fig, 1, 2, 0.07, 0.995, 0.33, 0.9, 0.06, 0)[0]
    x = np.arange(len(labels))
    top = max(r["hi"] for g in ("log", "other") for t in TYPES for r in p["type_by_confidence_group"][g][t])
    for i, (ax, g, title) in enumerate(zip(axs, ("log", "other"), ("Server logs", "Other wrappers"))):
        tb = p["type_by_confidence_group"][g]
        band(ax, x, p["null_lost_by_confidence"], REF, lw=0.9, ls=":", alpha=0.0)
        for t in ("wrong", "scope", "task"):
            band(ax, x, tb[t], TYPES[t][1], lw=1.5, marker="o")
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.2, len(labels) - 0.8)
        ax.set_ylim(0, top * 1.05)
        ax.set_xlabel("Probability of the answer without a wrapper")
        if i == 0:
            ax.set_ylabel("Share of wrapped prompts")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), title)
    handles = [Line2D([], [], color=TYPES[t][1], marker="o", ms=2.8, lw=1.5, label=TYPES[t][0])
               for t in ("wrong", "scope", "task")]
    handles.append(Line2D([], [], color=REF, lw=0.9, ls=":", label="Lost with one neutral sentence"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=4, columnspacing=1.6,
               handlelength=1.8, handletextpad=0.4)
    style.save(fig, out / "fig_confidence.pdf")


def swap(res, out):
    """Main figure: what a failed wrapped prompt answers after one patch, by the depth of the patch."""
    keys = models_in(res, "patching")
    fig, axs = panel_row(len(keys), 2.3, 0.34)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        t = res["models"][k]["patching"]["top1"]
        ax.axhline(t["donor_top1_in_wrapper"]["mean"], color=PATCH["D"][1], lw=0.9, ls=":")
        ax.axhline(t["BASE_own"]["mean"], color=REF, lw=0.9, ls=":")
        for s, key in (("D", "D_donor"), ("B", "B_own"), ("A", "A_own")):
            rows = t[key]
            band(ax, [r["depth"] for r in rows], rows, PATCH[s][1], ls=PATCH[s][2], lw=1.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(DEPTH)
        if i == 0:
            ax.set_ylabel("Predicts the target answer")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), NAMES[k])
    handles = [Line2D([], [], color=PATCH[s][1], ls=PATCH[s][2], lw=1.6, label=PATCH[s][0]) for s in "ADB"]
    handles += [Line2D([], [], color=PATCH["D"][1], lw=0.9, ls=":", label="Other question, own wrapped prompt"),
                Line2D([], [], color=REF, lw=0.9, ls=":", label="No patch")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=5, columnspacing=1.3,
               handlelength=2.0, handletextpad=0.4)
    style.save(fig, out / "fig_swap.pdf")


def repair(res, out):
    """Appendix: share of the lost answer's log-probability restored, by depth."""
    keys = models_in(res, "patching")
    fig, axs = panel_row(len(keys), 2.2, 0.32)
    for i, (ax, k) in enumerate(zip(axs, keys)):
        p = res["models"][k]["patching"]
        ax.axhline(0, color=REF, lw=0.7, ls=":")
        for s in ("D", "B", "A"):
            rows = p["curves"][s]
            band(ax, [r["depth"] for r in rows], rows, PATCH[s][1], ls=PATCH[s][2], lw=1.6)
        ax.plot([p["depth_star"]] * 2, [-1.05, -0.93], color=style.INK, lw=2.2, solid_capstyle="butt")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.05, 1.1)
        ax.set_xlabel(DEPTH)
        if i == 0:
            ax.set_ylabel("Share of lost answer restored")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), NAMES[k])
    handles = [Line2D([], [], color=PATCH[s][1], ls=PATCH[s][2], lw=1.6, label=PATCH[s][0]) for s in "ADB"]
    handles.append(Line2D([], [], color=style.INK, lw=2.2, label="Layer chosen on development items"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=4, columnspacing=1.6,
               handlelength=2.0)
    style.save(fig, out / "fig_repair.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    a = ap.parse_args()
    style.setup()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.load(open(Path(a.results) / "results.json"))
    behaviour(res, out)
    confidence(res, out)
    if models_in(res, "patching"):
        swap(res, out)
        repair(res, out)


if __name__ == "__main__":
    main()
