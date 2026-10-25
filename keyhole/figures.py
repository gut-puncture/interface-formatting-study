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
MCOL = {"q4": "#2471a3", "olmo": "#c0392b", "q8": "#1e8449", "g12": "#d68910"}
FAM = {"json": "Tool result", "html": "Web page", "log": "Server log", "email": "Email thread",
       "prose": "Plain prose"}
FCOL = {"json": "#2471a3", "html": "#1e8449", "log": "#c0392b", "email": "#7d3c98", "prose": "#8a8a8a"}
SITE = {"A": ("Own question, plain prompt", "#b2182b", "-"),
        "D": ("Other question, plain prompt", "#8a8a8a", "-"),
        "B": ("Last position, plain prompt", "#2471a3", "--")}
DEPTH = "Depth"


def axes_grid(fig, rows, cols, left, right, bottom, top, wgap, hgap, **kw):
    """Axes at exact positions (fractions of the figure), so nothing depends on automatic layout."""
    w = (right - left - wgap * (cols - 1)) / cols
    h = (top - bottom - hgap * (rows - 1)) / rows
    return [[fig.add_axes([left + c * (w + wgap), top - (r + 1) * h - r * hgap, w, h], **kw) for c in range(cols)]
            for r in range(rows)]


def band(ax, x, rows, color, lw=1.6, ls="-", label=None, alpha=0.15):
    x = np.asarray(x, float)
    m = np.array([r["mean"] if r and r.get("mean") is not None else np.nan for r in rows])
    lo = np.array([r["lo"] if r else np.nan for r in rows])
    hi = np.array([r["hi"] if r else np.nan for r in rows])
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, m, color=color, lw=lw, ls=ls, label=label)


def models_in(res):
    return [k for k in C.MODEL_ORDER if k in res["models"]]


def behaviour(res, out):
    """Figure: share of known answers still given, by prompt length, per wrapper family and model."""
    keys = models_in(res)
    fig = plt.figure(figsize=(style.PAGEW, 2.05))
    axs = axes_grid(fig, 1, len(keys), 0.07, 0.995, 0.33, 0.9, 0.035, 0)[0]
    for i, (ax, k) in enumerate(zip(axs, keys)):
        b = res["models"][k]["behaviour"]
        fl = b["floor"]
        ax.axhspan(fl["lo"], fl["hi"], color="#bdbdbd", alpha=0.45, linewidth=0)
        ax.axhline(fl["mean"], color="#7a7a7a", lw=0.9, ls=":")
        for f in C.FAMILIES:
            rows = [b["conditions"][f"{f}_{d}"] for d in C.DOSES]
            x = [r["tokens"] for r in rows]
            band(ax, x, [r["retained"] for r in rows], FCOL[f], lw=1.5)
            ax.plot(x, [r["retained"]["mean"] for r in rows], "o", color=FCOL[f], ms=2.8)
        ax.set_xscale("log", base=2)
        med = [np.median([b["conditions"][f"{f}_{d}"]["tokens"] for f in C.FAMILIES]) for d in C.DOSES]
        ax.set_xticks(med, [f"{round(med[0], -2):.0f}", f"{med[1] / 1000:.0f}k", f"{med[2] / 1000:.0f}k"])
        ax.minorticks_off()
        ax.set_xlim(med[0] / 1.5, med[2] * 1.5)
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("Prompt tokens")
        if i == 0:
            ax.set_ylabel("Known answers kept")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), NAMES[k])
    handles = [Line2D([], [], color=FCOL[f], marker="o", ms=2.8, lw=1.5, label=FAM[f]) for f in C.FAMILIES]
    handles.append(Line2D([], [], color="#7a7a7a", lw=0.9, ls=":", label="Noise floor"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=6, columnspacing=1.2,
               handlelength=1.8, handletextpad=0.4)
    style.save(fig, out / "fig_behaviour.pdf")


def repair(res, out):
    """Main figure: how much of the lost answer comes back when one site is restored, by depth."""
    keys = [k for k in models_in(res) if "patching" in res["models"][k]]
    fig = plt.figure(figsize=(style.PAGEW, 2.15))
    axs = axes_grid(fig, 1, len(keys), 0.07, 0.995, 0.3, 0.9, 0.035, 0)[0]
    for i, (ax, k) in enumerate(zip(axs, keys)):
        p = res["models"][k]["patching"]
        ax.axhline(0, color=style.AXIS, lw=0.6)
        ax.axhline(1, color="#bdbdbd", lw=0.6)
        for s in ("D", "B", "A"):
            rows = p["curves"][s]
            lab, c, ls = SITE[s]
            band(ax, [r["depth"] for r in rows], rows, c, ls=ls, lw=1.6)
        ax.plot([p["depth_star"]] * 2, [-0.42, -0.34], color=style.INK, lw=1.6, solid_capstyle="butt",
                clip_on=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.35, 1.15)
        ax.set_xlabel(DEPTH)
        if i == 0:
            ax.set_ylabel("Share of lost answer restored")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), NAMES[k])
    handles = [Line2D([], [], color=SITE[s][1], ls=SITE[s][2], lw=1.6, label=SITE[s][0]) for s in ("A", "D", "B")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3, columnspacing=1.6,
               handlelength=2.2)
    style.save(fig, out / "fig_repair.pdf")


SWAP = [("A_own", "Own question restored: gives its answer", "#b2182b", "-"),
        ("D_donor", "Other question swapped in: gives the other answer", "#2c2c2c", "-"),
        ("B_own", "Last position restored: gives its answer", "#2471a3", "--")]


def swap(res, out):
    """Main figure: what a failed wrapped prompt answers after one patch, by the depth of the patch."""
    keys = [k for k in models_in(res) if "patching" in res["models"][k]]
    fig = plt.figure(figsize=(style.PAGEW, 2.2))
    axs = axes_grid(fig, 1, len(keys), 0.07, 0.995, 0.31, 0.9, 0.035, 0)[0]
    for i, (ax, k) in enumerate(zip(axs, keys)):
        t = res["models"][k]["patching"]["top1"]
        for key, lab, c, ls in SWAP:
            rows = t[key]
            band(ax, [r["depth"] for r in rows], rows, c, ls=ls, lw=1.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(DEPTH)
        if i == 0:
            ax.set_ylabel("Gives the target answer")
        else:
            ax.set_yticklabels([])
        style.title(ax, chr(65 + i), NAMES[k])
    handles = [Line2D([], [], color=c, ls=ls, lw=1.6, label=lab) for _, lab, c, ls in SWAP]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3, columnspacing=1.4,
               handlelength=2.2)
    style.save(fig, out / "fig_swap.pdf")


def confidence(res, out):
    """Kept answers by how sure the model was of the answer without a wrapper (all models)."""
    p = res["pooled"]["type_by_confidence"]
    labels = ["0.5–0.8", "0.8–0.95", "0.95–0.99", "0.99–0.999", "> 0.999"]
    fig = plt.figure(figsize=(style.COLW, 2.1))
    ax = fig.add_axes([0.15, 0.29, 0.83, 0.68])
    x = np.arange(len(labels))
    for t in ("wrong", "scope", "task"):
        lab, c = TYPES[t]
        band(ax, x, p[t], c, lw=1.5)
        ax.plot(x, [r["mean"] for r in p[t]], "o", color=c, ms=2.8)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.25, len(labels) - 0.75)
    ax.set_ylim(0, None)
    ax.set_xlabel("Probability of the answer without a wrapper")
    ax.set_ylabel("Share of wrapped prompts")
    handles = [Line2D([], [], color=TYPES[t][1], marker="o", ms=2.8, lw=1.5, label=TYPES[t][0])
               for t in ("wrong", "scope", "task")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=3, columnspacing=1.0,
               handlelength=1.6, handletextpad=0.4)
    style.save(fig, out / "fig_confidence.pdf")


TYPES = {"wrong": ("Wrong answer", "#b2182b"), "scope": ("Not there", "#7d3c98"),
         "task": ("About the wrapper", "#2471a3")}


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
    if any("patching" in res["models"][k] for k in res["models"]):
        repair(res, out)
        swap(res, out)


if __name__ == "__main__":
    main()
