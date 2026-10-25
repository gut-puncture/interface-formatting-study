"""Figure style: serif type matching the paper's body (Times), bold lettered panel titles, firm lines,
no grid, dark axes; 3-D answer clouds over a soft floor with shadows."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

INK = "#222222"
AXIS = "#333333"
MUTED = "#8a8a8a"
FLOOR = "#ececee"
SHADOW = "#b4b4bb"
ACCENT = "#b2182b"
COLW = 3.3          # one column of the two-column page, inches
PAGEW = 6.95        # full page width, inches
# Twelve clearly distinct, saturated colours for categorical answers (no pale yellows on white).
DISTINCT = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a6b4", "#f032e6", "#7cb800",
            "#9a6324", "#000075", "#808000", "#ff6f91"]


def setup():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.7, "axes.grid": False, "legend.frameon": False,
        "axes.titlepad": 6, "axes.labelpad": 3,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def title(ax, letter, text):
    ax.set_title(f"{letter}. {text}", loc="left", fontweight="bold")


def floor_cloud(ax, P, colors, size=4, alpha=0.9, pad=0.15):
    """A 3-D cloud: points above a light square floor, each casting a faint shadow."""
    P = np.asarray(P, float)
    lo, hi = P.min(0), P.max(0)
    span = (hi - lo).max() * (1 + pad)
    mid = (hi + lo) / 2
    x0, x1 = mid[0] - span / 2, mid[0] + span / 2
    y0, y1 = mid[1] - span / 2, mid[1] + span / 2
    z0 = lo[2] - 0.08 * span
    # Fixed drawing order: the floor lies below every point, so it must never be painted over one.
    # (Matplotlib's automatic 3-D ordering sorts whole artists by depth and sometimes puts the floor on top.)
    ax.computed_zorder = False
    ax.plot_surface(np.array([[x0, x1], [x0, x1]]), np.array([[y0, y0], [y1, y1]]), np.full((2, 2), z0),
                    color=FLOOR, alpha=1.0, linewidth=0, antialiased=True, shade=False, zorder=0)
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), z0), s=size * 0.7, c=SHADOW, alpha=0.45, linewidths=0,
               depthshade=False, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=size, c=colors, alpha=alpha, linewidths=0, depthshade=False, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z0 + span)
    ax.view_init(elev=24, azim=-48)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.8), zoom=1.12)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
