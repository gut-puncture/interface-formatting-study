"""Build the camera-ready cross-model summary figure from frozen result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2166AC"
RED = "#B2182B"
OCHRE = "#D1771E"
DARK = "#4D4D4D"
LIGHT = "#E6E6E6"

MODELS = ["Mistral-7B", "Phi-3.5-mini", "Qwen2.5-1.5B"]


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(DARK)
    ax.tick_params(colors=DARK, labelsize=8)
    ax.grid(axis="x", color=LIGHT, linewidth=0.8, zorder=0)


def main() -> None:
    # Behavioral census: 3,000 items per model, eight wrapped prompts per item.
    conflict = np.array([66.2, 55.1, 62.0])
    adjusted = np.array([66.1, 54.8, 61.5])

    # Paired item-cluster bootstrap over 2,401 items; values are percentage points.
    position = np.array([-6.87, -4.27, -7.99])
    position_ci = np.array([[-7.84, -5.95], [-5.08, -3.43], [-9.01, -6.98]])
    label = np.array([-6.08, -4.38, -6.57])
    label_ci = np.array([[-6.88, -5.29], [-5.03, -3.75], [-7.35, -5.79]])

    # Candidate-minus-letter agreement AUC at the shared answer-prefix state.
    # The first-token subset is the clean root-state readout.
    letter_contract = np.array([6.89, 1.18, -4.83])
    letter_ci = np.array([[5.9, 7.9], [0.1, 2.3], [-6.1, -3.5]])
    text_contract = np.array([5.06, 5.04, -3.27])
    text_ci = np.array([[4.2, 6.0], [4.1, 6.0], [-4.5, -2.0]])

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.9), constrained_layout=True)
    y = np.arange(len(MODELS))

    ax = axes[0]
    ax.barh(y + 0.14, conflict, height=0.25, color=BLUE, label="All rows", zorder=3)
    ax.barh(y - 0.14, adjusted, height=0.25, color=DARK, label="After audit", zorder=3)
    ax.set_yticks(y, MODELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xlabel("Items with conflicting answers (%)", fontsize=8)
    ax.set_title("a  Wrapper instability", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.27))
    _style_axis(ax)

    ax = axes[1]
    for offset, values, cis, color, marker, name in [
        (0.12, position, position_ci, OCHRE, "o", "Move content"),
        (-0.12, label, label_ci, RED, "s", "Rotate labels"),
    ]:
        errors = np.vstack((values - cis[:, 0], cis[:, 1] - values))
        ax.errorbar(values, y + offset, xerr=errors, fmt=marker, color=color,
                    capsize=2.5, markersize=4, linewidth=1.2, label=name, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set_xlim(-10.5, 1)
    ax.set_xlabel("Change in accuracy (points)", fontsize=8)
    ax.set_title("b  Controlled rotations", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=1,
              bbox_to_anchor=(0.5, -0.27))
    _style_axis(ax)

    ax = axes[2]
    for offset, values, cis, color, marker, name in [
        (0.12, letter_contract, letter_ci, BLUE, "o", "Letter instruction"),
        (-0.12, text_contract, text_ci, RED, "s", "Answer-text instruction"),
    ]:
        errors = np.vstack((values - cis[:, 0], cis[:, 1] - values))
        ax.errorbar(values, y + offset, xerr=errors, fmt=marker, color=color,
                    capsize=2.5, markersize=4, linewidth=1.2, label=name, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.8)
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.set_xlim(-7.5, 9)
    ax.set_xlabel("Candidate minus letter stability\n(AUC points)", fontsize=8)
    ax.set_title("c  Clean first-token readout", loc="left", fontsize=9, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=1,
              bbox_to_anchor=(0.5, -0.27))
    _style_axis(ax)

    out = Path(__file__).resolve().parents[1] / "paper" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "camera_ready_summary.pdf", bbox_inches="tight")
    fig.savefig(out / "camera_ready_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
