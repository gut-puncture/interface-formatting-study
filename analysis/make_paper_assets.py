from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "gpu_artifacts" / "20260531T021109Z" / "extracted"
RESULTS = ARTIFACT_ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"

BLUE = "#356AA0"
RED = "#B54A4A"
GRAY = "#777777"
ORANGE = "#C77C2D"
DARK = "#333333"
LIGHT_GRAY = "#D8D8D8"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def plot_setup() -> None:
    cache = ROOT / "results" / "metadata" / "plot_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import rcParams

    rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": DARK,
            "axes.linewidth": 0.8,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", dpi=300)


def make_behavioral_figure(behavioral: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    wrapper_acc = behavioral.groupby("wrapper_name")["cal_correct"].mean().sort_values()
    item_correct_counts = behavioral.groupby("item_id")["cal_correct"].sum().astype(int)
    count_dist = item_correct_counts.value_counts().reindex(range(9), fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), gridspec_kw={"width_ratios": [1.2, 1.0]})

    labels = [name.replace("_", "\n") for name in wrapper_acc.index]
    bars = axes[0].barh(range(len(wrapper_acc)), 100 * wrapper_acc.values, color=BLUE, edgecolor=DARK, linewidth=0.4)
    axes[0].set_yticks(range(len(wrapper_acc)), labels)
    axes[0].set_xlabel("Calibrated accuracy (%)")
    axes[0].set_xlim(38, 55)
    axes[0].grid(axis="x", alpha=0.7)
    axes[0].set_title("Accuracy by interface wrapper")
    for bar, value in zip(bars, wrapper_acc.values):
        axes[0].text(100 * value + 0.25, bar.get_y() + bar.get_height() / 2, f"{100 * value:.1f}", va="center", fontsize=7)

    axes[1].bar(count_dist.index, count_dist.values, color=RED, edgecolor=DARK, linewidth=0.4)
    axes[1].set_xticks(range(9))
    axes[1].set_xlabel("Correct wrappers for same item")
    axes[1].set_ylabel("MMLU items")
    axes[1].set_title("Item-level instability")
    axes[1].grid(axis="y", alpha=0.7)
    axes[1].axvline(item_correct_counts.mean(), color=DARK, linestyle="--", linewidth=1.0)
    axes[1].text(item_correct_counts.mean() + 0.08, max(count_dist.values) * 0.93, "mean", fontsize=7, color=DARK)

    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "behavioral_instability")
    plt.close(fig)


def make_attention_figure(attention: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "clean_correct_wrapped": "successful wrapper",
        "corrupt_wrong_wrapped": "failed wrapper",
        "vanilla_same_item": "unwrapped same item",
    }
    colors = {
        "clean_correct_wrapped": BLUE,
        "corrupt_wrong_wrapped": RED,
        "vanilla_same_item": GRAY,
    }
    anchors = [("all_option_ends", "All option ends"), ("options_end", "Final option end")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharey=True)
    for ax, (anchor, title) in zip(axes, anchors):
        sub = attention[attention["anchor"] == anchor]
        for run_kind in ["clean_correct_wrapped", "corrupt_wrong_wrapped", "vanilla_same_item"]:
            series = sub[sub["run_kind"] == run_kind].sort_values("layer")
            ax.plot(
                series["layer"],
                series["attention_mass_question_options"],
                label=labels[run_kind],
                color=colors[run_kind],
                linewidth=2.0 if run_kind != "vanilla_same_item" else 1.5,
                linestyle={"clean_correct_wrapped": "-", "corrupt_wrong_wrapped": "--", "vanilla_same_item": (0, (4, 2))}[run_kind],
                marker={"clean_correct_wrapped": "o", "corrupt_wrong_wrapped": "s", "vanilla_same_item": None}[run_kind],
                markersize=2.4 if run_kind != "vanilla_same_item" else 0,
                markevery=3,
            )
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylim(0.15, 0.98)
        ax.grid(axis="y", alpha=0.65)
    axes[0].set_ylabel("Attention to question + options")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    save_figure(fig, "attention_content_mass")
    plt.close(fig)


def make_causal_figure(controls: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    all_option = controls[controls["anchor"] == "all_option_ends"].copy()
    order = [
        "same_item_clean_to_corrupt",
        "vanilla_to_corrupt",
        "cross_item_clean_to_corrupt",
        "same_label_cross_item_clean_to_corrupt",
        "different_label_cross_item_clean_to_corrupt",
    ]
    labels = {
        "same_item_clean_to_corrupt": "same item successful",
        "vanilla_to_corrupt": "same item unwrapped",
        "cross_item_clean_to_corrupt": "cross item successful",
        "same_label_cross_item_clean_to_corrupt": "same-label cross item",
        "different_label_cross_item_clean_to_corrupt": "different-label cross item",
    }
    colors = {
        "same_item_clean_to_corrupt": BLUE,
        "vanilla_to_corrupt": GRAY,
        "cross_item_clean_to_corrupt": DARK,
        "same_label_cross_item_clean_to_corrupt": RED,
        "different_label_cross_item_clean_to_corrupt": ORANGE,
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharex=True)
    for condition in order:
        sub = all_option[all_option["condition"] == condition].sort_values("layer")
        axes[0].plot(
            sub["layer"],
            100 * sub["patched_accuracy"],
            label=labels[condition],
            color=colors[condition],
            linewidth=1.8,
            marker="o",
            markersize=2.6,
        )
        axes[1].plot(
            sub["layer"],
            sub["mean_recovery"],
            label=labels[condition],
            color=colors[condition],
            linewidth=1.8,
            marker="o",
            markersize=2.6,
        )

    axes[0].set_title("Patched accuracy")
    axes[0].set_ylabel("Originally wrong cases fixed (%)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylim(0, 23)
    axes[0].grid(axis="y", alpha=0.65)

    axes[1].set_title("Mean recovery")
    axes[1].set_ylabel("Recovery")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylim(0, 0.28)
    axes[1].grid(axis="y", alpha=0.65)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.08, -0.24), ncol=2)

    fig.tight_layout(w_pad=1.1)
    save_figure(fig, "causal_controls_all_option_ends")
    plt.close(fig)


def make_tables(behavioral: pd.DataFrame, attention: pd.DataFrame, convergence: pd.DataFrame, controls: pd.DataFrame) -> None:
    full_items = behavioral.groupby("item_id")["cal_correct"].sum()
    full = {
        "rows": len(behavioral),
        "items": behavioral["item_id"].nunique(),
        "cal_acc": behavioral["cal_correct"].mean(),
        "raw_acc": behavioral["raw_correct"].mean(),
        "margin": behavioral["cal_margin"].mean(),
        "conflict": ((full_items > 0) & (full_items < 8)).mean(),
        "all_correct": (full_items == 8).mean(),
        "all_wrong": (full_items == 0).mean(),
        "mean_correct": full_items.mean(),
    }
    primary = behavioral[behavioral["content_free_calibration_kind"] == "same_wrapper_redaction"].copy()
    primary_counts = primary.groupby("item_id")["cal_correct"].sum()
    primary_sizes = primary.groupby("item_id").size()
    prim = {
        "rows": len(primary),
        "items": primary["item_id"].nunique(),
        "cal_acc": primary["cal_correct"].mean(),
        "raw_acc": primary["raw_correct"].mean(),
        "margin": primary["cal_margin"].mean(),
        "conflict": ((primary_counts > 0) & (primary_counts < primary_sizes)).mean(),
        "all_correct": (primary_counts == primary_sizes).mean(),
        "all_wrong": (primary_counts == 0).mean(),
        "mean_correct": primary_counts.mean(),
    }

    write(
        TABLES / "table_study_design.tex",
        "\n".join(
            [
                "\\begin{tabular}{ll}",
                "\\toprule",
                "Quantity & Value \\\\",
                "\\midrule",
                "Model under test & \\texttt{Qwen/Qwen2.5-1.5B-Instruct} \\\\",
                "Dataset & 3,000 MMLU source-test items \\\\",
                "Interface wrappers & 8 verified pure-interface wrappers \\\\",
                "Wrapped prompts scored & 24,000 \\\\",
                "Primary calibrated set & 18,577 rows over 2,672 items \\\\",
                "Mechanistic validation pairs & 285 same-item success/failure pairs \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
    )

    write(
        TABLES / "table_behavioral_summary.tex",
        "\n".join(
            [
                "\\begin{tabular}{lrrrrrrr}",
                "\\toprule",
                "Population & Rows & Items & Cal. acc. & Raw acc. & Conflict & All correct & All wrong \\\\",
                "\\midrule",
                f"Full scored set & {full['rows']:,} & {full['items']:,} & {pct(full['cal_acc'])} & {pct(full['raw_acc'])} & {pct(full['conflict'])} & {pct(full['all_correct'])} & {pct(full['all_wrong'])} \\\\",
                f"Primary calibrated set & {prim['rows']:,} & {prim['items']:,} & {pct(prim['cal_acc'])} & {pct(prim['raw_acc'])} & {pct(prim['conflict'])} & {pct(prim['all_correct'])} & {pct(prim['all_wrong'])} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
    )

    att_rows = []
    for anchor in ["all_option_ends", "options_end"]:
        sub = attention[attention["anchor"] == anchor]
        pivot = sub.pivot(index="layer", columns="run_kind", values="attention_mass_question_options")
        clean_corrupt = (pivot["clean_correct_wrapped"] - pivot["corrupt_wrong_wrapped"]).abs()
        vanilla_gap = pivot["vanilla_same_item"] - pivot["clean_correct_wrapped"]
        att_rows.append((anchor.replace("_", " "), clean_corrupt.mean(), clean_corrupt.max(), vanilla_gap.mean()))

    vc_rows = []
    for anchor in ["all_option_ends", "options_end"]:
        sub = convergence[convergence["anchor"] == anchor]
        vc_rows.append((anchor.replace("_", " "), sub["delta_clean_minus_corrupt"].min(), sub["delta_clean_minus_corrupt"].max()))

    same_item = controls[(controls["anchor"] == "all_option_ends") & (controls["condition"] == "same_item_clean_to_corrupt")]
    vanilla = controls[(controls["anchor"] == "all_option_ends") & (controls["condition"] == "vanilla_to_corrupt")]
    same_label = controls[(controls["anchor"] == "all_option_ends") & (controls["condition"] == "same_label_cross_item_clean_to_corrupt")]
    different_label = controls[
        (controls["anchor"] == "all_option_ends") & (controls["condition"] == "different_label_cross_item_clean_to_corrupt")
    ]

    write(
        TABLES / "table_mechanistic_summary.tex",
        "\n".join(
            [
                "\\begin{tabular}{lll}",
                "\\toprule",
                "Diagnostic & Anchor/condition & Value \\\\",
                "\\midrule",
                f"Attention success-failure gap, mean/max & {att_rows[0][0]} & {fmt(att_rows[0][1], 4)} / {fmt(att_rows[0][2], 4)} \\\\",
                f"Attention unwrapped-success gap, mean & {att_rows[0][0]} & {fmt(att_rows[0][3], 3)} \\\\",
                f"Attention success-failure gap, mean/max & {att_rows[1][0]} & {fmt(att_rows[1][1], 4)} / {fmt(att_rows[1][2], 4)} \\\\",
                f"Attention unwrapped-success gap, mean & {att_rows[1][0]} & {fmt(att_rows[1][3], 3)} \\\\",
                f"Vanilla cosine delta range & {vc_rows[0][0]} & {fmt(vc_rows[0][1], 4)}--{fmt(vc_rows[0][2], 4)} \\\\",
                f"Vanilla cosine delta range & {vc_rows[1][0]} & {fmt(vc_rows[1][1], 4)}--{fmt(vc_rows[1][2], 4)} \\\\",
                "\\midrule",
                f"Max patched accuracy & same item successful & {pct(same_item['patched_accuracy'].max())} \\\\",
                f"Max patched accuracy & same item unwrapped & {pct(vanilla['patched_accuracy'].max())} \\\\",
                f"Max patched accuracy & same-label cross item & {pct(same_label['patched_accuracy'].max())} \\\\",
                f"Max patched accuracy & different-label cross item & {pct(different_label['patched_accuracy'].max())} \\\\",
                f"Average recovery & same-label cross item & {fmt(same_label['mean_recovery'].mean(), 3)} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "",
            ]
        ),
    )


def main() -> None:
    plot_setup()
    behavioral = pd.read_parquet(RESULTS / "raw" / "behavioral_scores.parquet")
    attention = pd.read_csv(RESULTS / "tables" / "table_attention_diagnostics.csv")
    convergence = pd.read_csv(RESULTS / "tables" / "table_vanilla_convergence.csv")
    controls = pd.read_csv(RESULTS / "tables" / "table_focused_controls.csv")

    make_behavioral_figure(behavioral)
    make_attention_figure(attention)
    make_causal_figure(controls)
    make_tables(behavioral, attention, convergence, controls)
    print(f"wrote figures to {FIGURES}")
    print(f"wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
