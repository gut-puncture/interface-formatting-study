from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from interface_formatting_study.calibration import SAME_WRAPPER_REDACTION
from interface_formatting_study.conflicts import conflict_population_summary, item_conflict_outcomes


@dataclass(frozen=True)
class ModelRun:
    label: str
    root: Path
    model_id: str
    semantic_run_id: str | None


def _model_metadata(root: Path) -> tuple[str, str | None]:
    run_manifest = root / "metadata" / "run_manifest.json"
    if run_manifest.exists():
        payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("canary") is not False:
            raise ValueError(f"Model run is not full-complete: {root}")
        return str(payload["model"]["id"]), str(payload["semantic_identity"]["semantic_run_id"])
    legacy_manifest = root / "metadata" / "experiment_manifest.json"
    if legacy_manifest.exists():
        payload = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        return str(payload["model"]["name"]), None
    raise FileNotFoundError(f"No supported manifest under {root}")


def load_model_run(spec: str) -> ModelRun:
    if "=" not in spec:
        raise ValueError(f"Model run must be LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    root = Path(raw_path).expanduser().resolve()
    model_id, semantic_run_id = _model_metadata(root)
    return ModelRun(label=label.strip(), root=root, model_id=model_id, semantic_run_id=semantic_run_id)


def _read_required(run: ModelRun, relative: str) -> pd.DataFrame:
    path = run.root / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing {relative} for {run.label}: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _behavioral_summary(run: ModelRun) -> dict[str, object]:
    behavioral = _read_required(run, "raw/behavioral_scores.parquet")
    if len(behavioral) != 24000 or behavioral["item_id"].nunique() != 3000:
        raise ValueError(f"{run.label} behavioral table violates the 24,000-row/3,000-item contract")
    full = conflict_population_summary(
        item_conflict_outcomes(behavioral, expected_wrappers=8),
        population="full",
    )
    eligible = conflict_population_summary(
        item_conflict_outcomes(behavioral, calibration_kind=SAME_WRAPPER_REDACTION),
        population="calibration_eligible",
    )
    return {
        "model": run.label,
        "model_id": run.model_id,
        "semantic_run_id": run.semantic_run_id,
        "behavioral_rows": len(behavioral),
        "items": behavioral["item_id"].nunique(),
        "calibrated_accuracy": float(behavioral["cal_correct"].astype(bool).mean()),
        "mean_calibrated_margin": float(behavioral["cal_margin"].mean()),
        "full_conflict_items": full["conflict_items"],
        "full_conflict_denominator": full["denominator"],
        "full_conflict_rate": full["conflict_rate"],
        "eligible_conflict_items": eligible["conflict_items"],
        "eligible_conflict_denominator": eligible["denominator"],
        "eligible_conflict_rate": eligible["conflict_rate"],
    }


def _grouped_phase(
    run: ModelRun,
    relative: str,
    group_columns: list[str],
    metrics: list[str],
    *,
    exclude_skipped: bool = False,
) -> pd.DataFrame:
    frame = _read_required(run, relative)
    if exclude_skipped and "skipped" in frame:
        frame = frame[~frame["skipped"].astype(bool)].copy()
    if frame.empty:
        raise ValueError(f"{run.label} has no usable rows in {relative}")
    available_metrics = [metric for metric in metrics if metric in frame]
    grouped = frame.groupby(group_columns, as_index=False)[available_metrics].mean(numeric_only=True)
    grouped.insert(0, "model", run.label)
    grouped.insert(1, "model_id", run.model_id)
    return grouped


def build_comparison(runs: list[ModelRun], output_dir: Path, *, paper_dir: Path | None = None) -> dict[str, Path]:
    if len(runs) < 2:
        raise ValueError("Cross-model analysis requires at least two model runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    behavioral = pd.DataFrame([_behavioral_summary(run) for run in runs]).sort_values("model")
    attention = pd.concat(
        [
            _grouped_phase(
                run,
                "processed/attention_diagnostics.parquet",
                ["anchor", "layer", "run_kind"],
                ["attention_mass_question_options", "attention_mass_wrapper_syntax", "attention_entropy"],
            )
            for run in runs
        ],
        ignore_index=True,
    )
    vanilla = pd.concat(
        [
            _grouped_phase(
                run,
                "processed/vanilla_convergence.parquet",
                ["anchor", "layer"],
                ["cos_clean_vanilla", "cos_corrupt_vanilla", "delta_clean_minus_corrupt"],
            )
            for run in runs
        ],
        ignore_index=True,
    )
    controls = pd.concat(
        [
            _grouped_phase(
                run,
                "processed/focused_patching_controls.parquet",
                ["condition", "anchor", "layer"],
                ["patched_correct", "margin_improvement", "recovery"],
                exclude_skipped=True,
            )
            for run in runs
        ],
        ignore_index=True,
    )

    paths = {
        "behavioral": table_dir / "cross_model_behavioral_summary.csv",
        "attention": table_dir / "cross_model_attention_summary.csv",
        "vanilla": table_dir / "cross_model_vanilla_summary.csv",
        "controls": table_dir / "cross_model_controls_summary.csv",
    }
    behavioral.to_csv(paths["behavioral"], index=False)
    attention.to_csv(paths["attention"], index=False)
    vanilla.to_csv(paths["vanilla"], index=False)
    controls.to_csv(paths["controls"], index=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    sns.barplot(data=behavioral, x="model", y="calibrated_accuracy", ax=axes[0], color="#3269a8")
    sns.barplot(data=behavioral, x="model", y="full_conflict_rate", ax=axes[1], color="#c26d3a")
    axes[0].set(title="Calibrated wrapper accuracy", xlabel="", ylabel="Accuracy")
    axes[1].set(title="Items with wrapper conflict", xlabel="", ylabel="Conflict rate")
    for axis, values in zip(axes, (behavioral["calibrated_accuracy"], behavioral["full_conflict_rate"])):
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylim(0, max(0.65, float(values.max()) + 0.08))
        for patch, value in zip(axis.patches, values):
            axis.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.012,
                f"{value:.1%}",
                ha="center",
                va="bottom",
            )
    fig.tight_layout()
    behavioral_stem = figure_dir / "cross_model_behavioral"
    fig.savefig(behavioral_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(behavioral_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    attention_plot = attention[
        (attention["anchor"] == "all_option_ends")
        & attention["run_kind"].isin(("clean_correct_wrapped", "corrupt_wrong_wrapped"))
    ].copy()
    attention_plot["run_kind"] = attention_plot["run_kind"].map(
        {"clean_correct_wrapped": "Successful wrapper", "corrupt_wrong_wrapped": "Failed wrapper"}
    )
    models = list(behavioral["model"])
    fig, axes = plt.subplots(1, len(models), figsize=(8.2, 3.4), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for axis, model in zip(axes, models):
        sns.lineplot(
            data=attention_plot[attention_plot["model"] == model],
            x="layer",
            y="attention_mass_question_options",
            hue="run_kind",
            hue_order=("Successful wrapper", "Failed wrapper"),
            marker="o",
            ax=axis,
        )
        axis.set(title=model, xlabel="Layer", ylabel="Question + option attention mass")
        axis.legend(title="", loc="best", fontsize="small")
    fig.suptitle("Content attention is similar for successful and failed wrappers")
    fig.tight_layout()
    attention_stem = figure_dir / "cross_model_attention"
    fig.savefig(attention_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(attention_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    control_plot = controls[controls["anchor"] == "all_option_ends"].copy()
    control_plot = control_plot.groupby(["model", "condition"], as_index=False)["recovery"].mean()
    condition_labels = {
        "same_item_clean_to_corrupt": "Same item",
        "vanilla_to_corrupt": "Unwrapped item",
        "cross_item_clean_to_corrupt": "Cross item",
        "same_label_cross_item_clean_to_corrupt": "Same-label cross item",
        "different_label_cross_item_clean_to_corrupt": "Different-label cross item",
    }
    condition_order = [
        "Same item",
        "Unwrapped item",
        "Cross item",
        "Same-label cross item",
        "Different-label cross item",
    ]
    control_plot["condition"] = control_plot["condition"].map(condition_labels)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    sns.barplot(
        data=control_plot,
        x="condition",
        y="recovery",
        hue="model",
        order=condition_order,
        ax=ax,
    )
    ax.set(title="Focused causal-control recovery", xlabel="", ylabel="Mean recovery")
    ax.tick_params(axis="x", rotation=16)
    fig.tight_layout()
    controls_stem = figure_dir / "cross_model_controls"
    fig.savefig(controls_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(controls_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    latex_frame = behavioral[
        ["model", "calibrated_accuracy", "full_conflict_items", "full_conflict_denominator", "full_conflict_rate"]
    ].copy()
    latex_frame["calibrated_accuracy"] = latex_frame["calibrated_accuracy"].map(lambda value: f"{value:.1%}")
    latex_frame["full_conflict_rate"] = latex_frame["full_conflict_rate"].map(lambda value: f"{value:.1%}")
    latex_frame = latex_frame.rename(
        columns={
            "model": "Model",
            "calibrated_accuracy": "Calibrated accuracy",
            "full_conflict_items": "Full conflict items",
            "full_conflict_denominator": "Full denominator",
            "full_conflict_rate": "Full conflict rate",
        }
    )
    latex = latex_frame.to_latex(index=False, escape=True)
    latex_path = table_dir / "table_cross_model_behavioral.tex"
    latex_path.write_text(latex, encoding="utf-8")
    paths["latex"] = latex_path

    if paper_dir is not None:
        paper_dir = paper_dir.resolve()
        (paper_dir / "figures").mkdir(parents=True, exist_ok=True)
        (paper_dir / "tables").mkdir(parents=True, exist_ok=True)
        for stem in (behavioral_stem, attention_stem, controls_stem):
            for suffix in (".png", ".pdf"):
                shutil.copy2(stem.with_suffix(suffix), paper_dir / "figures" / stem.with_suffix(suffix).name)
        shutil.copy2(latex_path, paper_dir / "tables" / latex_path.name)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", action="append", required=True, help="LABEL=PATH; repeat for each model")
    parser.add_argument("--output-dir", default="results/cross_model")
    parser.add_argument("--paper-dir", default=None)
    args = parser.parse_args()
    runs = [load_model_run(spec) for spec in args.model_run]
    paths = build_comparison(
        runs,
        Path(args.output_dir),
        paper_dir=None if args.paper_dir is None else Path(args.paper_dir),
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
