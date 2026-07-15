from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from interface_formatting_study.run_identity import sha256_file
from interface_formatting_study.causal_runner import validate_causal_design
from interface_formatting_study.statistics import clustered_bootstrap_paired_diff
from interface_formatting_study.utils import EPSILON


@dataclass(frozen=True)
class CausalRun:
    label: str
    root: Path
    model_id: str
    semantic_run_id: str


def load_run(spec: str, *, allow_canary: bool = False) -> tuple[CausalRun, pd.DataFrame]:
    if "=" not in spec:
        raise ValueError(f"Run must be LABEL=PATH, got {spec!r}")
    label, raw_root = spec.split("=", 1)
    root = Path(raw_root).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Causal run is not complete: {root}")
    if manifest.get("canary") and not allow_canary:
        raise ValueError(f"Refusing canary as a full causal run: {root}")
    identity = manifest["semantic_identity"]
    raw_path = root / "raw" / "causal_behavior.parquet"
    raw_manifest_path = raw_path.with_name(raw_path.name + ".manifest.json")
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if raw_manifest.get("semantic_run_id") != identity["semantic_run_id"]:
        raise ValueError(f"Causal artifact identity mismatch: {raw_path}")
    if raw_manifest.get("sha256") != sha256_file(raw_path):
        raise ValueError(f"Causal artifact checksum mismatch: {raw_path}")
    frame = pd.read_parquet(raw_path)
    if len(frame) != int(raw_manifest.get("row_count", -1)):
        raise ValueError(f"Causal artifact row-count mismatch: {raw_path}")
    expected_rows = int(manifest.get("design", {}).get("run_rows", -1))
    if len(frame) != expected_rows or frame["work_key"].duplicated().any():
        raise ValueError(f"Complete causal artifact violates its work-row contract: {raw_path}")
    validate_causal_design(frame)
    for column, expected in (
        ("semantic_run_id", identity["semantic_run_id"]),
        ("model_id", manifest["model"]["id"]),
        ("model_revision", manifest["model"]["revision"]),
    ):
        if set(frame[column].astype(str)) != {str(expected)}:
            raise ValueError(f"Causal artifact row identity mismatch: {column}")
    return (
        CausalRun(
            label=label.strip(),
            root=root,
            model_id=str(manifest["model"]["id"]),
            semantic_run_id=str(identity["semantic_run_id"]),
        ),
        frame,
    )


def _mean_text_correct(row: pd.Series) -> float:
    if bool(row["text_ambiguous"]):
        return float("nan")
    scores = [float(value) for value in row["candidate_mean_logps"]]
    order = sorted(range(4), key=lambda index: (-scores[index], index))
    if abs(scores[order[0]] - scores[order[1]]) <= EPSILON:
        return 0.0
    return float(order[0] == int(row["correct_content_id"]))


def _effect(
    frame: pd.DataFrame,
    *,
    model: str,
    comparison: str,
    before: str,
    after: str,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    usable = frame.dropna(subset=[before, after]).copy()
    estimate = clustered_bootstrap_paired_diff(
        usable,
        before,
        after,
        cluster_col="item_id",
        n_boot=n_boot,
        seed=seed,
    )
    return {"model": model, "comparison": comparison, **estimate}


def _bootstrap_item_stat(frame: pd.DataFrame, statistic, *, n_boot: int, seed: int) -> dict[str, float | int]:
    items = frame["item_id"].astype(str).unique()
    if len(items) < 2:
        return {"estimate": float(statistic(frame)), "ci_low": float("nan"), "ci_high": float("nan"), "n": len(frame)}
    rng = np.random.default_rng(seed)
    estimate = float(statistic(frame))
    draws = []
    groups = {item: group for item, group in frame.groupby(frame["item_id"].astype(str), sort=False)}
    for _ in range(n_boot):
        sampled = rng.choice(items, size=len(items), replace=True)
        draw = pd.concat([groups[item] for item in sampled], ignore_index=True)
        draws.append(float(statistic(draw)))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n": len(frame),
    }


def summarize_run(run: CausalRun, frame: pd.DataFrame, *, n_boot: int, seed: int):
    keys = ["item_id", "wrapper_name"]
    letter = frame[frame["arm"] == "letter_intervention"].copy()
    text = frame[frame["arm"] == "answer_text"].copy()
    if len(letter) * 1 != len(text) * 7:
        raise ValueError(f"{run.label} does not have seven letter rows per text row")

    baseline_columns = [
        "raw_correct",
        "cal_correct",
        "raw_margin",
        "cal_margin",
        "raw_entropy",
        "cal_entropy",
        "raw_predicted_content_id",
        "cal_predicted_content_id",
    ]
    baseline = letter[letter["manipulation"] == "controlled_baseline"][keys + baseline_columns].copy()
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in baseline_columns})
    blocks = baseline
    for manipulation in ("position_only", "label_only"):
        intervention = letter[letter["manipulation"] == manipulation].merge(
            baseline,
            on=keys,
            validate="many_to_one",
        )
        intervention["raw_prediction_flip"] = (
            intervention["raw_predicted_content_id"] != intervention["baseline_raw_predicted_content_id"]
        )
        intervention["cal_prediction_flip"] = (
            intervention["cal_predicted_content_id"] != intervention["baseline_cal_predicted_content_id"]
        )
        grouped = (
            intervention
            .groupby(keys, as_index=False)
            .agg(
                **{
                    f"{manipulation}_{column}": (column, "mean")
                    for column in ("raw_correct", "cal_correct", "raw_margin", "cal_margin")
                },
                **{
                    f"{manipulation}_raw_distinct_predictions": ("raw_predicted_content_id", "nunique"),
                    f"{manipulation}_cal_distinct_predictions": ("cal_predicted_content_id", "nunique"),
                },
                **{
                    f"{manipulation}_raw_flip_rate": ("raw_prediction_flip", "mean"),
                    f"{manipulation}_cal_flip_rate": ("cal_prediction_flip", "mean"),
                },
            )
        )
        blocks = blocks.merge(grouped, on=keys, validate="one_to_one")

    text = text.copy()
    text["text_generated_correct"] = pd.to_numeric(text["generated_correct"], errors="coerce")
    text["text_total_correct"] = pd.to_numeric(text["candidate_total_correct"], errors="coerce")
    text["text_mean_correct"] = text.apply(_mean_text_correct, axis=1)
    blocks = blocks.merge(
        text[keys + ["text_generated_correct", "text_total_correct", "text_mean_correct", "text_ambiguous"]],
        on=keys,
        validate="one_to_one",
    )
    blocks.insert(0, "model", run.label)

    summary = {
        "model": run.label,
        "model_id": run.model_id,
        "semantic_run_id": run.semantic_run_id,
        "items": int(blocks["item_id"].nunique()),
        "item_format_blocks": len(blocks),
        "controlled_baseline_raw_accuracy": float(blocks["baseline_raw_correct"].mean()),
        "controlled_baseline_cal_accuracy": float(blocks["baseline_cal_correct"].mean()),
        "position_only_raw_accuracy": float(blocks["position_only_raw_correct"].mean()),
        "position_only_cal_accuracy": float(blocks["position_only_cal_correct"].mean()),
        "label_only_raw_accuracy": float(blocks["label_only_raw_correct"].mean()),
        "label_only_cal_accuracy": float(blocks["label_only_cal_correct"].mean()),
        "text_generated_accuracy": float(blocks["text_generated_correct"].mean()),
        "text_total_likelihood_accuracy": float(blocks["text_total_correct"].mean()),
        "text_mean_likelihood_accuracy": float(blocks["text_mean_correct"].mean()),
        "ambiguous_text_blocks": int(blocks["text_ambiguous"].astype(bool).sum()),
    }

    effects: list[dict[str, object]] = []
    for scoring in ("raw", "cal"):
        for manipulation in ("position_only", "label_only"):
            effects.append(
                _effect(
                    blocks,
                    model=run.label,
                    comparison=f"{manipulation}_minus_baseline_{scoring}",
                    before=f"baseline_{scoring}_correct",
                    after=f"{manipulation}_{scoring}_correct",
                    n_boot=n_boot,
                    seed=seed,
                )
            )
        effects.append(
            _effect(
                blocks,
                model=run.label,
                comparison=f"generated_text_minus_baseline_{scoring}_descriptive",
                before=f"baseline_{scoring}_correct",
                after="text_generated_correct",
                n_boot=n_boot,
                seed=seed,
            )
        )

    # Difference-in-differences asks whether wrappers amplify a sensitivity also seen in plain MCQs.
    plain = blocks[blocks["wrapper_name"] == "plain"].drop(columns="wrapper_name")
    wrapped = blocks[blocks["wrapper_name"] != "plain"].merge(
        plain,
        on=["model", "item_id"],
        suffixes=("", "_plain"),
        validate="many_to_one",
    )
    for scoring in ("raw", "cal"):
        for manipulation in ("position_only", "label_only"):
            wrapped["wrapper_delta"] = (
                wrapped[f"{manipulation}_{scoring}_correct"] - wrapped[f"baseline_{scoring}_correct"]
            )
            wrapped["plain_delta"] = (
                wrapped[f"{manipulation}_{scoring}_correct_plain"]
                - wrapped[f"baseline_{scoring}_correct_plain"]
            )
            effects.append(
                _effect(
                    wrapped,
                    model=run.label,
                    comparison=f"wrapper_amplification_{manipulation}_{scoring}",
                    before="plain_delta",
                    after="wrapper_delta",
                    n_boot=n_boot,
                    seed=seed,
                )
            )
        wrapped["wrapper_readout_delta"] = (
            wrapped["text_generated_correct"].astype(float)
            - wrapped[f"baseline_{scoring}_correct"].astype(float)
        )
        wrapped["plain_readout_delta"] = (
            wrapped["text_generated_correct_plain"].astype(float)
            - wrapped[f"baseline_{scoring}_correct_plain"].astype(float)
        )
        effects.append(
            _effect(
                wrapped,
                model=run.label,
                comparison=f"wrapper_amplification_generated_text_readout_{scoring}_descriptive",
                before="plain_readout_delta",
                after="wrapper_readout_delta",
                n_boot=n_boot,
                seed=seed,
            )
        )

    assignment = (
        letter.groupby(["manipulation", "correct_position", "correct_label"], as_index=False)
        .agg(raw_accuracy=("raw_correct", "mean"), cal_accuracy=("cal_correct", "mean"), rows=("work_key", "size"))
    )
    assignment.insert(0, "model", run.label)
    choice_rows = []
    for scoring in ("raw", "cal"):
        choices = (
            letter.groupby(["manipulation", f"{scoring}_predicted_label"], as_index=False)
            .agg(rows=("work_key", "size"))
            .rename(columns={f"{scoring}_predicted_label": "predicted_label"})
        )
        choices["rate"] = choices["rows"] / choices.groupby("manipulation")["rows"].transform("sum")
        choices.insert(0, "scoring", scoring)
        choice_rows.append(choices)
    choices = pd.concat(choice_rows, ignore_index=True)
    choices.insert(0, "model", run.label)

    confidence_percentile = (-blocks["baseline_cal_entropy"]).rank(method="average", pct=True)
    blocks["confidence_stratum"] = pd.cut(
        confidence_percentile,
        bins=[0.0, 1 / 3, 2 / 3, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    strata_rows = []
    for stratum, group in blocks.groupby("confidence_stratum", observed=True):
        for manipulation in ("position_only", "label_only"):
            estimate = _effect(
                group,
                model=run.label,
                comparison=manipulation,
                before="baseline_cal_correct",
                after=f"{manipulation}_cal_correct",
                n_boot=n_boot,
                seed=seed,
            )
            estimate["confidence_stratum"] = stratum
            strata_rows.append(estimate)
    return summary, blocks, pd.DataFrame(effects), assignment, choices, pd.DataFrame(strata_rows)


def build_bridge(
    run_label: str,
    behavioral: pd.DataFrame,
    blocks: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    required = {"item_id", "wrapper_name", "cal_correct"}
    if missing := required - set(behavioral.columns):
        raise ValueError(f"Original behavioral data for {run_label} is missing: {sorted(missing)}")
    if len(behavioral) != 24000 or behavioral.duplicated(["item_id", "wrapper_name"]).any():
        raise ValueError(f"Original behavioral data for {run_label} must have 24,000 unique item-wrapper rows")
    original = behavioral.copy()
    original["cal_correct"] = original["cal_correct"].astype(bool)
    item = (
        original.groupby("item_id", as_index=False)["cal_correct"]
        .agg(["min", "max"])
        .reset_index()
    )
    item["original_conflict"] = item["min"] != item["max"]
    controlled_item = (
        blocks.groupby("item_id", as_index=False)[
            ["position_only_cal_flip_rate", "label_only_cal_flip_rate"]
        ].mean()
    )
    item_bridge = item[["item_id", "original_conflict"]].merge(controlled_item, on="item_id", validate="one_to_one")
    rows = []
    for manipulation in ("position_only", "label_only"):
        value = f"{manipulation}_cal_flip_rate"
        stat = _bootstrap_item_stat(
            item_bridge,
            lambda draw, value=value: draw.loc[draw["original_conflict"], value].mean()
            - draw.loc[~draw["original_conflict"], value].mean(),
            n_boot=n_boot,
            seed=seed,
        )
        rows.append({"model": run_label, "bridge": f"original_conflict_sensitivity_{manipulation}", **stat})

    original["old_item_mean"] = original.groupby("item_id")["cal_correct"].transform("mean")
    original["old_wrapper_degradation"] = original["cal_correct"].astype(float) - original["old_item_mean"]
    plain = blocks[blocks["wrapper_name"] == "plain"].drop(columns="wrapper_name")
    wrapped = blocks[blocks["wrapper_name"] != "plain"].merge(
        plain,
        on=["model", "item_id"],
        suffixes=("", "_plain"),
        validate="many_to_one",
    )
    bridge = original.merge(
        wrapped,
        on=["item_id", "wrapper_name"],
        validate="one_to_one",
    )
    for manipulation in ("position_only", "label_only"):
        bridge["new_wrapper_amplification"] = (
            bridge[f"{manipulation}_cal_flip_rate"] - bridge[f"{manipulation}_cal_flip_rate_plain"]
        )
        stat = _bootstrap_item_stat(
            bridge,
            lambda draw: draw["old_wrapper_degradation"].corr(draw["new_wrapper_amplification"], method="spearman"),
            n_boot=n_boot,
            seed=seed,
        )
        rows.append({"model": run_label, "bridge": f"old_degradation_vs_amplification_{manipulation}_spearman", **stat})
    return pd.DataFrame(rows)


def analyze(
    specs: list[str],
    output_dir: Path,
    *,
    allow_canary: bool = False,
    n_boot: int = 2000,
    seed: int = 1729,
    behavioral_specs: list[str] | None = None,
):
    if not specs:
        raise ValueError("At least one causal run is required")
    behavioral_by_label = {}
    for spec in behavioral_specs or []:
        if "=" not in spec:
            raise ValueError(f"Behavioral run must be LABEL=PARQUET, got {spec!r}")
        label, path = spec.split("=", 1)
        behavioral_by_label[label.strip()] = pd.read_parquet(Path(path).expanduser())
    summaries, blocks, effects, assignments, choices, strata, bridges = [], [], [], [], [], [], []
    for spec in specs:
        run, frame = load_run(spec, allow_canary=allow_canary)
        result = summarize_run(run, frame, n_boot=n_boot, seed=seed)
        summaries.append(result[0])
        blocks.append(result[1])
        effects.append(result[2])
        assignments.append(result[3])
        choices.append(result[4])
        strata.append(result[5])
        if behavioral_specs is not None:
            if run.label not in behavioral_by_label:
                raise ValueError(f"Missing original behavioral data for causal model label {run.label}")
            bridges.append(
                build_bridge(
                    run.label,
                    behavioral_by_label[run.label],
                    result[1],
                    n_boot=n_boot,
                    seed=seed,
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_summary": output_dir / "model_summary.csv",
        "paired_effects": output_dir / "paired_effects.csv",
        "assignment_accuracy": output_dir / "assignment_accuracy.csv",
        "predicted_letter_rates": output_dir / "predicted_letter_rates.csv",
        "confidence_strata": output_dir / "confidence_strata.csv",
        "observational_causal_bridge": output_dir / "observational_causal_bridge.csv",
        "blocks": output_dir / "item_format_blocks.parquet",
    }
    pd.DataFrame(summaries).sort_values("model").to_csv(outputs["model_summary"], index=False)
    pd.concat(effects, ignore_index=True).sort_values(["model", "comparison"]).to_csv(
        outputs["paired_effects"], index=False
    )
    pd.concat(assignments, ignore_index=True).sort_values(
        ["model", "manipulation", "correct_position", "correct_label"]
    ).to_csv(outputs["assignment_accuracy"], index=False)
    pd.concat(choices, ignore_index=True).sort_values(
        ["model", "scoring", "manipulation", "predicted_label"]
    ).to_csv(outputs["predicted_letter_rates"], index=False)
    pd.concat(strata, ignore_index=True).sort_values(
        ["model", "confidence_stratum", "comparison"]
    ).to_csv(outputs["confidence_strata"], index=False)
    (pd.concat(bridges, ignore_index=True) if bridges else pd.DataFrame()).to_csv(
        outputs["observational_causal_bridge"], index=False
    )
    pd.concat(blocks, ignore_index=True).sort_values(["model", "item_id", "wrapper_name"]).to_parquet(
        outputs["blocks"], index=False
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-run", action="append", required=True, help="LABEL=PATH; repeat for each model")
    parser.add_argument("--output-dir", default="results/causal_followup")
    parser.add_argument("--allow-canary", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument(
        "--behavioral-run",
        action="append",
        help="LABEL=original behavioral_scores.parquet; repeat once per causal model",
    )
    args = parser.parse_args()
    outputs = analyze(
        args.model_run,
        Path(args.output_dir),
        allow_canary=args.allow_canary,
        n_boot=args.bootstrap_samples,
        behavioral_specs=args.behavioral_run,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
