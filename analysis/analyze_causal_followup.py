from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from interface_formatting_study.run_identity import sha256_file
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


def summarize_run(run: CausalRun, frame: pd.DataFrame, *, n_boot: int, seed: int):
    keys = ["item_id", "wrapper_name"]
    letter = frame[frame["arm"] == "letter_permutation"].copy()
    text = frame[frame["arm"] == "answer_text"].copy()
    if len(letter) * 1 != len(text) * 4:
        raise ValueError(f"{run.label} does not have four letter rows per text row")

    baseline = letter[letter["variant"].astype(int) == 0][keys + ["correct"]].rename(
        columns={"correct": "identity_letter_correct"}
    )
    rearranged = (
        letter[letter["variant"].astype(int) != 0]
        .groupby(keys, as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "rearranged_letter_accuracy"})
    )
    stability = (
        letter.groupby(keys, as_index=False)["predicted_content_id"]
        .nunique()
        .rename(columns={"predicted_content_id": "distinct_content_predictions"})
    )
    stability["content_stable_across_assignments"] = stability["distinct_content_predictions"] == 1
    text = text.copy()
    text["text_total_correct"] = pd.to_numeric(text["correct"], errors="coerce")
    text["text_mean_correct"] = text.apply(_mean_text_correct, axis=1)
    text_block = text[keys + ["text_total_correct", "text_mean_correct", "text_ambiguous"]]

    blocks = baseline.merge(rearranged, on=keys, validate="one_to_one")
    blocks = blocks.merge(stability, on=keys, validate="one_to_one")
    blocks = blocks.merge(text_block, on=keys, validate="one_to_one")
    blocks.insert(0, "model", run.label)

    summary = {
        "model": run.label,
        "model_id": run.model_id,
        "semantic_run_id": run.semantic_run_id,
        "items": int(blocks["item_id"].nunique()),
        "item_wrapper_blocks": len(blocks),
        "identity_letter_accuracy": float(blocks["identity_letter_correct"].mean()),
        "rearranged_letter_accuracy": float(blocks["rearranged_letter_accuracy"].mean()),
        "content_stability_rate": float(blocks["content_stable_across_assignments"].mean()),
        "text_total_accuracy": float(blocks["text_total_correct"].mean()),
        "text_mean_accuracy": float(blocks["text_mean_correct"].mean()),
        "ambiguous_text_blocks": int(blocks["text_ambiguous"].astype(bool).sum()),
    }

    effects = []
    comparisons = (
        ("rearranged_minus_identity", "identity_letter_correct", "rearranged_letter_accuracy"),
        ("text_total_minus_identity", "identity_letter_correct", "text_total_correct"),
        ("text_mean_minus_identity", "identity_letter_correct", "text_mean_correct"),
    )
    for name, before, after in comparisons:
        usable = blocks.dropna(subset=[before, after]).copy()
        estimate = clustered_bootstrap_paired_diff(
            usable,
            before,
            after,
            cluster_col="item_id",
            n_boot=n_boot,
            seed=seed,
        )
        effects.append({"model": run.label, "comparison": name, **estimate})

    assignment = (
        letter.groupby(["correct_position", "correct_label"], as_index=False)
        .agg(accuracy=("correct", "mean"), rows=("work_key", "size"))
    )
    assignment.insert(0, "model", run.label)
    choices = (
        letter.groupby(["predicted_label"], as_index=False)
        .agg(rows=("work_key", "size"))
    )
    choices["rate"] = choices["rows"] / choices["rows"].sum()
    choices.insert(0, "model", run.label)
    return summary, blocks, pd.DataFrame(effects), assignment, choices


def analyze(specs: list[str], output_dir: Path, *, allow_canary: bool = False, n_boot: int = 2000, seed: int = 1729):
    if not specs:
        raise ValueError("At least one causal run is required")
    summaries = []
    blocks = []
    effects = []
    assignments = []
    choices = []
    for spec in specs:
        run, frame = load_run(spec, allow_canary=allow_canary)
        result = summarize_run(run, frame, n_boot=n_boot, seed=seed)
        summaries.append(result[0])
        blocks.append(result[1])
        effects.append(result[2])
        assignments.append(result[3])
        choices.append(result[4])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_summary": output_dir / "model_summary.csv",
        "paired_effects": output_dir / "paired_effects.csv",
        "assignment_accuracy": output_dir / "assignment_accuracy.csv",
        "predicted_letter_rates": output_dir / "predicted_letter_rates.csv",
        "blocks": output_dir / "item_wrapper_blocks.parquet",
    }
    pd.DataFrame(summaries).sort_values("model").to_csv(outputs["model_summary"], index=False)
    pd.concat(effects, ignore_index=True).sort_values(["model", "comparison"]).to_csv(outputs["paired_effects"], index=False)
    pd.concat(assignments, ignore_index=True).sort_values(["model", "correct_position", "correct_label"]).to_csv(
        outputs["assignment_accuracy"], index=False
    )
    pd.concat(choices, ignore_index=True).sort_values(["model", "predicted_label"]).to_csv(
        outputs["predicted_letter_rates"], index=False
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
    args = parser.parse_args()
    outputs = analyze(
        args.model_run,
        Path(args.output_dir),
        allow_canary=args.allow_canary,
        n_boot=args.bootstrap_samples,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
