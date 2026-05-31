from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .anchors import anchor_is_leaky
from .calibration import SAME_WRAPPER_REDACTION
from .conflicts import item_outcome_summary
from .utils import read_json, write_table


def _require_non_null(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise ValueError(f"{name} has null provenance values in {column}")


def _validate_final_intervention_frame(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} is empty")
    expected = {
        "eval_split": "test",
        "location_selection_split": "validation",
        "alpha_selection_split": "validation",
        "vector_train_split": "train",
    }
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing provenance columns: {sorted(missing)}")
    _require_non_null(frame, set(expected), name=name)
    for column, value in expected.items():
        observed = set(frame[column].astype(str))
        if observed != {value}:
            raise ValueError(f"{name} has invalid {column}: {sorted(observed)}")
    required_singletons = {
        "model_name",
        "dataset_sha256",
        "selected_location_sha256",
        "selected_alpha_sha256",
        "vector_artifact_sha256",
        "vector_tensor_sha256",
    }
    missing_singletons = required_singletons - set(frame.columns)
    if missing_singletons:
        raise ValueError(f"{name} is missing artifact provenance columns: {sorted(missing_singletons)}")
    _require_non_null(frame, required_singletons, name=name)
    for column in required_singletons:
        values = set(frame[column].astype(str))
        if len(values) != 1:
            raise ValueError(f"{name} has non-singleton provenance column {column}: {sorted(values)}")


def _validate_conflict_bound_frame(frame: pd.DataFrame, *, name: str) -> None:
    if "conflict_pairs_sha256" not in frame.columns:
        raise ValueError(f"{name} is missing conflict_pairs_sha256")
    _require_non_null(frame, {"conflict_pairs_sha256"}, name=name)
    values = set(frame["conflict_pairs_sha256"].astype(str))
    if len(values) != 1:
        raise ValueError(f"{name} has non-singleton conflict_pairs_sha256: {sorted(values)}")


def make_dataset_wrapper_table(
    dataset_audit: Mapping[str, object],
    wrapper_audit: pd.DataFrame,
    *,
    run_type: str = "final",
    behavioral_rows_scored: int | None = None,
) -> pd.DataFrame:
    prompt_lengths = dataset_audit.get("prompt_length_chars", {})
    active_wrappers = set((dataset_audit.get("wrapper_counts") or {}).keys())
    active_audit = wrapper_audit[wrapper_audit["wrapper_name"].isin(active_wrappers)]
    return pd.DataFrame(
        [
            {
                "number_of_mmlu_items": int(dataset_audit["num_items"]),
                "number_of_wrappers": int(dataset_audit["num_wrappers"]),
                "number_of_prompts": int(dataset_audit["num_rows"]),
                "number_of_pure_interface_wrappers": int((active_audit["category"] == "pure_interface").sum()),
                "number_of_style_transforming_wrappers": int((active_audit["category"] == "style_transforming").sum()),
                "mean_prompt_chars": float(prompt_lengths.get("mean", 0.0)),
                "max_prompt_chars": int(prompt_lengths.get("max", 0)),
                "behavioral_run_type": run_type,
                "behavioral_rows_scored": int(behavioral_rows_scored) if behavioral_rows_scored is not None else int(dataset_audit["num_rows"]),
            }
        ]
    )


def make_behavioral_conflict_table(scored: pd.DataFrame, *, run_type: str = "final") -> pd.DataFrame:
    summary = item_outcome_summary(scored)
    rows: list[dict[str, object]] = []
    for category, group in scored.groupby("wrapper_category", sort=True):
        rows.append(
            {
                "group": f"category:{category}",
                "run_type": run_type,
                "calibrated_accuracy": float(group["cal_correct"].mean()),
                "raw_accuracy": float(group["raw_correct"].mean()) if "raw_correct" in group else float("nan"),
                "mean_calibrated_margin": float(group["cal_margin"].mean()),
                "conflict_item_rate": float("nan"),
                "all_correct_rate": float("nan"),
                "all_wrong_rate": float("nan"),
                "mean_correct_wrappers_per_item": float("nan"),
                "n_items": int(group["item_id"].nunique()),
                "n_prompts": int(len(group)),
            }
        )
    rows.append(
        {
            "group": "all_items",
            "run_type": run_type,
            "calibrated_accuracy": float(scored["cal_correct"].mean()),
            "raw_accuracy": float(scored["raw_correct"].mean()) if "raw_correct" in scored else float("nan"),
            "mean_calibrated_margin": float(scored["cal_margin"].mean()),
            "conflict_item_rate": float(summary["is_conflict"].mean()),
            "all_correct_rate": float(summary["all_correct"].mean()),
            "all_wrong_rate": float(summary["all_wrong"].mean()),
            "mean_correct_wrappers_per_item": float(summary["n_correct"].mean()),
            "n_items": int(scored["item_id"].nunique()),
            "n_prompts": int(len(scored)),
        }
    )
    return pd.DataFrame(rows)


def primary_behavioral_rows(scored: pd.DataFrame, *, run_type: str = "final") -> pd.DataFrame:
    if run_type != "final" or "content_free_calibration_kind" not in scored.columns:
        return scored
    primary = scored[scored["content_free_calibration_kind"] == SAME_WRAPPER_REDACTION].copy()
    if primary.empty:
        raise ValueError("Final behavioral summaries require same-wrapper calibrated rows")
    return primary


def make_intervention_table(access_results: pd.DataFrame, removal_results: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if access_results is not None and not access_results.empty:
        for condition, group in access_results.groupby("condition", sort=True):
            rows.append(
                {
                    "condition": condition,
                    "accuracy": float(group["intervention_correct"].mean()),
                    "margin": float(group["intervention_margin"].mean()),
                    "repair_or_damage_rate": float(group["repair"].mean()) if "repair" in group else float("nan"),
                    "n_items": int(group["item_id"].nunique()),
                    "n_rows": int(len(group)),
                }
            )
    if removal_results is not None and not removal_results.empty:
        for condition, group in removal_results.groupby("condition", sort=True):
            rows.append(
                {
                    "condition": condition,
                    "accuracy": float(group["intervention_correct"].mean()),
                    "margin": float(group["intervention_margin"].mean()),
                    "repair_or_damage_rate": float(group["damage"].mean()) if "damage" in group else float("nan"),
                    "n_items": int(group["item_id"].nunique()),
                    "n_rows": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def make_content_free_control_table(content_free: pd.DataFrame) -> pd.DataFrame:
    if content_free.empty:
        return pd.DataFrame(
            columns=["condition", "n_items", "n_rows", "max_label_share", "prediction_entropy", "mean_margin"]
        )
    import math

    counts = content_free["pred_label"].value_counts(normalize=True)
    entropy = -sum(float(p) * math.log(float(p)) for p in counts if p > 0)
    return pd.DataFrame(
        [
            {
                "condition": "content_free_interface_formatting_study",
                "n_items": int(content_free["item_id"].nunique()) if "item_id" in content_free else 0,
                "n_rows": int(len(content_free)),
                "max_label_share": float(counts.max()) if not counts.empty else float("nan"),
                "prediction_entropy": float(entropy),
                "mean_margin": float(content_free["cal_margin"].mean()) if "cal_margin" in content_free else float("nan"),
            }
        ]
    )


def make_semantic_patching_table(patching: pd.DataFrame) -> pd.DataFrame:
    if patching.empty:
        raise ValueError("patching_results.parquet is empty")
    source = patching.copy()
    if "skipped" in source.columns:
        source = source[~source["skipped"].astype(bool)].copy()
    if source.empty:
        raise ValueError("patching_results.parquet has no resolved patching rows")
    grouped = (
        source.groupby(["anchor", "layer"], as_index=False)
        .agg(
            n_items=("item_id", "nunique"),
            n_rows=("item_id", "size"),
            n_positions=("n_positions", "mean") if "n_positions" in source.columns else ("item_id", "size"),
            patched_accuracy=("patched_correct", "mean"),
            mean_margin_improvement=("margin_improvement", "mean"),
            mean_recovery=("recovery", "mean"),
        )
        .sort_values(["mean_margin_improvement", "mean_recovery"], ascending=[False, False])
    )
    grouped["is_leaky_anchor"] = grouped["anchor"].astype(str).map(anchor_is_leaky)
    return grouped[
        [
            "anchor",
            "layer",
            "is_leaky_anchor",
            "n_items",
            "n_rows",
            "n_positions",
            "patched_accuracy",
            "mean_margin_improvement",
            "mean_recovery",
        ]
    ]


def write_standard_tables(
    *,
    dataset_audit_path: str | Path,
    wrapper_audit_path: str | Path,
    behavioral_path: str | Path,
    output_dir: str | Path,
    patching_results_path: str | Path | None = None,
    access_results_path: str | Path | None = None,
    control_results_path: str | Path | None = None,
    removal_results_path: str | Path | None = None,
    content_free_control_path: str | Path | None = None,
) -> dict[str, Path]:
    out = Path(output_dir)
    dataset_audit = read_json(dataset_audit_path)
    wrapper_audit = pd.read_csv(wrapper_audit_path)
    scored = pd.read_parquet(behavioral_path)
    expected_prompts = int(dataset_audit["num_rows"])
    if len(scored) != expected_prompts:
        if not str(behavioral_path).endswith("behavioral_smoke.parquet"):
            raise ValueError(
                f"Behavioral table has {len(scored)} rows but dataset audit expects {expected_prompts}; "
                "refusing to write publication-style tables."
            )
        run_type = "smoke"
    else:
        run_type = "final"

    behavioral_summary_source = primary_behavioral_rows(scored, run_type=run_type)

    paths = {
        "dataset_wrapper": out / "table1_dataset_wrapper_audit.csv",
        "behavioral_conflicts": out / "table2_behavioral_conflicts.csv",
    }
    write_table(
        make_dataset_wrapper_table(dataset_audit, wrapper_audit, run_type=run_type, behavioral_rows_scored=len(scored)),
        paths["dataset_wrapper"],
    )
    write_table(make_behavioral_conflict_table(behavioral_summary_source, run_type=run_type), paths["behavioral_conflicts"])

    if patching_results_path and Path(patching_results_path).exists():
        patching = pd.read_parquet(patching_results_path)
        paths["semantic_patching"] = out / "table3_semantic_patching.csv"
        write_table(make_semantic_patching_table(patching), paths["semantic_patching"])

    access_frames = []
    if access_results_path and Path(access_results_path).exists():
        frame = pd.read_parquet(access_results_path)
        if run_type == "final":
            _validate_final_intervention_frame(frame, name=str(access_results_path))
            _validate_conflict_bound_frame(frame, name=str(access_results_path))
        access_frames.append(frame)
    if control_results_path and Path(control_results_path).exists():
        frame = pd.read_parquet(control_results_path)
        if run_type == "final":
            _validate_final_intervention_frame(frame, name=str(control_results_path))
            _validate_conflict_bound_frame(frame, name=str(control_results_path))
            if "condition" not in frame or frame["condition"].isna().any():
                raise ValueError("Final control results are missing non-null condition values")
            conditions = set(frame["condition"].astype(str))
            required_conditions = {"shuffled_pair_vector", "all_wrong_interface_formatting_study", "all_correct_interface_formatting_study"}
            missing_conditions = required_conditions - conditions
            if missing_conditions:
                raise ValueError(f"Final control results are missing conditions: {sorted(missing_conditions)}")
        access_frames.append(frame)
    removal = pd.read_parquet(removal_results_path) if removal_results_path and Path(removal_results_path).exists() else None
    if run_type == "final" and removal is not None:
        _validate_final_intervention_frame(removal, name=str(removal_results_path))
        _validate_conflict_bound_frame(removal, name=str(removal_results_path))
    if access_frames or removal is not None:
        paths["interventions"] = out / "table3_main_interventions.csv"
        access = pd.concat(access_frames, ignore_index=True) if access_frames else pd.DataFrame()
        write_table(make_intervention_table(access, removal), paths["interventions"])
    if content_free_control_path and Path(content_free_control_path).exists():
        content_free = pd.read_parquet(content_free_control_path)
        if run_type == "final":
            _validate_final_intervention_frame(content_free, name=str(content_free_control_path))
            if "behavioral_scores_sha256" not in content_free.columns:
                raise ValueError(f"{content_free_control_path} is missing behavioral_scores_sha256")
            _require_non_null(content_free, {"behavioral_scores_sha256"}, name=str(content_free_control_path))
            values = set(content_free["behavioral_scores_sha256"].astype(str))
            if len(values) != 1:
                raise ValueError(f"{content_free_control_path} has non-singleton behavioral_scores_sha256: {sorted(values)}")
        paths["content_free_control"] = out / "table4_content_free_control.csv"
        write_table(make_content_free_control_table(content_free), paths["content_free_control"])
    return paths
