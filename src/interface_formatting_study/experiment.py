from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import torch

from .access_vector import build_access_vector, random_vector_like, shuffled_pair_vector
from .anchors import resolve_anchor
from .calibration import SAME_WRAPPER_REDACTION, make_content_free_prompt
from .conflicts import construct_conflict_pairs, item_outcome_summary
from .data_loading import load_normalized
from .hooks import add_vector, remove_projection
from .interventions import bias_scores_from_row, score_prompt_with_optional_edit
from .patching import hidden_state_at
from .splits import assign_splits_by_item, assert_item_split_integrity
from .statistics import clustered_bootstrap_mean, clustered_bootstrap_paired_diff
from .utils import LABELS, ensure_parent, write_json, write_table
from .wrapper_audit import attach_wrapper_categories, audit_wrappers


def prepare_dataset(config: Mapping[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_normalized(str(config["dataset_path"]))
    wrapper_audit = audit_wrappers(df)
    df = attach_wrapper_categories(df, wrapper_audit)
    experiment_cfg = config.get("experiment", {})
    active_wrappers = experiment_cfg.get("active_wrappers")
    if active_wrappers:
        expected_wrappers = {str(wrapper) for wrapper in active_wrappers}
        df = df[df["wrapper_name"].isin(expected_wrappers)].copy()
        observed_wrappers = set(df["wrapper_name"].astype(str))
        if observed_wrappers != expected_wrappers:
            raise ValueError(
                f"Active wrapper contract mismatch: expected {sorted(expected_wrappers)}, "
                f"observed {sorted(observed_wrappers)}"
            )
    active_categories = set(experiment_cfg.get("wrapper_categories", ["pure_interface"]))
    df = df[df["wrapper_category"].isin(active_categories)].copy()
    if active_wrappers:
        categorized_wrappers = set(df["wrapper_name"].astype(str))
        expected_wrappers = {str(wrapper) for wrapper in active_wrappers}
        if categorized_wrappers != expected_wrappers:
            raise ValueError(
                "Configured active wrappers are not all in the active wrapper categories: "
                f"expected {sorted(expected_wrappers)}, observed {sorted(categorized_wrappers)}"
            )
    if df.empty:
        raise ValueError(f"No rows remain after filtering to wrapper categories {sorted(active_categories)}")
    split_cfg = config.get("splits", {})
    df = assign_splits_by_item(
        df,
        train_frac=float(split_cfg.get("train", 0.60)),
        validation_frac=float(split_cfg.get("validation", 0.20)),
        test_frac=float(split_cfg.get("test", 0.20)),
        seed=int(config.get("seed", 1729)),
    )
    assert_item_split_integrity(df)
    return df, wrapper_audit


def save_split_manifest(df: pd.DataFrame, output_path: str | Path) -> dict[str, object]:
    item_df = df.drop_duplicates("item_id")
    manifest = {
        "num_items": int(item_df["item_id"].nunique()),
        "num_prompts": int(len(df)),
        "source_split_counts": {k: int(v) for k, v in df["source_split"].value_counts(dropna=False).sort_index().items()},
        "split_item_counts": {k: int(v) for k, v in item_df["split"].value_counts().sort_index().items()},
        "split_prompt_counts": {k: int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "subject_by_split": {
            split: {subject: int(count) for subject, count in group["subject"].value_counts().sort_index().items()}
            for split, group in item_df.groupby("split")
        },
    }
    write_json(manifest, output_path)
    return manifest


def write_conflict_artifacts(scored: pd.DataFrame, processed_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(processed_dir)
    artifacts = {
        "pure_interface": out_dir / "conflict_pairs.parquet",
        "item_outcomes": out_dir / "item_outcomes.csv",
    }
    write_table(
        construct_conflict_pairs(scored, category="pure_interface", require_same_wrapper_calibration=True),
        artifacts["pure_interface"],
    )
    write_table(item_outcome_summary(scored), artifacts["item_outcomes"])
    return artifacts


def _collect_activations(
    model,
    tokenizer,
    pairs: pd.DataFrame,
    *,
    layer: int,
    anchor: str,
    split: str,
    limit: int | None = None,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    source = pairs[pairs["split"] == split].copy()
    if limit is not None:
        source = source.head(limit)
    clean_vectors: list[torch.Tensor] = []
    corrupt_vectors: list[torch.Tensor] = []
    meta_rows: list[dict[str, object]] = []
    for _, pair in source.iterrows():
        question = pair.get("question")
        choices = pair.get("choices")
        if resolve_anchor(tokenizer, str(pair["clean_prompt"]), anchor, question=question, choices=choices) is None:
            continue
        if resolve_anchor(tokenizer, str(pair["corrupt_prompt"]), anchor, question=question, choices=choices) is None:
            continue
        clean_vectors.append(
            hidden_state_at(
                model,
                tokenizer,
                str(pair["clean_prompt"]),
                layer=layer,
                anchor=anchor,
                question=question,
                choices=choices,
                device=device,
            )
        )
        corrupt_vectors.append(
            hidden_state_at(
                model,
                tokenizer,
                str(pair["corrupt_prompt"]),
                layer=layer,
                anchor=anchor,
                question=question,
                choices=choices,
                device=device,
            )
        )
        meta_rows.append(pair.to_dict())
    if not clean_vectors:
        raise ValueError(f"No conflict activations collected for split={split!r}, anchor={anchor!r}")
    return torch.stack(clean_vectors), torch.stack(corrupt_vectors), pd.DataFrame(meta_rows)


def train_access_vector(
    model,
    tokenizer,
    pairs: pd.DataFrame,
    *,
    layer: int,
    anchor: str,
    split: str = "train",
    balance_by: str = "label",
    limit: int | None = None,
    device=None,
) -> dict[str, object]:
    clean, corrupt, meta = _collect_activations(
        model, tokenizer, pairs, layer=layer, anchor=anchor, split=split, limit=limit, device=device
    )
    labels = list(meta["correct_label"].astype(str))
    secondary = None
    if balance_by == "label_subject":
        secondary = list(meta["subject"].astype(str))
    elif balance_by != "label":
        raise ValueError("balance_by must be 'label' or 'label_subject'")
    vector = build_access_vector(clean, corrupt, labels, secondary_groups=secondary)
    shuffled_vector = None
    shuffled_error = None
    try:
        shuffled_vector = shuffled_pair_vector(clean, corrupt, labels, secondary_groups=secondary, seed=1729)
    except ValueError as exc:
        shuffled_error = str(exc)
    composition = {
        "labels": {label: int(count) for label, count in meta["correct_label"].value_counts().sort_index().items()},
        "subjects": {subject: int(count) for subject, count in meta["subject"].value_counts().sort_index().items()},
        "clean_wrappers": {w: int(count) for w, count in meta["clean_wrapper"].value_counts().sort_index().items()},
        "corrupt_wrappers": {w: int(count) for w, count in meta["corrupt_wrapper"].value_counts().sort_index().items()},
    }
    missing_label_groups = [label for label in LABELS if label not in composition["labels"]]
    return {
        "vector": vector,
        "shuffled_vector": shuffled_vector,
        "metadata": {
            "layer": int(layer),
            "anchor": anchor,
            "split": split,
            "vector_kind": "interface_formatting_study_primary",
            "n_pairs": int(len(meta)),
            "balance_by": balance_by,
            "vector_norm": float(torch.linalg.vector_norm(vector)),
            "composition": composition,
            "missing_label_groups": missing_label_groups,
            "balance_limitation": (
                "all labels represented"
                if not missing_label_groups
                else "label balancing used only observed labels; interpret vector as under-balanced"
            ),
            "shuffled_control_available": shuffled_vector is not None,
            "shuffled_control_error": shuffled_error,
            "shuffled_control_seed": 1729,
            "shuffled_control_balance_by": balance_by,
        },
    }


def save_access_vector(payload: Mapping[str, object], path: str | Path) -> None:
    out = ensure_parent(path)
    torch.save({"vector": payload["vector"], "metadata": payload["metadata"]}, out)


def save_optional_shuffled_vector(payload: Mapping[str, object], path: str | Path) -> bool:
    vector = payload.get("shuffled_vector")
    if vector is None:
        return False
    out = ensure_parent(path)
    metadata = dict(payload["metadata"])
    metadata["vector_kind"] = "shuffled_pair_control"
    metadata["control_kind"] = "shuffled_pair"
    torch.save({"vector": vector, "metadata": metadata}, out)
    return True


def load_access_vector(path: str | Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = torch.load(path, map_location="cpu")
    return payload["vector"].float(), dict(payload.get("metadata", {}))


def _evaluate_pairs_with_vector(
    model,
    tokenizer,
    pairs: pd.DataFrame,
    *,
    vector: torch.Tensor,
    layer: int,
    anchor: str,
    alpha: float,
    split: str,
    condition: str,
    limit: int | None = None,
    batch_size: int = 40,
    device=None,
) -> pd.DataFrame:
    source = pairs[pairs["split"] == split].copy()
    if limit is not None:
        source = source.head(limit)
    rows: list[dict[str, object]] = []
    for _, pair in source.iterrows():
        row = pair.to_dict()
        if resolve_anchor(tokenizer, str(row["corrupt_prompt"]), anchor) is None:
            continue
        scored = score_prompt_with_optional_edit(
            model,
            tokenizer,
            str(row["corrupt_prompt"]),
            correct_label=str(row["correct_label"]),
            bias_scores=bias_scores_from_row(row, prefix="corrupt"),
            layer=layer,
            anchor=anchor,
            edit_fn=add_vector(vector, alpha),
            batch_size=batch_size,
            device=device,
        )
        rows.append(
            {
                "item_id": row["item_id"],
                "subject": row.get("subject"),
                "split": row["split"],
                "condition": condition,
                "alpha": float(alpha),
                "baseline_margin": float(row["corrupt_margin"]),
                "intervention_margin": float(scored["cal_margin"]),
                "margin_improvement": float(scored["cal_margin"]) - float(row["corrupt_margin"]),
                "baseline_correct": False,
                "intervention_correct": bool(scored["cal_correct"]),
                "repair": bool(scored["cal_correct"]),
                "pred_label": scored["cal_pred_label"],
            }
        )
    return pd.DataFrame(rows)


def tune_alpha(
    model,
    tokenizer,
    validation_pairs: pd.DataFrame,
    *,
    vector: torch.Tensor,
    layer: int,
    anchor: str,
    alpha_grid: Iterable[float],
    batch_size: int = 40,
    limit: int | None = None,
    device=None,
) -> dict[str, object]:
    alpha_results: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for alpha in alpha_grid:
        df = _evaluate_pairs_with_vector(
            model,
            tokenizer,
            validation_pairs,
            vector=vector,
            layer=layer,
            anchor=anchor,
            alpha=float(alpha),
            split="validation",
            condition="interface_formatting_study_positive",
            limit=limit,
            batch_size=batch_size,
            device=device,
        )
        detail_frames.append(df)
        alpha_results.append(
            {
                "alpha": float(alpha),
                "mean_margin_improvement": float(df["margin_improvement"].mean()) if not df.empty else float("-inf"),
                "repair_rate": float(df["repair"].mean()) if not df.empty else 0.0,
                "n_items": int(df["item_id"].nunique()) if not df.empty else 0,
            }
        )
    summary = pd.DataFrame(alpha_results).sort_values("mean_margin_improvement", ascending=False)
    if summary.empty:
        raise ValueError("No validation alpha results were produced")
    positive_summary = summary[summary["alpha"] > 0].copy()
    if positive_summary.empty:
        raise ValueError("Alpha grid must include positive alphas for the primary repair intervention")
    best = positive_summary.iloc[0]
    best_value = float(best["mean_margin_improvement"])
    tolerance = abs(best_value) * 0.05
    near = positive_summary[positive_summary["mean_margin_improvement"] >= best_value - tolerance].copy()
    near["abs_alpha"] = near["alpha"].abs()
    chosen = near.sort_values(["abs_alpha", "alpha"]).iloc[0]
    overall_best = summary.iloc[0]
    return {
        "selected_alpha": float(chosen["alpha"]),
        "selection_split": "validation",
        "selection_metric": "mean_calibrated_margin_improvement_positive_alpha_only",
        "positive_alpha_improved": best_value > 0.0,
        "best_positive_margin_improvement": best_value,
        "overall_best_alpha_diagnostic": float(overall_best["alpha"]),
        "overall_best_margin_improvement_diagnostic": float(overall_best["mean_margin_improvement"]),
        "claim_warning": (
            None
            if best_value > 0.0
            else "No positive alpha improved validation margin; reusable repair claim should be downgraded."
        ),
        "alpha_summary": alpha_results,
        "details": pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(),
    }


def evaluate_access_vector_and_controls(
    model,
    tokenizer,
    pairs: pd.DataFrame,
    *,
    vector: torch.Tensor,
    layer: int,
    anchor: str,
    alpha: float,
    split: str = "test",
    batch_size: int = 40,
    random_seeds: Iterable[int] = range(5),
    shuffled_vector: torch.Tensor | None = None,
    limit: int | None = None,
    device=None,
) -> pd.DataFrame:
    frames = [
        _evaluate_pairs_with_vector(
            model,
            tokenizer,
            pairs,
            vector=vector,
            layer=layer,
            anchor=anchor,
            alpha=alpha,
            split=split,
            condition="interface_formatting_study_positive",
            limit=limit,
            batch_size=batch_size,
            device=device,
        ),
        _evaluate_pairs_with_vector(
            model,
            tokenizer,
            pairs,
            vector=-vector,
            layer=layer,
            anchor=anchor,
            alpha=alpha,
            split=split,
            condition="wrong_direction",
            limit=limit,
            batch_size=batch_size,
            device=device,
        ),
    ]
    for seed in random_seeds:
        frames.append(
            _evaluate_pairs_with_vector(
                model,
                tokenizer,
                pairs,
                vector=random_vector_like(vector, seed=int(seed)),
                layer=layer,
                anchor=anchor,
                alpha=alpha,
                split=split,
                condition=f"random_seed_{seed}",
                limit=limit,
                batch_size=batch_size,
                device=device,
            )
        )
    if shuffled_vector is not None:
        frames.append(
            _evaluate_pairs_with_vector(
                model,
                tokenizer,
                pairs,
                vector=shuffled_vector,
                layer=layer,
                anchor=anchor,
                alpha=alpha,
                split=split,
                condition="shuffled_pair_vector",
                limit=limit,
                batch_size=batch_size,
                device=device,
            )
        )
    return pd.concat(frames, ignore_index=True)


def evaluate_clean_removal(
    model,
    tokenizer,
    pairs: pd.DataFrame,
    *,
    vector: torch.Tensor,
    layer: int,
    anchor: str,
    alpha: float,
    split: str = "test",
    batch_size: int = 40,
    limit: int | None = None,
    device=None,
) -> pd.DataFrame:
    source = pairs[pairs["split"] == split].copy()
    if limit is not None:
        source = source.head(limit)
    rows: list[dict[str, object]] = []
    conditions = {
        "negative_direction": add_vector(vector, -alpha),
        "projection_removal": remove_projection(vector),
    }
    for _, pair in source.iterrows():
        row = pair.to_dict()
        if resolve_anchor(tokenizer, str(row["clean_prompt"]), anchor) is None:
            continue
        for condition, edit_fn in conditions.items():
            scored = score_prompt_with_optional_edit(
                model,
                tokenizer,
                str(row["clean_prompt"]),
                correct_label=str(row["correct_label"]),
                bias_scores=bias_scores_from_row(row, prefix="clean"),
                layer=layer,
                anchor=anchor,
                edit_fn=edit_fn,
                batch_size=batch_size,
                device=device,
            )
            rows.append(
                {
                    "item_id": row["item_id"],
                    "subject": row.get("subject"),
                    "split": row["split"],
                    "condition": condition,
                    "baseline_margin": float(row["clean_margin"]),
                    "intervention_margin": float(scored["cal_margin"]),
                    "margin_drop": float(row["clean_margin"]) - float(scored["cal_margin"]),
                    "baseline_correct": True,
                    "intervention_correct": bool(scored["cal_correct"]),
                    "damage": not bool(scored["cal_correct"]),
                    "pred_label": scored["cal_pred_label"],
                }
            )
    return pd.DataFrame(rows)


def evaluate_content_free_control(
    model,
    tokenizer,
    scored_rows: pd.DataFrame,
    *,
    vector: torch.Tensor,
    layer: int,
    anchor: str,
    alpha: float,
    split: str = "test",
    category: str = "pure_interface",
    limit: int | None = None,
    batch_size: int = 40,
    device=None,
) -> pd.DataFrame:
    source = scored_rows[(scored_rows["split"] == split) & (scored_rows["wrapper_category"] == category)].copy()
    if "content_free_calibration_kind" in source.columns:
        source = source[source["content_free_calibration_kind"] == SAME_WRAPPER_REDACTION].copy()
    if limit is not None:
        source = source.head(limit)
    rows: list[dict[str, object]] = []
    for _, original in source.iterrows():
        row = original.to_dict()
        prompt = str(row.get("content_free_prompt") or make_content_free_prompt(row))
        if resolve_anchor(tokenizer, prompt, anchor) is None:
            continue
        bias = {label: 0.0 for label in LABELS}
        scored = score_prompt_with_optional_edit(
            model,
            tokenizer,
            prompt,
            correct_label=str(row["correct_label"]),
            bias_scores=bias,
            layer=layer,
            anchor=anchor,
            edit_fn=add_vector(vector, alpha),
            batch_size=batch_size,
            device=device,
        )
        rows.append(
            {
                "item_id": row["item_id"],
                "wrapper_name": row["wrapper_name"],
                "condition": "content_free_interface_formatting_study",
                "pred_label": scored["cal_pred_label"],
                "correct_label": row["correct_label"],
                "cal_margin": scored["cal_margin"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_item_outcome_subset_control(
    model,
    tokenizer,
    scored_rows: pd.DataFrame,
    *,
    vector: torch.Tensor,
    layer: int,
    anchor: str,
    alpha: float,
    outcome: str,
    split: str = "test",
    category: str = "pure_interface",
    limit: int | None = None,
    batch_size: int = 40,
    device=None,
) -> pd.DataFrame:
    if outcome not in {"all_wrong", "all_correct"}:
        raise ValueError("outcome must be 'all_wrong' or 'all_correct'")
    source = scored_rows[(scored_rows["split"] == split) & (scored_rows["wrapper_category"] == category)].copy()
    if "content_free_calibration_kind" in source.columns:
        source = source[source["content_free_calibration_kind"] == SAME_WRAPPER_REDACTION].copy()
    summary = item_outcome_summary(source)
    item_ids = set(summary.loc[summary[outcome], "item_id"])
    source = source[source["item_id"].isin(item_ids)].copy()
    if limit is not None:
        source = source.head(limit)
    rows: list[dict[str, object]] = []
    for _, original in source.iterrows():
        row = original.to_dict()
        prompt = str(row["wrapped_prompt"])
        if resolve_anchor(tokenizer, prompt, anchor) is None:
            continue
        scored = score_prompt_with_optional_edit(
            model,
            tokenizer,
            prompt,
            correct_label=str(row["correct_label"]),
            bias_scores=bias_scores_from_row(row),
            layer=layer,
            anchor=anchor,
            edit_fn=add_vector(vector, alpha),
            batch_size=batch_size,
            device=device,
        )
        baseline_correct = bool(row["cal_correct"])
        intervention_correct = bool(scored["cal_correct"])
        rows.append(
            {
                "item_id": row["item_id"],
                "wrapper_name": row["wrapper_name"],
                "subject": row.get("subject"),
                "split": row["split"],
                "condition": f"{outcome}_interface_formatting_study",
                "alpha": float(alpha),
                "baseline_margin": float(row["cal_margin"]),
                "intervention_margin": float(scored["cal_margin"]),
                "margin_improvement": float(scored["cal_margin"]) - float(row["cal_margin"]),
                "baseline_correct": baseline_correct,
                "intervention_correct": intervention_correct,
                "repair": (not baseline_correct) and intervention_correct,
                "damage": baseline_correct and not intervention_correct,
                "pred_label": scored["cal_pred_label"],
            }
        )
    return pd.DataFrame(rows)


def summarize_intervention_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, group in df.groupby("condition", sort=True):
        margin_stats = clustered_bootstrap_mean(group, "margin_improvement") if "margin_improvement" in group else {}
        repair_rate = float(group["repair"].mean()) if "repair" in group else float("nan")
        rows.append(
            {
                "condition": condition,
                "mean_margin_improvement": margin_stats.get("mean", float("nan")),
                "ci_low": margin_stats.get("ci_low", float("nan")),
                "ci_high": margin_stats.get("ci_high", float("nan")),
                "repair_rate": repair_rate,
                "n_items": int(group["item_id"].nunique()) if "item_id" in group else 0,
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)
