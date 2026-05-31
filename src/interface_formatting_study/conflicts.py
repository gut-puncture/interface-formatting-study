from __future__ import annotations

import pandas as pd

from .calibration import SAME_WRAPPER_REDACTION
from .utils import LABELS

BASE_CONFLICT_COLUMNS = [
    "item_id",
    "subject",
    "split",
    "question",
    "choices",
    "wrapper_category_subset",
    "clean_wrapper",
    "corrupt_wrapper",
    "clean_margin",
    "corrupt_margin",
    "clean_prompt",
    "corrupt_prompt",
    "correct_label",
    "clean_pred_label",
    "corrupt_pred_label",
    "clean_content_free_calibration_kind",
    "corrupt_content_free_calibration_kind",
]
SCORE_CONFLICT_COLUMNS = [
    f"{prefix}_{score_kind}_{label}"
    for label in LABELS
    for prefix in ("clean", "corrupt")
    for score_kind in ("raw_score", "bias_score", "cal_score")
]
CONFLICT_PAIR_COLUMNS = BASE_CONFLICT_COLUMNS + SCORE_CONFLICT_COLUMNS


def construct_conflict_pairs(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    correct_col: str = "cal_correct",
    margin_col: str = "cal_margin",
    category: str | None = None,
    require_same_wrapper_calibration: bool = False,
) -> pd.DataFrame:
    source = df
    if category is not None:
        source = source[source["wrapper_category"] == category]
    if require_same_wrapper_calibration:
        if "content_free_calibration_kind" not in source.columns:
            raise ValueError("Primary conflicts require content_free_calibration_kind provenance")
        source = source[source["content_free_calibration_kind"] == SAME_WRAPPER_REDACTION]

    rows: list[dict[str, object]] = []
    for item_id, group in source.groupby(item_col, sort=True):
        clean_candidates = group[group[correct_col].astype(bool)]
        corrupt_candidates = group[~group[correct_col].astype(bool)]
        if clean_candidates.empty or corrupt_candidates.empty:
            continue
        clean = clean_candidates.sort_values(margin_col, ascending=False).iloc[0]
        corrupt = corrupt_candidates.sort_values(margin_col, ascending=True).iloc[0]
        row = {
            "item_id": item_id,
            "subject": clean.get("subject"),
            "split": clean.get("split"),
            "question": clean.get("question"),
            "choices": clean.get("choices"),
            "wrapper_category_subset": category or "all",
            "clean_wrapper": clean["wrapper_name"],
            "corrupt_wrapper": corrupt["wrapper_name"],
            "clean_margin": float(clean[margin_col]),
            "corrupt_margin": float(corrupt[margin_col]),
            "clean_prompt": clean["wrapped_prompt"],
            "corrupt_prompt": corrupt["wrapped_prompt"],
            "correct_label": clean["correct_label"],
            "clean_pred_label": clean.get("cal_pred_label"),
            "corrupt_pred_label": corrupt.get("cal_pred_label"),
            "clean_content_free_calibration_kind": clean.get("content_free_calibration_kind"),
            "corrupt_content_free_calibration_kind": corrupt.get("content_free_calibration_kind"),
        }
        for label in LABELS:
            for prefix, selected in (("clean", clean), ("corrupt", corrupt)):
                for score_kind in ("raw_score", "bias_score", "cal_score"):
                    column = f"{score_kind}_{label}"
                    if column in selected:
                        row[f"{prefix}_{column}"] = float(selected[column])
        rows.append(row)
    return pd.DataFrame(rows, columns=CONFLICT_PAIR_COLUMNS)


def construct_all_pairs_weighted(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    correct_col: str = "cal_correct",
    margin_col: str = "cal_margin",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item_id, group in df.groupby(item_col, sort=True):
        clean_rows = group[group[correct_col].astype(bool)]
        corrupt_rows = group[~group[correct_col].astype(bool)]
        n_pairs = len(clean_rows) * len(corrupt_rows)
        if n_pairs == 0:
            continue
        base_weight = 1.0 / n_pairs
        emitted = 0
        for _, clean in clean_rows.iterrows():
            for _, corrupt in corrupt_rows.iterrows():
                emitted += 1
                weight = 1.0 - base_weight * (n_pairs - 1) if emitted == n_pairs else base_weight
                rows.append(
                    {
                        "item_id": item_id,
                        "clean_wrapper": clean["wrapper_name"],
                        "corrupt_wrapper": corrupt["wrapper_name"],
                        "clean_margin": float(clean[margin_col]),
                        "corrupt_margin": float(corrupt[margin_col]),
                        "pair_weight": weight,
                    }
                )
    return pd.DataFrame(rows)


def item_outcome_summary(df: pd.DataFrame, *, item_col: str = "item_id", correct_col: str = "cal_correct") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item_id, group in df.groupby(item_col, sort=True):
        n = len(group)
        n_correct = int(group[correct_col].astype(bool).sum())
        rows.append(
            {
                "item_id": item_id,
                "n_wrappers": n,
                "n_correct": n_correct,
                "is_conflict": 0 < n_correct < n,
                "all_correct": n_correct == n,
                "all_wrong": n_correct == 0,
            }
        )
    return pd.DataFrame(rows)
