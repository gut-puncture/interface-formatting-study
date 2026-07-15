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


def item_conflict_outcomes(
    df: pd.DataFrame,
    *,
    correct_col: str = "cal_correct",
    calibration_kind: str | None = None,
    expected_wrappers: int | None = None,
) -> pd.DataFrame:
    """Return one denominator-safe conflict row per eligible item."""

    required = {"item_id", "wrapper_name", correct_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Conflict population is missing columns: {sorted(missing)}")
    source = df
    if calibration_kind is not None:
        if "content_free_calibration_kind" not in source:
            raise ValueError("Calibration-eligible conflicts require calibration provenance")
        source = source[source["content_free_calibration_kind"] == calibration_kind]
    if source.empty:
        return pd.DataFrame(columns=["item_id", "observed_wrappers", "correct_wrappers", "is_conflict"])

    if source[correct_col].isna().any():
        raise ValueError(f"Conflict population contains missing {correct_col} values")
    invalid_outcomes = ~source[correct_col].isin([True, False])
    if invalid_outcomes.any():
        sample = source.loc[invalid_outcomes, correct_col].head(5).tolist()
        raise ValueError(f"Conflict population requires boolean {correct_col} values; found {sample}")

    duplicate = source.duplicated(["item_id", "wrapper_name"], keep=False)
    if duplicate.any():
        sample = source.loc[duplicate, ["item_id", "wrapper_name"]].head(5).to_dict("records")
        raise ValueError(f"Conflict population contains duplicate item-wrapper rows: {sample}")

    grouped = source.groupby("item_id", sort=True)
    observed = grouped["wrapper_name"].size()
    if expected_wrappers is not None:
        bad = observed[observed != int(expected_wrappers)]
        if not bad.empty:
            raise ValueError(
                f"Full conflict population expected {expected_wrappers} wrappers per item; "
                f"found {bad.head().to_dict()}"
            )
    correct = grouped[correct_col].sum().astype(int)
    outcomes = pd.DataFrame(
        {
            "item_id": observed.index.astype(str),
            "observed_wrappers": observed.to_numpy(dtype=int),
            "correct_wrappers": correct.to_numpy(dtype=int),
        }
    )
    outcomes["is_conflict"] = (
        (outcomes["correct_wrappers"] > 0)
        & (outcomes["correct_wrappers"] < outcomes["observed_wrappers"])
    )
    return outcomes


def conflict_population_summary(outcomes: pd.DataFrame, *, population: str) -> dict[str, object]:
    required = {"item_id", "is_conflict"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Item outcomes are missing columns: {sorted(missing)}")
    denominator = int(len(outcomes))
    conflicts = int(outcomes["is_conflict"].astype(bool).sum())
    return {
        "population": str(population),
        "eligible_items": denominator,
        "conflict_items": conflicts,
        "denominator": denominator,
        "conflict_rate": 0.0 if denominator == 0 else conflicts / denominator,
    }


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
