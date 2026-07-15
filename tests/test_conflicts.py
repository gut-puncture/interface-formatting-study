from __future__ import annotations

import pandas as pd
import pytest

from interface_formatting_study.conflicts import (
    conflict_population_summary,
    construct_all_pairs_weighted,
    construct_conflict_pairs,
    item_conflict_outcomes,
    item_outcome_summary,
)


def _rows(item_id: str, margins, corrects):
    return [
        {
            "item_id": item_id,
            "subject": "math",
            "split": "train",
            "wrapper_name": f"w{i}",
            "wrapper_category": "pure_interface",
            "question": "What is 1+1?",
            "choices": ["1", "2", "3", "4"],
            "cal_margin": margin,
            "cal_correct": correct,
            "wrapped_prompt": f"prompt-{i}",
            "correct_label": "A",
            "cal_pred_label": "A" if correct else "B",
            "content_free_calibration_kind": "same_wrapper_redaction",
        }
        for i, (margin, correct) in enumerate(zip(margins, corrects, strict=True))
    ]


def test_canonical_conflict_pair_uses_max_clean_and_min_corrupt():
    df = pd.DataFrame(_rows("i1", [0.1, 2.5, -0.2, -3.0, 1.0], [True, True, False, False, True]))
    pairs = construct_conflict_pairs(df)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["clean_wrapper"] == "w1"
    assert row["corrupt_wrapper"] == "w3"
    assert row["clean_margin"] == 2.5
    assert row["corrupt_margin"] == -3.0
    assert row["question"] == "What is 1+1?"
    assert row["choices"] == ["1", "2", "3", "4"]


def test_all_correct_and_all_wrong_do_not_emit_conflict_pairs():
    df = pd.DataFrame(
        _rows("all-correct", [1.0, 0.2], [True, True])
        + _rows("all-wrong", [-1.0, -0.2], [False, False])
    )
    assert construct_conflict_pairs(df).empty
    summary = item_outcome_summary(df)
    flags = summary.set_index("item_id")
    assert bool(flags.loc["all-correct", "all_correct"])
    assert bool(flags.loc["all-wrong", "all_wrong"])


def test_all_pairs_secondary_mode_weights_each_item_to_one():
    df = pd.DataFrame(_rows("i1", [1.0, 2.0, -1.0, -2.0, -3.0], [True, True, False, False, False]))
    pairs = construct_all_pairs_weighted(df)
    assert len(pairs) == 6
    assert pairs["pair_weight"].sum() == 1.0


def test_primary_conflicts_require_same_wrapper_calibration():
    df = pd.DataFrame(_rows("i1", [1.0, -1.0], [True, False]))
    df.loc[df["cal_correct"] == False, "content_free_calibration_kind"] = "generic_fallback"

    pairs = construct_conflict_pairs(df, category="pure_interface", require_same_wrapper_calibration=True)

    assert pairs.empty


def test_full_and_calibration_eligible_conflicts_have_explicit_denominators():
    frame = pd.DataFrame(
        _rows("mixed", [1.0, -1.0], [True, False])
        + _rows("stable", [1.0, 0.5], [True, True])
    )
    frame.loc[
        (frame["item_id"] == "mixed") & (frame["wrapper_name"] == "w1"),
        "content_free_calibration_kind",
    ] = "generic_fallback"

    full = item_conflict_outcomes(frame, expected_wrappers=2)
    eligible = item_conflict_outcomes(
        frame,
        calibration_kind="same_wrapper_redaction",
    )

    assert conflict_population_summary(full, population="full") == {
        "population": "full",
        "eligible_items": 2,
        "conflict_items": 1,
        "denominator": 2,
        "conflict_rate": 0.5,
    }
    assert conflict_population_summary(eligible, population="calibration_eligible") == {
        "population": "calibration_eligible",
        "eligible_items": 2,
        "conflict_items": 0,
        "denominator": 2,
        "conflict_rate": 0.0,
    }


def test_full_conflict_population_rejects_missing_or_duplicate_wrappers():
    incomplete = pd.DataFrame(_rows("item", [1.0], [True]))
    with pytest.raises(ValueError, match="expected 2"):
        item_conflict_outcomes(incomplete, expected_wrappers=2)

    duplicated = pd.concat([incomplete, incomplete], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate item-wrapper"):
        item_conflict_outcomes(duplicated, expected_wrappers=2)

    missing_outcome = incomplete.copy()
    missing_outcome["cal_correct"] = missing_outcome["cal_correct"].astype(object)
    missing_outcome.loc[0, "cal_correct"] = None
    with pytest.raises(ValueError, match="missing cal_correct"):
        item_conflict_outcomes(missing_outcome)
