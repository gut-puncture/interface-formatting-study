from __future__ import annotations

import pandas as pd

from interface_formatting_study.conflicts import construct_all_pairs_weighted, construct_conflict_pairs, item_outcome_summary


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
