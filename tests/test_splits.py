from __future__ import annotations

import pandas as pd

from interface_formatting_study.splits import assign_splits_by_item, assert_item_split_integrity, split_subject_proportions


def _split_df(n_items=100):
    rows = []
    for i in range(n_items):
        subject = "rare" if i < 10 else "common"
        for wrapper in range(20):
            rows.append({"item_id": f"item-{i}", "subject": subject, "wrapper_name": f"w{wrapper}", "source_split": "test"})
    return pd.DataFrame(rows)


def test_split_ignores_source_split_and_keeps_wrappers_together():
    df = assign_splits_by_item(_split_df(), seed=123)
    assert set(df["split"]) == {"train", "validation", "test"}
    assert_item_split_integrity(df)
    per_item_counts = df.groupby("item_id")["wrapper_name"].nunique()
    assert (per_item_counts == 20).all()


def test_subject_stratification_is_approximately_preserved():
    df = assign_splits_by_item(_split_df(200), seed=123)
    props = split_subject_proportions(df)
    rare_props = props.loc["rare"]
    assert rare_props.max() - rare_props.min() < 0.05

