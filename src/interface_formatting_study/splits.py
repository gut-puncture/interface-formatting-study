from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def assign_splits_by_item(
    df: pd.DataFrame,
    *,
    item_col: str = "item_id",
    subject_col: str = "subject",
    train_frac: float = 0.60,
    validation_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 1729,
) -> pd.DataFrame:
    total = train_frac + validation_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    item_subject = df[[item_col, subject_col]].drop_duplicates(item_col)
    rng = np.random.default_rng(seed)
    mapping: dict[str, str] = {}

    for _, group in item_subject.groupby(subject_col, dropna=False, sort=True):
        items = list(group[item_col].astype(str))
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_validation = int(round(n * validation_frac))
        if n_train + n_validation > n:
            n_validation = max(0, n - n_train)
        for item in items[:n_train]:
            mapping[item] = "train"
        for item in items[n_train : n_train + n_validation]:
            mapping[item] = "validation"
        for item in items[n_train + n_validation :]:
            mapping[item] = "test"

    out = df.copy()
    out["split"] = out[item_col].astype(str).map(mapping)
    if out["split"].isna().any():
        raise AssertionError("Internal split assignment left items unassigned")
    return out


def assert_item_split_integrity(df: pd.DataFrame, *, item_col: str = "item_id", split_col: str = "split") -> None:
    counts = df.groupby(item_col)[split_col].nunique()
    bad = counts[counts > 1]
    if not bad.empty:
        raise AssertionError(f"Items appear in multiple splits: {bad.index[:10].tolist()}")


def split_subject_proportions(df: pd.DataFrame, *, subject_col: str = "subject", split_col: str = "split") -> pd.DataFrame:
    counts = defaultdict(dict)
    for split, group in df.drop_duplicates(["item_id"]).groupby(split_col):
        props = group[subject_col].value_counts(normalize=True)
        for subject, value in props.items():
            counts[subject][split] = float(value)
    return pd.DataFrame.from_dict(counts, orient="index").fillna(0.0)

