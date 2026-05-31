from __future__ import annotations

import pandas as pd

from interface_formatting_study.statistics import benjamini_hochberg, clustered_bootstrap_mean, clustered_bootstrap_paired_diff


def test_clustered_bootstrap_reports_item_clusters_not_rows():
    df = pd.DataFrame({"item_id": ["i1", "i1", "i2"], "value": [1.0, 3.0, 10.0]})
    result = clustered_bootstrap_mean(df, "value", n_boot=20, seed=0)
    assert result["mean"] == 6.0
    assert result["n_clusters"] == 2
    assert result["n_rows"] == 3


def test_paired_diff_is_after_minus_before():
    df = pd.DataFrame({"item_id": ["i1", "i2"], "before": [1.0, 2.0], "after": [3.0, 5.0]})
    result = clustered_bootstrap_paired_diff(df, "before", "after", n_boot=20, seed=0)
    assert result["mean"] == 2.5


def test_benjamini_hochberg_monotone_adjustment():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1]
