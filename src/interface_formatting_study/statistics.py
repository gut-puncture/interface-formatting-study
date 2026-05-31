from __future__ import annotations

import numpy as np
import pandas as pd


def clustered_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    *,
    cluster_col: str = "item_id",
    n_boot: int = 1000,
    seed: int = 1729,
) -> dict[str, float | int]:
    if df.empty:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_clusters": 0, "n_rows": 0}
    rng = np.random.default_rng(seed)
    per_cluster = df.groupby(cluster_col)[value_col].mean().astype(float)
    clusters = per_cluster.index.to_numpy()
    grouped = per_cluster.to_dict()
    estimates = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.array([grouped[cluster] for cluster in sampled], dtype=float)
        estimates.append(float(np.mean(values)))
    return {
        "mean": float(per_cluster.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "n_clusters": int(len(clusters)),
        "n_rows": int(len(df)),
    }


def clustered_bootstrap_paired_diff(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    *,
    cluster_col: str = "item_id",
    n_boot: int = 1000,
    seed: int = 1729,
) -> dict[str, float | int]:
    tmp = df.copy()
    tmp["paired_diff"] = tmp[after_col].astype(float) - tmp[before_col].astype(float)
    return clustered_bootstrap_mean(tmp, "paired_diff", cluster_col=cluster_col, n_boot=n_boot, seed=seed)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        running = min(running, p[idx] * n / original_rank)
        adjusted[idx] = running
    return adjusted.tolist()
