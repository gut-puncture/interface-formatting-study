from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

from .utils import ensure_dir, ensure_parent


def _prepare_plot_environment(output_path: str | Path) -> None:
    output = Path(output_path)
    results_dir = output.parents[1] if len(output.parents) > 1 else output.parent
    cache_dir = ensure_dir(results_dir / "metadata" / "plot_cache")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def wrapper_accuracy_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    _prepare_plot_environment(output_path)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = (
        df.groupby(["wrapper_name", "wrapper_category"])
        .agg(
            raw_accuracy=("raw_correct", "mean"),
            calibrated_accuracy=("cal_correct", "mean"),
            mean_calibrated_margin=("cal_margin", "mean"),
        )
        .reset_index()
        .set_index("wrapper_name")
    )
    matrix = metrics[["raw_accuracy", "calibrated_accuracy", "mean_calibrated_margin"]]
    plt.figure(figsize=(8, max(4, 0.32 * len(matrix))))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="viridis")
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()


def conflict_histogram(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    _prepare_plot_environment(output_path)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    summary_df["n_correct"].hist(bins=range(0, int(summary_df["n_wrappers"].max()) + 2), align="left")
    plt.xlabel("Correct wrappers per item")
    plt.ylabel("Number of items")
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()


def patching_heatmap(results: pd.DataFrame, output_path: str | Path) -> None:
    _prepare_plot_environment(output_path)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = results.pivot_table(index="anchor", columns="layer", values="margin_improvement", aggfunc="mean")
    plt.figure(figsize=(10, 3 + len(matrix)))
    sns.heatmap(matrix, cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(ensure_parent(output_path), dpi=200)
    plt.close()
