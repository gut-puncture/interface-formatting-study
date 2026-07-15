from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

LABELS: tuple[str, ...] = ("A", "B", "C", "D")
EPSILON = 1e-6


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(obj: Any, path: str | Path) -> None:
    out = ensure_parent(path)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    out = ensure_parent(path)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out, index=False)
    elif suffix == ".csv":
        df.to_csv(out, index=False)
    elif suffix == ".jsonl":
        df.to_json(out, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported table suffix for {out}")


def write_table_atomic(df: pd.DataFrame, path: str | Path) -> None:
    destination = ensure_parent(path)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp{destination.suffix}")
    try:
        write_table(df, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_table(path: str | Path) -> pd.DataFrame:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(src)
    if suffix == ".csv":
        return pd.read_csv(src)
    if suffix == ".jsonl":
        return pd.read_json(src, orient="records", lines=True)
    raise ValueError(f"Unsupported table suffix for {src}")


def stable_group_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def label_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{label}" for label in LABELS]


def validate_label(label: str, *, name: str = "label") -> str:
    normalized = str(label).strip().upper()
    if normalized not in LABELS:
        raise ValueError(f"{name} must be one of {LABELS}, got {label!r}")
    return normalized
