#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from interface_formatting_study.causal_design import build_causal_design_v3
from interface_formatting_study.causal_option_audit import (
    build_causal_option_packets,
    load_causal_option_annotations,
)
from interface_formatting_study.experiment import prepare_dataset
from interface_formatting_study.utils import read_yaml


def _canary(status, rows: int):
    unresolved = status[
        ~status["position_applicable"].astype(bool)
        | ~status["label_applicable"].astype(bool)
    ].copy()
    unresolved["order"] = unresolved.apply(
        lambda row: hashlib.sha256(
            f"{row['item_id']}::{row['wrapper_name']}".encode()
        ).hexdigest(),
        axis=1,
    )
    groups = {
        name: group.sort_values("order", kind="mergesort").to_dict("records")
        for name, group in unresolved.groupby("wrapper_name", sort=True)
    }
    selected = []
    while len(selected) < rows and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(selected) < rows:
                selected.append(groups[name].pop(0))
    columns = unresolved.drop(columns="order").columns
    return pd.DataFrame(selected, columns=[*columns, "order"]).drop(columns="order")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or validate the causal option-map audit")
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=15)
    parser.add_argument("--max-chars", type=int, default=75_000)
    parser.add_argument("--canary-rows", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.command == "validate":
        annotations = load_causal_option_annotations(output_dir)
        result = {"annotations": len(annotations), "output_dir": str(output_dir)}
    else:
        frame, _ = prepare_dataset(read_yaml(args.config))
        _, status = build_causal_design_v3(frame, splits=("train", "validation"))
        if args.canary_rows:
            status = _canary(status, args.canary_rows)
        result = build_causal_option_packets(
            frame,
            status,
            output_dir,
            max_rows=args.max_rows,
            max_chars=args.max_chars,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
