#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from interface_formatting_study.experiment import prepare_dataset
from interface_formatting_study.utils import read_yaml
from interface_formatting_study.wrapper_row_audit import build_audit_packets, merge_audit_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or merge the one-time row-level wrapper audit")
    parser.add_argument("command", choices=("prepare", "merge"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=600_000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.command == "prepare":
        frame, _ = prepare_dataset(read_yaml(args.config))
        result = build_audit_packets(
            frame,
            output_dir,
            max_rows=args.max_rows,
            max_chars=args.max_chars,
        )
    else:
        merged = merge_audit_labels(output_dir)
        result = {"rows": len(merged), "output": str(output_dir / "wrapper_audit_labels.parquet")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
