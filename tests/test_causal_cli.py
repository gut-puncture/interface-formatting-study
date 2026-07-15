from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from interface_formatting_study import causal_cli
from interface_formatting_study.causal_design import build_causal_design
from interface_formatting_study.run_identity import sha256_file


def _write_design(tmp_path):
    design = build_causal_design(
        pd.DataFrame(
            [
                {
                    "item_id": "item-1",
                    "subject": "math",
                    "split": "train",
                    "question": "What is 2+2?",
                    "choices": ["3", "4", "5", "6"],
                    "correct_index": 1,
                }
            ]
        )
    )
    path = tmp_path / "design.parquet"
    design.to_parquet(path, index=False)
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps({"sha256": sha256_file(path), "rows": len(design)})
    )
    return path, design


def test_causal_cli_runs_identity_bound_canary(
    tmp_path,
    monkeypatch,
    causal_label_model,
    boundary_tokenizer,
):
    design_path, design = _write_design(tmp_path)
    monkeypatch.setattr(
        causal_cli,
        "load_model_and_tokenizer",
        lambda *args, **kwargs: (causal_label_model, boundary_tokenizer, torch.device("cpu")),
    )
    args = argparse.Namespace(
        design=str(design_path),
        profile="qwen",
        output_base=str(tmp_path / "runs"),
        batch_size=8,
        max_batch_tokens=2000,
        checkpoint_size=7,
        local_files_only=True,
        profile_timings=False,
        canary=True,
        canary_name="functional",
        canary_items=1,
    )

    causal_cli.cmd_run(args)

    manifests = list((tmp_path / "runs").glob("*/*/canaries/functional/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"]["id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert manifest["design"]["completed_rows"] == len(design)
    assert manifest["runtime"]["single_token_label_fast_path"]
