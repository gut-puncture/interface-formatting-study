from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from causal_fixtures import source_prompt_frame
from interface_formatting_study import causal_cli
from interface_formatting_study.causal_design import build_causal_design
from interface_formatting_study.run_identity import sha256_file


VERIFY_SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_causal_followup_artifacts.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_causal_followup_artifacts", VERIFY_SCRIPT)
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
sys.modules[VERIFY_SPEC.name] = VERIFY
VERIFY_SPEC.loader.exec_module(VERIFY)


def _write_design(tmp_path):
    design = build_causal_design(source_prompt_frame())
    path = tmp_path / "design.parquet"
    design.to_parquet(path, index=False)
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps(
            {
                "design_schema_version": 4,
                "scope": "source_prompt_counterfactual",
                "sha256": sha256_file(path),
                "rows": len(design),
            }
        )
    )
    return path, design


def test_load_design_rejects_old_canonical_template_manifest(tmp_path):
    path, design = _write_design(tmp_path)
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps(
            {
                "design_schema_version": 3,
                "scope": "controlled canonical templates; original prompts remain in the observational study",
                "sha256": sha256_file(path),
                "rows": len(design),
            }
        )
    )

    with pytest.raises(RuntimeError, match="source-preserving"):
        causal_cli._load_design(path)


def test_prepare_records_source_preservation_and_whole_item_exclusions(tmp_path, monkeypatch):
    source = source_prompt_frame()
    excluded = source.copy()
    excluded["item_id"] = "item-2"
    excluded.loc[excluded["wrapper_name"] == "protobuf_msg", "wrapped_prompt"] = (
        'message MCQ { repeated string options = 1 ["3", "4", "5", "6"]; }'
        "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "
    )
    frame = pd.concat([source, excluded], ignore_index=True)
    monkeypatch.setattr(causal_cli, "read_yaml", lambda _path: {})
    monkeypatch.setattr(causal_cli, "prepare_dataset", lambda _config: (frame, pd.DataFrame()))
    output = tmp_path / "design.parquet"

    causal_cli.cmd_prepare(
        argparse.Namespace(config="unused.yaml", splits="train,validation", output=str(output))
    )

    manifest = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert manifest["design_schema_version"] == 4
    assert manifest["scope"] == "source_prompt_counterfactual"
    assert manifest["source_items"] == 2
    assert manifest["eligible_items"] == 1
    assert manifest["excluded_items"] == 1
    exclusions = pd.read_parquet(output.with_name(output.stem + ".exclusions.parquet"))
    assert exclusions.to_dict("records") == [
        {
            "item_id": "item-2",
            "wrapper_name": "protobuf_msg",
            "reason": "option_labels_not_unambiguous",
        }
    ]


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
    monkeypatch.setattr(
        "interface_formatting_study.causal_runner._generate_answers_many",
        lambda _model, _tokenizer, records, **_kwargs: [str(row["correct_text"]) for row in records],
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
        allow_cpu=True,
    )

    causal_cli.cmd_run(args)

    manifests = list((tmp_path / "runs").glob("*/*/canaries/functional/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "complete"
    assert manifest["model"]["id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert manifest["design"]["completed_rows"] == len(design)
    assert manifest["runtime"]["single_token_label_fast_path"]
    assert "torch" in manifest["runtime"]["dependencies"]
    root = manifests[0].parent
    result = VERIFY.verify(
        root,
        manifest["semantic_identity"]["semantic_run_id"],
        manifest["model"]["slug"],
        "complete",
    )
    assert result["work_keys"] == len(design)

    shard_data = next((root / "shards" / "causal_behavior").glob("shard-*/data.parquet"))
    shard_data.write_bytes(shard_data.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        VERIFY.verify(
            root,
            manifest["semantic_identity"]["semantic_run_id"],
            manifest["model"]["slug"],
            "complete",
        )


def test_profiling_canary_spans_prompt_length_distribution():
    frame = pd.DataFrame(
        {
            "item_id": [f"item-{index}" for index in range(10)],
            "prompt": ["x" * (index + 1) for index in range(10)],
        }
    )

    subset = causal_cli._canary_subset(frame, 3)

    assert set(subset["item_id"]) == {"item-0", "item-4", "item-9"}
