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
from interface_formatting_study.causal_design import build_causal_design_v3
from interface_formatting_study.run_identity import sha256_file


VERIFY_SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_causal_followup_artifacts.py"
VERIFY_SPEC = importlib.util.spec_from_file_location("verify_causal_followup_artifacts", VERIFY_SCRIPT)
VERIFY = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
sys.modules[VERIFY_SPEC.name] = VERIFY
VERIFY_SPEC.loader.exec_module(VERIFY)


def _write_design(tmp_path):
    design, applicability = build_causal_design_v3(source_prompt_frame())
    path = tmp_path / "design.parquet"
    design.to_parquet(path, index=False)
    applicability_path = path.with_name(path.stem + ".applicability.parquet")
    applicability.to_parquet(applicability_path, index=False)
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps(
            {
                "design_schema_version": 5,
                "scope": "source_prompt_counterfactual",
                "sha256": sha256_file(path),
                "rows": len(design),
                "applicability": {
                    "path_name": applicability_path.name,
                    "sha256": sha256_file(applicability_path),
                    "rows": len(applicability),
                },
            }
        )
    )
    return path, design


def test_load_design_rejects_old_source_preserving_manifest(tmp_path):
    path, design = _write_design(tmp_path)
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

    with pytest.raises(RuntimeError, match="source-preserving"):
        causal_cli._load_design(path)


def test_load_design_rejects_canonical_rows_under_a_v5_manifest(tmp_path):
    path, design = _write_design(tmp_path)
    design["template_scope"] = "controlled_canonical_template"
    design.to_parquet(path, index=False)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="template_scope"):
        causal_cli._load_design(path)


def test_load_design_requires_the_checksum_bound_applicability_ledger(tmp_path):
    path, _ = _write_design(tmp_path)
    path.with_name(path.stem + ".applicability.parquet").unlink()

    with pytest.raises(RuntimeError, match="applicability"):
        causal_cli._load_design(path)


def test_load_design_rejects_duplicate_applicability_keys(tmp_path):
    path, _ = _write_design(tmp_path)
    applicability_path = path.with_name(path.stem + ".applicability.parquet")
    applicability = pd.read_parquet(applicability_path)
    applicability.iloc[1] = applicability.iloc[0]
    applicability.to_parquet(applicability_path, index=False)
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["applicability"]["sha256"] = sha256_file(applicability_path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="unique item-wrapper"):
        causal_cli._load_design(path)


def test_prepare_keeps_all_items_and_records_arm_applicability(tmp_path, monkeypatch):
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
    audit = tmp_path / "audit"
    (audit / "labels").mkdir(parents=True)
    (audit / "manifest.json").write_text('{"schema_version":1}\n')
    (audit / "labels" / "annotations.jsonl").write_text('{"annotation_id":"x"}\n')
    monkeypatch.setattr(causal_cli, "load_causal_option_annotations", lambda _path: {})
    choice_audit = tmp_path / "choice-audit"
    (choice_audit / "labels").mkdir(parents=True)
    (choice_audit / "manifest.json").write_text('{"schema_version":1}\n')
    (choice_audit / "labels" / "choices.jsonl").write_text('{"annotation_id":"y"}\n')
    monkeypatch.setattr(causal_cli, "load_causal_choice_overrides", lambda _path: {})

    causal_cli.cmd_prepare(
        argparse.Namespace(
            config="unused.yaml",
            splits="train,validation",
            output=str(output),
            option_audit=str(audit),
            choice_audit=str(choice_audit),
        )
    )

    manifest = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert manifest["design_schema_version"] == 5
    assert manifest["scope"] == "source_prompt_counterfactual"
    assert manifest["source_items"] == 2
    assert manifest["retained_items"] == 2
    assert manifest["option_audit"] == {
        "path": str(audit),
        "manifest_sha256": sha256_file(audit / "manifest.json"),
        "label_hashes": {
            "annotations.jsonl": sha256_file(audit / "labels" / "annotations.jsonl")
        },
    }
    assert manifest["choice_audit"] == {
        "path": str(choice_audit),
        "manifest_sha256": sha256_file(choice_audit / "manifest.json"),
        "label_hashes": {
            "choices.jsonl": sha256_file(choice_audit / "labels" / "choices.jsonl")
        },
    }
    applicability = pd.read_parquet(output.with_name(output.stem + ".applicability.parquet"))
    unresolved = applicability[
        (applicability["item_id"] == "item-2")
        & (applicability["wrapper_name"] == "protobuf_msg")
    ].iloc[0]
    assert not bool(unresolved["position_applicable"])
    assert not bool(unresolved["label_applicable"])


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
    assert manifest["design"]["applicability_sha256"]
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

    manifest["design"]["selected_work_sha256"] = "0" * 64
    manifests[0].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="selected work-key"):
        VERIFY.verify(
            root,
            manifest["semantic_identity"]["semantic_run_id"],
            manifest["model"]["slug"],
            "complete",
        )
    manifest["design"]["selected_work_sha256"] = causal_cli._selected_work_sha(design)
    manifests[0].write_text(json.dumps(manifest))

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
