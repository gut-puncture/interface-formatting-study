from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
import numpy as np
import torch

from interface_formatting_study import decision_binding_cli
from interface_formatting_study.decision_binding import CapturedReadouts, ProbeBank, save_probe_bank
from interface_formatting_study.run_identity import sha256_file


def test_parser_exposes_only_the_three_focused_commands():
    parser = decision_binding_cli.build_parser()

    prepare = parser.parse_args(
        [
            "prepare",
            "--scored",
            "scores.parquet",
            "--applicability",
            "app.parquet",
            "--stage",
            "discovery",
            "--output-dir",
            "bundle",
        ]
    )
    run = parser.parse_args(
        ["run-model", "--profile", "phi", "--bundle", "bundle"]
    )
    analyze = parser.parse_args(["analyze", "--run", "run-root"])

    assert prepare.command == "prepare"
    assert run.command == "run-model"
    assert analyze.command == "analyze"


def test_prepare_writes_checksum_bound_compact_bundle(tmp_path, monkeypatch):
    scored_path = tmp_path / "scored.parquet"
    applicability_path = tmp_path / "app.parquet"
    pd.DataFrame({"source": [1]}).to_parquet(scored_path, index=False)
    pd.DataFrame({"source": [2]}).to_parquet(applicability_path, index=False)
    readout = pd.DataFrame(
        {
            "readout_work_key": ["discovery|baseline", "discovery|variant"],
            "work_key": ["baseline", "variant"],
            "split": ["train", "validation"],
            "readout_role": ["probe_train", "layer_select"],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_work_key": ["pair|baseline|variant"],
            "selected_for_patching": [True],
            "pair_kind": ["answer_conflict"],
        }
    )
    monkeypatch.setattr(decision_binding_cli, "prepare_readout_ledger", lambda *_a, **_k: readout)
    monkeypatch.setattr(decision_binding_cli, "prepare_patch_pair_ledger", lambda *_a, **_k: pairs)
    output = tmp_path / "bundle"
    args = SimpleNamespace(
        scored=str(scored_path),
        applicability=str(applicability_path),
        stage="discovery",
        output_dir=str(output),
        seed="fixed",
        stable_controls=96,
        label_binding_controls=96,
    )

    decision_binding_cli.cmd_prepare(args)

    manifest = json.loads((output / "bundle_manifest.json").read_text())
    assert manifest["stage"] == "discovery"
    assert manifest["source_scored_sha256"] == sha256_file(scored_path)
    assert manifest["readout"]["rows"] == 2
    assert manifest["readout"]["sha256"] == sha256_file(output / "readout_ledger.parquet")
    assert manifest["pairs"]["selected_rows"] == 1
    assert manifest["pairs"]["sha256"] == sha256_file(output / "patch_pair_ledger.parquet")


def test_analysis_uses_within_pair_differences_not_unpaired_means(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    rows = []
    for item, pair, baseline, effect in (
        ("item-1", "p1", 100.0, 2.0),
        ("item-1", "p2", -100.0, 4.0),
        ("item-2", "p3", 0.0, 10.0),
    ):
        for condition, value in (("unpatched", baseline), ("probe", baseline + effect)):
            rows.append(
                {
                    "item_id": item,
                    "pair_work_key": pair,
                    "mechanism": "content",
                    "layer": 3,
                    "pair_kind": "answer_conflict",
                    "condition": condition,
                    "content_target_margin": value,
                    "symbol_target_margin": value / 2,
                }
            )
    pd.DataFrame(rows).to_parquet(run_root / "patch_results.parquet", index=False)
    output = tmp_path / "analysis"
    args = SimpleNamespace(run=[str(run_root)], output_dir=str(output), bootstrap_samples=200, seed=7)

    decision_binding_cli.cmd_analyze(args)

    summary = pd.read_csv(output / "patch_effects.csv")
    probe = summary[summary["condition"] == "probe"].iloc[0]
    assert probe["mean_content_margin_change"] == 6.5
    assert probe["pairs"] == 3
    assert probe["items"] == 2


def test_run_model_records_loader_failure_in_the_identity_bound_run_root(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    readout_path = bundle / "readout_ledger.parquet"
    pairs_path = bundle / "patch_pair_ledger.parquet"
    pd.DataFrame({"readout_work_key": ["x"]}).to_parquet(readout_path, index=False)
    pd.DataFrame({"pair_work_key": ["p"]}).to_parquet(pairs_path, index=False)
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "stage": "discovery",
                "model": {"id": None, "revision": None},
                "readout": {"rows": 1, "sha256": sha256_file(readout_path)},
                "pairs": {"rows": 1, "sha256": sha256_file(pairs_path)},
            }
        )
    )
    monkeypatch.setattr(
        decision_binding_cli,
        "load_model_and_tokenizer",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("synthetic load failure")),
    )
    output = tmp_path / "runs"
    args = SimpleNamespace(
        bundle=str(bundle),
        profile="qwen",
        frozen_run=None,
        output_base=str(output),
        bootstrap_samples=10,
        permutation_samples=10,
        stable_controls=96,
        label_binding_controls=96,
        max_readout_rows=None,
        max_pairs=None,
        local_files_only=True,
        allow_cpu=True,
        batch_size=2,
        max_batch_tokens=200,
        readout_chunk_size=2,
        patch_shard_size=1,
        seed=0,
    )

    with pytest.raises(RuntimeError, match="synthetic load failure"):
        decision_binding_cli.cmd_run_model(args)

    manifests = list(output.glob("qwen2.5-1.5b-instruct/*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "RuntimeError"


def test_run_model_resumes_readout_and_patch_shards_without_recomputing(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    readout_path = bundle / "readout_ledger.parquet"
    pairs_path = bundle / "patch_pair_ledger.parquet"
    pd.DataFrame(
        {
            "readout_work_key": ["validation|one"],
            "readout_role": ["layer_select"],
            "prompt": ["prompt"],
            "item_id": ["item-1"],
            "winner_content_id": [0],
            "winner_position": [1],
            "winner_label_index": [2],
            "winner_unique": [True],
            "manipulation": ["label_only"],
        }
    ).to_parquet(readout_path, index=False)
    pd.DataFrame(
        {
            "pair_work_key": ["pair-b", "pair-a"],
            "selected_for_patching": [True, True],
        }
    ).to_parquet(pairs_path, index=False)
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "stage": "discovery",
                "model": {"id": None, "revision": None},
                "readout": {"rows": 1, "sha256": sha256_file(readout_path)},
                "pairs": {"rows": 2, "sha256": sha256_file(pairs_path)},
            }
        )
    )
    dummy_model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        decision_binding_cli,
        "load_model_and_tokenizer",
        lambda *_a, **_k: (dummy_model, object(), torch.device("cpu")),
    )
    bank = ProbeBank(
        weights=np.zeros((1, 2, 4, 2)),
        intercepts=np.zeros((1, 2, 4)),
        means=np.zeros((1, 2, 2)),
        classes=np.arange(4),
        c=1e-2,
    )

    def fake_bank(root, *_args):
        save_probe_bank(root / "probe_bank.npz", bank)
        return bank

    monkeypatch.setattr(decision_binding_cli, "_fit_or_load_bank", fake_bank)
    monkeypatch.setattr(
        decision_binding_cli,
        "capture_layer_readouts",
        lambda _m, _t, prompts, **_k: CapturedReadouts(
            activations=torch.zeros((len(prompts), 1, 2, 2)),
            raw_log_probs=torch.zeros((len(prompts), 4)),
            batches=1,
            actual_tokens=1,
            padded_tokens=1,
        ),
    )
    selection = {
        "content": {"usable": True, "checkpoint": "format_end", "patch_layers": [0]},
        "label": {"usable": True, "checkpoint": "answer_prefix_end", "patch_layers": [0]},
    }
    monkeypatch.setattr(decision_binding_cli, "select_readout_layers", lambda *_a, **_k: selection)
    patch_calls = []
    monkeypatch.setattr(
        decision_binding_cli,
        "run_patch_pair",
        lambda _m, _t, row, *_a, **_k: patch_calls.append(row["pair_work_key"])
        or pd.DataFrame({"pair_work_key": [row["pair_work_key"]], "condition": ["probe"]}),
    )
    args = SimpleNamespace(
        bundle=str(bundle), profile="qwen", frozen_run=None, output_base=str(tmp_path / "runs"),
        bootstrap_samples=10, permutation_samples=10, stable_controls=96,
        label_binding_controls=96, max_readout_rows=None, max_pairs=None,
        local_files_only=True, allow_cpu=True, batch_size=2, max_batch_tokens=200,
        readout_chunk_size=1, patch_shard_size=1, seed=0,
    )

    decision_binding_cli.cmd_run_model(args)
    decision_binding_cli.cmd_run_model(args)

    manifests = list((tmp_path / "runs").glob("qwen2.5-1.5b-instruct/*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "complete"
    assert manifest["semantic_identity"]["semantic_run_id"] == manifests[0].parent.name
    assert patch_calls == ["pair-b", "pair-a"]
    root = manifests[0].parent
    assert len(pd.read_parquet(root / "readout_scores.parquet")) == 2
    patch_results = pd.read_parquet(root / "patch_results.parquet")
    assert patch_results["pair_work_key"].tolist() == ["pair-a", "pair-b"]
    assert manifest["artifacts"]["readout_scores_sha256"] == sha256_file(
        root / "readout_scores.parquet"
    )
    assert manifest["artifacts"]["patch_results_sha256"] == sha256_file(
        root / "patch_results.parquet"
    )
    progress = json.loads((root / "progress.json").read_text())
    assert progress["phase"] == "patches"
    assert progress["completed"] == 2
    assert progress["total"] == 2
