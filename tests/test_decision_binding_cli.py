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
            "--design",
            "design.parquet",
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
    design_path = tmp_path / "design.parquet"
    pd.DataFrame({"source": [1]}).to_parquet(scored_path, index=False)
    pd.DataFrame({"source": [2]}).to_parquet(applicability_path, index=False)
    pd.DataFrame({"source": [3]}).to_parquet(design_path, index=False)
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
    monkeypatch.setattr(decision_binding_cli, "_validate_prepare_sources", lambda *_a, **_k: {})
    output = tmp_path / "bundle"
    args = SimpleNamespace(
        scored=str(scored_path),
        applicability=str(applicability_path),
        design=str(design_path),
        stage="discovery",
        output_dir=str(output),
        seed="fixed",
        stable_controls=96,
        label_binding_controls=96,
    )

    decision_binding_cli.cmd_prepare(args)

    manifest = json.loads((output / "bundle_manifest.json").read_text())
    assert manifest["stage"] == "discovery"
    assert manifest["bundle_schema_version"] == 2
    assert manifest["source_scored_sha256"] == sha256_file(scored_path)
    assert manifest["source_design_sha256"] == sha256_file(design_path)
    assert manifest["readout"]["rows"] == 2
    assert manifest["readout"]["sha256"] == sha256_file(output / "readout_ledger.parquet")
    assert manifest["pairs"]["selected_rows"] == 1
    assert manifest["pairs"]["sha256"] == sha256_file(output / "patch_pair_ledger.parquet")


def test_run_refuses_a_legacy_bundle_without_canonical_source_attestation(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    readout = root / "readout_ledger.parquet"
    pairs = root / "patch_pair_ledger.parquet"
    pd.DataFrame({"readout_work_key": ["x"]}).to_parquet(readout, index=False)
    pd.DataFrame({"pair_work_key": ["p"]}).to_parquet(pairs, index=False)
    (root / "bundle_manifest.json").write_text(json.dumps({
        "bundle_schema_version": 1,
        "stage": "discovery",
        "readout": {"rows": 1, "sha256": sha256_file(readout)},
        "pairs": {"rows": 1, "sha256": sha256_file(pairs)},
    }))

    with pytest.raises(RuntimeError, match="canonical source attestation"):
        decision_binding_cli._load_bundle(root)


def test_run_refuses_schema_two_bundle_without_scored_source_attestation(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    readout = root / "readout_ledger.parquet"
    pairs = root / "patch_pair_ledger.parquet"
    pd.DataFrame(
        {
            "readout_work_key": [f"row-{index}" for index in range(2401)],
            "split": ["train"] * 1801 + ["validation"] * 600,
            "item_id": [f"train-{index}" for index in range(1801)]
            + [f"validation-{index}" for index in range(600)],
        }
    ).to_parquet(readout, index=False)
    pd.DataFrame({"pair_work_key": ["p"]}).to_parquet(pairs, index=False)
    (root / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "bundle_schema_version": 2,
                "stage": "discovery",
                "source_validation": {"split_items": {"train": 1801, "validation": 600}},
                "readout": {"rows": 2401, "sha256": sha256_file(readout)},
                "pairs": {"rows": 1, "sha256": sha256_file(pairs)},
            }
        )
    )

    with pytest.raises(RuntimeError, match="scored source attestation"):
        decision_binding_cli._load_bundle(root)


def test_scored_artifact_manifest_authenticates_every_row_and_file_hash(tmp_path):
    scored_path = tmp_path / "causal_behavior.parquet"
    frame = pd.DataFrame(
        {
            "semantic_run_id": ["run-1", "run-1"],
            "model_id": ["model", "model"],
            "model_revision": ["revision", "revision"],
        }
    )
    frame.to_parquet(scored_path, index=False)
    manifest_path = scored_path.with_name(scored_path.name + ".manifest.json")
    manifest_path.write_text(json.dumps({
        "semantic_run_id": "run-1", "model_id": "model", "model_revision": "revision",
        "row_count": 2, "sha256": sha256_file(scored_path),
    }))

    attestation = decision_binding_cli._validate_scored_manifest(scored_path, frame)
    assert attestation["semantic_run_id"] == "run-1"

    mixed = frame.copy()
    mixed.loc[1, "model_id"] = "other-model"
    with pytest.raises(ValueError, match="mixed or unauthenticated model identity"):
        decision_binding_cli._validate_scored_manifest(scored_path, mixed)

    scored_path.write_bytes(scored_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="scored artifact manifest"):
        decision_binding_cli._validate_scored_manifest(scored_path, frame)


def test_analysis_uses_within_pair_differences_not_unpaired_means(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    identity = {"semantic_run_id": "run", "model": {"slug": "toy-model"}}
    (run_root / "semantic_identity.json").write_text(json.dumps(identity))
    (run_root / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "stage": "discovery", "semantic_identity": identity
    }))
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
                    "split": "validation",
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


def test_analysis_keeps_discovery_and_confirmation_separate_and_reports_frozen_test_readout(
    tmp_path,
):
    roots = []
    for stage, split in (("discovery", "validation"), ("confirmation", "test")):
        root = tmp_path / "qwen2.5-1.5b-instruct" / f"{stage}-run"
        root.mkdir(parents=True)
        identity = {"semantic_run_id": f"{stage}-run", "model": {"slug": "qwen2.5-1.5b-instruct"}}
        (root / "semantic_identity.json").write_text(json.dumps(identity))
        (root / "run_manifest.json").write_text(json.dumps({
            "status": "complete", "stage": stage, "semantic_identity": identity
        }))
        pd.DataFrame([
            {
                "item_id": "item-1", "split": split, "pair_work_key": "pair-1",
                "mechanism": "content", "layer": 1, "pair_kind": "answer_conflict",
                "condition": condition, "content_target_margin": value,
                "symbol_target_margin": value,
            }
            for condition, value in (("unpatched", 0.0), ("probe", 1.0))
        ]).to_parquet(root / "patch_results.parquet", index=False)
        pd.DataFrame([
            {
                "readout_work_key": f"row-{checkpoint}-{manipulation}",
                "item_id": "item-1", "split": split, "manipulation": manipulation,
                "layer": 1, "checkpoint": checkpoint, "probe_pred_class": prediction,
                "winner_content_id": 0, "winner_position": 1, "winner_label_index": 2,
                "winner_unique": True, "text_identity_ambiguous": False,
                "content_log_prob": -0.1, "position_log_prob": -2.0, "label_log_prob": -3.0,
            }
            for checkpoint in ("format_end", "answer_prefix_end")
            for manipulation, prediction in (("controlled_baseline", 2), ("label_only", 0))
        ]).to_parquet(root / "readout_scores.parquet", index=False)
        (root / "frozen_selection.json").write_text(json.dumps({
            "selection": {
                "content": {"selected_layer": 1, "checkpoint": "format_end"},
                "label": {"selected_layer": 1, "checkpoint": "answer_prefix_end"},
            }
        }))
        roots.append(str(root))
    output = tmp_path / "analysis"

    decision_binding_cli.cmd_analyze(SimpleNamespace(
        run=roots, output_dir=str(output), bootstrap_samples=50, seed=4
    ))

    patch = pd.read_csv(output / "patch_effects.csv")
    assert set(patch["stage"]) == {"discovery", "confirmation"}
    assert len(patch[patch["condition"] == "probe"]) == 2
    readout = pd.read_csv(output / "readout_confirmation.csv")
    assert set(readout["stage"]) == {"confirmation"}
    assert readout.iloc[0]["split"] == "test"
    assert set(readout["controlled_variant_rows"]) == {1}


def test_canary_selection_uses_eight_distinct_items_and_is_deterministic():
    pairs = pd.DataFrame(
        {
            "item_id": [f"item-{index}" for index in range(12) for _ in range(2)],
            "subject": [f"subject-{index % 3}" for index in range(12) for _ in range(2)],
            "wrapper_name": [f"wrapper-{index % 4}" for index in range(12) for _ in range(2)],
            "selected_for_patching": True,
            "donor_prompt": ["x" * (index + 1) for index in range(12) for _ in range(2)],
            "receiver_prompt": ["y" * (index + 1) for index in range(12) for _ in range(2)],
        }
    )

    first = decision_binding_cli._select_canary_items(pairs, 8, seed=7)
    second = decision_binding_cli._select_canary_items(pairs, 8, seed=7)

    assert len(first) == 8
    assert len(set(first)) == 8
    assert first == second
    assert "item-11" in first
    assert decision_binding_cli._select_canary_items(pairs, 1, seed=7) == ["item-11"]


def test_confirmation_macro_accuracy_is_item_equal_before_class_averaging():
    frame = pd.DataFrame(
        [
            *(
                {"item_id": "item-1", "target": 0, "probe_pred_class": 1}
                for _ in range(10)
            ),
            {"item_id": "item-2", "target": 0, "probe_pred_class": 0},
            {"item_id": "item-3", "target": 1, "probe_pred_class": 1},
        ]
    )

    assert decision_binding_cli._item_clustered_macro_accuracy(frame, "target") == pytest.approx(
        0.75
    )


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
    monkeypatch.setattr(
        decision_binding_cli,
        "_load_bundle",
        lambda _root: (
            json.loads((bundle / "bundle_manifest.json").read_text()),
            pd.read_parquet(readout_path),
            pd.read_parquet(pairs_path),
        ),
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
    assert manifest["semantic_identity"] == json.loads(
        (manifests[0].parent / "semantic_identity.json").read_text()
    )


def test_seed_and_readout_chunking_are_bound_to_resume_identity(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    readout_path = bundle / "readout_ledger.parquet"
    pairs_path = bundle / "patch_pair_ledger.parquet"
    pd.DataFrame({"readout_work_key": ["x"]}).to_parquet(readout_path, index=False)
    pd.DataFrame({"pair_work_key": ["p"]}).to_parquet(pairs_path, index=False)
    (bundle / "bundle_manifest.json").write_text(json.dumps({
        "stage": "discovery", "model": {"id": None, "revision": None},
        "readout": {"rows": 1, "sha256": sha256_file(readout_path)},
        "pairs": {"rows": 1, "sha256": sha256_file(pairs_path)},
    }))
    monkeypatch.setattr(
        decision_binding_cli,
        "load_model_and_tokenizer",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stop after identity")),
    )
    monkeypatch.setattr(
        decision_binding_cli,
        "_load_bundle",
        lambda _root: (
            json.loads((bundle / "bundle_manifest.json").read_text()),
            pd.read_parquet(readout_path),
            pd.read_parquet(pairs_path),
        ),
    )
    base = dict(
        bundle=str(bundle), profile="qwen", frozen_run=None, output_base=str(tmp_path / "runs"),
        bootstrap_samples=10, permutation_samples=10, stable_controls=96,
        label_binding_controls=96, max_readout_rows=None, max_pairs=None, canary_items=None,
        local_files_only=True, allow_cpu=True, batch_size=2, max_batch_tokens=200,
        patch_shard_size=1,
    )
    for seed, chunk in ((0, 1), (1, 1), (0, 2)):
        with pytest.raises(RuntimeError, match="stop after identity"):
            decision_binding_cli.cmd_run_model(
                SimpleNamespace(**base, seed=seed, readout_chunk_size=chunk)
            )

    assert len(list((tmp_path / "runs" / "qwen2.5-1.5b-instruct").iterdir())) == 3


def test_frozen_mechanism_requires_a_complete_discovery_run(tmp_path):
    root = tmp_path / "discovery"
    root.mkdir()
    profile = decision_binding_cli.get_model_profile("qwen")
    identity = {
        "model": {"id": profile.model_id, "revision": profile.revision},
        "semantic_run_id": "frozen-id",
    }
    (root / "semantic_identity.json").write_text(json.dumps(identity))
    bank = ProbeBank(
        weights=np.zeros((1, 2, 4, 2)), intercepts=np.zeros((1, 2, 4)),
        means=np.zeros((1, 2, 2)), classes=np.arange(4), c=1e-2,
    )
    save_probe_bank(root / "probe_bank.npz", bank)
    (root / "frozen_selection.json").write_text(json.dumps({
        "probe_bank_sha256": sha256_file(root / "probe_bank.npz"), "selection": {}
    }))
    (root / "run_manifest.json").write_text(json.dumps({
        "status": "interrupted", "stage": "discovery", "semantic_identity": identity
    }))

    with pytest.raises(RuntimeError, match="complete discovery"):
        decision_binding_cli._load_frozen(root, "qwen")


def test_frozen_mechanism_rejects_a_completed_limited_canary(tmp_path):
    root = tmp_path / "canary"
    root.mkdir()
    profile = decision_binding_cli.get_model_profile("qwen")
    identity = {
        "model": {"id": profile.model_id, "revision": profile.revision},
        "semantic_run_id": "canary-id",
        "experiment_config": {
            "limits": {"canary_items": 8, "readout_rows": None, "pairs": None}
        },
    }
    (root / "semantic_identity.json").write_text(json.dumps(identity))
    (root / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "stage": "discovery", "semantic_identity": identity,
        "selected_pairs": 2, "completed_pairs": 2,
    }))

    with pytest.raises(RuntimeError, match="full discovery"):
        decision_binding_cli._load_frozen(root, "qwen")


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
    monkeypatch.setattr(
        decision_binding_cli,
        "_load_bundle",
        lambda _root: (
            json.loads((bundle / "bundle_manifest.json").read_text()),
            pd.read_parquet(readout_path),
            pd.read_parquet(pairs_path),
        ),
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
    assert manifest["telemetry"]["readout"]["padding_ratio"] == pytest.approx(1.0)
    assert manifest["telemetry"]["patches"]["completed_pairs_per_second"] > 0
    progress = json.loads((root / "progress.json").read_text())
    assert progress["phase"] == "patches"
    assert progress["completed"] == 2
    assert progress["total"] == 2
