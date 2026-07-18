from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from interface_formatting_study.decision_binding_content import (
    CANDIDATE_TARGET_STATE_COLUMNS,
    majority_candidate_scores,
    metadata_candidate_features,
)
from interface_formatting_study.decision_binding_content_cli import (
    _best_control,
    _categorical_parity_receipt,
    _capture_training,
    _eligible_training_inputs,
    _eligibility_receipt,
    _load_activation_shard,
    _load_prepared_bundle,
    _runtime_environment_receipt,
    _save_activation_shard,
    _score_selected_metadata_controls,
    _select_canary_sites,
    _verify_raw_winners,
    build_parser,
    verify_run_root,
)
from interface_formatting_study.run_identity import sha256_file


def _target_state_for_keys(work_keys: list[str]) -> pd.DataFrame:
    records = []
    for work_key in work_keys:
        records.append({
            "work_key": work_key,
            **{f"stored_raw_score_{label}": value for label, value in zip("ABCD", (-0.1, -1.0, -2.0, -3.0), strict=True)},
            **{f"fresh_raw_score_{label}": value for label, value in zip("ABCD", (-0.1, -1.0, -2.0, -3.0), strict=True)},
            "stored_winner_label": "A", "stored_winner_position": 0,
            "stored_winner_content_id": 0, "fresh_winner_label": "A",
            "fresh_winner_position": 0, "fresh_winner_content_id": 0,
            "stored_winner_unique": True, "fresh_winner_unique": True,
            "stored_winner_margin": 0.9, "fresh_winner_margin": 0.9,
            "max_abs_raw_score_drift": 0.0, "stored_fresh_content_agree": True,
            "reader_target_evaluable": True,
            "reader_target_ineligibility_reason": "eligible",
        })
    return pd.DataFrame.from_records(records, columns=CANDIDATE_TARGET_STATE_COLUMNS)


def test_content_cli_exposes_preparation_model_run_and_verification():
    parser = build_parser()

    prepared = parser.parse_args([
        "prepare", "--bundle", "bundle", "--applicability", "status",
        "--option-audit", "audit", "--output-dir", "out",
    ])
    run = parser.parse_args([
        "run-model", "--profile", "mistral", "--bundle", "prepared",
    ])
    verify = parser.parse_args([
        "verify", "--run-root", "run", "--run-id", "id", "--mode", "canary",
    ])

    assert prepared.command == "prepare"
    assert run.command == "run-model"
    assert verify.command == "verify"
    assert run.l2_grid == [1e-4, 1e-3, 1e-2, 1e-1]
    assert run.batch_size == 32
    assert run.max_batch_tokens == 40000
    with pytest.raises(SystemExit):
        parser.parse_args(["run-model", "--profile", "mistral", "--bundle", "x", "--stage", "confirmation"])


def test_raw_winner_instability_is_classified_instead_of_aborting():
    rows = pd.DataFrame({
        "work_key": ["stable", "unstable"],
        "raw_predicted_label": ["A", "A"],
        "winner_unique": [True, True],
        "raw_score_A": [-0.5, -1.0],
        "raw_score_B": [-1.0, -1.0625],
        "raw_score_C": [-2.0, -2.0],
        "raw_score_D": [-3.0, -3.0],
        "winner_position": [0, 0],
        "text_identity_ambiguous": [False, False],
        "content_target_evaluable": [True, True],
        "labels_by_position": [list("ABCD"), list("ABCD")],
        "actual_content_ids_by_position": [[0, 1, 2, 3], [0, 1, 2, 3]],
        "actual_winner_content_id": [0, 0],
    })
    fresh = __import__("torch").tensor([
        [-0.5, -1.0, -2.0, -3.0],
        [-1.0625, -1.0, -2.0, -3.0],
    ])

    classified = _verify_raw_winners(rows, fresh)

    assert classified["work_key"].tolist() == ["stable", "unstable"]
    assert classified["reader_target_evaluable"].tolist() == [True, False]
    assert classified["reader_target_ineligibility_reason"].tolist() == [
        "eligible", "stored_fresh_content_mismatch",
    ]


def test_prepared_bundle_is_checksum_and_role_bound(tmp_path):
    sites = pd.DataFrame({
        "work_key": ["a", "b", "c"],
        "item_id": ["train", "select", "gate"],
        "readout_role": ["probe_train", "layer_select", "reader_gate"],
    })
    path = tmp_path / "candidate_sites.parquet"
    sites.to_parquet(path, index=False)
    manifest = {
        "schema_version": 1,
        "stage": "discovery",
        "rows": 3,
        "sha256": sha256_file(path),
        "role_items": {"probe_train": 1, "layer_select": 1, "reader_gate": 1},
        "source_model": {"id": "model", "revision": "revision", "slug": "slug"},
    }
    (tmp_path / "prepared_manifest.json").write_text(json.dumps(manifest))

    loaded_manifest, loaded = _load_prepared_bundle(tmp_path)

    assert loaded_manifest == manifest
    assert loaded["work_key"].tolist() == ["a", "b", "c"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        _load_prepared_bundle(tmp_path)


def test_canary_sampling_is_role_balanced_and_never_changes_source_rows():
    rows = []
    for role in ("probe_train", "layer_select", "reader_gate"):
        for item in range(8):
            for variant in range(3):
                rows.append({
                    "work_key": f"{role}|{item}|{variant}",
                    "item_id": f"{role}|{item}",
                    "readout_role": role,
                    "variant": variant,
                    "prompt": (
                        f"exact-{role}-{item}-{variant}-" + ("λ" if item == 1 else "")
                        + ("x" * 200 if item == 2 else "")
                    ),
                    "candidate_texts": ["one", "two words", "three", "four"],
                })
    sites = pd.DataFrame(rows)

    canary = _select_canary_sites(sites, items_per_role=3, seed=7)

    assert canary.groupby("readout_role")["item_id"].nunique().to_dict() == {
        "layer_select": 3,
        "probe_train": 3,
        "reader_gate": 3,
    }
    expected = sites.set_index("work_key")["prompt"]
    assert all(expected[row.work_key] == row.prompt for row in canary.itertuples())
    assert canary["prompt"].str.contains("λ").any()
    assert canary["prompt"].str.len().max() > 200


def test_activation_shards_are_identity_and_checksum_bound(tmp_path):
    path = tmp_path / "capture.pt"
    values = __import__("torch").arange(24).reshape(1, 2, 4, 3)
    raw = torch.tensor([[-1.0, -1.0, -2.0, -3.0]])
    source = pd.DataFrame({
        "work_key": ["work"], "raw_predicted_label": ["A"],
        "winner_position": [0], "winner_unique": [True],
        "text_identity_ambiguous": [False], "content_target_evaluable": [True],
        "labels_by_position": [list("ABCD")],
        "actual_content_ids_by_position": [[0, 1, 2, 3]],
        "actual_winner_content_id": [0],
        "raw_score_A": [-0.5], "raw_score_B": [-1.0],
        "raw_score_C": [-2.0], "raw_score_D": [-3.0],
    })
    state = _verify_raw_winners(source, raw)
    _save_activation_shard(
        path, activations=values, raw_log_probs=raw, target_state=state,
        work_keys=["work"], semantic_sha256="a" * 64
    )

    loaded, loaded_state = _load_activation_shard(
        path, work_keys=["work"], semantic_sha256="a" * 64
    )

    assert loaded.equal(values)
    pd.testing.assert_frame_equal(loaded_state, state)
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        _load_activation_shard(path, work_keys=["work"], semantic_sha256="a" * 64)


@pytest.mark.parametrize(
    "activations",
    [
        torch.tensor([[[[float("nan")]] * 4]]),
        torch.zeros(2, 1, 4, 1),
    ],
)
def test_activation_shard_save_rejects_nonfinite_or_misaligned_values(
    tmp_path, activations
):
    path = tmp_path / "capture.pt"
    state = _target_state_for_keys(["work"])

    with pytest.raises(ValueError, match="activations are malformed"):
        _save_activation_shard(
            path,
            activations=activations,
            raw_log_probs=torch.tensor([[-0.1, -1.0, -2.0, -3.0]]),
            target_state=state,
            work_keys=["work"],
            semantic_sha256="a" * 64,
        )


@pytest.mark.parametrize("corruption", ["nonfinite", "cardinality"])
def test_activation_shard_load_rejects_nonfinite_or_misaligned_values(
    tmp_path, corruption
):
    path = tmp_path / "capture.pt"
    state = _target_state_for_keys(["work"])
    _save_activation_shard(
        path,
        activations=torch.zeros(1, 1, 4, 1),
        raw_log_probs=torch.tensor([[-0.1, -1.0, -2.0, -3.0]]),
        target_state=state,
        work_keys=["work"],
        semantic_sha256="a" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["activations"] = (
        torch.tensor([[[[float("inf")]] * 4]])
        if corruption == "nonfinite"
        else torch.zeros(2, 1, 4, 1)
    )
    torch.save(payload, path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = sha256_file(path)
    manifest["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="activation shard is malformed"):
        _load_activation_shard(
            path, work_keys=["work"], semantic_sha256="a" * 64
        )


def test_activation_shard_rejects_target_state_tampering(tmp_path):
    path = tmp_path / "capture.pt"
    source = pd.DataFrame({
        "work_key": ["work"], "raw_predicted_label": ["A"],
        "winner_position": [0], "winner_unique": [True],
        "text_identity_ambiguous": [False], "content_target_evaluable": [True],
        "labels_by_position": [list("ABCD")],
        "actual_content_ids_by_position": [[0, 1, 2, 3]],
        "actual_winner_content_id": [0],
        "raw_score_A": [-0.1], "raw_score_B": [-1.0],
        "raw_score_C": [-2.0], "raw_score_D": [-3.0],
    })
    state = _verify_raw_winners(source, torch.tensor([[-0.1, -1.0, -2.0, -3.0]]))
    _save_activation_shard(
        path, activations=torch.zeros(1, 2, 4, 3),
        raw_log_probs=torch.tensor([[-0.1, -1.0, -2.0, -3.0]]),
        target_state=state, work_keys=["work"], semantic_sha256="a" * 64,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_payload = json.loads(payload["target_state_json"])
    column = state_payload["columns"].index("reader_target_evaluable")
    state_payload["data"][0][column] = False
    payload["target_state_json"] = json.dumps(state_payload, separators=(",", ":"))
    payload["target_state_sha256"] = __import__("hashlib").sha256(
        payload["target_state_json"].encode()
    ).hexdigest()
    torch.save(payload, path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"] = sha256_file(path)
    manifest["bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="target state"):
        _load_activation_shard(path, work_keys=["work"], semantic_sha256="a" * 64)


def test_training_capture_resume_reuses_persisted_target_state(monkeypatch, tmp_path):
    training = pd.DataFrame({
        "work_key": ["a", "b"], "prompt": ["prompt-a", "prompt-b"],
        "raw_predicted_label": ["A", "A"], "winner_position": [0, 0],
        "winner_unique": [True, True], "text_identity_ambiguous": [False, False],
        "content_target_evaluable": [True, True],
        "labels_by_position": [list("ABCD"), list("ABCD")],
        "actual_content_ids_by_position": [[0, 1, 2, 3], [0, 1, 2, 3]],
        "actual_winner_content_id": [0, 0],
        "raw_score_A": [-0.1, -0.1], "raw_score_B": [-1.0, -1.0],
        "raw_score_C": [-2.0, -2.0], "raw_score_D": [-3.0, -3.0],
    })
    activations = torch.arange(48).reshape(2, 2, 4, 3)
    raw = torch.tensor([[-0.1, -1.0, -2.0, -3.0]] * 2)
    calls = []

    def capture(*_args, **_kwargs):
        calls.append(True)
        return SimpleNamespace(activations=activations, raw_log_probs=raw)

    monkeypatch.setattr(
        "interface_formatting_study.decision_binding_content_cli.capture_candidate_states",
        capture,
    )
    monkeypatch.setattr(
        "interface_formatting_study.decision_binding_content_cli._token_positions",
        lambda _tokenizer, frame: [[0, 1, 2, 3] for _ in range(len(frame))],
    )
    kwargs = dict(
        root=tmp_path, training=training, model=object(), tokenizer=object(),
        identity=SimpleNamespace(semantic_sha256="a" * 64), batch_size=2,
        max_batch_tokens=100, chunk_size=2,
        stop=SimpleNamespace(requested=False), progress=lambda *_args: None,
    )
    first_activations, first_state = _capture_training(**kwargs)
    monkeypatch.setattr(
        "interface_formatting_study.decision_binding_content_cli.capture_candidate_states",
        lambda *_args, **_kwargs: pytest.fail("resume repeated a completed forward"),
    )
    resumed_activations, resumed_state = _capture_training(**kwargs)

    assert len(calls) == 1
    assert resumed_activations.equal(first_activations)
    pd.testing.assert_frame_equal(resumed_state, first_state)


def test_one_runtime_eligibility_index_drives_every_training_consumer():
    training = pd.DataFrame({
        "work_key": ["eligible", "tie", "eligible-2"],
        "item_id": ["eligible", "tie", "eligible-2"],
        "wrapper_name": ["plain"] * 3,
        "readout_role": ["probe_train"] * 3,
        "manipulation": ["controlled_baseline"] * 3,
        "winner_position": [0, 0, 2],
        "actual_winner_content_id": [0, 0, 2],
        "raw_predicted_label": ["A", "A", "C"],
        "winner_unique": [True, True, True],
        "text_identity_ambiguous": [False, False, False],
        "content_target_evaluable": [True, True, True],
        "labels_by_position": [list("ABCD")] * 3,
        "actual_content_ids_by_position": [[0, 1, 2, 3]] * 3,
        "candidate_texts": [["a", "bb", "ccc", "dddd"]] * 3,
        "raw_score_A": [-0.1, -0.1, -1.0],
        "raw_score_B": [-1.0, -1.0, -2.0],
        "raw_score_C": [-2.0, -2.0, -0.1],
        "raw_score_D": [-3.0, -3.0, -3.0],
    })
    fresh = torch.tensor([
        [-0.1, -1.0, -2.0, -3.0],
        [-0.1, -0.1, -2.0, -3.0],
        [-1.0, -2.0, -0.1, -3.0],
    ])
    state = _verify_raw_winners(training, fresh)
    all_activations = torch.arange(72).reshape(3, 2, 4, 3)

    eligible, activations, targets = _eligible_training_inputs(
        training, all_activations, state
    )
    features = metadata_candidate_features(eligible)
    random_targets = (targets + np.random.default_rng(1000).integers(0, 4, len(targets))) % 4
    majority = majority_candidate_scores(eligible, eligible["actual_winner_content_id"])

    assert eligible["work_key"].tolist() == ["eligible", "eligible-2"]
    assert activations.shape[0] == len(targets) == len(random_targets) == 2
    assert all(values.shape[0] == 2 for values in features.values())
    assert majority["work_key"].nunique() == 2


def test_best_control_uses_only_runtime_eligible_rows():
    rows = []
    for l2 in (0.01, 0.1):
        for manipulation in ("position_only", "label_only"):
            for item in range(2):
                target = 0
                prediction = 0 if l2 == 0.01 else 1
                evaluable = True
                if item == 1:
                    evaluable = False
                    prediction = 1 if l2 == 0.01 else 0
                probability = [0.01] * 4
                probability[prediction] = 0.97
                rows.append({
                    "l2": l2, "item_id": f"{manipulation}-{item}",
                    "manipulation": manipulation,
                    "actual_winner_content_id": target,
                    "reader_target_evaluable": evaluable,
                    **{f"content_prob_{index}": probability[index] for index in range(4)},
                })

    selected = _best_control(pd.DataFrame(rows))

    assert selected["l2"].unique().tolist() == [0.01]
    assert len(selected) == 4  # selection filters, returned artifact retains every row


def test_gate_metadata_controls_never_treat_majority_as_a_fitted_ranker(monkeypatch):
    selected = {
        "position": pd.DataFrame({"l2": [0.1]}),
        "label": pd.DataFrame({"l2": [0.01]}),
        "majority": pd.DataFrame({"l2": [1.0]}),
    }
    fitted = {"position": {0.1: "position-model"}, "label": {0.01: "label-model"}}
    calls = []
    monkeypatch.setattr(
        "interface_formatting_study.decision_binding_content_cli._score_metadata_rankers",
        lambda _sites, models, name: calls.append((name, models)) or pd.DataFrame(),
    )

    observed = _score_selected_metadata_controls(pd.DataFrame(), selected, fitted)

    assert set(observed) == {"position", "label"}
    assert calls == [
        ("position", {0.1: "position-model"}),
        ("label", {0.01: "label-model"}),
    ]


def test_eligibility_receipt_is_order_independent_and_reason_bound():
    state = _target_state_for_keys(["b", "a"])
    state.loc[state["work_key"].eq("b"), "reader_target_evaluable"] = False
    state.loc[
        state["work_key"].eq("b"), "reader_target_ineligibility_reason"
    ] = "fresh_raw_tie"
    state.loc[state["work_key"].eq("b"), "fresh_winner_unique"] = False
    state.loc[state["work_key"].eq("b"), "fresh_winner_margin"] = 0.0
    first = _eligibility_receipt(state)
    second = _eligibility_receipt(state.iloc[::-1].reset_index(drop=True))
    changed = state.copy()
    changed.loc[0, "reader_target_ineligibility_reason"] = "stored_fresh_content_mismatch"

    assert first == second
    assert first["total_rows"] == 2
    assert first["evaluable_rows"] == 1
    assert first["reason_counts"] == {"eligible": 1, "fresh_raw_tie": 1}
    assert _eligibility_receipt(changed)["eligibility_sha256"] != first["eligibility_sha256"]


def test_categorical_parity_allows_only_drift_explainable_near_ties():
    near_left = np.array([[0.51, 0.49, 0.0, 0.0]])
    near_right = np.array([[0.49, 0.51, 0.0, 0.0]])
    near = _categorical_parity_receipt(near_left, near_right, ["near"])
    assert near["passes"] is True
    assert near["ambiguous_work_keys"] == ["near"]
    assert near["ambiguous_count"] == 1

    resolved_left = np.array([[0.8, 0.2, 0.0, 0.0]])
    resolved_right = np.array([[0.49, 0.51, 0.0, 0.0]])
    resolved = _categorical_parity_receipt(resolved_left, resolved_right, ["resolved"])
    assert resolved["passes"] is False

    too_far = _categorical_parity_receipt(
        np.array([[0.51, 0.49, 0.0, 0.0]]),
        np.array([[0.48, 0.52, 0.0, 0.0]]),
        ["far"], maximum_difference=0.02,
    )
    assert too_far["passes"] is False

    just_above = _categorical_parity_receipt(
        np.array([[0.50, 0.50, 0.0, 0.0]]),
        np.array([[0.5200001, 0.4799999, 0.0, 0.0]]),
        ["boundary"], maximum_difference=0.02,
    )
    assert just_above["observed_maximum_difference"] > 0.02
    assert just_above["passes"] is False


def test_runtime_environment_receipt_binds_gpu_class(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index=0: "NVIDIA A100-SXM4-80GB")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=0: (8, 0))
    a100 = _runtime_environment_receipt(attention_backend="sdpa")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index=0: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index=0: (9, 0))
    h100 = _runtime_environment_receipt(attention_backend="sdpa")

    assert a100["gpu_name"] != h100["gpu_name"]
    assert a100["gpu_compute_capability"] == "8.0"
    assert h100["gpu_compute_capability"] == "9.0"
    assert a100["attention_backend"] == h100["attention_backend"] == "sdpa"


def test_run_verifier_checks_identity_artifacts_and_claim_boundary(tmp_path):
    environment = {
        "device_type": "cuda", "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_compute_capability": "9.0", "bf16_supported": True,
        "attention_backend": "sdpa", "torch_version": "2.7.1",
        "cuda_version": "12.6", "transformers_version": "4.53.0",
        "tokenizers_version": "0.21.0", "inference_dtype": "bfloat16",
        "fit_dtype": "float32",
    }
    prepared_source_hashes = {
        "source_bundle_manifest_sha256": "b" * 64,
        "source_readout_sha256": "c" * 64,
        "source_applicability_sha256": "d" * 64,
        "option_audit_manifest_sha256": "e" * 64,
    }
    identity = {
        "semantic_run_id": "run-id",
        "semantic_sha256": "a" * 64,
        "model": {
            "id": "model", "revision": "revision", "slug": "slug",
            "expected_layers": 1,
        },
        "experiment_config": {
            "l2_grid": [0.1], "seed": 0, "runtime_environment": environment,
            "prepared_source_hashes": prepared_source_hashes,
        },
    }
    (tmp_path / "semantic_identity.json").write_text(json.dumps(identity))
    gate = {
        "claim": "candidate_local_linear_decodability",
        "selection_eligible": False,
        "stop_reason": "no_reader_beat_both_isolated_nuisance_controls",
        "reader_gate_opened": False,
        "selected_layer": 0,
        "selected_l2": 0.1,
        "tier_1_pass": False,
        "tier_2_pass": False,
        "patch_eligible": False,
        "opens_final_confirmation": False,
    }
    (tmp_path / "gate_report.json").write_text(json.dumps(gate))
    work_keys = [f"work-{index:03d}" for index in range(300)]
    training_keys = [f"train-{index:04d}" for index in range(1801)]
    layer_state = _target_state_for_keys(work_keys)
    training_state = _target_state_for_keys(training_keys)
    training_state.to_parquet(tmp_path / "training_target_state.parquet", index=False)
    training_receipt = _eligibility_receipt(training_state)
    layer_receipt = _eligibility_receipt(layer_state)
    frozen_selection = {
        "selection_eligible": False,
        "stop_reason": "no_reader_beat_both_isolated_nuisance_controls",
        "selected_layer": 0,
        "selected_l2": 0.1,
        "target_policy_version": 1,
        "training_eligibility": training_receipt,
        "layer_select_eligibility": layer_receipt,
        "selected_control_l2": {
            name: 0.1 for name in ("position", "label", "position_label", "answer_length")
        },
        "majority_training_distribution": [1.0, 0.0, 0.0, 0.0],
        "random_reader_seeds": [1000, 1001, 1002],
        "prepared_source_hashes": prepared_source_hashes,
    }
    scores = pd.DataFrame({
        "work_key": work_keys, "reader_name": "content", "l2": 0.1,
        "layer": 0, "readout_role": "layer_select",
    }).merge(layer_state, on="work_key", validate="one_to_one")
    scores.to_parquet(tmp_path / "layer_select_scores.parquet", index=False)
    control_names = ["position", "label", "position_label", "answer_length", "majority"]
    controls = pd.concat(
        [scores.assign(reader_name=name) for name in control_names], ignore_index=True
    ).sort_values(["reader_name", "l2", "layer", "work_key"], kind="mergesort")
    controls.to_parquet(tmp_path / "layer_select_control_scores.parquet", index=False)
    ranker_names = {
        "content-l2-1e-01.npz",
        "control-position-l2-1e-01.npz",
        "control-label-l2-1e-01.npz",
        "control-position_label-l2-1e-01.npz",
        "control-answer_length-l2-1e-01.npz",
    }
    (tmp_path / "rankers").mkdir()
    ranker_receipts = {}
    for name in ranker_names:
        path = tmp_path / "rankers" / name
        path.write_bytes(name.encode())
        ranker_receipts[name] = sha256_file(path)
    frozen_selection["ranker_sha256"] = ranker_receipts
    (tmp_path / "frozen_selection.json").write_text(json.dumps(frozen_selection))
    (tmp_path / "ranker_manifest.json").write_text(json.dumps({
        "schema_version": 1, "semantic_sha256": "a" * 64, "rankers": ranker_receipts,
    }))
    names = [
        "semantic_identity.json", "frozen_selection.json", "gate_report.json",
        "ranker_manifest.json", "layer_select_scores.parquet",
        "layer_select_control_scores.parquet", "training_target_state.parquet",
    ]
    artifacts = {}
    for name in names:
        path = tmp_path / name
        artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        if path.suffix == ".parquet":
            artifact_frame = pd.read_parquet(path)
            artifacts[name]["rows"] = len(artifact_frame)
            artifacts[name]["work_keys"] = artifact_frame["work_key"].nunique()
            artifacts[name]["eligibility"] = _eligibility_receipt(
                artifact_frame if name == "training_target_state.parquet"
                else artifact_frame[list(CANDIDATE_TARGET_STATE_COLUMNS)].drop_duplicates()
            )
    manifest = {
        "status": "canary_complete",
        "semantic_identity": identity,
        "canary": True,
        "role_items": {"probe_train": 1801, "layer_select": 300, "reader_gate": 300},
        "tier_1_pass": False,
        "tier_2_pass": False,
        "patch_eligible": False,
        "final_confirmation_opened": False,
        "reader_gate_opened": False,
        "training_pool_items": 1801,
        "training_evaluable_items": 1801,
        "training_eligibility": training_receipt,
        "target_evaluability_by_role": {"layer_select": layer_receipt},
        "artifacts": artifacts,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="required artifacts|work plan"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="canary")

    (tmp_path / "work_plan.json").write_text(json.dumps({
        "schema_version": 2,
        "semantic_sha256": "a" * 64,
        "reader_gate_opened": False,
        "role_items": {"probe_train": 1801, "layer_select": 300, "reader_gate": 300},
        "roles": {
            "layer_select": {"rows": 300, "work_keys": 300, "work_keys_sha256": "placeholder"},
            "reader_gate": {"rows": 300, "work_keys": 300, "work_keys_sha256": "placeholder"},
        },
        "layer_count": 1,
        "l2_grid": [0.1],
        "selected_layer": 0,
        "selected_l2": 0.1,
        "training_eligibility": training_receipt,
        "role_eligibility": {"layer_select": layer_receipt},
    }))
    manifest["artifacts"]["work_plan.json"] = {
        "sha256": sha256_file(tmp_path / "work_plan.json"),
        "bytes": (tmp_path / "work_plan.json").stat().st_size,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="expected work"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="canary")

    work_digest = __import__("hashlib").sha256("\n".join(work_keys).encode()).hexdigest()
    plan = json.loads((tmp_path / "work_plan.json").read_text())
    plan["roles"]["layer_select"]["work_keys_sha256"] = work_digest
    (tmp_path / "work_plan.json").write_text(json.dumps(plan))
    manifest["artifacts"]["work_plan.json"] = {
        "sha256": sha256_file(tmp_path / "work_plan.json"),
        "bytes": (tmp_path / "work_plan.json").stat().st_size,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="canary.*reader gate"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="canary")

    manifest["status"] = "content_readout_complete"
    manifest["canary"] = False
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    verified = verify_run_root(tmp_path, expected_run_id="run-id", mode="complete")

    assert verified["status"] == "content_readout_complete"
    complete_artifacts = dict(manifest["artifacts"])
    manifest["artifacts"] = {
        name: receipt
        for name, receipt in complete_artifacts.items()
        if name != "frozen_selection.json"
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="required artifacts"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="complete")
    manifest["artifacts"] = complete_artifacts
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "gate_report.json").write_text("{}")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_run_root(tmp_path, expected_run_id="run-id", mode="complete")
