from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from interface_formatting_study import decision_binding_logit_lens_cli as logit_cli
from interface_formatting_study.decision_binding_logit_lens_cli import (
    EXPECTED_FORMATS,
    _merge_telemetry_reports,
    _parity_report_from_frame,
    _validate_attempt_chain,
    audit_tokenizer_bundle,
    build_parser,
    build_logit_lens_identity,
    build_source_ledger,
    execute_model_run,
    load_design_audit_receipts,
    load_token_audit,
    load_prepared_bundle,
    prepare_bundle,
    run_atomic_chunks,
    score_lens_chunk,
    select_startup_work_keys,
    verify_run_root,
)
from interface_formatting_study.decision_binding_logit_lens import LayerwisePathScores
from interface_formatting_study.model_profiles import get_model_profile
from interface_formatting_study.shards import ShardStore
from interface_formatting_study.run_identity import sha256_file


LETTER_INSTRUCTION = "Return only the letter (A, B, C, or D)."
TEXT_INSTRUCTION = "Return only the exact answer text, not its letter."


def _prompt(instruction: str, item: str, wrapper: str) -> str:
    return (
        f"Question {item} in {wrapper}\n"
        "A. alpha\nB. beta\nC. gamma\nD. delta\n"
        f"{instruction}\nAnswer: "
    )


def _source_rows(*, split: str = "train", formats: tuple[str, ...] = ("plain", "wrapped")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = ["alpha", "beta", "gamma", "delta"]
    for wrapper in formats:
        common = {
            "item_id": "item-1",
            "subject": "subject",
            "split": split,
            "wrapper_name": wrapper,
            "variant": 0,
            "position_shift": 0,
            "label_shift": 0,
            "source_prompt_sha256": "source-" + wrapper,
            "content_ids_by_position": [0, 1, 2, 3],
            "labels_by_position": list("ABCD"),
            "candidate_texts": candidates,
            "correct_content_id": 0,
            "choice_provenance": "canonical",
        }
        letter_prompt = _prompt(LETTER_INSTRUCTION, "item-1", wrapper)
        text_prompt = _prompt(TEXT_INSTRUCTION, "item-1", wrapper)
        calibration_prompt = _prompt(LETTER_INSTRUCTION, "calibration", wrapper)
        rows.append({
            **common,
            "work_key": f"letter|item-1|{wrapper}|0",
            "arm": "letter_intervention",
            "manipulation": "controlled_baseline",
            "prompt": letter_prompt,
            "prompt_sha256": hashlib.sha256(letter_prompt.encode()).hexdigest(),
            "calibration_prompt": calibration_prompt,
            "calibration_prompt_sha256": hashlib.sha256(calibration_prompt.encode()).hexdigest(),
            "content_free_calibration_kind": "same_wrapper_redaction",
            "content_free_fallback_reason": "",
            "raw_predicted_content_id": 0,
            "cal_predicted_content_id": 0,
            "cal_correct": True,
            "cal_entropy": 0.5,
            "candidate_predicted_content_id": None,
            "candidate_total_tie": False,
        })
        rows.append({
            **common,
            "work_key": f"text|item-1|{wrapper}|0",
            "arm": "answer_text",
            "manipulation": "controlled_baseline",
            "prompt": text_prompt,
            "prompt_sha256": hashlib.sha256(text_prompt.encode()).hexdigest(),
            "calibration_prompt": None,
            "calibration_prompt_sha256": None,
            "content_free_calibration_kind": None,
            "content_free_fallback_reason": None,
            "raw_predicted_content_id": None,
            "cal_predicted_content_id": None,
            "cal_correct": None,
            "cal_entropy": None,
            "candidate_predicted_content_id": 0,
            "candidate_total_tie": False,
        })
    return pd.DataFrame(rows)


def test_build_source_ledger_refuses_protected_split_before_prompt_access():
    source = _source_rows(split="test")
    source["prompt"] = [pytest.fail] * len(source)

    with pytest.raises(ValueError, match="protected split"):
        build_source_ledger(
            source,
            expected_split_items={"test": 1},
            expected_formats=("plain", "wrapped"),
        )


def test_prepare_bundle_rejects_protected_causal_identity_before_parquet_access(
    tmp_path, monkeypatch
):
    causal_root = tmp_path / "causal"
    raw_path = causal_root / "raw" / "causal_behavior.parquet"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(b"must-not-be-opened")
    identity = {
        "semantic_run_id": "protected-run",
        "semantic_sha256": "a" * 64,
        "model": {
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
        },
        "experiment_config": {"design_splits": ["test"]},
    }
    (causal_root / "semantic_identity.json").write_text(json.dumps(identity))
    (causal_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "canary": False,
                "semantic_identity": identity,
                "design": {"sha256": "b" * 64, "applicability_sha256": "c" * 64},
            }
        )
    )
    raw_path.with_name(raw_path.name + ".manifest.json").write_text(
        json.dumps(
            {
                "semantic_run_id": identity["semantic_run_id"],
                "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "row_count": 1,
            }
        )
    )
    parquet_reads = 0

    def reject_parquet_read(*_args, **_kwargs):
        nonlocal parquet_reads
        parquet_reads += 1
        pytest.fail("protected source parquet was opened")

    monkeypatch.setattr(pd, "read_parquet", reject_parquet_read)

    with pytest.raises(RuntimeError, match="protected split"):
        prepare_bundle(
            causal_root,
            tmp_path / "v2",
            tmp_path / "v3",
            tmp_path / "design.json",
            tmp_path / "output",
        )
    assert parquet_reads == 0


def test_build_source_ledger_pairs_contracts_and_persists_identity_eligibility():
    source = _source_rows()

    ledger = build_source_ledger(
        source,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )

    assert ledger["block_work_key"].tolist() == ["item-1|plain", "item-1|wrapped"]
    assert ledger["letter_prompt"].str.endswith("\nAnswer: ").all()
    assert ledger["text_prompt"].str.endswith("\nAnswer: ").all()
    assert ledger["calibration_prompt"].str.endswith("\nAnswer: ").all()
    assert ledger["candidate_surfaces_match_plain"].tolist() == [True, True]
    assert ledger["cross_wrapper_identity_evaluable"].tolist() == [True, True]
    assert ledger["cross_wrapper_identity_reason"].tolist() == ["eligible", "eligible"]
    assert ledger["letter_prompt_sha256"].map(len).eq(64).all()
    assert ledger["text_prompt_sha256"].map(len).eq(64).all()


def test_build_source_ledger_preserves_plain_answer_prefix_without_trailing_space():
    source = _source_rows()
    plain = source["wrapper_name"].eq("plain")
    for index in source.index[plain]:
        if source.at[index, "arm"] == "letter_intervention":
            source.at[index, "prompt"] = source.at[index, "prompt"].removesuffix(" ")
            source.at[index, "calibration_prompt"] = source.at[
                index, "calibration_prompt"
            ].removesuffix(" ")
            source.at[index, "prompt_sha256"] = hashlib.sha256(
                source.at[index, "prompt"].encode()
            ).hexdigest()
            source.at[index, "calibration_prompt_sha256"] = hashlib.sha256(
                source.at[index, "calibration_prompt"].encode()
            ).hexdigest()
        else:
            source.at[index, "prompt"] = source.at[index, "prompt"].removesuffix(" ")
            source.at[index, "prompt_sha256"] = hashlib.sha256(
                source.at[index, "prompt"].encode()
            ).hexdigest()

    ledger = build_source_ledger(
        source,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )

    plain_row = ledger[ledger["wrapper_name"].eq("plain")].iloc[0]
    assert plain_row["letter_prompt"].endswith("\nAnswer:")
    assert plain_row["text_prompt"].endswith("\nAnswer:")
    assert plain_row["calibration_prompt"].endswith("\nAnswer:")


def test_build_source_ledger_retains_malformed_secondary_calibration_explicitly():
    source = _source_rows()
    row = source.index[
        source["arm"].eq("letter_intervention")
        & source["wrapper_name"].eq("wrapped")
    ][0]
    source.at[row, "calibration_prompt"] = source.at[row, "calibration_prompt"].replace(
        "Answer:", "OPTION_D_PLACEHOLDERswer:"
    )
    source.at[row, "calibration_prompt_sha256"] = hashlib.sha256(
        source.at[row, "calibration_prompt"].encode()
    ).hexdigest()

    ledger = build_source_ledger(
        source,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )

    wrapped = ledger[ledger["wrapper_name"].eq("wrapped")].iloc[0]
    assert not bool(wrapped["calibration_lens_evaluable"])
    assert wrapped["calibration_lens_ineligibility_reason"] == "missing_exact_answer_prefix"


def test_build_source_ledger_rejects_nonterminal_instruction_drift_and_bad_mapping():
    drift = _source_rows()
    text_index = drift.index[drift["arm"].eq("answer_text")][0]
    drift.at[text_index, "prompt"] = drift.at[text_index, "prompt"].replace("Question", "Changed")
    drift.at[text_index, "prompt_sha256"] = hashlib.sha256(
        drift.at[text_index, "prompt"].encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="terminal instruction"):
        build_source_ledger(
            drift,
            expected_split_items={"train": 1},
            expected_formats=("plain", "wrapped"),
        )

    mapping = _source_rows()
    mapping.at[0, "content_ids_by_position"] = [0, 0, 2, 3]
    with pytest.raises(ValueError, match="content mapping"):
        build_source_ledger(
            mapping,
            expected_split_items={"train": 1},
            expected_formats=("plain", "wrapped"),
        )


def test_build_source_ledger_retains_surface_and_override_ineligibility():
    source = _source_rows()
    wrapped = source["wrapper_name"].eq("wrapped")
    source.loc[wrapped, "choice_provenance"] = "displayed_choice_override"
    for index in source.index[wrapped]:
        source.at[index, "candidate_texts"] = ["ALPHA", "beta", "gamma", "delta"]

    ledger = build_source_ledger(
        source,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )

    wrapped_row = ledger[ledger["wrapper_name"].eq("wrapped")].iloc[0]
    assert not bool(wrapped_row["candidate_surfaces_match_plain"])
    assert not bool(wrapped_row["cross_wrapper_identity_evaluable"])
    assert wrapped_row["cross_wrapper_identity_reason"] == "mapping_unproven_and_surface_mismatch"


def test_actual_displayed_choice_audit_provenance_does_not_certify_cross_wrapper_identity():
    source = _source_rows()
    wrapped = source["wrapper_name"].eq("wrapped")
    source.loc[wrapped, "choice_provenance"] = (
        "gpt-5.6-luna:xhigh:019f-recorded-thread"
    )

    ledger = build_source_ledger(
        source,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )

    wrapped_row = ledger[ledger["wrapper_name"].eq("wrapped")].iloc[0]
    assert not bool(wrapped_row["cross_wrapper_identity_evaluable"])
    assert wrapped_row["cross_wrapper_identity_reason"] == "mapping_unproven"


class _ExactTokenizer:
    all_special_ids = [0]
    name_or_path = "mistral-fixture"

    def __init__(self):
        self._ids: dict[str, int] = {}
        self._pieces: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if not text:
            return []
        # Character tokens make every exact byte boundary deterministic while
        # still providing shared prefixes and multi-token candidates.
        result = []
        for piece in text:
            if piece not in self._ids:
                token_id = len(self._ids) + 1
                self._ids[piece] = token_id
                self._pieces[token_id] = piece
            result.append(self._ids[piece])
        return result

    def decode(self, token_ids, **_kwargs):
        return "".join(self._pieces[int(token_id)] for token_id in token_ids)


def test_token_audit_is_complete_hash_bound_and_persists_structural_eligibility(tmp_path):
    ledger = build_source_ledger(
        _source_rows(),
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    ledger.at[0, "candidate_texts"] = ["alpha", "alphabet", "C", "delta"]
    _write_prepared_bundle(tmp_path / "bundle", ledger)

    manifest = audit_tokenizer_bundle(
        tmp_path / "bundle",
        tmp_path / "tokenization",
        tokenizer=_ExactTokenizer(),
        max_context_tokens=4096,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    loaded_manifest, audit = load_token_audit(
        tmp_path / "tokenization",
        bundle_root=tmp_path / "bundle",
        expected_blocks=2,
    )

    assert manifest == loaded_manifest
    assert len(audit) == 6  # letter, text, and calibration for every source block
    assert set(audit["contract"]) == {"letter", "text", "calibration"}
    first = audit[
        audit["block_work_key"].eq("item-1|plain")
        & audit["contract"].eq("letter")
    ].iloc[0]
    assert list(first["candidate_path_token_counts"]) == [5, 8, 1, 5]
    assert isinstance(first["observation_token_text"], str)
    assert len(first["observation_character_span"]) == 2
    assert bool(first["prompt_roundtrip_exact"])
    assert len(first["decoded_prompt_sha256"]) == 64
    assert "prefix_collision" in first["candidate_identity_ineligibility_reasons"]
    assert "label_like_candidate" in first["candidate_identity_ineligibility_reasons"]
    assert not bool(first["primary_contrast_evaluable"])
    assert manifest["final_599_opened"] is False
    assert manifest["rows"] == 6
    assert manifest["continuation_tokenization_policy"].endswith(
        "without_implicit_prefix"
    )
    assert len(manifest["audit_sha256"]) == 64

    path = tmp_path / "tokenization" / "continuation_audit.parquet"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        load_token_audit(
            tmp_path / "tokenization",
            bundle_root=tmp_path / "bundle",
            expected_blocks=2,
        )


def test_structural_startup_selection_is_deterministic_and_outcome_free():
    rows = []
    for index in range(12):
        rows.append(
            {
                "block_work_key": f"item-{index}|plain",
                "contract": "letter",
                "candidate_path_token_counts": [1, 2 + index, 1, 1],
                "has_shared_first_token": index % 2 == 0,
                "has_prefix_collision": index == 3,
                "has_unicode_candidate": index == 4,
                "has_punctuation_candidate": index == 5,
                "candidate_identity_evaluable": index != 3,
            }
        )
    audit = pd.DataFrame(rows)

    selected = select_startup_work_keys(audit, count=8)
    shuffled = select_startup_work_keys(audit.sample(frac=1, random_state=7), count=8)

    assert selected == shuffled
    assert len(selected) == len(set(selected)) == 8
    assert "item-3|plain" in selected
    assert "item-4|plain" in selected
    assert "item-5|plain" in selected


def test_semantic_identity_binds_hardware_and_batching_but_not_max_chunks(tmp_path):
    dataset = tmp_path / "prompt_ledger.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(dataset, index=False)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    profile = get_model_profile("mistral")
    base = {
        "stage": "discovery",
        "runtime_environment": {
            "device_type": "cuda",
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "gpu_compute_capability": "9.0",
            "bf16_supported": True,
            "attention_backend": "sdpa",
            "torch_version": "2.7.1+cu126",
            "cuda_version": "12.6",
            "transformers_version": "4.53.2",
            "tokenizers_version": "0.21.2",
            "inference_dtype": "bfloat16",
        },
        "batch_size": 8,
        "max_batch_tokens": 24000,
        "capture_chunk_size": 8,
    }
    first = build_logit_lens_identity(
        profile, config={**base, "max_chunks_this_invocation": 1},
        dataset_path=dataset, source_paths=[source],
    )
    resumed = build_logit_lens_identity(
        profile, config={**base, "max_chunks_this_invocation": None},
        dataset_path=dataset, source_paths=[source],
    )
    h100 = dict(base)
    h100["runtime_environment"] = dict(base["runtime_environment"])
    h100["runtime_environment"]["gpu_name"] = "NVIDIA A100-SXM4-80GB"
    other_gpu = build_logit_lens_identity(
        profile, config=h100, dataset_path=dataset, source_paths=[source],
    )
    other_batch = build_logit_lens_identity(
        profile, config={**base, "batch_size": 4},
        dataset_path=dataset, source_paths=[source],
    )

    assert first.semantic_run_id == resumed.semantic_run_id
    assert first.semantic_run_id != other_gpu.semantic_run_id
    assert first.semantic_run_id != other_batch.semantic_run_id


def test_design_manifest_binds_both_option_and_displayed_choice_audits(tmp_path):
    path = tmp_path / "design.parquet.manifest.json"
    path.write_text(
        json.dumps(
            {
                "sha256": "a" * 64,
                "applicability": {"sha256": "b" * 64},
                "splits": ["train", "validation"],
                "retained_items": 2401,
                "option_audit": {"manifest_sha256": "c" * 64},
                "choice_audit": {"manifest_sha256": "d" * 64},
            }
        )
    )

    receipt = load_design_audit_receipts(
        path,
        expected_design_sha="a" * 64,
        expected_applicability_sha="b" * 64,
    )

    assert receipt["option_audit_manifest_sha256"] == "c" * 64
    assert receipt["choice_audit_manifest_sha256"] == "d" * 64
    assert receipt["design_manifest_sha256"] == sha256_file(path)

    with pytest.raises(RuntimeError, match="design manifest identity"):
        load_design_audit_receipts(
            path,
            expected_design_sha="0" * 64,
            expected_applicability_sha="b" * 64,
        )


def test_score_chunk_batches_equal_length_roots_and_persists_calibration_and_masks(tmp_path):
    ledger = build_source_ledger(
        _source_rows(),
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    _write_prepared_bundle(tmp_path / "bundle", ledger)
    audit_tokenizer_bundle(
        tmp_path / "bundle",
        tmp_path / "tokenization",
        tokenizer=_ExactTokenizer(),
        max_context_tokens=4096,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    _, audit = load_token_audit(
        tmp_path / "tokenization", bundle_root=tmp_path / "bundle", expected_blocks=2
    )
    calls: list[list[int]] = []

    def fake_scorer(_model, audits, *, expected_layers):
        calls.append([len(value.prompt_ids) for value in audits])
        outputs = []
        for index, value in enumerate(audits):
            base = torch.tensor(
                [[-0.1 - index, -1.0, -2.0, -3.0]] * expected_layers,
                dtype=torch.float32,
            )
            outputs.append(
                LayerwisePathScores(base, base - 0.1, base - 0.2, base - 0.3, 0.0)
            )
        return outputs

    frame, parity = score_lens_chunk(
        model=object(),
        ledger=ledger,
        token_audit=audit,
        work_keys=ledger["block_work_key"].tolist(),
        batch_size=2,
        max_batch_tokens=10000,
        expected_layers=2,
        scorer=fake_scorer,
        run_scalar_oracle=False,
    )

    assert len(frame) == 2 * 2 * 2
    assert not frame.duplicated(["block_work_key", "contract", "layer"]).any()
    assert all(len(set(lengths)) == 1 for lengths in calls)
    letter = frame[frame["contract"].eq("letter")]
    assert letter["letter_calibrated_logps"].map(lambda values: len(values) == 4).all()
    assert frame["primary_contrast_evaluable"].map(type).eq(bool).all()
    assert parity["max_final_native_difference"] == 0.0


def test_score_chunk_records_bf16_cross_forward_drift_without_rejecting_valid_same_forward_scores(
    tmp_path, monkeypatch
):
    ledger = build_source_ledger(
        _source_rows(),
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    _write_prepared_bundle(tmp_path / "bundle", ledger)
    audit_tokenizer_bundle(
        tmp_path / "bundle",
        tmp_path / "tokenization",
        tokenizer=_ExactTokenizer(),
        max_context_tokens=4096,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    _, audit = load_token_audit(
        tmp_path / "tokenization", bundle_root=tmp_path / "bundle", expected_blocks=2
    )

    def scores(value: float, layers: int, *, reverse_winner: bool = False) -> LayerwisePathScores:
        row = (
            [value - 1.0, value, value - 2.0, value - 3.0]
            if reverse_winner
            else [value, value - 1.0, value - 2.0, value - 3.0]
        )
        tensor = torch.tensor(
            [row] * layers,
            dtype=torch.float32,
        )
        return LayerwisePathScores(tensor, tensor, tensor, tensor, 0.0)

    def cached_scorer(_model, audits, *, expected_layers):
        return [scores(-0.1, expected_layers) for _audit in audits]

    monkeypatch.setattr(
        logit_cli,
        "score_candidate_paths_scalar",
        lambda _model, _audit, *, expected_layers: scores(
            -0.225, expected_layers, reverse_winner=True
        ),
    )

    frame, parity = score_lens_chunk(
        model=object(),
        ledger=ledger,
        token_audit=audit,
        work_keys=ledger["block_work_key"].tolist(),
        batch_size=2,
        max_batch_tokens=10000,
        expected_layers=2,
        scorer=cached_scorer,
        run_scalar_oracle=True,
    )

    assert parity["max_final_native_difference"] == 0.0
    assert parity["max_cached_scalar_letter_difference"] == pytest.approx(1.125)
    assert np.allclose(frame["parity_cached_scalar_letter_max"], 1.125)
    assert parity["cached_scalar_letter_argmax_disagreements"] == 8
    assert parity["cached_scalar_candidate_total_argmax_disagreements"] == 8
    assert frame["cached_scalar_letter_argmax_disagreement"].all()
    assert frame["cached_scalar_candidate_total_argmax_disagreement"].all()

    forged = frame.copy()
    forged["cached_scalar_letter_argmax_comparable"] = False
    with pytest.raises(ValueError, match="argmax comparability receipt mismatch"):
        _parity_report_from_frame(forged)


def test_atomic_chunk_resume_skips_completed_work_and_max_chunks_is_invocation_only(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(dataset, index=False)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    identity = build_logit_lens_identity(
        get_model_profile("mistral"),
        config={"stage": "discovery"},
        dataset_path=dataset,
        source_paths=[source],
    )
    store = ShardStore(tmp_path / "shards", identity)
    calls: list[list[str]] = []

    def process(keys):
        calls.append(list(keys))
        return pd.DataFrame({"block_work_key": list(keys), "value": range(len(keys))})

    first = run_atomic_chunks(
        store,
        ["a", "b", "c"],
        chunk_size=1,
        max_chunks_this_invocation=1,
        process_chunk=process,
    )
    second = run_atomic_chunks(
        store,
        ["a", "b", "c"],
        chunk_size=1,
        max_chunks_this_invocation=None,
        process_chunk=process,
    )

    assert first == ["a"]
    assert second == ["b", "c"]
    assert calls == [["a"], ["b"], ["c"]]
    assert store.completed_work_keys() == {"a", "b", "c"}

    nonprefix_store = ShardStore(tmp_path / "nonprefix", identity)
    nonprefix_store.write_shard(
        pd.DataFrame({"block_work_key": ["b"], "_work_key": ["b"]}),
        work_keys=["b"],
    )
    with pytest.raises(ValueError, match="prefix"):
        run_atomic_chunks(
            nonprefix_store,
            ["a", "b", "c"],
            chunk_size=1,
            max_chunks_this_invocation=None,
            process_chunk=process,
        )


def test_parity_coverage_survives_crash_after_atomic_shard_commit(tmp_path):
    dataset = tmp_path / "dataset.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(dataset, index=False)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    identity = build_logit_lens_identity(
        get_model_profile("mistral"),
        config={"stage": "discovery"},
        dataset_path=dataset,
        source_paths=[source],
    )
    store = ShardStore(tmp_path / "shards", identity)

    def process(keys):
        rows = {
                "block_work_key": list(keys),
                "parity_final_native_max": [0.001] * len(keys),
                "parity_cached_scalar_letter_max": [0.002] * len(keys),
                "parity_cached_scalar_first_token_max": [0.003] * len(keys),
                "parity_cached_scalar_mean_token_max": [0.004] * len(keys),
                "parity_cached_scalar_total_per_token_max": [0.005] * len(keys),
                "scalar_oracle_evaluated": [True] * len(keys),
                "letter_raw_logps": [[-0.1, -1.0, -2.0, -3.0]] * len(keys),
                "candidate_first_token_logps": [[-0.2, -1.1, -2.1, -3.1]] * len(keys),
                "candidate_mean_token_logps": [[-0.2, -1.1, -2.1, -3.1]] * len(keys),
                "candidate_path_total_logps": [[-0.2, -1.1, -2.1, -3.1]] * len(keys),
            }
        for readout in (
            "letter",
            "candidate_first_token",
            "candidate_mean_token",
            "candidate_total",
        ):
            rows[f"cached_scalar_{readout}_argmax_comparable"] = [True] * len(keys)
            rows[f"cached_scalar_{readout}_argmax_disagreement"] = [False] * len(keys)
        return pd.DataFrame(rows)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_atomic_chunks(
            store,
            ["a", "b"],
            chunk_size=1,
            max_chunks_this_invocation=1,
            process_chunk=process,
            on_flush=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("simulated crash")
            ),
        )
    assert store.completed_work_keys() == {"a"}

    run_atomic_chunks(
        store,
        ["a", "b"],
        chunk_size=1,
        max_chunks_this_invocation=None,
        process_chunk=process,
    )
    report = _parity_report_from_frame(store.merge())

    assert report["covered_work_keys"] == ["a", "b"]
    assert report["scalar_oracle_work_keys"] == ["a", "b"]


def test_telemetry_merge_covers_both_startup_halves_without_relabeling():
    first = {
        "covered_work_keys": ["a", "b", "c", "d"],
        "phase_seconds": {"root": 2.0, "branch": 1.0, "scalar_oracle": 3.0},
        "gpu_utilization": {
            "root": {"samples": 2, "mean_percent": 50.0, "max_percent": 60.0},
            "branch": {"samples": 1, "mean_percent": 40.0, "max_percent": 40.0},
            "scalar_oracle": {"samples": 1, "mean_percent": 30.0, "max_percent": 30.0},
        },
        "sample_errors": 0,
    }
    second = {
        "covered_work_keys": ["e", "f", "g", "h"],
        "phase_seconds": {"root": 1.0, "branch": 2.0, "scalar_oracle": 2.0},
        "gpu_utilization": {
            "root": {"samples": 1, "mean_percent": 80.0, "max_percent": 80.0},
            "branch": {"samples": 1, "mean_percent": 60.0, "max_percent": 60.0},
            "scalar_oracle": {"samples": 1, "mean_percent": 50.0, "max_percent": 50.0},
        },
        "sample_errors": 1,
    }

    merged = _merge_telemetry_reports(first, second)

    assert merged["covered_work_keys"] == list("abcdefgh")
    assert merged["phase_seconds"]["root"] == pytest.approx(3.0)
    assert merged["gpu_utilization"]["root"] == {
        "samples": 3,
        "mean_percent": 60.0,
        "max_percent": 80.0,
    }
    assert merged["sample_errors"] == 1


def test_startup_attempt_chain_requires_interrupted_four_then_resumed_four():
    keys = list("abcdefgh")
    one_shot = [
        {
            "started_at_unix": 1.0,
            "status": "startup_complete",
            "completed_before": 0,
            "processed_work_keys": keys,
            "completed_after": 8,
            "telemetry_work_keys": keys,
        }
    ]
    with pytest.raises(ValueError, match="four-plus-four"):
        _validate_attempt_chain(one_shot, keys, mode="startup")

    resumed = [
        {
            "started_at_unix": 1.0,
            "status": "interrupted",
            "completed_before": 0,
            "processed_work_keys": keys[:4],
            "completed_after": 4,
            "telemetry_work_keys": keys[:4],
        },
        {
            "started_at_unix": 2.0,
            "status": "startup_complete",
            "completed_before": 4,
            "processed_work_keys": keys[4:],
            "completed_after": 8,
            "telemetry_work_keys": keys[4:],
        },
    ]
    _validate_attempt_chain(resumed, keys, mode="startup")


def test_execute_model_run_requires_the_frozen_eight_by_four_startup_shape():
    common = {
        "profile": "mistral",
        "batch_size": 8,
        "max_batch_tokens": 24000,
        "max_chunks_this_invocation": None,
    }
    with pytest.raises(ValueError, match="eight items"):
        execute_model_run(
            SimpleNamespace(**common, startup_items=7, capture_chunk_size=4)
        )
    with pytest.raises(ValueError, match="four-work-unit chunks"):
        execute_model_run(
            SimpleNamespace(**common, startup_items=8, capture_chunk_size=2)
        )


def _write_prepared_bundle(root: Path, ledger: pd.DataFrame) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prompt_ledger.parquet"
    ledger.to_parquet(path, index=False)
    manifest = {
        "schema_version": 1,
        "stage": "discovery",
        "model": {
            "id": "mistralai/Mistral-7B-Instruct-v0.3",
            "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
            "slug": "mistral-7b-instruct-v0.3",
        },
        "rows": len(ledger),
        "items": int(ledger["item_id"].nunique()),
        "split_items": {"train": 1},
        "formats": ["plain", "wrapped"],
        "ledger_sha256": sha256_file(path),
        "source_receipts": {"causal_scored_sha256": "a" * 64},
    }
    (root / "prepared_manifest.json").write_text(json.dumps(manifest))


def test_load_prepared_bundle_is_stage_count_and_checksum_bound(tmp_path):
    ledger = build_source_ledger(
        _source_rows(),
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )
    _write_prepared_bundle(tmp_path, ledger)

    manifest, loaded = load_prepared_bundle(
        tmp_path,
        expected_split_items={"train": 1},
        expected_formats=("plain", "wrapped"),
    )

    assert manifest["stage"] == "discovery"
    pd.testing.assert_frame_equal(loaded, ledger)
    (tmp_path / "prompt_ledger.parquet").write_bytes(
        (tmp_path / "prompt_ledger.parquet").read_bytes() + b"corrupt"
    )
    with pytest.raises(RuntimeError, match="checksum"):
        load_prepared_bundle(
            tmp_path,
            expected_split_items={"train": 1},
            expected_formats=("plain", "wrapped"),
        )


def test_cli_exposes_only_discovery_logit_lens_commands():
    parser = build_parser()
    prepared = parser.parse_args([
        "prepare", "--causal-run", "run", "--v2-bundle", "v2",
        "--v3-bundle", "v3", "--design-manifest", "design-manifest",
        "--output-dir", "out",
    ])
    audit = parser.parse_args([
        "audit-tokenizer", "--bundle", "bundle", "--output", "audit",
    ])
    run = parser.parse_args([
        "run-model", "--bundle", "bundle", "--token-audit", "audit",
    ])
    analyze = parser.parse_args(["analyze", "--run-root", "run"])
    verify = parser.parse_args([
        "verify", "--run-root", "run", "--run-id", "id", "--mode", "partial",
    ])

    assert prepared.command == "prepare"
    assert audit.command == "audit-tokenizer"
    assert run.profile == "mistral"
    assert run.batch_size == 32
    assert run.max_batch_tokens == 40000
    assert run.capture_chunk_size == 64
    assert analyze.command == "analyze"
    assert verify.mode == "partial"
    with pytest.raises(SystemExit):
        parser.parse_args([
            "run-model", "--bundle", "bundle", "--token-audit", "audit",
            "--stage", "confirmation",
        ])
    with pytest.raises(SystemExit):
        parser.parse_args([
            "analyze", "--run-root", "run", "--bootstrap-samples", "100",
        ])


def test_verify_run_root_reconciles_complete_identity_shards_and_scores(tmp_path):
    # Minimal one-block, two-contract, two-layer fixture exercises the public verifier
    # without manufacturing model behavior.
    identity = {
        "model": {
            "id": "model", "revision": "revision", "slug": "mistral-7b-instruct-v0.3",
            "expected_layers": 2,
        },
        "experiment_config": {
            "stage": "discovery",
            "prepared_manifest_sha256": "pending",
            "continuation_audit_sha256": "pending",
            "tokenization_manifest_sha256": "pending",
            "ordered_work_keys_sha256": hashlib.sha256(
                json.dumps(
                    ["item|plain"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "analysis_policy": {"bootstrap_samples": 5000, "seed": 1729},
            "runtime_environment": {
                "device_type": "cuda", "gpu_name": "NVIDIA H100 80GB HBM3",
                "gpu_compute_capability": "9.0", "bf16_supported": True,
                "attention_backend": "sdpa", "torch_version": "2.7.1+cu126",
                "cuda_version": "12.6", "transformers_version": "4.53.2",
                "tokenizers_version": "0.21.2", "inference_dtype": "bfloat16",
            },
        },
    }
    (tmp_path / "prepared_manifest.json").write_text(json.dumps({"stage": "discovery"}))
    pd.DataFrame(
        [
            {
                "block_work_key": "item|plain",
                "contract": contract,
                "candidate_path_evaluable": [True, True, True, True],
            }
            for contract in ("letter", "text")
        ]
    ).to_parquet(tmp_path / "continuation_audit.parquet", index=False)
    (tmp_path / "tokenization_manifest.json").write_text(
        json.dumps(
            {
                "stage": "discovery",
                "audit_sha256": sha256_file(tmp_path / "continuation_audit.parquet"),
            }
        )
    )
    identity["experiment_config"]["prepared_manifest_sha256"] = sha256_file(
        tmp_path / "prepared_manifest.json"
    )
    identity["experiment_config"]["continuation_audit_sha256"] = sha256_file(
        tmp_path / "continuation_audit.parquet"
    )
    identity["experiment_config"]["tokenization_manifest_sha256"] = sha256_file(
        tmp_path / "tokenization_manifest.json"
    )
    semantic_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_id = semantic_digest[:20]
    identity = {
        "semantic_run_id": run_id,
        "semantic_sha256": semantic_digest,
        **identity,
    }
    (tmp_path / "semantic_identity.json").write_text(json.dumps(identity))
    (tmp_path / "analysis_spec.json").write_text(
        json.dumps(identity["experiment_config"]["analysis_policy"])
    )
    (tmp_path / "work_plan.json").write_text(
        json.dumps(
            {
                "work_keys": ["item|plain"],
                "ordered_work_keys_sha256": identity["experiment_config"][
                    "ordered_work_keys_sha256"
                ],
                "expected_rows": 4,
            }
        )
    )
    scores = pd.DataFrame([
        {
            "_work_key": "item|plain", "block_work_key": "item|plain",
            "contract": contract, "layer": layer,
            "letter_raw_logps": [-0.1, -1.0, -2.0, -3.0],
            "candidate_path_total_logps": [-0.2, -1.1, -2.1, -3.1],
            "candidate_mean_token_logps": [-0.2, -1.1, -2.1, -3.1],
            "candidate_first_token_logps": [-0.2, -1.1, -2.1, -3.1],
            "candidate_path_evaluable": [True, True, True, True],
            "candidate_ranking_evaluable": True,
            "primary_contrast_evaluable": True,
            "primary_contrast_exclusion_reason": "eligible",
            "first_token_contrast_evaluable": True,
            "first_token_contrast_exclusion_reason": "eligible",
            "letter_calibration_evaluable": True,
            "letter_calibration_exclusion_reason": "eligible",
            "letter_calibrated_logps": (
                [-0.05, -0.5, -1.0, -1.5]
                if contract == "letter"
                else [float("nan")] * 4
            ),
            "parity_final_native_max": 0.0,
            "parity_cached_scalar_letter_max": 0.0,
            "parity_cached_scalar_first_token_max": 0.0,
            "parity_cached_scalar_mean_token_max": 0.0,
            "parity_cached_scalar_total_per_token_max": 0.0,
            "scalar_oracle_evaluated": False,
        }
        for contract in ("letter", "text") for layer in range(2)
    ])
    for readout in (
        "letter",
        "candidate_first_token",
        "candidate_mean_token",
        "candidate_total",
    ):
        scores[f"cached_scalar_{readout}_argmax_comparable"] = False
        scores[f"cached_scalar_{readout}_argmax_disagreement"] = False
    shard = tmp_path / "shards" / "lens" / "shard-fixture"
    shard.mkdir(parents=True)
    scores.to_parquet(shard / "data.parquet", index=False)
    shard_manifest = {
        "shard_schema_version": 1,
        "semantic_run_id": run_id,
        "semantic_sha256": semantic_digest,
        "model_id": "model", "model_revision": "revision",
        "work_keys": ["item|plain"],
        "data_sha256": sha256_file(shard / "data.parquet"),
        "row_count": len(scores),
    }
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    scores.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    parity = {
        "schema_version": 2,
        "tolerance": 0.02,
        "cross_forward_sensitivity_policy": "record_only_bf16_execution_shape_sensitivity",
        "covered_work_keys": ["item|plain"],
        "scalar_oracle_work_keys": [],
        "max_final_native_difference": 0.0,
        "max_cached_scalar_letter_difference": 0.0,
        "max_cached_scalar_first_token_difference": 0.0,
        "max_cached_scalar_mean_token_difference": 0.0,
        "max_cached_scalar_total_per_token_difference": 0.0,
        **{
            f"cached_scalar_{readout}_argmax_{suffix}": 0
            for readout in (
                "letter",
                "candidate_first_token",
                "candidate_mean_token",
                "candidate_total",
            )
            for suffix in ("comparisons", "disagreements")
        },
    }
    (tmp_path / "parity_report.json").write_text(json.dumps(parity))
    telemetry = {
        "schema_version": 1,
        "semantic_run_id": run_id,
        "completed_work_units": 1,
        "root_input_tokens": 10,
        "branch_input_tokens": 2,
        "peak_vram_bytes": 1024,
        "phase_seconds": {"root": 1.0, "branch": 0.5, "scalar_oracle": 0.0},
        "gpu_utilization": {
            "root": {"samples": 2, "mean_percent": 70.0, "max_percent": 80.0},
            "branch": {"samples": 1, "mean_percent": 60.0, "max_percent": 60.0},
            "scalar_oracle": {"samples": 0, "mean_percent": None, "max_percent": None},
        },
        "covered_work_keys": ["item|plain"],
    }
    (tmp_path / "telemetry_report.json").write_text(json.dumps(telemetry))
    progress = {
        "semantic_run_id": run_id,
        "completed_work_units": 1,
        "root_input_tokens": 10,
        "branch_input_tokens": 2,
    }
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "attempt.json").write_text(
        json.dumps(
            {
                "semantic_run_id": run_id,
                "status": "complete",
                "started_at_unix": 1.0,
                "completed_before": 0,
                "processed_work_keys": ["item|plain"],
                "completed_after": 1,
                "telemetry_work_keys": ["item|plain"],
            }
        )
    )
    manifest = {
        "status": "complete",
        "semantic_identity": identity,
        "expected_work_keys": ["item|plain"],
        "expected_rows": 4,
        "completed_work_units": 1,
        "artifacts": {
            "layerwise_scores_sha256": sha256_file(tmp_path / "layerwise_scores.parquet"),
            "parity_report_sha256": sha256_file(tmp_path / "parity_report.json"),
            "telemetry_report_sha256": sha256_file(tmp_path / "telemetry_report.json"),
            "progress_sha256": sha256_file(tmp_path / "progress.json"),
        },
        "claim": "descriptive_layerwise_logit_lens_not_causal",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    receipt = verify_run_root(
        tmp_path,
        expected_run_id=run_id,
        mode="complete",
        require_analysis=False,
    )

    assert receipt == {"status": "complete", "work_units": 1, "rows": 4}
    with pytest.raises(ValueError, match="analysis artifact manifest is required"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    analysis_names = {
        "analysis_summary": "analysis_summary.json",
        "interpretation_memo": "interpretation_memo.md",
        "quality_gates": "quality_gates.json",
        "calibrated_letter": "calibrated_letter.csv",
        "diagnostics": "diagnostics.csv",
        "primary_contrasts": "primary_contrasts.csv",
        "secondary_contrasts": "secondary_contrasts.csv",
        "trajectories": "trajectories.csv",
        "timing": "timing.csv",
        "timing_distributions": "timing_distributions.csv",
        "trajectory_figure": "trajectory_2x2.png",
        "strata": "strata.csv",
    }
    for filename in analysis_names.values():
        (analysis_dir / filename).write_bytes(f"fixture:{filename}".encode())
    manifest["artifacts"]["analysis"] = {
        name: {
            "path": f"analysis/{filename}",
            "sha256": sha256_file(analysis_dir / filename),
        }
        for name, filename in analysis_names.items()
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    assert verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")["rows"] == 4
    rebound = json.loads(json.dumps(manifest))
    rebound["artifacts"]["analysis"]["quality_gates"] = dict(
        rebound["artifacts"]["analysis"]["analysis_summary"]
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(rebound))
    with pytest.raises(ValueError, match="canonical path"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (analysis_dir / "quality_gates.json").write_text("tampered")
    with pytest.raises(ValueError, match="analysis artifact"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")
    (analysis_dir / "quality_gates.json").write_bytes(
        b"fixture:quality_gates.json"
    )
    sensitive = scores.copy()
    for column in (
        "parity_cached_scalar_letter_max",
        "parity_cached_scalar_first_token_max",
        "parity_cached_scalar_mean_token_max",
        "parity_cached_scalar_total_per_token_max",
    ):
        sensitive[column] = 0.125
    sensitive.to_parquet(shard / "data.parquet", index=False)
    shard_manifest["data_sha256"] = sha256_file(shard / "data.parquet")
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    sensitive.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    manifest["artifacts"]["layerwise_scores_sha256"] = sha256_file(
        tmp_path / "layerwise_scores.parquet"
    )
    sensitive_parity = _parity_report_from_frame(sensitive)
    (tmp_path / "parity_report.json").write_text(json.dumps(sensitive_parity))
    manifest["artifacts"]["parity_report_sha256"] = sha256_file(
        tmp_path / "parity_report.json"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    assert verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")["rows"] == 4

    hard_failure = sensitive.copy()
    hard_failure["parity_final_native_max"] = 0.03
    hard_failure.to_parquet(shard / "data.parquet", index=False)
    shard_manifest["data_sha256"] = sha256_file(shard / "data.parquet")
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    hard_failure.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    manifest["artifacts"]["layerwise_scores_sha256"] = sha256_file(
        tmp_path / "layerwise_scores.parquet"
    )
    hard_parity = _parity_report_from_frame(hard_failure)
    (tmp_path / "parity_report.json").write_text(json.dumps(hard_parity))
    manifest["artifacts"]["parity_report_sha256"] = sha256_file(
        tmp_path / "parity_report.json"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same-forward final/native parity"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")

    scores.to_parquet(shard / "data.parquet", index=False)
    shard_manifest["data_sha256"] = sha256_file(shard / "data.parquet")
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    scores.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    manifest["artifacts"]["layerwise_scores_sha256"] = sha256_file(
        tmp_path / "layerwise_scores.parquet"
    )
    (tmp_path / "parity_report.json").write_text(json.dumps(parity))
    manifest["artifacts"]["parity_report_sha256"] = sha256_file(
        tmp_path / "parity_report.json"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    incomplete_parity = {**parity, "covered_work_keys": []}
    (tmp_path / "parity_report.json").write_text(json.dumps(incomplete_parity))
    manifest["artifacts"]["parity_report_sha256"] = sha256_file(
        tmp_path / "parity_report.json"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parity coverage"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")
    (tmp_path / "parity_report.json").write_text(json.dumps(parity))
    manifest["artifacts"]["parity_report_sha256"] = sha256_file(
        tmp_path / "parity_report.json"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    forged_identity = json.loads(json.dumps(identity))
    forged_identity["experiment_config"]["runtime_environment"]["gpu_name"] = "A100"
    (tmp_path / "semantic_identity.json").write_text(json.dumps(forged_identity))
    manifest["semantic_identity"] = forged_identity
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic identity digest"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")
    (tmp_path / "semantic_identity.json").write_text(json.dumps(identity))
    manifest["semantic_identity"] = identity
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    masked = scores.copy()
    candidate_columns = (
        "candidate_path_total_logps",
        "candidate_mean_token_logps",
        "candidate_first_token_logps",
    )
    for column in candidate_columns:
        masked[column] = pd.Series([[float("nan")] * 4 for _ in range(len(masked))])
    masked["primary_contrast_evaluable"] = False
    masked["primary_contrast_exclusion_reason"] = "prefix_collision"
    masked["first_token_contrast_evaluable"] = False
    masked["first_token_contrast_exclusion_reason"] = "shared_or_ineligible_first_token"
    masked.to_parquet(shard / "data.parquet", index=False)
    shard_manifest["data_sha256"] = sha256_file(shard / "data.parquet")
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    masked.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    manifest["artifacts"]["layerwise_scores_sha256"] = sha256_file(
        tmp_path / "layerwise_scores.parquet"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="non-finite candidate path"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")

    corrupted_mask = masked.copy()
    corrupted_mask["candidate_path_evaluable"] = pd.Series(
        [[False, False, False, False] for _ in range(len(corrupted_mask))]
    )
    corrupted_mask.to_parquet(shard / "data.parquet", index=False)
    shard_manifest["data_sha256"] = sha256_file(shard / "data.parquet")
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    corrupted_mask.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    manifest["artifacts"]["layerwise_scores_sha256"] = sha256_file(
        tmp_path / "layerwise_scores.parquet"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="authenticated continuation audit"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")

    letter_corrupt = scores.copy()
    letter_corrupt.at[0, "letter_raw_logps"] = [float("nan"), -1.0, -2.0, -3.0]
    letter_corrupt.to_parquet(shard / "data.parquet", index=False)
    shard_manifest["data_sha256"] = sha256_file(shard / "data.parquet")
    (shard / "manifest.json").write_text(json.dumps(shard_manifest))
    letter_corrupt.to_parquet(tmp_path / "layerwise_scores.parquet", index=False)
    manifest["artifacts"]["layerwise_scores_sha256"] = sha256_file(tmp_path / "layerwise_scores.parquet")
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="non-finite"):
        verify_run_root(tmp_path, expected_run_id=run_id, mode="complete")


def test_frame_reconciliation_handles_nested_parquet_arrays():
    left = pd.DataFrame(
        {"candidate_token_ids": [np.array([np.array([1, 2]), np.array([3])], dtype=object)]}
    )
    right = pd.DataFrame({"candidate_token_ids": [[[1, 2], [3]]]})

    assert logit_cli._frames_match(left, right)


def test_expected_formats_are_exactly_the_existing_nine():
    assert EXPECTED_FORMATS == (
        "plain", "csv_inline", "graphql_query", "html_form", "ini_file",
        "key_equals", "protobuf_msg", "shell_heredoc", "toml_config",
    )
