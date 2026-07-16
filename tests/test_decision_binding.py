from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd
import pytest
import torch

from interface_formatting_study.causal_design import balanced_assignments
from interface_formatting_study.decision_binding import (
    CheckpointIndices,
    ProbeBank,
    build_patch_vectors,
    capture_layer_readouts,
    evaluate_probe_bank,
    fit_probe_bank,
    load_probe_bank,
    patch_target_labels,
    prepare_patch_pair_ledger,
    prepare_readout_ledger,
    probe_subspace_basis,
    projected_replacement,
    resolve_readout_checkpoints,
    run_patch_pair,
    save_probe_bank,
    select_readout_layers,
    target_margin,
    validate_canonical_sources,
    winner_coordinates,
)


_SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "


class OffsetTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def __init__(self, *, mismatch: bool = False, cross_instruction: bool = False):
        self.mismatch = mismatch
        self.cross_instruction = cross_instruction

    def _pieces(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in re.finditer(r"\S+", text):
            ids.append(len(ids) + 10)
            offsets.append(match.span())
        if self.cross_instruction:
            marker = text.rfind("Return only the letter")
            content_end = len(text[:marker].rstrip())
            index = next(i for i, (start, end) in enumerate(offsets) if start <= content_end - 1 < end)
            offsets[index] = (offsets[index][0], marker + 6)
        return ids, offsets

    def encode(self, text: str, add_special_tokens: bool = False):
        ids, _ = self._pieces(text)
        return [*ids, 999] if self.mismatch else ids

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        ids, offsets = self._pieces(text)
        output = {"input_ids": ids}
        if return_offsets_mapping:
            output["offset_mapping"] = offsets
        return output


class CharacterOffsetTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "right"

    def __init__(self):
        self.vocab = {"<pad>": 0}

    def _id(self, character: str) -> int:
        if character not in self.vocab:
            self.vocab[character] = len(self.vocab)
        return self.vocab[character]

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self._id(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        output = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


def _scored_block(
    *,
    item_id: str,
    split: str,
    wrapper: str = "plain",
    separable: bool = True,
    duplicate_text: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = ["Photosynthesis", "photo-synthesis", "photosynthesis.", "Respiration"]
    if duplicate_text:
        candidates[2] = candidates[0]
    assignments = balanced_assignments() if separable else balanced_assignments()[:1]
    rows: list[dict[str, object]] = []
    for assignment in assignments:
        prompt = (
            f"Question {item_id} {wrapper}\n"
            + "\n".join(
                f"{label}) {candidates[content_id]}"
                for label, content_id in zip(
                    assignment.labels_by_position,
                    assignment.content_ids_by_position,
                    strict=True,
                )
            )
            + _SUFFIX
        )
        predicted_label = "B"
        predicted_position = assignment.labels_by_position.index(predicted_label)
        predicted_content = assignment.content_ids_by_position[predicted_position]
        rows.append(
            {
                "work_key": f"letter|{item_id}|{wrapper}|{assignment.variant}",
                "item_id": item_id,
                "subject": "biology",
                "split": split,
                "wrapper_name": wrapper,
                "arm": "letter_intervention",
                "variant": assignment.variant,
                "manipulation": assignment.manipulation,
                "position_shift": assignment.position_shift,
                "label_shift": assignment.label_shift,
                "template_scope": "source_prompt_counterfactual",
                "source_prompt_sha256": "",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "content_ids_by_position": list(assignment.content_ids_by_position),
                "labels_by_position": list(assignment.labels_by_position),
                "candidate_texts": candidates,
                "correct_content_id": 0,
                "correct_position": list(assignment.content_ids_by_position).index(0),
                "correct_label": assignment.labels_by_position[
                    list(assignment.content_ids_by_position).index(0)
                ],
                "raw_predicted_label": predicted_label,
                "raw_predicted_position": predicted_position,
                "raw_predicted_content_id": predicted_content,
                "raw_tie": False,
                **{
                    f"raw_score_{label}": score
                    for label, score in zip("ABCD", (1.0, 4.0, 2.0, 0.0), strict=True)
                },
                **{f"bias_score_{label}": 0.0 for label in "ABCD"},
                **{f"cal_score_{label}": float(4 - index) for index, label in enumerate("ABCD")},
            }
        )
    source_sha = str(rows[0]["prompt_sha256"])
    for row in rows:
        row["source_prompt_sha256"] = source_sha
    applicability = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "subject": "biology",
                "split": split,
                "wrapper_name": wrapper,
                "source_prompt_sha256": rows[0]["prompt_sha256"],
                "baseline_applicable": True,
                "position_applicable": separable,
                "label_applicable": separable,
                "not_applicable_reason": "" if separable else "option_structure_not_resolved",
            }
        ]
    )
    return pd.DataFrame(rows), applicability


def test_winner_coordinates_never_depend_on_answer_text_spelling():
    result = winner_coordinates(
        label="C",
        labels_by_position=["D", "A", "C", "B"],
        content_ids_by_position=[3, 0, 2, 1],
    )

    assert result.label_index == 2
    assert result.position == 2
    assert result.content_id == 2


def test_winner_coordinates_reject_non_permutations_and_unknown_labels():
    with pytest.raises(ValueError, match="labels_by_position"):
        winner_coordinates(
            label="A",
            labels_by_position=["A", "A", "C", "D"],
            content_ids_by_position=[0, 1, 2, 3],
        )
    with pytest.raises(ValueError, match="content_ids_by_position"):
        winner_coordinates(
            label="A",
            labels_by_position=["A", "B", "C", "D"],
            content_ids_by_position=[0, 0, 2, 3],
        )
    with pytest.raises(ValueError, match="Unknown winner label"):
        winner_coordinates(
            label="E",
            labels_by_position=["A", "B", "C", "D"],
            content_ids_by_position=[0, 1, 2, 3],
        )


def test_readout_checkpoints_use_offsets_from_the_complete_prompt():
    tokenizer = OffsetTokenizer()
    prompt = "A) alpha B) beta" + _SUFFIX

    checkpoints = resolve_readout_checkpoints(tokenizer, prompt)

    assert checkpoints.sequence_length == len(tokenizer.encode(prompt))
    assert checkpoints.format_end < checkpoints.answer_prefix_end
    assert checkpoints.padded(padded_length=checkpoints.sequence_length + 5, padding_side="left") == (
        checkpoints.format_end + 5,
        checkpoints.answer_prefix_end + 5,
    )
    assert checkpoints.padded(padded_length=checkpoints.sequence_length + 5, padding_side="right") == (
        checkpoints.format_end,
        checkpoints.answer_prefix_end,
    )


def test_readout_checkpoints_fail_closed_on_tokenization_drift_or_instruction_overlap():
    prompt = "A) alpha B) beta" + _SUFFIX
    with pytest.raises(ValueError, match="token IDs"):
        resolve_readout_checkpoints(OffsetTokenizer(mismatch=True), prompt)
    with pytest.raises(ValueError, match="response instruction"):
        resolve_readout_checkpoints(OffsetTokenizer(cross_instruction=True), prompt)
    with pytest.raises(ValueError, match="terminal letter instruction"):
        resolve_readout_checkpoints(OffsetTokenizer(), "No output contract")
    with pytest.raises(ValueError, match="padding_side"):
        CheckpointIndices(1, 2, 3).padded(padded_length=4, padding_side="middle")


def test_discovery_ledger_uses_plain_train_once_and_every_validation_rotation():
    train, train_app = _scored_block(item_id="train-1", split="train")
    validation, validation_app = _scored_block(item_id="validation-1", split="validation")

    ledger = prepare_readout_ledger(
        pd.concat([train, validation], ignore_index=True),
        pd.concat([train_app, validation_app], ignore_index=True),
        stage="discovery",
    )

    assert ledger[ledger["split"] == "train"]["variant"].tolist() == [0]
    assert set(ledger[ledger["split"] == "validation"]["variant"]) == set(range(7))
    assert set(ledger[ledger["split"] == "train"]["readout_role"]) == {"probe_train"}
    assert set(ledger[ledger["split"] == "validation"]["readout_role"]) == {"layer_select"}
    assert ledger["readout_work_key"].is_unique


def test_confirmation_ledger_retains_every_baseline_and_balances_one_rotation_per_arm():
    blocks = [
        _scored_block(item_id=f"test-{index}", split="test", wrapper="key_equals")
        for index in range(7)
    ]
    baseline_only = _scored_block(
        item_id="test-baseline-only",
        split="test",
        wrapper="protobuf_msg",
        separable=False,
    )
    scored = pd.concat([block[0] for block in [*blocks, baseline_only]], ignore_index=True)
    applicability = pd.concat([block[1] for block in [*blocks, baseline_only]], ignore_index=True)

    ledger = prepare_readout_ledger(scored, applicability, stage="confirmation", seed="fixed")

    assert set(ledger["item_id"]) == {f"test-{index}" for index in range(7)} | {
        "test-baseline-only"
    }
    assert len(ledger[ledger["manipulation"] == "controlled_baseline"]) == 8
    assert len(ledger[ledger["manipulation"] == "position_only"]) == 7
    assert len(ledger[ledger["manipulation"] == "label_only"]) == 7
    for manipulation in ("position_only", "label_only"):
        counts = ledger[ledger["manipulation"] == manipulation]["variant"].value_counts()
        assert counts.max() - counts.min() <= 1


def test_ledger_recomputes_coordinates_flags_duplicate_text_and_rejects_semantic_drift():
    scored, applicability = _scored_block(
        item_id="validation-1",
        split="validation",
        duplicate_text=True,
    )
    ledger = prepare_readout_ledger(scored, applicability, stage="discovery")

    assert ledger["text_identity_ambiguous"].all()
    assert ledger["winner_label_index"].eq(1).all()
    assert ledger["winner_position"].tolist() == scored["raw_predicted_position"].tolist()
    assert ledger["winner_content_id"].tolist() == scored["raw_predicted_content_id"].tolist()

    broken = scored.copy()
    broken.loc[0, "raw_predicted_content_id"] = 3
    with pytest.raises(ValueError, match="stored raw winner coordinates"):
        prepare_readout_ledger(broken, applicability, stage="discovery")

    broken = scored.copy()
    broken.loc[0, "prompt"] += "changed"
    with pytest.raises(ValueError, match="prompt checksum"):
        prepare_readout_ledger(broken, applicability, stage="discovery")

    broken = scored.copy()
    broken.loc[0, ["raw_score_A", "raw_score_B", "raw_score_C", "raw_score_D"]] = [
        9.0,
        3.0,
        2.0,
        1.0,
    ]
    with pytest.raises(ValueError, match="raw winner does not match raw scores"):
        prepare_readout_ledger(broken, applicability, stage="discovery")


def test_ledger_rejects_missing_or_extra_interventions_instead_of_silently_dropping_rows():
    scored, applicability = _scored_block(item_id="validation-1", split="validation")
    missing = scored[scored["variant"] != 3].copy()
    with pytest.raises(ValueError, match="expected 3 position_only"):
        prepare_readout_ledger(missing, applicability, stage="discovery")

    baseline_only, baseline_app = _scored_block(
        item_id="validation-2", split="validation", separable=False
    )
    extra_row = scored[scored["variant"] == 1].copy()
    extra_row["item_id"] = "validation-2"
    extra_row["work_key"] = "letter|validation-2|plain|1"
    extra_row["source_prompt_sha256"] = baseline_only.iloc[0]["source_prompt_sha256"]
    extra = pd.concat([baseline_only, extra_row], ignore_index=True)
    with pytest.raises(ValueError, match="expected 0 position_only"):
        prepare_readout_ledger(extra, baseline_app, stage="discovery")


def test_canonical_source_validation_rejects_missing_items_and_mapping_drift():
    train, train_app = _scored_block(item_id="train-1", split="train")
    validation, validation_app = _scored_block(item_id="validation-1", split="validation")
    canonical = pd.concat([train, validation], ignore_index=True)
    scored = canonical.copy()
    applicability = pd.concat([train_app, validation_app], ignore_index=True)

    validate_canonical_sources(
        scored,
        applicability,
        canonical,
        stage="discovery",
        expected_items={"train": 1, "validation": 1},
    )

    with pytest.raises(ValueError, match="canonical work keys"):
        validate_canonical_sources(
            scored[scored["item_id"] != "validation-1"],
            applicability[applicability["item_id"] != "validation-1"],
            canonical,
            stage="discovery",
            expected_items={"train": 1, "validation": 1},
        )

    changed = scored.copy()
    index = changed.index[changed["variant"] == 1][0]
    changed.at[index, "content_ids_by_position"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="canonical semantic columns"):
        validate_canonical_sources(
            changed,
            applicability,
            canonical,
            stage="discovery",
            expected_items={"train": 1, "validation": 1},
        )


def test_batched_capture_matches_transformer_block_outputs_at_both_checkpoints(tiny_hook_model):
    tokenizer = CharacterOffsetTokenizer()
    prompts = ["A) one B) two" + _SUFFIX, "A) red B) blue C) green" + _SUFFIX]

    captured = capture_layer_readouts(
        tiny_hook_model,
        tokenizer,
        prompts,
        batch_size=2,
        max_batch_tokens=512,
    )

    assert captured.activations.shape == (2, 2, 2, 3)
    assert captured.raw_log_probs.shape == (2, 4)
    assert captured.input_preparation_seconds >= 0
    assert captured.forward_seconds >= 0
    for row, prompt in enumerate(prompts):
        ids = torch.tensor([tokenizer.encode(prompt)])
        outputs = tiny_hook_model(input_ids=ids, output_hidden_states=True)
        checkpoints = resolve_readout_checkpoints(tokenizer, prompt)
        for layer in range(2):
            expected = outputs.hidden_states[layer + 1][0]
            assert torch.allclose(
                captured.activations[row, layer, 0], expected[checkpoints.format_end].float()
            )
            assert torch.allclose(
                captured.activations[row, layer, 1], expected[checkpoints.answer_prefix_end].float()
            )


def _probe_fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_train = 160
    y = np.arange(n_train) % 4
    train = rng.normal(scale=0.05, size=(n_train, 3, 2, 6)).astype(np.float32)
    for index, target in enumerate(y):
        train[index, 1, 0, target] += 8.0
        train[index, 2, 1, target] += 8.0

    n_validation = 96
    content = np.arange(n_validation) % 4
    label = (content + 1) % 4
    position = (content + 2) % 4
    validation = rng.normal(scale=0.05, size=(n_validation, 3, 2, 6)).astype(np.float32)
    for index in range(n_validation):
        validation[index, 1, 0, content[index]] += 8.0
        validation[index, 2, 1, label[index]] += 8.0
    ledger = pd.DataFrame(
        {
            "readout_work_key": [f"validation|{index}" for index in range(n_validation)],
            "item_id": [f"item-{index // 4}" for index in range(n_validation)],
            "winner_content_id": content,
            "winner_position": position,
            "winner_label_index": label,
            "winner_unique": True,
            "prompt": [f"large prompt {index}" for index in range(n_validation)],
            "candidate_texts": [["a", "b", "c", "d"]] * n_validation,
        }
    )
    return train, y, validation, ledger


def test_linear_probes_find_distinct_content_and_label_layers_without_item_leakage(tmp_path):
    train, y, validation, ledger = _probe_fixture()

    bank = fit_probe_bank(train, y, c=1e-2, max_iter=5000)
    evaluated = evaluate_probe_bank(bank, validation, ledger)
    assert "prompt" not in evaluated.columns
    assert "candidate_texts" not in evaluated.columns
    frozen = select_readout_layers(
        evaluated,
        bootstrap_samples=200,
        permutation_samples=200,
        seed=11,
    )

    assert frozen["content"]["selected_layer"] == 1
    assert frozen["content"]["checkpoint"] == "format_end"
    assert frozen["content"]["usable"] is True
    assert frozen["label"]["selected_layer"] == 2
    assert frozen["label"]["checkpoint"] == "answer_prefix_end"
    assert frozen["label"]["usable"] is True
    assert set(frozen["content"]["patch_layers"]) == {0, 1, 2}

    path = tmp_path / "probes.npz"
    save_probe_bank(path, bank)
    restored = load_probe_bank(path)
    assert isinstance(restored, ProbeBank)
    assert np.array_equal(restored.classes, bank.classes)
    assert np.allclose(restored.weights, bank.weights)
    assert np.allclose(restored.intercepts, bank.intercepts)
    assert np.allclose(restored.means, bank.means)
    assert np.allclose(restored.log_probabilities(validation), bank.log_probabilities(validation))


def test_probe_fit_rejects_missing_classes_and_nonconvergence():
    train, y, _, _ = _probe_fixture()
    with pytest.raises(ValueError, match="all four raw winner classes"):
        fit_probe_bank(train[y != 3], y[y != 3])
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_probe_bank(train, y, max_iter=1)


def test_layer_selection_marks_decodable_but_weak_probes_unusable():
    _, _, validation, ledger = _probe_fixture()
    weak = ProbeBank(
        weights=np.zeros((3, 2, 4, 6)),
        intercepts=np.zeros((3, 2, 4)),
        means=np.zeros((3, 2, 6)),
        classes=np.arange(4),
        c=1e-2,
    )

    evaluated = evaluate_probe_bank(weak, validation, ledger)
    frozen = select_readout_layers(
        evaluated,
        bootstrap_samples=100,
        permutation_samples=100,
        seed=13,
    )

    assert frozen["content"]["usable"] is False
    assert frozen["label"]["usable"] is False


def test_layer_selection_requires_target_selectivity_not_only_decodability():
    rows = []
    for item in range(32):
        content = item % 4
        label = content if item % 2 == 0 else (content + 1) % 4
        for checkpoint in ("format_end", "answer_prefix_end"):
            rows.append(
                {
                    "item_id": f"item-{item}",
                    "layer": 0,
                    "checkpoint": checkpoint,
                    "probe_pred_class": label,
                    "winner_content_id": content,
                    "winner_position": (content + 2) % 4,
                    "winner_label_index": label,
                    "content_log_prob": -4.0,
                    "position_log_prob": -3.0,
                    "label_log_prob": 0.0,
                    "winner_unique": True,
                    "text_identity_ambiguous": False,
                    "manipulation": "label_only" if item % 2 else "position_only",
                }
            )
    frozen = select_readout_layers(
        pd.DataFrame(rows), bootstrap_samples=200, permutation_samples=200, seed=3
    )

    assert frozen["content"]["macro_accuracy"] == pytest.approx(0.5)
    assert frozen["content"]["selectivity"] < 0
    assert frozen["content"]["usable"] is False
    assert frozen["label"]["usable"] is True


def test_duplicate_answer_text_is_retained_but_never_forced_into_content_inference():
    _, _, validation, ledger = _probe_fixture()
    ledger["text_identity_ambiguous"] = False
    ledger.loc[0, "text_identity_ambiguous"] = True
    bank = ProbeBank(
        weights=np.zeros((3, 2, 4, 6)),
        intercepts=np.zeros((3, 2, 4)),
        means=np.zeros((3, 2, 6)),
        classes=np.arange(4),
        c=1e-2,
    )

    evaluated = evaluate_probe_bank(bank, validation, ledger)
    ambiguous = evaluated[evaluated["readout_work_key"] == "validation|0"]

    assert len(ambiguous) == 6
    assert not ambiguous["content_evaluable"].any()
    assert ambiguous["probe_pred_class"].isna().all()
    assert ambiguous["content_log_prob"].isna().all()
    assert ambiguous["content_correct"].isna().all()
    assert ambiguous["position_log_prob"].isna().all()


def test_probe_subspace_is_class_centered_orthonormal_and_rank_at_most_three():
    weights = np.zeros((2, 2, 4, 6), dtype=np.float64)
    weights[1, 0, :, :4] = np.eye(4)
    weights[1, 0] += 9.0  # A shared offset is not a discriminating direction.
    bank = ProbeBank(
        weights=weights,
        intercepts=np.zeros((2, 2, 4)),
        means=np.zeros((2, 2, 6)),
        classes=np.arange(4),
        c=1e-2,
    )

    basis = probe_subspace_basis(bank, layer=1, checkpoint="format_end")

    assert basis.shape == (6, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-12)
    assert np.allclose(basis.T @ np.ones(6), np.zeros(3), atol=1e-12)
    assert np.array_equal(basis, probe_subspace_basis(bank, layer=1, checkpoint=0))


def test_projected_and_random_controls_have_the_prespecified_geometry():
    receiver = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0])
    donor = torch.tensor([4.0, 1.0, -0.5, 2.0, 5.0, -3.0])
    basis = np.eye(6, dtype=np.float64)[:, :2]

    projected = projected_replacement(receiver, donor, basis)
    vectors = build_patch_vectors(receiver, donor, basis, seed="pair|layer|checkpoint")
    repeated = build_patch_vectors(receiver, donor, basis, seed="pair|layer|checkpoint")

    delta = donor - receiver
    projected_delta = projected - receiver
    random_delta = vectors.random - receiver
    assert torch.allclose(projected[:2], donor[:2])
    assert torch.allclose(projected[2:], receiver[2:])
    assert torch.allclose(vectors.probe, projected)
    assert torch.allclose(vectors.full, donor)
    assert torch.allclose(vectors.identity, receiver)
    assert torch.allclose(vectors.random, repeated.random)
    assert torch.allclose(random_delta[:2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(random_delta.norm(), projected_delta.norm(), atol=1e-6)
    assert torch.allclose(projected_replacement(receiver, receiver, basis), receiver)
    assert not torch.allclose(delta, projected_delta)


def test_patch_targets_map_donor_content_into_receiver_without_string_matching():
    donor = {
        "raw_predicted_label": "B",
        "labels_by_position": ["D", "B", "A", "C"],
        "content_ids_by_position": [2, 3, 1, 0],
        "candidate_texts": ["same", "same", "other", "OTHER"],
    }
    receiver = {
        "labels_by_position": ["C", "A", "D", "B"],
        "content_ids_by_position": [3, 0, 2, 1],
        "candidate_texts": ["different", "values", "do", "not matter"],
    }

    targets = patch_target_labels(donor, receiver)

    assert targets.donor_content_id == 3
    assert targets.receiver_content_label == "C"
    assert targets.donor_symbol_label == "B"


def test_patch_targets_and_target_margin_fail_closed_on_invalid_inputs():
    valid = {
        "raw_predicted_label": "A",
        "labels_by_position": ["A", "B", "C", "D"],
        "content_ids_by_position": [0, 1, 2, 3],
    }
    broken = {**valid, "content_ids_by_position": [0, 1, 2, 2]}
    with pytest.raises(ValueError, match="content_ids_by_position"):
        patch_target_labels(valid, broken)
    with pytest.raises(ValueError, match="scores for exactly A, B, C, D"):
        target_margin({"A": 1.0, "B": 0.0}, "A")
    with pytest.raises(ValueError, match="Unknown target label"):
        target_margin({label: float(index) for index, label in enumerate("ABCD")}, "E")

    scores = {"A": 0.0, "B": 3.0, "C": 1.0, "D": -1.0}
    assert target_margin(scores, "B") == pytest.approx(2.0)
    assert target_margin(scores, "A") == pytest.approx(-3.0)


def _set_variant_winner_content(frame: pd.DataFrame, variant: int, content_id: int) -> None:
    index = frame.index[frame["variant"] == variant].item()
    contents = list(frame.at[index, "content_ids_by_position"])
    labels = list(frame.at[index, "labels_by_position"])
    position = contents.index(content_id)
    label = labels[position]
    frame.at[index, "raw_predicted_label"] = label
    frame.at[index, "raw_predicted_position"] = position
    frame.at[index, "raw_predicted_content_id"] = content_id
    for candidate_label in "ABCD":
        frame.at[index, f"raw_score_{candidate_label}"] = (
            4.0 if candidate_label == label else 0.0
        )


def test_patch_pair_ledger_keeps_every_candidate_and_selects_conflicts_and_controls():
    scored, applicability = _scored_block(
        item_id="validation-1", split="validation", wrapper="json_object"
    )
    _set_variant_winner_content(scored, variant=1, content_id=1)  # stable position control
    _set_variant_winner_content(scored, variant=4, content_id=1)  # content-stable label remap
    readout = prepare_readout_ledger(scored, applicability, stage="discovery")

    pairs = prepare_patch_pair_ledger(
        readout,
        split="validation",
        stable_control_count=1,
        label_binding_control_count=1,
        seed="fixed",
    )

    assert len(pairs) == 6
    assert pairs["pair_work_key"].is_unique
    assert (pairs["donor_work_key"] == "letter|validation-1|json_object|0").all()
    assert set(pairs.loc[pairs["selected_for_patching"], "pair_kind"]) == {
        "answer_conflict",
        "stable_control",
        "label_binding_control",
    }
    assert len(pairs[(pairs["pair_kind"] == "answer_conflict") & pairs["selected_for_patching"]]) == 2
    stable = pairs[pairs["pair_kind"] == "stable_control"].iloc[0]
    assert stable["manipulation"] == "position_only"
    binding = pairs[pairs["pair_kind"] == "label_binding_control"].iloc[0]
    assert binding["receiver_content_target_label"] == binding["receiver_raw_predicted_label"]
    assert binding["receiver_content_target_label"] != binding["donor_symbol_target_label"]


def test_patch_pair_ledger_marks_ties_and_unselected_rows_instead_of_dropping_them():
    scored, applicability = _scored_block(item_id="validation-1", split="validation")
    tied = scored["variant"] == 2
    scored.loc[tied, "raw_tie"] = True
    scored.loc[tied, ["raw_score_A", "raw_score_B", "raw_score_C", "raw_score_D"]] = [
        0.0,
        4.0,
        4.0,
        0.0,
    ]
    _set_variant_winner_content(scored, variant=1, content_id=1)
    _set_variant_winner_content(scored, variant=3, content_id=1)
    readout = prepare_readout_ledger(scored, applicability, stage="discovery")

    pairs = prepare_patch_pair_ledger(
        readout,
        split="validation",
        stable_control_count=1,
        label_binding_control_count=0,
    )

    assert len(pairs) == 6
    tied = pairs[pairs["receiver_work_key"].str.endswith("|2")].iloc[0]
    assert tied["pair_kind"] == "not_applicable_raw_tie"
    assert not bool(tied["selected_for_patching"])
    assert (pairs["pair_kind"] == "not_selected_stable").any()

    duplicate = pd.concat([readout, readout.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate readout work keys"):
        prepare_patch_pair_ledger(duplicate, split="validation")


def test_patch_pair_ledger_retains_ambiguous_text_items_but_never_selects_them():
    scored, applicability = _scored_block(
        item_id="validation-duplicate", split="validation", duplicate_text=True
    )
    _set_variant_winner_content(scored, variant=1, content_id=1)
    ledger = prepare_readout_ledger(scored, applicability, stage="discovery")

    pairs = prepare_patch_pair_ledger(ledger, split="validation")

    assert len(pairs) == 6
    assert not pairs["selected_for_patching"].any()
    assert set(pairs["pair_kind"]) == {"not_applicable_ambiguous_content"}


def test_patch_pair_runs_all_prespecified_conditions_and_identity_matches_unpatched(
    tiny_hook_model,
):
    tokenizer = CharacterOffsetTokenizer()
    donor_prompt = "A)a B)b C)c D)d" + _SUFFIX
    receiver_prompt = "C)d A)a D)c B)b" + _SUFFIX
    pair = {
        "pair_work_key": "pair|donor|receiver",
        "item_id": "validation-1",
        "subject": "toy",
        "split": "validation",
        "wrapper_name": "plain",
        "manipulation": "label_only",
        "variant": 4,
        "pair_kind": "answer_conflict",
        "selected_for_patching": True,
        "donor_work_key": "donor",
        "receiver_work_key": "receiver",
        "donor_prompt": donor_prompt,
        "receiver_prompt": receiver_prompt,
        "donor_prompt_sha256": hashlib.sha256(donor_prompt.encode()).hexdigest(),
        "receiver_prompt_sha256": hashlib.sha256(receiver_prompt.encode()).hexdigest(),
        "donor_labels_by_position": ["A", "B", "C", "D"],
        "receiver_labels_by_position": ["C", "A", "D", "B"],
        "donor_content_ids_by_position": [0, 1, 2, 3],
        "receiver_content_ids_by_position": [3, 0, 2, 1],
        "donor_raw_predicted_label": "B",
        "receiver_raw_predicted_label": "D",
        "donor_raw_predicted_content_id": 1,
        "receiver_raw_predicted_content_id": 2,
        "correct_content_id": 1,
        "receiver_content_target_label": "B",
        "donor_symbol_target_label": "B",
        "raw_tie": False,
        **{f"receiver_bias_score_{label}": 0.0 for label in "ABCD"},
    }
    weights = np.zeros((2, 2, 4, 3), dtype=np.float64)
    weights[:, :, :, :] = np.asarray(
        [[-1.0, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    bank = ProbeBank(
        weights=weights,
        intercepts=np.zeros((2, 2, 4)),
        means=np.zeros((2, 2, 3)),
        classes=np.arange(4),
        c=1e-2,
    )
    frozen = {
        "content": {
            "usable": True,
            "checkpoint": "format_end",
            "patch_layers": [0],
        },
        "label": {
            "usable": True,
            "checkpoint": "answer_prefix_end",
            "patch_layers": [1],
        },
    }

    batch_sizes = []

    def record_batch_size(_module, _args, kwargs):
        batch_sizes.append(int(kwargs["input_ids"].shape[0]))

    handle = tiny_hook_model.register_forward_pre_hook(record_batch_size, with_kwargs=True)
    try:
        rows = run_patch_pair(tiny_hook_model, tokenizer, pair, bank, frozen)
    finally:
        handle.remove()

    assert batch_sizes == [2, *([1] * 8)]
    assert len(rows) == 10
    assert set(rows["mechanism"]) == {"content", "label"}
    assert set(rows["condition"]) == {"unpatched", "identity", "probe", "full", "random"}
    for _, group in rows.groupby("mechanism"):
        unpatched = group[group["condition"] == "unpatched"].iloc[0]
        identity = group[group["condition"] == "identity"].iloc[0]
        for label in "ABCD":
            assert identity[f"raw_score_{label}"] == pytest.approx(unpatched[f"raw_score_{label}"])
        assert identity["intervention_norm"] == pytest.approx(0.0)
        assert group[group["condition"] == "probe"]["intervention_norm"].iloc[0] == pytest.approx(
            group[group["condition"] == "random"]["intervention_norm"].iloc[0]
        )
    assert rows["content_target_margin"].notna().all()
    assert rows["symbol_target_margin"].notna().all()
