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
    capture_layer_readouts,
    evaluate_probe_bank,
    fit_probe_bank,
    load_probe_bank,
    prepare_readout_ledger,
    resolve_readout_checkpoints,
    save_probe_bank,
    select_readout_layers,
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
                "source_prompt_sha256": "",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "content_ids_by_position": list(assignment.content_ids_by_position),
                "labels_by_position": list(assignment.labels_by_position),
                "candidate_texts": candidates,
                "correct_content_id": 0,
                "raw_predicted_label": predicted_label,
                "raw_predicted_position": predicted_position,
                "raw_predicted_content_id": predicted_content,
                "raw_tie": False,
                **{f"raw_score_{label}": float(4 - index) for index, label in enumerate("ABCD")},
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
        }
    )
    return train, y, validation, ledger


def test_linear_probes_find_distinct_content_and_label_layers_without_item_leakage(tmp_path):
    train, y, validation, ledger = _probe_fixture()

    bank = fit_probe_bank(train, y, c=1e-2, max_iter=5000)
    evaluated = evaluate_probe_bank(bank, validation, ledger)
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
