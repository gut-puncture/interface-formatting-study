from __future__ import annotations

import math

import pytest
import torch

from interface_formatting_study.scoring import (
    VARIANT_TEMPLATES,
    completion_logps,
    correct_answer_margin,
    label_variants,
    predict_from_scores,
    score_labels,
    score_labels_many,
    single_token_label_ids,
    tokenize_text,
)


class StaticLabelModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.forward_batch_sizes = []

    def forward(self, input_ids=None, attention_mask=None, **_kwargs):
        self.forward_batch_sizes.append(int(input_ids.shape[0]))
        vocab_size = max(256, len(self.tokenizer.vocab) + 64)
        logits = torch.full((*input_ids.shape, vocab_size), -10.0, device=input_ids.device)
        for rank, label in enumerate(("A", "B", "C", "D")):
            for token_id in self.tokenizer.encode(label, add_special_tokens=False):
                logits[..., token_id] = float(4 - rank)
        return type("Output", (), {"logits": logits})()


def test_label_variants_are_symmetric_for_all_labels():
    variant_counts = {label: len(label_variants(label)) for label in ["A", "B", "C", "D"]}
    assert set(variant_counts.values()) == {len(VARIANT_TEMPLATES)}
    assert label_variants("A") == ["A"]
    assert label_variants("D") == ["D"]


def test_completion_scoring_uses_manual_token_concatenation(boundary_tokenizer, rule_model):
    prompt = "X"
    completion = "A"
    separate = tokenize_text(boundary_tokenizer, prompt) + tokenize_text(boundary_tokenizer, completion)
    merged = tokenize_text(boundary_tokenizer, prompt + completion)
    assert separate != merged
    [score] = completion_logps(rule_model, boundary_tokenizer, prompt, [completion])
    assert score > -1e-4


def test_completion_logps_sums_only_completion_tokens(boundary_tokenizer, rule_model):
    prompt = "Q"
    completion = " AB"
    scores = completion_logps(rule_model, boundary_tokenizer, prompt, [completion])
    assert len(scores) == 1
    assert scores[0] > -1e-4


def test_score_labels_returns_every_label(boundary_tokenizer, rule_model):
    scores = score_labels(rule_model, boundary_tokenizer, "Q")
    assert set(scores) == {"A", "B", "C", "D"}
    assert all(math.isfinite(value) for value in scores.values())


def test_score_labels_many_matches_single_prompt_scoring(boundary_tokenizer, causal_label_model):
    prompts = ["Q", "X", "Question text"]
    batched = score_labels_many(causal_label_model, boundary_tokenizer, prompts, batch_size=8, max_batch_tokens=128)
    singles = [score_labels(causal_label_model, boundary_tokenizer, prompt, batch_size=8) for prompt in prompts]
    assert batched == singles


def test_one_token_fast_path_matches_reference_and_uses_one_sequence_per_prompt(boundary_tokenizer):
    model = StaticLabelModel(boundary_tokenizer)
    prompts = ["Q", "longer question", "X"]

    fast = score_labels_many(model, boundary_tokenizer, prompts, batch_size=16, max_batch_tokens=256)
    reference = [score_labels(model, boundary_tokenizer, prompt) for prompt in prompts]

    assert fast == reference
    assert single_token_label_ids(boundary_tokenizer) is not None
    assert model.forward_batch_sizes[0] == len(prompts)


def test_multi_token_label_automatically_uses_generic_scorer(boundary_tokenizer):
    original_encode = boundary_tokenizer.encode

    def encode(text, add_special_tokens=False):
        if text == "D":
            return original_encode("DD", add_special_tokens=add_special_tokens)
        return original_encode(text, add_special_tokens=add_special_tokens)

    boundary_tokenizer.encode = encode
    model = StaticLabelModel(boundary_tokenizer)
    assert single_token_label_ids(boundary_tokenizer) is None
    scores = score_labels_many(model, boundary_tokenizer, ["Q", "X"], batch_size=16)
    assert len(scores) == 2
    assert model.forward_batch_sizes[0] == 8


def test_margin_math_and_tie_handling():
    good = correct_answer_margin({"A": -4.0, "B": -2.0, "C": -5.0, "D": -3.0}, "B")
    assert good.margin == pytest.approx(1.0)
    assert good.correct
    bad = correct_answer_margin({"A": -1.0, "B": -2.0, "C": -5.0, "D": -3.0}, "B")
    assert bad.margin == pytest.approx(-1.0)
    assert not bad.correct
    tie = correct_answer_margin({"A": 0.0, "B": 1e-7, "C": -1.0, "D": -2.0}, "A")
    assert tie.is_tie
    assert not tie.correct
    pred, is_tie = predict_from_scores({"A": 0.0, "B": 5e-7, "C": -1, "D": -2})
    assert pred == "B"
    assert is_tie
