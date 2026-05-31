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
    tokenize_text,
)


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


def test_score_labels_many_matches_single_prompt_scoring(boundary_tokenizer, rule_model):
    prompts = ["Q", "X", "Question text"]
    batched = score_labels_many(rule_model, boundary_tokenizer, prompts, batch_size=8, max_batch_tokens=128)
    singles = [score_labels(rule_model, boundary_tokenizer, prompt, batch_size=8) for prompt in prompts]
    assert batched == singles


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
