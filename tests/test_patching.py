from __future__ import annotations

import pandas as pd
import pytest
import torch

from interface_formatting_study.patching import (
    hidden_states_at_layers_and_positions,
    recovery,
    score_layers_with_position_replacements,
    score_prompt_with_position_replacements,
    select_best_location,
)


def test_recovery_formula_and_invalid_denominator():
    assert recovery(0.0, -2.0, 2.0) == 0.5
    assert recovery(0.0, 1.0, 1.0 + 1e-7) is None


def test_select_best_location_prefers_small_layer_and_content_end_within_tolerance():
    df = pd.DataFrame(
        [
            {"item_id": "i1", "layer": 5, "anchor": "answer_anchor", "margin_improvement": 1.0, "recovery": 0.4},
            {"item_id": "i2", "layer": 3, "anchor": "content_end", "margin_improvement": 0.96, "recovery": 0.3},
            {"item_id": "i3", "layer": 8, "anchor": "answer_anchor", "margin_improvement": 0.1, "recovery": 0.1},
        ]
    )
    selected = select_best_location(df)
    assert selected["selected_layer"] == 3
    assert selected["selected_anchor"] == "content_end"


def test_select_best_location_filters_answer_anchor_by_default():
    df = pd.DataFrame(
        [
            {"item_id": "i1", "layer": 22, "anchor": "answer_anchor", "margin_improvement": 9.0, "recovery": 1.0, "patched_correct": True, "n_positions": 1},
            {"item_id": "i2", "layer": 8, "anchor": "question_end", "margin_improvement": 1.0, "recovery": 0.2, "patched_correct": True, "n_positions": 1},
        ]
    )

    selected = select_best_location(df)

    assert selected["selected_layer"] == 8
    assert selected["selected_anchor"] == "question_end"


def test_select_best_location_can_allow_answer_anchor_for_diagnostics():
    df = pd.DataFrame(
        [
            {"item_id": "i1", "layer": 22, "anchor": "answer_anchor", "margin_improvement": 9.0, "recovery": 1.0, "patched_correct": True, "n_positions": 1},
        ]
    )

    with pytest.raises(ValueError, match="non-leaky"):
        select_best_location(df)

    selected = select_best_location(df, allow_leaky_anchors=True)
    assert selected["selected_anchor"] == "answer_anchor"


def test_hidden_states_capture_multiple_positions(tiny_hook_model, boundary_tokenizer):
    prompt = "Question A B C"
    states = hidden_states_at_layers_and_positions(
        tiny_hook_model,
        boundary_tokenizer,
        prompt,
        layers=[0, 1],
        positions=[0, 2],
    )

    assert set(states) == {0, 1}
    assert set(states[0]) == {0, 2}
    assert states[0][0].shape == states[0][2].shape


def test_batched_layer_patching_matches_scalar_patching(tiny_hook_model, boundary_tokenizer):
    prompt = "Question A B C"
    bias_scores = {"A": 0.1, "B": -0.1, "C": 0.05, "D": 0.0}
    replacements_by_layer = {
        0: {0: torch.tensor([1.5, -0.5, 0.25]), 2: torch.tensor([0.0, 1.0, -1.0])},
        1: {0: torch.tensor([-0.25, 0.75, 1.25]), 2: torch.tensor([1.0, -1.0, 0.5])},
    }

    batched = score_layers_with_position_replacements(
        tiny_hook_model,
        boundary_tokenizer,
        prompt,
        correct_label="A",
        bias_scores=bias_scores,
        replacements_by_layer=replacements_by_layer,
    )

    for layer, replacements in replacements_by_layer.items():
        scalar = score_prompt_with_position_replacements(
            tiny_hook_model,
            boundary_tokenizer,
            prompt,
            correct_label="A",
            bias_scores=bias_scores,
            layer=layer,
            replacements=replacements,
        )
        for key, value in scalar.items():
            if isinstance(value, float):
                assert batched[layer][key] == pytest.approx(value)
            else:
                assert batched[layer][key] == value
