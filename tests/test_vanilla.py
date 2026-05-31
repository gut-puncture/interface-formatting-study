from __future__ import annotations

import pytest

from interface_formatting_study.vanilla import build_vanilla_prompt, build_vanilla_prompt_from_row, choices_by_label


def test_build_vanilla_prompt_preserves_question_choices_and_instruction():
    prompt = build_vanilla_prompt(
        "Which value is prime?",
        {"A": "4", "B": "6", "C": "7", "D": "8"},
    )

    assert prompt.startswith("Which value is prime?\n\n")
    assert "A) 4" in prompt
    assert "B) 6" in prompt
    assert "C) 7" in prompt
    assert "D) 8" in prompt
    assert "Return only the letter (A, B, C, or D)." in prompt
    assert prompt.endswith("Answer:")


def test_build_vanilla_prompt_from_row_accepts_json_choices():
    prompt = build_vanilla_prompt_from_row(
        {
            "question": "Pick the color.",
            "choices": '{"A": "red", "B": "blue", "C": "green", "D": "gold"}',
        }
    )

    assert "Pick the color." in prompt
    assert "A) red" in prompt
    assert "D) gold" in prompt


def test_choices_by_label_rejects_missing_choice():
    with pytest.raises(ValueError, match="missing choices"):
        choices_by_label({"A": "x", "B": "y", "C": "z", "D": ""})
