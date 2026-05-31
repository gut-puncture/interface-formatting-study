from __future__ import annotations

import pytest

from interface_formatting_study.generation_parser import parse_generation


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("A", "A"),
        (" A", "A"),
        ("A.", "A"),
        ("A)", "A"),
        ("(A)", "A"),
        ("Option A", "A"),
        ("The answer is A.", "A"),
        ("The correct option is B", "B"),
        ("I think the answer is (C).", "C"),
        ("answer: D", "D"),
    ],
)
def test_parser_valid_label_patterns(text, label):
    parsed = parse_generation(text)
    assert parsed.valid
    assert parsed.label == label


@pytest.mark.parametrize("text", ["A or B", "", "No idea"])
def test_parser_invalid_outputs(text):
    parsed = parse_generation(text)
    assert not parsed.valid


def test_parser_exact_answer_text():
    parsed = parse_generation("I pick Paris.", choices={"A": "London", "B": "Paris", "C": "Rome", "D": "Berlin"})
    assert parsed.valid
    assert parsed.label == "B"

