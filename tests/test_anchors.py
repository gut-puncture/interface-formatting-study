from __future__ import annotations

import pytest

from interface_formatting_study.anchors import answer_anchor, content_end_anchor, resolve_anchor, resolve_anchor_positions
from interface_formatting_study.scoring import tokenize_text


def test_answer_anchor_is_last_prompt_token(boundary_tokenizer):
    prompt = "Question?\nAnswer: "
    assert answer_anchor(boundary_tokenizer, prompt) == len(tokenize_text(boundary_tokenizer, prompt)) - 1


def test_content_end_anchor_points_before_shared_suffix(boundary_tokenizer):
    prompt = "CONTENT ABC\n\nReturn only the letter (A, B, C, or D).\nAnswer: "
    anchor = content_end_anchor(boundary_tokenizer, prompt)
    prefix = "CONTENT ABC"
    assert anchor == len(tokenize_text(boundary_tokenizer, prefix)) - 1


def test_content_end_missing_returns_none(boundary_tokenizer):
    assert content_end_anchor(boundary_tokenizer, "No suffix here") is None
    assert resolve_anchor(boundary_tokenizer, "No suffix here", "content_end") is None


def test_content_end_uses_final_shared_suffix(boundary_tokenizer):
    prompt = (
        "Wrapper quotes: Return only the letter in the source text.\n"
        "Actual content ends here.\n\n"
        "Return only the letter (A, B, C, or D).\nAnswer: "
    )
    anchor = content_end_anchor(boundary_tokenizer, prompt)
    prefix = "Wrapper quotes: Return only the letter in the source text.\nActual content ends here."
    assert anchor == len(tokenize_text(boundary_tokenizer, prefix)) - 1


def _verified_wrapper_prompt(wrapper: str, question: str, choices: list[str]) -> str:
    a, b, c, d = choices
    prompts = {
        "csv_inline": (
            f"Question,Option,Value\n{question},A,{a}\n{question},B,{b}\n{question},C,{c}\n{question},D,{d}\n"
            "Select the correct option letter.\n\nReturn only the letter (A, B, C, or D).\nAnswer: "
        ),
        "graphql_query": (
            f"query {{\n  question: \"{question}\"\n  answer(options: [\n"
            f"    {{ letter: \"A\", value: \"{a}\" }},\n    {{ letter: \"B\", value: \"{b}\" }},\n"
            f"    {{ letter: \"C\", value: \"{c}\" }},\n    {{ letter: \"D\", value: \"{d}\" }}\n  ])\n}}\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
        "html_form": (
            f"<form>\n<p>{question}</p>\n<label for=\"answer\">Select your answer:</label>\n<select id=\"answer\" name=\"answer\">\n"
            f"<option value=\"A\">A) {a}</option>\n<option value=\"B\">B) {b}</option>\n"
            f"<option value=\"C\">C) {c}</option>\n<option value=\"D\">D) {d}</option>\n</select>\n</form>\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
        "ini_file": (
            f"[question]\ntext={question}\n\n[options]\nA={a}\nB={b}\nC={c}\nD={d}\n\n"
            "[instruction]\nSelect the correct option by responding with a single letter (A, B, C, or D).\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
        "key_equals": (
            f"QUESTION={question} OPTION_A={a} OPTION_B={b} OPTION_C={c} OPTION_D={d}\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
        "protobuf_msg": (
            f"message MCQ {{\n  required string question = 1 [default = \"{question}\"];\n  repeated string options = 2 [\n"
            f"    (option) = \"A) {a}\",\n    (option) = \"B) {b}\",\n    (option) = \"C) {c}\",\n"
            f"    (option) = \"D) {d}\"\n  ];\n}}\n\nReturn only the letter (A, B, C, or D).\nAnswer: "
        ),
        "shell_heredoc": (
            f"#!/bin/bash\ncat << 'EOF'\n{question}\n\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n\nEOF\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
        "toml_config": (
            f"[question]\ntext = \"{question}\"\n\n[options]\nA = \"{a}\"\nB = \"{b}\"\nC = \"{c}\"\nD = \"{d}\"\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
    }
    return prompts[wrapper]


@pytest.mark.parametrize(
    "wrapper",
    [
        "csv_inline",
        "graphql_query",
        "html_form",
        "ini_file",
        "key_equals",
        "protobuf_msg",
        "shell_heredoc",
        "toml_config",
    ],
)
def test_semantic_anchors_resolve_for_verified_wrappers(boundary_tokenizer, wrapper):
    question = "Add 17 and 25."
    choices = ["39", "40", "42", "52"]
    prompt = _verified_wrapper_prompt(wrapper, question, choices)

    assert resolve_anchor(boundary_tokenizer, prompt, "question_end", question=question, choices=choices) is not None
    assert resolve_anchor(boundary_tokenizer, prompt, "options_end", question=question, choices=choices) is not None
    for label in "ABCD":
        anchor = resolve_anchor(boundary_tokenizer, prompt, f"option_{label}_end", question=question, choices=choices)
        assert anchor is not None


def test_grouped_semantic_anchor_resolves_multiple_positions(boundary_tokenizer):
    question = "Add 17 and 25."
    choices = ["39", "40", "42", "52"]
    prompt = _verified_wrapper_prompt("key_equals", question, choices)

    grouped = resolve_anchor_positions(
        boundary_tokenizer,
        prompt,
        "question_end+options_end",
        question=question,
        choices=choices,
    )

    assert grouped is not None
    assert [position.name for position in grouped] == ["question_end", "options_end"]
    assert grouped[0].position != grouped[1].position


def test_semantic_anchor_missing_metadata_returns_none(boundary_tokenizer):
    prompt = _verified_wrapper_prompt("key_equals", "Add 17 and 25.", ["39", "40", "42", "52"])

    assert resolve_anchor(boundary_tokenizer, prompt, "question_end") is None
    assert resolve_anchor(boundary_tokenizer, prompt, "option_A_end", question="Add 17 and 25.") is None


def test_grouped_anchor_rejects_duplicate_positions(boundary_tokenizer):
    prompt = _verified_wrapper_prompt(
        "csv_inline",
        "What is the remainder of 21 divided by 7?",
        ["21", "7", "1", "None of these"],
    )

    assert (
        resolve_anchor_positions(
            boundary_tokenizer,
            prompt,
            "all_option_ends",
            question="What is the remainder of 21 divided by 7?",
            choices=["21", "7", "1", "None of these"],
        )
        is None
    )
