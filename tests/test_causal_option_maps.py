from __future__ import annotations

import hashlib

from interface_formatting_study.causal_option_maps import (
    parse_prompt_options,
    transform_with_option_map,
)


SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "


def test_csv_parser_handles_labels_in_the_last_column_and_preserves_source_bytes():
    prompt = (
        "Question,Option,Option_Letter\n"
        "Which value is prime?,four,A\n"
        "Which value is prime?,six,B\n"
        "Which value is prime?,seven,C\n"
        "Which value is prime?,eight,D"
        + SUFFIX
    )

    parsed = parse_prompt_options(prompt, "csv_inline", ["four", "six", "seven", "eight"])

    assert parsed.source_sha256 == hashlib.sha256(prompt.encode()).hexdigest()
    assert parsed.separable
    assert [slot.content_id for slot in parsed.representations[0].slots] == [0, 1, 2, 3]
    assert [slot.payload_texts(prompt) for slot in parsed.representations[0].slots] == [
        ("four",),
        ("six",),
        ("seven",),
        ("eight",),
    ]


def test_csv_parser_handles_unquoted_commas_in_option_payloads():
    prompt = (
        "Option,Description\n"
        "A,one.\n"
        "B,two, with detail.\n"
        "C,three, with more, detail.\n"
        "D,four."
        + SUFFIX
    )

    parsed = parse_prompt_options(
        prompt,
        "csv_inline",
        ["one.", "two, with detail.", "three, with more, detail.", "four."],
    )

    assert [slot.payload_texts(prompt) for slot in parsed.representations[0].slots] == [
        ("one.",),
        ("two, with detail.",),
        ("three, with more, detail.",),
        ("four.",),
    ]


def test_protobuf_parser_handles_label_prefix_inside_repeated_option_strings():
    prompt = (
        "message MCQ {\n"
        '  repeated string options = 2 [(option) = "A) one.",\n'
        '    (option) = "B) two, detailed.",\n'
        '    (option) = "C) three.",\n'
        '    (option) = "D) four."];\n'
        "}"
        + SUFFIX
    )

    parsed = parse_prompt_options(
        prompt,
        "protobuf_msg",
        ["one.", "two, detailed.", "three.", "four."],
    )
    transformed = transform_with_option_map(parsed, prompt, position_shift=1, label_shift=0)

    assert '(option) = "D) four."' in transformed
    assert '(option) = "A) one."' in transformed


def test_ini_parser_attaches_description_on_the_line_after_each_option_section():
    prompt = (
        "[option_A]\ndescription = one\n\n"
        "[option_B]\ndescription = two\n\n"
        "[option_C]\ndescription = three\n\n"
        "[option_D]\ndescription = four"
        + SUFFIX
    )

    parsed = parse_prompt_options(prompt, "ini_file", ["one", "two", "three", "four"])
    transformed = transform_with_option_map(parsed, prompt, position_shift=1, label_shift=0)

    assert [slot.payload_texts(prompt) for slot in parsed.representations[0].slots] == [
        ("one",),
        ("two",),
        ("three",),
        ("four",),
    ]
    assert "[option_D]\ndescription = four" in transformed
    assert "[option_A]\ndescription = one" in transformed


def test_toml_parser_moves_the_complete_escaped_string_value():
    prompt = (
        '[options]\nA = "\\"one\\""\nB = "\\"two\\""\n'
        'C = "\\"three\\""\nD = "\\"four\\""'
        + SUFFIX
    )

    parsed = parse_prompt_options(prompt, "toml_config", ['"one"', '"two"', '"three"', '"four"'])
    transformed = transform_with_option_map(parsed, prompt, position_shift=1, label_shift=0)

    assert transformed == (
        '[options]\nD = "\\"four\\""\nA = "\\"one\\""\n'
        'B = "\\"two\\""\nC = "\\"three\\""'
        + SUFFIX
    )


def test_parser_supports_repeated_protobuf_representations_without_editing_distractors():
    prompt = (
        "message MCQ {\n"
        "  required string optionA = 1;\n"
        "  required string optionB = 2;\n"
        "  required string optionC = 3;\n"
        "  required string optionD = 4;\n"
        "}\n"
        "// optionA: alpha\n// optionB: beta\n// optionC: gamma\n// optionD: delta\n"
        "// Instruction example: A, B, C, or D"
        + SUFFIX
    )

    parsed = parse_prompt_options(prompt, "protobuf_msg", ["alpha", "beta", "gamma", "delta"])
    transformed = transform_with_option_map(parsed, prompt, position_shift=0, label_shift=1)

    assert parsed.separable
    assert len(parsed.representations) == 2
    assert "required string optionB = 1" in transformed
    assert "// optionB: alpha" in transformed
    assert "Instruction example: A, B, C, or D" in transformed


def test_position_rotation_moves_exact_payload_and_attached_label_only():
    prompt = "OPTION_A=alpha\nOPTION_B=beta\nOPTION_C=gamma\nOPTION_D=delta" + SUFFIX

    parsed = parse_prompt_options(prompt, "shell_heredoc", ["alpha", "beta", "gamma", "delta"])
    transformed = transform_with_option_map(parsed, prompt, position_shift=1, label_shift=0)

    assert transformed == (
        "OPTION_D=delta\nOPTION_A=alpha\nOPTION_B=beta\nOPTION_C=gamma" + SUFFIX
    )


def test_unlabelled_graphql_array_is_retained_but_marked_nonseparable():
    prompt = (
        'query { answer(options: ["alpha", "beta", "gamma", "delta"]) }' + SUFFIX
    )

    parsed = parse_prompt_options(prompt, "graphql_query", ["alpha", "beta", "gamma", "delta"])

    assert not parsed.separable
    assert parsed.not_applicable_reason == "independent_label_position_not_identifiable"
    assert [slot.payload_texts(prompt) for slot in parsed.representations[0].slots] == [
        ("alpha",),
        ("beta",),
        ("gamma",),
        ("delta",),
    ]


def test_html_table_with_repeated_labels_changes_every_option_label_consistently():
    prompt = (
        "<table>"
        '<tr><td><input value="A">No effect</td><td><input value="A">No effect</td></tr>'
        '<tr><td><input value="B">No effect</td><td><input value="B">Understated</td></tr>'
        '<tr><td><input value="C">Understated</td><td><input value="C">No effect</td></tr>'
        '<tr><td><input value="D">Understated</td><td><input value="D">Understated</td></tr>'
        "</table>"
        + SUFFIX
    )
    choices = [
        "No effect, No effect",
        "No effect, Understated",
        "Understated, No effect",
        "Understated, Understated",
    ]

    parsed = parse_prompt_options(prompt, "html_form", choices)
    transformed = transform_with_option_map(parsed, prompt, position_shift=0, label_shift=1)

    assert parsed.separable
    assert transformed.count('value="B"') == 2
    assert transformed.count('value="C"') == 2
    assert transformed.count('value="D"') == 2
    assert transformed.count('value="A"') == 2


def test_html_option_parser_moves_nested_markup_as_one_payload():
    prompt = (
        '<select name="answer">'
        '<option value="A">A) Alpha H<sub>2</sub>.</option>'
        '<option value="B">B) Beta.</option>'
        '<option value="C">C) Gamma.</option>'
        '<option value="D">D) Delta H<sub>3</sub>.</option>'
        "</select>"
        + SUFFIX
    )

    parsed = parse_prompt_options(
        prompt,
        "html_form",
        ["Alpha H2.", "Beta.", "Gamma.", "Delta H3."],
    )
    transformed = transform_with_option_map(parsed, prompt, position_shift=1, label_shift=0)

    assert [slot.payload_texts(prompt) for slot in parsed.representations[0].slots] == [
        ("Alpha H<sub>2</sub>.",),
        ("Beta.",),
        ("Gamma.",),
        ("Delta H<sub>3</sub>.",),
    ]
    assert '<option value="D">D) Delta H<sub>3</sub>.</option>' in transformed


def test_wrong_source_bytes_are_rejected_before_transformation():
    prompt = "A=alpha\nB=beta\nC=gamma\nD=delta" + SUFFIX
    parsed = parse_prompt_options(prompt, "key_equals", ["alpha", "beta", "gamma", "delta"])

    try:
        transform_with_option_map(parsed, prompt + "changed", position_shift=0, label_shift=1)
    except ValueError as exc:
        assert "source prompt checksum mismatch" in str(exc)
    else:
        raise AssertionError("changed source prompt was accepted")
