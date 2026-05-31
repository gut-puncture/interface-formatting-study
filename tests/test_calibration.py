from __future__ import annotations

from interface_formatting_study.calibration import calibrate_scores, content_free_contains_answer_text, make_content_free_prompt


def test_calibration_subtracts_label_specific_bias():
    raw = {"A": 10.0, "B": 9.0, "C": 8.0, "D": 7.0}
    bias = {"A": 9.0, "B": 1.0, "C": 7.0, "D": 0.0}
    assert calibrate_scores(raw, bias) == {"A": 1.0, "B": 8.0, "C": 1.0, "D": 7.0}


def test_content_free_prompt_removes_question_and_choice_text_but_preserves_instruction():
    row = {
        "wrapper_name": "ini_file",
        "question": "What is the capital of France?",
        "choices": ["Paris", "London", "Rome", "Berlin"],
        "wrapped_prompt": (
            "[question]\ntext=What is the capital of France?\n"
            "A=Paris\nB=London\nC=Rome\nD=Berlin\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
    }
    prompt = make_content_free_prompt(row)
    assert "QUESTION_TEXT_PLACEHOLDER" in prompt
    assert "OPTION_A_PLACEHOLDER" in prompt
    assert "Return only the letter" in prompt
    assert "Paris" not in prompt
    assert "capital of France" not in prompt
    assert not content_free_contains_answer_text(prompt, row["choices"])


def test_content_free_fallback_keeps_labels_and_final_instruction():
    prompt = make_content_free_prompt({"wrapper_name": "unknown", "wrapped_prompt": "opaque", "choices": None, "question": None})
    assert prompt.startswith("[CONTENT_FREE_WRAPPER]")
    assert "A = OPTION_A_PLACEHOLDER" in prompt
    assert "D = OPTION_D_PLACEHOLDER" in prompt
    assert "Return only the letter" in prompt


def test_content_free_falls_back_when_paraphrased_question_would_leak():
    row = {
        "wrapper_name": "academic_abstract",
        "question": "Find the degree for Q(sqrt(2)) over Q.",
        "choices": ["0", "2", "4", "6"],
        "wrapped_prompt": (
            "The study determines the field extension Q(√2) over Q. "
            "The correct value is one of A) 0, B) 2, C) 4, D) 6.\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
    }
    prompt = make_content_free_prompt(row)
    assert prompt.startswith("[CONTENT_FREE_WRAPPER]")
    assert "√2" not in prompt
    assert "sqrt(2)" not in prompt
    assert "QUESTION_TEXT_PLACEHOLDER" in prompt


def test_content_free_redacts_case_insensitive_question_residue():
    row = {
        "wrapper_name": "regex_match",
        "question": "Chlorine gas reacts most readily with",
        "choices": ["Sodium", "Helium", "Neon", "Argon"],
        "wrapped_prompt": (
            "Match the chemical that chlorine gas reacts most readily with to the correct option: "
            "/Chlorine gas reacts most readily with (Sodium|Helium|Neon|Argon)/\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
    }

    prompt = make_content_free_prompt(row)

    assert "chlorine gas reacts most readily with" not in prompt.lower()
    assert "QUESTION_TEXT_PLACEHOLDER" in prompt


def test_content_free_detector_ignores_required_labels_and_short_symbols():
    prompt = "[CONTENT_FREE_WRAPPER]\nA = OPTION_A_PLACEHOLDER\nAnswer: "
    assert not content_free_contains_answer_text(prompt, ["n", "A", "n + 1", "2n"])
    assert content_free_contains_answer_text("choice = representational", ["representational", "other", "third", "fourth"])
    assert content_free_contains_answer_text("leaked value is 4", ["4", "0", "2", "6"])
    assert content_free_contains_answer_text("leaked symbolic answer 2n", ["n", "n + 1", "n + 2", "2n"])
    assert content_free_contains_answer_text("the option value is A", ["A", "B", "C", "D"])
    assert content_free_contains_answer_text("enum City { NEW_YORK = 1; }", ["New York", "Los Angeles", "Chicago", "Houston"])
    assert content_free_contains_answer_text("const correctCity = newYork;", ["New York", "Los Angeles", "Chicago", "Houston"])


def test_content_free_falls_back_when_enum_names_leak_choices():
    row = {
        "wrapper_name": "protobuf_msg",
        "question": "Which attachment style is shown?",
        "choices": ["Secure", "Insecure avoidant", "Insecure resistant", "Insecure disorganized"],
        "wrapped_prompt": (
            "message AttachmentQuestion { // Which attachment style is shown? "
            "enum AttachmentType { SECURE = 0; INSECURE_AVOIDANT = 1; "
            "INSECURE_RESISTANT = 2; INSECURE_DISORGANIZED = 3; } }\n\n"
            "Return only the letter (A, B, C, or D).\nAnswer: "
        ),
    }
    prompt = make_content_free_prompt(row)
    assert prompt.startswith("[CONTENT_FREE_WRAPPER]")
    assert not content_free_contains_answer_text(prompt, row["choices"])
