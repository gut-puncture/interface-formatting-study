from __future__ import annotations

import pandas as pd

from interface_formatting_study.wrapper_audit import audit_wrappers, classify_wrapper_name, contains_all_choices


def test_actual_wrapper_name_taxonomy_fails_closed_for_style_wrappers():
    pure = ["csv_inline", "html_form", "ini_file", "key_equals", "toml_config", "graphql_query", "protobuf_msg"]
    style = ["haiku_riddle", "quest_briefing", "legal_clause", "meeting_minutes", "academic_abstract", "tweet_thread"]
    for name in pure:
        assert classify_wrapper_name(name) == "pure_interface"
    for name in style:
        assert classify_wrapper_name(name) == "style_transforming"


def test_allowlisted_wrappers_audit_as_pure_when_present():
    from interface_formatting_study.wrapper_audit import PURE_INTERFACE_ALLOWLIST

    rows = []
    for name in PURE_INTERFACE_ALLOWLIST:
        rows.append(
            {
                "wrapper_name": name,
                "wrapped_prompt": "Question text. A) alpha B) beta C) gamma D) delta",
                "question": "Question text.",
                "choices": ["alpha", "beta", "gamma", "delta"],
            }
        )
    audit = audit_wrappers(pd.DataFrame(rows))
    assert set(audit["category"]) == {"pure_interface"}
    assert audit["verified_pure_interface"].all()


def test_allowlisted_wrapper_without_canonical_evidence_is_not_primary_pure():
    audit = audit_wrappers(
        pd.DataFrame(
            [
                {
                    "wrapper_name": "regex_match",
                    "wrapped_prompt": "Use a pattern to choose one option.",
                    "question": "What is the answer?",
                    "choices": ["alpha", "beta", "gamma", "delta"],
                }
            ]
        )
    )

    row = audit.iloc[0]
    assert row["category"] == "pure_interface_unverified"
    assert not bool(row["verified_pure_interface"])


def test_audit_uses_canonical_evidence_and_name_heuristics():
    df = pd.DataFrame(
        [
            {
                "wrapper_name": "csv_inline",
                "wrapped_prompt": "Q?, A) alpha, B) beta, C) gamma, D) delta\nReturn only the letter",
                "question": "Q?",
                "choices": ["alpha", "beta", "gamma", "delta"],
            },
            {
                "wrapper_name": "haiku_riddle",
                "wrapped_prompt": "moonlit puzzle whispers; choose A alpha B beta C gamma D delta",
                "question": "Q?",
                "choices": ["alpha", "beta", "gamma", "delta"],
            },
        ]
    )
    audit = audit_wrappers(df)
    categories = audit.set_index("wrapper_name")["category"].to_dict()
    assert categories["csv_inline"] == "pure_interface"
    assert categories["haiku_riddle"] == "style_transforming"


def test_contains_all_choices_requires_all_option_texts():
    assert contains_all_choices("A alpha B beta C gamma D delta", ["alpha", "beta", "gamma", "delta"])
    assert not contains_all_choices("A alpha B beta C gamma", ["alpha", "beta", "gamma", "delta"])


def test_question_verbatim_check_does_not_accept_loose_character_overlap():
    from interface_formatting_study.wrapper_audit import contains_nearly_verbatim

    paraphrase = "Three roots join the field; over rational ground, tell what degree is held?"
    question = "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
    assert not contains_nearly_verbatim(paraphrase, question)
