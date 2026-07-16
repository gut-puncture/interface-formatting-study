from __future__ import annotations

import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from interface_formatting_study.causal_design import (
    ACTIVE_WRAPPERS,
    CONTROLLED_FORMATS,
    balanced_assignments,
    build_causal_design,
    build_causal_design_v3,
    build_causal_design_with_exclusions,
)
from interface_formatting_study.vanilla import build_vanilla_prompt
from interface_formatting_study.causal_option_maps import parse_prompt_options


QUESTION = "What is 2+2?"
CHOICES = ["3", "4", "5", "6"]
SUFFIX = "\n\nReturn only the letter (A, B, C, or D).\nAnswer: "


def _source_prompts() -> dict[str, str]:
    return {
        "csv_inline": "Option,Statement\nA,3\nB,4\nC,5\nD,6\n\nWhat is 2+2?" + SUFFIX,
        "graphql_query": (
            'query { question(description: "What is 2+2?") { answer(options: '
            '{ A: "3", B: "4", C: "5", D: "6" }) } }' + SUFFIX
        ),
        "html_form": (
            '<form>\n<p>What is 2+2?</p>\n<select name="answer">\n'
            '<option value="A">A) 3</option>\n<option value="B">B) 4</option>\n'
            '<option value="C">C) 5</option>\n<option value="D">D) 6</option>\n'
            "</select>\n</form>" + SUFFIX
        ),
        "ini_file": "[Question]\nstatement=What is 2+2?\n\n[Options]\nA=3\nB=4\nC=5\nD=6" + SUFFIX,
        "key_equals": "QUESTION=What is 2+2?\nA=3\nB=4\nC=5\nD=6" + SUFFIX,
        "protobuf_msg": (
            'message MCQ {\nOPTION_A = 0 [label = "3"];\nOPTION_B = 1 [label = "4"];\n'
            'OPTION_C = 2 [label = "5"];\nOPTION_D = 3 [label = "6"];\n}' + SUFFIX
        ),
        "shell_heredoc": "cat <<'EOF'\nWhat is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6\nEOF" + SUFFIX,
        "toml_config": (
            '[question]\ntext = "What is 2+2?"\n\n[options]\n'
            'A = "3"\nB = "4"\nC = "5"\nD = "6"' + SUFFIX
        ),
    }


def _source_frame(*, split: str = "train") -> pd.DataFrame:
    prompts = _source_prompts()
    return pd.DataFrame(
        [
            {
                "item_id": "item-1",
                "subject": "math",
                "split": split,
                "question": QUESTION,
                "choices": CHOICES,
                "correct_index": 1,
                "wrapper_name": wrapper,
                "wrapped_prompt": prompts[wrapper],
            }
            for wrapper in ACTIVE_WRAPPERS
        ]
    )


def test_balanced_assignments_isolate_position_and_letter():
    assignments = balanced_assignments()

    assert len(assignments) == 7
    identity = assignments[0]
    assert identity.manipulation == "controlled_baseline"
    assert identity.content_ids_by_position == (0, 1, 2, 3)
    assert identity.labels_by_position == ("A", "B", "C", "D")

    position_only = [assignment for assignment in assignments if assignment.manipulation == "position_only"]
    label_only = [assignment for assignment in assignments if assignment.manipulation == "label_only"]
    assert len(position_only) == len(label_only) == 3
    assert {assignment.label_shift for assignment in position_only} == {0}
    assert {assignment.position_shift for assignment in label_only} == {0}

    for content_id in range(4):
        assert {
            assignment.content_ids_by_position.index(content_id) for assignment in [identity, *position_only]
        } == {0, 1, 2, 3}
        assert {
            assignment.labels_by_position[assignment.content_ids_by_position.index(content_id)]
            for assignment in [identity, *label_only]
        } == {"A", "B", "C", "D"}


def test_build_design_has_seven_isolated_letter_variants_text_and_plain_control():
    source = _source_frame()

    design = build_causal_design(source, splits=("train",))

    assert len(design) == len(CONTROLLED_FORMATS) * 8
    assert set(design["item_id"]) == {"item-1"}
    assert set(design["wrapper_name"]) == set(CONTROLLED_FORMATS)
    assert len(design[design["arm"] == "letter_intervention"]) == len(CONTROLLED_FORMATS) * 7
    assert len(design[design["arm"] == "answer_text"]) == len(CONTROLLED_FORMATS)
    assert design["work_key"].is_unique
    for _, group in design.groupby("wrapper_name"):
        letter = group[group["arm"] == "letter_intervention"]
        assert set(letter["manipulation"]) == {"controlled_baseline", "position_only", "label_only"}
        assert set(letter["variant"].astype(int)) == set(range(7))
        assert group[group["arm"] == "answer_text"]["variant"].tolist() == [0]


def test_build_design_uses_exact_source_and_existing_vanilla_prompts_for_baselines():
    source = _source_frame()

    design = build_causal_design(source)

    baselines = design[
        (design["arm"] == "letter_intervention")
        & (design["manipulation"] == "controlled_baseline")
    ].set_index("wrapper_name")
    expected = source.set_index("wrapper_name")["wrapped_prompt"].to_dict()
    for wrapper in ACTIVE_WRAPPERS:
        assert baselines.loc[wrapper, "prompt"] == expected[wrapper]
        assert baselines.loc[wrapper, "prompt_sha256"] == baselines.loc[wrapper, "source_prompt_sha256"]
    assert baselines.loc["plain", "prompt"] == build_vanilla_prompt(QUESTION, CHOICES)


def test_source_counterfactuals_change_only_the_requested_key_equals_feature():
    design = build_causal_design(_source_frame())
    block = design[design["wrapper_name"] == "key_equals"].set_index(["arm", "variant"])
    source = _source_prompts()["key_equals"]

    assert block.loc[("letter_intervention", 0), "prompt"] == source
    assert block.loc[("letter_intervention", 1), "prompt"] == (
        "QUESTION=What is 2+2?\nD=6\nA=3\nB=4\nC=5" + SUFFIX
    )
    assert block.loc[("letter_intervention", 4), "prompt"] == (
        "QUESTION=What is 2+2?\nB=3\nC=4\nD=5\nA=6" + SUFFIX
    )
    assert block.loc[("letter_intervention", 1), "correct_label"] == "B"
    assert block.loc[("letter_intervention", 4), "correct_label"] == "C"
    assert block.loc[("answer_text", 0), "prompt"] == source.replace(
        "Return only the letter (A, B, C, or D).",
        "Return only the exact answer text, not its letter.",
    )
    assert block.loc[("answer_text", 0), "correct_text"] == "4"


def test_v3_uses_displayed_choice_override_for_reordered_source_prompt():
    source = _source_frame()
    prompt = "QUESTION=What is 2+2?\nA=6\nB=3\nC=4\nD=5" + SUFFIX
    source.loc[source["wrapper_name"] == "key_equals", "wrapped_prompt"] = prompt

    design, _ = build_causal_design_v3(
        source,
        choice_overrides={
            ("item-1", "key_equals"): {
                "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "candidate_texts": ["6", "3", "4", "5"],
                "correct_position": 2,
                "provenance": "gpt-5.6-luna:xhigh:thread-1",
            }
        },
    )
    block = design[design["wrapper_name"] == "key_equals"].set_index(["arm", "variant"])

    assert block.loc[("letter_intervention", 0), "correct_label"] == "C"
    assert block.loc[("answer_text", 0), "correct_text"] == "4"
    assert block.loc[("answer_text", 0), "candidate_texts"] == ["6", "3", "4", "5"]
    assert block.loc[("letter_intervention", 1), "correct_label"] == "C"
    assert block.loc[("letter_intervention", 1), "correct_position"] == 3
    assert block.loc[("letter_intervention", 4), "correct_label"] == "D"


def test_v3_scores_the_answer_text_displayed_in_the_stored_prompt():
    source = _source_frame()
    prompt = "QUESTION=What is 2+2?\nA=3\nB=4 points\nC=5\nD=6" + SUFFIX
    source.loc[source["wrapper_name"] == "key_equals", "wrapped_prompt"] = prompt

    design, _ = build_causal_design_v3(source)
    answer_text = design[
        (design["wrapper_name"] == "key_equals") & (design["arm"] == "answer_text")
    ].iloc[0]

    assert answer_text["candidate_texts"] == ["3", "4 points", "5", "6"]
    assert answer_text["correct_text"] == "4 points"


def test_v3_scores_multicolumn_csv_rows_as_the_displayed_answer_text():
    source = _source_frame()
    source.loc[source["wrapper_name"] == "csv_inline", "wrapped_prompt"] = (
        "Option,Value,Unit\nA,3,points\nB,4,points\nC,5,points\nD,6,points" + SUFFIX
    )

    design, _ = build_causal_design_v3(source)
    answer_text = design[
        (design["wrapper_name"] == "csv_inline") & (design["arm"] == "answer_text")
    ].iloc[0]

    assert answer_text["candidate_texts"] == [
        "3,points", "4,points", "5,points", "6,points"
    ]
    assert answer_text["correct_text"] == "4,points"


def test_source_counterfactuals_support_real_wide_csv_and_graphql_record_shapes():
    prompts = _source_prompts()
    prompts["csv_inline"] = (
        "Question,Option A,Option B,Option C,Option D\n"
        "What is 2+2?,3,4,5,6" + SUFFIX
    )
    prompts["graphql_query"] = (
        'query { question: "What is 2+2?" answer(options: ['
        '{ letter: "A", value: "3" }, { letter: "B", value: "4" }, '
        '{ letter: "C", value: "5" }, { letter: "D", value: "6" }]) }' + SUFFIX
    )
    source = _source_frame()
    source["wrapped_prompt"] = source["wrapper_name"].map(prompts)

    design = build_causal_design(source)

    csv_rows = design[design["wrapper_name"] == "csv_inline"].set_index(["arm", "variant"])
    assert csv_rows.loc[("letter_intervention", 1), "prompt"] == (
        "Question,Option D,Option A,Option B,Option C\n"
        "What is 2+2?,6,3,4,5" + SUFFIX
    )
    assert csv_rows.loc[("letter_intervention", 4), "prompt"] == (
        "Question,Option B,Option C,Option D,Option A\n"
        "What is 2+2?,3,4,5,6" + SUFFIX
    )
    graphql = design[design["wrapper_name"] == "graphql_query"].set_index(["arm", "variant"])
    assert '{ letter: "D", value: "6" }' in graphql.loc[("letter_intervention", 1), "prompt"]
    assert '{ letter: "B", value: "3" }' in graphql.loc[("letter_intervention", 4), "prompt"]


def test_untransformable_wrapper_excludes_the_whole_item_with_a_reason():
    source = _source_frame()
    source.loc[source["wrapper_name"] == "protobuf_msg", "wrapped_prompt"] = (
        "message MCQ { repeated string options = 1 [\"3\", \"4\", \"5\", \"6\"]; }"
        + SUFFIX
    )

    design, exclusions = build_causal_design_with_exclusions(source)

    assert design.empty
    assert exclusions.to_dict("records") == [
        {
            "item_id": "item-1",
            "split": "train",
            "subject": "math",
            "correct_label": "B",
            "wrapper_name": "protobuf_msg",
            "reason": "option_labels_not_unambiguous",
        }
    ]
    with pytest.raises(ValueError, match="build_causal_design_with_exclusions"):
        build_causal_design(source)


@pytest.mark.parametrize(
    ("wrapper", "distracting"),
    [
        ("csv_inline", "A,extra\nB,extra\nC,extra\nD,extra\n"),
        ("graphql_query", '{ A: "extra", B: "extra", C: "extra", D: "extra" }\n'),
        ("ini_file", "A=extra\nB=extra\nC=extra\nD=extra\n"),
        ("key_equals", "A=extra\nB=extra\nC=extra\nD=extra\n"),
        ("protobuf_msg", "OPTION_A=extra\nOPTION_B=extra\nOPTION_C=extra\nOPTION_D=extra\n"),
        ("shell_heredoc", "A) extra\nB) extra\nC) extra\nD) extra\n"),
        ("toml_config", "A=\"extra\"\nB=\"extra\"\nC=\"extra\"\nD=\"extra\"\n"),
    ],
)
def test_balanced_non_option_label_construct_is_excluded_instead_of_edited(wrapper, distracting):
    source = _source_frame()
    source.loc[source["wrapper_name"] == wrapper, "wrapped_prompt"] = (
        distracting + _source_prompts()[wrapper]
    )

    design, exclusions = build_causal_design_with_exclusions(source)

    assert design.empty
    assert exclusions[exclusions["wrapper_name"] == wrapper]["reason"].tolist() == [
        "option_labels_not_unambiguous"
    ]


def test_html_changes_only_labels_inside_the_answer_option_containers():
    source = _source_frame()
    original = _source_prompts()["html_form"]
    distracting = "<legend>A) alpha B) beta C) gamma D) delta</legend>\n"
    source.loc[source["wrapper_name"] == "html_form", "wrapped_prompt"] = (
        original.replace("<form>\n", "<form>\n" + distracting)
    )

    design = build_causal_design(source)
    prompt = design[
        (design["wrapper_name"] == "html_form")
        & (design["arm"] == "letter_intervention")
        & (design["variant"] == 4)
    ]["prompt"].item()

    assert distracting in prompt
    assert '<option value="B">B) 3</option>' in prompt
    assert '<option value="C">C) 4</option>' in prompt


def test_source_prompt_hashes_bind_original_wrappers_without_calling_them_identity():
    source = _source_frame()
    design = build_causal_design(source)

    wrappers = design[design["wrapper_name"] != "plain"]
    assert wrappers["source_prompt_sha256"].notna().all()
    assert set(wrappers["template_scope"]) == {"source_prompt_counterfactual"}
    assert design[design["wrapper_name"] == "plain"]["source_prompt_sha256"].notna().all()


def test_v3_retains_whole_item_when_one_format_has_implicit_labels():
    source = _source_frame()
    source.loc[source["wrapper_name"] == "graphql_query", "wrapped_prompt"] = (
        'query { answer(options: ["3", "4", "5", "6"]) }' + SUFFIX
    )

    design, applicability = build_causal_design_v3(source)

    assert set(design["wrapper_name"]) == set(CONTROLLED_FORMATS)
    graphql = design[design["wrapper_name"] == "graphql_query"]
    assert set(graphql["arm"]) == {"letter_intervention", "answer_text"}
    assert graphql[graphql["arm"] == "letter_intervention"]["manipulation"].tolist() == [
        "controlled_baseline"
    ]
    row = applicability.set_index("wrapper_name").loc["graphql_query"]
    assert bool(row["baseline_applicable"])
    assert bool(row["text_applicable"])
    assert not bool(row["position_applicable"])
    assert not bool(row["label_applicable"])
    assert row["not_applicable_reason"] == "independent_label_position_not_identifiable"


def test_v3_generates_complete_rotations_for_separable_formats():
    design, applicability = build_causal_design_v3(_source_frame())

    assert len(design) == len(CONTROLLED_FORMATS) * 8
    columns = ["baseline_applicable", "text_applicable", "position_applicable", "label_applicable"]
    assert applicability[columns].all().all()
    for _, group in design.groupby(["item_id", "wrapper_name"]):
        assert set(group[group["manipulation"] == "position_only"]["variant"]) == {1, 2, 3}
        assert set(group[group["manipulation"] == "label_only"]["variant"]) == {4, 5, 6}


def test_v3_preserves_shapes_already_supported_by_the_legacy_transformer():
    source = _source_frame()
    source.loc[source["wrapper_name"] == "csv_inline", "wrapped_prompt"] = (
        "Question,Option A,Option B,Option C,Option D\n"
        "What is 2+2?,3,4,5,6" + SUFFIX
    )

    legacy = build_causal_design(source)
    design, applicability = build_causal_design_v3(source)

    csv_legacy = legacy[legacy["wrapper_name"] == "csv_inline"].set_index("work_key")
    csv_v3 = design[design["wrapper_name"] == "csv_inline"].set_index("work_key")
    assert csv_v3["prompt"].to_dict() == csv_legacy["prompt"].to_dict()
    status = applicability.set_index("wrapper_name").loc["csv_inline"]
    assert bool(status["position_applicable"])
    assert bool(status["label_applicable"])
    assert status["parse_provenance"] == "legacy_deterministic"


def test_v3_records_supplied_option_map_provenance():
    source = _source_frame()
    prompt = _source_prompts()["key_equals"]
    parsed = replace(
        parse_prompt_options(prompt, "key_equals", CHOICES),
        provenance="gpt-5.6-luna:xhigh:thread-1",
    )

    _, applicability = build_causal_design_v3(
        source,
        option_maps={("item-1", "key_equals"): parsed},
    )

    status = applicability.set_index("wrapper_name").loc["key_equals"]
    assert status["parse_provenance"] == "gpt-5.6-luna:xhigh:thread-1"
