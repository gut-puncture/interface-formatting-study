from __future__ import annotations

import pandas as pd

from interface_formatting_study.causal_design import (
    ACTIVE_WRAPPERS,
    CONTROLLED_FORMATS,
    balanced_assignments,
    build_causal_design,
    render_causal_prompt,
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


def test_all_controlled_formats_render_both_readouts_and_calibration():
    question = 'Which token means "less than" & remains semantic?'
    choices = ["<", ">", "&", "newline\nvalue"]
    assignment = balanced_assignments()[2]

    for format_name in CONTROLLED_FORMATS:
        letter = render_causal_prompt(
            format_name, question, choices, assignment, correct_content_id=0, readout="letter"
        )
        text = render_causal_prompt(
            format_name, question, choices, assignment, correct_content_id=0, readout="text"
        )
        position = assignment.content_ids_by_position.index(0)
        assert letter.correct_output == assignment.labels_by_position[position]
        assert text.correct_output == "<"
        assert letter.calibration_prompt is not None
        assert question not in letter.calibration_prompt
        assert all(choice not in letter.calibration_prompt for choice in choices if len(choice) > 1)
        assert text.calibration_prompt is None
        assert letter.prompt_sha256 != text.prompt_sha256

    graphql = render_causal_prompt(
        "graphql_query", question, choices, assignment, correct_content_id=0, readout="letter"
    )
    assert all(f"slot{index}: option(" in graphql.prompt for index in range(4))


def test_build_design_has_seven_isolated_letter_variants_text_and_plain_control():
    source = pd.DataFrame(
        [
            {
                "item_id": "item-1",
                "subject": "math",
                "split": "train",
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "correct_index": 1,
                "wrapper_name": wrapper,
                "wrapped_prompt": f"original-{wrapper}",
            }
            for wrapper in ACTIVE_WRAPPERS
        ]
        + [
            {
                "item_id": "item-test",
                "subject": "math",
                "split": "test",
                "question": "What is 3+3?",
                "choices": ["5", "6", "7", "8"],
                "correct_index": 1,
                "wrapper_name": wrapper,
                "wrapped_prompt": f"original-test-{wrapper}",
            }
            for wrapper in ACTIVE_WRAPPERS
        ]
    )

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


def test_source_prompt_hashes_bind_original_wrappers_without_calling_them_identity():
    source = pd.DataFrame(
        [
            {
                "item_id": "item-1",
                "subject": "math",
                "split": "train",
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "correct_index": 1,
                "wrapper_name": wrapper,
                "wrapped_prompt": f"original-{wrapper}",
            }
            for wrapper in ACTIVE_WRAPPERS
        ]
    )
    design = build_causal_design(source)

    wrappers = design[design["wrapper_name"] != "plain"]
    assert wrappers["source_prompt_sha256"].notna().all()
    assert set(wrappers["template_scope"]) == {"controlled_canonical"}
    assert design[design["wrapper_name"] == "plain"]["source_prompt_sha256"].isna().all()
