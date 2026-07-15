from __future__ import annotations

import pandas as pd

from interface_formatting_study.causal_design import (
    ACTIVE_WRAPPERS,
    balanced_assignments,
    build_causal_design,
    render_causal_prompt,
)


def test_balanced_assignments_independently_cycle_content_position_and_letter():
    assignments = balanced_assignments("item-17", "csv_inline")

    assert len(assignments) == 4
    assert assignments[0].content_ids_by_position == (0, 1, 2, 3)
    assert assignments[0].labels_by_position == ("A", "B", "C", "D")
    for content_id in range(4):
        positions = []
        labels = []
        for assignment in assignments:
            position = assignment.content_ids_by_position.index(content_id)
            positions.append(position)
            labels.append(assignment.labels_by_position[position])
        assert set(positions) == {0, 1, 2, 3}
        assert set(labels) == {"A", "B", "C", "D"}
    assert sorted(assignment.position_shift for assignment in assignments) == [0, 1, 2, 3]
    assert sorted(assignment.label_shift for assignment in assignments) == [0, 1, 2, 3]


def test_all_canonical_wrappers_render_both_readouts_with_correct_mapping():
    question = 'Which token means "less than" & remains semantic?'
    choices = ["<", ">", "&", "newline\nvalue"]
    assignment = balanced_assignments("item-special")[2]

    for wrapper in ACTIVE_WRAPPERS:
        letter = render_causal_prompt(wrapper, question, choices, assignment, correct_content_id=0, readout="letter")
        text = render_causal_prompt(wrapper, question, choices, assignment, correct_content_id=0, readout="text")
        position = assignment.content_ids_by_position.index(0)
        assert letter.correct_output == assignment.labels_by_position[position]
        assert text.correct_output == "<"
        assert "Return only the letter A, B, C, or D." in letter.prompt
        assert "Return only the exact answer text, not its letter." in text.prompt
        assert letter.prompt_sha256 != text.prompt_sha256


def test_build_design_has_four_letter_variants_one_text_variant_and_no_test_leakage():
    source = pd.DataFrame(
        [
            {
                "item_id": "item-1",
                "subject": "math",
                "split": "train",
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "correct_index": 1,
            },
            {
                "item_id": "item-test",
                "subject": "math",
                "split": "test",
                "question": "What is 3+3?",
                "choices": ["5", "6", "7", "8"],
                "correct_index": 1,
            },
        ]
    )

    design = build_causal_design(source, splits=("train",))

    assert len(design) == len(ACTIVE_WRAPPERS) * 5
    assert set(design["item_id"]) == {"item-1"}
    assert len(design[design["arm"] == "letter_permutation"]) == len(ACTIVE_WRAPPERS) * 4
    assert len(design[design["arm"] == "answer_text"]) == len(ACTIVE_WRAPPERS)
    assert design["work_key"].is_unique
    assert set(design[design["arm"] == "answer_text"]["variant"]) == {0}
