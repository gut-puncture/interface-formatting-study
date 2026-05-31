from __future__ import annotations

import pandas as pd
import torch

from interface_formatting_study.component_patching import run_focused_patching_controls, select_control_donor, summarize_focused_controls


def _pair(item_id: str, label: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "subject": "toy",
        "split": "validation",
        "question": f"Question {item_id}?",
        "choices": ["a", "b", "c", "d"],
        "correct_label": label,
        "clean_wrapper": f"clean_{item_id}",
        "corrupt_wrapper": f"corrupt_{item_id}",
        "clean_prompt": f"clean prompt {item_id}",
        "corrupt_prompt": f"corrupt prompt {item_id}",
        "clean_margin": 1.0,
        "corrupt_margin": -1.0,
    }


def test_control_donor_selection_keeps_cross_item_donors_off_target_item():
    pool = pd.DataFrame([_pair("i0", "A"), _pair("i1", "A"), _pair("i2", "B")])
    target = pool.iloc[0].to_dict()

    cross = select_control_donor(pool, target, condition="cross_item_clean_to_corrupt")
    same_label = select_control_donor(pool, target, condition="same_label_cross_item_clean_to_corrupt")
    different_label = select_control_donor(pool, target, condition="different_label_cross_item_clean_to_corrupt")

    assert cross is not None
    assert cross.donor_item_id != "i0"
    assert same_label is not None
    assert same_label.donor_item_id != "i0"
    assert same_label.donor_label == "A"
    assert different_label is not None
    assert different_label.donor_item_id != "i0"
    assert different_label.donor_label != "A"


def test_same_item_and_vanilla_donors_record_target_provenance():
    pool = pd.DataFrame([_pair("i0", "A"), _pair("i1", "B")])
    target = pool.iloc[0].to_dict()

    same_item = select_control_donor(pool, target, condition="same_item_clean_to_corrupt")
    vanilla = select_control_donor(pool, target, condition="vanilla_to_corrupt")

    assert same_item is not None
    assert same_item.donor_item_id == "i0"
    assert same_item.donor_wrapper == "clean_i0"
    assert vanilla is not None
    assert vanilla.donor_item_id == "i0"
    assert vanilla.donor_wrapper == "vanilla"
    assert "A) a" in vanilla.donor_prompt


def test_focused_controls_summary_uses_non_skipped_rows_and_non_null_provenance():
    controls = pd.DataFrame(
        [
            {
                "item_id": "i0",
                "condition": "vanilla_to_corrupt",
                "anchor": "options_end",
                "layer": 2,
                "n_positions": 1,
                "patched_correct": True,
                "margin_improvement": 0.5,
                "recovery": 0.25,
                "donor_item_equals_target": True,
                "donor_label_relation": "same",
                "skipped": False,
            },
            {
                "item_id": "i1",
                "condition": "vanilla_to_corrupt",
                "anchor": "options_end",
                "layer": 2,
                "n_positions": 0,
                "patched_correct": False,
                "margin_improvement": 0.0,
                "recovery": 0.0,
                "donor_item_equals_target": True,
                "donor_label_relation": "same",
                "skipped": True,
            },
        ]
    )

    summary = summarize_focused_controls(controls)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["condition"] == "vanilla_to_corrupt"
    assert row["n_items"] == 1
    assert row["patched_accuracy"] == 1.0
    assert row["donor_item_equals_target_rate"] == 1.0


def test_cross_item_controls_resolve_donor_anchors_with_donor_content(boundary_tokenizer, tiny_hook_model):
    tiny_hook_model.embed = torch.nn.Embedding(512, 3)
    tiny_hook_model.lm_head = torch.nn.Linear(3, 512, bias=False)
    rows = [
        {
            **_pair("i0", "A"),
            "question": "Question zero?",
            "choices": ["aa", "bb", "cc", "dd"],
            "clean_prompt": "Question zero?\nA) aa\nB) bb\nC) cc\nD) dd\n\nReturn only the letter (A, B, C, or D).\nAnswer:",
            "corrupt_prompt": "wrap Question zero? option A=aa option B=bb option C=cc option D=dd\n\nReturn only the letter (A, B, C, or D).\nAnswer:",
        },
        {
            **_pair("i1", "A"),
            "question": "Question one?",
            "choices": ["ee", "ff", "gg", "hh"],
            "clean_prompt": "Question one?\nA) ee\nB) ff\nC) gg\nD) hh\n\nReturn only the letter (A, B, C, or D).\nAnswer:",
            "corrupt_prompt": "wrap Question one? option A=ee option B=ff option C=gg option D=hh\n\nReturn only the letter (A, B, C, or D).\nAnswer:",
        },
    ]
    for row in rows:
        for label in ("A", "B", "C", "D"):
            row[f"corrupt_bias_score_{label}"] = 0.0

    results = run_focused_patching_controls(
        tiny_hook_model,
        boundary_tokenizer,
        pd.DataFrame(rows),
        layers=[0],
        split="validation",
        cap=None,
        anchors=("options_end",),
        conditions=("same_label_cross_item_clean_to_corrupt",),
    )

    assert not results.empty
    assert not results["skipped"].any()
    assert set(results["donor_item_equals_target"]) == {False}
    assert set(results["donor_label_relation"]) == {"same"}
