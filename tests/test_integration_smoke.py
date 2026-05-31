from __future__ import annotations

import pandas as pd
import torch

from interface_formatting_study.access_vector import build_access_vector
from interface_formatting_study.calibration import calibrate_scores, make_content_free_prompt
from interface_formatting_study.conflicts import construct_conflict_pairs
from interface_formatting_study.experiment import prepare_dataset
from interface_formatting_study.splits import assign_splits_by_item, assert_item_split_integrity


def test_end_to_end_fake_pipeline_contract():
    rows = []
    for item in range(5):
        for wrapper in range(4):
            correct = wrapper % 2 == 0
            rows.append(
                {
                    "item_id": f"i{item}",
                    "subject": "math",
                    "wrapper_name": f"w{wrapper}",
                    "wrapper_category": "pure_interface",
                    "question": "Q?",
                    "choices": ["a", "b", "c", "d"],
                    "wrapped_prompt": "Q? A) a B) b C) c D) d Return only the letter\nAnswer: ",
                    "correct_label": "A",
                    "cal_correct": correct,
                    "cal_margin": 1.0 if correct else -1.0 - wrapper,
                    "cal_pred_label": "A" if correct else "B",
                }
            )
    df = assign_splits_by_item(pd.DataFrame(rows), seed=0)
    assert_item_split_integrity(df)
    pairs = construct_conflict_pairs(df)
    assert len(pairs) == 5
    prompt = make_content_free_prompt(df.iloc[0].to_dict())
    assert "Return only the letter" in prompt
    assert calibrate_scores({"A": 2, "B": 1, "C": 0, "D": -1}, {"A": 1, "B": 1, "C": 1, "D": 1})["A"] == 1
    vector = build_access_vector(torch.eye(4), torch.zeros(4, 4), ["A", "B", "C", "D"])
    assert vector.shape == (4,)


def test_prepare_dataset_filters_to_verified_pure_interface_wrappers(tmp_path):
    path = tmp_path / "mini.jsonl"
    rows = []
    for wrapper, prompt in [
        ("csv_inline", "Q?,A,alpha\nQ?,B,beta\nQ?,C,gamma\nQ?,D,delta\nAnswer: "),
        ("haiku_riddle", "moonlit puzzle whispers\nA) alpha B) beta C) gamma D) delta\nAnswer: "),
    ]:
        rows.append(
            {
                "example_id": "i1",
                "subject": "math",
                "question": "Q?",
                "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                "correct_key": "A",
                "wrapper_id": wrapper,
                "prompt_text": prompt,
                "split": "test",
            }
        )
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")

    df, audit = prepare_dataset(
        {
            "dataset_path": str(path),
            "seed": 0,
            "experiment": {"wrapper_categories": ["pure_interface"]},
        }
    )

    assert set(df["wrapper_name"]) == {"csv_inline"}
    assert audit.set_index("wrapper_name").loc["haiku_riddle", "category"] == "style_transforming"
