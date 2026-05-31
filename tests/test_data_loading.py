from __future__ import annotations

import json

from interface_formatting_study.data_loading import dataset_audit, dataset_audit_from_frame, load_normalized, normalize_record


def test_normalize_actual_dataset_schema():
    record = {
        "example_id": "mmlu::test::0",
        "subject": "abstract_algebra",
        "question": "Q?",
        "options": {"A": "zero", "B": "four", "C": "two", "D": "six"},
        "correct_key": "B",
        "wrapper_id": "csv_inline",
        "prompt_text": "question, A, B, C, D\nQ?, zero, four, two, six\nAnswer: ",
        "split": "test",
    }
    ex = normalize_record(record)
    assert ex.item_id == "mmlu::test::0"
    assert ex.choices == ["zero", "four", "two", "six"]
    assert ex.correct_label == "B"
    assert ex.correct_index == 1
    assert ex.wrapper_name == "csv_inline"
    assert ex.wrapped_prompt.startswith("question")


def test_normalize_rejects_inconsistent_correct_index():
    record = {
        "example_id": "mmlu::test::0",
        "options": {"A": "zero", "B": "four", "C": "two", "D": "six"},
        "correct_key": "B",
        "correct_index": 2,
        "wrapper_id": "csv_inline",
        "prompt_text": "prompt",
    }
    try:
        normalize_record(record)
    except ValueError as exc:
        assert "inconsistent" in str(exc)
    else:
        raise AssertionError("Expected inconsistent correct_index to be rejected")


def test_dataset_audit_counts_items_wrappers_and_labels(tmp_path):
    path = tmp_path / "mini.jsonl"
    rows = []
    for item_id in ["i1", "i2"]:
        for wrapper in ["csv_inline", "html_form"]:
            rows.append(
                {
                    "example_id": item_id,
                    "subject": "math",
                    "question": "Q?",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                    "correct_key": "A",
                    "wrapper_id": wrapper,
                    "prompt_text": "Q? A) a B) b C) c D) d Answer: ",
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    audit = dataset_audit(path)
    assert audit["num_rows"] == 4
    assert audit["num_items"] == 2
    assert audit["wrapper_counts"] == {"csv_inline": 2, "html_form": 2}
    assert audit["correct_label_distribution"]["A"] == 4
    df = load_normalized(path)
    assert len(df) == 4


def test_dataset_audit_from_frame_counts_active_wrapper_subset(tmp_path):
    path = tmp_path / "mini.jsonl"
    rows = []
    for item_id in ["i1", "i2"]:
        rows.append(
            {
                "example_id": item_id,
                "subject": "math",
                "question": "Q?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "correct_key": "A",
                "wrapper_id": "csv_inline",
                "prompt_text": "Q? A) a B) b C) c D) d Answer: ",
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    df = load_normalized(path)
    df["wrapper_category"] = "pure_interface"

    audit = dataset_audit_from_frame(
        df,
        dataset_path=path,
        wrapper_policy="verified_pure_interface_only",
        source_num_rows=4,
    )

    assert audit["num_rows"] == 2
    assert audit["num_items"] == 2
    assert audit["num_wrappers"] == 1
    assert audit["wrapper_category_counts"] == {"pure_interface": 2}
