from __future__ import annotations

import pandas as pd
import pytest
import torch

from interface_formatting_study.behavioral_eval import _score_records_reference, evaluate_behavioral


def _mini_behavioral_frame(n_rows: int = 5) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        rows.append(
            {
                "item_id": f"i{index}",
                "subject": "math",
                "wrapper_name": "key_equals",
                "wrapper_category": "pure_interface",
                "question": f"Question {index}?",
                "choices": ["zero", "one", "two", "three"],
                "wrapped_prompt": (
                    f"Question {index}? A=zero B=one C=two D=three\n"
                    "Return only the letter (A, B, C, or D).\nAnswer: "
                ),
                "correct_label": "A",
                "split": "train",
            }
        )
    return pd.DataFrame(rows)


def test_behavioral_eval_writes_resume_shards(tmp_path, boundary_tokenizer, rule_model):
    df = _mini_behavioral_frame()
    first = evaluate_behavioral(
        df,
        rule_model,
        boundary_tokenizer,
        sequence_batch_size=8,
        checkpoint_size=2,
        checkpoint_dir=tmp_path,
        resume=False,
    )
    assert len(first) == len(df)
    assert len(list(tmp_path.glob("part_*.parquet"))) == 3

    class ExplodingModel(torch.nn.Module):
        def forward(self, *args, **kwargs):  # pragma: no cover - should never run
            raise AssertionError("resume should load existing shards")

    resumed = evaluate_behavioral(
        df,
        ExplodingModel(),
        boundary_tokenizer,
        sequence_batch_size=8,
        checkpoint_size=2,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    pd.testing.assert_frame_equal(first.reset_index(drop=True), resumed.reset_index(drop=True))


def test_behavioral_eval_uses_cached_content_free_prompts(boundary_tokenizer, rule_model):
    df = _mini_behavioral_frame(4)
    scored = evaluate_behavioral(df, rule_model, boundary_tokenizer, sequence_batch_size=8, checkpoint_size=4)
    assert len(scored) == 4
    assert set(scored["content_free_calibration_kind"]) == {"same_wrapper_redaction"}
    assert all(column in scored for column in ["raw_score_A", "bias_score_A", "cal_score_A", "cal_margin"])


def test_batched_behavioral_matches_row_wise_reference(boundary_tokenizer, causal_label_model):
    df = _mini_behavioral_frame(4)
    batched = evaluate_behavioral(
        df,
        causal_label_model,
        boundary_tokenizer,
        sequence_batch_size=16,
        checkpoint_size=4,
    )
    reference = _score_records_reference(
        df.to_dict("records"),
        causal_label_model,
        boundary_tokenizer,
        batch_size=16,
        bias_cache={},
    )
    pd.testing.assert_frame_equal(batched, reference)
