from __future__ import annotations

import json

import pandas as pd
import pytest

from causal_fixtures import source_prompt_frame
from interface_formatting_study.causal_design import build_causal_design
from interface_formatting_study.causal_runner import run_causal_design, validate_causal_design
from interface_formatting_study.model_profiles import get_model_profile
from interface_formatting_study.run_identity import build_semantic_identity


def _design() -> pd.DataFrame:
    return build_causal_design(source_prompt_frame())


def _identity(tmp_path, design: pd.DataFrame):
    design_path = tmp_path / "design.parquet"
    design.to_parquet(design_path, index=False)
    source = tmp_path / "source.py"
    source.write_text("version = 1\n")
    return build_semantic_identity(
        get_model_profile("qwen"),
        config={"study": "causal-v1", "scoring": {"batch_size": 8, "max_batch_tokens": 256}},
        dataset_path=design_path,
        source_paths=[source],
    )


def test_causal_runner_interrupt_resume_matches_uninterrupted(
    tmp_path,
    boundary_tokenizer,
    causal_label_model,
    monkeypatch,
):
    monkeypatch.setattr(
        "interface_formatting_study.causal_runner._generate_answers_many",
        lambda _model, _tokenizer, records, **_kwargs: [str(row["correct_text"]) for row in records],
    )
    design = _design()
    identity = _identity(tmp_path, design)

    full = run_causal_design(
        design,
        causal_label_model,
        boundary_tokenizer,
        identity,
        tmp_path / "full",
        batch_size=8,
        max_batch_tokens=256,
        checkpoint_size=7,
    )

    checks = 0

    def stop_after_first_shard():
        nonlocal checks
        checks += 1
        return checks >= 2

    partial = run_causal_design(
        design,
        causal_label_model,
        boundary_tokenizer,
        identity,
        tmp_path / "resumed",
        batch_size=8,
        max_batch_tokens=256,
        checkpoint_size=7,
        should_stop=stop_after_first_shard,
    )
    assert 0 < len(partial) < len(design)

    resumed = run_causal_design(
        design,
        causal_label_model,
        boundary_tokenizer,
        identity,
        tmp_path / "resumed",
        batch_size=8,
        max_batch_tokens=256,
        checkpoint_size=7,
    )

    pd.testing.assert_frame_equal(full, resumed)
    assert len(resumed) == len(design)
    assert set(resumed["arm"]) == {"letter_intervention", "answer_text"}
    progress = json.loads((tmp_path / "resumed" / "progress.json").read_text())
    assert progress["status"] == "complete"
    assert progress["completed"] == len(design)


def test_causal_design_validation_rejects_duplicate_or_inconsistent_work():
    design = _design()
    duplicate = pd.concat([design, design.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate work keys"):
        validate_causal_design(duplicate)

    broken = design.copy()
    broken.loc[broken["arm"] == "letter_intervention", "correct_label"] = "Z"
    with pytest.raises(ValueError, match="correct_label"):
        validate_causal_design(broken)

    unbalanced = design.copy()
    first_letter = unbalanced[unbalanced["arm"] == "letter_intervention"].index[0]
    unbalanced.at[first_letter, "content_ids_by_position"] = [0, 1, 3, 2]
    correct_content = int(unbalanced.at[first_letter, "correct_content_id"])
    correct_position = unbalanced.at[first_letter, "content_ids_by_position"].index(correct_content)
    unbalanced.at[first_letter, "correct_position"] = correct_position
    unbalanced.at[first_letter, "correct_label"] = unbalanced.at[first_letter, "labels_by_position"][correct_position]
    with pytest.raises(ValueError, match="not balanced"):
        validate_causal_design(unbalanced)
