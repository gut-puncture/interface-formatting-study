from __future__ import annotations

from interface_formatting_study.experiment import prepare_dataset
from interface_formatting_study.token_roles import (
    ROLE_ANSWER_PREFIX,
    ROLE_INSTRUCTION,
    ROLE_QUESTION,
    ROLE_WRAPPER_SYNTAX,
    label_token_roles,
)
from interface_formatting_study.utils import LABELS, read_yaml
from interface_formatting_study.vanilla import build_vanilla_prompt


def test_token_role_labeling_covers_all_active_wrappers(boundary_tokenizer):
    cfg = read_yaml("configs/default.yaml")
    df, _ = prepare_dataset(cfg)

    for wrapper, group in df.groupby("wrapper_name", sort=True):
        row = group.iloc[0]
        labeling = label_token_roles(
            boundary_tokenizer,
            str(row["wrapped_prompt"]),
            question=row["question"],
            choices=row["choices"],
        )

        assert labeling.role_positions[ROLE_QUESTION], wrapper
        for label in LABELS:
            assert labeling.role_positions[f"option_{label}"], (wrapper, label)
        assert labeling.role_positions[ROLE_WRAPPER_SYNTAX], wrapper
        assert labeling.role_positions[ROLE_INSTRUCTION], wrapper
        assert labeling.role_positions[ROLE_ANSWER_PREFIX], wrapper


def test_vanilla_prompt_roles_have_no_wrapper_syntax_except_spacing(boundary_tokenizer):
    prompt = build_vanilla_prompt("Question?", {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"})
    labeling = label_token_roles(
        boundary_tokenizer,
        prompt,
        question="Question?",
        choices={"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
    )

    assert labeling.role_positions[ROLE_QUESTION]
    assert labeling.role_positions["option_A"]
    assert labeling.role_positions[ROLE_INSTRUCTION]
    assert labeling.role_positions[ROLE_ANSWER_PREFIX]
    assert len(labeling.role_positions[ROLE_WRAPPER_SYNTAX]) < len(labeling.token_roles)
