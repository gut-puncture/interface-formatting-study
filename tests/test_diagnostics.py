from __future__ import annotations

import math

import pandas as pd
import torch

from interface_formatting_study.diagnostics import attention_role_metrics, run_vanilla_convergence, safe_cosine_similarity


def test_attention_role_metrics_aggregate_expected_positions():
    attention = torch.tensor(
        [
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.25, 0.25, 0.25, 0.25],
                [0.70, 0.10, 0.10, 0.10],
                [0.40, 0.30, 0.20, 0.10],
            ],
            [
                [0.40, 0.30, 0.20, 0.10],
                [0.10, 0.20, 0.30, 0.40],
                [0.25, 0.25, 0.25, 0.25],
                [0.10, 0.10, 0.10, 0.70],
            ],
        ]
    )
    roles = ["question", "option_A", "wrapper_syntax", "instruction"]

    metrics = attention_role_metrics(attention, [1], roles)

    head0 = metrics[metrics["head"] == 0].iloc[0]
    assert head0["attention_mass_question"] == 0.25
    assert head0["attention_mass_options"] == 0.25
    assert head0["attention_mass_question_options"] == 0.50
    assert head0["attention_mass_wrapper_syntax"] == 0.25
    assert head0["attention_entropy"] > 0


def test_safe_cosine_similarity_handles_zero_vectors():
    assert math.isnan(safe_cosine_similarity(torch.zeros(3), torch.ones(3)))
    assert safe_cosine_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 1.0


def test_vanilla_convergence_resolves_clean_corrupt_and_vanilla_anchors(boundary_tokenizer, tiny_hook_model):
    tiny_hook_model.embed = torch.nn.Embedding(512, 3)
    tiny_hook_model.lm_head = torch.nn.Linear(3, 512, bias=False)
    pairs = pd.DataFrame(
        [
            {
                "item_id": "i0",
                "subject": "toy",
                "split": "validation",
                "question": "Q?",
                "choices": ["a", "b", "c", "d"],
                "correct_label": "A",
                "clean_wrapper": "clean",
                "corrupt_wrapper": "corrupt",
                "clean_prompt": "Q?\nA) a\nB) b\nC) c\nD) d\n\nReturn only the letter (A, B, C, or D).\nAnswer:",
                "corrupt_prompt": "wrap { Q? option A=a option B=b option C=c option D=d }\n\nReturn only the letter (A, B, C, or D).\nAnswer:",
            }
        ]
    )

    results = run_vanilla_convergence(
        tiny_hook_model,
        boundary_tokenizer,
        pairs,
        layers=[0],
        split="validation",
        cap=None,
        anchors=("options_end", "all_option_ends"),
    )

    assert set(results["anchor"]) == {"options_end", "all_option_ends"}
    assert results["clean_positions"].notna().all()
    assert results["corrupt_positions"].notna().all()
    assert results["vanilla_positions"].notna().all()
