from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "analysis" / "compare_model_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_model_runs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _make_run(root: Path, model: str, offset: float) -> None:
    (root / "metadata").mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "processed").mkdir()
    (root / "metadata" / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "canary": False,
                "model": {"id": model},
                "semantic_identity": {"semantic_run_id": model},
            }
        )
    )
    items = [f"i{index}" for index in range(3000)]
    wrappers = [f"wrapper-{index}" for index in range(8)]
    rows = []
    for item_index, item in enumerate(items):
        for wrapper_index, wrapper in enumerate(wrappers):
            correct = not (item_index in {0, 1} and wrapper_index == 0)
            calibration_kind = "same_wrapper_redaction"
            if item_index == 1 and wrapper_index > 0:
                calibration_kind = "generic_fallback"
            rows.append(
                {
                    "item_id": item,
                    "wrapper_name": wrapper,
                    "cal_correct": correct,
                    "cal_margin": offset,
                    "content_free_calibration_kind": calibration_kind,
                }
            )
    behavioral = pd.DataFrame(rows)
    behavioral.to_parquet(root / "raw" / "behavioral_scores.parquet", index=False)
    pd.DataFrame({"item_id": items[:10]}).to_parquet(root / "processed" / "conflict_pairs.parquet", index=False)
    pd.DataFrame(
        {
            "anchor": ["all_option_ends"],
            "layer": [2],
            "run_kind": ["corrupt_wrong_wrapped"],
            "attention_mass_question_options": [0.5 + offset],
            "attention_mass_wrapper_syntax": [0.2],
            "attention_entropy": [1.0],
        }
    ).to_parquet(root / "processed" / "attention_diagnostics.parquet", index=False)
    pd.DataFrame(
        {
            "anchor": ["all_option_ends"],
            "layer": [2],
            "cos_clean_vanilla": [0.5],
            "cos_corrupt_vanilla": [0.4],
            "delta_clean_minus_corrupt": [0.1],
        }
    ).to_parquet(root / "processed" / "vanilla_convergence.parquet", index=False)
    pd.DataFrame(
        {
            "condition": ["same_item_clean_to_corrupt"] * 2,
            "anchor": ["all_option_ends"] * 2,
            "layer": [2, 2],
            "patched_correct": [True, False],
            "margin_improvement": [0.2, -99.0],
            "recovery": [0.3 + offset, -99.0],
            "skipped": [False, True],
        }
    ).to_parquet(root / "processed" / "focused_patching_controls.parquet", index=False)


def test_cross_model_analysis_writes_tables_figures_and_paper_assets(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _make_run(first, "model-a", 0.0)
    _make_run(second, "model-b", 0.1)
    output = tmp_path / "comparison"
    paper = tmp_path / "paper"

    paths = MODULE.build_comparison(
        [MODULE.load_model_run(f"A={first}"), MODULE.load_model_run(f"B={second}")],
        output,
        paper_dir=paper,
    )

    assert all(path.exists() for path in paths.values())
    assert (output / "figures" / "cross_model_behavioral.png").exists()
    assert (paper / "figures" / "cross_model_controls.pdf").exists()
    assert (paper / "tables" / "table_cross_model_behavioral.tex").exists()
    controls = pd.read_csv(output / "tables" / "cross_model_controls_summary.csv")
    assert controls.sort_values("model")["recovery"].tolist() == [0.3, 0.4]
    behavioral = pd.read_csv(output / "tables" / "cross_model_behavioral_summary.csv")
    assert set(behavioral["full_conflict_items"]) == {2}
    assert set(behavioral["full_conflict_denominator"]) == {3000}
    assert behavioral["full_conflict_rate"].tolist() == pytest.approx([2 / 3000, 2 / 3000])
    assert set(behavioral["eligible_conflict_items"]) == {1}
    assert set(behavioral["eligible_conflict_denominator"]) == {3000}


def test_wrapper_audit_sensitivity_excludes_only_declared_rows(tmp_path):
    root = tmp_path / "run"
    _make_run(root, "model-a", 0.0)
    behavioral = pd.read_parquet(root / "raw" / "behavioral_scores.parquet")
    audit = behavioral[["item_id", "wrapper_name"]].copy()
    audit["final_label"] = "formatting_only"
    audit.loc[(audit["item_id"] == "i0") & (audit["wrapper_name"] == "wrapper-0"), "final_label"] = "content_changed"

    result = MODULE.build_audit_sensitivity([MODULE.load_model_run(f"A={root}")], audit)

    meaning = result[result["population"] == "meaning_preserving"].iloc[0]
    assert meaning["rows"] == 23999
    assert meaning["conflict_items"] == 1
    assert meaning["denominator"] == 3000
