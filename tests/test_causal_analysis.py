from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from causal_fixtures import source_prompt_frame
from interface_formatting_study.causal_design import build_causal_design
from interface_formatting_study.run_identity import sha256_file


SCRIPT = Path(__file__).parents[1] / "analysis" / "analyze_causal_followup.py"
SPEC = importlib.util.spec_from_file_location("analyze_causal_followup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _run_fixture(root: Path) -> None:
    design = build_causal_design(source_prompt_frame())
    design["raw_predicted_content_id"] = design["correct_content_id"]
    design["cal_predicted_content_id"] = design["correct_content_id"]
    design["raw_predicted_label"] = design["correct_label"]
    design["cal_predicted_label"] = design["correct_label"]
    design["raw_correct"] = True
    design["cal_correct"] = True
    design["raw_margin"] = 1.0
    design["cal_margin"] = 1.0
    design["raw_entropy"] = 0.2
    design["cal_entropy"] = 0.2
    design["text_ambiguous"] = False
    design["candidate_mean_logps"] = design["candidate_texts"].map(lambda _: [-4.0, -1.0, -3.0, -5.0])
    design["candidate_total_correct"] = True
    design["generated_correct"] = True
    design["semantic_run_id"] = "semantic-one"
    design["model_id"] = "model-one"
    design["model_revision"] = "revision-one"
    raw = root / "raw" / "causal_behavior.parquet"
    raw.parent.mkdir(parents=True)
    design.to_parquet(raw, index=False)
    raw.with_name(raw.name + ".manifest.json").write_text(
        json.dumps({"semantic_run_id": "semantic-one", "sha256": sha256_file(raw), "row_count": len(design)})
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "canary": False,
                "model": {"id": "model-one", "revision": "revision-one"},
                "semantic_identity": {"semantic_run_id": "semantic-one"},
                "design": {"run_rows": len(design)},
            }
        )
    )


def test_causal_analysis_consumes_verified_run_and_writes_effects(tmp_path):
    run = tmp_path / "run"
    _run_fixture(run)

    outputs = MODULE.analyze([f"Tiny={run}"], tmp_path / "analysis", n_boot=20)

    assert all(path.exists() for path in outputs.values())
    summary = pd.read_csv(outputs["model_summary"])
    assert summary.loc[0, "controlled_baseline_cal_accuracy"] == 1.0
    assert summary.loc[0, "text_generated_accuracy"] == 1.0
    effects = pd.read_csv(outputs["paired_effects"])
    assert set(effects["comparison"]) == {
        "position_only_minus_baseline_raw",
        "position_only_minus_baseline_cal",
        "label_only_minus_baseline_raw",
        "label_only_minus_baseline_cal",
        "wrapper_amplification_position_only_raw",
        "wrapper_amplification_position_only_cal",
        "wrapper_amplification_label_only_raw",
        "wrapper_amplification_label_only_cal",
        "generated_text_minus_baseline_raw_descriptive",
        "generated_text_minus_baseline_cal_descriptive",
        "wrapper_amplification_generated_text_readout_raw_descriptive",
        "wrapper_amplification_generated_text_readout_cal_descriptive",
    }


def test_causal_analysis_retains_blocks_when_one_intervention_arm_is_not_applicable(tmp_path):
    run = tmp_path / "run"
    _run_fixture(run)
    raw = run / "raw" / "causal_behavior.parquet"
    frame = pd.read_parquet(raw)
    missing_wrapper = "graphql_query"
    frame = frame[
        ~(
            (frame["wrapper_name"] == missing_wrapper)
            & (frame["manipulation"] == "label_only")
        )
    ].copy()
    frame.to_parquet(raw, index=False)
    raw.with_name(raw.name + ".manifest.json").write_text(
        json.dumps({"semantic_run_id": "semantic-one", "sha256": sha256_file(raw), "row_count": len(frame)})
    )
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["design"]["run_rows"] = len(frame)
    manifest_path.write_text(json.dumps(manifest))

    outputs = MODULE.analyze([f"Tiny={run}"], tmp_path / "analysis", n_boot=20)

    blocks = pd.read_parquet(outputs["blocks"])
    graphql = blocks[blocks["wrapper_name"] == missing_wrapper].iloc[0]
    assert pd.isna(graphql["label_only_cal_correct"])
    assert not pd.isna(graphql["position_only_cal_correct"])
    effects = pd.read_csv(outputs["paired_effects"])
    label = effects[effects["comparison"] == "label_only_minus_baseline_cal"].iloc[0]
    position = effects[effects["comparison"] == "position_only_minus_baseline_cal"].iloc[0]
    assert label["n_rows"] == position["n_rows"] - 1
