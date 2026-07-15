from __future__ import annotations

import os
from types import SimpleNamespace

import pandas as pd
import pytest

from interface_formatting_study.cli import cmd_content_free_control, cmd_eval_vector, cmd_train_vector, cmd_tune_alpha, build_parser, cmd_figures, cmd_write_manifest
from interface_formatting_study.deliverables import missing_final_artifacts
from interface_formatting_study.utils import read_json, read_yaml, write_json


def test_parser_exposes_manifest_and_deliverables_commands():
    parser = build_parser()

    manifest_args = parser.parse_args(["write-manifest"])
    deliverable_args = parser.parse_args(["deliverables", "--no-compile"])
    attention_args = parser.parse_args(["attention-diagnostics", "--cap", "1"])
    convergence_args = parser.parse_args(["vanilla-convergence", "--layers", "2,4"])
    controls_args = parser.parse_args(["focused-patching-controls", "--anchors", "options_end"])
    tables_args = parser.parse_args(["tables"])

    assert manifest_args.command == "write-manifest"
    assert deliverable_args.command == "deliverables"
    assert deliverable_args.no_compile is True
    assert attention_args.command == "attention-diagnostics"
    assert convergence_args.command == "vanilla-convergence"
    assert controls_args.command == "focused-patching-controls"
    assert tables_args.include_legacy_semantic_patching is False


def test_write_manifest_records_dataset_hash_and_source_hashes(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"item_id":"i0"}\n')
    metadata_dir = tmp_path / "metadata"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"dataset_path: {dataset}",
                "model_name: fake-model",
                "outputs:",
                f"  metadata_dir: {metadata_dir}",
            ]
        )
    )
    write_json({"num_rows": 1}, metadata_dir / "dataset_audit.json")
    write_json({"train": {"num_items": 1}}, metadata_dir / "split_manifest.json")

    cmd_write_manifest(SimpleNamespace(config=str(config)))

    manifest = read_json(metadata_dir / "experiment_manifest.json")
    assert manifest["manifest_schema_version"] == 1
    assert len(manifest["dataset"]["sha256"]) == 64
    assert manifest["split_manifest"]["train"]["num_items"] == 1
    assert "src/interface_formatting_study/cli.py" in manifest["source_hashes"]


def test_write_manifest_source_hashes_do_not_depend_on_cwd(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"item_id":"i0"}\n')
    metadata_dir = tmp_path / "metadata"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"dataset_path: {dataset}",
                "model_name: fake-model",
                "outputs:",
                f"  metadata_dir: {metadata_dir}",
            ]
        )
    )
    write_json({"num_rows": 1}, metadata_dir / "dataset_audit.json")
    write_json({"train": {"num_items": 1}}, metadata_dir / "split_manifest.json")

    previous = os.getcwd()
    try:
        os.chdir(tmp_path)
        cmd_write_manifest(SimpleNamespace(config=str(config)))
    finally:
        os.chdir(previous)

    manifest = read_json(metadata_dir / "experiment_manifest.json")
    assert "src/interface_formatting_study/cli.py" in manifest["source_hashes"]


def test_write_manifest_records_cli_selected_model_instead_of_config_default(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"item_id":"i0"}\n')
    metadata_dir = tmp_path / "metadata"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"dataset_path: {dataset}\nmodel_name: default-qwen\noutputs:\n  metadata_dir: {metadata_dir}\n"
    )
    write_json({"num_rows": 1}, metadata_dir / "dataset_audit.json")
    write_json({"train": {"num_items": 1}}, metadata_dir / "split_manifest.json")

    cmd_write_manifest(SimpleNamespace(config=str(config), model="selected-model"))

    assert read_json(metadata_dir / "experiment_manifest.json")["model"]["name"] == "selected-model"


def test_figures_refuse_partial_non_smoke_behavioral_table(tmp_path):
    metadata_dir = tmp_path / "metadata"
    figures_dir = tmp_path / "figures"
    raw_dir = tmp_path / "raw"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "dataset_path: ignored.jsonl",
                "model_name: fake-model",
                "outputs:",
                f"  metadata_dir: {metadata_dir}",
                f"  figures_dir: {figures_dir}",
            ]
        )
    )
    write_json({"num_rows": 2}, metadata_dir / "dataset_audit.json")
    raw_dir.mkdir(parents=True)
    behavioral = raw_dir / "behavioral_scores.parquet"
    pd.DataFrame({"item_id": ["i0"], "wrapper_name": ["w0"], "cal_correct": [True]}).to_parquet(behavioral)

    with pytest.raises(ValueError, match="refusing to write publication-style figures"):
        cmd_figures(SimpleNamespace(config=str(config), behavioral=str(behavioral)))


@pytest.mark.parametrize(
    ("command", "name"),
    [
        (cmd_train_vector, "train-vector"),
        (cmd_tune_alpha, "tune-alpha"),
        (cmd_eval_vector, "eval-vector"),
        (cmd_content_free_control, "content-free-control"),
    ],
)
def test_final_mechanistic_commands_refuse_limit(tmp_path, command, name):
    config = tmp_path / "config.yaml"
    config.write_text("dataset_path: dataset.jsonl\nmodel_name: fake\noutputs:\n  metadata_dir: metadata\n")
    args = SimpleNamespace(config=str(config), limit=1)

    with pytest.raises(ValueError, match=f"{name} writes final artifacts"):
        command(args)


def test_missing_final_artifacts_reports_required_outputs(tmp_path):
    missing = missing_final_artifacts(tmp_path)

    assert "results/metadata/experiment_manifest.json" in missing
    assert "results/raw/behavioral_scores.parquet" in missing
    assert "results/processed/attention_diagnostics.parquet" in missing
    assert "results/processed/vanilla_convergence.parquet" in missing
    assert "results/processed/focused_patching_controls.parquet" in missing
    assert "results/tables/table_focused_controls.csv" in missing
    assert "results/processed/interface_formatting_study_vector.pt" not in missing


def test_default_config_uses_non_leaky_semantic_patching():
    cfg = read_yaml("configs/default.yaml")
    activation = cfg["activation"]
    focused = cfg["focused_mechanistic"]

    assert activation["layers"] == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    assert "answer_anchor" not in activation["anchors"]
    assert "question_end" in activation["anchors"]
    assert "all_option_ends" in activation["anchors"]
    assert focused["layers"] == [2, 4, 6, 8, 10, 12, 14, 16]
    assert focused["anchors"] == ["options_end", "all_option_ends"]
    assert "content_end" not in focused["anchors"]
    assert "same_item_clean_to_corrupt" in focused["conditions"]
    assert "vanilla_to_corrupt" in focused["conditions"]
    assert "same_label_cross_item_clean_to_corrupt" in focused["conditions"]


def test_gpu_script_default_pipeline_skips_vector_stages():
    script = open("scripts/run_interface_formatting_study_focused_mechanistic_gpu.sh", encoding="utf-8").read()

    assert "Attention diagnostics" in script
    assert "Vanilla convergence diagnostics" in script
    assert "Focused causal controls" in script
    assert "patching-sweep" not in script
    assert "train-vector" not in script
    assert "tune-alpha" not in script
    assert "eval-vector" not in script
    assert "content-free-control" not in script
    package_section = script.split("tar -czf", maxsplit=1)[1]
    assert "results/tables/table3_semantic_patching.csv" not in package_section
