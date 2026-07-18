from __future__ import annotations

import os
import subprocess
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


def test_causal_fetch_and_control_scripts_cover_canaries_and_persist_mode():
    fetch = open("scripts/fetch_causal_followup_artifacts.sh", encoding="utf-8").read()
    control = open("scripts/control_causal_followup_gpu.sh", encoding="utf-8").read()

    assert 'RUN_SUFFIX="/canaries/${CANARY_NAME}"' in fetch
    assert "MODE_FILE" in control
    assert 'echo "$MODE" > "$MODE_FILE"' in control


def test_causal_sync_copies_source_preserving_design_sidecars():
    sync = open("scripts/sync_interface_formatting_study_to_gpu.sh", encoding="utf-8").read()

    assert 'ACTIVE_DATASET_PATH%.*}.applicability.parquet' in sync
    assert 'ACTIVE_DATASET_PATH}.manifest.json' in sync
    assert 'cd "$ROOT_DIR"' in sync
    assert '"./$relative_path"' in sync


def test_gpu_bootstrap_supports_plain_ubuntu_without_changing_the_runtime_contract():
    bootstrap = open("scripts/bootstrap_causal_followup_gpu.sh", encoding="utf-8").read()
    run = open("scripts/run_decision_binding_content_gpu.sh", encoding="utf-8").read()

    assert "UV_UNMANAGED_INSTALL" in bootstrap
    assert '"$UV_BIN" python install 3.11' in bootstrap
    assert '"$UV_BIN" venv --python 3.11' in bootstrap
    assert 'torch==2.7.1' in bootstrap
    assert "https://download.pytorch.org/whl/cu126" in bootstrap
    assert '.venv/bin/python' in run
    assert '"$PYTHON_BIN" -m interface_formatting_study.decision_binding_content_cli' in run
    assert 'BATCH_SIZE:-32' in run
    assert 'MAX_BATCH_TOKENS:-40000' in run
    assert 'MAX_ITER:-100' in run
    assert 'torch.__version__.split("+")[0] == "2.7.1"' in bootstrap
    assert 'torch.version.cuda == "12.6"' in bootstrap


def test_gpu_bootstrap_rerun_uses_uv_for_an_existing_uv_environment(tmp_path):
    project = tmp_path / "project"
    home = tmp_path / "home"
    python_log = tmp_path / "python.log"
    uv_log = tmp_path / "uv.log"
    (project / ".venv/bin").mkdir(parents=True)
    (home / ".local/bin").mkdir(parents=True)
    fake_python = project / ".venv/bin/python"
    fake_python.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{python_log}"\ncat >/dev/null\n',
        encoding="utf-8",
    )
    fake_uv = home / ".local/bin/uv"
    fake_uv.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{uv_log}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_uv.chmod(0o755)

    subprocess.run(
        ["bash", os.path.abspath("scripts/bootstrap_causal_followup_gpu.sh")],
        cwd=project,
        env={**os.environ, "HOME": str(home)},
        check=True,
    )

    assert "-m pip" not in python_log.read_text(encoding="utf-8")
    assert uv_log.read_text(encoding="utf-8").count("pip install --python .venv/bin/python") == 2


def test_gpu_bootstrap_wraps_a_compatible_system_runtime_in_a_venv(tmp_path):
    project = tmp_path / "project"
    home = tmp_path / "home"
    fake_bin = tmp_path / "bin"
    python_log = tmp_path / "python.log"
    uv_log = tmp_path / "uv.log"
    project.mkdir()
    fake_bin.mkdir()
    (home / ".local/bin").mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{python_log}"\ncat >/dev/null\n',
        encoding="utf-8",
    )
    fake_uv = home / ".local/bin/uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >> "{uv_log}"\n'
        'if [[ "$1" == "venv" ]]; then\n'
        '  mkdir -p .venv/bin\n'
        '  cp "$FAKE_SYSTEM_PYTHON" .venv/bin/python\n'
        "fi\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_uv.chmod(0o755)

    subprocess.run(
        ["bash", os.path.abspath("scripts/bootstrap_causal_followup_gpu.sh")],
        cwd=project,
        env={
            **os.environ,
            "HOME": str(home),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_SYSTEM_PYTHON": str(fake_python),
        },
        check=True,
    )

    assert (project / ".venv/bin/python").is_file()
    assert "venv --system-site-packages --python" in uv_log.read_text(encoding="utf-8")
    assert "-m pip" not in python_log.read_text(encoding="utf-8")


def test_decision_binding_operator_is_thin_resumable_and_fetch_verified():
    run = open("scripts/run_decision_binding_gpu.sh", encoding="utf-8").read()
    control = open("scripts/control_decision_binding_gpu.sh", encoding="utf-8").read()
    fetch = open("scripts/fetch_decision_binding_artifacts.sh", encoding="utf-8").read()
    sync = open("scripts/sync_interface_formatting_study_to_gpu.sh", encoding="utf-8").read()
    bootstrap = open("scripts/bootstrap_causal_followup_gpu.sh", encoding="utf-8").read()

    assert "interface_formatting_study.decision_binding_cli" in run
    assert 'COMMON+=(--canary-items "${CANARY_ITEMS:-8}")' in run
    assert '[[ "$MODE" =~ ^(functional|readout|full)$ ]]' in run
    assert 'COMMON+=(--stop-after-readout)' in run
    assert "complete|readout|partial" in fetch
    assert "MODE_FILE" in control and 'echo "$MODE" > "$MODE_FILE"' in control
    assert '[[ "$MODE" =~ ^(functional|readout|full)$ ]]' in control
    assert 'ps -p "$pid" -o command=' in control
    assert "expected_bundle" in control and '"--profile $PROFILE"' in control
    assert "stop)" in control and "kill)" in control and "status)" in control
    assert "results/decision_binding_runs" in fetch
    assert "verify_decision_binding_artifacts.py" in fetch
    assert "run_decision_binding_gpu.sh" in sync
    assert '(3, 11) <= sys.version_info[:2]' in bootstrap
    assert "control_decision_binding_gpu.sh" in sync
    assert 'DECISION_PAIR_PATH="$(dirname "$ACTIVE_DATASET_PATH")/patch_pair_ledger.parquet"' in sync
    assert 'DECISION_MANIFEST_PATH="$(dirname "$ACTIVE_DATASET_PATH")/bundle_manifest.json"' in sync


def test_content_operator_status_surfaces_latest_durable_progress(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    profile = "mistral"
    mode = "functional"
    bundle = "/prepared/mistral"
    (state / f"{profile}.pid").write_text(str(os.getpid()))
    (state / f"{profile}.mode").write_text(mode)
    (state / f"{profile}.args").write_text(bundle)
    progress = {
        "phase": "training_capture",
        "completed": 1,
        "total": 8,
        "elapsed_seconds": 3.5,
        "units_per_second": 0.29,
        "peak_vram_bytes": 1234,
    }
    (state / f"{profile}-{mode}.log").write_text(
        __import__("json").dumps(progress) + "\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        f"echo 'python -m decision_binding_content --profile {profile} {bundle}'\n"
    )
    fake_ps.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            os.path.abspath("scripts/control_decision_binding_content_gpu.sh"),
            "status",
            profile,
        ],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "DECISION_CONTENT_STATE_DIR": str(state),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "running profile=mistral" in result.stdout
    assert '"phase": "training_capture"' in result.stdout
    assert '"completed": 1' in result.stdout
    assert '"peak_vram_bytes": 1234' in result.stdout
