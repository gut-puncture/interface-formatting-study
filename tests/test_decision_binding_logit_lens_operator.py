from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/run_decision_binding_logit_lens_gpu.sh"
CONTROL = ROOT / "scripts/control_decision_binding_logit_lens_gpu.sh"
FETCH = ROOT / "scripts/fetch_decision_binding_logit_lens_artifacts.sh"
SYNC = ROOT / "scripts/sync_interface_formatting_study_to_gpu.sh"


def test_logit_lens_runner_is_mistral_only_and_uses_runtime_configuration():
    script = RUN.read_text(encoding="utf-8")

    assert ".venv/bin/python" in script
    assert "interface_formatting_study.decision_binding_logit_lens_cli" in script
    assert '--profile "mistral"' in script
    assert "qwen" not in script.lower()
    assert "phi" not in script.lower()
    assert '[[ "$MODE" =~ ^(startup|full)$ ]]' in script
    assert 'BATCH_SIZE:-8' in script
    assert 'MAX_BATCH_TOKENS:-24000' in script
    assert 'STARTUP_CAPTURE_CHUNK_SIZE:-4' in script
    assert 'FULL_CAPTURE_CHUNK_SIZE:-64' in script
    assert 'STARTUP_ITEMS:-8' in script
    assert "--max-chunks-this-invocation" in script


def test_logit_lens_runner_passes_frozen_arguments_through_real_entrypoint(tmp_path):
    project = tmp_path / "project"
    fake_python = project / ".venv/bin/python"
    log = tmp_path / "args.log"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" > "{log}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    subprocess.run(
        ["bash", str(RUN), "startup", "/prepared/v4", "/prepared/tokenization"],
        cwd=project,
        env={
            **os.environ,
            "BATCH_SIZE": "3",
            "MAX_BATCH_TOKENS": "1234",
            "CAPTURE_CHUNK_SIZE": "2",
            "STARTUP_ITEMS": "8",
            "MAX_CHUNKS_THIS_INVOCATION": "1",
        },
        check=True,
    )

    args = log.read_text(encoding="utf-8")
    assert "-m interface_formatting_study.decision_binding_logit_lens_cli run-model" in args
    assert "--profile mistral" in args
    assert "--bundle /prepared/v4" in args
    assert "--token-audit /prepared/tokenization" in args
    assert "--batch-size 3" in args
    assert "--max-batch-tokens 1234" in args
    assert "--capture-chunk-size 2" in args
    assert "--startup-items 8" in args
    assert "--max-chunks-this-invocation 1" in args
    assert "--local-files-only" in args


def test_logit_lens_control_is_owned_resumable_and_surfaces_progress(tmp_path):
    script = CONTROL.read_text(encoding="utf-8")
    assert "DECISION_LOGIT_LENS_STATE_DIR" in script
    assert "MODE_FILE" in script and 'echo "$MODE" > "$MODE_FILE"' in script
    assert "ARGS_FILE" in script
    assert '[[ "$MODE" =~ ^(startup|full)$ ]]' in script
    assert 'ps -p "$pid" -o command=' in script
    assert "decision_binding_logit_lens" in script
    assert "run_decision_binding_logit_lens_gpu.sh" in script
    assert "flock -n" in script
    assert "stop)" in script and "status)" in script and "tail)" in script

    state = tmp_path / "state"
    fake_bin = tmp_path / "bin"
    state.mkdir()
    fake_bin.mkdir()
    (state / "mistral.pid").write_text(str(os.getpid()), encoding="utf-8")
    (state / "mistral.mode").write_text("startup", encoding="utf-8")
    (state / "mistral.args").write_text(
        "/prepared/v4\n/prepared/tokenization\n",
        encoding="utf-8",
    )
    (state / "mistral-startup.log").write_text(
        '{"phase":"scoring","completed":1,"total":8,"peak_vram_bytes":1234}\n',
        encoding="utf-8",
    )
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'python -m interface_formatting_study.decision_binding_logit_lens_cli "
        "run-model --profile mistral --bundle /prepared/v4 "
        "--token-audit /prepared/tokenization'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    result = subprocess.run(
        ["bash", str(CONTROL), "status"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "DECISION_LOGIT_LENS_STATE_DIR": str(state),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "running model=mistral" in result.stdout
    assert '"phase":"scoring"' in result.stdout
    assert '"completed":1' in result.stdout
    assert '"peak_vram_bytes":1234' in result.stdout


def test_logit_lens_fetch_supports_partial_startup_and_complete_verification():
    script = FETCH.read_text(encoding="utf-8")

    assert "partial|startup|complete" in script
    assert "results/decision_binding_logit_lens_runs" in script
    assert "interface_formatting_study.decision_binding_logit_lens_cli" in script
    assert 'verify' in script
    assert '--run-root "$DEST"' in script
    assert '--run-id "$RUN_ID"' in script
    assert '--mode "$MODE"' in script
    assert ".venv/bin/python" in script
    assert 'missing executable Python runtime' in script


def test_logit_lens_operator_scripts_are_in_the_thin_gpu_sync_whitelist():
    script = SYNC.read_text(encoding="utf-8")

    assert "/scripts/run_decision_binding_logit_lens_gpu.sh" in script
    assert "/scripts/control_decision_binding_logit_lens_gpu.sh" in script
    assert "/scripts/fetch_decision_binding_logit_lens_artifacts.sh" in script
    assert "/analysis/analyze_decision_binding_logit_lens.py" in script
    assert "/DECISION_BINDING_LOGIT_LENS_RUN_CARD.md" in script
    assert "/SCIENTIFIC_NORTH_STAR.md" in script
