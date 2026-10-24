from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts/run_decision_binding_logit_lens_gpu.sh"
CONTROL = ROOT / "scripts/control_decision_binding_logit_lens_gpu.sh"
FETCH = ROOT / "scripts/fetch_decision_binding_logit_lens_artifacts.sh"
SYNC = ROOT / "scripts/sync_interface_formatting_study_to_gpu.sh"
CHECKLIST = ROOT / "DECISION_BINDING_LOGIT_LENS_EXECUTION_CHECKLIST.md"
RUN_CARD = ROOT / "DECISION_BINDING_LOGIT_LENS_RUN_CARD.md"
NORTH_STAR = ROOT / "SCIENTIFIC_NORTH_STAR.md"


def test_logit_lens_runner_uses_explicit_pinned_profile_and_runtime_configuration():
    script = RUN.read_text(encoding="utf-8")

    assert ".venv/bin/python" in script
    assert "interface_formatting_study.decision_binding_logit_lens_cli" in script
    assert 'PROFILE="$2"' in script
    assert '[[ "$PROFILE" =~ ^(mistral|phi|qwen)$ ]]' in script
    assert '--profile "$PROFILE"' in script
    assert '[[ "$MODE" =~ ^(startup|full)$ ]]' in script
    assert 'BATCH_SIZE:-8' in script
    assert 'MAX_BATCH_TOKENS:-24000' in script
    assert '--capture-chunk-size "4"' in script
    assert 'FULL_CAPTURE_CHUNK_SIZE:-64' in script
    assert '--startup-items "8"' in script
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
        ["bash", str(RUN), "startup", "phi", "/prepared/v4", "/prepared/tokenization"],
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
    assert "--profile phi" in args
    assert "--bundle /prepared/v4" in args
    assert "--token-audit /prepared/tokenization" in args
    assert "--batch-size 3" in args
    assert "--max-batch-tokens 1234" in args
    assert "--capture-chunk-size 4" in args
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
    assert 'HOST_GPU_LOCK_FILE="$STATE_DIR/gpu.lock"' in script
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


def test_logit_lens_control_keeps_live_startup_owned_when_status_requests_full(tmp_path):
    state = tmp_path / "state"
    fake_bin = tmp_path / "bin"
    state.mkdir()
    fake_bin.mkdir()
    (state / "mistral.pid").write_text(str(os.getpid()), encoding="utf-8")
    (state / "mistral.mode").write_text("startup", encoding="utf-8")
    (state / "mistral.args").write_text(
        "/prepared/v4\n/prepared/tokenization\n", encoding="utf-8"
    )
    (state / "mistral-startup.log").write_text("startup alive\n", encoding="utf-8")
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'bash scripts/run_decision_binding_logit_lens_gpu.sh startup "
        "mistral /prepared/v4 /prepared/tokenization'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    result = subprocess.run(
        ["bash", str(CONTROL), "status", "mistral", "full"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "DECISION_LOGIT_LENS_STATE_DIR": str(state),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    assert "running model=mistral mode=startup" in result.stdout
    assert (state / "mistral.pid").read_text(encoding="utf-8") == str(os.getpid())


def test_canonical_checklist_uses_profile_before_mode_and_documented_launch_parses(
    tmp_path,
):
    checklist = CHECKLIST.read_text(encoding="utf-8")
    assert "control_decision_binding_logit_lens_gpu.sh start mistral startup" in checklist
    assert "control_decision_binding_logit_lens_gpu.sh status mistral" in checklist
    assert "control_decision_binding_logit_lens_gpu.sh tail mistral" in checklist
    assert "control_decision_binding_logit_lens_gpu.sh start mistral full" in checklist
    assert "fetch_decision_binding_logit_lens_artifacts.sh \\\n  mistral ubuntu@<host>" in checklist

    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_decision_binding_logit_lens_gpu.sh"
    runner.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(CONTROL),
            "start",
            "mistral",
            "startup",
            "/prepared/v4",
            "/prepared/tokenization",
        ],
        cwd=project,
        env={
            **os.environ,
            "DECISION_LOGIT_LENS_STATE_DIR": str(tmp_path / "state"),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "started model=mistral mode=startup" in result.stdout


def test_governing_docs_authorize_only_the_pinned_cross_model_extension():
    run_card = RUN_CARD.read_text(encoding="utf-8")
    north_star = NORTH_STAR.read_text(encoding="utf-8")
    normalized_run_card = " ".join(run_card.split())
    normalized_north_star = " ".join(north_star.split())

    assert "Mistral-only" not in run_card
    assert "one Mistral logit-lens scorer" not in run_card
    assert "all 32 layers" not in run_card
    assert "Phi and Mistral use 32 transformer blocks (layers 0-31)" in normalized_run_card
    assert "Qwen uses 28 transformer blocks (layers 0-27)" in normalized_run_card
    assert "profile-bound final layer" in normalized_run_card
    assert "final 599" in normalized_run_card
    assert "same scoring, eligibility, parity, and claim boundaries" in normalized_run_card
    assert "one host-wide gpu lock" in normalized_run_card.lower()

    assert "Mistral-only internal diagnostic" not in north_star
    assert "pinned Mistral, Phi, and Qwen" in normalized_north_star
    assert "final 599" in normalized_north_star


def test_single_gpu_host_lock_prevents_phi_and_qwen_running_together(tmp_path):
    project = tmp_path / "project"
    scripts = project / "scripts"
    state = tmp_path / "state"
    marker = tmp_path / "started.txt"
    fake_bin = tmp_path / "bin"
    fake_locks = tmp_path / "fake-locks"
    scripts.mkdir(parents=True)
    fake_bin.mkdir()
    fake_locks.mkdir()
    runner = scripts / "run_decision_binding_logit_lens_gpu.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$2" >> "$LOGIT_LENS_TEST_MARKER"\n'
        "sleep 30\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    fake_flock = fake_bin / "flock"
    fake_flock.write_text(
        "#!/usr/bin/env bash\n"
        'command="$(ps -p "$PPID" -o command=)"\n'
        'case "$command" in\n'
        '  *gpu.lock*) key=gpu ;;\n'
        '  *phi.lock*) key=phi ;;\n'
        '  *qwen.lock*) key=qwen ;;\n'
        '  *) key=unknown ;;\n'
        'esac\n'
        'mkdir "$FAKE_FLOCK_ROOT/$key.held"\n',
        encoding="utf-8",
    )
    fake_flock.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DECISION_LOGIT_LENS_STATE_DIR": str(state),
        "LOGIT_LENS_TEST_MARKER": str(marker),
        "FAKE_FLOCK_ROOT": str(fake_locks),
    }

    try:
        subprocess.run(
            [
                "bash", str(CONTROL), "start", "phi", "startup",
                "/prepared/phi", "/audit/phi",
            ],
            cwd=project,
            env=environment,
            check=True,
        )
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(0.01)
        assert marker.read_text(encoding="utf-8").splitlines() == ["phi"]

        subprocess.run(
            [
                "bash", str(CONTROL), "start", "qwen", "startup",
                "/prepared/qwen", "/audit/qwen",
            ],
            cwd=project,
            env=environment,
            check=True,
        )
        for _ in range(100):
            qwen_log = state / "qwen-startup.log"
            if qwen_log.exists() and "host GPU lock is held" in qwen_log.read_text(
                encoding="utf-8"
            ):
                break
            time.sleep(0.01)
        assert marker.read_text(encoding="utf-8").splitlines() == ["phi"]
        assert "host GPU lock is held" in qwen_log.read_text(encoding="utf-8")
    finally:
        for profile in ("phi", "qwen"):
            pid_path = state / f"{profile}.pid"
            if pid_path.exists():
                try:
                    os.kill(int(pid_path.read_text(encoding="utf-8")), 9)
                except (ProcessLookupError, ValueError):
                    pass


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
    assert 'phi-3.5-mini-instruct' in script
    assert 'qwen2.5-1.5b-instruct' in script


def test_logit_lens_operator_scripts_are_in_the_thin_gpu_sync_whitelist():
    script = SYNC.read_text(encoding="utf-8")

    assert "/scripts/run_decision_binding_logit_lens_gpu.sh" in script
    assert "/scripts/control_decision_binding_logit_lens_gpu.sh" in script
    assert "/scripts/fetch_decision_binding_logit_lens_artifacts.sh" in script
    assert "/analysis/analyze_decision_binding_logit_lens.py" in script
    assert "/DECISION_BINDING_LOGIT_LENS_RUN_CARD.md" in script
    assert "/SCIENTIFIC_NORTH_STAR.md" in script
