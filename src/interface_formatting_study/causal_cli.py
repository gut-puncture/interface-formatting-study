from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .causal_design import CONTROLLED_FORMATS, build_causal_design
from .causal_runner import run_causal_design, validate_causal_design
from .experiment import prepare_dataset
from .model_loader import load_model_and_tokenizer
from .model_profiles import MODEL_PROFILES, get_model_profile
from .model_runner import ForwardTimer, StopState
from .run_identity import build_semantic_identity, default_semantic_source_paths, sha256_file
from .scoring import single_token_label_ids
from .utils import read_table, read_yaml, write_table_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESIGN = Path("artifacts/causal_followup/v1/design_train_validation.parquet")


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_identity(identity, path: Path) -> None:
    payload = identity.as_dict()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(f"Semantic identity mismatch at {path}")
        return
    _atomic_json(payload, path)


def cmd_prepare(args) -> None:
    config = read_yaml(args.config)
    frame, _ = prepare_dataset(config)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits or set(splits) - {"train", "validation", "test"}:
        raise ValueError("--splits must contain train, validation, and/or test")
    design = build_causal_design(frame, splits=splits)
    validate_causal_design(design)
    items = int(design["item_id"].nunique())
    if len(design) != items * len(CONTROLLED_FORMATS) * 8:
        raise RuntimeError("Causal design row count violates the 9 formats x 8 rows contract")
    output = Path(args.output)
    write_table_atomic(design, output)
    _atomic_json(
        {
            "design_schema_version": 3,
            "sha256": sha256_file(output),
            "rows": len(design),
            "items": items,
            "splits": list(splits),
            "formats": list(CONTROLLED_FORMATS),
            "rows_per_item_format": 8,
            "letter_rows": {
                "controlled_baseline": 1,
                "position_only": 3,
                "label_only": 3,
            },
            "answer_text_rows": 1,
            "scope": "controlled canonical templates; original prompts remain in the observational study",
        },
        output.with_name(output.name + ".manifest.json"),
    )
    print(json.dumps({"output": str(output), "rows": len(design), "items": items}, sort_keys=True))


def _load_design(path: Path) -> pd.DataFrame:
    manifest_path = path.with_name(path.name + ".manifest.json")
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing causal design or manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError("Causal design checksum mismatch")
    design = read_table(path)
    if len(design) != int(manifest.get("rows", -1)):
        raise RuntimeError("Causal design row count does not match its manifest")
    validate_causal_design(design)
    return design


def _canary_subset(design: pd.DataFrame, item_count: int) -> pd.DataFrame:
    if item_count < 1:
        raise ValueError("--canary-items must be positive")
    item_lengths = (
        design.assign(prompt_chars=design["prompt"].astype(str).str.len())
        .groupby("item_id", as_index=False)["prompt_chars"]
        .max()
        .sort_values(["prompt_chars", "item_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    count = min(item_count, len(item_lengths))
    positions = np.linspace(0, len(item_lengths) - 1, num=count, dtype=int)
    item_ids = item_lengths.iloc[positions]["item_id"].astype(str).tolist()
    return design[design["item_id"].astype(str).isin(item_ids)].copy()


def _dependency_versions() -> dict[str, str]:
    versions = {}
    for package in (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface-hub",
        "pandas",
        "pyarrow",
        "numpy",
        "PyYAML",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _selected_work_sha(frame: pd.DataFrame) -> str:
    keys = "\n".join(sorted(frame["work_key"].astype(str))) + "\n"
    import hashlib

    return hashlib.sha256(keys.encode("utf-8")).hexdigest()


def _validate_run_args(args) -> None:
    for name in ("batch_size", "max_batch_tokens", "checkpoint_size"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.canary and int(args.canary_items) <= 0:
        raise ValueError("--canary-items must be positive")


def _assert_gpu_contract(model, device, *, allow_cpu: bool) -> str:
    dtype = str(next(model.parameters()).dtype)
    if allow_cpu:
        return dtype
    if not torch.cuda.is_available() or getattr(device, "type", str(device)) != "cuda":
        raise RuntimeError("Causal GPU runs require CUDA; refusing CPU execution")
    if next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError(f"Causal GPU runs require BF16, got {dtype}")
    return dtype


def cmd_run(args) -> None:
    _validate_run_args(args)
    design_path = Path(args.design)
    design = _load_design(design_path)
    profile = get_model_profile(args.profile)
    source = design
    if args.canary:
        if not args.canary_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("--canary-name may contain only letters, numbers, dashes, and underscores")
        source = _canary_subset(design, args.canary_items)

    total_started = time.monotonic()
    model_load_started = time.monotonic()
    model, tokenizer, device = load_model_and_tokenizer(
        profile.model_id,
        revision=profile.revision,
        tokenizer_name=profile.tokenizer_id,
        tokenizer_revision=profile.tokenizer_revision,
        local_files_only=args.local_files_only,
        attn_implementation=profile.attention_backend,
        expected_layers=profile.expected_layers,
        trust_remote_code=profile.trust_remote_code,
    )
    model_load_seconds = time.monotonic() - model_load_started
    actual_dtype = _assert_gpu_contract(model, device, allow_cpu=bool(getattr(args, "allow_cpu", False)))
    dependencies = _dependency_versions()
    scientific_config = {
        "study": "causal_followup_v2_controlled",
        "design_splits": sorted(design["split"].astype(str).unique()),
        "formats": list(CONTROLLED_FORMATS),
        "arms": {
            "controlled_baseline": 1,
            "position_only": 3,
            "label_only": 3,
            "answer_text": 1,
        },
        "scoring": {
            "letter_primary": "content_free_calibrated_conditional_log_likelihood",
            "letter_secondary": "raw_conditional_log_likelihood",
            "answer_text_primary": "deterministic_generation_exact_match",
            "answer_text_secondary": ["total_log_likelihood", "mean_token_log_likelihood"],
            "batch_size": int(args.batch_size),
            "max_batch_tokens": args.max_batch_tokens,
        },
        "selected_work_sha256": _selected_work_sha(source),
        "runtime_contract": {
            "python": sys.version.split()[0],
            "torch": dependencies["torch"],
            "transformers": dependencies["transformers"],
            "numpy": dependencies["numpy"],
            "pandas": dependencies["pandas"],
            "pyarrow": dependencies["pyarrow"],
            "tokenizers": dependencies["tokenizers"],
            "safetensors": dependencies["safetensors"],
            "huggingface_hub": dependencies["huggingface-hub"],
            "cuda": torch.version.cuda,
            "dtype": actual_dtype,
            "attention_backend": profile.attention_backend,
            "gpu_lock_sha256": sha256_file(PROJECT_ROOT / "requirements-gpu.lock"),
        },
    }
    identity = build_semantic_identity(
        profile,
        config=scientific_config,
        dataset_path=design_path,
        source_paths=default_semantic_source_paths(PROJECT_ROOT),
    )
    run_root = Path(args.output_base) / profile.slug / identity.semantic_run_id
    if args.canary:
        run_root = run_root / "canaries" / args.canary_name
    _write_identity(identity, run_root / "semantic_identity.json")

    base_manifest = {
        "run_schema_version": 2,
        "status": "starting",
        "canary": bool(args.canary),
        "canary_name": args.canary_name if args.canary else None,
        "model": {
            "id": profile.model_id,
            "revision": profile.revision,
            "slug": profile.slug,
            "expected_layers": profile.expected_layers,
        },
        "semantic_identity": identity.as_dict(),
        "design": {
            "path_name": design_path.name,
            "sha256": sha256_file(design_path),
            "source_rows": len(design),
            "run_rows": len(source),
            "selected_work_sha256": _selected_work_sha(source),
        },
        "runtime": {
            "device": str(device),
            "dtype": actual_dtype,
            "model_load_seconds": model_load_seconds,
            "dependencies": dependencies,
        },
    }
    _atomic_json(base_manifest, run_root / "run_manifest.json")

    stop_state = StopState()
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, stop_state.request)

    timer = None
    try:
        label_ids = single_token_label_ids(tokenizer)
        timer = ForwardTimer(model, enabled=args.profile_timings)
        scoring_started = time.monotonic()
        with timer:
            merged = run_causal_design(
                source,
                model,
                tokenizer,
                identity,
                run_root,
                batch_size=args.batch_size,
                max_batch_tokens=args.max_batch_tokens,
                checkpoint_size=args.checkpoint_size,
                device=device,
                should_stop=lambda: stop_state.requested,
            )
        scoring_seconds = time.monotonic() - scoring_started
        complete = len(merged) == len(source)
        manifest = {
            **base_manifest,
            "status": "complete" if complete else "interrupted",
            "design": {
                **base_manifest["design"],
                "completed_rows": len(merged),
            },
            "runtime": {
                **base_manifest["runtime"],
                "single_token_label_fast_path": label_ids is not None,
                "label_token_ids": label_ids,
                "total_wall_seconds": time.monotonic() - total_started,
                "scoring_seconds": scoring_seconds,
                "forward_seconds": None if timer is None else timer.seconds,
                "forward_calls": None if timer is None else timer.calls,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                "batch_size": args.batch_size,
                "max_batch_tokens": args.max_batch_tokens,
                "checkpoint_size": args.checkpoint_size,
            },
            "stop_signal": stop_state.signal_name,
        }
        _atomic_json(manifest, run_root / "run_manifest.json")
        print(json.dumps({"run_root": str(run_root), "status": manifest["status"]}, sort_keys=True))
    except BaseException as error:
        failed = {
            **base_manifest,
            "status": "failed",
            "failure": {"type": type(error).__name__, "message": str(error)},
            "stop_signal": stop_state.signal_name,
        }
        _atomic_json(failed, run_root / "run_manifest.json")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface-formatting-causal")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", default="configs/default.yaml")
    prepare.add_argument("--splits", default="train,validation")
    prepare.add_argument("--output", default=str(DEFAULT_DESIGN))
    prepare.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run")
    run.add_argument("--profile", choices=sorted(MODEL_PROFILES), required=True)
    run.add_argument("--design", default=str(DEFAULT_DESIGN))
    run.add_argument("--output-base", default="results/causal_runs")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-batch-tokens", type=int, default=40000)
    run.add_argument("--checkpoint-size", type=int, default=256)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--profile-timings", action="store_true")
    run.add_argument("--canary", action="store_true")
    run.add_argument("--canary-name", default="functional")
    run.add_argument("--canary-items", type=int, default=8)
    run.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
