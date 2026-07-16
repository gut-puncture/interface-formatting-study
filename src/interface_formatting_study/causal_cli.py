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

from .causal_design import CONTROLLED_FORMATS, build_causal_design_v3
from .causal_option_audit import load_causal_choice_overrides, load_causal_option_annotations
from .causal_runner import run_causal_design, validate_causal_design
from .experiment import prepare_dataset
from .model_loader import load_model_and_tokenizer
from .model_profiles import MODEL_PROFILES, get_model_profile
from .model_runner import ForwardTimer, StopState
from .run_identity import build_semantic_identity, default_semantic_source_paths, sha256_file
from .scoring import single_token_label_ids
from .utils import read_table, read_yaml, write_table_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESIGN = Path(
    "artifacts/causal_followup/v3_source_preserving/design_train_validation.parquet"
)
DESIGN_SCHEMA_VERSION = 5
DESIGN_SCOPE = "source_prompt_counterfactual"


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


def _eligibility_summary(
    frame: pd.DataFrame, design: pd.DataFrame, splits: tuple[str, ...]
) -> dict[str, object]:
    items = frame[frame["split"].astype(str).isin(splits)].drop_duplicates("item_id").copy()
    items["correct_label"] = items["correct_index"].map(dict(enumerate("ABCD")))
    eligible_ids = set(design["item_id"].astype(str))
    items["eligible"] = items["item_id"].astype(str).isin(eligible_ids)

    def summarize(column: str) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for value, group in items.groupby(column, sort=True):
            source_count = int(len(group))
            eligible_count = int(group["eligible"].sum())
            result[str(value)] = {
                "source_items": source_count,
                "eligible_items": eligible_count,
                "eligibility_rate": eligible_count / source_count,
            }
        return result

    return {
        "by_split": summarize("split"),
        "by_correct_label": summarize("correct_label"),
        "by_subject": summarize("subject"),
    }


def cmd_prepare(args) -> None:
    config = read_yaml(args.config)
    frame, _ = prepare_dataset(config)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits or set(splits) - {"train", "validation", "test"}:
        raise ValueError("--splits must contain train, validation, and/or test")
    option_maps = (
        load_causal_option_annotations(args.option_audit)
        if getattr(args, "option_audit", None)
        else None
    )
    choice_overrides = (
        load_causal_choice_overrides(args.choice_audit)
        if getattr(args, "choice_audit", None)
        else None
    )
    design, applicability = build_causal_design_v3(
        frame,
        splits=splits,
        option_maps=option_maps,
        choice_overrides=choice_overrides,
    )
    source_items = int(frame[frame["split"].astype(str).isin(splits)]["item_id"].nunique())
    validate_causal_design(design)
    items = int(design["item_id"].nunique())
    if items != source_items or len(applicability) != items * len(CONTROLLED_FORMATS):
        raise RuntimeError("Causal v3 design did not retain every source item-format block")
    output = Path(args.output)
    write_table_atomic(design, output)
    applicability_path = output.with_name(output.stem + ".applicability.parquet")
    write_table_atomic(applicability, applicability_path)
    def audit_identity(value: str | None) -> dict[str, object] | None:
        if not value:
            return None
        audit_root = Path(value)
        return {
            "path": str(audit_root),
            "manifest_sha256": sha256_file(audit_root / "manifest.json"),
            "label_hashes": {
                path.name: sha256_file(path)
                for path in sorted((audit_root / "labels").glob("*.jsonl"))
            },
        }

    option_audit = audit_identity(getattr(args, "option_audit", None))
    choice_audit = audit_identity(getattr(args, "choice_audit", None))
    _atomic_json(
        {
            "design_schema_version": DESIGN_SCHEMA_VERSION,
            "sha256": sha256_file(output),
            "rows": len(design),
            "source_items": source_items,
            "retained_items": items,
            "applicability": {
                "path_name": applicability_path.name,
                "sha256": sha256_file(applicability_path),
                "rows": len(applicability),
                "position_applicable": int(applicability["position_applicable"].sum()),
                "label_applicable": int(applicability["label_applicable"].sum()),
            },
            "splits": list(splits),
            "formats": list(CONTROLLED_FORMATS),
            "rows_per_item_format": "2 + 3*position_applicable + 3*label_applicable",
            "letter_rows": {
                "controlled_baseline": 1,
                "position_only": 3,
                "label_only": 3,
            },
            "answer_text_rows": 1,
            "option_audit": option_audit,
            "choice_audit": choice_audit,
            "scope": DESIGN_SCOPE,
        },
        output.with_name(output.name + ".manifest.json"),
    )
    print(json.dumps({"output": str(output), "rows": len(design), "items": items}, sort_keys=True))


def _load_design(path: Path) -> pd.DataFrame:
    manifest_path = path.with_name(path.name + ".manifest.json")
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing causal design or manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("design_schema_version") != DESIGN_SCHEMA_VERSION
        or manifest.get("scope") != DESIGN_SCOPE
    ):
        raise RuntimeError("Causal run requires a schema-v5 source-preserving design")
    if sha256_file(path) != manifest.get("sha256"):
        raise RuntimeError("Causal design checksum mismatch")
    applicability_meta = manifest.get("applicability")
    expected_applicability_name = path.stem + ".applicability.parquet"
    if not isinstance(applicability_meta, dict) or applicability_meta.get("path_name") != expected_applicability_name:
        raise RuntimeError("Causal design applicability ledger metadata is missing or invalid")
    applicability_path = path.with_name(expected_applicability_name)
    if not applicability_path.exists():
        raise RuntimeError("Causal design applicability ledger is missing")
    if sha256_file(applicability_path) != applicability_meta.get("sha256"):
        raise RuntimeError("Causal design applicability ledger checksum mismatch")
    applicability = read_table(applicability_path)
    if len(applicability) != int(applicability_meta.get("rows", -1)):
        raise RuntimeError("Causal design applicability ledger row count mismatch")
    design = read_table(path)
    if len(design) != int(manifest.get("rows", -1)):
        raise RuntimeError("Causal design row count does not match its manifest")
    validate_causal_design(design)
    ledger_keys = list(zip(applicability["item_id"], applicability["wrapper_name"], strict=True))
    if len(ledger_keys) != len(set(ledger_keys)):
        raise RuntimeError("Causal applicability ledger requires unique item-wrapper keys")
    design_keys = set(zip(design["item_id"], design["wrapper_name"], strict=True))
    if set(ledger_keys) != design_keys:
        raise RuntimeError("Causal applicability ledger keys do not match the design")
    for item_id, group in design.groupby("item_id", sort=False):
        if set(group["wrapper_name"]) != set(CONTROLLED_FORMATS):
            raise RuntimeError(f"Causal design is missing an expected format for item {item_id}")
    observed = design.groupby(["item_id", "wrapper_name", "manipulation"]).size()
    for row in applicability.to_dict("records"):
        key = (row["item_id"], row["wrapper_name"])
        if observed.get((*key, "position_only"), 0) != (3 if row["position_applicable"] else 0):
            raise RuntimeError(f"Position applicability disagrees with design rows: {key}")
        if observed.get((*key, "label_only"), 0) != (3 if row["label_applicable"] else 0):
            raise RuntimeError(f"Label applicability disagrees with design rows: {key}")
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
    design_manifest = json.loads(
        design_path.with_name(design_path.name + ".manifest.json").read_text(encoding="utf-8")
    )
    applicability_meta = design_manifest["applicability"]
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
        "study": "causal_followup_v3_source_preserving",
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
        "design_applicability_sha256": applicability_meta["sha256"],
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
            "applicability_path_name": applicability_meta["path_name"],
            "applicability_sha256": applicability_meta["sha256"],
            "applicability_rows": applicability_meta["rows"],
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
    prepare.add_argument("--option-audit")
    prepare.add_argument("--choice-audit")
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
