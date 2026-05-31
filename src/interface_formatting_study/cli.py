from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

from .behavioral_eval import evaluate_behavioral
from .calibration import SAME_WRAPPER_REDACTION
from .conflicts import construct_conflict_pairs, item_outcome_summary
from .component_patching import FOCUSED_CONTROL_CONDITIONS, run_focused_patching_controls, summarize_focused_controls
from .data_loading import dataset_audit, dataset_audit_from_frame, load_normalized
from .deliverables import REQUIRED_FINAL_ARTIFACTS, prepare_deliverables
from .diagnostics import (
    run_attention_diagnostics,
    run_vanilla_convergence,
    summarize_attention_diagnostics,
    summarize_vanilla_convergence,
)
from .experiment import (
    evaluate_access_vector_and_controls,
    evaluate_clean_removal,
    evaluate_content_free_control,
    evaluate_item_outcome_subset_control,
    load_access_vector,
    prepare_dataset,
    save_access_vector,
    save_optional_shuffled_vector,
    save_split_manifest,
    summarize_intervention_results,
    train_access_vector,
    tune_alpha,
    write_conflict_artifacts,
)
from .model_loader import load_model_and_tokenizer
from .patching import run_validation_patching_sweep, select_best_location
from .plots import conflict_histogram, wrapper_accuracy_heatmap
from .splits import assign_splits_by_item
from .tables import primary_behavioral_rows, write_standard_tables
from .utils import LABELS, read_json, read_table, read_yaml, write_json, write_table
from .wrapper_audit import attach_wrapper_categories, audit_wrappers

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(args) -> dict:
    cfg = read_yaml(args.config)
    labels = cfg.get("labels")
    if labels is not None and tuple(str(label) for label in labels) != tuple(LABELS):
        raise ValueError(f"Config labels must match scorer labels {list(LABELS)}, got {labels!r}")
    policy = cfg.get("scoring", {}).get("answer_variant_policy")
    if policy not in (None, "canonical_letter_only"):
        raise ValueError(f"Unsupported answer_variant_policy {policy!r}; scorer uses canonical_letter_only")
    return cfg


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package_name in ["numpy", "pandas", "pyarrow", "PyYAML", "torch", "transformers"]:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions


def _project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _source_hashes(config_path: str | Path) -> dict[str, str]:
    paths = sorted((PROJECT_ROOT / "src/interface_formatting_study").glob("*.py")) + [Path(config_path), PROJECT_ROOT / "pyproject.toml"]
    return {_project_relative(path): _sha256_file(path) for path in paths if path.exists()}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _existing_artifact_hashes() -> dict[str, str]:
    return {
        path: _sha256_file(PROJECT_ROOT / path)
        for path in REQUIRED_FINAL_ARTIFACTS
        if path != "results/metadata/experiment_manifest.json" and (PROJECT_ROOT / path).exists()
    }


def _is_full_behavioral_table(scored, cfg: dict) -> bool:
    dataset_audit_path = Path(cfg["outputs"]["metadata_dir"]) / "dataset_audit.json"
    if not dataset_audit_path.exists():
        return False
    return len(scored) == int(read_json(dataset_audit_path)["num_rows"])


def _assert_primary_conflicts(pairs, *, command_name: str) -> None:
    if pairs.empty:
        raise ValueError(f"{command_name} requires non-empty primary pure-interface conflict pairs")
    required = {
        "wrapper_category_subset",
        "clean_content_free_calibration_kind",
        "corrupt_content_free_calibration_kind",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"{command_name} conflict file is missing provenance columns: {sorted(missing)}")
    subsets = set(pairs["wrapper_category_subset"].dropna().astype(str))
    if subsets != {"pure_interface"}:
        raise ValueError(f"{command_name} requires pure_interface conflict pairs, got {sorted(subsets)}")
    for column in ("clean_content_free_calibration_kind", "corrupt_content_free_calibration_kind"):
        kinds = set(pairs[column].dropna().astype(str))
        if kinds != {SAME_WRAPPER_REDACTION}:
            raise ValueError(f"{command_name} requires same-wrapper calibrated conflict pairs, got {column}={sorted(kinds)}")


def _assert_behavioral_has_primary_provenance(scored, *, command_name: str) -> None:
    required = {"wrapper_category", "content_free_calibration_kind"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"{command_name} behavioral file is missing provenance columns: {sorted(missing)}")
    primary = scored[scored["wrapper_category"] == "pure_interface"]
    if primary.empty:
        raise ValueError(f"{command_name} behavioral file has no verified pure-interface rows")
    same_wrapper_primary = primary[primary["content_free_calibration_kind"] == SAME_WRAPPER_REDACTION]
    if same_wrapper_primary.empty:
        raise ValueError(f"{command_name} behavioral file has no same-wrapper calibrated pure-interface rows")


def cmd_audit_dataset(args) -> None:
    cfg = _config(args)
    df, audit_df = prepare_dataset(cfg)
    source_audit = dataset_audit(cfg["dataset_path"], max_examples_per_wrapper=0)
    out = Path(cfg["outputs"]["metadata_dir"]) / "dataset_audit.json"
    audit = dataset_audit_from_frame(
        df,
        dataset_path=cfg["dataset_path"],
        wrapper_policy=str(cfg.get("experiment", {}).get("wrapper_policy", "verified_pure_interface_only")),
        source_num_rows=int(source_audit["num_rows"]),
    )
    write_json(audit, out)
    write_table(audit_df, Path(cfg["outputs"]["metadata_dir"]) / "wrapper_audit.csv")
    save_split_manifest(df, Path(cfg["outputs"]["metadata_dir"]) / "split_manifest.json")
    print(f"wrote {out} ({audit['num_rows']} active rows, {audit['num_items']} items, {audit['num_wrappers']} wrappers)")


def cmd_wrapper_audit(args) -> None:
    cfg = _config(args)
    df, audit = prepare_dataset(cfg)
    out = Path(cfg["outputs"]["metadata_dir"]) / "wrapper_audit.csv"
    write_table(audit, out)
    save_split_manifest(df, Path(cfg["outputs"]["metadata_dir"]) / "split_manifest.json")
    print(f"wrote {out}")


def cmd_behavioral_smoke(args) -> None:
    cfg = _config(args)
    df, audit = prepare_dataset(cfg)
    model_name = args.model or cfg["model_name"]
    model, tokenizer, device = load_model_and_tokenizer(model_name, local_files_only=args.local_files_only)
    scoring_cfg = cfg.get("scoring", {})
    scored = evaluate_behavioral(
        df,
        model,
        tokenizer,
        batch_size=int(scoring_cfg["batch_size"]),
        sequence_batch_size=int(scoring_cfg.get("sequence_batch_size", scoring_cfg["batch_size"])),
        checkpoint_size=int(scoring_cfg.get("checkpoint_size", 256)),
        max_batch_tokens=scoring_cfg.get("max_batch_tokens"),
        limit=args.limit,
        device=device,
    )
    out = Path("results/smoke/raw") / "behavioral_smoke.parquet"
    write_table(scored, out)
    print(f"wrote {out}")


def cmd_behavioral(args) -> None:
    cfg = _config(args)
    df, _audit = prepare_dataset(cfg)
    model_name = args.model or cfg["model_name"]
    model, tokenizer, device = load_model_and_tokenizer(model_name, local_files_only=args.local_files_only)
    scoring_cfg = cfg.get("scoring", {})
    checkpoint_dir = None
    if args.limit is None:
        checkpoint_dir = Path(cfg["outputs"]["raw_dir"]) / "behavioral_shards"
    scored = evaluate_behavioral(
        df,
        model,
        tokenizer,
        batch_size=int(scoring_cfg["batch_size"]),
        sequence_batch_size=int(scoring_cfg.get("sequence_batch_size", scoring_cfg["batch_size"])),
        checkpoint_size=int(scoring_cfg.get("checkpoint_size", 256)),
        max_batch_tokens=scoring_cfg.get("max_batch_tokens"),
        checkpoint_dir=checkpoint_dir,
        resume=bool(args.resume and args.limit is None),
        limit=args.limit,
        device=device,
    )
    if args.limit is None:
        out = Path(cfg["outputs"]["raw_dir"]) / "behavioral_scores.parquet"
    else:
        out = Path("results/smoke/raw") / "behavioral_limited.parquet"
    write_table(scored, out)
    if args.limit is None:
        write_conflict_artifacts(scored, cfg["outputs"]["processed_dir"])
    print(f"wrote {out}")


def cmd_conflicts(args) -> None:
    cfg = _config(args)
    scored = read_table(args.behavioral)
    output_dir = Path(cfg["outputs"]["processed_dir"])
    if Path(args.behavioral).name != "behavioral_scores.parquet" or not _is_full_behavioral_table(scored, cfg):
        output_dir = Path("results/smoke/processed")
    artifacts = write_conflict_artifacts(scored, output_dir)
    print("wrote " + ", ".join(str(path) for path in artifacts.values()))


def _load_model_for_experiment(args, cfg):
    model_name = args.model or cfg["model_name"]
    return load_model_and_tokenizer(model_name, local_files_only=args.local_files_only)


def _csv_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if not value.strip():
        raise ValueError("layer list must not be empty")
    return [int(part.strip()) for part in value.split(",")]


def _csv_strings(value: str | None, default: list[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(str(item) for item in default)
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("anchor/condition list must not be empty")
    return values


def _focused_cfg(cfg: dict) -> dict:
    return cfg.get(
        "focused_mechanistic",
        {
            "layers": [2, 4, 6, 8, 10, 12, 14, 16],
            "anchors": ["options_end", "all_option_ends"],
            "diagnostic_cap": 300,
            "patching_cap": 300,
        },
    )


def cmd_patching_sweep(args) -> None:
    cfg = _config(args)
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="patching-sweep")
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    activation_cfg = cfg.get("activation", {})
    if args.layers is None:
        configured_layers = activation_cfg.get("layers")
        layers = [int(x) for x in configured_layers] if configured_layers else None
    elif args.layers == "all":
        layers = None
    else:
        layers = [int(x) for x in args.layers.split(",")]
    cap = args.cap if args.cap is not None else int(activation_cfg.get("validation_conflict_cap", 300))
    anchors = tuple(args.anchors.split(",")) if args.anchors else tuple(activation_cfg.get("anchors", ["question_end", "options_end"]))
    results = run_validation_patching_sweep(
        model,
        tokenizer,
        pairs,
        layers=layers,
        anchors=anchors,
        cap=cap,
        seed=int(cfg.get("seed", 1729)),
        batch_size=int(cfg["scoring"]["batch_size"]),
        device=device,
    )
    out = Path(cfg["outputs"]["processed_dir"]) / "patching_results.parquet"
    selected = select_best_location(results, allow_leaky_anchors=args.allow_leaky_anchors)
    write_table(results, out)
    write_json(selected, Path(cfg["outputs"]["metadata_dir"]) / "selected_location.json")
    print(f"wrote {out} and selected layer={selected['selected_layer']} anchor={selected['selected_anchor']}")


def cmd_attention_diagnostics(args) -> None:
    cfg = _config(args)
    focused = _focused_cfg(cfg)
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="attention-diagnostics")
    if args.behavioral and Path(args.behavioral).exists():
        scored = read_table(args.behavioral)
        _assert_behavioral_has_primary_provenance(scored, command_name="attention-diagnostics")
    anchors = _csv_strings(args.anchors, list(focused.get("anchors", ["options_end", "all_option_ends"])))
    cap = args.cap if args.cap is not None else int(focused.get("diagnostic_cap", 300))
    model_name = args.model or cfg["model_name"]
    attn_implementation = None if args.attn_implementation == "default" else args.attn_implementation
    model, tokenizer, device = load_model_and_tokenizer(
        model_name,
        local_files_only=args.local_files_only,
        attn_implementation=attn_implementation,
    )
    results = run_attention_diagnostics(
        model,
        tokenizer,
        pairs,
        split=args.split,
        cap=cap,
        anchors=anchors,
        seed=int(cfg.get("seed", 1729)),
        device=device,
    )
    out = Path(cfg["outputs"]["processed_dir"]) / "attention_diagnostics.parquet"
    table_out = Path(cfg["outputs"]["tables_dir"]) / "table_attention_diagnostics.csv"
    write_table(results, out)
    write_table(summarize_attention_diagnostics(results), table_out)
    print(f"wrote {out} and {table_out}")


def cmd_vanilla_convergence(args) -> None:
    cfg = _config(args)
    focused = _focused_cfg(cfg)
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="vanilla-convergence")
    layers = _csv_ints(args.layers, [int(layer) for layer in focused.get("layers", [2, 4, 6, 8, 10, 12, 14, 16])])
    anchors = _csv_strings(args.anchors, list(focused.get("anchors", ["options_end", "all_option_ends"])))
    cap = args.cap if args.cap is not None else int(focused.get("diagnostic_cap", 300))
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    results = run_vanilla_convergence(
        model,
        tokenizer,
        pairs,
        layers=layers,
        split=args.split,
        cap=cap,
        anchors=anchors,
        seed=int(cfg.get("seed", 1729)),
        device=device,
    )
    out = Path(cfg["outputs"]["processed_dir"]) / "vanilla_convergence.parquet"
    table_out = Path(cfg["outputs"]["tables_dir"]) / "table_vanilla_convergence.csv"
    write_table(results, out)
    write_table(summarize_vanilla_convergence(results), table_out)
    print(f"wrote {out} and {table_out}")


def cmd_focused_patching_controls(args) -> None:
    cfg = _config(args)
    focused = _focused_cfg(cfg)
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="focused-patching-controls")
    layers = _csv_ints(args.layers, [int(layer) for layer in focused.get("layers", [2, 4, 6, 8, 10, 12, 14, 16])])
    anchors = _csv_strings(args.anchors, list(focused.get("anchors", ["options_end", "all_option_ends"])))
    conditions = _csv_strings(args.conditions, list(focused.get("conditions", FOCUSED_CONTROL_CONDITIONS)))
    unknown_conditions = set(conditions) - set(FOCUSED_CONTROL_CONDITIONS)
    if unknown_conditions:
        raise ValueError(f"unknown focused control conditions: {sorted(unknown_conditions)}")
    cap = args.cap if args.cap is not None else int(focused.get("patching_cap", 300))
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    results = run_focused_patching_controls(
        model,
        tokenizer,
        pairs,
        layers=layers,
        split=args.split,
        cap=cap,
        anchors=anchors,
        conditions=conditions,
        seed=int(cfg.get("seed", 1729)),
        device=device,
    )
    out = Path(cfg["outputs"]["processed_dir"]) / "focused_patching_controls.parquet"
    table_out = Path(cfg["outputs"]["tables_dir"]) / "table_focused_controls.csv"
    write_table(results, out)
    write_table(summarize_focused_controls(results), table_out)
    print(f"wrote {out} and {table_out}")


def cmd_train_vector(args) -> None:
    cfg = _config(args)
    if args.limit is not None:
        raise ValueError("train-vector writes final artifacts and does not accept --limit")
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="train-vector")
    selected = read_yaml(args.selection) if str(args.selection).endswith((".yaml", ".yml")) else __import__("json").loads(Path(args.selection).read_text())
    model_name = args.model or cfg["model_name"]
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    payload = train_access_vector(
        model,
        tokenizer,
        pairs,
        layer=int(selected["selected_layer"]),
        anchor=str(selected["selected_anchor"]),
        balance_by=args.balance_by,
        limit=args.limit,
        device=device,
    )
    payload["metadata"].update(
        {
            "model_name": model_name,
            "dataset_sha256": _sha256_file(cfg["dataset_path"]),
            "conflict_pairs_sha256": _sha256_file(args.conflicts),
            "selected_location_sha256": _sha256_file(args.selection),
            "vector_tensor_sha256": _tensor_sha256(payload["vector"]),
        }
    )
    out = Path(cfg["outputs"]["processed_dir"]) / "interface_formatting_study_vector.pt"
    save_access_vector(payload, out)
    shuffled_out = Path(cfg["outputs"]["processed_dir"]) / "shuffled_pair_vector.pt"
    save_optional_shuffled_vector(payload, shuffled_out)
    write_json(payload["metadata"], Path(cfg["outputs"]["metadata_dir"]) / "interface_formatting_study_vector_metadata.json")
    print(f"wrote {out}")


def cmd_tune_alpha(args) -> None:
    cfg = _config(args)
    if args.limit is not None:
        raise ValueError("tune-alpha writes final artifacts and does not accept --limit")
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="tune-alpha")
    selected = __import__("json").loads(Path(args.selection).read_text())
    vector, metadata = load_access_vector(args.vector)
    if int(metadata.get("layer", -1)) != int(selected["selected_layer"]) or metadata.get("anchor") != selected["selected_anchor"]:
        raise ValueError("Loaded access vector metadata does not match selected layer/anchor")
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    result = tune_alpha(
        model,
        tokenizer,
        pairs,
        vector=vector,
        layer=int(selected["selected_layer"]),
        anchor=str(selected["selected_anchor"]),
        alpha_grid=cfg["activation"]["alpha_grid"],
        batch_size=int(cfg["scoring"]["batch_size"]),
        limit=args.limit,
        device=device,
    )
    out = Path(cfg["outputs"]["metadata_dir"]) / "selected_alpha.json"
    serializable = {k: v for k, v in result.items() if k != "details"}
    serializable.update(
        {
            "model_name": args.model or cfg["model_name"],
            "dataset_sha256": _sha256_file(cfg["dataset_path"]),
            "conflict_pairs_sha256": _sha256_file(args.conflicts),
            "selected_location_sha256": _sha256_file(args.selection),
            "vector_artifact_sha256": _sha256_file(args.vector),
            "vector_tensor_sha256": metadata.get("vector_tensor_sha256") or _tensor_sha256(vector),
            "vector_metadata": metadata,
        }
    )
    write_json(serializable, out)
    write_table(result["details"], Path(cfg["outputs"]["processed_dir"]) / "alpha_tuning_details.parquet")
    print(f"wrote {out} selected_alpha={result['selected_alpha']}")


def cmd_eval_vector(args) -> None:
    cfg = _config(args)
    if args.limit is not None:
        raise ValueError("eval-vector writes final artifacts and does not accept --limit")
    pairs = read_table(args.conflicts)
    _assert_primary_conflicts(pairs, command_name="eval-vector")
    selected = __import__("json").loads(Path(args.selection).read_text())
    if args.split != "test":
        raise ValueError("eval-vector is for final test evaluation; use test split only or a separate diagnostic command")
    selected_split = selected.get("selection_split")
    if selected_split != "validation":
        raise ValueError(f"selected location must come from validation, got {selected_split!r}")
    alpha_payload = __import__("json").loads(Path(args.alpha).read_text())
    if alpha_payload.get("selection_split") != "validation":
        raise ValueError("selected alpha must come from validation")
    if not alpha_payload.get("positive_alpha_improved", False) and not args.allow_nonimproving_alpha:
        raise ValueError("No positive alpha improved validation margin; refusing final primary intervention evaluation")
    alpha = alpha_payload["selected_alpha"]
    vector, vector_metadata = load_access_vector(args.vector)
    if vector_metadata.get("split") != "train":
        raise ValueError(f"access vector must be trained on train split, got {vector_metadata.get('split')!r}")
    if vector_metadata.get("vector_kind") != "interface_formatting_study_primary":
        raise ValueError(f"--vector must be interface_formatting_study_primary, got {vector_metadata.get('vector_kind')!r}")
    if int(vector_metadata.get("layer", -1)) != int(selected["selected_layer"]) or vector_metadata.get("anchor") != selected["selected_anchor"]:
        raise ValueError("Loaded access vector metadata does not match selected layer/anchor")
    expected_provenance = {
        "model_name": args.model or cfg["model_name"],
        "dataset_sha256": _sha256_file(cfg["dataset_path"]),
        "conflict_pairs_sha256": _sha256_file(args.conflicts),
        "selected_location_sha256": _sha256_file(args.selection),
    }
    for key, expected in expected_provenance.items():
        if vector_metadata.get(key) != expected:
            raise ValueError(f"access vector provenance mismatch for {key}")
        if alpha_payload.get(key) != expected:
            raise ValueError(f"selected alpha provenance mismatch for {key}")
    vector_tensor_hash = vector_metadata.get("vector_tensor_sha256") or _tensor_sha256(vector)
    if alpha_payload.get("vector_tensor_sha256") != vector_tensor_hash:
        raise ValueError("selected alpha was not tuned on the loaded access vector")
    if alpha_payload.get("vector_artifact_sha256") != _sha256_file(args.vector):
        raise ValueError("selected alpha vector artifact hash does not match loaded vector file")
    shuffled_vector = None
    if not Path(args.shuffled_vector).exists():
        raise FileNotFoundError(f"Missing required shuffled-pair control vector: {args.shuffled_vector}")
    shuffled_vector, shuffled_metadata = load_access_vector(args.shuffled_vector)
    if shuffled_metadata.get("vector_kind") != "shuffled_pair_control":
        raise ValueError(f"--shuffled-vector must be shuffled_pair_control, got {shuffled_metadata.get('vector_kind')!r}")
    if int(shuffled_metadata.get("layer", -1)) != int(selected["selected_layer"]) or shuffled_metadata.get("anchor") != selected["selected_anchor"]:
        raise ValueError("Loaded shuffled vector metadata does not match selected layer/anchor")
    for key, expected in expected_provenance.items():
        if shuffled_metadata.get(key) != expected:
            raise ValueError(f"shuffled vector provenance mismatch for {key}")
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    results = evaluate_access_vector_and_controls(
        model,
        tokenizer,
        pairs,
        vector=vector,
        layer=int(selected["selected_layer"]),
        anchor=str(selected["selected_anchor"]),
        alpha=float(alpha),
        split=args.split,
        batch_size=int(cfg["scoring"]["batch_size"]),
        shuffled_vector=shuffled_vector,
        limit=args.limit,
        device=device,
    )
    removal = evaluate_clean_removal(
        model,
        tokenizer,
        pairs,
        vector=vector,
        layer=int(selected["selected_layer"]),
        anchor=str(selected["selected_anchor"]),
        alpha=float(alpha),
        split=args.split,
        batch_size=int(cfg["scoring"]["batch_size"]),
        limit=args.limit,
        device=device,
    )
    access_out = Path(cfg["outputs"]["processed_dir"]) / "access_vector_results.parquet"
    control_out = Path(cfg["outputs"]["processed_dir"]) / "control_results.parquet"
    provenance = {
        "selected_layer": int(selected["selected_layer"]),
        "selected_anchor": str(selected["selected_anchor"]),
        "selected_alpha": float(alpha),
        "vector_train_split": vector_metadata.get("split"),
        "location_selection_split": selected.get("selection_split"),
        "alpha_selection_split": alpha_payload.get("selection_split"),
        "eval_split": args.split,
        "model_name": args.model or cfg["model_name"],
        "dataset_sha256": expected_provenance["dataset_sha256"],
        "conflict_pairs_sha256": expected_provenance["conflict_pairs_sha256"],
        "selected_location_sha256": expected_provenance["selected_location_sha256"],
        "selected_alpha_sha256": _sha256_file(args.alpha),
        "vector_artifact_sha256": _sha256_file(args.vector),
        "vector_tensor_sha256": vector_tensor_hash,
    }
    for frame in (results, removal):
        for key, value in provenance.items():
            frame[key] = value
    control_frames = [results[results["condition"] != "interface_formatting_study_positive"]]
    if args.behavioral and Path(args.behavioral).exists():
        scored = read_table(args.behavioral)
        _assert_behavioral_has_primary_provenance(scored, command_name="eval-vector controls")
        for outcome in ("all_wrong", "all_correct"):
            control_frames.append(
                evaluate_item_outcome_subset_control(
                    model,
                    tokenizer,
                    scored,
                    vector=vector,
                    layer=int(selected["selected_layer"]),
                    anchor=str(selected["selected_anchor"]),
                    alpha=float(alpha),
                    outcome=outcome,
                    split=args.split,
                    batch_size=int(cfg["scoring"]["batch_size"]),
                    limit=args.limit,
                    device=device,
                )
            )
    controls = __import__("pandas").concat(control_frames, ignore_index=True)
    for key, value in provenance.items():
        controls[key] = value
    write_table(results[results["condition"] == "interface_formatting_study_positive"], access_out)
    write_table(controls, control_out)
    write_table(removal, Path(cfg["outputs"]["processed_dir"]) / "removal_results.parquet")
    interface_formatting_study_only = results[results["condition"] == "interface_formatting_study_positive"]
    write_table(summarize_intervention_results(__import__("pandas").concat([interface_formatting_study_only, controls], ignore_index=True)), control_out.with_name("control_summary.csv"))
    print(f"wrote {access_out}, {control_out}, {control_out.with_name('control_summary.csv')}, and removal_results.parquet")


def cmd_content_free_control(args) -> None:
    cfg = _config(args)
    if args.limit is not None:
        raise ValueError("content-free-control writes final artifacts and does not accept --limit")
    scored = read_table(args.behavioral)
    _assert_behavioral_has_primary_provenance(scored, command_name="content-free-control")
    selected = __import__("json").loads(Path(args.selection).read_text())
    if args.split != "test":
        raise ValueError("content-free-control is for final test controls; use test split only")
    if selected.get("selection_split") != "validation":
        raise ValueError("selected location must come from validation")
    alpha_payload = __import__("json").loads(Path(args.alpha).read_text())
    if alpha_payload.get("selection_split") != "validation":
        raise ValueError("selected alpha must come from validation")
    alpha = alpha_payload["selected_alpha"]
    vector, metadata = load_access_vector(args.vector)
    if metadata.get("split") != "train":
        raise ValueError("access vector must be trained on train split")
    if metadata.get("vector_kind") != "interface_formatting_study_primary":
        raise ValueError(f"--vector must be interface_formatting_study_primary, got {metadata.get('vector_kind')!r}")
    if int(metadata.get("layer", -1)) != int(selected["selected_layer"]) or metadata.get("anchor") != selected["selected_anchor"]:
        raise ValueError("Loaded access vector metadata does not match selected layer/anchor")
    expected_provenance = {
        "model_name": args.model or cfg["model_name"],
        "dataset_sha256": _sha256_file(cfg["dataset_path"]),
        "selected_location_sha256": _sha256_file(args.selection),
    }
    for key, expected in expected_provenance.items():
        if metadata.get(key) != expected:
            raise ValueError(f"access vector provenance mismatch for {key}")
        if alpha_payload.get(key) != expected:
            raise ValueError(f"selected alpha provenance mismatch for {key}")
    if alpha_payload.get("vector_tensor_sha256") != (metadata.get("vector_tensor_sha256") or _tensor_sha256(vector)):
        raise ValueError("selected alpha was not tuned on the loaded access vector")
    if alpha_payload.get("vector_artifact_sha256") != _sha256_file(args.vector):
        raise ValueError("selected alpha vector artifact hash does not match loaded vector file")
    model, tokenizer, device = _load_model_for_experiment(args, cfg)
    results = evaluate_content_free_control(
        model,
        tokenizer,
        scored,
        vector=vector,
        layer=int(selected["selected_layer"]),
        anchor=str(selected["selected_anchor"]),
        alpha=float(alpha),
        split=args.split,
        limit=args.limit,
        batch_size=int(cfg["scoring"]["batch_size"]),
        device=device,
    )
    provenance = {
        "selected_layer": int(selected["selected_layer"]),
        "selected_anchor": str(selected["selected_anchor"]),
        "selected_alpha": float(alpha),
        "vector_train_split": metadata.get("split"),
        "location_selection_split": selected.get("selection_split"),
        "alpha_selection_split": alpha_payload.get("selection_split"),
        "eval_split": args.split,
        "model_name": args.model or cfg["model_name"],
        "dataset_sha256": expected_provenance["dataset_sha256"],
        "selected_location_sha256": expected_provenance["selected_location_sha256"],
        "selected_alpha_sha256": _sha256_file(args.alpha),
        "vector_artifact_sha256": _sha256_file(args.vector),
        "vector_tensor_sha256": metadata.get("vector_tensor_sha256") or _tensor_sha256(vector),
        "behavioral_scores_sha256": _sha256_file(args.behavioral),
    }
    for key, value in provenance.items():
        results[key] = value
    out = Path(cfg["outputs"]["processed_dir"]) / "content_free_control.parquet"
    write_table(results, out)
    print(f"wrote {out}")


def cmd_tables(args) -> None:
    cfg = _config(args)
    behavioral_name = Path(args.behavioral).name
    output_dir = Path(cfg["outputs"]["tables_dir"])
    if behavioral_name == "behavioral_smoke.parquet":
        output_dir = Path("results/smoke/tables")
    patching_results_path = None
    if args.include_legacy_semantic_patching:
        patching_results_path = Path(cfg["outputs"]["processed_dir"]) / "patching_results.parquet"
    paths = write_standard_tables(
        dataset_audit_path=Path(cfg["outputs"]["metadata_dir"]) / "dataset_audit.json",
        wrapper_audit_path=Path(cfg["outputs"]["metadata_dir"]) / "wrapper_audit.csv",
        behavioral_path=args.behavioral,
        output_dir=output_dir,
        patching_results_path=patching_results_path,
    )
    print("wrote " + ", ".join(str(path) for path in paths.values()))


def cmd_figures(args) -> None:
    cfg = _config(args)
    scored = read_table(args.behavioral)
    out_dir = Path(cfg["outputs"]["figures_dir"])
    if Path(args.behavioral).name == "behavioral_smoke.parquet":
        out_dir = Path("results/smoke/figures")
    else:
        dataset_audit = read_json(Path(cfg["outputs"]["metadata_dir"]) / "dataset_audit.json")
        expected_rows = int(dataset_audit["num_rows"])
        if len(scored) != expected_rows:
            raise ValueError(
                f"Behavioral table has {len(scored)} rows but dataset audit expects {expected_rows}; "
                "refusing to write publication-style figures."
            )
    run_type = "smoke" if Path(args.behavioral).name == "behavioral_smoke.parquet" else "final"
    plot_source = primary_behavioral_rows(scored, run_type=run_type)
    wrapper_accuracy_heatmap(plot_source, out_dir / "wrapper_accuracy_heatmap.png")
    conflict_histogram(item_outcome_summary(plot_source), out_dir / "format_conflict_distribution.png")
    print(f"wrote figures under {out_dir}")


def cmd_write_manifest(args) -> None:
    cfg = _config(args)
    metadata_dir = Path(cfg["outputs"]["metadata_dir"])
    split_manifest_path = metadata_dir / "split_manifest.json"
    dataset_audit_path = metadata_dir / "dataset_audit.json"
    if not dataset_audit_path.exists():
        raise FileNotFoundError(f"Missing dataset audit: {dataset_audit_path}")
    if not split_manifest_path.exists():
        raise FileNotFoundError(f"Missing split manifest: {split_manifest_path}")
    output_cfg = cfg.get("outputs", {})
    if {"raw_dir", "processed_dir", "figures_dir", "tables_dir"}.issubset(output_cfg):
        missing = [
            path
            for path in REQUIRED_FINAL_ARTIFACTS
            if path != "results/metadata/experiment_manifest.json" and not (PROJECT_ROOT / path).exists()
        ]
        if missing:
            raise FileNotFoundError("Cannot write final manifest before required artifacts exist:\n" + "\n".join(missing))
    manifest = {
        "manifest_schema_version": 1,
        "project": "Interface Formatting Study",
        "description": "Diagnostics for interface-formatting effects in multiple-choice language-model evaluation.",
        "dataset": {
            "path": cfg["dataset_path"],
            "sha256": _sha256_file(cfg["dataset_path"]),
            "audit": read_json(dataset_audit_path),
        },
        "model": {"name": cfg["model_name"]},
        "config": cfg,
        "split_manifest": read_json(split_manifest_path),
        "dependency_versions": _dependency_versions(),
        "source_hashes": _source_hashes(args.config),
        "artifact_hashes": _existing_artifact_hashes(),
        "pipeline": [
            "audit-dataset",
            "wrapper-audit",
            "behavioral",
            "conflicts",
            "attention-diagnostics",
            "vanilla-convergence",
            "focused-patching-controls",
            "figures",
            "tables",
            "write-manifest",
            "deliverables",
        ],
        "scientific_invariants": [
            "score answer labels by conditional likelihood, never primary generation parsing",
            "calibrate with content-free wrapper priors only when same-wrapper redaction succeeds",
            "split by item_id into internal partitions and never tune on the internal test partition",
            "primary mechanistic claims use verified pure-interface conflicts with same-wrapper calibration",
            "attention is diagnostic and not treated as causal evidence by itself",
            "focused causal controls distinguish same-item answer-state transfer from task-representation restoration",
            "primary focused patching uses options_end and all_option_ends content anchors rather than final answer-token states",
            "final tables and figures require full behavioral row coverage",
        ],
    }
    out = metadata_dir / "experiment_manifest.json"
    write_json(manifest, out)
    print(f"wrote {out}")


def cmd_deliverables(args) -> None:
    result = prepare_deliverables(compile_pdf=not args.no_compile)
    print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface_formatting_study")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit-dataset").set_defaults(func=cmd_audit_dataset)
    sub.add_parser("wrapper-audit").set_defaults(func=cmd_wrapper_audit)

    smoke = sub.add_parser("behavioral-smoke")
    smoke.add_argument("--limit", type=int, default=20)
    smoke.add_argument("--model", default=None)
    smoke.add_argument("--local-files-only", action="store_true")
    smoke.set_defaults(func=cmd_behavioral_smoke)

    behavioral = sub.add_parser("behavioral")
    behavioral.add_argument("--limit", type=int, default=None)
    behavioral.add_argument("--model", default=None)
    behavioral.add_argument("--local-files-only", action="store_true")
    behavioral.add_argument("--resume", action="store_true")
    behavioral.set_defaults(func=cmd_behavioral)

    conflicts = sub.add_parser("conflicts")
    conflicts.add_argument("--behavioral", required=True)
    conflicts.set_defaults(func=cmd_conflicts)

    figures = sub.add_parser("figures")
    figures.add_argument("--behavioral", required=True)
    figures.set_defaults(func=cmd_figures)

    tables = sub.add_parser("tables")
    tables.add_argument("--behavioral", default="results/raw/behavioral_scores.parquet")
    tables.add_argument(
        "--include-legacy-semantic-patching",
        action="store_true",
        help="also write the old broad semantic patching table when patching_results.parquet exists",
    )
    tables.set_defaults(func=cmd_tables)

    patching = sub.add_parser("patching-sweep")
    patching.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    patching.add_argument("--layers", default=None, help="'all' or comma-separated layer ids; defaults to config activation.layers")
    patching.add_argument("--anchors", default=None, help="comma-separated anchors; defaults to config activation.anchors")
    patching.add_argument("--cap", type=int, default=None)
    patching.add_argument("--model", default=None)
    patching.add_argument("--local-files-only", action="store_true")
    patching.add_argument("--allow-leaky-anchors", action="store_true")
    patching.set_defaults(func=cmd_patching_sweep)

    attention = sub.add_parser("attention-diagnostics")
    attention.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    attention.add_argument("--behavioral", default="results/raw/behavioral_scores.parquet")
    attention.add_argument("--split", default="validation")
    attention.add_argument("--cap", type=int, default=None)
    attention.add_argument("--anchors", default=None, help="comma-separated anchors; defaults to focused_mechanistic.anchors")
    attention.add_argument("--model", default=None)
    attention.add_argument("--local-files-only", action="store_true")
    attention.add_argument("--attn-implementation", default="eager", help="'eager' is safest for output_attentions; use 'default' to leave model default")
    attention.set_defaults(func=cmd_attention_diagnostics)

    convergence = sub.add_parser("vanilla-convergence")
    convergence.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    convergence.add_argument("--split", default="validation")
    convergence.add_argument("--cap", type=int, default=None)
    convergence.add_argument("--layers", default=None, help="comma-separated layer ids; defaults to focused_mechanistic.layers")
    convergence.add_argument("--anchors", default=None, help="comma-separated anchors; defaults to focused_mechanistic.anchors")
    convergence.add_argument("--model", default=None)
    convergence.add_argument("--local-files-only", action="store_true")
    convergence.set_defaults(func=cmd_vanilla_convergence)

    focused_controls = sub.add_parser("focused-patching-controls")
    focused_controls.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    focused_controls.add_argument("--split", default="validation")
    focused_controls.add_argument("--cap", type=int, default=None)
    focused_controls.add_argument("--layers", default=None, help="comma-separated layer ids; defaults to focused_mechanistic.layers")
    focused_controls.add_argument("--anchors", default=None, help="comma-separated anchors; defaults to focused_mechanistic.anchors")
    focused_controls.add_argument("--conditions", default=None, help="comma-separated focused control conditions")
    focused_controls.add_argument("--model", default=None)
    focused_controls.add_argument("--local-files-only", action="store_true")
    focused_controls.set_defaults(func=cmd_focused_patching_controls)

    train = sub.add_parser("train-vector")
    train.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    train.add_argument("--selection", default="results/metadata/selected_location.json")
    train.add_argument("--balance-by", choices=["label", "label_subject"], default="label")
    train.add_argument("--limit", type=int, default=None)
    train.add_argument("--model", default=None)
    train.add_argument("--local-files-only", action="store_true")
    train.set_defaults(func=cmd_train_vector)

    tune = sub.add_parser("tune-alpha")
    tune.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    tune.add_argument("--selection", default="results/metadata/selected_location.json")
    tune.add_argument("--vector", default="results/processed/interface_formatting_study_vector.pt")
    tune.add_argument("--limit", type=int, default=None)
    tune.add_argument("--model", default=None)
    tune.add_argument("--local-files-only", action="store_true")
    tune.set_defaults(func=cmd_tune_alpha)

    ev = sub.add_parser("eval-vector")
    ev.add_argument("--conflicts", default="results/processed/conflict_pairs.parquet")
    ev.add_argument("--selection", default="results/metadata/selected_location.json")
    ev.add_argument("--alpha", default="results/metadata/selected_alpha.json")
    ev.add_argument("--vector", default="results/processed/interface_formatting_study_vector.pt")
    ev.add_argument("--shuffled-vector", default="results/processed/shuffled_pair_vector.pt")
    ev.add_argument("--behavioral", default="results/raw/behavioral_scores.parquet")
    ev.add_argument("--split", default="test")
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--model", default=None)
    ev.add_argument("--local-files-only", action="store_true")
    ev.add_argument("--allow-nonimproving-alpha", action="store_true")
    ev.set_defaults(func=cmd_eval_vector)

    cfree = sub.add_parser("content-free-control")
    cfree.add_argument("--behavioral", default="results/raw/behavioral_scores.parquet")
    cfree.add_argument("--selection", default="results/metadata/selected_location.json")
    cfree.add_argument("--alpha", default="results/metadata/selected_alpha.json")
    cfree.add_argument("--vector", default="results/processed/interface_formatting_study_vector.pt")
    cfree.add_argument("--split", default="test")
    cfree.add_argument("--limit", type=int, default=None)
    cfree.add_argument("--model", default=None)
    cfree.add_argument("--local-files-only", action="store_true")
    cfree.set_defaults(func=cmd_content_free_control)

    sub.add_parser("write-manifest").set_defaults(func=cmd_write_manifest)

    deliverables = sub.add_parser("deliverables")
    deliverables.add_argument("--no-compile", action="store_true")
    deliverables.set_defaults(func=cmd_deliverables)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
