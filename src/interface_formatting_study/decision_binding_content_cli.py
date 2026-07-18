from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .causal_option_audit import load_causal_option_annotations
from .decision_binding_cli import _load_bundle
from .decision_binding_content import (
    CANDIDATE_TARGET_STATE_COLUMNS,
    CandidateRanker,
    candidate_score_rows,
    capture_candidate_states,
    classify_candidate_target_state,
    evaluate_candidate_ranker,
    fit_candidate_ranker,
    gate_candidate_reader,
    load_candidate_ranker,
    locate_content_token_indices,
    majority_candidate_scores,
    metadata_candidate_features,
    prepare_candidate_sites,
    save_candidate_ranker,
    select_candidate_reader,
)
from .model_loader import load_model_and_tokenizer
from .model_profiles import MODEL_PROFILES, get_model_profile
from .model_runner import StopState
from .run_identity import build_semantic_identity, default_semantic_source_paths, sha256_file
from .shards import ShardStore, run_sharded_phase
from .utils import read_table, write_table_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_ITEM_COUNTS = {"probe_train": 1801, "layer_select": 300, "reader_gate": 300}
L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
CONTROL_READERS = ("position", "label", "position_label", "answer_length", "majority")
TARGET_POLICY_VERSION = 1
PARITY_MAX_DIFFERENCE = 2e-2
PARITY_COMPARISON_EPSILON = 1e-12


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _role_items(frame: pd.DataFrame) -> dict[str, int]:
    return {
        role: int(frame.loc[frame["readout_role"].astype(str) == role, "item_id"].nunique())
        for role in ROLE_ITEM_COUNTS
    }


def _profile_for_bundle(manifest: Mapping[str, object]):
    model = manifest.get("model", {})
    for profile in MODEL_PROFILES.values():
        if (
            isinstance(model, Mapping)
            and str(model.get("id")) == profile.model_id
            and str(model.get("revision")) == profile.revision
        ):
            return profile
    raise ValueError("decision-binding source model does not match a pinned profile")


def cmd_prepare(args) -> None:
    source_root = Path(args.bundle)
    source_manifest, ledger, _ = _load_bundle(source_root)
    if source_manifest.get("stage") != "discovery":
        raise ValueError("candidate-local preparation accepts discovery data only")
    applicability_path = Path(args.applicability)
    if sha256_file(applicability_path) != source_manifest.get("source_applicability_sha256"):
        raise ValueError("applicability checksum does not match the source bundle")
    applicability = read_table(applicability_path)
    option_maps = load_causal_option_annotations(args.option_audit)
    sites = prepare_candidate_sites(ledger, applicability, option_maps=option_maps)
    if len(sites) != len(ledger) or set(sites["work_key"]) != set(ledger["work_key"]):
        raise RuntimeError("candidate preparation did not preserve every source work key")
    observed_roles = _role_items(sites)
    if observed_roles != ROLE_ITEM_COUNTS:
        raise RuntimeError(f"candidate preparation changed frozen roles: {observed_roles}")
    profile = _profile_for_bundle(source_manifest)
    output = Path(args.output_dir)
    path = output / "candidate_sites.parquet"
    write_table_atomic(sites, path)
    _atomic_json({
        "schema_version": 1,
        "stage": "discovery",
        "rows": int(len(sites)),
        "sha256": sha256_file(path),
        "work_keys_sha256": hashlib.sha256(
            "\n".join(sorted(sites["work_key"].astype(str))).encode()
        ).hexdigest(),
        "role_items": observed_roles,
        "target_evaluable_rows": int(sites["content_target_evaluable"].sum()),
        "mapping_mismatch_rows": int((~sites["mapping_matches_ledger"].astype(bool)).sum()),
        "source_bundle_manifest_sha256": sha256_file(source_root / "bundle_manifest.json"),
        "source_readout_sha256": source_manifest["readout"]["sha256"],
        "source_applicability_sha256": sha256_file(applicability_path),
        "option_audit_manifest_sha256": sha256_file(Path(args.option_audit) / "manifest.json"),
        "source_model": {
            "id": profile.model_id,
            "revision": profile.revision,
            "slug": profile.slug,
        },
    }, output / "prepared_manifest.json")


def _load_prepared_bundle(root: str | Path) -> tuple[dict[str, object], pd.DataFrame]:
    bundle = Path(root)
    manifest_path = bundle / "prepared_manifest.json"
    sites_path = bundle / "candidate_sites.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "discovery":
        raise RuntimeError("candidate prepared manifest has an unsupported identity")
    if sha256_file(sites_path) != manifest.get("sha256"):
        raise RuntimeError("candidate prepared bundle checksum mismatch")
    sites = read_table(sites_path)
    if int(manifest.get("rows", -1)) != len(sites) or sites["work_key"].duplicated().any():
        raise RuntimeError("candidate prepared bundle row identity mismatch")
    if _role_items(sites) != {
        str(key): int(value) for key, value in manifest.get("role_items", {}).items()
    }:
        raise RuntimeError("candidate prepared bundle role counts mismatch")
    return manifest, sites


def _select_canary_sites(frame: pd.DataFrame, *, items_per_role: int, seed: int) -> pd.DataFrame:
    if items_per_role <= 0:
        raise ValueError("canary items per role must be positive")
    chosen: list[pd.DataFrame] = []
    for role in ROLE_ITEM_COUNTS:
        group = frame[frame["readout_role"].astype(str) == role]
        records = []
        for item_id, item in group.groupby("item_id", sort=False):
            prompts = item["prompt"].astype(str)
            candidates = [str(value) for values in item.get("candidate_texts", []) for value in values]
            records.append({
                "item_id": str(item_id),
                "max_length": int(prompts.str.len().max()),
                "non_ascii": any(not prompt.isascii() for prompt in prompts),
                "multiword": any(any(character.isspace() for character in value.strip()) for value in candidates),
                "digest": hashlib.sha256(f"{seed}|{role}|{item_id}".encode()).hexdigest(),
            })
        selected: list[str] = []
        if records:
            selected.append(str(max(records, key=lambda row: row["max_length"])["item_id"]))
        for feature in ("non_ascii", "multiword"):
            matching = sorted(
                (row for row in records if row[feature] and row["item_id"] not in selected),
                key=lambda row: row["digest"],
            )
            if matching and len(selected) < items_per_role:
                selected.append(str(matching[0]["item_id"]))
        for row in sorted(records, key=lambda value: value["digest"]):
            if row["item_id"] not in selected and len(selected) < items_per_role:
                selected.append(str(row["item_id"]))
        item_ids = selected
        if len(item_ids) != items_per_role:
            raise ValueError(f"canary lacks {items_per_role} items for {role}")
        chosen.append(group[group["item_id"].astype(str).isin(item_ids)])
    return pd.concat(chosen, ignore_index=True).sort_values("work_key", kind="mergesort")


def _chunks(frame: pd.DataFrame, size: int) -> list[tuple[str, pd.DataFrame]]:
    work: list[tuple[str, pd.DataFrame]] = []
    ordered = frame.sort_values("work_key", kind="mergesort").reset_index(drop=True)
    for start in range(0, len(ordered), size):
        chunk = ordered.iloc[start : start + size].copy()
        digest = hashlib.sha256("\n".join(chunk["work_key"].astype(str)).encode()).hexdigest()
        work.append((digest[:20], chunk))
    return work


def _verify_raw_winners(
    chunk: pd.DataFrame, raw_log_probs: torch.Tensor
) -> pd.DataFrame:
    """Validate and classify a fresh forward without dropping unstable rows."""

    return classify_candidate_target_state(chunk, raw_log_probs)


def _eligibility_receipt(state: pd.DataFrame) -> dict[str, object]:
    required = {
        "work_key", "reader_target_evaluable", "reader_target_ineligibility_reason",
    }
    if missing := required - set(state.columns):
        raise ValueError(f"candidate target state is missing columns: {sorted(missing)}")
    if state["work_key"].duplicated().any():
        raise ValueError("candidate target state has duplicate work keys")
    allowed_reasons = {
        "stored_raw_tie", "ambiguous_answer_content", "fresh_raw_tie",
        "stored_fresh_content_mismatch", "eligible",
    }
    reasons = state["reader_target_ineligibility_reason"].astype(str)
    evaluable = state["reader_target_evaluable"].astype(bool)
    if not set(reasons).issubset(allowed_reasons) or not (evaluable == reasons.eq("eligible")).all():
        raise ValueError("candidate target state has inconsistent eligibility reasons")
    rows = sorted(
        (
            str(row.work_key),
            bool(row.reader_target_evaluable),
            str(row.reader_target_ineligibility_reason),
        )
        for row in state.itertuples()
    )
    reason_counts = {
        str(key): int(value)
        for key, value in state["reader_target_ineligibility_reason"].astype(str)
        .value_counts().sort_index().items()
    }
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    canonical_state = state[list(CANDIDATE_TARGET_STATE_COLUMNS)].sort_values(
        "work_key", kind="mergesort"
    ).to_json(orient="records", double_precision=15)
    return {
        "target_policy_version": TARGET_POLICY_VERSION,
        "total_rows": int(len(state)),
        "evaluable_rows": int(state["reader_target_evaluable"].astype(bool).sum()),
        "reason_counts": reason_counts,
        "eligibility_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "target_state_sha256": hashlib.sha256(
            canonical_state.encode("utf-8")
        ).hexdigest(),
    }


def _runtime_environment_receipt(*, attention_backend: str) -> dict[str, object]:
    cuda = bool(torch.cuda.is_available())
    return {
        "device_type": "cuda" if cuda else "cpu",
        "gpu_name": str(torch.cuda.get_device_name(0)) if cuda else None,
        "gpu_compute_capability": (
            ".".join(map(str, torch.cuda.get_device_capability(0))) if cuda else None
        ),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda else False,
        "attention_backend": str(attention_backend),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "inference_dtype": "bfloat16" if cuda else "float32",
        "fit_dtype": "float32",
    }


def _categorical_parity_receipt(
    left: np.ndarray,
    right: np.ndarray,
    work_keys: Sequence[str],
    *,
    maximum_difference: float = PARITY_MAX_DIFFERENCE,
) -> dict[str, object]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if (
        left_values.shape != right_values.shape
        or left_values.ndim != 2
        or left_values.shape[0] != len(work_keys)
        or left_values.shape[1] != 4
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        raise ValueError("categorical parity requires finite aligned four-way scores")
    row_differences = np.max(np.abs(left_values - right_values), axis=1)
    ambiguous: list[dict[str, object]] = []
    resolvable: list[str] = []
    for index, work_key in enumerate(map(str, work_keys)):
        if int(left_values[index].argmax()) == int(right_values[index].argmax()):
            continue
        left_sorted = np.sort(left_values[index])
        right_sorted = np.sort(right_values[index])
        left_margin = float(left_sorted[-1] - left_sorted[-2])
        right_margin = float(right_sorted[-1] - right_sorted[-2])
        explained = (
            left_margin <= 2.0 * row_differences[index]
            and right_margin <= 2.0 * row_differences[index]
        )
        if explained:
            ambiguous.append({
                "work_key": work_key,
                "left_margin": left_margin,
                "right_margin": right_margin,
                "max_coordinate_difference": float(row_differences[index]),
            })
        else:
            resolvable.append(work_key)
    observed = float(row_differences.max(initial=0.0))
    return {
        "passes": bool(
            observed <= float(maximum_difference) + PARITY_COMPARISON_EPSILON
            and not resolvable
        ),
        "maximum_difference": float(maximum_difference),
        "observed_maximum_difference": observed,
        "ambiguous_work_keys": [row["work_key"] for row in ambiguous],
        "ambiguous_count": int(len(ambiguous)),
        "ambiguous_rows": ambiguous,
        "resolvable_disagreement_work_keys": resolvable,
        "resolvable_disagreement_count": int(len(resolvable)),
    }


def _token_positions(tokenizer, chunk: pd.DataFrame) -> list[list[int]]:
    return [
        locate_content_token_indices(tokenizer, str(row.prompt), row.content_char_spans)
        for row in chunk.itertuples()
    ]


def _save_activation_shard(
    path: Path,
    *,
    activations: torch.Tensor,
    raw_log_probs: torch.Tensor,
    target_state: pd.DataFrame,
    work_keys: Sequence[str],
    semantic_sha256: str,
) -> None:
    normalized_keys = list(map(str, work_keys))
    if list(target_state["work_key"].astype(str)) != normalized_keys:
        raise ValueError("training activation target state is not work-key aligned")
    if tuple(target_state.columns) != tuple(CANDIDATE_TARGET_STATE_COLUMNS):
        raise ValueError("training activation target state has an unsupported schema")
    raw = raw_log_probs.detach().float().cpu()
    if raw.shape != (len(normalized_keys), 4) or not torch.isfinite(raw).all():
        raise ValueError("training activation raw scores are malformed")
    state_json = target_state.to_json(orient="split", index=False, double_precision=15)
    state_sha256 = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 2,
        "semantic_sha256": semantic_sha256,
        "work_keys": normalized_keys,
        "activations": activations.cpu(),
        "raw_log_probs": raw,
        "target_state_json": state_json,
        "target_state_sha256": state_sha256,
    }, temporary)
    os.replace(temporary, path)
    _atomic_json({
        "schema_version": 2,
        "semantic_sha256": semantic_sha256,
        "work_keys": normalized_keys,
        "target_state_sha256": state_sha256,
        "eligibility": _eligibility_receipt(target_state),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }, path.with_suffix(path.suffix + ".manifest.json"))


def _load_activation_shard(
    path: Path,
    *,
    work_keys: Sequence[str],
    semantic_sha256: str,
) -> tuple[torch.Tensor, pd.DataFrame]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise RuntimeError(f"training activation shard checksum manifest is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 2
        or manifest.get("semantic_sha256") != semantic_sha256
        or manifest.get("work_keys") != list(map(str, work_keys))
        or manifest.get("sha256") != sha256_file(path)
        or int(manifest.get("bytes", -1)) != path.stat().st_size
    ):
        raise RuntimeError(f"training activation shard checksum mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != 2
        or payload.get("semantic_sha256") != semantic_sha256
        or payload.get("work_keys") != list(map(str, work_keys))
    ):
        raise RuntimeError(f"training activation shard identity mismatch: {path}")
    values = payload.get("activations")
    if not isinstance(values, torch.Tensor) or values.ndim != 4 or values.shape[2] != 4:
        raise RuntimeError(f"training activation shard is malformed: {path}")
    raw = payload.get("raw_log_probs")
    state_json = payload.get("target_state_json")
    if (
        not isinstance(raw, torch.Tensor)
        or raw.shape != (len(work_keys), 4)
        or not torch.isfinite(raw).all()
        or not isinstance(state_json, str)
        or hashlib.sha256(state_json.encode("utf-8")).hexdigest()
        != payload.get("target_state_sha256")
        or payload.get("target_state_sha256") != manifest.get("target_state_sha256")
    ):
        raise RuntimeError(f"training activation shard target state mismatch: {path}")
    state_payload = json.loads(state_json)
    state = pd.DataFrame(state_payload["data"], columns=state_payload["columns"])
    if (
        tuple(state.columns) != tuple(CANDIDATE_TARGET_STATE_COLUMNS)
        or list(state["work_key"].astype(str)) != list(map(str, work_keys))
        or _eligibility_receipt(state) != manifest.get("eligibility")
        or not np.allclose(
            state[[f"fresh_raw_score_{label}" for label in "ABCD"]].to_numpy(dtype=float),
            raw.numpy(), atol=1e-7, rtol=0.0,
        )
    ):
        raise RuntimeError(f"training activation shard target state mismatch: {path}")
    return values, state


def _capture_training(
    root: Path,
    training: pd.DataFrame,
    model,
    tokenizer,
    *,
    identity,
    batch_size: int,
    max_batch_tokens: int,
    chunk_size: int,
    stop: StopState,
    progress,
) -> tuple[torch.Tensor, pd.DataFrame] | None:
    values: list[torch.Tensor] = []
    states: list[pd.DataFrame] = []
    work = _chunks(training, chunk_size)
    for index, (key, chunk) in enumerate(work):
        path = root / "activation_shards" / "training" / f"{key}.pt"
        if path.exists():
            captured, target_state = _load_activation_shard(
                path, work_keys=chunk["work_key"], semantic_sha256=identity.semantic_sha256
            )
        else:
            if stop.requested:
                return None
            result = capture_candidate_states(
                model,
                tokenizer,
                chunk["prompt"].astype(str).tolist(),
                _token_positions(tokenizer, chunk),
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
            )
            target_state = _verify_raw_winners(chunk, result.raw_log_probs)
            captured = result.activations
            _save_activation_shard(
                path,
                activations=captured,
                raw_log_probs=result.raw_log_probs,
                target_state=target_state,
                work_keys=chunk["work_key"],
                semantic_sha256=identity.semantic_sha256,
            )
        values.append(captured)
        states.append(target_state)
        progress("training_capture", index + 1, len(work))
    return torch.cat(values, dim=0), pd.concat(states, ignore_index=True)


def _eligible_training_inputs(
    training_pool: pd.DataFrame,
    activations: torch.Tensor,
    target_state: pd.DataFrame,
) -> tuple[pd.DataFrame, torch.Tensor, np.ndarray]:
    if len(activations) != len(training_pool):
        raise RuntimeError("candidate training activations do not match the frozen pool")
    training = _join_target_state(training_pool, target_state)
    mask = training["reader_target_evaluable"].astype(bool).to_numpy(copy=True)
    if not mask.any():
        raise RuntimeError("candidate training pool has no runtime-eligible content targets")
    eligible = training.loc[mask].reset_index(drop=True)
    return (
        eligible,
        activations[torch.from_numpy(mask)],
        eligible["winner_position"].to_numpy(dtype=int),
    )


def _fit_rankers(
    root: Path,
    activations: torch.Tensor,
    targets: np.ndarray,
    *,
    l2_grid: Sequence[float],
    semantic_sha256: str,
    prefix: str,
    max_iter: int,
    fit_device=None,
    should_stop=None,
) -> dict[float, CandidateRanker]:
    rankers: dict[float, CandidateRanker] = {}
    for l2 in l2_grid:
        if should_stop is not None and should_stop():
            break
        path = root / "rankers" / f"{prefix}-l2-{float(l2):.0e}.npz"
        if path.exists():
            ranker = load_candidate_ranker(path, semantic_sha256=semantic_sha256)
        else:
            ranker = fit_candidate_ranker(
                activations,
                targets,
                l2=float(l2),
                max_iter=max_iter,
                device=fit_device,
            )
            save_candidate_ranker(path, ranker, semantic_sha256=semantic_sha256)
        rankers[float(l2)] = ranker
    return rankers


def _score_model_phase(
    root: Path,
    phase: str,
    sites: pd.DataFrame,
    rankers: Mapping[float, CandidateRanker],
    model,
    tokenizer,
    *,
    identity,
    batch_size: int,
    max_batch_tokens: int,
    chunk_size: int,
    stop: StopState,
    progress,
    selected_layer: int | None = None,
    extra_rankers: Mapping[str, CandidateRanker] | None = None,
) -> tuple[pd.DataFrame, bool]:
    store = ShardStore(root / "shards" / phase, identity)
    work = _chunks(sites, chunk_size)

    def process(_key: str, chunk: pd.DataFrame) -> pd.DataFrame:
        captured = capture_candidate_states(
            model,
            tokenizer,
            chunk["prompt"].astype(str).tolist(),
            _token_positions(tokenizer, chunk),
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        target_state = _verify_raw_winners(chunk, captured.raw_log_probs)
        scored_sites = chunk.drop(
            columns=[
                column for column in CANDIDATE_TARGET_STATE_COLUMNS
                if column != "work_key" and column in chunk
            ],
            errors="ignore",
        ).merge(target_state, on="work_key", how="left", validate="one_to_one")
        if len(scored_sites) != len(chunk):
            raise RuntimeError("candidate score target state did not preserve every source row")
        rows = []
        for l2, ranker in rankers.items():
            scored = candidate_score_rows(
                scored_sites,
                evaluate_candidate_ranker(ranker, captured.activations),
                l2=l2,
                reader_name="content",
            )
            if selected_layer is not None:
                scored = scored[scored["layer"].astype(int) == selected_layer]
            rows.append(scored)
        for name, ranker in (extra_rankers or {}).items():
            scored = candidate_score_rows(
                scored_sites,
                evaluate_candidate_ranker(ranker, captured.activations),
                l2=ranker.l2,
                reader_name=name,
            )
            if selected_layer is not None:
                scored = scored[scored["layer"].astype(int) == selected_layer]
            rows.append(scored)
        return pd.concat(rows, ignore_index=True)

    merged = run_sharded_phase(
        store,
        work,
        process,
        max_work_units=1,
        max_seconds=300,
        should_stop=lambda: stop.requested,
        on_flush=lambda completed, _path: progress(phase, len(completed), len(work)),
    )
    complete = store.completed_work_keys() == {key for key, _ in work}
    return merged, complete


def _score_metadata_rankers(
    sites: pd.DataFrame,
    rankers: Mapping[float, CandidateRanker],
    feature_name: str,
) -> pd.DataFrame:
    features = metadata_candidate_features(sites)[feature_name]
    return pd.concat([
        candidate_score_rows(
            sites,
            evaluate_candidate_ranker(ranker, features),
            l2=l2,
            reader_name=feature_name,
        )
        for l2, ranker in rankers.items()
    ], ignore_index=True)


def _score_selected_metadata_controls(
    sites: pd.DataFrame,
    selected_controls: Mapping[str, pd.DataFrame],
    fitted_controls: Mapping[str, Mapping[float, CandidateRanker]],
) -> dict[str, pd.DataFrame]:
    scored: dict[str, pd.DataFrame] = {}
    for name, fitted in fitted_controls.items():
        selected_l2 = float(selected_controls[name].iloc[0]["l2"])
        scored[name] = _score_metadata_rankers(
            sites, {selected_l2: fitted[selected_l2]}, name
        )
    return scored


def _target_state_from_scores(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(CANDIDATE_TARGET_STATE_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"candidate scores are missing target state: {sorted(missing)}")
    state = frame[list(CANDIDATE_TARGET_STATE_COLUMNS)].drop_duplicates().copy()
    if state["work_key"].duplicated().any():
        raise RuntimeError("candidate scores disagree on repeated work-key target state")
    return state.sort_values("work_key", kind="mergesort").reset_index(drop=True)


def _join_target_state(sites: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    base = sites.drop(
        columns=[
            column for column in CANDIDATE_TARGET_STATE_COLUMNS
            if column != "work_key" and column in sites
        ],
        errors="ignore",
    )
    joined = base.merge(state, on="work_key", how="left", validate="one_to_one")
    if len(joined) != len(sites) or joined["reader_target_evaluable"].isna().any():
        raise RuntimeError("candidate target state did not cover every source row")
    return joined


def _best_control(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = []
    for l2, group in frame.groupby("l2", sort=True):
        if "reader_target_evaluable" not in group:
            raise ValueError("control selection requires runtime target eligibility")
        eligible = group[group["reader_target_evaluable"].astype(bool)]
        if eligible.empty:
            raise ValueError("control selection has no runtime-eligible rows")
        probabilities = eligible[[f"content_prob_{value}" for value in range(4)]].to_numpy()
        target = eligible["actual_winner_content_id"].to_numpy(dtype=int)
        correct = probabilities.argmax(axis=1) == target
        scored = eligible.assign(_correct=correct)
        arms = []
        for manipulation in ("position_only", "label_only"):
            arm = scored[scored["manipulation"].astype(str) == manipulation]
            arms.append(float(arm.groupby("item_id")["_correct"].mean().mean()))
        candidates.append((min(arms), float(l2), group))
    return max(candidates, key=lambda value: (value[0], value[1]))[2].copy()


def _run_parity_checks(
    root: Path,
    ranker: CandidateRanker,
    activations: torch.Tensor,
    targets: np.ndarray,
    model,
    tokenizer,
    gate_sites: pd.DataFrame,
    *,
    identity,
    batch_size: int,
    max_batch_tokens: int,
    max_iter: int,
    fit_device,
    seed: int,
    selected_layer: int,
) -> dict[str, object]:
    sample = activations[: min(32, len(activations))]
    expected = evaluate_candidate_ranker(ranker, sample)
    path = root / "rankers" / f"content-l2-{ranker.l2:.0e}.npz"
    loaded = load_candidate_ranker(path, semantic_sha256=identity.semantic_sha256)
    loaded_scores = evaluate_candidate_ranker(loaded, sample)
    save_load_difference = float(np.max(np.abs(expected - loaded_scores)))
    save_load = bool(
        np.array_equal(expected.argmax(axis=2), loaded_scores.argmax(axis=2))
        and save_load_difference <= 1e-7
    )

    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        torch.manual_seed(seed + 9001)
        repeated = fit_candidate_ranker(
            activations,
            targets,
            l2=ranker.l2,
            max_iter=max_iter,
            device=fit_device,
        )
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    repeated_scores = evaluate_candidate_ranker(repeated, sample)
    seed_difference = float(np.max(np.abs(expected - repeated_scores)))
    seed_parity = bool(
        np.array_equal(expected.argmax(axis=2), repeated_scores.argmax(axis=2))
        and seed_difference <= 1e-5
    )

    eligible_gate = gate_sites[gate_sites["reader_target_evaluable"].astype(bool)]
    batch_sites = eligible_gate.sort_values("work_key", kind="mergesort").iloc[:4].copy()
    if batch_sites.empty:
        raise RuntimeError("batch parity has no runtime-eligible reader-gate rows")
    positions = _token_positions(tokenizer, batch_sites)
    batched = capture_candidate_states(
        model, tokenizer, batch_sites["prompt"].astype(str).tolist(), positions,
        batch_size=min(batch_size, len(batch_sites)), max_batch_tokens=max_batch_tokens,
    )
    scalar = [
        capture_candidate_states(
            model, tokenizer, [str(row.prompt)], [position],
            batch_size=1, max_batch_tokens=max_batch_tokens,
        )
        for row, position in zip(batch_sites.itertuples(), positions, strict=True)
    ]
    scalar_activations = torch.cat([capture.activations for capture in scalar])
    scalar_raw = torch.cat([capture.raw_log_probs for capture in scalar])
    activation_difference = float(
        (batched.activations[:, selected_layer].float()
         - scalar_activations[:, selected_layer].float()).abs().max()
    )
    raw_difference = float((batched.raw_log_probs - scalar_raw).abs().max())
    batched_reader = evaluate_candidate_ranker(ranker, batched.activations)[:, selected_layer]
    scalar_reader = evaluate_candidate_ranker(ranker, scalar_activations)[:, selected_layer]
    reader_receipt = _categorical_parity_receipt(
        batched_reader, scalar_reader, batch_sites["work_key"],
    )
    raw_receipt = _categorical_parity_receipt(
        batched.raw_log_probs.float().cpu().numpy(),
        scalar_raw.float().cpu().numpy(),
        batch_sites["work_key"],
    )
    batch_parity = bool(
        reader_receipt["passes"]
        and raw_receipt["passes"]
        and activation_difference <= PARITY_MAX_DIFFERENCE
        and raw_difference <= PARITY_MAX_DIFFERENCE
    )
    return {
        "schema_version": 2,
        "semantic_sha256": identity.semantic_sha256,
        "save_load": save_load,
        "batch": batch_parity,
        "seed": seed_parity,
        "max_save_load_probability_difference": save_load_difference,
        "max_seed_probability_difference": seed_difference,
        "max_batch_activation_difference": activation_difference,
        "max_batch_raw_log_probability_difference": raw_difference,
        "max_batch_reader_probability_difference": reader_receipt[
            "observed_maximum_difference"
        ],
        "batch_reader_categorical": reader_receipt,
        "batch_raw_categorical": raw_receipt,
        "selected_layer": int(selected_layer),
        "batch_work_keys": batch_sites["work_key"].astype(str).tolist(),
        "batch_rows": int(len(batch_sites)),
    }


def _manifest_artifacts(root: Path, names: Sequence[str]) -> dict[str, dict[str, object]]:
    artifacts = {}
    for name in names:
        path = root / name
        metadata: dict[str, object] = {
            "sha256": sha256_file(path), "bytes": path.stat().st_size
        }
        if path.suffix == ".parquet":
            frame = read_table(path)
            metadata["rows"] = int(len(frame))
            metadata["work_keys"] = int(frame["work_key"].nunique()) if "work_key" in frame else 0
            if {
                "reader_target_evaluable", "reader_target_ineligibility_reason",
            }.issubset(frame.columns):
                metadata["eligibility"] = _eligibility_receipt(
                    _target_state_from_scores(frame)
                    if "reader_name" in frame else frame
                )
        artifacts[name] = metadata
    return artifacts


def verify_run_root(
    root: str | Path,
    *,
    expected_run_id: str,
    mode: str,
) -> dict[str, object]:
    run_root = Path(root)
    identity = json.loads((run_root / "semantic_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    expected_status = "canary_complete" if mode == "canary" else "content_readout_complete"
    if mode not in {"canary", "complete"} or manifest.get("status") != expected_status:
        raise RuntimeError("candidate run has not reached the requested terminal state")
    if (
        identity.get("semantic_run_id") != expected_run_id
        or manifest.get("semantic_identity") != identity
    ):
        raise RuntimeError("candidate run semantic identity mismatch")
    if bool(manifest.get("canary")) != (mode == "canary"):
        raise RuntimeError("candidate run mode mismatch")
    required = {
        "semantic_identity.json", "frozen_selection.json", "gate_report.json",
        "ranker_manifest.json", "work_plan.json", "layer_select_scores.parquet",
        "layer_select_control_scores.parquet", "training_target_state.parquet",
    }
    if bool(manifest.get("reader_gate_opened")):
        required.update({
            "reader_gate_scores.parquet", "reader_gate_control_scores.parquet",
            "reader_gate_random_scores.parquet", "parity_report.json",
        })
    declared = set(map(str, manifest.get("artifacts", {})))
    if not required.issubset(declared):
        raise RuntimeError(f"candidate run is missing required artifacts: {sorted(required - declared)}")
    for name, metadata in manifest.get("artifacts", {}).items():
        path = run_root / str(name)
        if not path.is_file() or sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"candidate artifact checksum mismatch: {name}")
        if path.stat().st_size != int(metadata.get("bytes", -1)):
            raise RuntimeError(f"candidate artifact size mismatch: {name}")
        if path.suffix == ".parquet":
            frame = read_table(path)
            if len(frame) != int(metadata.get("rows", -1)):
                raise RuntimeError(f"candidate artifact row count mismatch: {name}")
            if name == "training_target_state.parquet":
                if (
                    tuple(frame.columns) != tuple(CANDIDATE_TARGET_STATE_COLUMNS)
                    or frame["work_key"].duplicated().any()
                    or not frame["work_key"].astype(str).is_monotonic_increasing
                    or _eligibility_receipt(frame) != metadata.get("eligibility")
                ):
                    raise RuntimeError("candidate training target state mismatch")
                continue
            identity_columns = [
                column for column in ("reader_name", "l2", "layer", "work_key")
                if column in frame
            ]
            if len(identity_columns) < 4 or frame.duplicated(identity_columns).any():
                raise RuntimeError(f"candidate artifact work identity mismatch: {name}")
            sorted_frame = frame.sort_values(identity_columns, kind="mergesort").reset_index(drop=True)
            if not frame.reset_index(drop=True).equals(sorted_frame):
                raise RuntimeError(f"candidate artifact ordering mismatch: {name}")
            role = "reader_gate" if name.startswith("reader_gate") else "layer_select"
            if not frame["readout_role"].astype(str).eq(role).all():
                raise RuntimeError(f"candidate artifact role mismatch: {name}")
            if int(frame["work_key"].nunique()) != int(metadata.get("work_keys", -1)):
                raise RuntimeError(f"candidate artifact work-key count mismatch: {name}")
            state = _target_state_from_scores(frame)
            if _eligibility_receipt(state) != metadata.get("eligibility"):
                raise RuntimeError(f"candidate artifact target state mismatch: {name}")
    plan_path = run_root / "work_plan.json"
    if not plan_path.is_file():
        raise RuntimeError("candidate work plan is missing")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selection = json.loads((run_root / "frozen_selection.json").read_text(encoding="utf-8"))
    config_l2 = [float(value) for value in identity.get("experiment_config", {}).get("l2_grid", [])]
    if (
        plan.get("schema_version") != 2
        or plan.get("semantic_sha256") != identity.get("semantic_sha256")
        or bool(plan.get("reader_gate_opened")) != bool(manifest.get("reader_gate_opened"))
        or plan.get("role_items") != manifest.get("role_items")
        or int(plan.get("layer_count", -1)) != int(identity.get("model", {}).get("expected_layers", -2))
        or [float(value) for value in plan.get("l2_grid", [])] != config_l2
        or int(plan.get("selected_layer", -1)) != int(selection.get("selected_layer", -2))
        or not np.isclose(float(plan.get("selected_l2", -1)), float(selection.get("selected_l2", -2)))
        or plan.get("training_eligibility") != manifest.get("training_eligibility")
        or plan.get("training_eligibility") != selection.get("training_eligibility")
        or plan.get("role_eligibility") != manifest.get("target_evaluability_by_role")
    ):
        raise RuntimeError("candidate work plan identity mismatch")
    training_state = read_table(run_root / "training_target_state.parquet")
    training_receipt = _eligibility_receipt(training_state)
    if (
        training_receipt != plan.get("training_eligibility")
        or int(training_receipt["total_rows"]) != int(manifest.get("training_pool_items", -1))
        or int(training_receipt["evaluable_rows"])
        != int(manifest.get("training_evaluable_items", -1))
    ):
        raise RuntimeError("candidate training eligibility does not match expected work")
    environment = identity.get("experiment_config", {}).get("runtime_environment", {})
    required_environment = {
        "device_type", "gpu_name", "gpu_compute_capability", "bf16_supported",
        "attention_backend", "torch_version", "cuda_version", "transformers_version",
        "tokenizers_version", "inference_dtype", "fit_dtype",
    }
    if (
        set(environment) != required_environment
        or environment.get("attention_backend") != "sdpa"
        or environment.get("fit_dtype") != "float32"
        or (
            environment.get("device_type") == "cuda"
            and (
                not environment.get("gpu_name")
                or not environment.get("gpu_compute_capability")
                or environment.get("bf16_supported") is not True
                or environment.get("inference_dtype") != "bfloat16"
            )
        )
    ):
        raise RuntimeError("candidate runtime environment identity mismatch")
    expected_frames = {
        "layer_select_scores.parquet": {
            "role": "layer_select", "readers": {"content"},
            "rows_per_work": int(plan["layer_count"]) * len(config_l2),
        },
        "layer_select_control_scores.parquet": {
            "role": "layer_select", "readers": set(CONTROL_READERS),
            "rows_per_work": len(CONTROL_READERS),
        },
    }
    if bool(plan["reader_gate_opened"]):
        expected_frames.update({
            "reader_gate_scores.parquet": {
                "role": "reader_gate", "readers": {"content"}, "rows_per_work": 1,
            },
            "reader_gate_control_scores.parquet": {
                "role": "reader_gate", "readers": set(CONTROL_READERS),
                "rows_per_work": len(CONTROL_READERS),
            },
            "reader_gate_random_scores.parquet": {
                "role": "reader_gate", "readers": {f"random_{index}" for index in range(3)},
                "rows_per_work": 3,
            },
        })
    for name, expectation in expected_frames.items():
        frame = read_table(run_root / name)
        role_plan = plan.get("roles", {}).get(expectation["role"], {})
        observed_summary = _work_key_summary(frame)
        if (
            observed_summary["work_keys"] != int(role_plan.get("work_keys", -1))
            or observed_summary["work_keys_sha256"] != role_plan.get("work_keys_sha256")
            or len(frame) != int(role_plan.get("rows", -1)) * int(expectation["rows_per_work"])
            or set(frame["reader_name"].astype(str)) != expectation["readers"]
        ):
            raise RuntimeError(f"candidate artifact does not match expected work: {name}")
        observed_eligibility = _eligibility_receipt(_target_state_from_scores(frame))
        if observed_eligibility != plan["role_eligibility"].get(expectation["role"]):
            raise RuntimeError(f"candidate artifact eligibility does not match expected work: {name}")
    ranker_manifest_path = run_root / "ranker_manifest.json"
    if ranker_manifest_path.exists():
        ranker_manifest = json.loads(ranker_manifest_path.read_text(encoding="utf-8"))
        if ranker_manifest.get("semantic_sha256") != identity.get("semantic_sha256"):
            raise RuntimeError("candidate ranker manifest identity mismatch")
        ranker_receipts = ranker_manifest.get("rankers", {})
        if set(map(str, ranker_receipts)) != _expected_ranker_names(plan):
            raise RuntimeError("candidate ranker set does not match expected work")
        for name, digest in ranker_receipts.items():
            path = run_root / "rankers" / str(name)
            if not path.is_file() or sha256_file(path) != digest:
                raise RuntimeError(f"candidate ranker checksum mismatch: {name}")
        if selection.get("ranker_sha256") != ranker_receipts:
            raise RuntimeError("candidate frozen selection ranker hashes mismatch")
    expected_parity_policy = {
        "schema_version": 2,
        "maximum_activation_difference": PARITY_MAX_DIFFERENCE,
        "maximum_raw_log_probability_difference": PARITY_MAX_DIFFERENCE,
        "maximum_reader_probability_difference": PARITY_MAX_DIFFERENCE,
        "categorical_disagreement_rule": "both_margins_lte_twice_observed_coordinate_difference",
    }
    expected_source_keys = {
        "source_bundle_manifest_sha256", "source_readout_sha256",
        "source_applicability_sha256", "option_audit_manifest_sha256",
    }
    if (
        selection.get("target_policy_version") != TARGET_POLICY_VERSION
        or set(selection.get("selected_control_l2", {}))
        != {reader for reader in CONTROL_READERS if reader != "majority"}
        or len(selection.get("majority_training_distribution", [])) != 4
        or not np.isclose(sum(selection.get("majority_training_distribution", [])), 1.0)
        or selection.get("random_reader_seeds") != [
            int(identity.get("experiment_config", {}).get("seed", 0)) + 1000 + index
            for index in range(3)
        ]
        or set(selection.get("prepared_source_hashes", {})) != expected_source_keys
        or selection.get("prepared_source_hashes")
        != identity.get("experiment_config", {}).get("prepared_source_hashes")
        or selection.get("layer_select_eligibility")
        != plan.get("role_eligibility", {}).get("layer_select")
        or (
            bool(manifest.get("reader_gate_opened"))
            and selection.get("reader_gate_eligibility")
            != plan.get("role_eligibility", {}).get("reader_gate")
        )
        or (
            bool(manifest.get("reader_gate_opened"))
            and selection.get("parity_policy") != expected_parity_policy
        )
    ):
        raise RuntimeError("candidate frozen selection policy mismatch")
    if bool(manifest.get("reader_gate_opened")):
        parity = json.loads((run_root / "parity_report.json").read_text(encoding="utf-8"))
        if (
            parity.get("schema_version") != 2
            or parity.get("semantic_sha256") != identity.get("semantic_sha256")
            or int(parity.get("selected_layer", -1)) != int(selection.get("selected_layer", -2))
            or any(
                not np.isclose(
                    float(parity.get(key, {}).get("maximum_difference", -1)),
                    PARITY_MAX_DIFFERENCE,
                )
                for key in ("batch_reader_categorical", "batch_raw_categorical")
            )
            or bool(parity.get("batch"))
            != bool(
                parity.get("batch_reader_categorical", {}).get("passes")
                and parity.get("batch_raw_categorical", {}).get("passes")
                and float(parity.get("max_batch_activation_difference", float("inf")))
                <= PARITY_MAX_DIFFERENCE
            )
        ):
            raise RuntimeError("candidate parity report violates the frozen policy")
    gate = json.loads((run_root / "gate_report.json").read_text(encoding="utf-8"))
    if mode == "canary" and not bool(manifest.get("reader_gate_opened")):
        raise RuntimeError("candidate canary did not reach the reader gate")
    if (
        gate.get("claim") != "candidate_local_linear_decodability"
        or bool(gate.get("reader_gate_opened", True)) != bool(manifest.get("reader_gate_opened"))
        or (
            not bool(manifest.get("reader_gate_opened"))
            and (
                mode != "complete"
                or gate.get("selection_eligible") is not False
                or selection.get("selection_eligible") is not False
                or not gate.get("stop_reason")
                or gate.get("stop_reason") != selection.get("stop_reason")
            )
        )
        or int(gate.get("selected_layer", selection.get("selected_layer", -1)))
        != int(selection.get("selected_layer", -2))
        or not np.isclose(
            float(gate.get("selected_l2", selection.get("selected_l2", -1))),
            float(selection.get("selected_l2", -2)),
        )
        or bool(gate.get("patch_eligible"))
        or bool(manifest.get("patch_eligible"))
        or bool(gate.get("opens_final_confirmation")) != bool(gate.get("tier_2_pass"))
        or bool(manifest.get("final_confirmation_opened"))
        or bool(manifest.get("tier_1_pass")) != bool(gate.get("tier_1_pass"))
        or bool(manifest.get("tier_2_pass")) != bool(gate.get("tier_2_pass"))
    ):
        raise RuntimeError("candidate run violates the frozen scientific claim boundary")
    if mode == "complete" and {
        str(key): int(value) for key, value in manifest.get("role_items", {}).items()
    } != ROLE_ITEM_COUNTS:
        raise RuntimeError("candidate run changed the frozen discovery split")
    return manifest


def cmd_verify(args) -> None:
    manifest = verify_run_root(
        args.run_root, expected_run_id=args.run_id, mode=args.mode
    )
    print(json.dumps({
        "status": manifest["status"],
        "tier_1_pass": manifest["tier_1_pass"],
        "tier_2_pass": manifest["tier_2_pass"],
    }, sort_keys=True))


def _stable_score_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["reader_name", "l2", "layer", "work_key"]
    if missing := set(columns) - set(frame.columns):
        raise ValueError(f"candidate score frame is missing sort columns: {sorted(missing)}")
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def _work_key_summary(frame: pd.DataFrame) -> dict[str, object]:
    keys = sorted(frame["work_key"].astype(str).unique())
    return {
        "rows": int(len(frame)),
        "work_keys": int(len(keys)),
        "work_keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }


def _write_work_plan(
    root: Path,
    sites: pd.DataFrame,
    *,
    semantic_sha256: str,
    layer_count: int,
    l2_grid: Sequence[float],
    reader_gate_opened: bool,
    selected_layer: int | None,
    selected_l2: float | None,
    training_eligibility: Mapping[str, object],
    role_eligibility: Mapping[str, Mapping[str, object]],
) -> None:
    roles = {
        role: _work_key_summary(
            sites[sites["readout_role"].astype(str).eq(role)]
        )
        for role in ("layer_select", "reader_gate")
    }
    _atomic_json({
        "schema_version": 2,
        "semantic_sha256": semantic_sha256,
        "reader_gate_opened": bool(reader_gate_opened),
        "role_items": _role_items(sites),
        "roles": roles,
        "layer_count": int(layer_count),
        "l2_grid": [float(value) for value in l2_grid],
        "selected_layer": selected_layer,
        "selected_l2": selected_l2,
        "training_eligibility": dict(training_eligibility),
        "role_eligibility": {
            str(role): dict(receipt) for role, receipt in role_eligibility.items()
        },
    }, root / "work_plan.json")


def _ranker_name(prefix: str, l2: float) -> str:
    return f"{prefix}-l2-{float(l2):.0e}.npz"


def _expected_ranker_names(plan: Mapping[str, object]) -> set[str]:
    l2_grid = [float(value) for value in plan["l2_grid"]]
    names = {_ranker_name("content", l2) for l2 in l2_grid}
    for reader in CONTROL_READERS:
        if reader != "majority":
            names.update(_ranker_name(f"control-{reader}", l2) for l2 in l2_grid)
    if bool(plan["reader_gate_opened"]):
        selected_l2 = float(plan["selected_l2"])
        names.update(_ranker_name(f"random-{index}", selected_l2) for index in range(3))
    return names


def cmd_run_model(args) -> None:
    prepared_manifest, all_sites = _load_prepared_bundle(args.bundle)
    profile = get_model_profile(args.profile)
    source_model = prepared_manifest["source_model"]
    if (
        source_model.get("id") != profile.model_id
        or source_model.get("revision") != profile.revision
        or source_model.get("slug") != profile.slug
    ):
        raise ValueError("CLI-selected model does not match the prepared source model")
    canary = args.canary_items is not None
    sites = (
        _select_canary_sites(all_sites, items_per_role=args.canary_items, seed=args.seed)
        if canary else all_sites
    )
    runtime_environment = _runtime_environment_receipt(
        attention_backend=profile.attention_backend
    )
    prepared_source_hashes = {
        key: prepared_manifest[key]
        for key in (
            "source_bundle_manifest_sha256", "source_readout_sha256",
            "source_applicability_sha256", "option_audit_manifest_sha256",
        )
    }
    config = {
        "experiment": "decision_binding_candidate_local_content_v3",
        "prepared_manifest_sha256": sha256_file(Path(args.bundle) / "prepared_manifest.json"),
        "prepared_source_hashes": prepared_source_hashes,
        "token_position": "final_full_token_within_audited_candidate_payload",
        "l2_grid": list(map(float, args.l2_grid)),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutation_samples": int(args.permutation_samples),
        "canary_items": args.canary_items,
        "batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "capture_chunk_size": int(args.capture_chunk_size),
        "max_iter": int(args.max_iter),
        "target_policy_version": TARGET_POLICY_VERSION,
        "runtime_environment": runtime_environment,
    }
    identity = build_semantic_identity(
        profile,
        config=config,
        dataset_path=Path(args.bundle) / "candidate_sites.parquet",
        source_paths=[
            *default_semantic_source_paths(PROJECT_ROOT),
            PROJECT_ROOT / "requirements-gpu.lock",
        ],
    )
    root = Path(args.output_base) / profile.slug / identity.semantic_run_id
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "semantic_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity.as_dict():
        raise RuntimeError("candidate run-root semantic identity mismatch")
    _atomic_json(identity.as_dict(), identity_path)
    started = time.monotonic()
    previous_manifest = root / "run_manifest.json"
    _atomic_json({
        "status": "starting",
        "profile": args.profile,
        "semantic_identity": identity.as_dict(),
        "canary": canary,
    }, previous_manifest)

    stop = StopState()
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, stop.request)

    def progress(phase: str, completed: int, total: int) -> None:
        elapsed = max(time.monotonic() - started, 1e-9)
        _atomic_json({
            "phase": phase,
            "completed": int(completed),
            "total": int(total),
            "elapsed_seconds": elapsed,
            "units_per_second": completed / elapsed,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }, root / "progress.json")

    try:
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
        if not args.allow_cpu and device.type != "cuda":
            raise RuntimeError("candidate-local model runs require CUDA")
        if device.type != runtime_environment["device_type"]:
            raise RuntimeError("candidate-local loaded device differs from bound runtime identity")
        if device.type == "cuda" and (
            not torch.cuda.is_bf16_supported()
            or next(model.parameters()).dtype != torch.bfloat16
        ):
            raise RuntimeError("candidate-local CUDA runs require actual BF16 model weights")
        training_pool = sites[
            sites["readout_role"].astype(str).eq("probe_train")
            & sites["manipulation"].astype(str).eq("controlled_baseline")
            & sites["wrapper_name"].astype(str).eq("plain")
        ].sort_values("work_key", kind="mergesort")
        if training_pool["item_id"].duplicated().any() or (
            not canary and len(training_pool) != 1801
        ):
            raise RuntimeError("candidate training rows do not match the frozen plain baseline")
        captured_training = _capture_training(
            root, training_pool, model, tokenizer,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            chunk_size=args.capture_chunk_size,
            stop=stop,
            progress=progress,
        )
        if captured_training is None:
            _atomic_json({
                "status": "interrupted", "phase": "training_capture",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        all_activations, training_state = captured_training
        training_eligibility_receipt = _eligibility_receipt(training_state)
        write_table_atomic(
            training_state.sort_values("work_key", kind="mergesort").reset_index(drop=True),
            root / "training_target_state.parquet",
        )
        training, activations, targets = _eligible_training_inputs(
            training_pool, all_activations, training_state
        )
        rankers = _fit_rankers(
            root, activations, targets,
            l2_grid=args.l2_grid,
            semantic_sha256=identity.semantic_sha256,
            prefix="content",
            max_iter=args.max_iter,
            fit_device=device,
            should_stop=lambda: stop.requested,
        )
        if stop.requested or set(rankers) != set(map(float, args.l2_grid)):
            _atomic_json({
                "status": "interrupted", "phase": "content_fit",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        train_features = metadata_candidate_features(training)
        control_rankers = {
            name: _fit_rankers(
                root,
                torch.from_numpy(features),
                targets,
                l2_grid=args.l2_grid,
                semantic_sha256=identity.semantic_sha256,
                prefix=f"control-{name}",
                max_iter=args.max_iter,
                fit_device=device,
                should_stop=lambda: stop.requested,
            )
            for name, features in train_features.items()
        }
        if stop.requested or any(
            set(fitted) != set(map(float, args.l2_grid)) for fitted in control_rankers.values()
        ):
            _atomic_json({
                "status": "interrupted", "phase": "control_fit",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        layer_sites = sites[
            sites["readout_role"].astype(str).eq("layer_select")
        ].copy()
        layer_scores, complete = _score_model_phase(
            root, "layer_select", layer_sites, rankers, model, tokenizer,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            chunk_size=args.capture_chunk_size,
            stop=stop,
            progress=progress,
        )
        if not complete:
            _atomic_json({
                "status": "interrupted", "phase": "layer_select",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        layer_scores = _stable_score_frame(layer_scores)
        layer_state = _target_state_from_scores(layer_scores)
        layer_eligibility_receipt = _eligibility_receipt(layer_state)
        layer_sites_with_state = _join_target_state(layer_sites, layer_state)
        layer_controls = {
            name: _best_control(_score_metadata_rankers(layer_sites_with_state, fitted, name))
            for name, fitted in control_rankers.items()
        }
        layer_controls["majority"] = majority_candidate_scores(
            layer_sites_with_state, training["actual_winner_content_id"].to_numpy(dtype=int)
        )
        selection_stop = False
        try:
            selection = select_candidate_reader(
                layer_scores,
                layer_controls,
                allow_ineligible=canary,
            )
        except ValueError as error:
            if "beats both isolated nuisance controls" not in str(error):
                raise
            selection = select_candidate_reader(
                layer_scores, layer_controls, allow_ineligible=True
            )
            selection["stop_reason"] = "no_reader_beat_both_isolated_nuisance_controls"
            selection_stop = True
        selection_payload = {
            **selection,
            "semantic_sha256": identity.semantic_sha256,
            "prepared_manifest_sha256": config["prepared_manifest_sha256"],
            "token_position": config["token_position"],
            "l2_grid": config["l2_grid"],
            "seed": args.seed,
            "target_policy_version": TARGET_POLICY_VERSION,
            "training_eligibility": training_eligibility_receipt,
            "layer_select_eligibility": layer_eligibility_receipt,
            "selected_control_l2": {
                name: float(frame.iloc[0]["l2"])
                for name, frame in layer_controls.items()
                if name != "majority"
            },
            "majority_training_distribution": (
                np.bincount(
                    training["actual_winner_content_id"].to_numpy(dtype=int), minlength=4
                ) / len(training)
            ).tolist(),
            "random_reader_seeds": [args.seed + 1000 + index for index in range(3)],
            "prepared_source_hashes": prepared_source_hashes,
            "final_confirmation_opened": False,
            "ranker_sha256": {
                path.name: sha256_file(path) for path in sorted((root / "rankers").glob("*.npz"))
            },
        }
        _atomic_json(selection_payload, root / "frozen_selection.json")
        _write_work_plan(
            root,
            sites,
            semantic_sha256=identity.semantic_sha256,
            layer_count=profile.expected_layers,
            l2_grid=args.l2_grid,
            reader_gate_opened=not selection_stop,
            selected_layer=int(selection["selected_layer"]),
            selected_l2=float(selection["selected_l2"]),
            training_eligibility=training_eligibility_receipt,
            role_eligibility={"layer_select": layer_eligibility_receipt},
        )

        if selection_stop:
            report = {
                "schema_version": 1,
                "claim": "candidate_local_linear_decodability",
                "selection_eligible": False,
                "stop_reason": selection["stop_reason"],
                "tier_1_pass": False,
                "tier_2_pass": False,
                "content_reader_usable": False,
                "patch_eligible": False,
                "opens_final_confirmation": False,
                "reader_gate_opened": False,
                "selected_layer": int(selection["selected_layer"]),
                "selected_l2": float(selection["selected_l2"]),
            }
            write_table_atomic(layer_scores, root / "layer_select_scores.parquet")
            write_table_atomic(
                _stable_score_frame(pd.concat(layer_controls.values(), ignore_index=True)),
                root / "layer_select_control_scores.parquet",
            )
            _atomic_json(report, root / "gate_report.json")
            _atomic_json({
                "schema_version": 1,
                "semantic_sha256": identity.semantic_sha256,
                "rankers": {
                    path.name: sha256_file(path)
                    for path in sorted((root / "rankers").glob("*.npz"))
                },
            }, root / "ranker_manifest.json")
            artifact_names = (
                "semantic_identity.json", "frozen_selection.json", "gate_report.json",
                "ranker_manifest.json", "work_plan.json", "layer_select_scores.parquet",
                "layer_select_control_scores.parquet", "training_target_state.parquet",
            )
            _atomic_json({
                "status": "content_readout_complete",
                "profile": args.profile,
                "semantic_identity": identity.as_dict(),
                "canary": False,
                "role_items": _role_items(sites),
                "training_pool_items": int(training_pool["item_id"].nunique()),
                "training_evaluable_items": int(training["item_id"].nunique()),
                "training_eligibility": training_eligibility_receipt,
                "target_evaluability_by_role": {
                    "layer_select": layer_eligibility_receipt,
                },
                "tier_1_pass": False,
                "tier_2_pass": False,
                "content_reader_usable": False,
                "patch_eligible": False,
                "final_confirmation_opened": False,
                "reader_gate_opened": False,
                "stop_reason": selection["stop_reason"],
                "artifacts": _manifest_artifacts(root, artifact_names),
                "wall_seconds": time.monotonic() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available() else 0,
            }, previous_manifest)
            return

        selected_l2 = float(selection["selected_l2"])
        selected_layer = int(selection["selected_layer"])
        gate_sites = sites[
            sites["readout_role"].astype(str).eq("reader_gate")
        ].copy()
        random_rankers: dict[str, CandidateRanker] = {}
        for random_index in range(3):
            rng = np.random.default_rng(args.seed + 1000 + random_index)
            random_targets = (targets + rng.integers(0, 4, size=len(targets))) % 4
            fitted_random = _fit_rankers(
                root, activations, random_targets,
                l2_grid=[selected_l2],
                semantic_sha256=identity.semantic_sha256,
                prefix=f"random-{random_index}",
                max_iter=args.max_iter,
                fit_device=device,
                should_stop=lambda: stop.requested,
            )
            if selected_l2 not in fitted_random:
                break
            random_rankers[f"random_{random_index}"] = fitted_random[selected_l2]
        if stop.requested:
            _atomic_json({
                "status": "interrupted", "phase": "random_fit",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        all_gate_scores, complete = _score_model_phase(
            root, "reader_gate", gate_sites, {selected_l2: rankers[selected_l2]}, model, tokenizer,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            chunk_size=args.capture_chunk_size,
            stop=stop,
            progress=progress,
            selected_layer=selected_layer,
            extra_rankers=random_rankers,
        )
        if not complete:
            _atomic_json({
                "status": "interrupted", "phase": "reader_gate",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        all_gate_scores = _stable_score_frame(all_gate_scores)
        gate_state = _target_state_from_scores(all_gate_scores)
        gate_eligibility_receipt = _eligibility_receipt(gate_state)
        gate_sites_with_state = _join_target_state(gate_sites, gate_state)
        parity_report = _run_parity_checks(
            root,
            rankers[selected_l2],
            activations,
            targets,
            model,
            tokenizer,
            gate_sites_with_state,
            identity=identity,
            batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens,
            max_iter=args.max_iter,
            fit_device=device,
            seed=args.seed,
            selected_layer=selected_layer,
        )
        _atomic_json(parity_report, root / "parity_report.json")
        selection_payload.update({
            "reader_gate_eligibility": gate_eligibility_receipt,
            "parity_policy": {
                "schema_version": 2,
                "maximum_activation_difference": PARITY_MAX_DIFFERENCE,
                "maximum_raw_log_probability_difference": PARITY_MAX_DIFFERENCE,
                "maximum_reader_probability_difference": PARITY_MAX_DIFFERENCE,
                "categorical_disagreement_rule": "both_margins_lte_twice_observed_coordinate_difference",
            },
            "ranker_sha256": {
                path.name: sha256_file(path) for path in sorted((root / "rankers").glob("*.npz"))
            },
        })
        _atomic_json(selection_payload, root / "frozen_selection.json")
        _write_work_plan(
            root,
            sites,
            semantic_sha256=identity.semantic_sha256,
            layer_count=profile.expected_layers,
            l2_grid=args.l2_grid,
            reader_gate_opened=True,
            selected_layer=selected_layer,
            selected_l2=selected_l2,
            training_eligibility=training_eligibility_receipt,
            role_eligibility={
                "layer_select": layer_eligibility_receipt,
                "reader_gate": gate_eligibility_receipt,
            },
        )
        if stop.requested:
            _atomic_json({
                "status": "interrupted", "phase": "parity",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        gate_scores = all_gate_scores[all_gate_scores["reader_name"].astype(str) == "content"].copy()
        random_frames = [
            all_gate_scores[all_gate_scores["reader_name"].astype(str) == name].copy()
            for name in sorted(random_rankers)
        ]
        gate_controls = _score_selected_metadata_controls(
            gate_sites_with_state, layer_controls, control_rankers
        )
        gate_controls["majority"] = majority_candidate_scores(
            gate_sites_with_state, training["actual_winner_content_id"].to_numpy(dtype=int)
        )
        bootstrap = min(args.bootstrap_samples, 300) if canary else args.bootstrap_samples
        permutations = min(args.permutation_samples, 100) if canary else args.permutation_samples
        report = gate_candidate_reader(
            gate_scores,
            selection,
            gate_controls,
            random_frames,
            bootstrap_samples=bootstrap,
            permutation_samples=permutations,
            seed=args.seed,
            parity={key: bool(parity_report[key]) for key in ("save_load", "batch", "seed")},
        )
        if stop.requested:
            _atomic_json({
                "status": "interrupted", "phase": "gate_statistics",
                "semantic_identity": identity.as_dict(), "stop_signal": stop.signal_name,
            }, previous_manifest)
            return
        write_table_atomic(layer_scores, root / "layer_select_scores.parquet")
        write_table_atomic(gate_scores, root / "reader_gate_scores.parquet")
        write_table_atomic(
            _stable_score_frame(pd.concat(layer_controls.values(), ignore_index=True)),
            root / "layer_select_control_scores.parquet",
        )
        write_table_atomic(
            _stable_score_frame(pd.concat(gate_controls.values(), ignore_index=True)),
            root / "reader_gate_control_scores.parquet",
        )
        write_table_atomic(
            _stable_score_frame(pd.concat(random_frames, ignore_index=True)),
            root / "reader_gate_random_scores.parquet",
        )
        _atomic_json(report, root / "gate_report.json")
        _atomic_json({
            "schema_version": 1,
            "semantic_sha256": identity.semantic_sha256,
            "rankers": {
                path.name: sha256_file(path) for path in sorted((root / "rankers").glob("*.npz"))
            },
        }, root / "ranker_manifest.json")
        artifact_names = (
            "semantic_identity.json", "frozen_selection.json", "gate_report.json",
            "ranker_manifest.json", "work_plan.json", "parity_report.json", "layer_select_scores.parquet", "reader_gate_scores.parquet",
            "layer_select_control_scores.parquet", "reader_gate_control_scores.parquet",
            "reader_gate_random_scores.parquet", "training_target_state.parquet",
        )
        _atomic_json({
            "status": "canary_complete" if canary else "content_readout_complete",
            "profile": args.profile,
            "semantic_identity": identity.as_dict(),
            "canary": canary,
            "role_items": _role_items(sites),
            "training_pool_items": int(training_pool["item_id"].nunique()),
            "training_evaluable_items": int(training["item_id"].nunique()),
            "training_eligibility": training_eligibility_receipt,
            "target_evaluability_by_role": {
                "layer_select": layer_eligibility_receipt,
                "reader_gate": gate_eligibility_receipt,
            },
            "tier_1_pass": bool(report["tier_1_pass"]),
            "tier_2_pass": bool(report["tier_2_pass"]),
            "content_reader_usable": bool(report["content_reader_usable"]),
            "patch_eligible": False,
            "final_confirmation_opened": False,
            "reader_gate_opened": True,
            "artifacts": _manifest_artifacts(root, artifact_names),
            "wall_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        }, previous_manifest)
    except BaseException as error:
        _atomic_json({
            "status": "failed",
            "profile": args.profile,
            "semantic_identity": identity.as_dict(),
            "failure": {"type": type(error).__name__, "message": str(error)},
            "wall_seconds": time.monotonic() - started,
            "stop_signal": stop.signal_name,
        }, previous_manifest)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface-formatting-decision-content")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--bundle", required=True)
    prepare.add_argument("--applicability", required=True)
    prepare.add_argument("--option-audit", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=cmd_prepare)
    run = sub.add_parser("run-model")
    run.add_argument("--profile", choices=sorted(MODEL_PROFILES), required=True)
    run.add_argument("--bundle", required=True)
    run.add_argument("--output-base", default="results/decision_binding_content_runs")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-batch-tokens", type=int, default=40000)
    run.add_argument("--capture-chunk-size", type=int, default=64)
    run.add_argument("--max-iter", type=int, default=100)
    run.add_argument("--l2-grid", type=float, nargs="+", default=list(L2_GRID))
    run.add_argument("--bootstrap-samples", type=int, default=5000)
    run.add_argument("--permutation-samples", type=int, default=1000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--canary-items", type=int)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--allow-cpu", action="store_true", help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run_model)
    verify = sub.add_parser("verify")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--mode", choices=("canary", "complete"), required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "capture_chunk_size", 1) <= 0:
        raise ValueError("capture chunk size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
