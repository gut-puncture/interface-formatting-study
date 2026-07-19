from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .decision_binding_logit_lens import (
    CandidateContinuation,
    ContinuationAudit,
    PARITY_ATOL,
    audit_fixed_root_continuations,
    score_candidate_paths_cached_many,
    score_candidate_paths_scalar,
)
from .model_loader import load_model_and_tokenizer
from .model_profiles import ModelProfile, get_model_profile
from .run_identity import (
    SemanticIdentity,
    build_semantic_identity,
    default_semantic_source_paths,
    sha256_file,
)
from .shards import ShardStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LETTER_INSTRUCTION = "Return only the letter (A, B, C, or D)."
TEXT_INSTRUCTION = "Return only the exact answer text, not its letter."
EXPECTED_FORMATS = (
    "plain",
    "csv_inline",
    "graphql_query",
    "html_form",
    "ini_file",
    "key_equals",
    "protobuf_msg",
    "shell_heredoc",
    "toml_config",
)
EXPECTED_SPLIT_ITEMS = {"train": 1801, "validation": 600}
MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
MISTRAL_SLUG = "mistral-7b-instruct-v0.3"
CLAIM_BOUNDARY = "descriptive_layerwise_logit_lens_not_causal"
TOKEN_AUDIT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
CROSS_FORWARD_SENSITIVITY_POLICY = "record_only_bf16_execution_shape_sensitivity"
EXPECTED_ANALYSIS_ARTIFACTS = {
    "analysis_summary": "analysis/analysis_summary.json",
    "interpretation_memo": "analysis/interpretation_memo.md",
    "quality_gates": "analysis/quality_gates.json",
    "calibrated_letter": "analysis/calibrated_letter.csv",
    "diagnostics": "analysis/diagnostics.csv",
    "primary_contrasts": "analysis/primary_contrasts.csv",
    "secondary_contrasts": "analysis/secondary_contrasts.csv",
    "trajectories": "analysis/trajectories.csv",
    "timing": "analysis/timing.csv",
    "timing_distributions": "analysis/timing_distributions.csv",
    "trajectory_figure": "analysis/trajectory_2x2.png",
    "strata": "analysis/strata.csv",
}


class GpuPhaseTelemetry:
    """Small task-local sampler for the root/branch execution phases."""

    def __init__(self, semantic_run_id: str, *, sample_interval_seconds: float = 0.1):
        self.semantic_run_id = semantic_run_id
        self.sample_interval_seconds = sample_interval_seconds
        self._phase = "idle"
        self._phase_started = time.monotonic()
        self._phase_seconds = {"root": 0.0, "branch": 0.0, "scalar_oracle": 0.0}
        self._samples = {
            phase: [] for phase in ("root", "branch", "scalar_oracle")
        }
        self._sample_errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        if phase not in {"idle", "root", "branch", "scalar_oracle"}:
            raise ValueError(f"unknown telemetry phase: {phase}")
        now = time.monotonic()
        with self._lock:
            self._accrue_locked(now)
            self._phase = phase
            self._phase_started = now

    def _accrue_locked(self, now: float) -> None:
        if self._phase in self._phase_seconds:
            self._phase_seconds[self._phase] += max(now - self._phase_started, 0.0)

    @staticmethod
    def _gpu_utilization() -> float:
        if hasattr(torch.cuda, "utilization"):
            try:
                return float(torch.cuda.utilization(0))
            except Exception:
                pass
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(completed.stdout.splitlines()[0].strip())

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            with self._lock:
                phase = self._phase
            if phase not in self._samples:
                continue
            try:
                value = self._gpu_utilization()
                if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise ValueError("GPU utilization is outside [0, 100]")
                with self._lock:
                    self._samples[phase].append(value)
            except Exception:
                with self._lock:
                    self._sample_errors += 1

    def snapshot(
        self,
        *,
        completed_work_units: int,
        root_input_tokens: int,
        branch_input_tokens: int,
        peak_vram_bytes: int,
    ) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            self._accrue_locked(now)
            self._phase_started = now
            seconds = dict(self._phase_seconds)
            samples = {phase: list(values) for phase, values in self._samples.items()}
            errors = self._sample_errors
        utilization = {}
        for phase, values in samples.items():
            utilization[phase] = {
                "samples": len(values),
                "mean_percent": None if not values else float(np.mean(values)),
                "max_percent": None if not values else float(max(values)),
            }
        return {
            "schema_version": 1,
            "semantic_run_id": self.semantic_run_id,
            "completed_work_units": int(completed_work_units),
            "root_input_tokens": int(root_input_tokens),
            "branch_input_tokens": int(branch_input_tokens),
            "peak_vram_bytes": int(peak_vram_bytes),
            "phase_seconds": seconds,
            "gpu_utilization": utilization,
            "sample_errors": int(errors),
        }

    def stop(self) -> None:
        self.set_phase("idle")
        self._stop.set()
        self._thread.join(timeout=3)


def _merge_telemetry_reports(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> dict[str, object]:
    phases = ("root", "branch", "scalar_oracle")
    previous_utilization = previous.get("gpu_utilization", {})
    current_utilization = current.get("gpu_utilization", {})
    utilization: dict[str, dict[str, object]] = {}
    for phase in phases:
        left = previous_utilization.get(phase, {})
        right = current_utilization.get(phase, {})
        left_count = int(left.get("samples", 0))
        right_count = int(right.get("samples", 0))
        count = left_count + right_count
        left_mean = 0.0 if left_count == 0 else float(left["mean_percent"])
        right_mean = 0.0 if right_count == 0 else float(right["mean_percent"])
        maxima = [
            float(value)
            for value in (left.get("max_percent"), right.get("max_percent"))
            if value is not None
        ]
        utilization[phase] = {
            "samples": count,
            "mean_percent": (
                None
                if count == 0
                else (left_count * left_mean + right_count * right_mean) / count
            ),
            "max_percent": None if not maxima else max(maxima),
        }
    previous_seconds = previous.get("phase_seconds", {})
    current_seconds = current.get("phase_seconds", {})
    semantic_ids = {
        str(value)
        for value in (
            previous.get("semantic_run_id"),
            current.get("semantic_run_id"),
        )
        if value not in {None, ""}
    }
    if len(semantic_ids) > 1:
        raise ValueError("telemetry reports have conflicting semantic identities")
    return {
        "schema_version": 2,
        "semantic_run_id": next(iter(semantic_ids), ""),
        "completed_work_units": int(current.get("completed_work_units", 0)),
        "covered_work_keys": sorted(
            set(map(str, previous.get("covered_work_keys", [])))
            | set(map(str, current.get("covered_work_keys", [])))
        ),
        "root_input_tokens": int(current.get("root_input_tokens", 0)),
        "branch_input_tokens": int(current.get("branch_input_tokens", 0)),
        "peak_vram_bytes": max(
            int(previous.get("peak_vram_bytes", 0)),
            int(current.get("peak_vram_bytes", 0)),
        ),
        "phase_seconds": {
            phase: float(previous_seconds.get(phase, 0.0))
            + float(current_seconds.get(phase, 0.0))
            for phase in phases
        },
        "gpu_utilization": utilization,
        "sample_errors": int(previous.get("sample_errors", 0))
        + int(current.get("sample_errors", 0)),
    }


def _validate_attempt_chain(
    receipts: Sequence[Mapping[str, object]],
    work_keys: Sequence[str],
    *,
    mode: str,
) -> None:
    ordered_keys = [str(value) for value in work_keys]
    advancing: list[Mapping[str, object]] = []
    cursor = 0
    for receipt in sorted(receipts, key=lambda value: float(value["started_at_unix"])):
        if receipt.get("status") not in {"interrupted", "startup_complete", "complete"}:
            continue
        before = int(receipt.get("completed_before", -1))
        after = int(receipt.get("completed_after", -1))
        processed = list(map(str, receipt.get("processed_work_keys", [])))
        telemetry_keys = list(map(str, receipt.get("telemetry_work_keys", [])))
        if before != cursor or after != before + len(processed):
            raise ValueError("attempt receipts do not form a deterministic resume chain")
        if processed != ordered_keys[before:after] or telemetry_keys != processed:
            raise ValueError("attempt receipts do not bind work and telemetry coverage")
        cursor = after
        advancing.append(receipt)
    if cursor != len(ordered_keys):
        raise ValueError("attempt receipts do not cover the verified work prefix")
    if mode == "startup" and len(ordered_keys) == 8:
        shape = [
            (
                int(receipt["completed_before"]),
                int(receipt["completed_after"]),
                str(receipt["status"]),
            )
            for receipt in advancing
        ]
        if shape != [(0, 4, "interrupted"), (4, 8, "startup_complete")]:
            raise ValueError("startup attempt receipts do not prove the frozen four-plus-four resume")


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(values: object) -> str:
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _as_four_strings(value: object, *, work_key: str) -> list[str]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        raise ValueError(f"candidate surfaces are malformed for {work_key}")
    if any(item is None or not isinstance(item, str) for item in value):
        raise ValueError(f"candidate surfaces are malformed for {work_key}")
    return [str(item) for item in value]


def _as_permutation(value: object, *, work_key: str) -> list[int]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        raise ValueError(f"content mapping is malformed for {work_key}")
    try:
        normalized = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"content mapping is malformed for {work_key}") from error
    if sorted(normalized) != [0, 1, 2, 3]:
        raise ValueError(f"content mapping is not a four-way permutation for {work_key}")
    return normalized


def _validate_prompt(
    row: Mapping[str, object],
    *,
    field: str,
    hash_field: str,
    require_answer_prefix: bool = True,
) -> str:
    prompt = row.get(field)
    if not isinstance(prompt, str):
        raise ValueError(f"{field} is missing for {row.get('work_key')}")
    if _sha256_text(prompt) != str(row.get(hash_field)):
        raise ValueError(f"{field} checksum mismatch for {row.get('work_key')}")
    if require_answer_prefix and not (
        prompt.endswith("\nAnswer:") or prompt.endswith("\nAnswer: ")
    ):
        raise ValueError(f"{field} lacks the exact terminal Answer prefix")
    return prompt


def _terminal_contract_matches(letter_prompt: str, text_prompt: str) -> bool:
    return (
        letter_prompt.count(LETTER_INSTRUCTION) == 1
        and text_prompt.count(TEXT_INSTRUCTION) == 1
        and letter_prompt.replace(LETTER_INSTRUCTION, TEXT_INSTRUCTION, 1)
        == text_prompt
    )


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _susceptibility_map(
    source: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    manipulation: str,
) -> dict[tuple[str, str], bool | None]:
    if "cal_predicted_content_id" not in source:
        return {}
    keys = ["item_id", "wrapper_name"]
    reference = baseline[keys + ["cal_predicted_content_id"]].rename(
        columns={"cal_predicted_content_id": "baseline_content_id"}
    )
    treatment = source[source["manipulation"].astype(str).eq(manipulation)][
        keys + ["cal_predicted_content_id"]
    ].merge(reference, on=keys, how="left", validate="many_to_one")
    treatment["changed"] = (
        pd.to_numeric(treatment["cal_predicted_content_id"], errors="coerce")
        != pd.to_numeric(treatment["baseline_content_id"], errors="coerce")
    )
    treatment["evaluable"] = treatment[
        ["cal_predicted_content_id", "baseline_content_id"]
    ].notna().all(axis=1)
    result: dict[tuple[str, str], bool | None] = {}
    for key, group in treatment.groupby(keys, sort=False):
        result[(str(key[0]), str(key[1]))] = (
            bool(group.loc[group["evaluable"], "changed"].any())
            if bool(group["evaluable"].any())
            else None
        )
    return result


def build_source_ledger(
    source: pd.DataFrame,
    *,
    expected_split_items: Mapping[str, int] = EXPECTED_SPLIT_ITEMS,
    expected_formats: Sequence[str] = EXPECTED_FORMATS,
) -> pd.DataFrame:
    """Build the source-bound two-contract ledger without touching protected splits."""

    _require_columns(source, {"split", "item_id", "wrapper_name"}, name="causal source")
    observed_splits = set(source["split"].astype(str))
    protected = observed_splits - {"train", "validation"}
    if protected:
        raise ValueError(f"causal source contains a protected split: {sorted(protected)}")
    if observed_splits != set(expected_split_items):
        raise ValueError(
            f"causal source split set mismatch: {sorted(observed_splits)} != "
            f"{sorted(expected_split_items)}"
        )

    required = {
        "work_key", "subject", "arm", "manipulation", "variant", "prompt",
        "prompt_sha256", "calibration_prompt", "calibration_prompt_sha256",
        "source_prompt_sha256", "content_ids_by_position", "labels_by_position",
        "candidate_texts", "correct_content_id", "choice_provenance",
    }
    _require_columns(source, required, name="causal source")
    if source["work_key"].astype(str).duplicated().any():
        raise ValueError("causal source contains duplicate work keys")

    item_splits = source[["item_id", "split"]].drop_duplicates()
    if item_splits["item_id"].astype(str).duplicated().any():
        raise ValueError("one item appears in more than one split")
    counts = item_splits.groupby("split")["item_id"].nunique().to_dict()
    if {str(key): int(value) for key, value in counts.items()} != {
        str(key): int(value) for key, value in expected_split_items.items()
    }:
        raise ValueError(f"causal source split-item counts mismatch: {counts}")

    expected_format_set = set(map(str, expected_formats))
    observed_formats = set(source["wrapper_name"].astype(str))
    if observed_formats != expected_format_set:
        raise ValueError("causal source format set mismatch")
    item_formats = source[["item_id", "wrapper_name"]].drop_duplicates()
    per_item = item_formats.groupby("item_id")["wrapper_name"].agg(
        lambda values: set(map(str, values))
    )
    if any(values != expected_format_set for values in per_item):
        raise ValueError("causal source does not contain every expected item-format block")

    baseline = source[
        source["arm"].astype(str).eq("letter_intervention")
        & source["manipulation"].astype(str).eq("controlled_baseline")
        & pd.to_numeric(source["variant"], errors="coerce").eq(0)
    ].copy()
    text = source[source["arm"].astype(str).eq("answer_text")].copy()
    keys = ["item_id", "wrapper_name"]
    expected_blocks = len(item_splits) * len(expected_formats)
    for name, frame in (("letter baseline", baseline), ("text contract", text)):
        if len(frame) != expected_blocks or frame.duplicated(keys).any():
            raise ValueError(f"{name} does not contain exactly one row per item-format block")

    position_susceptibility = _susceptibility_map(
        source, baseline, manipulation="position_only"
    )
    label_susceptibility = _susceptibility_map(
        source, baseline, manipulation="label_only"
    )

    text_by_key = {
        (str(row.item_id), str(row.wrapper_name)): row._asdict()
        for row in text.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for letter_row in baseline.sort_values(keys, kind="mergesort").to_dict("records"):
        item_id = str(letter_row["item_id"])
        wrapper_name = str(letter_row["wrapper_name"])
        text_row = text_by_key[(item_id, wrapper_name)]
        letter_prompt = _validate_prompt(
            letter_row, field="prompt", hash_field="prompt_sha256"
        )
        text_prompt = _validate_prompt(text_row, field="prompt", hash_field="prompt_sha256")
        calibration_prompt = _validate_prompt(
            letter_row,
            field="calibration_prompt",
            hash_field="calibration_prompt_sha256",
            require_answer_prefix=False,
        )
        calibration_evaluable = calibration_prompt.endswith(
            "\nAnswer:"
        ) or calibration_prompt.endswith("\nAnswer: ")
        if not _terminal_contract_matches(letter_prompt, text_prompt):
            raise ValueError(
                f"prompt contracts differ outside the terminal instruction: {item_id}|{wrapper_name}"
            )
        # Validate each contract independently before comparing their values so a
        # malformed mapping cannot masquerade as ordinary contract drift.
        _as_permutation(
            letter_row["content_ids_by_position"], work_key=str(letter_row["work_key"])
        )
        _as_permutation(
            text_row["content_ids_by_position"], work_key=str(text_row["work_key"])
        )
        for field in ("source_prompt_sha256", "candidate_texts", "content_ids_by_position"):
            left = letter_row[field]
            right = text_row[field]
            if isinstance(left, np.ndarray):
                left = left.tolist()
            if isinstance(right, np.ndarray):
                right = right.tolist()
            if left != right:
                raise ValueError(f"prompt contracts disagree on {field}: {item_id}|{wrapper_name}")
        candidates = _as_four_strings(
            letter_row["candidate_texts"], work_key=str(letter_row["work_key"])
        )
        content_ids = _as_permutation(
            letter_row["content_ids_by_position"], work_key=str(letter_row["work_key"])
        )
        labels = list(letter_row["labels_by_position"])
        if labels != list("ABCD"):
            raise ValueError(f"baseline displayed labels are malformed: {item_id}|{wrapper_name}")
        baseline_content = _optional_int(letter_row.get("cal_predicted_content_id"))
        text_content = _optional_int(text_row.get("candidate_predicted_content_id"))
        records.append(
            {
                "block_work_key": f"{item_id}|{wrapper_name}",
                "item_id": item_id,
                "subject": str(letter_row["subject"]),
                "split": str(letter_row["split"]),
                "wrapper_name": wrapper_name,
                "source_prompt_sha256": str(letter_row["source_prompt_sha256"]),
                "letter_source_work_key": str(letter_row["work_key"]),
                "text_source_work_key": str(text_row["work_key"]),
                "letter_prompt": letter_prompt,
                "letter_prompt_sha256": _sha256_text(letter_prompt),
                "answer_prefix_trailing_space": letter_prompt.endswith("Answer: "),
                "text_prompt": text_prompt,
                "text_prompt_sha256": _sha256_text(text_prompt),
                "calibration_prompt": calibration_prompt,
                "calibration_prompt_sha256": _sha256_text(calibration_prompt),
                "calibration_lens_evaluable": calibration_evaluable,
                "calibration_lens_ineligibility_reason": (
                    "eligible" if calibration_evaluable else "missing_exact_answer_prefix"
                ),
                "content_free_calibration_kind": letter_row.get(
                    "content_free_calibration_kind"
                ),
                "content_free_fallback_reason": letter_row.get(
                    "content_free_fallback_reason"
                ),
                "content_ids_by_position": content_ids,
                "labels_by_position": labels,
                "candidate_texts": candidates,
                "correct_content_id": int(letter_row["correct_content_id"]),
                "choice_provenance": str(letter_row["choice_provenance"]),
                "stored_letter_raw_content_id": _optional_int(
                    letter_row.get("raw_predicted_content_id")
                ),
                "stored_letter_cal_content_id": baseline_content,
                "stored_letter_cal_correct": (
                    None
                    if letter_row.get("cal_correct") is None
                    else bool(letter_row.get("cal_correct"))
                ),
                "stored_letter_cal_entropy": (
                    None
                    if letter_row.get("cal_entropy") is None
                    else float(letter_row.get("cal_entropy"))
                ),
                "stored_text_candidate_content_id": text_content,
                "stored_text_candidate_tie": bool(
                    text_row.get("candidate_total_tie", False)
                ),
                "stored_text_letter_relation": (
                    "undefined"
                    if baseline_content is None
                    or text_content is None
                    or bool(text_row.get("candidate_total_tie", False))
                    else "agree" if baseline_content == text_content else "disagree"
                ),
                "position_susceptible": position_susceptibility.get(
                    (item_id, wrapper_name)
                ),
                "label_susceptible": label_susceptibility.get(
                    (item_id, wrapper_name)
                ),
            }
        )

    ledger = pd.DataFrame.from_records(records).sort_values(
        ["item_id", "wrapper_name"], kind="mergesort"
    ).reset_index(drop=True)
    plain = ledger[ledger["wrapper_name"].eq("plain")].set_index("item_id")
    if len(plain) != len(item_splits):
        raise ValueError("source ledger lacks exactly one plain row per item")
    surface_match: list[bool] = []
    identity_evaluable: list[bool] = []
    identity_reasons: list[str] = []
    conflict_status: list[str] = []
    correctness_status: list[str] = []
    for row in ledger.itertuples(index=False):
        plain_row = plain.loc[str(row.item_id)]
        surfaces_match = list(row.candidate_texts) == list(plain_row["candidate_texts"])
        provenance = str(row.choice_provenance)
        # Canonical source choices and parser-backed payload extraction preserve
        # the canonical content IDs. The separate displayed-choice audit proves
        # physical A/B/C/D text and the correct position only; its stored
        # provenance is a model/thread receipt and cannot certify a four-way
        # semantic permutation across wrappers.
        mapping_proven = provenance == "canonical" or provenance.endswith(":payload")
        reasons = []
        if not mapping_proven:
            reasons.append("mapping_unproven")
        if not surfaces_match:
            reasons.append("surface_mismatch")
        surface_match.append(surfaces_match)
        identity_evaluable.append(not reasons)
        identity_reasons.append("eligible" if not reasons else "_and_".join(reasons))
        plain_prediction = _optional_int(plain_row["stored_letter_cal_content_id"])
        wrapped_prediction = _optional_int(row.stored_letter_cal_content_id)
        conflict_status.append(
            "undefined"
            if plain_prediction is None or wrapped_prediction is None
            else "stable" if plain_prediction == wrapped_prediction else "conflict"
        )
        plain_correct = plain_row["stored_letter_cal_correct"]
        wrapped_correct = row.stored_letter_cal_correct
        if plain_correct is None or wrapped_correct is None:
            correctness_status.append("undefined")
        elif bool(plain_correct) and bool(wrapped_correct):
            correctness_status.append("both_correct")
        elif bool(plain_correct):
            correctness_status.append("plain_only_correct")
        elif bool(wrapped_correct):
            correctness_status.append("wrapped_only_correct")
        else:
            correctness_status.append("both_wrong")
    ledger["candidate_surfaces_match_plain"] = surface_match
    ledger["cross_wrapper_identity_evaluable"] = identity_evaluable
    ledger["cross_wrapper_identity_reason"] = identity_reasons
    ledger["stored_plain_wrapped_status"] = conflict_status
    ledger["stored_correctness_status"] = correctness_status

    entropy = pd.to_numeric(ledger["stored_letter_cal_entropy"], errors="coerce")
    if entropy.notna().any():
        percentile = (-entropy).rank(method="average", pct=True)
        ledger["stored_confidence_stratum"] = pd.cut(
            percentile,
            bins=[0.0, 1 / 3, 2 / 3, 1.0],
            labels=["low", "medium", "high"],
            include_lowest=True,
        ).astype(str)
    else:
        ledger["stored_confidence_stratum"] = "undefined"
    return ledger


def _read_json(path: Path, *, name: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return payload


def load_prepared_bundle(
    root: str | Path,
    *,
    expected_split_items: Mapping[str, int] = EXPECTED_SPLIT_ITEMS,
    expected_formats: Sequence[str] = EXPECTED_FORMATS,
) -> tuple[dict[str, object], pd.DataFrame]:
    bundle = Path(root)
    manifest = _read_json(bundle / "prepared_manifest.json", name="prepared manifest")
    if manifest.get("schema_version") != 1 or manifest.get("stage") != "discovery":
        raise RuntimeError("prepared bundle is not a discovery logit-lens bundle")
    ledger_path = bundle / "prompt_ledger.parquet"
    if not ledger_path.exists() or sha256_file(ledger_path) != manifest.get("ledger_sha256"):
        raise RuntimeError("prepared ledger checksum mismatch")
    ledger = pd.read_parquet(ledger_path)
    if len(ledger) != int(manifest.get("rows", -1)):
        raise RuntimeError("prepared ledger row count mismatch")
    splits = ledger[["item_id", "split"]].drop_duplicates()
    counts = splits.groupby("split")["item_id"].nunique().to_dict()
    if {str(k): int(v) for k, v in counts.items()} != {
        str(k): int(v) for k, v in expected_split_items.items()
    }:
        raise RuntimeError("prepared ledger split boundary mismatch")
    if set(ledger["wrapper_name"].astype(str)) != set(map(str, expected_formats)):
        raise RuntimeError("prepared ledger format boundary mismatch")
    if ledger["block_work_key"].astype(str).duplicated().any():
        raise RuntimeError("prepared ledger contains duplicate block work keys")
    expected_rows = sum(map(int, expected_split_items.values())) * len(expected_formats)
    if len(ledger) != expected_rows:
        raise RuntimeError("prepared ledger population cardinality mismatch")
    return manifest, ledger


def _displayed_candidates(row: Mapping[str, object]) -> list[str]:
    candidates = _as_four_strings(
        row["candidate_texts"], work_key=str(row["block_work_key"])
    )
    content_ids = _as_permutation(
        row["content_ids_by_position"], work_key=str(row["block_work_key"])
    )
    return [candidates[content_id] for content_id in content_ids]


def _audit_record(
    source_row: Mapping[str, object],
    *,
    contract: str,
    audit: ContinuationAudit | None,
    root_reason: str,
) -> dict[str, object]:
    candidates = [] if audit is None else list(audit.candidates)
    surfaces = [] if audit is None else [candidate.surface for candidate in candidates]
    counts = [] if audit is None else [len(candidate.token_ids) for candidate in candidates]
    reasons = [] if audit is None else list(audit.identity_ineligibility_reasons)
    root_evaluable = audit is not None
    candidate_identity_evaluable = bool(
        audit is not None and audit.identity_comparison_evaluable
    )
    source_identity = bool(source_row["cross_wrapper_identity_evaluable"])
    primary_evaluable = bool(
        contract in {"letter", "text"}
        and root_evaluable
        and source_identity
        and candidate_identity_evaluable
    )
    first_token_evaluable = bool(
        primary_evaluable and audit is not None and not audit.shared_first_token_indices
    )
    return {
        "block_work_key": str(source_row["block_work_key"]),
        "item_id": str(source_row["item_id"]),
        "split": str(source_row["split"]),
        "wrapper_name": str(source_row["wrapper_name"]),
        "contract": contract,
        "root_evaluable": root_evaluable,
        "root_ineligibility_reason": root_reason,
        "prompt": "" if audit is None else audit.prompt,
        "prompt_sha256": "" if audit is None else audit.prompt_sha256,
        "prompt_ids": [] if audit is None else list(audit.prompt_ids),
        "prompt_token_sha256": "" if audit is None else audit.prompt_token_sha256,
        "prompt_token_count": 0 if audit is None else len(audit.prompt_ids),
        "prompt_roundtrip_exact": None,
        "decoded_prompt_sha256": "",
        "observation_token_id": None if audit is None else int(audit.prompt_ids[-1]),
        "observation_token_text": "",
        "observation_character_span": [None, None],
        "label_token_ids": [] if audit is None else list(audit.label_token_ids),
        "candidate_surfaces_by_position": surfaces,
        "candidate_surface_sha256s": [candidate.surface_sha256 for candidate in candidates],
        "candidate_utf8_bytes": [candidate.utf8_bytes for candidate in candidates],
        "candidate_token_ids": [list(candidate.token_ids) for candidate in candidates],
        "candidate_path_token_counts": counts,
        "candidate_path_evaluable": [candidate.eligible for candidate in candidates],
        "candidate_path_ineligibility_reasons": [
            candidate.eligibility_reason for candidate in candidates
        ],
        "duplicate_surface_indices": (
            [] if audit is None else [list(value) for value in audit.duplicate_surface_indices]
        ),
        "identical_token_indices": (
            [] if audit is None else [list(value) for value in audit.identical_token_indices]
        ),
        "prefix_collision_indices": (
            [] if audit is None else [list(value) for value in audit.prefix_collision_indices]
        ),
        "shared_first_token_indices": (
            [] if audit is None else [list(value) for value in audit.shared_first_token_indices]
        ),
        "label_like_indices": [] if audit is None else list(audit.label_like_indices),
        "has_shared_first_token": bool(audit and audit.shared_first_token_indices),
        "has_prefix_collision": bool(audit and audit.prefix_collision_indices),
        "has_unicode_candidate": any(any(ord(char) > 127 for char in value) for value in surfaces),
        "has_punctuation_candidate": any(
            any(unicodedata.category(char).startswith(("P", "S")) for char in value)
            for value in surfaces
        ),
        "candidate_identity_evaluable": candidate_identity_evaluable,
        "candidate_identity_ineligibility_reasons": reasons,
        "primary_contrast_evaluable": primary_evaluable,
        "first_token_contrast_evaluable": first_token_evaluable,
    }


def _tokenizer_receipt(tokenizer) -> dict[str, object]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = ""
    if backend is not None and hasattr(backend, "to_str"):
        backend_json = str(backend.to_str())
    receipt = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "special_token_ids": [int(value) for value in getattr(tokenizer, "all_special_ids", ())],
        "backend_sha256": _sha256_text(backend_json) if backend_json else None,
    }
    return {**receipt, "receipt_sha256": _canonical_sha256(receipt)}


def _add_observation_receipt(record: dict[str, object], tokenizer) -> dict[str, object]:
    if not bool(record["root_evaluable"]):
        return record
    token_id = int(record["observation_token_id"])
    try:
        decoded_prompt = tokenizer.decode(
            list(record["prompt_ids"]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded_prompt = tokenizer.decode(list(record["prompt_ids"]))
    record["prompt_roundtrip_exact"] = decoded_prompt == str(record["prompt"])
    record["decoded_prompt_sha256"] = _sha256_text(str(decoded_prompt))
    try:
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        token_text = tokenizer.decode([token_id])
    span: list[int | None] = [None, None]
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                str(record["prompt"]),
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = [int(value) for value in encoded["input_ids"]]
            offsets = encoded["offset_mapping"]
            if ids != list(record["prompt_ids"]):
                raise ValueError("tokenizer offset receipt changed the exact prompt IDs")
            span = [int(offsets[-1][0]), int(offsets[-1][1])]
        except (TypeError, KeyError, NotImplementedError):
            span = [None, None]
    record["observation_token_text"] = str(token_text)
    record["observation_character_span"] = span
    return record


def audit_tokenizer_bundle(
    bundle_root: str | Path,
    output_dir: str | Path,
    *,
    tokenizer,
    max_context_tokens: int,
    expected_split_items: Mapping[str, int] = EXPECTED_SPLIT_ITEMS,
    expected_formats: Sequence[str] = EXPECTED_FORMATS,
) -> dict[str, object]:
    """Audit every exact continuation before model weights are loaded."""

    if int(max_context_tokens) < 1:
        raise ValueError("max_context_tokens must be positive")
    prepared, ledger = load_prepared_bundle(
        bundle_root,
        expected_split_items=expected_split_items,
        expected_formats=expected_formats,
    )
    records: list[dict[str, object]] = []
    for source_row in ledger.to_dict("records"):
        displayed_candidates = _displayed_candidates(source_row)
        for contract, prompt_field in (
            ("letter", "letter_prompt"),
            ("text", "text_prompt"),
        ):
            audit = audit_fixed_root_continuations(
                tokenizer,
                str(source_row[prompt_field]),
                displayed_candidates,
                max_context_tokens=max_context_tokens,
            )
            records.append(
                _add_observation_receipt(
                    _audit_record(
                        source_row, contract=contract, audit=audit, root_reason="eligible"
                    ),
                    tokenizer,
                )
            )
        if bool(source_row["calibration_lens_evaluable"]):
            calibration_audit = audit_fixed_root_continuations(
                tokenizer,
                str(source_row["calibration_prompt"]),
                list("ABCD"),
                max_context_tokens=max_context_tokens,
            )
            calibration_reason = "eligible"
        else:
            calibration_audit = None
            calibration_reason = str(
                source_row["calibration_lens_ineligibility_reason"]
            )
        records.append(
            _add_observation_receipt(
                _audit_record(
                    source_row,
                    contract="calibration",
                    audit=calibration_audit,
                    root_reason=calibration_reason,
                ),
                tokenizer,
            )
        )

    frame = pd.DataFrame.from_records(records).sort_values(
        ["block_work_key", "contract"], kind="mergesort"
    ).reset_index(drop=True)
    expected_rows = len(ledger) * 3
    if len(frame) != expected_rows or frame.duplicated(
        ["block_work_key", "contract"]
    ).any():
        raise RuntimeError("token audit does not preserve the complete block-contract product")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "continuation_audit.parquet"
    frame.to_parquet(audit_path, index=False)
    token_counts = [
        int(value)
        for values in frame.loc[frame["contract"].isin(["letter", "text"]), "candidate_path_token_counts"]
        for value in values
    ]
    manifest = {
        "schema_version": TOKEN_AUDIT_SCHEMA_VERSION,
        "stage": "discovery",
        "model": prepared["model"],
        "rows": len(frame),
        "blocks": len(ledger),
        "contracts": ["letter", "text", "calibration"],
        "audit_sha256": sha256_file(audit_path),
        "prepared_manifest_sha256": sha256_file(Path(bundle_root) / "prepared_manifest.json"),
        "prepared_ledger_sha256": prepared["ledger_sha256"],
        "max_context_tokens": int(max_context_tokens),
        "tokenizer": _tokenizer_receipt(tokenizer),
        "continuation_tokenization_policy": (
            "fixed_root_mistral_metaspace_without_implicit_prefix"
        ),
        "prompt_roundtrip_mismatches": int(
            frame["prompt_roundtrip_exact"].eq(False).sum()
        ),
        "candidate_paths": len(token_counts),
        "one_token_paths": sum(value == 1 for value in token_counts),
        "multi_token_paths": sum(value > 1 for value in token_counts),
        "maximum_candidate_tokens": max(token_counts, default=0),
        "root_ineligible": int((~frame["root_evaluable"]).sum()),
        "primary_ineligible": int(
            (~frame.loc[frame["contract"].isin(["letter", "text"]), "primary_contrast_evaluable"]).sum()
        ),
        "final_599_opened": False,
    }
    _atomic_json(manifest, output / "tokenization_manifest.json")
    return manifest


def load_token_audit(
    root: str | Path,
    *,
    bundle_root: str | Path,
    expected_blocks: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    audit_root = Path(root)
    manifest = _read_json(
        audit_root / "tokenization_manifest.json", name="tokenization manifest"
    )
    if (
        manifest.get("schema_version") != TOKEN_AUDIT_SCHEMA_VERSION
        or manifest.get("stage") != "discovery"
        or manifest.get("final_599_opened") is not False
    ):
        raise RuntimeError("token audit has an invalid discovery boundary")
    prepared_path = Path(bundle_root) / "prepared_manifest.json"
    prepared = _read_json(prepared_path, name="prepared manifest")
    if (
        sha256_file(prepared_path) != manifest.get("prepared_manifest_sha256")
        or prepared.get("ledger_sha256") != manifest.get("prepared_ledger_sha256")
    ):
        raise RuntimeError("token audit prepared-bundle identity mismatch")
    path = audit_root / "continuation_audit.parquet"
    if not path.exists() or sha256_file(path) != manifest.get("audit_sha256"):
        raise RuntimeError("token audit checksum mismatch")
    frame = pd.read_parquet(path)
    if (
        len(frame) != int(manifest.get("rows", -1))
        or int(manifest.get("blocks", -1)) != int(expected_blocks)
        or len(frame) != int(expected_blocks) * 3
        or frame.duplicated(["block_work_key", "contract"]).any()
        or set(frame["contract"].astype(str)) != {"letter", "text", "calibration"}
    ):
        raise RuntimeError("token audit structure mismatch")
    return manifest, frame


def _audit_from_record(row: Mapping[str, object]) -> ContinuationAudit:
    if not bool(row["root_evaluable"]):
        raise ValueError("cannot reconstruct an ineligible root audit")
    candidates = tuple(
        CandidateContinuation(
            surface=str(surface),
            token_ids=tuple(int(value) for value in token_ids),
            surface_sha256=str(surface_sha),
            utf8_bytes=int(utf8_bytes),
            eligible=bool(eligible),
            eligibility_reason=str(reason),
        )
        for surface, token_ids, surface_sha, utf8_bytes, eligible, reason in zip(
            row["candidate_surfaces_by_position"],
            row["candidate_token_ids"],
            row["candidate_surface_sha256s"],
            row["candidate_utf8_bytes"],
            row["candidate_path_evaluable"],
            row["candidate_path_ineligibility_reasons"],
            strict=True,
        )
    )
    return ContinuationAudit(
        prompt=str(row["prompt"]),
        prompt_ids=tuple(int(value) for value in row["prompt_ids"]),
        prompt_sha256=str(row["prompt_sha256"]),
        prompt_token_sha256=str(row["prompt_token_sha256"]),
        label_token_ids=tuple(int(value) for value in row["label_token_ids"]),  # type: ignore[arg-type]
        candidates=candidates,
        duplicate_surface_indices=tuple(
            tuple(int(value) for value in group) for group in row["duplicate_surface_indices"]
        ),
        identical_token_indices=tuple(
            tuple(int(value) for value in group) for group in row["identical_token_indices"]
        ),
        prefix_collision_indices=tuple(
            tuple(int(value) for value in group) for group in row["prefix_collision_indices"]
        ),
        shared_first_token_indices=tuple(
            tuple(int(value) for value in group) for group in row["shared_first_token_indices"]
        ),
        label_like_indices=tuple(int(value) for value in row["label_like_indices"]),
        identity_comparison_evaluable=bool(row["candidate_identity_evaluable"]),
        identity_ineligibility_reasons=tuple(
            str(value) for value in row["candidate_identity_ineligibility_reasons"]
        ),
    )


def select_startup_work_keys(audit: pd.DataFrame, *, count: int = 8) -> list[str]:
    """Choose the single startup sample from tokenizer structure only."""

    if count < 1:
        raise ValueError("startup item count must be positive")
    letter = audit[audit["contract"].astype(str).eq("letter")].copy()
    if letter["block_work_key"].astype(str).duplicated().any() or len(letter) < count:
        raise ValueError("token audit cannot supply the requested startup sample")
    letter["max_path_tokens"] = letter["candidate_path_token_counts"].map(
        lambda values: max((int(value) for value in values), default=0)
    )
    letter = letter.sort_values("block_work_key", kind="mergesort")
    selected: list[str] = []

    def add(frame: pd.DataFrame) -> None:
        for value in frame["block_work_key"].astype(str):
            if value not in selected:
                selected.append(value)
                break

    for column in (
        "has_prefix_collision",
        "has_unicode_candidate",
        "has_punctuation_candidate",
        "has_shared_first_token",
    ):
        add(letter[letter[column].astype(bool)])
    add(letter[letter["candidate_path_token_counts"].map(lambda values: any(int(v) == 1 for v in values))])
    add(letter[letter["candidate_path_token_counts"].map(lambda values: any(int(v) > 1 for v in values))])
    add(letter.sort_values(["max_path_tokens", "block_work_key"], ascending=[False, True]))
    add(letter[~letter["candidate_identity_evaluable"].astype(bool)])
    for value in letter["block_work_key"].astype(str):
        if len(selected) >= count:
            break
        if value not in selected:
            selected.append(value)
    return sorted(selected[:count])


def build_logit_lens_identity(
    profile: ModelProfile,
    *,
    config: Mapping[str, object],
    dataset_path: str | Path,
    source_paths: Iterable[str | Path],
) -> SemanticIdentity:
    identity_config = dict(config)
    identity_config.pop("max_chunks_this_invocation", None)
    return build_semantic_identity(
        profile,
        config=identity_config,
        dataset_path=dataset_path,
        source_paths=source_paths,
    )


def _max_finite_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or not torch.equal(torch.isnan(left), torch.isnan(right)):
        raise ValueError("cached/scalar score shapes or eligibility masks differ")
    finite = torch.isfinite(left) & torch.isfinite(right)
    if torch.isinf(left).any() or torch.isinf(right).any():
        raise ValueError("cached/scalar score contains infinity")
    if not bool(finite.any()):
        return 0.0
    return float((left[finite] - right[finite]).abs().max())


def _argmax_set_disagreement(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[bool, bool]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("cached/scalar argmax comparison shape mismatch")
    left_finite = torch.isfinite(left)
    right_finite = torch.isfinite(right)
    if not torch.equal(left_finite, right_finite):
        raise ValueError("cached/scalar argmax eligibility masks differ")
    if not bool(left_finite.any()):
        return False, False
    indices = torch.nonzero(left_finite, as_tuple=False).flatten()
    left_values = left[indices]
    right_values = right[indices]
    left_winners = set(indices[left_values == left_values.max()].tolist())
    right_winners = set(indices[right_values == right_values.max()].tolist())
    return True, left_winners != right_winners


def score_lens_chunk(
    *,
    model,
    ledger: pd.DataFrame,
    token_audit: pd.DataFrame,
    work_keys: Sequence[str],
    batch_size: int,
    max_batch_tokens: int,
    expected_layers: int,
    scorer=score_candidate_paths_cached_many,
    run_scalar_oracle: bool,
    phase_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Score atomic item-format work units through the authenticated audit."""

    if batch_size < 1 or max_batch_tokens < 1 or expected_layers < 1:
        raise ValueError("runtime batching and expected layers must be positive")
    normalized_keys = [str(value) for value in work_keys]
    if len(normalized_keys) != len(set(normalized_keys)):
        raise ValueError("chunk work keys must be unique")
    source = ledger[ledger["block_work_key"].astype(str).isin(normalized_keys)].copy()
    if set(source["block_work_key"].astype(str)) != set(normalized_keys):
        raise ValueError("chunk references an unknown prepared work key")
    source_by_key = {
        str(row["block_work_key"]): row for row in source.to_dict("records")
    }
    audit_rows = token_audit[
        token_audit["block_work_key"].astype(str).isin(normalized_keys)
    ].copy()
    if len(audit_rows) != len(normalized_keys) * 3:
        raise ValueError("chunk token audit is incomplete")

    tasks: list[tuple[str, str, ContinuationAudit]] = []
    audit_metadata: dict[tuple[str, str], dict[str, object]] = {}
    for row in audit_rows.to_dict("records"):
        key = (str(row["block_work_key"]), str(row["contract"]))
        audit_metadata[key] = row
        if bool(row["root_evaluable"]):
            tasks.append((key[0], key[1], _audit_from_record(row)))
    scores: dict[tuple[str, str], object] = {}
    scalar_references: dict[tuple[str, str], object] = {}
    oracle_maxima = {
        "max_cached_scalar_letter_difference": 0.0,
        "max_cached_scalar_first_token_difference": 0.0,
        "max_cached_scalar_mean_token_difference": 0.0,
        "max_cached_scalar_total_per_token_difference": 0.0,
    }
    by_length: dict[int, list[tuple[str, str, ContinuationAudit]]] = {}
    for task in tasks:
        by_length.setdefault(len(task[2].prompt_ids), []).append(task)
    for prompt_length in sorted(by_length):
        ordered = sorted(by_length[prompt_length], key=lambda value: (value[0], value[1]))
        start = 0
        while start < len(ordered):
            max_count = min(batch_size, max(1, max_batch_tokens // prompt_length))
            batch = ordered[start : start + max_count]
            scorer_kwargs: dict[str, object] = {"expected_layers": expected_layers}
            if phase_callback is not None:
                scorer_kwargs["phase_callback"] = phase_callback
            observed = scorer(model, [value[2] for value in batch], **scorer_kwargs)
            if len(observed) != len(batch):
                raise RuntimeError("batched scorer returned the wrong number of rows")
            for task, result in zip(batch, observed, strict=True):
                scores[(task[0], task[1])] = result
                if run_scalar_oracle:
                    if phase_callback is not None:
                        phase_callback("scalar_oracle")
                    try:
                        reference = score_candidate_paths_scalar(
                            model, task[2], expected_layers=expected_layers
                        )
                    finally:
                        if phase_callback is not None:
                            phase_callback("idle")
                    scalar_references[(task[0], task[1])] = reference
                    letter_difference = _max_finite_difference(
                        result.letter_logp, reference.letter_logp
                    )
                    first_difference = _max_finite_difference(
                        result.first_token_logp, reference.first_token_logp
                    )
                    mean_difference = _max_finite_difference(
                        result.mean_token_logp, reference.mean_token_logp
                    )
                    total_difference = 0.0
                    for candidate_index, candidate in enumerate(task[2].candidates):
                        if not candidate.eligible:
                            continue
                        difference = _max_finite_difference(
                            result.total_logp[:, candidate_index],
                            reference.total_logp[:, candidate_index],
                        ) / len(candidate.token_ids)
                        total_difference = max(total_difference, difference)
                    for name, value in (
                        ("max_cached_scalar_letter_difference", letter_difference),
                        ("max_cached_scalar_first_token_difference", first_difference),
                        ("max_cached_scalar_mean_token_difference", mean_difference),
                        ("max_cached_scalar_total_per_token_difference", total_difference),
                    ):
                        oracle_maxima[name] = max(oracle_maxima[name], value)
            start += len(batch)

    output: list[dict[str, object]] = []
    max_native = 0.0
    for block_key in normalized_keys:
        source_row = source_by_key[block_key]
        calibration = scores.get((block_key, "calibration"))
        calibration_evaluable = calibration is not None
        for contract in ("letter", "text"):
            result = scores.get((block_key, contract))
            if result is None:
                raise ValueError(f"primary root is not evaluable: {block_key}/{contract}")
            max_native = max(max_native, float(result.max_final_native_difference))
            metadata = audit_metadata[(block_key, contract)]
            primary_evaluable = bool(metadata["primary_contrast_evaluable"])
            first_evaluable = bool(metadata["first_token_contrast_evaluable"])
            if contract == "letter" and calibration_evaluable:
                calibrated = result.letter_logp - calibration.letter_logp
            else:
                calibrated = torch.full_like(result.letter_logp, float("nan"))
            position_flag = source_row.get("position_susceptible")
            label_flag = source_row.get("label_susceptible")
            if position_flag is None or label_flag is None:
                susceptibility = "undefined"
            elif bool(position_flag) and bool(label_flag):
                susceptibility = "both"
            elif bool(position_flag):
                susceptibility = "position_only"
            elif bool(label_flag):
                susceptibility = "label_only"
            else:
                susceptibility = "neither"
            candidate_reasons = [
                str(value) for value in metadata["candidate_identity_ineligibility_reasons"]
            ]
            primary_reason = (
                "eligible"
                if primary_evaluable
                else "|".join(
                    dict.fromkeys(
                        [
                            str(source_row["cross_wrapper_identity_reason"]),
                            *candidate_reasons,
                        ]
                    )
                )
            )
            for layer in range(expected_layers):
                sensitivity: dict[str, bool] = {}
                if run_scalar_oracle:
                    reference = scalar_references[(block_key, contract)]
                    readouts = (
                        ("letter", result.letter_logp[layer], reference.letter_logp[layer]),
                        (
                            "candidate_first_token",
                            result.first_token_logp[layer],
                            reference.first_token_logp[layer],
                        ),
                        (
                            "candidate_mean_token",
                            result.mean_token_logp[layer],
                            reference.mean_token_logp[layer],
                        ),
                        (
                            "candidate_total",
                            result.total_logp[layer],
                            reference.total_logp[layer],
                        ),
                    )
                    for readout, cached_values, scalar_values in readouts:
                        comparable, disagreement = _argmax_set_disagreement(
                            cached_values, scalar_values
                        )
                        sensitivity[f"cached_scalar_{readout}_argmax_comparable"] = comparable
                        sensitivity[f"cached_scalar_{readout}_argmax_disagreement"] = disagreement
                else:
                    for readout in (
                        "letter",
                        "candidate_first_token",
                        "candidate_mean_token",
                        "candidate_total",
                    ):
                        sensitivity[f"cached_scalar_{readout}_argmax_comparable"] = False
                        sensitivity[f"cached_scalar_{readout}_argmax_disagreement"] = False
                output.append(
                    {
                        "block_work_key": block_key,
                        "item_id": str(source_row["item_id"]),
                        "subject": str(source_row["subject"]),
                        "split": str(source_row["split"]),
                        "wrapper_name": str(source_row["wrapper_name"]),
                        "contract": contract,
                        "layer": layer,
                        "content_ids_by_position": list(source_row["content_ids_by_position"]),
                        "candidate_surfaces_by_position": list(
                            metadata["candidate_surfaces_by_position"]
                        ),
                        "candidate_token_ids": [
                            list(value) for value in metadata["candidate_token_ids"]
                        ],
                        "candidate_path_token_counts": list(
                            metadata["candidate_path_token_counts"]
                        ),
                        "candidate_path_evaluable": list(
                            metadata["candidate_path_evaluable"]
                        ),
                        "letter_raw_logps": result.letter_logp[layer].tolist(),
                        "letter_calibrated_logps": calibrated[layer].tolist(),
                        "candidate_path_total_logps": result.total_logp[layer].tolist(),
                        "candidate_mean_token_logps": result.mean_token_logp[layer].tolist(),
                        "candidate_first_token_logps": result.first_token_logp[layer].tolist(),
                        "primary_contrast_evaluable": primary_evaluable,
                        "primary_contrast_exclusion_reason": primary_reason,
                        "first_token_contrast_evaluable": first_evaluable,
                        "first_token_contrast_exclusion_reason": (
                            "eligible" if first_evaluable else "shared_or_ineligible_first_token"
                        ),
                        # Block-level availability is repeated across both
                        # contracts so population receipts cannot drift. The
                        # calibrated score itself is defined only on letter rows.
                        "letter_calibration_evaluable": calibration_evaluable,
                        "letter_calibration_exclusion_reason": (
                            "eligible"
                            if calibration_evaluable
                            else str(
                                audit_metadata[(block_key, "calibration")][
                                    "root_ineligibility_reason"
                                ]
                            )
                        ),
                        "conflict_status": str(source_row["stored_plain_wrapped_status"]),
                        "confidence_stratum": str(source_row["stored_confidence_stratum"]),
                        "correctness_stratum": str(source_row["stored_correctness_status"]),
                        "position_susceptible": position_flag,
                        "label_susceptible": label_flag,
                        "susceptibility_stratum": susceptibility,
                        "text_letter_status": str(source_row["stored_text_letter_relation"]),
                        "incorrect_wrapped_decision": bool(
                            str(source_row["wrapper_name"]) != "plain"
                            and source_row["stored_letter_cal_correct"] is False
                        ),
                        "exact_surface_match": bool(
                            source_row["candidate_surfaces_match_plain"]
                        ),
                        "audit_provenance": str(source_row["choice_provenance"]),
                        **sensitivity,
                    }
                )
    frame = pd.DataFrame.from_records(output)
    parity = {
        "max_final_native_difference": max_native,
        **oracle_maxima,
    }
    for readout in (
        "letter",
        "candidate_first_token",
        "candidate_mean_token",
        "candidate_total",
    ):
        comparable_column = f"cached_scalar_{readout}_argmax_comparable"
        disagreement_column = f"cached_scalar_{readout}_argmax_disagreement"
        parity[f"cached_scalar_{readout}_argmax_comparisons"] = int(
            frame[comparable_column].sum()
        )
        parity[f"cached_scalar_{readout}_argmax_disagreements"] = int(
            (frame[comparable_column] & frame[disagreement_column]).sum()
        )
    frame["parity_final_native_max"] = float(max_native)
    frame["parity_cached_scalar_letter_max"] = float(
        oracle_maxima["max_cached_scalar_letter_difference"]
    )
    frame["parity_cached_scalar_first_token_max"] = float(
        oracle_maxima["max_cached_scalar_first_token_difference"]
    )
    frame["parity_cached_scalar_mean_token_max"] = float(
        oracle_maxima["max_cached_scalar_mean_token_difference"]
    )
    frame["parity_cached_scalar_total_per_token_max"] = float(
        oracle_maxima["max_cached_scalar_total_per_token_difference"]
    )
    frame["scalar_oracle_evaluated"] = bool(run_scalar_oracle)
    return frame, parity


def run_atomic_chunks(
    store: ShardStore,
    ordered_work_keys: Sequence[str],
    *,
    chunk_size: int,
    max_chunks_this_invocation: int | None,
    process_chunk,
    should_stop=lambda: False,
    on_flush=None,
) -> list[str]:
    """Write complete chunks atomically and skip authenticated completed keys."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_chunks_this_invocation is not None and max_chunks_this_invocation < 1:
        raise ValueError("max_chunks_this_invocation must be positive")
    completed = store.completed_work_keys()
    ordered = [str(value) for value in ordered_work_keys]
    prefix_length = 0
    while prefix_length < len(ordered) and ordered[prefix_length] in completed:
        prefix_length += 1
    if completed != set(ordered[:prefix_length]):
        raise ValueError("resume work must be an exact completed prefix of the work plan")
    if prefix_length < len(ordered) and prefix_length % chunk_size != 0:
        raise ValueError("resume prefix does not end on an atomic chunk boundary")
    pending = ordered[prefix_length:]
    processed: list[str] = []
    chunks = 0
    for start in range(0, len(pending), chunk_size):
        if should_stop():
            break
        if max_chunks_this_invocation is not None and chunks >= max_chunks_this_invocation:
            break
        keys = pending[start : start + chunk_size]
        frame = process_chunk(keys)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("process_chunk must return a pandas DataFrame")
        frame = frame.copy()
        frame["_work_key"] = frame["block_work_key"].astype(str)
        if set(frame["_work_key"]) != set(keys):
            raise ValueError("scored chunk rows do not cover exactly the declared work keys")
        shard = store.write_shard(frame, work_keys=keys)
        processed.extend(keys)
        chunks += 1
        if on_flush is not None:
            on_flush(set(completed).union(processed), shard)
        if should_stop():
            break
    return processed


def _runtime_environment() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Mistral logit-lens execution requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Mistral logit-lens execution requires CUDA BF16 support")
    capability = torch.cuda.get_device_capability(0)

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "missing"

    return {
        "device_type": "cuda",
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
        "bf16_supported": True,
        "attention_backend": "sdpa",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "transformers_version": version("transformers"),
        "tokenizers_version": version("tokenizers"),
        "inference_dtype": "bfloat16",
    }


def _analysis_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "primary_score": "candidate_path_total_logp",
        "sensitivity_score": "candidate_mean_token_logp",
        "root_diagnostic": "candidate_first_token_logp",
        "letter_primary": "raw_logp",
        "letter_secondary": "content_free_calibrated_logp",
        "pre_final_layers": list(range(31)),
        "parity_layer": 31,
        "same_forward_final_native_parity_atol": PARITY_ATOL,
        "cross_forward_sensitivity_policy": CROSS_FORWARD_SENSITIVITY_POLICY,
        "ambiguity_reference": 0.04,
        "quality_gates": {
            "minimum_primary_items_per_contract": 200,
            "minimum_primary_item_fraction": 0.8,
            "maximum_final_layer_ambiguity_rate": 0.2,
            "ambiguity_reference": 0.04,
            "required_contracts": ["letter", "text"],
            "required_final_layer_readouts": ["candidate_total", "letter_raw"],
            "require_resolved_primary_estimates": True,
            "require_total_mean_direction_agreement_by_contract": True,
        },
        "bootstrap_samples": 5000,
        "bootstrap_seed": 1729,
        "claim": CLAIM_BOUNDARY,
    }


def _run_source_paths() -> list[Path]:
    return [
        *default_semantic_source_paths(PROJECT_ROOT),
        PROJECT_ROOT / "analysis" / "analyze_decision_binding_logit_lens.py",
        PROJECT_ROOT / "DECISION_BINDING_LOGIT_LENS_RUN_CARD.md",
        PROJECT_ROOT / "SCIENTIFIC_NORTH_STAR.md",
        PROJECT_ROOT / "requirements-gpu.lock",
    ]


PARITY_COLUMNS = {
    "max_final_native_difference": "parity_final_native_max",
    "max_cached_scalar_letter_difference": "parity_cached_scalar_letter_max",
    "max_cached_scalar_first_token_difference": "parity_cached_scalar_first_token_max",
    "max_cached_scalar_mean_token_difference": "parity_cached_scalar_mean_token_max",
    "max_cached_scalar_total_per_token_difference": (
        "parity_cached_scalar_total_per_token_max"
    ),
}

SENSITIVITY_ARGMAX_READOUTS = (
    "letter",
    "candidate_first_token",
    "candidate_mean_token",
    "candidate_total",
)
SENSITIVITY_SCORE_COLUMNS = {
    "letter": "letter_raw_logps",
    "candidate_first_token": "candidate_first_token_logps",
    "candidate_mean_token": "candidate_mean_token_logps",
    "candidate_total": "candidate_path_total_logps",
}


def _parity_report_from_frame(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "schema_version": 2,
            "tolerance": PARITY_ATOL,
            "cross_forward_sensitivity_policy": CROSS_FORWARD_SENSITIVITY_POLICY,
            "covered_work_keys": [],
            "scalar_oracle_work_keys": [],
            **{name: 0.0 for name in PARITY_COLUMNS},
            **{
                f"cached_scalar_{readout}_argmax_{suffix}": 0
                for readout in SENSITIVITY_ARGMAX_READOUTS
                for suffix in ("comparisons", "disagreements")
            },
        }
    sensitivity_columns = {
        f"cached_scalar_{readout}_argmax_{suffix}"
        for readout in SENSITIVITY_ARGMAX_READOUTS
        for suffix in ("comparable", "disagreement")
    }
    required = {
        "block_work_key",
        "scalar_oracle_evaluated",
        *PARITY_COLUMNS.values(),
        *sensitivity_columns,
        *SENSITIVITY_SCORE_COLUMNS.values(),
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"parity receipts are missing from score shards: {sorted(missing)}")
    covered = sorted(frame["block_work_key"].astype(str).unique().tolist())
    scalar_by_key = frame.groupby("block_work_key", sort=False)[
        "scalar_oracle_evaluated"
    ].agg(["min", "max"])
    if (scalar_by_key["min"] != scalar_by_key["max"]).any():
        raise ValueError("scalar-oracle parity state changes within a work unit")
    scalar_keys = sorted(
        str(key) for key, row in scalar_by_key.iterrows() if bool(row["min"])
    )
    maxima: dict[str, float] = {}
    for name, column in PARITY_COLUMNS.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError("parity receipts contain non-finite values")
        maxima[name] = float(values.max())
    scalar_mask = frame["scalar_oracle_evaluated"].astype(bool)
    scalar_frame = frame[scalar_mask]
    argmax_counts: dict[str, int] = {}
    for readout in SENSITIVITY_ARGMAX_READOUTS:
        comparable_column = f"cached_scalar_{readout}_argmax_comparable"
        disagreement_column = f"cached_scalar_{readout}_argmax_disagreement"
        if frame.loc[~scalar_mask, [comparable_column, disagreement_column]].astype(bool).any().any():
            raise ValueError("argmax sensitivity receipt exists without scalar evaluation")
        comparable = scalar_frame[comparable_column].astype(bool)
        disagreement = scalar_frame[
            disagreement_column
        ].astype(bool)
        expected_comparable = scalar_frame[SENSITIVITY_SCORE_COLUMNS[readout]].map(
            lambda values: bool(np.isfinite(np.asarray(values, dtype=float)).any())
        )
        if not comparable.equals(expected_comparable.astype(bool)):
            raise ValueError("argmax comparability receipt mismatch")
        if bool((disagreement & ~comparable).any()):
            raise ValueError("argmax disagreement lacks a comparable score vector")
        argmax_counts[f"cached_scalar_{readout}_argmax_comparisons"] = int(
            comparable.sum()
        )
        argmax_counts[f"cached_scalar_{readout}_argmax_disagreements"] = int(
            (comparable & disagreement).sum()
        )
    return {
        "schema_version": 2,
        "tolerance": PARITY_ATOL,
        "cross_forward_sensitivity_policy": CROSS_FORWARD_SENSITIVITY_POLICY,
        "covered_work_keys": covered,
        "scalar_oracle_work_keys": scalar_keys,
        **maxima,
        **argmax_counts,
    }


def _merge_parity_reports(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> dict[str, object]:
    if float(previous.get("tolerance", PARITY_ATOL)) != PARITY_ATOL or float(
        current.get("tolerance", PARITY_ATOL)
    ) != PARITY_ATOL:
        raise ValueError("parity report tolerance drift")
    policies = {
        previous.get("cross_forward_sensitivity_policy", CROSS_FORWARD_SENSITIVITY_POLICY),
        current.get("cross_forward_sensitivity_policy", CROSS_FORWARD_SENSITIVITY_POLICY),
    }
    if policies != {CROSS_FORWARD_SENSITIVITY_POLICY}:
        raise ValueError("cross-forward sensitivity policy drift")
    return {
        "schema_version": 2,
        "tolerance": PARITY_ATOL,
        "cross_forward_sensitivity_policy": CROSS_FORWARD_SENSITIVITY_POLICY,
        "covered_work_keys": sorted(
            set(map(str, previous.get("covered_work_keys", [])))
            | set(map(str, current.get("covered_work_keys", [])))
        ),
        "scalar_oracle_work_keys": sorted(
            set(map(str, previous.get("scalar_oracle_work_keys", [])))
            | set(map(str, current.get("scalar_oracle_work_keys", [])))
        ),
        **{
            name: max(float(previous.get(name, 0.0)), float(current.get(name, 0.0)))
            for name in PARITY_COLUMNS
        },
        **{
            f"cached_scalar_{readout}_argmax_{suffix}": int(
                previous.get(f"cached_scalar_{readout}_argmax_{suffix}", 0)
            )
            + int(current.get(f"cached_scalar_{readout}_argmax_{suffix}", 0))
            for readout in SENSITIVITY_ARGMAX_READOUTS
            for suffix in ("comparisons", "disagreements")
        },
    }


def execute_model_run(args: argparse.Namespace) -> dict[str, object]:
    profile = get_model_profile(args.profile)
    if profile.name != "mistral":
        raise ValueError("the logit-lens experiment is frozen to Mistral only")
    if args.batch_size < 1 or args.max_batch_tokens < 1 or args.capture_chunk_size < 1:
        raise ValueError("runtime batch size, token cap, and chunk size must be positive")
    if args.startup_items is not None and args.startup_items < 1:
        raise ValueError("startup-items must be positive")
    if args.startup_items is not None and args.startup_items != 8:
        raise ValueError("the structural startup is frozen to exactly eight items")
    if args.startup_items is not None and args.capture_chunk_size != 4:
        raise ValueError("the structural startup is frozen to four-work-unit chunks")
    if (
        args.max_chunks_this_invocation is not None
        and args.max_chunks_this_invocation < 1
    ):
        raise ValueError("max-chunks-this-invocation must be positive")

    prepared, ledger = load_prepared_bundle(args.bundle)
    token_manifest, token_audit = load_token_audit(
        args.token_audit,
        bundle_root=args.bundle,
        expected_blocks=len(ledger),
    )
    environment = _runtime_environment()
    if args.startup_items is None:
        mode = "full"
        work_keys = ledger["block_work_key"].astype(str).tolist()
    else:
        mode = "startup"
        work_keys = select_startup_work_keys(token_audit, count=args.startup_items)
    config = {
        "experiment": "mistral_two_contract_logit_lens_v4",
        "stage": "discovery",
        "mode": mode,
        "prepared_manifest_sha256": sha256_file(
            Path(args.bundle) / "prepared_manifest.json"
        ),
        "prepared_ledger_sha256": prepared["ledger_sha256"],
        "tokenization_manifest_sha256": sha256_file(
            Path(args.token_audit) / "tokenization_manifest.json"
        ),
        "continuation_audit_sha256": token_manifest["audit_sha256"],
        "ordered_work_keys_sha256": _canonical_sha256(work_keys),
        "startup_work_keys": work_keys if mode == "startup" else [],
        "scoring_policy": {
            "layers": list(range(profile.expected_layers)),
            "observation": "final_prompt_token_at_answer_prefix",
            "projection": "post_block_then_model_final_norm_then_lm_head_log_softmax",
            "candidate_primary": "teacher_forced_path_total_logp",
            "candidate_sensitivity": "mean_token_logp",
            "first_token": "root_state_diagnostic",
            "surface_policy": "exact_canonical_token_sequence_no_variants_no_eos",
            "same_forward_final_native_parity_atol": PARITY_ATOL,
            "cross_forward_sensitivity_policy": CROSS_FORWARD_SENSITIVITY_POLICY,
        },
        "analysis_policy": _analysis_spec(),
        "runtime_environment": environment,
        "batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "capture_chunk_size": int(args.capture_chunk_size),
        "max_chunks_this_invocation": args.max_chunks_this_invocation,
    }
    identity = build_logit_lens_identity(
        profile,
        config=config,
        dataset_path=Path(args.bundle) / "prompt_ledger.parquet",
        source_paths=_run_source_paths(),
    )
    run_root = (
        Path(args.output_base) / profile.slug / identity.semantic_run_id
    )
    run_root.mkdir(parents=True, exist_ok=True)
    identity_path = run_root / "semantic_identity.json"
    if identity_path.exists():
        if _read_json(identity_path, name="semantic identity") != identity.as_dict():
            raise RuntimeError("existing run root has a different semantic identity")
    else:
        _atomic_json(identity.as_dict(), identity_path)
    for source_path, destination in (
        (Path(args.bundle) / "prepared_manifest.json", run_root / "prepared_manifest.json"),
        (
            Path(args.token_audit) / "tokenization_manifest.json",
            run_root / "tokenization_manifest.json",
        ),
        (
            Path(args.token_audit) / "continuation_audit.parquet",
            run_root / "continuation_audit.parquet",
        ),
    ):
        if destination.exists() and sha256_file(destination) != sha256_file(source_path):
            raise RuntimeError(f"existing run input receipt conflicts: {destination.name}")
        if not destination.exists():
            shutil.copy2(source_path, destination)
    work_plan = {
        "schema_version": 1,
        "mode": mode,
        "work_keys": work_keys,
        "ordered_work_keys_sha256": _canonical_sha256(work_keys),
        "expected_rows": len(work_keys) * profile.expected_layers * 2,
    }
    _atomic_json(work_plan, run_root / "work_plan.json")
    _atomic_json(_analysis_spec(), run_root / "analysis_spec.json")

    store = ShardStore(run_root / "shards" / "lens", identity)
    completed_before = store.completed_work_keys()
    started = time.time()
    attempt_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started)) + f"-{uuid.uuid4().hex[:8]}"
    attempt_path = run_root / "attempts" / f"{attempt_id}.json"
    attempt = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "semantic_run_id": identity.semantic_run_id,
        "started_at_unix": started,
        "max_chunks_this_invocation": args.max_chunks_this_invocation,
        "completed_before": len(completed_before),
        "status": "running",
    }
    _atomic_json(attempt, attempt_path)
    parity_path = run_root / "parity_report.json"
    if completed_before:
        parity = _parity_report_from_frame(
            store.merge(sort_by=["block_work_key", "contract", "layer"])
        )
    else:
        parity = _parity_report_from_frame(pd.DataFrame())
    _atomic_json(parity, parity_path)
    telemetry_path = run_root / "telemetry_report.json"
    telemetry_base = (
        _read_json(telemetry_path, name="telemetry report")
        if telemetry_path.exists()
        else {}
    )
    if telemetry_base:
        if telemetry_base.get("semantic_run_id") != identity.semantic_run_id:
            raise RuntimeError("existing telemetry report has a different identity")
        if not set(map(str, telemetry_base.get("covered_work_keys", []))).issubset(
            completed_before
        ):
            raise RuntimeError("existing telemetry covers unauthenticated work keys")
    telemetry = GpuPhaseTelemetry(identity.semantic_run_id)
    telemetry.start()
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        value: signal.signal(value, request_stop) for value in (signal.SIGTERM, signal.SIGINT)
    }
    model = None
    try:
        model, tokenizer, device = load_model_and_tokenizer(
            profile.model_id,
            revision=profile.revision,
            tokenizer_name=profile.tokenizer_id,
            tokenizer_revision=profile.tokenizer_revision,
            local_files_only=bool(args.local_files_only),
            attn_implementation=profile.attention_backend,
            expected_layers=profile.expected_layers,
            trust_remote_code=profile.trust_remote_code,
        )
        if device.type != "cuda" or next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("loaded Mistral does not match CUDA BF16 runtime identity")
        current_tokenizer = _tokenizer_receipt(tokenizer)
        expected_tokenizer = token_manifest["tokenizer"]
        for field in ("class", "special_token_ids", "backend_sha256"):
            if current_tokenizer.get(field) != expected_tokenizer.get(field):
                raise RuntimeError(f"loaded tokenizer differs from token audit: {field}")
        observed_context = int(getattr(model.config, "max_position_embeddings", -1))
        if observed_context != int(token_manifest["max_context_tokens"]):
            raise RuntimeError("model context limit differs from tokenizer audit")

        ledger_subset = ledger[
            ledger["block_work_key"].astype(str).isin(work_keys)
        ].copy()
        audit_subset = token_audit[
            token_audit["block_work_key"].astype(str).isin(work_keys)
        ].copy()

        def process(keys: Sequence[str]) -> pd.DataFrame:
            frame, _current_parity = score_lens_chunk(
                model=model,
                ledger=ledger_subset,
                token_audit=audit_subset,
                work_keys=keys,
                batch_size=int(args.batch_size),
                max_batch_tokens=int(args.max_batch_tokens),
                expected_layers=profile.expected_layers,
                run_scalar_oracle=mode == "startup",
                phase_callback=telemetry.set_phase,
            )
            return frame

        def write_runtime_receipts(
            completed: set[str], shard_path: Path | None
        ) -> None:
            elapsed = max(time.time() - started, 1e-9)
            completed_audit = audit_subset[
                audit_subset["block_work_key"].astype(str).isin(completed)
                & audit_subset["root_evaluable"].astype(bool)
            ]
            root_input_tokens = int(completed_audit["prompt_token_count"].sum())
            branch_input_tokens = 0
            for token_paths in completed_audit["candidate_token_ids"]:
                unique_prefixes = {
                    tuple(int(value) for value in path[:depth])
                    for path in token_paths
                    for depth in range(1, len(path))
                }
                branch_input_tokens += len(unique_prefixes)
            peak_vram = int(torch.cuda.max_memory_allocated())
            telemetry_report = telemetry.snapshot(
                completed_work_units=len(completed),
                root_input_tokens=root_input_tokens,
                branch_input_tokens=branch_input_tokens,
                peak_vram_bytes=peak_vram,
            )
            telemetry_report["covered_work_keys"] = sorted(
                completed - completed_before
            )
            telemetry_report = _merge_telemetry_reports(
                telemetry_base, telemetry_report
            )
            _atomic_json(telemetry_report, telemetry_path)
            rate = len(completed - completed_before) / elapsed
            remaining = len(work_keys) - len(completed)
            progress = {
                "phase": "lens",
                "mode": mode,
                "semantic_run_id": identity.semantic_run_id,
                "completed_work_units": len(completed),
                "total_work_units": len(work_keys),
                "work_units_per_second": rate,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": None if rate <= 0 else remaining / rate,
                "root_prompts": int(len(completed_audit)),
                "root_input_tokens": root_input_tokens,
                "branch_input_tokens": int(branch_input_tokens),
                "padding_tokens": 0,
                "last_shard": (
                    None if shard_path is None else str(shard_path.relative_to(run_root))
                ),
                "peak_vram_bytes": peak_vram,
                "batch_size": int(args.batch_size),
                "max_batch_tokens": int(args.max_batch_tokens),
                "capture_chunk_size": int(args.capture_chunk_size),
                "phase_seconds": telemetry_report["phase_seconds"],
                "gpu_utilization": telemetry_report["gpu_utilization"],
            }
            _atomic_json(progress, run_root / "progress.json")
            print(json.dumps(progress, sort_keys=True), flush=True)

        def on_flush(completed: set[str], shard_path: Path) -> None:
            nonlocal parity
            shard_frame = pd.read_parquet(shard_path / "data.parquet")
            parity = _merge_parity_reports(
                parity, _parity_report_from_frame(shard_frame)
            )
            _atomic_json(parity, parity_path)
            write_runtime_receipts(completed, shard_path)

        processed = run_atomic_chunks(
            store,
            work_keys,
            chunk_size=int(args.capture_chunk_size),
            max_chunks_this_invocation=args.max_chunks_this_invocation,
            process_chunk=process,
            should_stop=lambda: stop_requested,
            on_flush=on_flush,
        )
        completed = store.completed_work_keys()
        reconciled = (
            store.merge(sort_by=["block_work_key", "contract", "layer"])
            if completed
            else pd.DataFrame()
        )
        parity = _parity_report_from_frame(reconciled)
        _atomic_json(parity, parity_path)
        write_runtime_receipts(completed, None)
        is_complete = completed == set(work_keys)
        artifacts: dict[str, object] = {
            "parity_report_sha256": sha256_file(parity_path),
            "progress_sha256": sha256_file(run_root / "progress.json"),
            "telemetry_report_sha256": sha256_file(
                telemetry_path
            ),
            "continuation_audit_sha256": sha256_file(
                run_root / "continuation_audit.parquet"
            ),
        }
        if is_complete:
            merged_path = run_root / "layerwise_scores.parquet"
            reconciled.to_parquet(merged_path, index=False)
            artifacts["layerwise_scores_sha256"] = sha256_file(merged_path)
        status = (
            "startup_complete"
            if is_complete and mode == "startup"
            else "complete" if is_complete else "interrupted"
        )
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": status,
            "mode": mode,
            "semantic_identity": identity.as_dict(),
            "expected_work_keys": work_keys,
            "expected_rows": work_plan["expected_rows"],
            "completed_work_units": len(completed),
            "artifacts": artifacts,
            "claim": CLAIM_BOUNDARY,
            "final_599_opened": False,
        }
        _atomic_json(manifest, run_root / "run_manifest.json")
        attempt.update(
            {
                "status": status,
                "finished_at_unix": time.time(),
                "processed_work_keys": processed,
                "completed_after": len(completed),
                "telemetry_work_keys": processed,
            }
        )
        _atomic_json(attempt, attempt_path)
        return {
            "status": status,
            "mode": mode,
            "semantic_run_id": identity.semantic_run_id,
            "run_root": str(run_root),
            "processed_this_invocation": len(processed),
            "completed_work_units": len(completed),
            "total_work_units": len(work_keys),
        }
    except BaseException as error:
        attempt.update(
            {
                "status": "failed",
                "finished_at_unix": time.time(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _atomic_json(attempt, attempt_path)
        raise
    finally:
        telemetry.stop()
        for value, handler in previous_handlers.items():
            signal.signal(value, handler)
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_design_audit_receipts(
    path: str | Path,
    *,
    expected_design_sha: str,
    expected_applicability_sha: str,
) -> dict[str, str]:
    manifest_path = Path(path)
    manifest = _read_json(manifest_path, name="causal design manifest")
    if (
        manifest.get("sha256") != expected_design_sha
        or manifest.get("applicability", {}).get("sha256")
        != expected_applicability_sha
        or set(map(str, manifest.get("splits", []))) != {"train", "validation"}
        or int(manifest.get("retained_items", -1)) != sum(EXPECTED_SPLIT_ITEMS.values())
    ):
        raise RuntimeError("causal design manifest identity or protected split mismatch")
    option_sha = manifest.get("option_audit", {}).get("manifest_sha256")
    choice_sha = manifest.get("choice_audit", {}).get("manifest_sha256")
    if not isinstance(option_sha, str) or len(option_sha) != 64:
        raise RuntimeError("causal design manifest lacks the option-audit hash")
    if not isinstance(choice_sha, str) or len(choice_sha) != 64:
        raise RuntimeError("causal design manifest lacks the displayed-choice-audit hash")
    return {
        "design_manifest_sha256": sha256_file(manifest_path),
        "option_audit_manifest_sha256": option_sha,
        "choice_audit_manifest_sha256": choice_sha,
    }


def prepare_bundle(
    causal_run_root: str | Path,
    v2_bundle_root: str | Path,
    v3_bundle_root: str | Path,
    design_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    causal_root = Path(causal_run_root)
    run_manifest = _read_json(causal_root / "run_manifest.json", name="causal run manifest")
    identity = _read_json(causal_root / "semantic_identity.json", name="causal identity")
    raw_path = causal_root / "raw" / "causal_behavior.parquet"
    raw_manifest = _read_json(
        raw_path.with_name(raw_path.name + ".manifest.json"), name="causal raw manifest"
    )
    if run_manifest.get("status") != "complete" or run_manifest.get("canary"):
        raise RuntimeError("causal source must be a completed non-canary run")
    if run_manifest.get("semantic_identity") != identity:
        raise RuntimeError("causal run identity mismatch")
    model = identity.get("model", {})
    if model.get("id") != MISTRAL_ID or model.get("revision") != MISTRAL_REVISION:
        raise RuntimeError("causal source is not the pinned Mistral run")
    identity_splits = set(
        map(str, identity.get("experiment_config", {}).get("design_splits", []))
    )
    if identity_splits != {"train", "validation"}:
        raise RuntimeError(
            "causal identity contains a protected split or omits discovery splits"
        )

    v2_root = Path(v2_bundle_root)
    v3_root = Path(v3_bundle_root)
    v2 = _read_json(v2_root / "bundle_manifest.json", name="v2 bundle manifest")
    v3 = _read_json(v3_root / "prepared_manifest.json", name="v3 prepared manifest")
    if v2.get("stage") != "discovery" or v3.get("stage") != "discovery":
        raise RuntimeError("source bundles must remain discovery-only")
    if (
        v2.get("model", {}).get("id") != MISTRAL_ID
        or v2.get("model", {}).get("revision") != MISTRAL_REVISION
        or v2.get("source_scored_sha256") != raw_manifest.get("sha256")
        or v3.get("source_readout_sha256") != v2.get("readout", {}).get("sha256")
        or v3.get("source_model", {}).get("id") != MISTRAL_ID
        or v3.get("source_model", {}).get("revision") != MISTRAL_REVISION
    ):
        raise RuntimeError("source bundle identity chain mismatch")
    audit_receipts = load_design_audit_receipts(
        design_manifest_path,
        expected_design_sha=str(run_manifest.get("design", {}).get("sha256")),
        expected_applicability_sha=str(
            run_manifest.get("design", {}).get("applicability_sha256")
        ),
    )
    if audit_receipts["option_audit_manifest_sha256"] != v3.get(
        "option_audit_manifest_sha256"
    ):
        raise RuntimeError("v3 bundle and causal design disagree on option-audit identity")

    # Only after every metadata identity and protected-split receipt has passed
    # may the scored causal table be opened.
    if (
        raw_manifest.get("semantic_run_id") != identity.get("semantic_run_id")
        or raw_manifest.get("sha256") != sha256_file(raw_path)
    ):
        raise RuntimeError("causal scored artifact identity or checksum mismatch")
    source = pd.read_parquet(raw_path)
    if len(source) != int(raw_manifest.get("row_count", -1)):
        raise RuntimeError("causal scored artifact row count mismatch")

    ledger = build_source_ledger(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "prompt_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    split_items = (
        ledger[["item_id", "split"]]
        .drop_duplicates()
        .groupby("split")["item_id"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    source_receipts = {
        "causal_semantic_run_id": identity["semantic_run_id"],
        "causal_semantic_sha256": identity["semantic_sha256"],
        "causal_scored_sha256": raw_manifest["sha256"],
        "causal_design_sha256": run_manifest.get("design", {}).get("sha256"),
        "causal_applicability_sha256": run_manifest.get("design", {}).get(
            "applicability_sha256"
        ),
        "v2_bundle_manifest_sha256": sha256_file(v2_root / "bundle_manifest.json"),
        "v2_readout_sha256": v2["readout"]["sha256"],
        "v3_prepared_manifest_sha256": sha256_file(v3_root / "prepared_manifest.json"),
        "option_audit_manifest_sha256": v3["option_audit_manifest_sha256"],
        **audit_receipts,
        "scientific_north_star_sha256": sha256_file(
            PROJECT_ROOT / "SCIENTIFIC_NORTH_STAR.md"
        ),
    }
    manifest = {
        "schema_version": 1,
        "stage": "discovery",
        "experiment": "mistral_two_contract_logit_lens_v4",
        "model": {"id": MISTRAL_ID, "revision": MISTRAL_REVISION, "slug": MISTRAL_SLUG},
        "rows": len(ledger),
        "items": int(ledger["item_id"].nunique()),
        "split_items": {str(k): int(v) for k, v in split_items.items()},
        "formats": list(EXPECTED_FORMATS),
        "ledger_sha256": sha256_file(ledger_path),
        "ordered_work_keys_sha256": _canonical_sha256(
            ledger["block_work_key"].astype(str).tolist()
        ),
        "prompt_hashes_sha256": _canonical_sha256(
            ledger[
                [
                    "block_work_key",
                    "letter_prompt_sha256",
                    "text_prompt_sha256",
                    "calibration_prompt_sha256",
                ]
            ].to_dict("records")
        ),
        "source_receipts": source_receipts,
        "final_599_opened": False,
    }
    _atomic_json(manifest, output / "prepared_manifest.json")
    return manifest


def _frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if len(left) != len(right) or set(left.columns) != set(right.columns):
        return False
    columns = sorted(left.columns)

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported nested Parquet value: {type(value).__name__}")

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame[columns].copy()
        for column in columns:
            if result[column].dtype != object:
                continue
            result[column] = result[column].map(
                lambda value: (
                    "nested:"
                    + json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        default=json_default,
                    )
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else "scalar:" + json.dumps(value, default=json_default)
                )
            )
        return result

    left_hash = pd.util.hash_pandas_object(normalized(left), index=False).value_counts().sort_index()
    right_hash = pd.util.hash_pandas_object(normalized(right), index=False).value_counts().sort_index()
    return left_hash.equals(right_hash)


def _load_lens_shards(
    root: Path, identity: Mapping[str, object]
) -> tuple[set[str], pd.DataFrame]:
    keys: set[str] = set()
    exact_seen: set[tuple[str, tuple[str, ...]]] = set()
    frames: list[pd.DataFrame] = []
    phase_root = root / "shards" / "lens"
    for shard in sorted(phase_root.glob("shard-*")):
        data_path = shard / "data.parquet"
        metadata = _read_json(shard / "manifest.json", name="lens shard manifest")
        if not data_path.exists():
            raise ValueError("lens shard is incomplete")
        if (
            metadata.get("semantic_run_id") != identity.get("semantic_run_id")
            or metadata.get("semantic_sha256") != identity.get("semantic_sha256")
            or metadata.get("model_id") != identity.get("model", {}).get("id")
            or metadata.get("model_revision") != identity.get("model", {}).get("revision")
        ):
            raise ValueError("lens shard identity mismatch")
        digest = sha256_file(data_path)
        if digest != metadata.get("data_sha256"):
            raise ValueError("lens shard checksum mismatch")
        work_keys = tuple(str(value) for value in metadata.get("work_keys", []))
        if not work_keys or len(set(work_keys)) != len(work_keys):
            raise ValueError("lens shard work keys are empty or duplicated")
        exact_key = (digest, work_keys)
        if exact_key in exact_seen:
            continue
        if keys.intersection(work_keys):
            raise ValueError("lens shards contain conflicting work keys")
        frame = pd.read_parquet(data_path)
        if len(frame) != int(metadata.get("row_count", -1)):
            raise ValueError("lens shard row count mismatch")
        if "_work_key" not in frame or set(frame["_work_key"].astype(str)) != set(work_keys):
            raise ValueError("lens shard rows do not match declared work keys")
        exact_seen.add(exact_key)
        keys.update(work_keys)
        frames.append(frame)
    return keys, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _four_array(value: object, *, column: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} contains a malformed four-way value") from error
    if array.shape != (4,) or np.isinf(array).any():
        raise ValueError(f"{column} contains a malformed or infinite four-way value")
    return array


def _four_bools(value: object, *, column: str) -> np.ndarray:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 4:
        raise ValueError(f"{column} contains a malformed four-way value")
    if any(not isinstance(item, (bool, np.bool_)) for item in value):
        raise ValueError(f"{column} must contain exactly four booleans")
    return np.asarray(value, dtype=bool)


def _validate_score_vectors(frame: pd.DataFrame) -> None:
    required = {
        "letter_raw_logps",
        "candidate_path_total_logps",
        "candidate_mean_token_logps",
        "candidate_first_token_logps",
        "letter_calibrated_logps",
        "candidate_path_evaluable",
        "primary_contrast_evaluable",
        "first_token_contrast_evaluable",
        "letter_calibration_evaluable",
        "scalar_oracle_evaluated",
        "primary_contrast_exclusion_reason",
        "first_token_contrast_exclusion_reason",
        "letter_calibration_exclusion_reason",
        *PARITY_COLUMNS.values(),
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"lens score artifact is missing columns: {sorted(missing)}")
    for column in (
        "primary_contrast_evaluable",
        "first_token_contrast_evaluable",
        "letter_calibration_evaluable",
        "scalar_oracle_evaluated",
    ):
        if frame[column].isna().any() or not frame[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).all():
            raise ValueError(f"{column} must be boolean and non-null")
    if (frame["first_token_contrast_evaluable"] & ~frame["primary_contrast_evaluable"]).any():
        raise ValueError("first-token eligibility cannot exceed primary eligibility")
    for row in frame.itertuples(index=False):
        letter = _four_array(row.letter_raw_logps, column="letter_raw_logps")
        if not np.isfinite(letter).all():
            raise ValueError("lens scores contain non-finite letter_raw_logps")
        path_evaluable = _four_bools(
            row.candidate_path_evaluable, column="candidate_path_evaluable"
        )
        for column in (
            "candidate_path_total_logps",
            "candidate_mean_token_logps",
            "candidate_first_token_logps",
        ):
            values = _four_array(getattr(row, column), column=column)
            finite = np.isfinite(values)
            if not np.array_equal(finite, path_evaluable):
                raise ValueError(
                    f"lens scores contain non-finite candidate path values in {column}"
                )
        calibrated = _four_array(
            row.letter_calibrated_logps, column="letter_calibrated_logps"
        )
        if (
            str(row.contract) == "letter"
            and bool(row.letter_calibration_evaluable)
            and not np.isfinite(calibrated).all()
        ):
            raise ValueError("lens scores contain non-finite eligible calibrated-letter values")
    stable_columns = (
        "primary_contrast_evaluable",
        "primary_contrast_exclusion_reason",
        "first_token_contrast_evaluable",
        "first_token_contrast_exclusion_reason",
        "letter_calibration_evaluable",
        "letter_calibration_exclusion_reason",
        "candidate_path_evaluable",
        "scalar_oracle_evaluated",
        *PARITY_COLUMNS.values(),
    )
    for key, group in frame.groupby(["block_work_key", "contract"], sort=False):
        for column in stable_columns:
            values = group[column]
            if column == "candidate_path_evaluable":
                values = values.map(
                    lambda value: json.dumps([bool(item) for item in value])
                )
            if values.nunique(dropna=False) != 1:
                raise ValueError(f"eligibility state changes across layers: {key}/{column}")


def _validate_semantic_identity_digest(identity: Mapping[str, object]) -> None:
    payload = {
        str(key): value
        for key, value in identity.items()
        if key not in {"semantic_run_id", "semantic_sha256"}
    }
    digest = _canonical_sha256(payload)
    if (
        identity.get("semantic_sha256") != digest
        or identity.get("semantic_run_id") != digest[:20]
    ):
        raise ValueError("semantic identity digest or run ID is invalid")


def verify_run_root(
    root: str | Path,
    *,
    expected_run_id: str,
    mode: str,
    require_analysis: bool = True,
) -> dict[str, object]:
    if mode not in {"partial", "startup", "complete"}:
        raise ValueError("verification mode must be partial, startup, or complete")
    run_root = Path(root)
    identity = _read_json(run_root / "semantic_identity.json", name="semantic identity")
    manifest = _read_json(run_root / "run_manifest.json", name="run manifest")
    _validate_semantic_identity_digest(identity)
    if identity.get("semantic_run_id") != expected_run_id:
        raise ValueError("semantic run ID mismatch")
    if manifest.get("semantic_identity") != identity:
        raise ValueError("run manifest identity mismatch")
    if manifest.get("claim") != CLAIM_BOUNDARY:
        raise ValueError("run manifest violates the bounded claim contract")
    if manifest.get("final_599_opened", False) is not False:
        raise ValueError("run manifest violates the protected final-set boundary")
    environment = identity.get("experiment_config", {}).get("runtime_environment", {})
    required_environment = {
        "device_type", "gpu_name", "gpu_compute_capability", "bf16_supported",
        "attention_backend", "torch_version", "cuda_version", "transformers_version",
        "tokenizers_version", "inference_dtype",
    }
    if (
        set(environment) != required_environment
        or environment.get("device_type") != "cuda"
        or not environment.get("gpu_name")
        or not environment.get("gpu_compute_capability")
        or environment.get("bf16_supported") is not True
        or environment.get("attention_backend") != "sdpa"
        or environment.get("inference_dtype") != "bfloat16"
    ):
        raise ValueError("runtime environment identity is incomplete or invalid")
    experiment_config = identity.get("experiment_config", {})
    input_receipts = (
        ("prepared_manifest.json", "prepared_manifest_sha256"),
        ("continuation_audit.parquet", "continuation_audit_sha256"),
        ("tokenization_manifest.json", "tokenization_manifest_sha256"),
    )
    for filename, field in input_receipts:
        path = run_root / filename
        if not path.exists() or sha256_file(path) != experiment_config.get(field):
            raise ValueError(f"run input receipt mismatch: {filename}")
    token_manifest = _read_json(
        run_root / "tokenization_manifest.json", name="tokenization manifest"
    )
    if token_manifest.get("audit_sha256") != experiment_config.get(
        "continuation_audit_sha256"
    ):
        raise ValueError("tokenization manifest does not bind the continuation audit")
    continuation_audit = pd.read_parquet(run_root / "continuation_audit.parquet")
    audit_primary = continuation_audit[
        continuation_audit["contract"].astype(str).isin(["letter", "text"])
    ].copy()
    if audit_primary.duplicated(["block_work_key", "contract"]).any():
        raise ValueError("continuation audit has duplicate primary coordinates")
    analysis_spec = _read_json(run_root / "analysis_spec.json", name="analysis specification")
    if analysis_spec != experiment_config.get("analysis_policy"):
        raise ValueError("analysis specification differs from semantic identity")
    work_plan = _read_json(run_root / "work_plan.json", name="work plan")
    work_keys = [str(value) for value in work_plan.get("work_keys", [])]
    if (
        work_plan.get("ordered_work_keys_sha256")
        != experiment_config.get("ordered_work_keys_sha256")
        or _canonical_sha256(work_keys) != work_plan.get("ordered_work_keys_sha256")
        or work_keys != [str(value) for value in manifest.get("expected_work_keys", [])]
        or int(work_plan.get("expected_rows", -1))
        != int(manifest.get("expected_rows", -1))
    ):
        raise ValueError("work plan differs from semantic identity or run manifest")

    shard_keys, shard_frame = _load_lens_shards(run_root, identity)
    merged_path = run_root / "layerwise_scores.parquet"
    merged = None
    if merged_path.exists():
        expected_sha = manifest.get("artifacts", {}).get("layerwise_scores_sha256")
        if expected_sha is not None and sha256_file(merged_path) != expected_sha:
            raise ValueError("merged lens artifact checksum mismatch")
        merged = pd.read_parquet(merged_path)
        if not _frames_match(merged, shard_frame):
            raise ValueError("merged lens artifact does not reconcile with shards")
    elif mode in {"startup", "complete"}:
        raise ValueError("complete verification requires the merged lens artifact")
    frame = shard_frame if merged is None else merged
    if not frame.empty:
        _validate_score_vectors(frame)
        score_masks = frame[
            ["block_work_key", "contract", "candidate_path_evaluable"]
        ].drop_duplicates(["block_work_key", "contract"])
        audit_masks = audit_primary[
            ["block_work_key", "contract", "candidate_path_evaluable"]
        ]
        mask_comparison = score_masks.merge(
            audit_masks,
            on=["block_work_key", "contract"],
            how="left",
            suffixes=("_score", "_audit"),
            validate="one_to_one",
            indicator=True,
        )
        if (mask_comparison["_merge"] != "both").any() or any(
            not np.array_equal(
                _four_bools(score, column="candidate_path_evaluable_score"),
                _four_bools(audit, column="candidate_path_evaluable_audit"),
            )
            for score, audit in zip(
                mask_comparison["candidate_path_evaluable_score"],
                mask_comparison["candidate_path_evaluable_audit"],
                strict=True,
            )
        ):
            raise ValueError(
                "score eligibility differs from the authenticated continuation audit"
            )
    structural = ["block_work_key", "contract", "layer"]
    if set(structural) - set(frame.columns) or frame.duplicated(structural).any():
        raise ValueError("lens scores contain missing or duplicate structural coordinates")
    expected_layers = set(range(int(identity.get("model", {}).get("expected_layers", -1))))
    expected_structure = {
        (contract, layer) for contract in ("letter", "text") for layer in expected_layers
    }
    for _, group in frame.groupby("block_work_key", sort=False):
        observed = set(zip(group["contract"].astype(str), group["layer"].astype(int), strict=True))
        if observed != expected_structure:
            raise ValueError("lens scores do not form a complete contract-layer product")

    status = str(manifest.get("status"))
    if mode == "complete" and status != "complete":
        raise ValueError("run is not complete")
    if mode == "startup" and status != "startup_complete":
        raise ValueError("startup run is not complete")
    expected_keys = {str(value) for value in manifest.get("expected_work_keys", [])}
    if mode in {"startup", "complete"} and shard_keys != expected_keys:
        raise ValueError("completed work keys do not match the work plan")
    if mode in {"startup", "complete"} and len(frame) != int(manifest.get("expected_rows", -1)):
        raise ValueError("completed row count does not match the work plan")
    if int(manifest.get("completed_work_units", -1)) != len(shard_keys):
        raise ValueError("completed work-unit count mismatch")
    verified_prefix = work_keys[: len(shard_keys)]
    if shard_keys != set(verified_prefix):
        raise ValueError("verified shards are not the deterministic work-plan prefix")
    if frame.empty and mode in {"startup", "complete"}:
        raise ValueError("completed run contains no score rows")
    parity_path = run_root / "parity_report.json"
    if frame.empty and mode == "partial":
        parity = None
    else:
        if (
            not parity_path.exists()
            or sha256_file(parity_path)
            != manifest.get("artifacts", {}).get("parity_report_sha256")
        ):
            raise ValueError("parity report is missing or checksum-mismatched")
        parity = _read_json(parity_path, name="parity report")
        if float(parity.get("tolerance", -1.0)) != PARITY_ATOL:
            raise ValueError("parity report tolerance drift")
        if parity.get("cross_forward_sensitivity_policy") != CROSS_FORWARD_SENSITIVITY_POLICY:
            raise ValueError("cross-forward sensitivity policy drift")
        maxima = [
            float(value)
            for key, value in parity.items()
            if str(key).startswith("max_")
        ]
        if not maxima or not np.isfinite(maxima).all():
            raise ValueError("parity report contains non-finite sensitivity values")
        if float(parity.get("max_final_native_difference", math.inf)) > PARITY_ATOL:
            raise ValueError("same-forward final/native parity exceeds the frozen tolerance")
        expected_parity = _parity_report_from_frame(frame)
        if parity != expected_parity:
            raise ValueError("parity coverage or maxima do not reconcile with score shards")
        if (
            experiment_config.get("mode") == "startup"
            and set(map(str, parity.get("scalar_oracle_work_keys", []))) != shard_keys
        ):
            raise ValueError("startup scalar parity coverage is incomplete")

        artifacts = manifest.get("artifacts", {})
        progress_path = run_root / "progress.json"
        telemetry_path = run_root / "telemetry_report.json"
        if (
            not progress_path.exists()
            or sha256_file(progress_path) != artifacts.get("progress_sha256")
            or not telemetry_path.exists()
            or sha256_file(telemetry_path) != artifacts.get("telemetry_report_sha256")
        ):
            raise ValueError("runtime telemetry receipts are missing or checksum-mismatched")
        progress = _read_json(progress_path, name="progress receipt")
        telemetry = _read_json(telemetry_path, name="telemetry report")
        if (
            progress.get("semantic_run_id") != identity.get("semantic_run_id")
            or telemetry.get("semantic_run_id") != identity.get("semantic_run_id")
            or int(progress.get("completed_work_units", -1)) != len(shard_keys)
            or int(telemetry.get("completed_work_units", -1)) != len(shard_keys)
            or set(map(str, telemetry.get("covered_work_keys", []))) != shard_keys
            or int(telemetry.get("root_input_tokens", -1)) <= 0
            or int(telemetry.get("branch_input_tokens", -1)) <= 0
            or int(telemetry.get("peak_vram_bytes", -1)) < 0
        ):
            raise ValueError("runtime telemetry does not cover the verified shards")
        phase_seconds = telemetry.get("phase_seconds", {})
        utilization = telemetry.get("gpu_utilization", {})
        for phase in ("root", "branch"):
            seconds = float(phase_seconds.get(phase, -1.0))
            phase_utilization = utilization.get(phase, {})
            samples = int(phase_utilization.get("samples", -1))
            mean = phase_utilization.get("mean_percent")
            maximum = phase_utilization.get("max_percent")
            if (
                not math.isfinite(seconds)
                or seconds <= 0
                or samples < 1
                or mean is None
                or maximum is None
                or not 0.0 <= float(mean) <= 100.0
                or not 0.0 <= float(maximum) <= 100.0
            ):
                raise ValueError(f"runtime telemetry lacks valid {phase} utilization")
        attempt_receipts = []
        for path in sorted((run_root / "attempts").glob("*.json")):
            receipt = _read_json(path, name="attempt receipt")
            if receipt.get("semantic_run_id") == identity.get("semantic_run_id"):
                attempt_receipts.append(receipt)
        _validate_attempt_chain(
            attempt_receipts,
            verified_prefix,
            mode=str(experiment_config.get("mode", "full")),
        )

    analysis_artifacts = manifest.get("artifacts", {}).get("analysis")
    if mode == "complete" and require_analysis and analysis_artifacts is None:
        raise ValueError("analysis artifact manifest is required for complete verification")
    if analysis_artifacts is not None:
        if set(analysis_artifacts) != set(EXPECTED_ANALYSIS_ARTIFACTS):
            raise ValueError("analysis artifact manifest is incomplete or contains extras")
        for name, receipt in analysis_artifacts.items():
            relative = Path(str(receipt.get("path", "")))
            if relative.as_posix() != EXPECTED_ANALYSIS_ARTIFACTS[name]:
                raise ValueError(f"analysis artifact does not use its canonical path: {name}")
            path = run_root / relative
            if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
                raise ValueError(f"analysis artifact checksum mismatch: {name}")
    sums = []
    for path in sorted(run_root.rglob("*")):
        if path.is_file() and path.name != "LOCAL_SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.relative_to(run_root)}")
    (run_root / "LOCAL_SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return {"status": status, "work_units": len(shard_keys), "rows": len(frame)}


def _cmd_prepare(args: argparse.Namespace) -> None:
    manifest = prepare_bundle(
        args.causal_run,
        args.v2_bundle,
        args.v3_bundle,
        args.design_manifest,
        args.output_dir,
    )
    print(json.dumps(manifest, sort_keys=True))


def _cmd_audit_tokenizer(args: argparse.Namespace) -> None:
    if args.profile != "mistral":
        raise ValueError("the tokenizer audit is frozen to Mistral only")
    from transformers import AutoConfig, AutoTokenizer

    profile = get_model_profile(args.profile)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.tokenizer_id,
        revision=profile.tokenizer_revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=profile.trust_remote_code,
        use_fast=True,
    )
    config = AutoConfig.from_pretrained(
        profile.model_id,
        revision=profile.revision,
        local_files_only=bool(args.local_files_only),
        trust_remote_code=profile.trust_remote_code,
    )
    max_context_tokens = int(getattr(config, "max_position_embeddings", -1))
    manifest = audit_tokenizer_bundle(
        args.bundle,
        args.output,
        tokenizer=tokenizer,
        max_context_tokens=max_context_tokens,
    )
    print(json.dumps(manifest, sort_keys=True))


def _cmd_run_model(args: argparse.Namespace) -> None:
    print(json.dumps(execute_model_run(args), sort_keys=True))


def _load_analysis_module():
    import importlib.util
    import sys

    path = PROJECT_ROOT / "analysis" / "analyze_decision_binding_logit_lens.py"
    spec = importlib.util.spec_from_file_location(
        "analyze_decision_binding_logit_lens_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen logit-lens analysis")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cmd_analyze(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    identity = _read_json(run_root / "semantic_identity.json", name="semantic identity")
    run_id = str(identity.get("semantic_run_id", ""))
    verify_run_root(
        run_root,
        expected_run_id=run_id,
        mode="complete",
        require_analysis=False,
    )
    module = _load_analysis_module()
    output_dir = run_root / "analysis"
    outputs = module.analyze(
        run_root / "layerwise_scores.parquet",
        output_dir,
        n_boot=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    manifest = _read_json(run_root / "run_manifest.json", name="run manifest")
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts["analysis"] = {
        str(name): {
            "path": str(path.relative_to(run_root)),
            "sha256": sha256_file(path),
        }
        for name, path in outputs.items()
    }
    manifest["artifacts"] = artifacts
    _atomic_json(manifest, run_root / "run_manifest.json")
    verify_run_root(run_root, expected_run_id=run_id, mode="complete")
    print(json.dumps({name: str(path) for name, path in outputs.items()}, sort_keys=True))


def _cmd_verify(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            verify_run_root(
                args.run_root, expected_run_id=args.run_id, mode=args.mode
            ),
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interface-formatting-decision-logit-lens")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--causal-run", required=True)
    prepare.add_argument("--v2-bundle", required=True)
    prepare.add_argument("--v3-bundle", required=True)
    prepare.add_argument("--design-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=_cmd_prepare)

    audit = sub.add_parser("audit-tokenizer")
    audit.add_argument("--profile", choices=["mistral"], default="mistral")
    audit.add_argument("--bundle", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--local-files-only", action="store_true")
    audit.set_defaults(func=_cmd_audit_tokenizer)

    run = sub.add_parser("run-model")
    run.add_argument("--profile", choices=["mistral"], default="mistral")
    run.add_argument("--bundle", required=True)
    run.add_argument("--token-audit", required=True)
    run.add_argument("--output-base", default="results/decision_binding_logit_lens_runs")
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-batch-tokens", type=int, default=40000)
    run.add_argument("--capture-chunk-size", type=int, default=64)
    run.add_argument("--startup-items", type=int)
    run.add_argument("--max-chunks-this-invocation", type=int)
    run.add_argument("--local-files-only", action="store_true")
    run.set_defaults(func=_cmd_run_model)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--run-root", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, choices=[5000], default=5000)
    analyze.add_argument("--seed", type=int, choices=[1729], default=1729)
    analyze.set_defaults(func=_cmd_analyze)

    verify = sub.add_parser("verify")
    verify.add_argument("--run-root", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--mode", choices=["partial", "startup", "complete"], required=True)
    verify.set_defaults(func=_cmd_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
