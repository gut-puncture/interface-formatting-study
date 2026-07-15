from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Callable

import pandas as pd

from .run_identity import SemanticIdentity, sha256_file
from .calibration import calibrate_scores
from .scoring import (
    ScoringTelemetry,
    correct_answer_margin,
    predict_from_scores,
    score_candidate_sets_many,
    score_labels_many,
)
from .shards import ShardStore
from .utils import EPSILON, write_table_atomic


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_causal_design(design: pd.DataFrame) -> None:
    required = {
        "work_key",
        "item_id",
        "split",
        "wrapper_name",
        "arm",
        "variant",
        "position_shift",
        "label_shift",
        "manipulation",
        "prompt",
        "prompt_sha256",
        "template_scope",
        "source_prompt_sha256",
        "calibration_prompt",
        "calibration_prompt_sha256",
        "content_ids_by_position",
        "labels_by_position",
        "candidate_texts",
        "correct_content_id",
        "correct_position",
        "correct_label",
        "correct_text",
    }
    missing = required - set(design.columns)
    if missing:
        raise ValueError(f"Causal design is missing columns: {sorted(missing)}")
    if design.empty:
        raise ValueError("Causal design is empty")
    if design["work_key"].duplicated().any():
        raise ValueError("Causal design contains duplicate work keys")
    if set(design["arm"]) - {"letter_intervention", "answer_text"}:
        raise ValueError("Causal design contains an unknown experiment arm")
    if set(design["template_scope"].astype(str)) != {"source_prompt_counterfactual"}:
        raise ValueError("Causal design template_scope is not source-preserving")
    for row in design.itertuples(index=False):
        contents = [int(value) for value in row.content_ids_by_position]
        labels = [str(value) for value in row.labels_by_position]
        candidates = [str(value) for value in row.candidate_texts]
        if sorted(contents) != [0, 1, 2, 3] or sorted(labels) != ["A", "B", "C", "D"]:
            raise ValueError(f"Causal design has an invalid assignment for {row.work_key}")
        if len(candidates) != 4:
            raise ValueError(f"Causal design requires four candidate texts for {row.work_key}")
        correct_content = int(row.correct_content_id)
        correct_position = contents.index(correct_content)
        if int(row.correct_position) != correct_position:
            raise ValueError(f"Causal design has an inconsistent correct_position for {row.work_key}")
        if str(row.correct_label) != labels[correct_position]:
            raise ValueError(f"Causal design has an inconsistent correct_label for {row.work_key}")
        if str(row.correct_text) != candidates[correct_content]:
            raise ValueError(f"Causal design has an inconsistent correct_text for {row.work_key}")
        actual_prompt_sha = hashlib.sha256(str(row.prompt).encode("utf-8")).hexdigest()
        if str(row.prompt_sha256) != actual_prompt_sha:
            raise ValueError(f"Causal design has a prompt checksum mismatch for {row.work_key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.source_prompt_sha256)):
            raise ValueError(f"Causal design has an invalid source prompt checksum for {row.work_key}")
        if row.manipulation == "controlled_baseline" and row.arm == "letter_intervention":
            if str(row.prompt_sha256) != str(row.source_prompt_sha256):
                raise ValueError(f"Causal baseline does not match its source prompt for {row.work_key}")
        if row.arm == "letter_intervention":
            if not isinstance(row.calibration_prompt, str) or not row.calibration_prompt:
                raise ValueError(f"Letter row is missing calibration_prompt for {row.work_key}")
            calibration_sha = hashlib.sha256(row.calibration_prompt.encode("utf-8")).hexdigest()
            if str(row.calibration_prompt_sha256) != calibration_sha:
                raise ValueError(f"Causal design has a calibration checksum mismatch for {row.work_key}")
        if row.arm == "answer_text" and int(row.variant) != 0:
            raise ValueError("Answer-text rows must use the identity assignment")

    for (item_id, wrapper_name), group in design.groupby(["item_id", "wrapper_name"], sort=True):
        letter = group[group["arm"] == "letter_intervention"]
        text = group[group["arm"] == "answer_text"]
        if set(letter["variant"].astype(int)) != set(range(7)) or len(letter) != 7 or len(text) != 1:
            raise ValueError(
                f"Causal design requires seven letter variants and one text row for {item_id}/{wrapper_name}"
            )
        expected = {"controlled_baseline": 1, "position_only": 3, "label_only": 3}
        if letter["manipulation"].value_counts().to_dict() != expected:
            raise ValueError(f"Causal design manipulations are not isolated for {item_id}/{wrapper_name}")
        identity = letter[letter["manipulation"] == "controlled_baseline"]
        position = letter[letter["manipulation"] == "position_only"]
        label = letter[letter["manipulation"] == "label_only"]
        if set(position["label_shift"].astype(int)) != {0} or set(label["position_shift"].astype(int)) != {0}:
            raise ValueError(f"Causal design main effects are confounded for {item_id}/{wrapper_name}")
        for content_id in range(4):
            position_block = pd.concat([identity, position])
            positions = [list(values).index(content_id) for values in position_block["content_ids_by_position"]]
            labels = [
                list(labels_by_position)[position]
                for labels_by_position, position in zip(
                    pd.concat([identity, label])["labels_by_position"],
                    [list(values).index(content_id) for values in pd.concat([identity, label])["content_ids_by_position"]],
                    strict=True,
                )
            ]
            if sorted(positions) != [0, 1, 2, 3] or sorted(labels) != ["A", "B", "C", "D"]:
                raise ValueError(f"Causal design is not balanced for {item_id}/{wrapper_name}")


def _choice_entropy(scores: dict[str, float]) -> float:
    values = list(scores.values())
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return float(-sum(value * math.log(value) for value in probabilities if value > 0) / math.log(4))


def _score_letter_rows(
    frame: pd.DataFrame,
    model,
    tokenizer,
    *,
    batch_size: int,
    max_batch_tokens: int | None,
    device,
    telemetry: ScoringTelemetry,
) -> list[dict[str, object]]:
    records = frame.to_dict("records")
    prompts = [str(row["prompt"]) for row in records]
    calibration_prompts = [str(row["calibration_prompt"]) for row in records]
    all_scores = score_labels_many(
        model,
        tokenizer,
        [*prompts, *calibration_prompts],
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        device=device,
        telemetry=telemetry,
    )
    raw_scores_by_row = all_scores[: len(records)]
    bias_scores_by_row = all_scores[len(records) :]
    output = []
    for row, raw_scores, bias_scores in zip(records, raw_scores_by_row, bias_scores_by_row, strict=True):
        cal_scores = calibrate_scores(raw_scores, bias_scores)
        raw_predicted_label, raw_tie = predict_from_scores(raw_scores)
        cal_predicted_label, cal_tie = predict_from_scores(cal_scores)
        labels = [str(value) for value in row["labels_by_position"]]
        contents = [int(value) for value in row["content_ids_by_position"]]
        raw_predicted_position = labels.index(raw_predicted_label)
        cal_predicted_position = labels.index(cal_predicted_label)
        raw_predicted_content = contents[raw_predicted_position]
        cal_predicted_content = contents[cal_predicted_position]
        raw_margin = correct_answer_margin(raw_scores, str(row["correct_label"]))
        cal_margin = correct_answer_margin(cal_scores, str(row["correct_label"]))
        output.append(
            {
                **row,
                **{f"raw_score_{label}": float(raw_scores[label]) for label in ("A", "B", "C", "D")},
                **{f"bias_score_{label}": float(bias_scores[label]) for label in ("A", "B", "C", "D")},
                **{f"cal_score_{label}": float(cal_scores[label]) for label in ("A", "B", "C", "D")},
                "raw_predicted_label": raw_predicted_label,
                "raw_predicted_position": raw_predicted_position,
                "raw_predicted_content_id": raw_predicted_content,
                "raw_correct": bool(raw_predicted_content == int(row["correct_content_id"]) and not raw_tie),
                "raw_tie": bool(raw_tie),
                "raw_margin": float(raw_margin.margin),
                "raw_entropy": _choice_entropy(raw_scores),
                "cal_predicted_label": cal_predicted_label,
                "cal_predicted_position": cal_predicted_position,
                "cal_predicted_content_id": cal_predicted_content,
                "cal_correct": bool(cal_predicted_content == int(row["correct_content_id"]) and not cal_tie),
                "cal_tie": bool(cal_tie),
                "cal_margin": float(cal_margin.margin),
                "cal_entropy": _choice_entropy(cal_scores),
                "text_ambiguous": False,
            }
        )
    return output


def _normalize_generated_answer(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip())


def _generate_answers_many(
    model,
    tokenizer,
    records: list[dict[str, object]],
    *,
    batch_size: int,
    max_batch_tokens: int | None,
    device,
    telemetry: ScoringTelemetry,
) -> list[str]:
    import torch

    prompt_lengths = [len(tokenizer.encode(str(row["prompt"]), add_special_tokens=False)) for row in records]
    generation_limits = [
        min(
            192,
            max(
                16,
                max(
                    len(tokenizer.encode(str(candidate), add_special_tokens=False))
                    for candidate in row["candidate_texts"]
                )
                + 8,
            ),
        )
        for row in records
    ]
    answers: list[str | None] = [None] * len(records)
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"

    def generate(indices: list[int]) -> None:
        if not indices:
            return
        preparation_started = time.monotonic()
        prompts = [str(records[index]["prompt"]) for index in indices]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        max_new_tokens = max(generation_limits[index] for index in indices)
        telemetry.batches += 1
        telemetry.actual_tokens += sum(prompt_lengths[index] + generation_limits[index] for index in indices)
        telemetry.padded_tokens += len(indices) * (encoded["input_ids"].shape[1] + max_new_tokens)
        telemetry.input_preparation_seconds += time.monotonic() - preparation_started
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            if len(indices) == 1:
                raise
            torch.cuda.empty_cache()
            midpoint = len(indices) // 2
            generate(indices[:midpoint])
            generate(indices[midpoint:])
            return
        suffix = generated[:, encoded["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        for index, answer in zip(indices, decoded, strict=True):
            answers[index] = answer

    try:
        ordered = sorted(
            range(len(records)),
            key=lambda index: (prompt_lengths[index] + generation_limits[index], index),
        )
        current: list[int] = []
        current_max = 0
        for index in ordered:
            length = prompt_lengths[index] + generation_limits[index]
            candidate_max = max(current_max, length)
            count_exceeded = len(current) >= batch_size
            token_exceeded = (
                max_batch_tokens is not None
                and current
                and candidate_max * (len(current) + 1) > max_batch_tokens
            )
            if count_exceeded or token_exceeded:
                generate(current)
                current = []
                current_max = 0
            current.append(index)
            current_max = max(current_max, length)
        generate(current)
    finally:
        tokenizer.padding_side = old_padding_side
    if any(answer is None for answer in answers):
        raise RuntimeError("Generation failed to produce every answer")
    return [str(answer) for answer in answers]


def _score_text_rows(
    frame: pd.DataFrame,
    model,
    tokenizer,
    *,
    batch_size: int,
    max_batch_tokens: int | None,
    device,
    telemetry: ScoringTelemetry,
) -> list[dict[str, object]]:
    records = frame.to_dict("records")
    candidate_sets = [[str(value) for value in row["candidate_texts"]] for row in records]
    scored = score_candidate_sets_many(
        model,
        tokenizer,
        [str(row["prompt"]) for row in records],
        candidate_sets,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        device=device,
        telemetry=telemetry,
    )
    generated_answers = _generate_answers_many(
        model,
        tokenizer,
        records,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        device=device,
        telemetry=telemetry,
    )
    output = []
    for row, candidates, candidate_scores, generated in zip(
        records, candidate_sets, scored, generated_answers, strict=True
    ):
        totals = [float(score.total_logp) for score in candidate_scores]
        ranked = sorted(range(4), key=lambda index: (-totals[index], index))
        predicted_content = ranked[0]
        is_tie = abs(totals[ranked[0]] - totals[ranked[1]]) <= EPSILON
        correct_content = int(row["correct_content_id"])
        best_wrong = max(score for index, score in enumerate(totals) if index != correct_content)
        ambiguous = len(set(candidates)) != len(candidates)
        first_line = next((line.strip() for line in generated.splitlines() if line.strip()), "")
        generated_correct = _normalize_generated_answer(first_line) == _normalize_generated_answer(
            str(row["correct_text"])
        )
        output.append(
            {
                **row,
                "candidate_total_logps": totals,
                "candidate_mean_logps": [float(score.mean_logp) for score in candidate_scores],
                "candidate_token_counts": [int(score.token_count) for score in candidate_scores],
                "generated_text": generated,
                "generated_answer": first_line,
                "generated_correct": bool(generated_correct),
                "candidate_predicted_content_id": predicted_content,
                "candidate_total_correct": None if ambiguous else bool(predicted_content == correct_content and not is_tie),
                "candidate_total_tie": bool(is_tie),
                "candidate_total_margin": float(totals[correct_content] - best_wrong),
                "text_ambiguous": ambiguous,
            }
        )
    return output


def _score_chunk(
    frame: pd.DataFrame,
    model,
    tokenizer,
    *,
    batch_size: int,
    max_batch_tokens: int | None,
    device,
    telemetry: ScoringTelemetry,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    letter = frame[frame["arm"] == "letter_intervention"]
    text = frame[frame["arm"] == "answer_text"]
    if not letter.empty:
        rows.extend(
            _score_letter_rows(
                letter,
                model,
                tokenizer,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                device=device,
                telemetry=telemetry,
            )
        )
    if not text.empty:
        rows.extend(
            _score_text_rows(
                text,
                model,
                tokenizer,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                device=device,
                telemetry=telemetry,
            )
        )
    return pd.DataFrame(rows).sort_values("work_key", kind="mergesort").reset_index(drop=True)


def run_causal_design(
    design: pd.DataFrame,
    model,
    tokenizer,
    identity: SemanticIdentity,
    run_root: str | Path,
    *,
    batch_size: int,
    max_batch_tokens: int | None,
    checkpoint_size: int = 256,
    device=None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    validate_causal_design(design)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_batch_tokens is not None and max_batch_tokens < 1:
        raise ValueError("max_batch_tokens must be positive")
    if checkpoint_size < 1:
        raise ValueError("checkpoint_size must be positive")
    stop = should_stop or (lambda: False)
    root = Path(run_root)
    store = ShardStore(root / "shards" / "causal_behavior", identity)
    completed = store.completed_work_keys()
    source = design.sort_values("work_key", kind="mergesort").reset_index(drop=True)
    unexpected = completed - set(source["work_key"].astype(str))
    if unexpected:
        raise RuntimeError(f"Stored shards contain {len(unexpected)} work keys outside this run")
    pending = source[~source["work_key"].astype(str).isin(completed)].reset_index(drop=True)
    telemetry = ScoringTelemetry()
    started = time.monotonic()

    for start in range(0, len(pending), checkpoint_size):
        if stop():
            break
        chunk = pending.iloc[start : start + checkpoint_size].copy()
        scored = _score_chunk(
            chunk,
            model,
            tokenizer,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            device=device,
            telemetry=telemetry,
        )
        scored["semantic_run_id"] = identity.semantic_run_id
        scored["model_id"] = identity.payload["model"]["id"]
        scored["model_revision"] = identity.payload["model"]["revision"]
        keys = [str(value) for value in chunk["work_key"]]
        store.write_shard(scored, work_keys=keys)
        completed.update(keys)
        _atomic_json(
            {
                "status": "stopping" if stop() else "running",
                "completed": len(completed),
                "total": len(source),
                "last_work_key": keys[-1],
                "elapsed_seconds": time.monotonic() - started,
                "batches": telemetry.batches,
                "padding_ratio": telemetry.padding_ratio,
                "input_preparation_seconds": telemetry.input_preparation_seconds,
            },
            root / "progress.json",
        )
        if stop():
            break

    merged = store.merge(sort_by=["work_key"])
    output = root / "raw" / "causal_behavior.parquet"
    write_table_atomic(merged, output)
    complete = len(merged) == len(source)
    _atomic_json(
        {
            "status": "complete" if complete else "interrupted",
            "completed": len(merged),
            "total": len(source),
            "elapsed_seconds": time.monotonic() - started,
            "batches": telemetry.batches,
            "actual_tokens": telemetry.actual_tokens,
            "padded_tokens": telemetry.padded_tokens,
            "padding_ratio": telemetry.padding_ratio,
            "input_preparation_seconds": telemetry.input_preparation_seconds,
            "output_write_seconds": store.write_seconds,
        },
        root / "progress.json",
    )
    _atomic_json(
        {
            "artifact_schema_version": 1,
            "semantic_run_id": identity.semantic_run_id,
            "semantic_sha256": identity.semantic_sha256,
            "model_id": identity.payload["model"]["id"],
            "model_revision": identity.payload["model"]["revision"],
            "row_count": len(merged),
            "sha256": sha256_file(output),
        },
        output.with_name(output.name + ".manifest.json"),
    )
    return merged
