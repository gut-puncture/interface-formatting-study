from __future__ import annotations

import json
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import Callable

import pandas as pd

from .run_identity import SemanticIdentity, sha256_file
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
        "prompt",
        "prompt_sha256",
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
    if set(design["arm"]) - {"letter_permutation", "answer_text"}:
        raise ValueError("Causal design contains an unknown experiment arm")
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
        if row.arm == "answer_text" and int(row.variant) != 0:
            raise ValueError("Answer-text rows must use the identity assignment")

    for (item_id, wrapper_name), group in design.groupby(["item_id", "wrapper_name"], sort=True):
        letter = group[group["arm"] == "letter_permutation"]
        text = group[group["arm"] == "answer_text"]
        if set(letter["variant"].astype(int)) != {0, 1, 2, 3} or len(letter) != 4 or len(text) != 1:
            raise ValueError(
                f"Causal design requires four letter variants and one text row for {item_id}/{wrapper_name}"
            )
        if sorted(letter["position_shift"].astype(int)) != [0, 1, 2, 3]:
            raise ValueError(f"Causal design position shifts are not balanced for {item_id}/{wrapper_name}")
        if sorted(letter["label_shift"].astype(int)) != [0, 1, 2, 3]:
            raise ValueError(f"Causal design label shifts are not balanced for {item_id}/{wrapper_name}")
        for content_id in range(4):
            positions = [list(values).index(content_id) for values in letter["content_ids_by_position"]]
            labels = [
                list(labels_by_position)[position]
                for labels_by_position, position in zip(letter["labels_by_position"], positions, strict=True)
            ]
            if sorted(positions) != [0, 1, 2, 3] or sorted(labels) != ["A", "B", "C", "D"]:
                raise ValueError(f"Causal design is not balanced for {item_id}/{wrapper_name}")


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
    scores_by_row = score_labels_many(
        model,
        tokenizer,
        [str(row["prompt"]) for row in records],
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        device=device,
        telemetry=telemetry,
    )
    output = []
    for row, scores in zip(records, scores_by_row, strict=True):
        predicted_label, is_tie = predict_from_scores(scores)
        labels = [str(value) for value in row["labels_by_position"]]
        contents = [int(value) for value in row["content_ids_by_position"]]
        predicted_position = labels.index(predicted_label)
        predicted_content = contents[predicted_position]
        margin = correct_answer_margin(scores, str(row["correct_label"]))
        output.append(
            {
                **row,
                **{f"score_{label}": float(scores[label]) for label in ("A", "B", "C", "D")},
                "predicted_label": predicted_label,
                "predicted_position": predicted_position,
                "predicted_content_id": predicted_content,
                "correct": bool(predicted_content == int(row["correct_content_id"]) and not is_tie),
                "is_tie": bool(is_tie),
                "correct_margin": float(margin.margin),
                "text_ambiguous": False,
            }
        )
    return output


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
    output = []
    for row, candidates, candidate_scores in zip(records, candidate_sets, scored, strict=True):
        totals = [float(score.total_logp) for score in candidate_scores]
        ranked = sorted(range(4), key=lambda index: (-totals[index], index))
        predicted_content = ranked[0]
        is_tie = abs(totals[ranked[0]] - totals[ranked[1]]) <= EPSILON
        correct_content = int(row["correct_content_id"])
        best_wrong = max(score for index, score in enumerate(totals) if index != correct_content)
        ambiguous = len(set(candidates)) != len(candidates)
        output.append(
            {
                **row,
                "candidate_total_logps": totals,
                "candidate_mean_logps": [float(score.mean_logp) for score in candidate_scores],
                "candidate_token_counts": [int(score.token_count) for score in candidate_scores],
                "predicted_label": None,
                "predicted_position": None,
                "predicted_content_id": predicted_content,
                "correct": None if ambiguous else bool(predicted_content == correct_content and not is_tie),
                "is_tie": bool(is_tie),
                "correct_margin": float(totals[correct_content] - best_wrong),
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
    letter = frame[frame["arm"] == "letter_permutation"]
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
    if checkpoint_size < 1:
        raise ValueError("checkpoint_size must be positive")
    stop = should_stop or (lambda: False)
    root = Path(run_root)
    store = ShardStore(root / "shards" / "causal_behavior", identity)
    completed = store.completed_work_keys()
    source = design.sort_values("work_key", kind="mergesort").reset_index(drop=True)
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
