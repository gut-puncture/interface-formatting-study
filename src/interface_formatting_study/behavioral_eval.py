from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .calibration import calibrate_scores, make_content_free_prompt_with_metadata, score_with_calibration
from .scoring import correct_answer_margin, score_labels, score_labels_many, tokenize_text
from .utils import LABELS, read_table, write_table


def _score_labels_many_with_backoff(
    model,
    tokenizer,
    prompts: list[str],
    *,
    sequence_batch_size: int,
    max_batch_tokens: int | None,
    device=None,
) -> list[dict[str, float]]:
    batch_size = max(1, int(sequence_batch_size))
    token_cap = None if max_batch_tokens is None else max(1, int(max_batch_tokens))
    while True:
        try:
            return score_labels_many(
                model,
                tokenizer,
                prompts,
                batch_size=batch_size,
                max_batch_tokens=token_cap,
                device=device,
                use_cache=True,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message or (batch_size == 1 and token_cap in (None, 1)):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            if token_cap is not None:
                token_cap = max(1, token_cap // 2)
            print(f"CUDA OOM while scoring; retrying with sequence_batch_size={batch_size}, max_batch_tokens={token_cap}")


def _score_records_batched(
    records: list[dict[str, object]],
    model,
    tokenizer,
    *,
    sequence_batch_size: int,
    max_batch_tokens: int | None,
    bias_cache: dict[str, dict[str, float]],
    device=None,
) -> pd.DataFrame:
    raw_scores_list = _score_labels_many_with_backoff(
        model,
        tokenizer,
        [str(row["wrapped_prompt"]) for row in records],
        sequence_batch_size=sequence_batch_size,
        max_batch_tokens=max_batch_tokens,
        device=device,
    )
    content_free_meta = [make_content_free_prompt_with_metadata(row) for row in records]
    missing_bias_prompts = sorted(
        {
            str(meta["content_free_prompt"])
            for meta in content_free_meta
            if str(meta["content_free_prompt"]) not in bias_cache
        }
    )
    if missing_bias_prompts:
        bias_scores_list = _score_labels_many_with_backoff(
            model,
            tokenizer,
            missing_bias_prompts,
            sequence_batch_size=sequence_batch_size,
            max_batch_tokens=max_batch_tokens,
            device=device,
        )
        bias_cache.update(zip(missing_bias_prompts, bias_scores_list, strict=True))

    scored_rows: list[dict[str, object]] = []
    for row, raw_scores, meta in zip(records, raw_scores_list, content_free_meta, strict=True):
        result = dict(row)
        result["prompt_token_count"] = len(tokenize_text(tokenizer, str(row["wrapped_prompt"])))
        content_free_prompt = str(meta["content_free_prompt"])
        bias_scores = bias_cache[content_free_prompt]
        cal_scores = calibrate_scores(raw_scores, bias_scores)
        raw_margin = correct_answer_margin(raw_scores, str(row["correct_label"]))
        cal_margin = correct_answer_margin(cal_scores, str(row["correct_label"]))
        result.update(
            {
                "content_free_prompt": content_free_prompt,
                "content_free_calibration_kind": meta["content_free_calibration_kind"],
                "content_free_fallback_reason": meta["content_free_fallback_reason"],
                "raw_pred_label": raw_margin.pred_label,
                "cal_pred_label": cal_margin.pred_label,
                "raw_correct": raw_margin.correct,
                "cal_correct": cal_margin.correct,
                "raw_margin": raw_margin.margin,
                "cal_margin": cal_margin.margin,
                "raw_tie": raw_margin.is_tie,
                "cal_tie": cal_margin.is_tie,
            }
        )
        for label in LABELS:
            result[f"raw_score_{label}"] = raw_scores[label]
            result[f"bias_score_{label}"] = bias_scores[label]
            result[f"cal_score_{label}"] = cal_scores[label]
        scored_rows.append(result)
    return pd.DataFrame(scored_rows)


def _score_records_reference(
    records: list[dict[str, object]],
    model,
    tokenizer,
    *,
    batch_size: int,
    bias_cache: dict[str, dict[str, float]],
    device=None,
) -> pd.DataFrame:
    scored_rows: list[dict[str, object]] = []
    for row in records:
        result = dict(row)
        result["prompt_token_count"] = len(tokenize_text(tokenizer, str(row["wrapped_prompt"])))
        raw_scores = score_labels(
            model,
            tokenizer,
            str(row["wrapped_prompt"]),
            batch_size=batch_size,
            device=device,
        )
        meta = make_content_free_prompt_with_metadata(row)
        content_free_prompt = str(meta["content_free_prompt"])
        if content_free_prompt not in bias_cache:
            bias_cache[content_free_prompt] = score_labels(
                model,
                tokenizer,
                content_free_prompt,
                batch_size=batch_size,
                device=device,
            )
        bias_scores = bias_cache[content_free_prompt]
        cal_scores = calibrate_scores(raw_scores, bias_scores)
        raw_margin = correct_answer_margin(raw_scores, str(row["correct_label"]))
        cal_margin = correct_answer_margin(cal_scores, str(row["correct_label"]))
        result.update(
            {
                "content_free_prompt": content_free_prompt,
                "content_free_calibration_kind": meta["content_free_calibration_kind"],
                "content_free_fallback_reason": meta["content_free_fallback_reason"],
                "raw_pred_label": raw_margin.pred_label,
                "cal_pred_label": cal_margin.pred_label,
                "raw_correct": raw_margin.correct,
                "cal_correct": cal_margin.correct,
                "raw_margin": raw_margin.margin,
                "cal_margin": cal_margin.margin,
                "raw_tie": raw_margin.is_tie,
                "cal_tie": cal_margin.is_tie,
            }
        )
        for label in LABELS:
            result[f"raw_score_{label}"] = raw_scores[label]
            result[f"bias_score_{label}"] = bias_scores[label]
            result[f"cal_score_{label}"] = cal_scores[label]
        scored_rows.append(result)
    return pd.DataFrame(scored_rows)


def evaluate_behavioral(
    df: pd.DataFrame,
    model,
    tokenizer,
    *,
    batch_size: int = 40,
    sequence_batch_size: int | None = None,
    checkpoint_size: int = 256,
    max_batch_tokens: int | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    limit: int | None = None,
    device=None,
) -> pd.DataFrame:
    source = (df.head(limit).copy() if limit is not None else df.copy()).reset_index(drop=True)
    if sequence_batch_size is None:
        sequence_batch_size = batch_size
    if checkpoint_dir is None:
        rows: list[dict[str, object]] = []
        # Keep the old narrow path for tiny smoke/test calls unless batching is explicitly useful.
        if len(source) <= 1:
            for _, row in tqdm(source.iterrows(), total=len(source), desc="behavioral"):
                result = row.to_dict()
                result["prompt_token_count"] = len(tokenize_text(tokenizer, str(row["wrapped_prompt"])))
                result.update(score_with_calibration(model, tokenizer, row.to_dict(), batch_size=batch_size, device=device))
                rows.append(result)
            return pd.DataFrame(rows)
    shard_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    if shard_dir is not None:
        shard_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    bias_cache: dict[str, dict[str, float]] = {}
    chunk_size = max(1, int(checkpoint_size))
    for start in tqdm(range(0, len(source), chunk_size), desc="behavioral shards"):
        end = min(start + chunk_size, len(source))
        shard_path = None if shard_dir is None else shard_dir / f"part_{start:06d}_{end:06d}.parquet"
        if resume and shard_path is not None and shard_path.exists():
            frames.append(read_table(shard_path))
            continue
        frame = _score_records_reference(
            source.iloc[start:end].to_dict("records"),
            model,
            tokenizer,
            batch_size=int(batch_size),
            bias_cache=bias_cache,
            device=device,
        )
        if shard_path is not None:
            tmp_path = shard_path.with_suffix(".tmp.parquet")
            write_table(frame, tmp_path)
            tmp_path.replace(shard_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
