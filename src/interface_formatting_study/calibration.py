from __future__ import annotations

import re
from typing import Mapping

from .scoring import correct_answer_margin, score_labels
from .utils import LABELS


def _placeholder_for(label: str) -> str:
    return f"OPTION_{label}_PLACEHOLDER"


SAME_WRAPPER_REDACTION = "same_wrapper_redaction"
GENERIC_FALLBACK = "generic_fallback"


def _generic_content_free_prompt(row: Mapping[str, object]) -> str:
    return (
        "[CONTENT_FREE_WRAPPER]\n"
        "question = QUESTION_TEXT_PLACEHOLDER\n"
        "A = OPTION_A_PLACEHOLDER\n"
        "B = OPTION_B_PLACEHOLDER\n"
        "C = OPTION_C_PLACEHOLDER\n"
        "D = OPTION_D_PLACEHOLDER\n\n"
        "Return only the letter (A, B, C, or D).\n"
        "Answer: "
    )


def make_content_free_prompt_with_metadata(row: Mapping[str, object]) -> dict[str, object]:
    prompt = str(row.get("wrapped_prompt", ""))
    question = row.get("question")
    choices = row.get("choices")

    if question:
        prompt = re.sub(re.escape(str(question)), "QUESTION_TEXT_PLACEHOLDER", prompt, flags=re.IGNORECASE)

    if isinstance(choices, (list, tuple)):
        for label, choice in zip(LABELS, choices, strict=False):
            choice_text = str(choice)
            if choice_text:
                prompt = prompt.replace(choice_text, _placeholder_for(label))

    has_question_placeholder = "QUESTION_TEXT_PLACEHOLDER" in prompt
    has_option_placeholders = all(_placeholder_for(label) in prompt for label in LABELS)
    canonical_question_available = bool(question)
    canonical_choices_available = isinstance(choices, (list, tuple)) and len(choices) == len(LABELS)
    if not canonical_question_available and not canonical_choices_available:
        return {
            "content_free_prompt": _generic_content_free_prompt(row),
            "content_free_calibration_kind": GENERIC_FALLBACK,
            "content_free_fallback_reason": "canonical_question_and_choices_missing",
        }

    if (
        (not canonical_question_available or has_question_placeholder)
        and (not canonical_choices_available or has_option_placeholders)
    ):
        if canonical_choices_available and content_free_contains_answer_text(prompt, list(choices)):
            return {
                "content_free_prompt": _generic_content_free_prompt(row),
                "content_free_calibration_kind": GENERIC_FALLBACK,
                "content_free_fallback_reason": "answer_text_leak_after_redaction",
            }
        return {
            "content_free_prompt": prompt,
            "content_free_calibration_kind": SAME_WRAPPER_REDACTION,
            "content_free_fallback_reason": "",
        }

    return {
        "content_free_prompt": _generic_content_free_prompt(row),
        "content_free_calibration_kind": GENERIC_FALLBACK,
        "content_free_fallback_reason": "incomplete_canonical_redaction",
    }


def make_content_free_prompt(row: Mapping[str, object]) -> str:
    return str(make_content_free_prompt_with_metadata(row)["content_free_prompt"])


def content_free_contains_answer_text(prompt: str, choices: list[str] | None) -> bool:
    if not choices:
        return False
    normalized = re.sub(r"\s+", " ", prompt).lower()
    safe = normalized
    safe = re.sub(r"option_[abcd]_placeholder", " ", safe)
    safe = safe.replace("question_text_placeholder", " ")
    safe = safe.replace("question_placeholder", " ")
    safe = safe.replace("content_free_wrapper", " ")
    safe = re.sub(r"return only the letter\s*\(\s*a\s*,\s*b\s*,\s*c\s*,\s*or\s*d\s*\)\s*\.?", " ", safe)
    safe = re.sub(r"\banswer\s*:\s*", " ", safe)
    safe = re.sub(r"(?<![a-z0-9])[abcd]\s*[\)=:\-]\s*(?![a-z0-9])", " ", safe)
    safe = re.sub(r"\s+", " ", safe)
    safe_compact = "".join(re.findall(r"[a-z0-9]+", safe))
    for choice in choices:
        text = str(choice).strip().lower()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        pattern = r"(?<![a-z0-9])" + re.escape(text) + r"(?![a-z0-9])"
        if re.search(pattern, safe):
            return True
        choice_compact = "".join(re.findall(r"[a-z0-9]+", text))
        if len(choice_compact) >= 4 and choice_compact in safe_compact:
            return True
    return False


def calibrate_scores(raw_scores: Mapping[str, float], bias_scores: Mapping[str, float]) -> dict[str, float]:
    return {label: float(raw_scores[label]) - float(bias_scores[label]) for label in LABELS}


def score_with_calibration(
    model,
    tokenizer,
    row: Mapping[str, object],
    *,
    batch_size: int = 40,
    device=None,
) -> dict[str, object]:
    raw_scores = score_labels(model, tokenizer, str(row["wrapped_prompt"]), batch_size=batch_size, device=device)
    content_free = make_content_free_prompt_with_metadata(row)
    content_free_prompt = str(content_free["content_free_prompt"])
    bias_scores = score_labels(model, tokenizer, content_free_prompt, batch_size=batch_size, device=device)
    cal_scores = calibrate_scores(raw_scores, bias_scores)
    raw_margin = correct_answer_margin(raw_scores, str(row["correct_label"]))
    cal_margin = correct_answer_margin(cal_scores, str(row["correct_label"]))

    out: dict[str, object] = {
        "content_free_prompt": content_free_prompt,
        "content_free_calibration_kind": content_free["content_free_calibration_kind"],
        "content_free_fallback_reason": content_free["content_free_fallback_reason"],
    }
    for label in LABELS:
        out[f"raw_score_{label}"] = raw_scores[label]
        out[f"bias_score_{label}"] = bias_scores[label]
        out[f"cal_score_{label}"] = cal_scores[label]
    out.update(
        {
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
    return out
