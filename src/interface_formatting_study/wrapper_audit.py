from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

PURE_INTERFACE_ALLOWLIST = {
    "ascii_box",
    "csv_inline",
    "graphql_query",
    "html_form",
    "ini_file",
    "key_equals",
    "protobuf_msg",
    "regex_match",
    "s_expression",
    "shell_heredoc",
    "toml_config",
}

STYLE_TRANSFORMING_ALLOWLIST = {
    "academic_abstract",
    "changelog_entry",
    "haiku_riddle",
    "irc_log",
    "legal_clause",
    "meeting_minutes",
    "quest_briefing",
    "recipe_instruction",
    "tweet_thread",
}

PURE_NAME_RE = re.compile(
    r"json|yaml|xml|html|csv|ini|table|markdown|ascii|key|form|list|bullet|box|graphql|protobuf|regex|s_expression|shell|toml",
    re.I,
)
STYLE_NAME_RE = re.compile(
    r"haiku|riddle|quest|story|poem|roleplay|mission|briefing|persona|abstract|legal|meeting|recipe|tweet|changelog|irc",
    re.I,
)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _subsequence_coverage(haystack: str, needle: str) -> float:
    hay_tokens = _word_tokens(haystack)
    needle_tokens = _word_tokens(needle)
    if not needle_tokens:
        return 0.0
    hay = " ".join(hay_tokens)
    matched = sum(1 for token in needle_tokens if re.search(rf"\b{re.escape(token)}\b", hay))
    return matched / len(needle_tokens)


def contains_nearly_verbatim(haystack: str, needle: str | None, *, threshold: float = 0.78) -> bool:
    if not needle:
        return False
    hay = _norm_text(haystack)
    ned = _norm_text(needle)
    if not ned:
        return False
    if ned in hay:
        return True
    return _subsequence_coverage(hay, ned) >= threshold


def contains_all_choices(prompt: str, choices: Iterable[str] | None) -> bool:
    if choices is None:
        return False
    return all(contains_nearly_verbatim(prompt, str(choice), threshold=0.95) for choice in choices)


@dataclass(frozen=True)
class WrapperAuditRow:
    wrapper_name: str
    category: str
    verified_pure_interface: bool
    contains_question_verbatim: bool
    contains_all_choices_verbatim: bool
    question_preservation_rate: float
    choice_preservation_rate: float
    mean_prompt_chars: float
    notes: str


def classify_wrapper_name(wrapper_name: str) -> str | None:
    if wrapper_name in PURE_INTERFACE_ALLOWLIST:
        return "pure_interface"
    if wrapper_name in STYLE_TRANSFORMING_ALLOWLIST:
        return "style_transforming"
    if STYLE_NAME_RE.search(wrapper_name):
        return "style_transforming"
    if PURE_NAME_RE.search(wrapper_name):
        return "pure_interface"
    return None


def audit_wrappers(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[WrapperAuditRow] = []
    for wrapper_name, group in df.groupby("wrapper_name", sort=True):
        question_flags = [
            contains_nearly_verbatim(prompt, question)
            for prompt, question in zip(group["wrapped_prompt"], group["question"], strict=False)
        ]
        choice_flags = [
            contains_all_choices(prompt, choices)
            for prompt, choices in zip(group["wrapped_prompt"], group["choices"], strict=False)
        ]
        question_rate = sum(question_flags) / max(1, len(question_flags))
        choice_rate = sum(choice_flags) / max(1, len(choice_flags))
        name_category = classify_wrapper_name(wrapper_name)
        canonical_verified = question_rate >= 0.80 and choice_rate >= 0.98

        if (wrapper_name in PURE_INTERFACE_ALLOWLIST or name_category == "pure_interface") and canonical_verified:
            category = "pure_interface"
            notes = "verified pure-interface: canonical-field checks pass"
        elif wrapper_name in PURE_INTERFACE_ALLOWLIST or name_category == "pure_interface":
            category = "pure_interface_unverified"
            notes = "manual/name pure-interface candidate; excluded from primary claims because canonical-field checks fail"
        elif name_category == "style_transforming":
            category = "style_transforming"
            notes = "manual/name heuristic indicates semantic or stylistic transformation"
        else:
            category = "style_transforming"
            notes = "conservative fallback: not enough canonical preservation evidence"

        rows.append(
            WrapperAuditRow(
                wrapper_name=str(wrapper_name),
                category=category,
                verified_pure_interface=category == "pure_interface",
                contains_question_verbatim=question_rate >= 0.80,
                contains_all_choices_verbatim=choice_rate >= 0.98,
                question_preservation_rate=float(question_rate),
                choice_preservation_rate=float(choice_rate),
                mean_prompt_chars=float(group["wrapped_prompt"].str.len().mean()),
                notes=notes,
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def attach_wrapper_categories(df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    category_map = audit_df.set_index("wrapper_name")["category"].to_dict()
    verified_map = audit_df.set_index("wrapper_name")["verified_pure_interface"].to_dict()
    out = df.copy()
    out["wrapper_category"] = out["wrapper_name"].map(category_map).fillna("unknown")
    out["wrapper_is_verified_pure_interface"] = (
        out["wrapper_name"].map(verified_map).fillna(False).astype(bool)
    )
    return out
