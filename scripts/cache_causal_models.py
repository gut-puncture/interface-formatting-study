#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from interface_formatting_study.model_profiles import MODEL_PROFILES, get_model_profile


ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.py",
    "*.txt",
    "*.tiktoken",
    "*.jinja",
    "model*.safetensors",
    "tokenizer*",
]
IGNORE_PATTERNS = [
    "consolidated.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "original/*",
]


def cache_profile(name: str):
    profile = get_model_profile(name)
    return snapshot_download(
        repo_id=profile.model_id,
        revision=profile.revision,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache pinned causal-follow-up model snapshots")
    parser.add_argument("--profile", action="append", choices=sorted(MODEL_PROFILES))
    args = parser.parse_args()
    names = args.profile or ["mistral", "phi", "qwen"]
    for name in names:
        profile = get_model_profile(name)
        path = cache_profile(name)
        print(f"cached {profile.model_id}@{profile.revision} at {path}")


if __name__ == "__main__":
    main()
