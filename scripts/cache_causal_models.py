#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from interface_formatting_study.model_profiles import MODEL_PROFILES, get_model_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache pinned causal-follow-up model snapshots")
    parser.add_argument("--profile", action="append", choices=sorted(MODEL_PROFILES))
    args = parser.parse_args()
    names = args.profile or ["mistral", "phi", "qwen"]
    for name in names:
        profile = get_model_profile(name)
        path = snapshot_download(repo_id=profile.model_id, revision=profile.revision)
        print(f"cached {profile.model_id}@{profile.revision} at {path}")


if __name__ == "__main__":
    main()
