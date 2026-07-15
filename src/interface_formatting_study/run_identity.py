from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .model_profiles import ModelProfile


EXECUTION_ONLY_CONFIG_KEYS = {
    "checkpoint_size",
    "prefetch_batches",
}
NON_SEMANTIC_TOP_LEVEL_KEYS = {"outputs", "bootstrap", "model_name", "fallback_model_name"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _scientific_config(config: Mapping[str, object]) -> dict[str, object]:
    def clean(value: object, *, parent: str | None = None) -> object:
        if isinstance(value, Mapping):
            out: dict[str, object] = {}
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                key_str = str(key)
                if parent is None and key_str in NON_SEMANTIC_TOP_LEVEL_KEYS:
                    continue
                if key_str in EXECUTION_ONLY_CONFIG_KEYS:
                    continue
                out[key_str] = clean(child, parent=key_str)
            return out
        if isinstance(value, (list, tuple)):
            return [clean(child, parent=parent) for child in value]
        if isinstance(value, Path):
            return str(value)
        return value

    cleaned = clean(config)
    assert isinstance(cleaned, dict)
    return cleaned


@dataclass(frozen=True)
class SemanticIdentity:
    semantic_run_id: str
    semantic_sha256: str
    payload: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "semantic_run_id": self.semantic_run_id,
            "semantic_sha256": self.semantic_sha256,
            **self.payload,
        }


def default_semantic_source_paths(project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    return sorted((root / "src" / "interface_formatting_study").glob("*.py")) + [root / "pyproject.toml"]


def build_semantic_identity(
    profile: ModelProfile,
    *,
    config: Mapping[str, object],
    dataset_path: str | Path,
    source_paths: Iterable[str | Path],
) -> SemanticIdentity:
    dataset = Path(dataset_path)

    def stable_source_key(path: Path) -> str:
        parts = path.parts
        if "src" in parts:
            return str(Path(*parts[parts.index("src") :]))
        return path.name

    source_hashes = {
        stable_source_key(Path(path)): sha256_file(path)
        for path in sorted((Path(path) for path in source_paths), key=lambda item: str(item))
        if Path(path).exists()
    }
    payload: dict[str, object] = {
        "identity_schema_version": 1,
        "model": {
            "id": profile.model_id,
            "revision": profile.revision,
            "slug": profile.slug,
            "expected_layers": profile.expected_layers,
            "trust_remote_code": profile.trust_remote_code,
        },
        "tokenizer": {
            "id": profile.tokenizer_id,
            "revision": profile.tokenizer_revision,
        },
        "dataset": {
            "path_name": dataset.name,
            "sha256": sha256_file(dataset),
        },
        "experiment_config": _scientific_config(config),
        "semantic_source_hashes": source_hashes,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return SemanticIdentity(semantic_run_id=digest[:20], semantic_sha256=digest, payload=payload)
