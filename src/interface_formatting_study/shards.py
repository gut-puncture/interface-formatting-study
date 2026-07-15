from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

import pandas as pd

from .run_identity import SemanticIdentity, sha256_file


class ShardError(RuntimeError):
    pass


class ShardIdentityError(ShardError):
    pass


class ShardConflictError(ShardError):
    pass


@dataclass(frozen=True)
class ShardRecord:
    path: Path
    work_keys: tuple[str, ...]
    data_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ShardStore:
    """Atomic, identity-checked phase shards with deterministic work keys."""

    def __init__(self, root: str | Path, identity: SemanticIdentity):
        self.root = Path(root)
        self.identity = identity
        self.write_seconds = 0.0
        self.root.mkdir(parents=True, exist_ok=True)
        identity_path = self.root / "semantic_identity.json"
        if identity_path.exists():
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
            if existing != identity.as_dict():
                raise ShardIdentityError(f"Shard root identity does not match {identity.semantic_run_id}: {self.root}")
        else:
            _atomic_json(identity.as_dict(), identity_path)

    def write_shard(
        self,
        frame: pd.DataFrame,
        *,
        work_keys: Sequence[str],
        shard_hint: str | None = None,
    ) -> Path:
        write_started = time.monotonic()
        normalized_keys = tuple(sorted({str(key) for key in work_keys}))
        if not normalized_keys:
            raise ValueError("work_keys must not be empty")
        if len(normalized_keys) != len(work_keys):
            raise ValueError("work_keys must be unique within a shard")

        temporary = self.root / f".incoming-{uuid.uuid4().hex}"
        temporary.mkdir(parents=False)
        try:
            data_path = temporary / "data.parquet"
            frame.to_parquet(data_path, index=False)
            data_sha256 = sha256_file(data_path)
            key_sha256 = hashlib.sha256(_canonical_json(normalized_keys).encode("utf-8")).hexdigest()
            manifest = {
                "shard_schema_version": 1,
                "semantic_run_id": self.identity.semantic_run_id,
                "semantic_sha256": self.identity.semantic_sha256,
                "model_id": self.identity.payload["model"]["id"],
                "model_revision": self.identity.payload["model"]["revision"],
                "work_keys": list(normalized_keys),
                "data_sha256": data_sha256,
                "row_count": int(len(frame)),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            hint = "" if shard_hint is None else f"-{str(shard_hint).replace('/', '_')}"
            destination = self.root / f"shard-{key_sha256[:12]}-{data_sha256[:12]}{hint}"
            if destination.exists():
                existing_manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
                if existing_manifest != manifest or sha256_file(destination / "data.parquet") != data_sha256:
                    raise ShardConflictError(f"Existing deterministic shard conflicts with new data: {destination}")
                return destination
            os.replace(temporary, destination)
            return destination
        finally:
            self.write_seconds += time.monotonic() - write_started
            if temporary.exists():
                shutil.rmtree(temporary)

    def _records(self) -> list[ShardRecord]:
        records: list[ShardRecord] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir() or path.name.startswith(".incoming-"):
                continue
            manifest_path = path / "manifest.json"
            data_path = path / "data.parquet"
            if not manifest_path.exists() or not data_path.exists():
                raise ShardError(f"Incomplete shard directory: {path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("semantic_run_id") != self.identity.semantic_run_id
                or manifest.get("semantic_sha256") != self.identity.semantic_sha256
                or manifest.get("model_id") != self.identity.payload["model"]["id"]
                or manifest.get("model_revision") != self.identity.payload["model"]["revision"]
            ):
                raise ShardIdentityError(f"Shard identity mismatch: {path}")
            actual_sha256 = sha256_file(data_path)
            if actual_sha256 != manifest.get("data_sha256"):
                raise ShardError(f"Shard checksum mismatch: {path}")
            work_keys = tuple(str(key) for key in manifest.get("work_keys", []))
            if not work_keys or len(set(work_keys)) != len(work_keys):
                raise ShardError(f"Shard has missing or duplicate work keys: {path}")
            records.append(ShardRecord(path=path, work_keys=work_keys, data_sha256=actual_sha256))
        return records

    def _unique_records(self) -> list[ShardRecord]:
        unique: list[ShardRecord] = []
        exact_seen: set[tuple[str, tuple[str, ...]]] = set()
        work_key_owner: dict[str, ShardRecord] = {}
        for record in self._records():
            exact_key = (record.data_sha256, record.work_keys)
            if exact_key in exact_seen:
                continue
            for work_key in record.work_keys:
                owner = work_key_owner.get(work_key)
                if owner is not None:
                    raise ShardConflictError(
                        f"Conflicting duplicate work key {work_key!r}: {owner.path.name} vs {record.path.name}"
                    )
            exact_seen.add(exact_key)
            unique.append(record)
            for work_key in record.work_keys:
                work_key_owner[work_key] = record
        return unique

    def completed_work_keys(self) -> set[str]:
        return {key for record in self._unique_records() for key in record.work_keys}

    def merge(self, *, sort_by: Sequence[str] | None = None) -> pd.DataFrame:
        frames = [pd.read_parquet(record.path / "data.parquet") for record in self._unique_records()]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        if sort_by:
            missing = set(sort_by) - set(merged.columns)
            if missing and not merged.empty:
                raise ShardError(f"Cannot sort merged shards; missing columns: {sorted(missing)}")
            if not merged.empty:
                merged = merged.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
        return merged


T = TypeVar("T")


def run_sharded_phase(
    store: ShardStore,
    work_items: Iterable[tuple[str, T]],
    process_one: Callable[[str, T], pd.DataFrame],
    *,
    max_work_units: int = 16,
    max_seconds: float = 300.0,
    should_stop: Callable[[], bool] | None = None,
    on_flush: Callable[[set[str], Path], None] | None = None,
) -> pd.DataFrame:
    if max_work_units < 1:
        raise ValueError("max_work_units must be positive")
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    stop = should_stop or (lambda: False)
    completed = store.completed_work_keys()
    frames: list[pd.DataFrame] = []
    keys: list[str] = []
    opened_at = time.monotonic()

    def flush() -> None:
        nonlocal frames, keys, opened_at
        if not keys:
            return
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        shard_path = store.write_shard(combined, work_keys=keys)
        completed.update(keys)
        if on_flush is not None:
            on_flush(set(completed), shard_path)
        frames = []
        keys = []
        opened_at = time.monotonic()

    for work_key, payload in work_items:
        normalized_key = str(work_key)
        if normalized_key in completed:
            continue
        if stop():
            flush()
            break
        frame = process_one(normalized_key, payload)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("process_one must return a pandas DataFrame")
        if not frame.empty:
            frame = frame.copy()
            frame["_work_key"] = normalized_key
            frames.append(frame)
        keys.append(normalized_key)
        if len(keys) >= max_work_units or time.monotonic() - opened_at >= max_seconds or stop():
            flush()
        if stop():
            break
    flush()
    return store.merge()
