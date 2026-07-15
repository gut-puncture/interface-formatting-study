from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "cache_causal_models.py"
SPEC = importlib.util.spec_from_file_location("cache_causal_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_model_cache_excludes_duplicate_consolidated_weights(monkeypatch):
    observed = {}
    monkeypatch.setattr(MODULE, "snapshot_download", lambda **kwargs: observed.update(kwargs) or "/cache")

    MODULE.cache_profile("mistral")

    assert "model*.safetensors" in observed["allow_patterns"]
    assert "consolidated.safetensors" in observed["ignore_patterns"]
    assert observed["revision"]
