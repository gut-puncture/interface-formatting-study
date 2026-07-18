#!/usr/bin/env bash
set -euo pipefail

runtime_is_ready() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys, torch
assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()
assert torch.__version__.split("+")[0] == "2.7.1"
assert torch.version.cuda == "12.6"
assert (3, 11) <= sys.version_info[:2] < (3, 13)
PY
}

UV_BIN="${HOME}/.local/bin/uv"
ensure_uv() {
  if [[ ! -x "$UV_BIN" ]]; then
    mkdir -p "$(dirname "$UV_BIN")"
    curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_UNMANAGED_INSTALL="$(dirname "$UV_BIN")" sh
  fi
}

PYTHON_BIN=""
if [[ -x .venv/bin/python ]] && runtime_is_ready .venv/bin/python; then
  PYTHON_BIN=.venv/bin/python
else
  SYSTEM_PYTHON_BIN=""
  if command -v python >/dev/null 2>&1 && runtime_is_ready python; then
    SYSTEM_PYTHON_BIN="$(command -v python)"
  fi
  ensure_uv
  if [[ -n "$SYSTEM_PYTHON_BIN" ]]; then
    "$UV_BIN" venv --system-site-packages --python "$SYSTEM_PYTHON_BIN" --clear .venv
  else
    "$UV_BIN" python install 3.11
    "$UV_BIN" venv --python 3.11 --clear .venv
    "$UV_BIN" pip install --python .venv/bin/python \
      torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  fi
  PYTHON_BIN=.venv/bin/python
fi
ensure_uv
PIP=("$UV_BIN" pip install --python "$PYTHON_BIN")

# Preserve the image's CUDA-matched Torch. The small non-Torch environment is
# exact so resumed shards cannot silently mix numerical stacks.
"${PIP[@]}" -e . --no-deps
"${PIP[@]}" --requirement requirements-gpu.lock

"$PYTHON_BIN" - <<'PY'
import importlib.metadata, pandas, pyarrow, torch, transformers
import sys
assert torch.cuda.is_available(), "Prime image does not expose a CUDA GPU"
assert torch.cuda.is_bf16_supported(), "GPU does not support BF16"
assert torch.__version__.split("+")[0] == "2.7.1", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "transformers": transformers.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
})
expected = {}
for line in open("requirements-gpu.lock", encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#"):
        name, version = line.split("==", 1)
        expected[name.lower()] = version
for name, version in expected.items():
    actual = importlib.metadata.version(name)
    assert actual == version, (name, actual, version)
PY
