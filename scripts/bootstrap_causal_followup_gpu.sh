#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "Prime image does not expose a CUDA GPU"
assert torch.cuda.is_bf16_supported(), "GPU does not support BF16"
assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) == (2, 7), torch.__version__
assert (3, 11) <= sys.version_info[:2] < (3, 13), sys.version
print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})
PY

# Preserve the image's CUDA-matched Torch. The small non-Torch environment is
# exact so resumed shards cannot silently mix numerical stacks.
python -m pip install -e . --no-deps
python -m pip install --requirement requirements-gpu.lock

python - <<'PY'
import importlib.metadata, pandas, pyarrow, torch, transformers
print({
    "torch": torch.__version__,
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
