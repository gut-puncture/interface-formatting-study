#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
assert torch.cuda.is_available(), "Prime image does not expose a CUDA GPU"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)})
PY

# Preserve the image's CUDA-matched Torch. Install this project without dependency
# resolution, then add only lightweight packages that the image is missing.
python -m pip install -e . --no-deps
MISSING="$(python - <<'PY'
import importlib.util
modules = {
    "numpy": "numpy>=1.24",
    "pandas": "pandas>=2.0",
    "yaml": "pyyaml>=6.0",
    "transformers": "transformers>=4.44",
    "pyarrow": "pyarrow>=14.0",
    "huggingface_hub": "huggingface-hub>=0.24",
}
print(" ".join(package for module, package in modules.items() if importlib.util.find_spec(module) is None))
PY
)"
if [[ -n "$MISSING" ]]; then
  # shellcheck disable=SC2086
  python -m pip install $MISSING
fi

python - <<'PY'
import pandas, pyarrow, torch, transformers
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
})
PY
