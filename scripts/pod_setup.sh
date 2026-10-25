#!/bin/bash
# Run on the GPU pod from the repo root. Installs pinned packages, downloads the four models
# at their pinned revisions (in the background), and starts the experiment once downloads finish.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
mkdir -p runs/main
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee runs/main/gpu.txt
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV" -ge 570 ]; then IDX=https://download.pytorch.org/whl/cu128; else IDX=https://download.pytorch.org/whl/cu126; fi
pip install -q "torch==2.10.0" --index-url $IDX || pip install -q torch --index-url $IDX
pip install -q "transformers==5.20.2" accelerate hf_transfer huggingface_hub pandas pyarrow numpy
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.is_available(),'transformers',transformers.__version__)" | tee runs/main/versions.txt
pip freeze > runs/main/pip_freeze.txt
# download all models in order (first one first, so the run can start early)
python - <<'EOF' > runs/main/download.log 2>&1 &
from huggingface_hub import snapshot_download
from keyhole import config as C
for k in C.MODEL_ORDER:
    m = C.MODELS[k]
    p = snapshot_download(m["repo"], revision=m["rev"], allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
    open(f"runs/main/{k}.downloaded", "w").write(p)
    print("downloaded", k, p, flush=True)
EOF
echo "setup ok"
