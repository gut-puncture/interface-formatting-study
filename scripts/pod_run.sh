#!/bin/bash
# Start the experiment on the pod (after pod_setup.sh). Waits for each model's download.
cd "$(dirname "$0")/.."
. .venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=0
for k in q4 olmo q8 g12; do
  while [ ! -f runs/main/$k.downloaded ]; do sleep 20; done
  python -m keyhole.run --out runs/main --models $k --hours ${HOURS:-5.25} >> runs/main/stdout.log 2>&1
done
python -m keyhole.run --out runs/main --models --finish >> runs/main/stdout.log 2>&1
