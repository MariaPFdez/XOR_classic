#!/usr/bin/env bash
# Option A — Python venv on the DGX Spark (aarch64, GB10).
# If you already have a working CUDA PyTorch env on the Sparks, skip this and
# just `pip install lpips matplotlib` into it, then run check_env.py.
set -euo pipefail
cd "$(dirname "$0")"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url "$TORCH_INDEX"
pip install lpips matplotlib numpy
python check_env.py || {
  echo
  echo "If CUDA is not available with these wheels, use the NGC container instead:"
  echo "  docker build -t xorlab -f Dockerfile .    (see README)"
  exit 1; }
