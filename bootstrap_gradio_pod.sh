#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_PREFER_BINARY=1
export HF_XET_HIGH_PERFORMANCE=1
export BERNINI_DIR="${BERNINI_DIR:-/opt/Bernini}"
export BERNINI_MODEL_DIR="${BERNINI_MODEL_DIR:-/models/Bernini-R-Diffusers}"
export BERNINI_GRADIO_PORT="${BERNINI_GRADIO_PORT:-7860}"

apt-get update
apt-get install -y --no-install-recommends \
  build-essential git git-lfs curl wget aria2 ffmpeg libgl1 libglib2.0-0 \
  ninja-build ca-certificates
git lfs install

python -m pip install --upgrade pip setuptools wheel packaging
python - <<'PY' || pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
import torch
assert torch.__version__.startswith("2.8.0")
assert torch.version.cuda and torch.version.cuda.startswith("12.8")
PY

PY_TAG="$(python - <<'PY'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}")
PY
)"
pip install \
  "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"

if [ ! -d "$BERNINI_DIR/.git" ]; then
  rm -rf "$BERNINI_DIR"
  git clone --depth=1 https://github.com/bytedance/Bernini.git "$BERNINI_DIR"
fi

grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==)' "$BERNINI_DIR/requirements.txt" > /tmp/bernini-requirements.txt
pip install -r /tmp/bernini-requirements.txt
pip install --upgrade "huggingface_hub[hf_xet]"
pip install -e "$BERNINI_DIR"

if [ ! -f "$BERNINI_MODEL_DIR/model_index.json" ]; then
  mkdir -p "$BERNINI_MODEL_DIR"
  hf download ByteDance/Bernini-R-Diffusers --local-dir "$BERNINI_MODEL_DIR"
fi

cd "$BERNINI_DIR"
exec python gradio_demo.py \
  --config "$BERNINI_MODEL_DIR" \
  --port "$BERNINI_GRADIO_PORT" \
  --save_dir /workspace/bernini-gradio-outputs
