#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_PREFER_BINARY=1
export HF_XET_HIGH_PERFORMANCE=1
export BERNINI_DIR="${BERNINI_DIR:-/opt/Bernini}"
export BERNINI_RUNTIME_DIR="${BERNINI_RUNTIME_DIR:-/opt/bernini-runtime}"
export BERNINI_RUNTIME_REPO="${BERNINI_RUNTIME_REPO:-https://github.com/AvivK5498/Bernini-Runtime.git}"
export BERNINI_RUNTIME_BRANCH="${BERNINI_RUNTIME_BRANCH:-main}"
export BERNINI_MODEL_DIR="${BERNINI_MODEL_DIR:-/models/Bernini-R-Diffusers}"
export PATH="/opt/venv/bin:$PATH"

apt-get update
apt-get install -y --no-install-recommends \
  python3.12 python3.12-dev python3.12-venv python3-pip \
  build-essential git git-lfs curl wget aria2 ffmpeg libgl1 libglib2.0-0 \
  ninja-build ca-certificates
git lfs install

if [ ! -d /opt/venv ]; then
  python3.12 -m venv /opt/venv
fi

python -m pip install --upgrade pip setuptools wheel packaging
pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install \
  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-linux_x86_64.whl

if [ ! -d "$BERNINI_DIR/.git" ]; then
  rm -rf "$BERNINI_DIR"
  git clone --depth=1 https://github.com/bytedance/Bernini.git "$BERNINI_DIR"
fi
grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==)' "$BERNINI_DIR/requirements.txt" > /tmp/bernini-requirements.txt
pip install -r /tmp/bernini-requirements.txt
pip install --upgrade "huggingface_hub[hf_xet]" runpod requests
pip install -e "$BERNINI_DIR"

if [ -d "$BERNINI_RUNTIME_DIR/.git" ]; then
  git -C "$BERNINI_RUNTIME_DIR" fetch --depth=1 origin "$BERNINI_RUNTIME_BRANCH"
  git -C "$BERNINI_RUNTIME_DIR" reset --hard "origin/$BERNINI_RUNTIME_BRANCH"
else
  rm -rf "$BERNINI_RUNTIME_DIR"
  git clone --depth=1 --branch "$BERNINI_RUNTIME_BRANCH" "$BERNINI_RUNTIME_REPO" "$BERNINI_RUNTIME_DIR"
fi
pip install -e "$BERNINI_RUNTIME_DIR"

python "$BERNINI_RUNTIME_DIR/scripts/prefetch_model.py"
exec python -m bernini_runtime.handler

