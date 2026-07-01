#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${TRADUZAI_PROJECT_ROOT:-/workspace/TraduzAI}"
REPO_URL="${TRADUZAI_REPO_URL:-https://github.com/OTheVyzx/TraduzAi.git}"
BRANCH="${TRADUZAI_REPO_BRANCH:-main}"
VENV_DIR="${TRADUZAI_VENV_DIR:-$PROJECT_ROOT/.venv}"
MODELS_DIR="${TRADUZAI_MODELS_DIR:-$PROJECT_ROOT/pipeline/models}"

echo "[vast] project_root=$PROJECT_ROOT"
echo "[vast] models_dir=$MODELS_DIR"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg
fi

if [ ! -d "$PROJECT_ROOT/.git" ]; then
  mkdir -p "$(dirname "$PROJECT_ROOT")"
  git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_ROOT"
else
  git -C "$PROJECT_ROOT" fetch --depth 1 origin "$BRANCH"
  git -C "$PROJECT_ROOT" checkout "$BRANCH"
  git -C "$PROJECT_ROOT" pull --ff-only origin "$BRANCH"
fi

cd "$PROJECT_ROOT"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# PaddleOCR is the primary OCR engine. The current Paddle CUDA 12.6 wheels pin
# nvidia-nccl-cu12==2.25.1, while matching PyTorch CUDA 12.6 wheels pin 2.26.2.
# Install Paddle's stack first, then install Torch without dependency resolution
# so both frameworks can be imported in the same single-GPU worker process.
PADDLE_INDEX_URL="${TRADUZAI_PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"
TORCH_INDEX_URL="${TRADUZAI_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
python -m pip install --extra-index-url "$PADDLE_INDEX_URL" paddlepaddle-gpu==3.2.2
python -m pip install --index-url "$TORCH_INDEX_URL" --no-deps \
  torch==2.7.1+cu126 \
  torchvision==0.22.1+cu126 \
  torchaudio==2.7.1+cu126
python -m pip install -r "$PROJECT_ROOT/scripts/vast/requirements-vast.txt"
python "$PROJECT_ROOT/scripts/vast/verify-gpu-stack.py"

mkdir -p "$MODELS_DIR" "$PROJECT_ROOT/data/worker"
export TRADUZAI_MODELS_DIR="$MODELS_DIR"
export PYTHONPATH="$PROJECT_ROOT/pipeline:${PYTHONPATH:-}"

cd "$PROJECT_ROOT/pipeline"
python download_models.py

echo "[vast] bootstrap pronto"
