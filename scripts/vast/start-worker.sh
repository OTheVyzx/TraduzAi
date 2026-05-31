#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE="${TRADUZAI_ENV_FILE:-/workspace/traduzai-worker.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PROJECT_ROOT="${TRADUZAI_PROJECT_ROOT:-/workspace/TraduzAI}"
VENV_DIR="${TRADUZAI_VENV_DIR:-$PROJECT_ROOT/.venv}"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

: "${TRADUZAI_API_URL:?defina TRADUZAI_API_URL}"
: "${TRADUZAI_WORKER_TOKEN:?defina TRADUZAI_WORKER_TOKEN}"

export TRADUZAI_PROJECT_ROOT="$PROJECT_ROOT"
export TRADUZAI_PIPELINE_MAIN="${TRADUZAI_PIPELINE_MAIN:-$PROJECT_ROOT/pipeline/main.py}"
export TRADUZAI_PIPELINE_PYTHON="${TRADUZAI_PIPELINE_PYTHON:-$VENV_DIR/bin/python}"
export TRADUZAI_WORKER_NAME="${TRADUZAI_WORKER_NAME:-vast-worker-$(hostname)}"
export TRADUZAI_WORKER_WORK_DIR="${TRADUZAI_WORKER_WORK_DIR:-/workspace/traduzai-worker}"
export TRADUZAI_MODELS_DIR="${TRADUZAI_MODELS_DIR:-$PROJECT_ROOT/pipeline/models}"
export TRADUZAI_FAST_PAGE_SERVER="${TRADUZAI_FAST_PAGE_SERVER:-1}"
export TRADUZAI_WORKER_WARMUP_ON_START="${TRADUZAI_WORKER_WARMUP_ON_START:-1}"
export TRADUZAI_WARMUP_PROFILE="${TRADUZAI_WARMUP_PROFILE:-quality}"
export TRADUZAI_WARMUP_LANG="${TRADUZAI_WARMUP_LANG:-en}"
export TRADUZAI_REQUIRE_GPU="${TRADUZAI_REQUIRE_GPU:-1}"
export TRADUZAI_ARTIFACT_PROFILE="${TRADUZAI_ARTIFACT_PROFILE:-fast}"
export TRADUZAI_ARTIFACT_UPLOAD_WORKERS="${TRADUZAI_ARTIFACT_UPLOAD_WORKERS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/pipeline:${PYTHONPATH:-}"

mkdir -p "$TRADUZAI_WORKER_WORK_DIR"

python -m worker doctor
python "$PROJECT_ROOT/scripts/vast/verify-gpu-stack.py"

if [ "${TRADUZAI_WORKER_ONCE:-0}" = "1" ]; then
  exec python -m worker --once
fi

exec python -m worker
