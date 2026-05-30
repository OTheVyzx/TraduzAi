#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${TRADUZAI_PROJECT_ROOT:-/workspace/TraduzAI}"
VENV_DIR="${TRADUZAI_VENV_DIR:-$PROJECT_ROOT/.venv}"
MODELS_DIR="${TRADUZAI_MODELS_DIR:-$PROJECT_ROOT/pipeline/models}"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"

export TRADUZAI_PROJECT_ROOT="$PROJECT_ROOT"
export TRADUZAI_PIPELINE_MAIN="${TRADUZAI_PIPELINE_MAIN:-$PROJECT_ROOT/pipeline/main.py}"
export TRADUZAI_PIPELINE_PYTHON="${TRADUZAI_PIPELINE_PYTHON:-$VENV_DIR/bin/python}"
export TRADUZAI_MODELS_DIR="$MODELS_DIR"
export PYTHONPATH="$PROJECT_ROOT/pipeline:${PYTHONPATH:-}"

python -m worker doctor
python - <<'PY'
import json
import os

from worker.config import WorkerSettings
from worker.fast_page import FastPageProcessClient

settings = WorkerSettings.from_env()
client = FastPageProcessClient(settings)
try:
    events = client.warmup(
        {
            "models_dir": os.environ.get("TRADUZAI_MODELS_DIR") or str(settings.project_root / "pipeline" / "models"),
            "profile": os.environ.get("TRADUZAI_WARMUP_PROFILE", "quality"),
            "lang": os.environ.get("TRADUZAI_WARMUP_LANG", "en"),
        }
    )
    print(json.dumps(events[-1] if events else {"type": "ready"}, ensure_ascii=False))
finally:
    client.close()
PY
