from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT / "pipeline"
for path in (str(ROOT), str(PIPELINE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline.main import _preload_detect_ocr_page, _run_detect_page  # noqa: E402


def _quiet_call(fn, *args, **kwargs):
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        return fn(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_json")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--engine-preset", default="")
    args = parser.parse_args()

    project_path = Path(args.project_json)
    page = int(args.page)
    options = {
        "idioma_origem": args.source_lang,
        "engine_preset_id": args.engine_preset,
    }

    t0 = time.perf_counter()
    _quiet_call(_preload_detect_ocr_page, project_path, page, options)
    preload_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    _quiet_call(_run_detect_page, project_path, page, None, options)
    cached_click_sec = time.perf_counter() - t1

    print(
        json.dumps(
            {
                "project_json": str(project_path),
                "page": page,
                "preload_sec": round(preload_sec, 3),
                "cached_click_sec": round(cached_click_sec, 3),
                "speedup": round(preload_sec / cached_click_sec, 2) if cached_click_sec > 0 else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
