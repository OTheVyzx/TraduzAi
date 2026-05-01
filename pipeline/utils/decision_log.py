from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TRACE_PATH: Path | None = None
_QA_PATH: Path | None = None
_ENTRIES: list[dict[str, Any]] = []
_RUN_META: dict[str, Any] = {}


def infer_page_number(image_ref: str | Path | None) -> int | None:
    if image_ref is None:
        return None
    text = str(image_ref)
    match = re.search(r"(?<!\d)(\d{3})(?:__\d{3})?(?!\d)", Path(text).stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def configure_decision_trace(work_dir: str | Path, run_meta: dict[str, Any] | None = None) -> None:
    global _TRACE_PATH, _QA_PATH, _ENTRIES, _RUN_META

    base = Path(work_dir)
    base.mkdir(parents=True, exist_ok=True)
    trace_path = base / "decision_trace.jsonl"
    qa_path = base / "qa_report.json"

    with _LOCK:
        _TRACE_PATH = trace_path
        _QA_PATH = qa_path
        _ENTRIES = []
        _RUN_META = dict(run_meta or {})
        _TRACE_PATH.write_text("", encoding="utf-8")


def record_decision(
    *,
    stage: str,
    action: str,
    reason: str,
    page: int | None = None,
    layer: str | None = None,
    text: str | None = None,
    bbox: list[int] | None = None,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "action": action,
        "reason": reason,
    }
    if page is not None:
        entry["page"] = int(page)
    if layer:
        entry["layer"] = str(layer)
    if text:
        entry["text"] = str(text)[:240]
    if bbox:
        entry["bbox"] = [int(v) for v in bbox]
    if details:
        entry["details"] = details

    message_parts = [
        f"stage={stage}",
        f"action={action}",
        f"reason={reason}",
    ]
    if page is not None:
        message_parts.append(f"page={page}")
    if layer:
        message_parts.append(f"layer={layer}")
    if text:
        compact = " ".join(str(text).split())
        if len(compact) > 80:
            compact = compact[:77] + "..."
        message_parts.append(f'text="{compact}"')

    logger.log(level, "DECISAO | " + " | ".join(message_parts))

    with _LOCK:
        _ENTRIES.append(entry)
        if _TRACE_PATH is not None:
            try:
                _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with _TRACE_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except OSError:
                pass

    return entry


def finalize_decision_trace(run_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    global _RUN_META
    with _LOCK:
        if run_meta:
            _RUN_META.update(run_meta)

        by_stage = Counter(entry["stage"] for entry in _ENTRIES)
        by_action = Counter(entry["action"] for entry in _ENTRIES)
        by_reason = Counter(entry["reason"] for entry in _ENTRIES)
        flagged_pages = sorted(
            {
                int(entry["page"])
                for entry in _ENTRIES
                if isinstance(entry.get("page"), int)
            }
        )

        report = {
            "run": dict(_RUN_META),
            "summary": {
                "total_decisions": len(_ENTRIES),
                "by_stage": dict(by_stage),
                "by_action": dict(by_action),
                "by_reason": dict(by_reason),
            },
            "flagged_pages": flagged_pages,
            "entries_preview": _ENTRIES[:200],
        }

        if _QA_PATH is not None:
            _QA_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
