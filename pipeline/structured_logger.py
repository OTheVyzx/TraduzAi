"""Structured JSONL logger with deduplication and run summary."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, logs_root: str | Path, run_id: str):
        self.logs_root = Path(logs_root)
        self.run_id = run_id
        self.run_dir = self.logs_root / run_id
        self.path = self.run_dir / "structured_log.jsonl"
        self.start = time.perf_counter()
        self._seen: set[str] = set()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def event_id(self, stage: str, page: int | None, region_id: str | None, event: str, payload: dict[str, Any]) -> str:
        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        raw = f"{self.run_id}|{stage}|{page}|{region_id}|{event}|{payload_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def log(
        self,
        *,
        stage: str,
        event: str,
        page: int | None = None,
        region_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = payload or {}
        event_id = self.event_id(stage, page, region_id, event, payload)
        record = {
            "event_id": event_id,
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "page": page,
            "region_id": region_id,
            "event": event,
            "payload": payload,
            "duration_seconds": max(0.0, time.perf_counter() - self.start),
        }
        if event_id in self._seen:
            return record
        self._seen.add(event_id)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def build_log_summary(project: dict[str, Any]) -> dict[str, Any]:
    pages = project.get("paginas") or []
    regions = [layer for page in pages for layer in page.get("text_layers", []) or []]
    qa_flags = [flag for region in regions for flag in region.get("qa_flags", []) or []]
    critical_flags = int(((project.get("qa") or {}).get("summary") or {}).get("critical_count", 0) or 0)
    return {
        "actual_pages": int((project.get("estatisticas") or {}).get("total_paginas", len(pages)) or len(pages)),
        "processed_pages": len(pages),
        "translated_regions": sum(1 for region in regions if region.get("translated") or region.get("traduzido")),
        "qa_flags": len(qa_flags),
        "critical_flags": critical_flags,
        "glossary_hits": sum(len(region.get("glossary_hits", []) or []) for region in regions),
        "entity_flags": sum(len(region.get("entity_flags", []) or []) for region in regions),
        "export_status": "with_warnings" if qa_flags else "clean",
    }
