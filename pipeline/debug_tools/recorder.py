from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
import hashlib
import json
import logging
import os
import re
import threading

from .schemas import with_schema_header

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


@dataclass(frozen=True)
class DebugLevel:
    STANDARD = "standard"
    FULL = "full"
    MINIMAL = "minimal"


class DebugRecorder:
    """Central E2E debug recorder; debug failures never interrupt the pipeline."""

    def __init__(
        self,
        work_dir: Path,
        enabled: bool,
        run_id: str,
        *,
        level: str = DebugLevel.STANDARD,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.enabled = bool(enabled)
        self.run_id = run_id
        self.level = level
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._root = self.work_dir / "debug" / "e2e"
        self._manifest_path = self._root / "debug_manifest.json"
        self._events_path = self._root / "events.jsonl"
        self._errors_path = self._root / "debug_errors.jsonl"
        self._artifacts_path = self._root / "artifacts.jsonl"
        self._runtime_fingerprints_path = self._root / "00_run" / "runtime_fingerprints.jsonl"
        self._canonical_history_path = self._root / "00_run" / "canonical_artifacts.jsonl"
        self._canonical_manifest_path = self._root / "00_run" / "canonical_manifest.json"
        self._artifacts: list[dict[str, Any]] = []
        self._runtime_fingerprints: dict[str, dict[str, Any]] = {}
        self._runtime_fingerprint_summary: dict[str, dict[str, Any]] = {}
        self._canonical_entries: dict[str, dict[str, Any]] = {}
        self._canonical_text_metrics: dict[str, dict[str, Any]] = {}
        self._canonical_expected_page_ids: set[str] = set()
        self._canonical_expected_band_ids: set[str] = set()
        self._runtime_fingerprint_enabled = _env_flag("TRADUZAI_FLAG_RUNTIME_FINGERPRINT_V2")
        self._visual_baseline_enabled = _env_flag("TRADUZAI_FLAG_VISUAL_BASELINE_LOSSLESS_V2")
        self._runtime_fingerprint_lock = threading.RLock()
        self._canonical_lock = threading.RLock()
        self._stage_durations: dict[str, float] = {}
        if self.enabled:
            self._bootstrap_tree()

    def event(self, stage: str, action: str, payload: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            record = {
                "schema_version": 1,
                "run_id": self.run_id,
                "ts": self._clock(),
                "stage": stage,
                "action": action,
                **(payload or {}),
            }
            self._append_jsonl(self._events_path, record)
        except Exception as exc:
            self._record_error(stage=stage, action=action, exc=exc)

    def write_json(self, rel_path: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        stage = self._stage_from_rel(rel_path)
        try:
            target = self._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(self._header(payload, stage=stage), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.register_artifact(stage=stage, rel_path=rel_path, kind="json")
        except Exception as exc:
            self._record_error(stage=stage, action="write_json", exc=exc, rel_path=rel_path)

    def write_jsonl(self, rel_path: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        stage = self._stage_from_rel(rel_path)
        try:
            self._append_jsonl(self._root / rel_path, self._header(payload, stage=stage))
        except Exception as exc:
            self._record_error(stage=stage, action="write_jsonl", exc=exc, rel_path=rel_path)

    def write_image(self, rel_path: str, image: Any, *, quality: int = 88) -> None:
        if not self.enabled:
            return
        stage = self._stage_from_rel(rel_path)
        try:
            import cv2

            target = self._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            ext = target.suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            else:
                cv2.imwrite(str(target), image)
            self.register_artifact(stage=stage, rel_path=rel_path, kind="image")
        except Exception as exc:
            self._record_error(stage=stage, action="write_image", exc=exc, rel_path=rel_path)

    def register_artifact(
        self,
        stage: str,
        rel_path: str,
        kind: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "schema_version": 1,
            "stage": stage,
            "rel_path": rel_path,
            "kind": kind,
            "meta": meta or {},
        }
        self._artifacts.append(entry)
        try:
            self._append_jsonl(self._artifacts_path, entry)
        except Exception as exc:
            self._record_error(stage=stage, action="register_artifact", exc=exc, rel_path=rel_path)

    def record_runtime_fingerprint(self, stage: str, fingerprint: dict[str, Any]) -> None:
        if not self.enabled or not self._runtime_fingerprint_enabled:
            return
        stage_name = str(stage or "").strip() or "unknown"
        try:
            with self._runtime_fingerprint_lock:
                record = dict(fingerprint or {})
                record.setdefault("schema_version", 1)
                record["run_id"] = self.run_id
                record["stage"] = stage_name
                record["ts"] = self._clock()
                summary_fields = {
                    key: record.get(key)
                    for key in (
                        "stage",
                        "requested_engine",
                        "resolved_engine",
                        "executed_backend",
                        "resolution_status",
                        "execution_status",
                        "result_status",
                        "execution_context",
                        "fallback_used",
                        "fallback_reason",
                    )
                }
                summary_key = json.dumps(
                    summary_fields,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                summary = self._runtime_fingerprint_summary.setdefault(
                    summary_key,
                    {**summary_fields, "event_count": 0},
                )
                summary["event_count"] = int(summary.get("event_count", 0)) + 1
                previous = self._runtime_fingerprints.get(stage_name)
                if previous is not None:
                    comparable_previous = {
                        key: value for key, value in previous.items() if key not in {"run_id", "ts"}
                    }
                    comparable_record = {
                        key: value for key, value in record.items() if key not in {"run_id", "ts"}
                    }
                    if comparable_previous == comparable_record:
                        return
                self._append_jsonl(self._runtime_fingerprints_path, record)
                self._runtime_fingerprints[stage_name] = record
        except Exception as exc:
            self._record_error(stage=stage_name, action="record_runtime_fingerprint", exc=exc)

    def write_canonical_image(
        self,
        kind: str,
        image: Any,
        *,
        page_id: str = "",
        band_id: str = "",
        color_space: str = "bgr",
    ) -> dict[str, Any]:
        if not self.enabled or not self._visual_baseline_enabled:
            return {}
        normalized_kind = str(kind or "").strip().lower()
        try:
            if normalized_kind not in {"page", "final_band"}:
                raise ValueError(f"unsupported canonical image kind: {kind!r}")
            import cv2
            import numpy as np

            if not isinstance(image, np.ndarray) or image.size == 0:
                raise ValueError("canonical image must be a non-empty numpy array")
            canonical = np.ascontiguousarray(image)
            normalized_page_id = str(page_id or "").strip()
            normalized_band_id = str(band_id or "").strip()
            if normalized_kind == "page" and not normalized_page_id:
                raise ValueError("page_id is required for canonical pages")
            if normalized_kind == "final_band" and not normalized_band_id:
                raise ValueError("band_id is required for canonical final bands")

            identity = normalized_page_id if normalized_kind == "page" else normalized_band_id
            safe_identity = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity).strip("._") or "unknown"
            subdir = "canonical_pages" if normalized_kind == "page" else "canonical_final_bands"
            rel_path = f"00_run/{subdir}/{safe_identity}.png"
            target = self._root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded_ok, encoded = cv2.imencode(
                ".png",
                canonical,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not encoded_ok:
                raise RuntimeError(f"failed to encode canonical PNG: {identity}")
            png_bytes = encoded.tobytes()
            with self._canonical_lock:
                target.write_bytes(png_bytes)
                buffer_digest = hashlib.sha256()
                buffer_digest.update(str(canonical.dtype).encode("ascii", errors="replace") + b"\0")
                buffer_digest.update(",".join(str(value) for value in canonical.shape).encode("ascii") + b"\0")
                buffer_digest.update(canonical.tobytes(order="C"))
                key = f"{normalized_kind}:{normalized_page_id}:{normalized_band_id}"
                entry = {
                    "schema_version": 1,
                    "key": key,
                    "kind": normalized_kind,
                    "page_id": normalized_page_id,
                    "band_id": normalized_band_id,
                    "rel_path": rel_path,
                    "shape": [int(value) for value in canonical.shape],
                    "dtype": str(canonical.dtype),
                    "buffer_color_space": str(color_space or "").strip().lower(),
                    "encoded_color_space": "rgb",
                    "buffer_sha256": buffer_digest.hexdigest(),
                    "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
                }
                first_write = key not in self._canonical_entries
                self._canonical_entries[key] = entry
                self._append_jsonl(self._canonical_history_path, {**entry, "run_id": self.run_id})
                if first_write:
                    self.register_artifact(stage="run", rel_path=rel_path, kind="canonical_png", meta={"key": key})
            return entry
        except Exception as exc:
            self._record_error(stage="run", action="write_canonical_image", exc=exc, kind=normalized_kind)
            return {}

    def set_canonical_expected_coverage(
        self,
        *,
        page_ids: list[str] | tuple[str, ...] | set[str],
        band_ids: list[str] | tuple[str, ...] | set[str],
    ) -> None:
        if not self.enabled or not self._visual_baseline_enabled:
            return
        with self._canonical_lock:
            self._canonical_expected_page_ids.update({
                str(value).strip() for value in page_ids if str(value).strip()
            })
            self._canonical_expected_band_ids.update({
                str(value).strip() for value in band_ids if str(value).strip()
            })

    def record_canonical_text_metrics(
        self,
        records: list[dict[str, Any]],
        *,
        replace_existing: bool = False,
    ) -> None:
        if not self.enabled or not self._visual_baseline_enabled:
            return
        metric_fields = {
            "page_id",
            "band_id",
            "text_id",
            "text_instance_id",
            "trace_id",
            "source_bbox",
            "text_pixel_bbox",
            "target_bbox",
            "safe_text_box",
            "render_bbox",
            "font_size_final",
            "line_count",
            "line_height",
            "wrapped_lines",
            "fit_status",
            "render_balloon_containment",
            "render_outside_balloon",
            "qa_flags",
            "qa_metrics",
        }
        with self._canonical_lock:
            seen_in_batch: set[str] = set()
            for index, raw in enumerate(records or []):
                if not isinstance(raw, dict):
                    continue
                metric = {
                    key: _json_safe(raw.get(key))
                    for key in metric_fields
                    if key in raw
                }
                metric_id = str(metric.get("text_instance_id") or metric.get("trace_id") or "").strip()
                if not metric_id:
                    page_id = str(metric.get("page_id") or "").strip()
                    band_id = str(metric.get("band_id") or "").strip()
                    text_id = str(metric.get("text_id") or raw.get("id") or index).strip()
                    metric_id = "@".join(value for value in (text_id, band_id or page_id) if value)
                    metric["trace_id"] = metric_id
                metric["metric_id"] = metric_id
                existing = self._canonical_text_metrics.get(metric_id)
                duplicate_in_batch = metric_id in seen_in_batch
                seen_in_batch.add(metric_id)
                if duplicate_in_batch or (existing is not None and existing != metric and not replace_existing):
                    self._record_error(
                        stage="run",
                        action="duplicate_canonical_text_metric",
                        exc=ValueError(f"duplicate canonical text metric: {metric_id}"),
                        metric_id=metric_id,
                    )
                    continue
                if existing == metric:
                    continue
                self._canonical_text_metrics[metric_id] = metric

    @contextmanager
    def time_stage(self, stage_name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        import time

        started = time.perf_counter()
        try:
            yield
        finally:
            self._stage_durations[stage_name] = round(
                float(self._stage_durations.get(stage_name, 0.0)) + (time.perf_counter() - started),
                4,
            )

    def finalize(self, *, config_snapshot: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            manifest = {
                "schema_version": 1,
                "run_id": self.run_id,
                "created_at": self._clock(),
                "level": self.level,
                "stage_durations_sec": self._stage_durations,
                "artifact_count": len(self._artifacts),
                "config_snapshot": config_snapshot or {},
                **(extra or {}),
            }
            if self._runtime_fingerprint_enabled:
                manifest["runtime_fingerprint_count"] = len(self._runtime_fingerprints)
                manifest["runtime_fingerprints"] = [
                    self._runtime_fingerprints[key]
                    for key in sorted(self._runtime_fingerprints)
                ]
                manifest["runtime_fingerprint_summary"] = [
                    self._runtime_fingerprint_summary[key]
                    for key in sorted(self._runtime_fingerprint_summary)
                ]
            if self._visual_baseline_enabled:
                manifest["canonical_entry_count"] = len(self._canonical_entries)
            self._root.mkdir(parents=True, exist_ok=True)
            if self._visual_baseline_enabled:
                self._canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                self._canonical_manifest_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_id": self.run_id,
                            "entry_count": len(self._canonical_entries),
                            "expected_page_ids": sorted(self._canonical_expected_page_ids),
                            "expected_final_band_ids": sorted(self._canonical_expected_band_ids),
                            "text_metric_count": len(self._canonical_text_metrics),
                            "text_metrics": [
                                self._canonical_text_metrics[key]
                                for key in sorted(self._canonical_text_metrics)
                            ],
                            "entries": [self._canonical_entries[key] for key in sorted(self._canonical_entries)],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            self._manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("DebugRecorder.finalize falhou: %s", exc)

    def _bootstrap_tree(self) -> None:
        for sub in [
            "00_run",
            "01_input_extract",
            "02_strip_detect",
            "03_ocr",
            "04_text_normalization_router",
            "05_layout_geometry",
            "06_mask_segmentation",
            "07_translation",
            "08_inpaint",
            "09_typeset",
            "10_copyback_reassemble",
            "11_qa_export_gate",
            "12_contact_sheets",
            "13_report",
        ]:
            (self._root / sub).mkdir(parents=True, exist_ok=True)

    def _header(self, payload: dict[str, Any], stage: str) -> dict[str, Any]:
        return with_schema_header(payload if isinstance(payload, dict) else {"value": payload}, run_id=self.run_id, stage=stage)

    def _stage_from_rel(self, rel_path: str) -> str:
        head = rel_path.replace("\\", "/").split("/", 1)[0]
        return head[3:] if len(head) > 3 and head[:2].isdigit() and head[2] == "_" else "misc"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _record_error(self, **kwargs: Any) -> None:
        try:
            import traceback

            exc = kwargs.pop("exc", None)
            payload = {
                "schema_version": 1,
                "ts": self._clock(),
                "run_id": self.run_id,
                "traceback": traceback.format_exc(limit=4) if exc else "",
                **kwargs,
            }
            self._errors_path.parent.mkdir(parents=True, exist_ok=True)
            with self._errors_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
