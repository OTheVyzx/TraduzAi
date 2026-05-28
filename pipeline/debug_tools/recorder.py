from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
import json
import logging

from .schemas import with_schema_header

logger = logging.getLogger(__name__)


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
        self._artifacts: list[dict[str, Any]] = []
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
            self._root.mkdir(parents=True, exist_ok=True)
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
