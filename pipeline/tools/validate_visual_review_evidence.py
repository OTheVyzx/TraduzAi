"""Validate completeness and lineage of manually recorded visual verdicts."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Sequence

from ownership.hash_contract import canonical_page_sha256, sha256_file

from PIL import Image


class VisualReviewEvidenceError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VisualReviewEvidenceError(f"evidence must be an object: {path}")
    return payload


def _validate_items(items, expected_ids: list[str], *, label: str) -> None:
    if not isinstance(items, list):
        raise VisualReviewEvidenceError(f"{label} items are missing")
    ids = [str(item.get("page_id") or "") for item in items if isinstance(item, dict)]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise VisualReviewEvidenceError(f"{label} page set/order differs from audit")
    for item in items:
        verdict = str(item.get("visual_verdict") or "")
        if verdict not in {"GO", "NO_GO"} or not item.get("reviewer") or not item.get("reviewed_at"):
            raise VisualReviewEvidenceError(f"{label}/{item.get('page_id')}: manual verdict is incomplete")
        try:
            datetime.fromisoformat(str(item["reviewed_at"]).replace("Z", "+00:00"))
        except ValueError as exc:
            raise VisualReviewEvidenceError(f"{label}/{item.get('page_id')}: reviewed_at is invalid") from exc
        for prefix in ("source", "final"):
            path = Path(str(item.get(f"{prefix}_path") or ""))
            expected_hash = str(item.get(f"{prefix}_file_sha256") or "")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise VisualReviewEvidenceError(f"{label}/{item.get('page_id')}: {prefix} artifact is stale")
        with Image.open(Path(str(item["final_path"]))) as opened:
            final_pixel_sha256 = canonical_page_sha256(opened.convert("RGB"))
        if str(item.get("final_pixel_sha256") or "") != final_pixel_sha256:
            raise VisualReviewEvidenceError(f"{label}/{item.get('page_id')}: final pixel hash is stale")


def validate_visual_review_files(
    *, off_audit: Path, render_audit: Path, matrix_manifest: Path,
    matrix_audit_dir: Path, off_evidence: Path, render_evidence: Path,
    matrix_evidence: Path, required_sentinel_pages: Sequence[int], require_complete: bool,
) -> dict[str, Any]:
    off, render = _load(off_audit), _load(render_audit)
    off_review, render_review = _load(off_evidence), _load(render_evidence)
    expected_off = [str(value) for value in off.get("source_manifest_page_ids") or ()]
    expected_render = [str(value) for value in render.get("source_manifest_page_ids") or ()]
    if require_complete:
        _validate_items(off_review.get("items"), expected_off, label="off")
        _validate_items(render_review.get("items"), expected_render, label="render")
        for label, audit, evidence in (("off", off, off_review), ("render", render, render_review)):
            expected_execution = str(audit.get("execution_id") or "")
            if not expected_execution or any(
                str(item.get("execution_id") or "") != expected_execution
                for item in evidence.get("items") or ()
            ):
                raise VisualReviewEvidenceError(f"{label}: review execution lineage mismatch")
    matrix = _load(matrix_manifest)
    matrix_review = _load(matrix_evidence)
    expected_entries = [str(item.get("entry_id") or "") for item in matrix.get("entries") or ()]
    reviewed_entries = [str(item.get("entry_id") or "") for item in matrix_review.get("entries") or ()]
    if reviewed_entries != expected_entries or len(reviewed_entries) != len(set(reviewed_entries)):
        raise VisualReviewEvidenceError("matrix review entry set differs from manifest")
    missing_audits = [entry for entry in expected_entries if not (matrix_audit_dir / f"{entry}.json").is_file()]
    if missing_audits:
        raise VisualReviewEvidenceError("matrix audits missing: " + ", ".join(missing_audits))
    matrix_by_id = {
        str(item.get("entry_id") or ""): item
        for item in matrix_review.get("entries") or ()
        if isinstance(item, dict)
    }
    if require_complete:
        for entry_id in expected_entries:
            audit = _load(matrix_audit_dir / f"{entry_id}.json")
            expected_pages = [str(value) for value in audit.get("source_manifest_page_ids") or ()]
            _validate_items(matrix_by_id[entry_id].get("items"), expected_pages, label=f"matrix/{entry_id}")
    sentinels = {f"page_{int(value):03d}" for value in required_sentinel_pages}
    for label, review in (("off", off_review), ("render", render_review)):
        stage_rows = review.get("sentinel_stages") or []
        by_page = {}
        for row in stage_rows:
            by_page.setdefault(str(row.get("page_id") or ""), set()).add(str(row.get("stage") or ""))
        expected_stages = {"original", "coverage_ocr_overlay", "inpaint", "typeset", "page_composition", "persisted_final"}
        for page_id in sentinels:
            if by_page.get(page_id) != expected_stages:
                raise VisualReviewEvidenceError(f"{label}/{page_id}: six-stage evidence is incomplete")
        for row in stage_rows:
            path = Path(str(row.get("path") or ""))
            if not path.is_file() or sha256_file(path) != str(row.get("file_sha256") or ""):
                raise VisualReviewEvidenceError(
                    f"{label}/{row.get('page_id')}/{row.get('stage')}: stage artifact is stale"
                )
            page = next(
                (item for item in review.get("items") or () if item.get("page_id") == row.get("page_id")),
                None,
            )
            if page is None or str(row.get("execution_id") or "") != str(page.get("execution_id") or ""):
                raise VisualReviewEvidenceError(
                    f"{label}/{row.get('page_id')}/{row.get('stage')}: stage lineage mismatch"
                )
    return {
        "schema_version": 1,
        "status": "PASS",
        "off_page_count": len(expected_off),
        "render_page_count": len(expected_render),
        "matrix_entry_count": len(expected_entries),
        "sentinel_page_count": len(sentinels),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "off-audit", "render-audit", "matrix-manifest", "matrix-audit-dir",
        "off-evidence", "render-evidence", "matrix-evidence", "report",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--required-sentinel-pages", default="")
    parser.add_argument("--require-complete", action="store_true")
    try:
        args = parser.parse_args(argv)
        result = validate_visual_review_files(
            off_audit=Path(args.off_audit).resolve(), render_audit=Path(args.render_audit).resolve(),
            matrix_manifest=Path(args.matrix_manifest).resolve(), matrix_audit_dir=Path(args.matrix_audit_dir).resolve(),
            off_evidence=Path(args.off_evidence).resolve(), render_evidence=Path(args.render_evidence).resolve(),
            matrix_evidence=Path(args.matrix_evidence).resolve(),
            required_sentinel_pages=tuple(int(value) for value in args.required_sentinel_pages.split(",") if value.strip()),
            require_complete=bool(args.require_complete),
        )
        report = Path(args.report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        print(f"visual review evidence error: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
