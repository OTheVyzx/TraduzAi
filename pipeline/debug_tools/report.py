from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SEVERITY_RANK = {
    "critical": 4,
    "warning": 3,
    "high": 3,
    "medium": 2,
    "low": 1,
    "none": 0,
}

# Canonical confidence keys — see DBG2-01/03 and tools/analyze_e2e_debug.py.
CONFIDENCE_KEYS = ("confidence_raw", "ocr_confidence", "confianca_ocr", "confidence")


def generate_debug_report(
    e2e_root: str | Path,
    *,
    top_n: int = 10,
    project_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate debug_report.md/json from whatever E2E artifacts are present.

    When ``project_path`` is supplied, the per-run report adds:

    - ``text_count`` and ``content_class_counts`` derived from the **canonical**
      source per page (``page.text_layers`` if present, else ``page.textos``).
      Never sums the two — see DBG2-02.
    - ``confidence_zero_count`` and ``confidence_missing_count`` derived via
      ``canonical_confidence`` (``confidence_raw`` -> ``ocr_confidence`` ->
      ``confianca_ocr`` -> ``confidence``). Resolves DBG2-01/03.
    - ``13_report/debug_report_consistency.json`` describing stage-level
      vs aggregator divergences (DBG2-23 + §5b invariants).
    """

    root = Path(e2e_root)
    report_dir = root / "13_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    export_gate = _read_json(root / "11_qa_export_gate" / "export_gate.json")
    consistency = _read_json(root / "11_qa_export_gate" / "qa_export_gate_consistency.json")
    issues = _load_issues(root, export_gate)
    top_issues = [_normalize_issue(issue, root, report_dir) for issue in _sort_issues(issues)[:top_n]]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": str(export_gate.get("status") or "UNKNOWN"),
        "issue_count": len(issues),
        "top_issue_count": len(top_issues),
        "export_gate": {
            "status": export_gate.get("status"),
            "critical_issue_count": export_gate.get("critical_issue_count", 0),
            "review_issue_count": export_gate.get("review_issue_count", 0),
            "issue_count": export_gate.get("issue_count", len(issues)),
        },
        "qa_export_gate_consistency": consistency.get("consistency", {}),
        "top_issues": top_issues,
    }

    project_data: dict[str, Any] = {}
    if project_path is not None:
        project_data = _read_json(Path(project_path))

    if project_data:
        canonical_layers = _iter_canonical_text_layers(project_data)
        zero, missing = _confidence_counts(canonical_layers)
        payload["text_count"] = len(canonical_layers)
        payload["confidence_zero_count"] = zero
        payload["confidence_missing_count"] = missing
        payload["content_class_counts"] = dict(_content_class_counts(canonical_layers))
        payload["project_estatisticas_total_textos"] = _safe_int(
            (project_data.get("estatisticas") or {}).get("total_textos")
        )
        ocr_audit = _read_json(root / "03_ocr" / "ocr_confidence_audit.json")
        ocr_audit_summary = ocr_audit.get("summary") if isinstance(ocr_audit, dict) else {}
        if isinstance(ocr_audit_summary, dict):
            payload["ocr_confidence_audit_zero_count"] = _safe_int(
                ocr_audit_summary.get("blocks_with_confidence_zero")
            )

        stage_overreach_count = _count_jsonl_lines(
            root / "05_layout_geometry" / "source_bbox_balloon_overreach.jsonl"
        )
        source_bbox_count = _canonical_stage_count(
            stage_overreach_count,
            _source_bbox_equals_balloon_bbox_count(canonical_layers),
        )
        balloon_missing_count = _canonical_stage_count(
            _count_jsonl_lines(root / "09_typeset" / "balloon_bbox_missing_audit.jsonl"),
            0,
        )
        skip_summary = _skip_inpaint_summary(root)
        payload["source_bbox_equals_balloon_bbox_count"] = source_bbox_count
        payload["balloon_bbox_missing_count"] = balloon_missing_count
        payload["skip_inpaint_honored_bands"] = skip_summary["honored_bands"]
        payload["skip_inpaint_total_bands"] = skip_summary["total_bands"]
        payload["skip_inpaint_honored"] = skip_summary["honored"]
        payload["skip_inpaint_requested"] = skip_summary["requested"]

        consistency_payload = _build_consistency_payload(
            root,
            project_data=project_data,
            canonical_layers=canonical_layers,
            ocr_audit_summary=ocr_audit_summary if isinstance(ocr_audit_summary, dict) else {},
            source_bbox_equals_balloon_bbox_count=source_bbox_count,
            balloon_bbox_missing_count=balloon_missing_count,
            skip_summary=skip_summary,
        )
        (report_dir / "debug_report_consistency.json").write_text(
            json.dumps(
                {"schema_version": 1, **consistency_payload},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        payload["debug_report_consistency"] = consistency_payload

    (report_dir / "debug_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "debug_report.md").write_text(_render_markdown(payload), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Canonical sources (mirror of tools/analyze_e2e_debug.py helpers)
# ---------------------------------------------------------------------------


def _iter_canonical_text_layers(project: dict[str, Any]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    pages = project.get("paginas") or project.get("pages") or []
    if not isinstance(pages, list):
        return layers
    for page in pages:
        if not isinstance(page, dict):
            continue
        for candidate in (page.get("text_layers"), page.get("textos"), page.get("texts")):
            if isinstance(candidate, list) and candidate:
                layers.extend(item for item in candidate if isinstance(item, dict))
                break
    return layers


def canonical_confidence(layer: dict[str, Any]) -> float | None:
    for key in CONFIDENCE_KEYS:
        value = layer.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _confidence_counts(layers: list[dict[str, Any]]) -> tuple[int, int]:
    zero, missing = 0, 0
    for layer in layers:
        value = canonical_confidence(layer)
        if value is None:
            missing += 1
        elif value == 0.0:
            zero += 1
    return zero, missing


def _content_class_counts(layers: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for layer in layers:
        raw = (
            layer.get("content_class")
            or layer.get("classe_conteudo")
            or layer.get("tipo")
            or "unknown"
        )
        counts[str(raw)] += 1
    return counts


def _build_consistency_payload(
    e2e_root: Path,
    *,
    project_data: dict[str, Any],
    canonical_layers: list[dict[str, Any]],
    ocr_audit_summary: dict[str, Any],
    source_bbox_equals_balloon_bbox_count: int,
    balloon_bbox_missing_count: int,
    skip_summary: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    # text_count vs estatisticas (DBG2-02)
    aggregator_count = len(canonical_layers)
    total_textos = _safe_int((project_data.get("estatisticas") or {}).get("total_textos"))
    checks.append(
        {
            "name": "text_count_vs_estatisticas",
            "stage_level": total_textos,
            "aggregator": aggregator_count,
            "consistent": total_textos is None or aggregator_count == total_textos,
        }
    )

    # confidence zero count vs OCR audit (DBG2-01)
    zero, _ = _confidence_counts(canonical_layers)
    audit_zero = _safe_int(ocr_audit_summary.get("blocks_with_confidence_zero"))
    checks.append(
        {
            "name": "confidence_zero_count",
            "stage_level": audit_zero,
            "aggregator": zero,
            "consistent": audit_zero is None or audit_zero == zero,
        }
    )

    # source_bbox_balloon_overreach.jsonl vs aggregator (DBG2-23)
    overreach_path = e2e_root / "05_layout_geometry" / "source_bbox_balloon_overreach.jsonl"
    stage_overreach = _count_jsonl_lines(overreach_path)
    aggregator_overreach = source_bbox_equals_balloon_bbox_count
    checks.append(
        {
            "name": "source_bbox_overreach_count",
            "stage_level": stage_overreach,
            "aggregator": aggregator_overreach,
            "consistent": stage_overreach is None
            or stage_overreach == aggregator_overreach,
        }
    )

    # balloon_bbox_missing_audit.jsonl vs aggregate metric (DBG2-23)
    stage_missing = _count_jsonl_lines(
        e2e_root / "09_typeset" / "balloon_bbox_missing_audit.jsonl"
    )
    checks.append(
        {
            "name": "balloon_bbox_missing_count",
            "stage_level": stage_missing,
            "aggregator": balloon_bbox_missing_count,
            "consistent": stage_missing is None
            or stage_missing == balloon_bbox_missing_count,
        }
    )

    # skip_inpaint_honored from per-band decisions vs aggregate metric (DBG2-21)
    requested = skip_summary["requested"]
    stage_honored = True if requested else False
    aggregator_honored = skip_summary["honored"]
    checks.append(
        {
            "name": "skip_inpaint_honored",
            "stage_level": stage_honored,
            "aggregator": aggregator_honored,
            "consistent": aggregator_honored is None
            or aggregator_honored == stage_honored,
        }
    )

    all_consistent = all(check["consistent"] is True for check in checks)
    return {"all_consistent": all_consistent, "checks": checks}


def _count_jsonl_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    count = 0
    for line in text.splitlines():
        if line.strip():
            count += 1
    return count


def _canonical_stage_count(stage_count: int | None, fallback_count: int) -> int:
    return stage_count if stage_count is not None else fallback_count


def _source_bbox_equals_balloon_bbox_count(layers: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for layer in layers
        if _bbox_equals(layer.get("source_bbox"), layer.get("balloon_bbox"))
    )


def _skip_inpaint_summary(e2e_root: Path) -> dict[str, Any]:
    runner_config = _read_json(e2e_root / "00_run" / "runner_config_snapshot.json")
    requested = bool(runner_config.get("skip_inpaint"))
    decisions = _collect_inpaint_decisions(e2e_root)
    total_bands = len(decisions)
    honored_bands = sum(
        1 for decision in decisions if decision.get("skip_inpaint_honored") is True
    )

    if requested:
        honored = True if total_bands == 0 else honored_bands == total_bands
    elif total_bands == 0:
        honored = None
    else:
        honored = honored_bands == total_bands

    return {
        "requested": requested,
        "honored": honored,
        "honored_bands": honored_bands,
        "total_bands": total_bands,
    }


def _collect_inpaint_decisions(e2e_root: Path) -> list[dict[str, Any]]:
    base = e2e_root / "08_inpaint"
    if not base.exists():
        return []
    decisions: list[dict[str, Any]] = []
    try:
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            payload = _read_json(child / "inpaint_decision.json")
            if payload:
                decisions.append(payload)
    except OSError:
        return []
    return decisions


def _bbox_equals(a: Any, b: Any) -> bool:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return False
    try:
        return tuple(float(x) for x in a) == tuple(float(x) for x in b)
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Existing helpers (unchanged)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _load_issues(root: Path, export_gate: dict[str, Any]) -> list[dict[str, Any]]:
    issues = export_gate.get("issues")
    if isinstance(issues, list):
        return [issue for issue in issues if isinstance(issue, dict)]

    issues_path = root / "11_qa_export_gate" / "qa_issues.jsonl"
    loaded: list[dict[str, Any]] = []
    try:
        for line in issues_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            issue = json.loads(line)
            if isinstance(issue, dict):
                loaded.append(issue)
    except Exception:
        return []
    return loaded


def _sort_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        issues,
        key=lambda issue: (
            SEVERITY_RANK.get(str(issue.get("severity") or "low"), 0),
            -_int_or_zero(issue.get("page")),
        ),
        reverse=True,
    )


def _normalize_issue(issue: dict[str, Any], root: Path, report_dir: Path) -> dict[str, Any]:
    flags = issue.get("flags") if isinstance(issue.get("flags"), list) else []
    normalized = {
        "severity": str(issue.get("severity") or "unknown"),
        "type": str(issue.get("type") or "unknown"),
        "page": issue.get("page"),
        "layer": issue.get("layer"),
        "flags": [str(flag) for flag in flags],
        "text": str(issue.get("text") or ""),
        "artifact_links": _artifact_links(issue, root, report_dir),
    }
    for key in (
        "trace_id",
        "text_id",
        "text_instance_id",
        "page_id",
        "band_id",
        "coordinate_space",
    ):
        value = issue.get(key)
        if value:
            normalized[key] = value
    bbox = issue.get("bbox")
    if bbox is not None:
        normalized["bbox"] = bbox
    for key in ("source_bbox", "balloon_bbox", "safe_text_box", "render_bbox"):
        value = issue.get(key)
        if value is not None:
            normalized[key] = value
    return normalized


def _artifact_links(issue: dict[str, Any], root: Path, report_dir: Path) -> list[dict[str, str]]:
    links = []
    for rel_path in _iter_artifact_paths(issue):
        target = Path(rel_path)
        if target.is_absolute():
            try:
                display_path = target.relative_to(root)
            except ValueError:
                continue
        else:
            display_path = target
            target = root / target
        href = os.path.relpath(target, report_dir).replace("\\", "/")
        label = display_path.as_posix()
        links.append({"label": label, "href": href})
    return links


def _iter_artifact_paths(issue: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("linked_artifacts", "artifact_links", "artifacts"):
        value = issue.get(key)
        if isinstance(value, list):
            values.extend(value)
    for key in ("artifact", "rel_path", "path"):
        value = issue.get(key)
        if value:
            values.append(value)

    paths: list[str] = []
    for value in values:
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict):
            for key in ("rel_path", "path", "href"):
                if value.get(key):
                    paths.append(str(value[key]))
                    break
    return paths


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Debug Report",
        "",
        f"- Status: {payload['status']}",
        f"- Issues: {payload['issue_count']}",
    ]
    if "text_count" in payload:
        lines.append(f"- Texts (canonical): {payload['text_count']}")
    if "confidence_zero_count" in payload:
        lines.append(
            "- Confidence: zero=%s, missing=%s"
            % (payload.get("confidence_zero_count"), payload.get("confidence_missing_count"))
        )
    if "debug_report_consistency" in payload:
        consistency = payload["debug_report_consistency"]
        lines.append(
            f"- Consistency: {'OK' if consistency.get('all_consistent') else 'MISMATCH'}"
        )
    lines.extend([
        "",
        "## Top Issues",
        "",
    ])
    top_issues = payload.get("top_issues") or []
    if not top_issues:
        lines.append("Nenhuma issue encontrada.")
        lines.append("")
        return "\n".join(lines)

    for index, issue in enumerate(top_issues, start=1):
        flags = ", ".join(issue.get("flags") or []) or "sem flags"
        page = issue.get("page") or "?"
        layer = issue.get("layer") or "?"
        lines.append(f"{index}. **{issue['severity']}** `{issue['type']}` page `{page}` layer `{layer}`")
        trace_id = issue.get("trace_id")
        if trace_id:
            lines.append(f"   - Trace: `{trace_id}`")
        lines.append(f"   - Flags: {flags}")
        artifact_links = issue.get("artifact_links") or []
        if artifact_links:
            rendered = ", ".join(f"[{link['label']}]({link['href']})" for link in artifact_links)
            lines.append(f"   - Artifacts: {rendered}")
    lines.append("")
    return "\n".join(lines)
