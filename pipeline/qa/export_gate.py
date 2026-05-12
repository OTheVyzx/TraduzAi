"""Export blocking policy for known P0 render issues."""

from __future__ import annotations

import re
from typing import Any


SOURCE_SCRIPT_RE = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)

P0_FLAGS = {
    "vlm_failure_phrase",
    "translation_fallback_phrase",
    "glossary_violation",
    "forbidden_translation",
    "placeholder_lost",
    "unrestored_placeholder",
    "speech_cjk_preserved_inside_balloon",
    "source_script_leak",
}

HIGH_LAYOUT_FLAGS = {
    "text_overflow_high",
    "outline_damage_high",
}


def evaluate_export_gate(project: dict[str, Any], *, override: bool = False) -> dict[str, Any]:
    issues = collect_export_blocking_issues(project)
    status = "PASS"
    if issues:
        status = "OVERRIDDEN" if override else "BLOCK"
    return {
        "status": status,
        "allowed": status != "BLOCK",
        "override": bool(override),
        "issue_count": len(issues),
        "issues": issues,
    }


def collect_export_blocking_issues(project: dict[str, Any]) -> list[dict[str, Any]]:
    source_lang = str(project.get("idioma_origem") or "").lower()
    cjk_source = source_lang in {"ja", "jp", "ko", "kr", "zh", "zh-cn", "zh-tw"}
    issues: list[dict[str, Any]] = []
    for page_index, page in enumerate(project.get("paginas") or [], start=1):
        layers = page.get("text_layers") or page.get("textos") or []
        for layer_index, layer in enumerate(layers, start=1):
            if not isinstance(layer, dict):
                continue
            if layer.get("skip_processing"):
                continue
            translated = str(layer.get("translated") or layer.get("traduzido") or "")
            flags = {str(flag) for flag in layer.get("qa_flags") or [] if flag}
            matched_flags = sorted(flags & P0_FLAGS)
            if cjk_source and translated and SOURCE_SCRIPT_RE.search(translated):
                matched_flags.append("speech_cjk_preserved_inside_balloon")
            if flags & HIGH_LAYOUT_FLAGS:
                matched_flags.extend(sorted(flags & HIGH_LAYOUT_FLAGS))
            if not matched_flags:
                continue
            issues.append(
                {
                    "page": int(page.get("numero") or page_index),
                    "layer": layer.get("id") or f"t{layer_index}",
                    "type": "p0_render_blocker",
                    "severity": "critical",
                    "flags": sorted(set(matched_flags)),
                    "text": translated[:160],
                    "bbox": layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox"),
                }
            )
    return issues
