"""Export blocking policy for known P0 render issues."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from qa.translation_qa import severity_for_flag


SOURCE_SCRIPT_RE = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)
EXPORT_BLOCKING_REVIEW_FLAGS = {
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "untranslated_english",
    "gibberish_detected",
}
CONFIRMED_VISUAL_DAMAGE_FLAGS = {
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "render_outside_balloon",
    "render_bbox_far_from_target_bbox",
    "page_space_rerender_mixed_coordinates",
    "missing_render_bbox",
    "text_residual_after_inpaint",
    "text_residual_after_inpaint_confirmed",
    "fast_fill_unverified_residual",
    "fast_fill_insufficient_coverage",
}
IGNORED_LEGACY_FLAGS = {
    "low_confidence_visual_noise",
    "cover_title_logo",
    "mask_density_high",
}
ROUTE_ACTIONS_REQUIRING_EXPORT_GATE = {
    "review_required",
    "translate_inpaint_render",
    "translate_render_only",
    "translate_sfx_inpaint_render",
}
SCANLATION_CREDIT_RE = re.compile(
    r"\b(?:"
    r"RESET\s*SCANS?|SCANSHOMEMANGA|SCANLATOR|SCANS?|UTOON|HIVE(?:TOON)?|TOON\s*\.?\s*(?:NET|NETS|NETE|net)|"
    r"NEW\s*TOKI?|[A-Z0-9]*TOK[A-Z0-9]*|[A-Z0-9]*IOKI[A-Z0-9]*|NEWTO\w*|NEVTO\w*|NEYTO\w*|NWTOK\w*|WTOK\w*|TOKLJ?G?O|"
    r"(?=[A-Z0-9]*\d)[A-Z0-9]{5,}\s*\.?\s*COM|"
    r"DISCORD|PATREON|PAYPAL|KO-?FI|DEVMAX|SUPPORT\s*US|SPECIAL\s+THANKS|"
    r"CONTACT|INVITE|RECRUITING|TRANSLATORS?|EDITORS?|TYPESETTERS?|READ\s+ON|"
    r"FOR\s+FASTER\s+UPP?ATE|CONTENTS?\s+LAB|"
    r"BRONZE|SILVER|GOLD|DIAMOND|PLATINUM|GOATBEARDS|DRAGENDAVE|SILICONMAGE|GUDPLAYUR|"
    r"ALL\s+COMICS\s+ON\s+THIS\s+WEBSITE|ORIGINAL\s+VERSION|FOR\s+THE\s+ORIGINAL"
    r")\b",
    re.IGNORECASE,
)
SCANLATION_VISUAL_REVIEW_ONLY_FLAGS = {
    "render_on_art_suspected",
    "TEXT_CLIPPED",
    "TEXT_OVERFLOW",
    "render_outside_balloon",
    "render_bbox_far_from_target_bbox",
    "text_residual_after_inpaint",
    "text_residual_after_inpaint_confirmed",
    "fast_fill_unverified_residual",
    "fast_fill_insufficient_coverage",
    "fast_fill_no_glyph_evidence",
    "fit_below_minimum_legible",
    "missing_render_bbox",
    "layout_bbox_coordinate_mismatch",
    "mask_outside_balloon_critical",
    "source_glyph_area_ratio_critical",
    "mask_outside_balloon",
    "weak_text_residual_after_inpaint",
}
SCANLATION_HARMLESS_CONTEXT_FLAGS = {
    "compact_small_text_capacity",
    "connected_lobe_boxes_missing_source_anchor_fallback",
    "ocr_art_fragment_suspected",
    "ocr_partial_low_confidence_fragment",
    "ocr_run_on_suspect",
    "rotated_text_recovery",
    "safe_text_box_recomputed",
    "tiny_bubble_inner_bbox_rejected",
}
SCANLATION_TIER_WORDS = {"BRONZE", "SILVER", "GOLD", "DIAMOND", "PLATINUM"}
SCANLATION_CONTEXTUAL_RE = re.compile(
    r"\b(?:WARNING|NOTICE|READ\s+THIS|OFFICIAL\s+SITE|FASTER\s+UPDATES?)\b",
    re.IGNORECASE,
)
FINAL_PIXEL_CONTRACTS = {
    "source_coverage_contract",
    "owner_graph_contract",
    "route_state_contract",
    "pixel_ownership_contract",
    "final_language_contract",
    "layout_legibility_contract",
    "residual_cleanup_contract",
    "protected_art_contract",
    "qa_integrity_contract",
}


def _final_pixel_blocker(
    *,
    page_id: str,
    page_number: int | None,
    reason: str,
    owner_id: str | None = None,
    component_ids: list[str] | None = None,
    offenders: list[str] | None = None,
    issue_id: str | None = None,
    contract: str | None = None,
) -> dict[str, Any]:
    stable_owner = owner_id or "page"
    stable_trace = f"{page_id}:{stable_owner}:{reason}"
    artifact_links = [
        "04_text_normalization_router/page_owner_graph.json",
        "04_text_normalization_router/source_evidence_ledger.jsonl",
        "06_mask_segmentation/owner_masks",
        "08_inpaint/owner_cleanup_contracts.jsonl",
        "09_typeset/render_plan_final.jsonl",
        "11_qa_export_gate/final_pixel_ocr.jsonl",
        "11_qa_export_gate/persisted_artifact_hashes.jsonl",
    ]
    issue: dict[str, Any] = {
        "page": page_number,
        "page_id": page_id,
        "type": "final_pixel_contract",
        "issue_scope": "page",
        "severity": "critical",
        "blocks_export": True,
        "source": "final_pixel_qa",
        "reason": reason,
        "flags": [reason],
        "component_ids": list(component_ids or []),
        "offenders": list(offenders or [reason]),
        "trace_id": stable_trace,
        "artifact_links": artifact_links,
        "linked_artifacts": artifact_links,
    }
    if owner_id:
        issue["owner_id"] = owner_id
    if issue_id:
        issue["issue_id"] = issue_id
    if contract:
        issue["contract"] = contract
    return issue


def _collect_final_pixel_report_issues(project: dict[str, Any]) -> list[dict[str, Any]]:
    """Fail closed on final persisted pixels for verified owner-graph projects."""
    if str(project.get("owner_graph_status") or "").strip().lower() != "verified":
        return []

    pages: list[tuple[str, int | None]] = []
    for index, page in enumerate(project.get("paginas") or [], start=1):
        if not isinstance(page, dict):
            continue
        try:
            page_number = int(page.get("numero") or index)
        except (TypeError, ValueError):
            page_number = index
        pages.append((str(page.get("page_id") or f"page_{page_number:03d}"), page_number))

    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    raw_reports = qa.get("final_pixel_reports")
    reports = raw_reports if isinstance(raw_reports, list) else []
    reports_by_page: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        report_page_id = str(report.get("page_id") or "").strip()
        if report_page_id:
            reports_by_page.setdefault(report_page_id, []).append(report)

    issues: list[dict[str, Any]] = []
    expected_page_ids = {page_id for page_id, _ in pages}
    for page_id, page_number in pages:
        matches = reports_by_page.get(page_id, [])
        if len(matches) != 1:
            reason = "final_pixel_report_missing" if not matches else "final_pixel_report_duplicate"
            issues.append(
                _final_pixel_blocker(page_id=page_id, page_number=page_number, reason=reason)
            )
            continue
        report = matches[0]
        observer = str(report.get("observer") or "").strip()
        if report.get("observer_available") is not True or not observer:
            issues.append(
                _final_pixel_blocker(
                    page_id=page_id,
                    page_number=page_number,
                    reason="final_pixel_observer_unavailable",
                )
            )
            continue
        if report.get("observation_complete") is not True:
            issues.append(
                _final_pixel_blocker(
                    page_id=page_id,
                    page_number=page_number,
                    reason="final_pixel_observation_incomplete",
                )
            )
            continue
        if report.get("coverage_complete") is False:
            failures = [
                str(value)
                for value in report.get("coverage_failures") or []
                if str(value).strip()
            ] or ["final_pixel_observation_incomplete"]
            for failure in failures:
                issues.append(
                    _final_pixel_blocker(
                        page_id=page_id,
                        page_number=page_number,
                        reason=failure,
                        offenders=[
                            f"detected_blocks:{int(report.get('detected_block_count', 0) or 0)}",
                            f"ocr_records:{int(report.get('ocr_record_count', 0) or 0)}",
                        ],
                        contract="qa_integrity_contract",
                    )
                )
            continue

        artifact_path = Path(str(report.get("artifact_path") or ""))
        expected_hash = str(report.get("persisted_sha256") or "").strip().lower()
        hash_is_well_formed = bool(re.fullmatch(r"[0-9a-f]{64}", expected_hash))
        try:
            actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except (OSError, ValueError):
            actual_hash = ""
        if not hash_is_well_formed or actual_hash != expected_hash:
            issues.append(
                _final_pixel_blocker(
                    page_id=page_id,
                    page_number=page_number,
                    reason="final_pixel_artifact_hash_stale",
                )
            )
            continue

        contracts = report.get("contracts")
        if not isinstance(contracts, dict) or not FINAL_PIXEL_CONTRACTS.issubset(contracts):
            issues.append(
                _final_pixel_blocker(
                    page_id=page_id,
                    page_number=page_number,
                    reason="final_pixel_contracts_incomplete",
                )
            )
            continue

        report_issues = report.get("issues") if isinstance(report.get("issues"), list) else []
        blocked_contracts_with_issue: set[str] = set()
        for report_issue in report_issues:
            if not isinstance(report_issue, dict):
                continue
            reason = str(report_issue.get("reason") or "final_pixel_contract_blocked").strip()
            contract = str(report_issue.get("contract") or "").strip() or None
            if contract:
                blocked_contracts_with_issue.add(contract)
            component_ids = [
                str(value) for value in report_issue.get("component_ids") or [] if str(value).strip()
            ]
            offenders = [
                str(value) for value in report_issue.get("offenders") or [] if str(value).strip()
            ]
            issues.append(
                _final_pixel_blocker(
                    page_id=page_id,
                    page_number=page_number,
                    reason=reason,
                    owner_id=str(report_issue.get("owner_id") or "").strip() or None,
                    component_ids=component_ids,
                    offenders=offenders,
                    issue_id=str(report_issue.get("issue_id") or "").strip() or None,
                    contract=contract,
                )
            )
        for contract in sorted(FINAL_PIXEL_CONTRACTS):
            status = str(contracts.get(contract) or "").strip().upper()
            if status not in {"PASS", "BLOCK"}:
                issues.append(
                    _final_pixel_blocker(
                        page_id=page_id,
                        page_number=page_number,
                        reason="final_pixel_contract_status_invalid",
                        contract=contract,
                    )
                )
            elif status == "BLOCK" and contract not in blocked_contracts_with_issue:
                issues.append(
                    _final_pixel_blocker(
                        page_id=page_id,
                        page_number=page_number,
                        reason="final_pixel_contract_blocked",
                        contract=contract,
                    )
                )

    for extra_page_id in sorted(set(reports_by_page) - expected_page_ids):
        issues.append(
            _final_pixel_blocker(
                page_id=extra_page_id,
                page_number=None,
                reason="final_pixel_report_unexpected_page",
            )
        )
    return issues


def _collect_owner_functional_contract_issues(
    project: dict[str, Any],
) -> list[dict[str, Any]]:
    if str(project.get("owner_graph_status") or "").strip().lower() != "verified":
        return []
    issues: list[dict[str, Any]] = []
    for page_index, page in enumerate(project.get("paginas") or [], start=1):
        if not isinstance(page, dict):
            continue
        page_number = int(page.get("numero") or page_index)
        page_id = str(page.get("page_id") or f"page_{page_number:03d}")
        for layer in page.get("text_layers") or page.get("textos") or []:
            if not isinstance(layer, dict):
                continue
            owner_id = str(layer.get("owner_id") or "").strip()
            if not owner_id or layer.get("render_completed") is not True:
                continue
            component_ids = [str(value) for value in layer.get("component_ids") or []]

            def add(reason: str, contract: str, offenders: list[str] | None = None) -> None:
                issues.append(
                    _final_pixel_blocker(
                        page_id=page_id,
                        page_number=page_number,
                        reason=reason,
                        owner_id=owner_id,
                        component_ids=component_ids,
                        offenders=offenders or [],
                        contract=contract,
                    )
                )

            quality = layer.get("owner_render_quality")
            if not isinstance(quality, dict):
                render_contract = layer.get("render_layout_contract")
                quality = (
                    render_contract.get("owner_render_quality")
                    if isinstance(render_contract, dict)
                    else None
                )
            if not isinstance(quality, dict):
                add(
                    "missing_owner_render_quality_contract",
                    "layout_legibility_contract",
                )
            else:
                status = str(quality.get("status") or "").strip()
                if status != "ok":
                    add(status or "invalid_owner_render_quality_contract", "layout_legibility_contract")
                if int(quality.get("outside_safe_pixels", 0) or 0) > 0:
                    add(
                        "core_pixels_outside_safe_polygon",
                        "layout_legibility_contract",
                        [f"outside_safe_pixels:{quality.get('outside_safe_pixels')}"],
                    )
                if not quality.get("rendered_line_core_heights_px"):
                    add("missing_rendered_line_core_metrics", "layout_legibility_contract")
                try:
                    source_scale_ratio = float(quality.get("source_scale_ratio"))
                except (TypeError, ValueError):
                    source_scale_ratio = None
                if (
                    source_scale_ratio is not None
                    and source_scale_ratio < 0.75
                    and status != "under_source_scale"
                ):
                    add("under_source_scale", "layout_legibility_contract")
                try:
                    x_height_ratio = float(quality.get("x_height_ratio"))
                except (TypeError, ValueError):
                    x_height_ratio = None
                if x_height_ratio is not None and x_height_ratio < 0.75:
                    add("under_source_x_height", "layout_legibility_contract")
            if str(layer.get("fit_status") or "") == "below_proportional_legibility":
                add("below_proportional_legibility", "layout_legibility_contract")

            if str(layer.get("route_action") or "") in {
                "translate_inpaint_render",
                "translate_sfx_inpaint_render",
            }:
                residual = layer.get("residual_cleanup_contract")
                if not isinstance(residual, dict) or residual.get("residual_verified") is not True:
                    add("unverified_owner_residual", "residual_cleanup_contract")
                else:
                    try:
                        score = float(residual.get("residual_score"))
                        threshold = float(residual.get("residual_threshold"))
                    except (TypeError, ValueError):
                        add("invalid_owner_residual_contract", "residual_cleanup_contract")
                    else:
                        if score > threshold:
                            add("owner_residual_above_threshold", "residual_cleanup_contract")
                protected = layer.get("protected_art_contract")
                if not isinstance(protected, dict):
                    add("missing_protected_art_contract", "protected_art_contract")
                elif (
                    int(protected.get("protected_art_changed_pixels", 0) or 0) > 0
                    or int(protected.get("action_protected_overlap_pixels", 0) or 0) > 0
                ):
                    add("protected_art_contract_violation", "protected_art_contract")
    return issues


def _page_id_from_identity(identity: str) -> str | None:
    match = re.search(r"(page_\d{3})_band_\d{3}", identity)
    return match.group(1) if match else None


def _band_id_from_identity(identity: str) -> str | None:
    match = re.search(r"(page_\d{3}_band_\d{3})", identity)
    return match.group(1) if match else None


def _trace_id_from_identity(identity: str) -> str | None:
    value = str(identity or "").strip()
    return value if "@" in value else None


def _text_id_from_identity(identity: str) -> str | None:
    value = str(identity or "").strip()
    if "@" in value:
        return value.split("@", 1)[0] or None
    return value or None


def _clean_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _layer_text_blob(layer: dict[str, Any]) -> str:
    return " ".join(
        str(layer.get(key) or "")
        for key in (
            "text",
            "original",
            "raw_ocr",
            "normalized_ocr",
            "normalized_text_final",
            "translated",
            "traduzido",
        )
    )


def _is_strong_scanlation_credit_layer(layer: dict[str, Any]) -> bool:
    blob = _layer_text_blob(layer)
    if not blob.strip():
        return False
    matches = [str(match or "").upper().strip() for match in SCANLATION_CREDIT_RE.findall(blob)]
    if layer.get("_scanlation_credit_context") is True and any(
        match in SCANLATION_TIER_WORDS for match in matches
    ):
        return True
    if layer.get("_scanlation_credit_context") is True and SCANLATION_CONTEXTUAL_RE.search(blob):
        return True
    if any(match and match not in SCANLATION_TIER_WORDS for match in matches):
        return True
    compact = re.sub(r"[^A-Z0-9]+", "", blob.upper())
    return bool(
        "RESETSCAN" in compact
        or "SCANSHOMEMANGA" in compact
        or "HIVETOON" in compact
        or "HIVESCAN" in compact
        or "PATREONCOM" in compact
        or "DISCORDCOM" in compact
        or "PAYPALCOM" in compact
    )


def _is_suppressed_scanlation_credit_layer(layer: dict[str, Any]) -> bool:
    reason = str(layer.get("skip_reason") or layer.get("route_reason") or "").strip().lower()
    if reason == "scanlation_credit_suppressed":
        return True
    return any(
        str(flag or "").strip().lower() == "scanlation_credit_suppressed"
        for flag in layer.get("qa_flags") or []
    )


def _layer_is_export_gate_candidate(layer: dict[str, Any]) -> bool:
    if _is_suppressed_scanlation_credit_layer(layer):
        return False
    # A layer explicitly hidden by the automatic safety guard is not emitted
    # into the translated image.  Its diagnostic flags remain useful in the
    # project, but must not block exporting pixels that are absent by design.
    if layer.get("visible") is False:
        return False
    route_action = _clean_string(layer.get("route_action"))
    if route_action:
        return (
            route_action.startswith("translate_")
            or route_action in ROUTE_ACTIONS_REQUIRING_EXPORT_GATE
        )
    return bool(
        layer.get("qa_flags")
        or layer.get("text")
        or layer.get("original")
        or layer.get("raw_ocr")
        or layer.get("translated")
    )


def _sfx_inpaint_requires_review(layer: dict[str, Any]) -> bool:
    if _clean_string(layer.get("route_action")) != "translate_sfx_inpaint_render":
        return False
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    return sfx.get("inpaint_allowed") is False


def _sfx_review_flags(layer: dict[str, Any]) -> list[str]:
    sfx = layer.get("sfx") if isinstance(layer.get("sfx"), dict) else {}
    flags = [str(flag) for flag in sfx.get("qa_flags") or [] if flag]
    if "sfx_inpaint_review_required" not in flags:
        flags.insert(0, "sfx_inpaint_review_required")
    return list(dict.fromkeys(flags))


def _first_list_string(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for item in value:
        text = _clean_string(item)
        if text:
            return text
    return None


def _resolve_trace_id(layer: dict[str, Any], text_id: str | None, band_id: str | None) -> str | None:
    for value in (
        layer.get("trace_id"),
        layer.get("text_instance_id"),
        _first_list_string(layer.get("source_trace_ids")),
        _first_list_string(layer.get("_source_trace_ids")),
        _first_list_string(layer.get("trace_ids")),
    ):
        trace_id = _clean_string(value)
        if trace_id and "@" in trace_id:
            return trace_id
    if text_id and band_id:
        return f"{text_id}@{band_id}"
    return None


def _synthetic_band_id(page_id: str, layer_index: int) -> str:
    return f"{page_id}_layer_{layer_index:03d}"


def _stable_rel_path(value: Any) -> str | None:
    if not value:
        return None
    rel_path = str(value).strip().replace("\\", "/")
    if not rel_path or re.match(r"^[A-Za-z]:/", rel_path) or rel_path.startswith("/"):
        return None
    return rel_path


def _translated_page_ref(page: dict[str, Any]) -> str | None:
    for value in (
        page.get("arquivo_traduzido"),
        page.get("translated_path"),
        page.get("output_path"),
    ):
        rel_path = _stable_rel_path(value)
        if rel_path:
            return rel_path
    image_layers = page.get("image_layers")
    if isinstance(image_layers, dict):
        rendered = image_layers.get("rendered")
        if isinstance(rendered, dict):
            return _stable_rel_path(rendered.get("path"))
    return None


def _layer_qa_flags(layer: dict[str, Any]) -> set[str]:
    flags = {str(flag) for flag in layer.get("qa_flags") or [] if flag}
    top_level_render_flags = set(flags & {"TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"})
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    render_fit_flags = {str(flag) for flag in render_fit.get("flags") or [] if flag}
    render_fit_stale = _render_fit_evidence_is_stale(layer)
    if _final_render_text_flags_are_review_only(layer):
        flags.difference_update({"TEXT_CLIPPED", "TEXT_OVERFLOW", "render_outside_balloon"})
    if not render_fit_stale:
        flags.update(render_fit_flags)
    if _final_render_text_flags_are_review_only(layer) and top_level_render_flags:
        flags.difference_update(top_level_render_flags)
    flags.difference_update(IGNORED_LEGACY_FLAGS)
    if layer.get("ocr_repair_status") != "repair_failed":
        flags.discard("ocr_truncated_or_joined")
    return flags


def _bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _bbox_area(value: list[int] | None) -> int:
    if value is None:
        return 0
    return max(0, value[2] - value[0]) * max(0, value[3] - value[1])


def _bbox_contains(outer: list[int] | None, inner: list[int] | None, margin: int = 4) -> bool:
    if outer is None or inner is None:
        return False
    return bool(
        inner[0] >= outer[0] - margin
        and inner[1] >= outer[1] - margin
        and inner[2] <= outer[2] + margin
        and inner[3] <= outer[3] + margin
    )


def _containment_target_bboxes(layer: dict[str, Any]) -> list[list[int]]:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    dark_bubble_metrics = qa_metrics.get("image_dark_bubble_mask") if isinstance(qa_metrics.get("image_dark_bubble_mask"), dict) else {}
    dark_panel_metrics = qa_metrics.get("image_dark_panel_mask") if isinstance(qa_metrics.get("image_dark_panel_mask"), dict) else {}
    candidates = [
        _bbox4(layer.get("bubble_inner_bbox")),
        _bbox4(layer.get("bubble_mask_bbox")),
        _bbox4(dark_bubble_metrics.get("mask_bbox")),
        _bbox4(dark_panel_metrics.get("mask_bbox")),
        _bbox4(layer.get("target_bbox")),
        _bbox4(layer.get("capacity_bbox")),
        _bbox4(layer.get("layout_bbox")),
        _bbox4(layer.get("balloon_bbox")),
        _bbox4(layer.get("bbox")),
    ]
    seen: set[tuple[int, int, int, int]] = set()
    targets: list[list[int]] = []
    for bbox in candidates:
        if bbox is None:
            continue
        key = tuple(bbox)
        if key in seen:
            continue
        seen.add(key)
        targets.append(bbox)
    return targets

def _bbox_intersection_area(a: list[int] | None, b: list[int] | None) -> int:
    if a is None or b is None:
        return 0
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _bbox_center_distance(a: list[int] | None, b: list[int] | None) -> tuple[float, float]:
    if a is None or b is None:
        return (0.0, 0.0)
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return (abs(ax - bx), abs(ay - by))


def _render_fit_evidence_is_stale(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    if not render_fit:
        return False
    current_target = (
        _bbox4(layer.get("target_bbox"))
        or _bbox4(layer.get("balloon_bbox"))
        or _bbox4(layer.get("layout_bbox"))
        or _bbox4(layer.get("capacity_bbox"))
    )
    fit_target = _bbox4(render_fit.get("target_bbox")) or _bbox4(render_fit.get("balloon_bbox"))
    render_bbox = _bbox4(layer.get("render_bbox"))
    if current_target is None or fit_target is None or render_bbox is None:
        return False
    current_area = _bbox_area(current_target)
    fit_area = _bbox_area(fit_target)
    if current_area <= 0 or fit_area <= 0:
        return False
    return bool(
        _bbox_contains(current_target, render_bbox)
        and _bbox_contains(current_target, fit_target)
        and fit_area < int(current_area * 0.40)
    )


def _final_render_text_flags_are_review_only(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    target = (
        _bbox4(layer.get("balloon_bbox"))
        or _bbox4(render_fit.get("balloon_bbox"))
        or _bbox4(layer.get("target_bbox"))
        or _bbox4(render_fit.get("target_bbox"))
        or _bbox4(layer.get("capacity_bbox"))
        or _bbox4(layer.get("layout_bbox"))
        or _bbox4(layer.get("bbox"))
    )
    safe_bbox = _bbox4(layer.get("safe_text_box")) or _bbox4(render_fit.get("safe_text_box"))
    render_bbox = _bbox4(layer.get("render_bbox")) or _bbox4(render_fit.get("render_bbox"))
    if target is None or safe_bbox is None or render_bbox is None:
        return False
    attempts = [item for item in list(layer.get("fit_attempts") or []) if isinstance(item, dict)]
    fit_ok = str(layer.get("fit_status") or "").strip().lower() == "ok" or any(
        str(item.get("status") or "").strip().lower() == "ok" for item in attempts
    )
    if fit_ok and _bbox_contains(safe_bbox, render_bbox, margin=2):
        return True
    if _is_dark_bubble_or_panel_layer(layer) and _bbox_contains(safe_bbox, render_bbox, margin=2):
        return True
    if not _bbox_contains(target, render_bbox, margin=2):
        return False
    sx1, sy1, sx2, sy2 = safe_bbox
    rx1, ry1, rx2, ry2 = render_bbox
    safe_w = max(1, sx2 - sx1)
    safe_h = max(1, sy2 - sy1)
    overhang_px = max(0, sx1 - rx1, rx2 - sx2, sy1 - ry1, ry2 - sy2)
    return overhang_px <= max(8, int(round(min(safe_w, safe_h) * 0.04)))


def _is_dark_bubble_or_panel_layer(layer: dict[str, Any]) -> bool:
    profiles = {
        str(layer.get("layout_profile") or "").strip().lower(),
        str(layer.get("block_profile") or "").strip().lower(),
        str(layer.get("background_type") or "").strip().lower(),
    }
    source = str(layer.get("bubble_mask_source") or layer.get("balloon_mask_source") or "").strip().lower()
    if source in {"image_dark_bubble_mask", "image_dark_panel_mask", "derived_card_panel_mask"}:
        return True
    return bool(profiles & {"dark_bubble", "dark_panel", "colored_status_panel"})


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_real_lobe_assignment_evidence(layer: dict[str, Any]) -> bool:
    confidence = _float_or_none(layer.get("lobe_assignment_confidence"))
    if confidence is not None:
        return confidence < 0.6
    return bool(_clean_string(layer.get("connected_balloon_id")) or _clean_string(layer.get("lobe_id")))


def _warning_flags_blocking_export(flags: set[str], layer: dict[str, Any]) -> set[str]:
    if _scanlation_credit_flags_are_review_only(flags, layer):
        return set()
    blocking = set(flags & EXPORT_BLOCKING_REVIEW_FLAGS)
    if "mask_outside_balloon" in flags and _render_balloon_containment_is_low(layer):
        blocking.add("mask_outside_balloon")
    return blocking


def _layer_has_confirmed_visual_damage(flags: set[str]) -> bool:
    return bool(flags & CONFIRMED_VISUAL_DAMAGE_FLAGS)


def _scanlation_credit_flags_are_review_only(flags: set[str], layer: dict[str, Any]) -> bool:
    if not flags:
        return False
    if not _is_strong_scanlation_credit_layer(layer):
        return False
    decision_flags = flags - SCANLATION_HARMLESS_CONTEXT_FLAGS
    if not decision_flags:
        return True
    return decision_flags.issubset(SCANLATION_VISUAL_REVIEW_ONLY_FLAGS)


def _render_geometry_contained(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    if render_bbox is None:
        return False
    safe_bbox = _bbox4(layer.get("safe_text_box"))
    for target in _containment_target_bboxes(layer):
        if not _bbox_contains(target, render_bbox, margin=4):
            continue
        if safe_bbox is not None and not _bbox_contains(target, safe_bbox, margin=6):
            continue
        return True
    return False


def _render_inside_safe_box(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    safe_bbox = _bbox4(layer.get("safe_text_box"))
    return bool(render_bbox is not None and safe_bbox is not None and _bbox_contains(safe_bbox, render_bbox, margin=2))


def _rendered_background_is_white_balloon(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    if bool(qa_metrics.get("render_flat_balloon_background")):
        return True
    luma = _float_or_none(qa_metrics.get("render_balloon_background_luma"))
    std = _float_or_none(qa_metrics.get("render_balloon_background_luma_std"))
    return bool(luma is not None and luma >= 245.0 and (std is None or std <= 24.0))


def _traceability_missing_entry_is_review_only(missing: dict[str, Any]) -> bool:
    if bool(missing.get("is_review_only")):
        return True
    flag = str(missing.get("flag") or "").strip()
    if flag != "fast_fill_no_glyph_evidence":
        return False
    source = str(missing.get("source") or "").strip()
    return source in {"render_plan", "inpaint_decision", "mask_decision"}


def _render_balloon_containment_is_low(layer: dict[str, Any], *, threshold: float = 0.90) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    validated_containment = _float_or_none(qa_metrics.get("render_validated_containment"))
    if validated_containment is not None and validated_containment >= threshold and _render_geometry_contained(layer):
        return False
    if _render_geometry_contained(layer):
        return False
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    if containment is not None:
        return containment < threshold
    target = _bbox4(layer.get("balloon_bbox")) or _bbox4(layer.get("target_bbox"))
    render_bbox = _bbox4(layer.get("render_bbox"))
    if target is None or render_bbox is None:
        return False
    render_area = _bbox_area(render_bbox)
    if render_area <= 0:
        return False
    containment_ratio = _bbox_intersection_area(target, render_bbox) / float(render_area)
    return containment_ratio < threshold


def _render_validated_containment_is_good(layer: dict[str, Any], *, threshold: float = 0.92) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    validated_containment = _float_or_none(qa_metrics.get("render_validated_containment"))
    return bool(validated_containment is not None and validated_containment >= threshold)


def _source_and_render_are_displaced(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    source_bbox = _best_source_alignment_bbox(layer, render_bbox)
    if source_bbox is None or render_bbox is None:
        return False
    for target in (
        _bbox4(layer.get("bubble_mask_bbox")),
        _bbox4(layer.get("bubble_inner_bbox")),
    ):
        if _bbox_contains(target, source_bbox, margin=8) and _bbox_contains(target, render_bbox, margin=8):
            return False
    source_area = _bbox_area(source_bbox)
    render_area = _bbox_area(render_bbox)
    if source_area <= 0 or render_area <= 0:
        return False
    overlap = _bbox_intersection_area(source_bbox, render_bbox)
    if overlap >= int(min(source_area, render_area) * 0.20):
        return False
    dx, dy = _bbox_center_distance(source_bbox, render_bbox)
    source_w = max(1, source_bbox[2] - source_bbox[0])
    source_h = max(1, source_bbox[3] - source_bbox[1])
    render_w = max(1, render_bbox[2] - render_bbox[0])
    render_h = max(1, render_bbox[3] - render_bbox[1])
    return bool(
        dx > max(24.0, min(source_w, render_w) * 0.45)
        or dy > max(18.0, min(source_h, render_h) * 0.75)
    )


def _best_source_alignment_bbox(layer: dict[str, Any], render_bbox: list[int] | None) -> list[int] | None:
    candidates = [
        _bbox4(layer.get("source_bbox")),
        _bbox4(layer.get("layout_bbox")),
        _bbox4(layer.get("bbox")),
        _bbox4(layer.get("text_pixel_bbox")),
    ]
    candidates = [bbox for bbox in candidates if bbox is not None]
    if not candidates:
        return None
    if render_bbox is None:
        return candidates[0]
    best = max(candidates, key=lambda bbox: _bbox_intersection_area(bbox, render_bbox))
    if _bbox_intersection_area(best, render_bbox) > 0:
        return best
    return candidates[0]


def _is_microtext_layer(layer: dict[str, Any]) -> bool:
    candidates = [
        _bbox4(layer.get("text_pixel_bbox")),
        _bbox4(layer.get("source_bbox")),
        _bbox4(layer.get("bbox")),
        _bbox4(layer.get("layout_bbox")),
    ]
    heights = [bbox[3] - bbox[1] for bbox in candidates if bbox is not None]
    if not heights or min(heights) > 34:
        return False
    target = _bbox4(layer.get("balloon_bbox")) or _bbox4(layer.get("target_bbox"))
    if target is None:
        return True
    target_h = target[3] - target[1]
    target_area = _bbox_area(target)
    return bool(target_h <= 80 or target_area <= 20000)


def _microtext_render_is_upscaled(layer: dict[str, Any]) -> bool:
    render_bbox = _bbox4(layer.get("render_bbox"))
    if render_bbox is None:
        return False
    candidates = [
        _bbox4(layer.get("text_pixel_bbox")),
        _bbox4(layer.get("bbox")),
        _bbox4(layer.get("layout_bbox")),
    ]
    small_heights = [bbox[3] - bbox[1] for bbox in candidates if bbox is not None]
    if not small_heights:
        return False
    source_h = min(small_heights)
    if source_h > 16:
        return False
    render_h = render_bbox[3] - render_bbox[1]
    return render_h > max(18, source_h * 3)


def _render_replaces_source_text_area(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    render_bbox = _bbox4(layer.get("render_bbox")) or _bbox4(render_fit.get("render_bbox"))
    if render_bbox is None:
        return False
    for source_bbox in (
        _bbox4(layer.get("text_pixel_bbox")),
        _bbox4(layer.get("source_bbox")),
        _bbox4(layer.get("layout_bbox")),
        _bbox4(layer.get("bbox")),
    ):
        if source_bbox is None:
            continue
        if _bbox_contains(source_bbox, render_bbox, margin=16):
            return True
        source_area = _bbox_area(source_bbox)
        render_area = _bbox_area(render_bbox)
        if source_area <= 0 or render_area <= 0:
            continue
        overlap = _bbox_intersection_area(source_bbox, render_bbox)
        if overlap >= int(min(source_area, render_area) * 0.55):
            return True
    return False


def _dark_layer_render_is_contained(layer: dict[str, Any]) -> bool:
    if not _is_dark_bubble_or_panel_layer(layer):
        return False
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    render_fit = qa_metrics.get("render_fit") if isinstance(qa_metrics.get("render_fit"), dict) else {}
    render_bbox = _bbox4(layer.get("render_bbox")) or _bbox4(render_fit.get("render_bbox"))
    safe_bbox = _bbox4(layer.get("safe_text_box")) or _bbox4(render_fit.get("safe_text_box"))
    if render_bbox is None:
        return False
    if safe_bbox is not None and not _bbox_contains(safe_bbox, render_bbox, margin=3):
        return False
    for target in _containment_target_bboxes(layer):
        if _bbox_contains(target, render_bbox, margin=8):
            return True
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    return bool(containment is not None and containment >= 0.90)


def _critical_flag_can_be_review_only(flag: str, flags: set[str], layer: dict[str, Any]) -> bool:
    if _scanlation_credit_flags_are_review_only(flags, layer):
        return True
    if flag == "text_residual_after_inpaint" and _contained_dark_residual_is_review_only(layer):
        return True
    if _layer_has_confirmed_visual_damage(flags):
        return False
    if flag == "mask_outside_balloon_critical":
        if _source_and_render_are_displaced(layer):
            return False
        return _render_geometry_contained(layer)
    if flag == "missing_real_bubble_mask":
        # A rejected derived mask is not enough to prove inpaint damage after
        # the final render.  Preserve it for review, but do not block export
        # when the final text is demonstrably inside both its safe box and the
        # detected balloon, and no other confirmed visual-damage flag exists.
        source = str(layer.get("bubble_mask_source") or "").strip().lower()
        return source in {"rejected_derived_bubble_mask", "derived_white_crop_rejected"} and _render_inside_safe_box(layer) and _render_geometry_contained(layer)
    if flag == "fit_below_minimum_legible":
        return _translator_note_fit_is_review_only(layer)
    if flag == "bbox_overreach_critical":
        if _microtext_render_is_upscaled(layer):
            return False
        return _is_microtext_layer(layer) and _render_geometry_contained(layer)
    if flag == "render_on_art_suspected":
        if _dark_layer_render_is_contained(layer):
            return True
        return _render_replaces_source_text_area(layer)
    if flag == "fast_fill_no_glyph_evidence":
        if _dark_layer_render_is_contained(layer):
            return True
        if layer.get("_render_metadata_group_sibling_geometry") and _render_inside_safe_box(layer):
            return True
        return _render_replaces_source_text_area(layer)
    return False


def _translator_note_fit_is_review_only(layer: dict[str, Any]) -> bool:
    text = str(layer.get("translated") or layer.get("text") or "").strip().lower()
    if not (text.startswith("t/n:") or text.startswith("tn:") or text.startswith("n/t:")):
        return False
    try:
        final_font_px = int(layer.get("font_size_final", 0) or 0)
        minimum_font_px = int(layer.get("minimum_legible_font_px", 0) or 0)
    except (TypeError, ValueError):
        return False
    if minimum_font_px <= 0 or final_font_px < minimum_font_px:
        return False
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    bg_luma = _float_or_none(qa_metrics.get("render_background_luma"))
    bg_std = _float_or_none(qa_metrics.get("render_background_luma_std"))
    flat_bg = bool(qa_metrics.get("render_flat_balloon_background"))
    if containment is not None and containment < 0.96:
        return False
    if bg_luma is not None and bg_luma < 235.0:
        return False
    if bg_std is not None and bg_std > 8.0:
        return False
    return flat_bg or bg_luma is not None


def _contained_dark_residual_is_review_only(layer: dict[str, Any]) -> bool:
    qa_metrics = layer.get("qa_metrics") if isinstance(layer.get("qa_metrics"), dict) else {}
    luma = _float_or_none(qa_metrics.get("render_background_luma"))
    luma_std = _float_or_none(qa_metrics.get("render_background_luma_std"))
    containment = _float_or_none(qa_metrics.get("render_balloon_containment"))
    if luma is None or luma > 96.0:
        return False
    if luma_std is not None and luma_std > 24.0:
        return False
    if containment is not None and containment < 0.90:
        return False
    return bool(_render_inside_safe_box(layer) or _render_geometry_contained(layer))


def _artifact_links_for_issue(
    flags: set[str],
    *,
    page: dict[str, Any],
    page_id: str | None,
    band_id: str | None,
    trace_id: str | None,
) -> list[str]:
    links: list[str] = ["11_qa_export_gate/qa_issues.jsonl"]
    render_flags = {
        "TEXT_CLIPPED",
        "TEXT_OVERFLOW",
        "render_outside_balloon",
        "render_outside_bubble_mask",
        "render_bbox_far_from_target_bbox",
        "render_on_art_suspected",
        "page_space_rerender_mixed_coordinates",
    }
    geometry_flags = {
        "bbox_overreach_critical",
        "layout_bbox_coordinate_mismatch",
        "bubble_inner_bbox_coordinate_mismatch",
        "source_bbox_assigned_from_balloon",
        "safe_text_box_recomputed",
        "balloon_bbox_collapsed_to_text",
        "balloon_bbox_missing",
    }
    mask_flags = {
        "mask_outside_balloon_critical",
        "source_glyph_area_ratio_critical",
        "mask_outside_balloon",
        "bbox_fallback_bubble_mask",
        "glyph_mask_outside_bubble",
        "missing_real_bubble_mask",
        "fast_fill_insufficient_coverage",
        "fast_fill_unverified_residual",
        "low_inpaint_coverage",
    }
    residual_flags = {
        "weak_text_residual_after_inpaint",
        "text_residual_after_inpaint",
        "text_residual_after_inpaint_confirmed",
        "text_residual_after_inpaint_suspected",
        "fast_fill_unverified_residual",
        "fast_fill_insufficient_coverage",
        "inpaint_texture_flattened",
    }
    translation_flags = {
        "vlm_failure_phrase",
        "translation_fallback_phrase",
        "glossary_violation",
        "forbidden_translation",
        "placeholder_lost",
        "unrestored_placeholder",
        "entity_mistranslated",
        "untranslated_english",
        "empty_translation",
        "mojibake_in_translation",
        "source_script_leak",
        "speech_cjk_preserved_inside_balloon",
    }

    if flags & render_flags:
        translated_ref = _translated_page_ref(page)
        if translated_ref:
            links.append(translated_ref)
        links.extend(
            [
                "09_typeset/render_plan_final.jsonl",
                "05_layout_geometry/layout_blocks.jsonl",
                f"12_contact_sheets/{band_id}.jpg",
            ]
        )

    if flags & geometry_flags:
        links.append("05_layout_geometry/layout_blocks.jsonl")
        if band_id:
            links.append(f"12_contact_sheets/{band_id}.jpg")

    if flags & mask_flags:
        links.append("06_mask_segmentation/mask_chain_summary.json")
        if band_id:
            links.append(f"06_mask_segmentation/{band_id}/mask_overlay.jpg")

    if flags & residual_flags:
        if band_id:
            links.extend(
                [
                    f"08_inpaint/{band_id}/03_inpaint_mask_overlay.jpg",
                    f"08_inpaint/{band_id}/inpaint_decision.json",
                    f"08_inpaint/{band_id}/06_band_after_inpaint.jpg",
                ]
            )

    if "inpaint_texture_flattened" in flags and band_id:
        links.extend(
            [
                f"08_inpaint/{band_id}/00_band_before_inpaint.jpg",
                f"08_inpaint/{band_id}/05_inpaint_mask_overlay.jpg",
                f"08_inpaint/{band_id}/06_band_after_inpaint.jpg",
                f"08_inpaint/{band_id}/inpaint_decision.json",
            ]
        )

    if flags & translation_flags:
        links.extend(
            [
                "07_translation/translation_inputs.jsonl",
                "07_translation/translation_outputs.jsonl",
            ]
        )

    if trace_id:
        links.append("11_qa_export_gate/export_gate.json")

    existing = set()
    deduped: list[str] = []
    for rel_path in links:
        stable = _stable_rel_path(rel_path)
        if stable and stable not in existing:
            deduped.append(stable)
            existing.add(stable)
    return deduped


def _terminal_owner_gate(project: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate terminal source-language evidence by exact proof-to-probe linkage."""

    source_manifest = project.get("chapter_source_manifest")
    if not isinstance(source_manifest, dict):
        return [], {}
    expected_pages = source_manifest.get("pages")
    pages = project.get("paginas")
    if not isinstance(expected_pages, list) or not isinstance(pages, list):
        return [_terminal_integrity_issue("source_manifest_page_set_mismatch")], {
            "english_dialogue_residual_count": 0
        }
    expected_identity = [
        (str(item.get("page_id") or ""), str(item.get("page_source_sha256") or ""))
        for item in expected_pages if isinstance(item, dict)
    ]
    actual_identity = [
        (str(item.get("page_id") or ""), str(item.get("page_source_sha256") or ""))
        for item in pages if isinstance(item, dict)
    ]
    gate_issues: list[dict[str, Any]] = []
    if expected_identity != actual_identity or len(actual_identity) != len(pages):
        gate_issues.append(_terminal_integrity_issue("source_manifest_page_set_mismatch"))
        return gate_issues, {"english_dialogue_residual_count": 0}

    source_issue_count = 0
    source_kinds = {
        "source_language_visible",
        "mixed_language_overlay",
        "independently_detected_text_without_owner",
        "cleanup_incomplete",
    }
    for page in pages:
        result = page.get("owner_page_result") or page.get("page_execution_result")
        if not isinstance(result, dict):
            gate_issues.append(_terminal_integrity_issue("page_lifecycle_incomplete", page))
            continue
        if str(result.get("status") or "") != "final_verified":
            gate_issues.append(_terminal_integrity_issue("page_lifecycle_incomplete", page))
            continue
        proof = result.get("terminal_proof")
        probes = result.get("qa_probes")
        residuals = result.get("language_residual_issues")
        if not isinstance(proof, dict) or not isinstance(probes, list) or not isinstance(residuals, list):
            gate_issues.append(_terminal_integrity_issue("terminal_probe_integrity_error", page))
            continue
        probe_id = str(proof.get("final_qa_probe_id") or "")
        matches = [item for item in probes if isinstance(item, dict) and item.get("probe_id") == probe_id]
        if len(matches) != 1:
            gate_issues.append(_terminal_integrity_issue("terminal_probe_integrity_error", page))
            continue
        probe = matches[0]
        exact_fields = (
            ("ocr_invocation_id", "fresh_ocr_invocation_id"),
            ("root_input_pixel_sha256", "fresh_ocr_root_input_pixel_sha256"),
            ("fresh_ocr_attempt_ids", "fresh_ocr_attempt_ids"),
            ("fresh_ocr_attempt_chain_sha256", "fresh_ocr_attempt_chain_sha256"),
        )
        if any(probe.get(left) != proof.get(right) for left, right in exact_fields):
            gate_issues.append(_terminal_integrity_issue("terminal_probe_integrity_error", page))
            continue
        issue_by_id: dict[str, list[dict[str, Any]]] = {}
        for item in residuals:
            if isinstance(item, dict):
                issue_by_id.setdefault(str(item.get("issue_id") or ""), []).append(item)
        terminal_issue_ids = [str(value) for value in probe.get("issue_ids") or []]
        if len(terminal_issue_ids) != len(set(terminal_issue_ids)) or any(
            len(issue_by_id.get(issue_id, ())) != 1 for issue_id in terminal_issue_ids
        ):
            gate_issues.append(_terminal_integrity_issue("terminal_probe_integrity_error", page))
            continue
        terminal_issues = [issue_by_id[issue_id][0] for issue_id in terminal_issue_ids]
        visible = [item for item in terminal_issues if str(item.get("kind") or "") in source_kinds]
        source_issue_count += len(visible)
        for item in visible:
            gate_issues.append({
                "page_id": page.get("page_id"),
                "type": "terminal_source_language_residual",
                "reason": "terminal_source_language_residual",
                "severity": "critical",
                "blocks_export": True,
                "flags": [str(item.get("kind") or "source_language_visible")],
                "issue_id": item.get("issue_id"),
                "owner_id": item.get("owner_id"),
            })
    return gate_issues, {"english_dialogue_residual_count": source_issue_count}


def _terminal_integrity_issue(reason: str, page: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "page_id": (page or {}).get("page_id"),
        "type": "owner_terminal_integrity",
        "reason": reason,
        "severity": "critical",
        "blocks_export": True,
        "flags": [reason],
    }


def evaluate_export_gate(project: dict[str, Any], *, override: bool = False) -> dict[str, Any]:
    issues = collect_export_blocking_issues(project)
    terminal_issues, terminal_metrics = _terminal_owner_gate(project)
    issues.extend(terminal_issues)
    review_issues = [issue for issue in issues if issue.get("severity") == "warning"]
    blocking_issues = [
        issue
        for issue in issues
        if issue.get("severity") == "critical" or bool(issue.get("blocks_export"))
    ]
    from qa.gate_composition import normalize_export_gate

    result = normalize_export_gate(
        issues,
        blocked=bool(blocking_issues),
        review=any(issue.get("type") == "sfx_inpaint_review" for issue in review_issues),
        override=override,
    )
    result.update(terminal_metrics)
    return result


def append_qa_integrity_failure(
    export_gate: dict[str, Any], failures: list[str]
) -> dict[str, Any]:
    """Add one fail-closed row and make all exported gate counts self-consistent."""
    failures = list(dict.fromkeys(str(value) for value in failures if str(value).strip()))
    if not failures:
        return export_gate
    issues = [item for item in export_gate.get("issues") or [] if isinstance(item, dict)]
    if not any(issue.get("reason") == "qa_integrity_failure" for issue in issues):
        issues.append(
            {
                "page": None,
                "page_id": "run",
                "owner_id": None,
                "component_ids": [],
                "trace_id": "run:qa_integrity:failure",
                "coordinate_space": "page",
                "type": "qa_integrity_failure",
                "issue_scope": "run",
                "severity": "critical",
                "blocks_export": True,
                "source": "owner_artifacts",
                "reason": "qa_integrity_failure",
                "flags": ["qa_integrity_failure"],
                "offenders": failures,
                "artifact_links": [
                    "11_qa_export_gate/owner_invariant_report.json",
                    "11_qa_export_gate/qa_export_gate_consistency.json",
                ],
            }
        )
    critical = [item for item in issues if item.get("severity") == "critical"]
    review = [item for item in issues if item.get("severity") == "warning"]
    blocking = [
        item
        for item in issues
        if item.get("severity") == "critical" or bool(item.get("blocks_export"))
    ]
    export_gate.update(
        {
            "status": "BLOCK",
            "allowed": False,
            "issue_count": len(issues),
            "blocking_issue_count": len(blocking),
            "blocking_flag_count": sum(len(item.get("flags") or []) for item in blocking),
            "critical_issue_count": len(critical),
            "critical_flag_count": sum(len(item.get("flags") or []) for item in critical),
            "review_issue_count": len(review),
            "review_flag_count": sum(len(item.get("flags") or []) for item in review),
            "needs_review": bool(review),
            "issues": issues,
        }
    )
    return export_gate


def _collect_final_visual_contract_issues(project: dict[str, Any], existing_issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    contract = qa.get("post_rerender_final_visual_contract") if isinstance(qa, dict) else None
    visual_qa = contract.get("qa") if isinstance(contract, dict) and isinstance(contract.get("qa"), dict) else {}
    rows = visual_qa.get("rows") if isinstance(visual_qa, dict) else []
    if not isinstance(rows, list):
        return []

    layers_by_trace: dict[str, dict[str, Any]] = {}
    for page in project.get("paginas") or []:
        for layer in page.get("text_layers") or page.get("textos") or []:
            if not isinstance(layer, dict):
                continue
            trace_id = _clean_string(layer.get("trace_id") or layer.get("text_instance_id"))
            if trace_id:
                layers_by_trace[trace_id] = layer

    generated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("status") or "").strip().lower() != "fail":
            continue
        band_id = _clean_string(row.get("band_id")) or "unresolved"
        trace_ids = list(dict.fromkeys(
            str(value).strip() for value in row.get("trace_ids") or [] if str(value).strip()
        ))
        flags = list(dict.fromkeys(str(flag).strip() for flag in row.get("flags") or [] if str(flag).strip()))
        if not flags:
            continue
        already_reported = {
            str(flag)
            for issue in [*existing_issues, *generated]
            if str(issue.get("band_id") or "") == band_id
            and (
                not trace_ids
                or str(issue.get("trace_id") or "") in trace_ids
                or bool(set(issue.get("trace_ids") or []) & set(trace_ids))
            )
            for flag in issue.get("flags") or []
        }
        flags = [flag for flag in flags if flag not in already_reported]
        if not flags:
            continue
        page_match = re.search(r"page_(\d+)", band_id, re.IGNORECASE)
        page_number = int(page_match.group(1)) if page_match else None
        page_id = f"page_{page_number:03d}" if page_number is not None else "unresolved"
        matched_layers = [layers_by_trace[trace_id] for trace_id in trace_ids if trace_id in layers_by_trace]
        artifact_links = list(dict.fromkeys([
            "11_qa_export_gate/final_rerender_visual_qa.json",
            "11_qa_export_gate/final_rerender_visual_qa.jsonl",
            "10_copyback_reassemble/final_band_crops.jsonl",
            *[str(link) for link in row.get("artifact_links") or [] if str(link).strip()],
        ]))
        generated.append(
            {
                "page": page_number,
                "page_id": page_id,
                "band_id": band_id,
                "layer": (
                    matched_layers[0].get("id") or matched_layers[0].get("text_id")
                    if matched_layers else band_id
                ),
                "text_id": matched_layers[0].get("text_id") or matched_layers[0].get("id") if matched_layers else None,
                "trace_id": trace_ids[0] if trace_ids else None,
                "trace_ids": trace_ids,
                "coordinate_space": "page",
                "type": "p0_final_visual_blocker",
                "issue_scope": "band",
                "severity": "critical",
                "blocks_export": True,
                "source": "post_rerender_final_visual_contract",
                "flags": flags,
                "metrics": dict(row.get("metrics") or {}),
                "artifact_links": artifact_links,
                "linked_artifacts": artifact_links,
            }
        )
    return generated


def collect_export_blocking_issues(project: dict[str, Any]) -> list[dict[str, Any]]:
    source_lang = str(project.get("idioma_origem") or "").lower()
    cjk_source = source_lang in {"ja", "jp", "ko", "kr", "zh", "zh-cn", "zh-tw"}
    issues: list[dict[str, Any]] = []
    for page_index, page in enumerate(project.get("paginas") or [], start=1):
        layers = page.get("text_layers") or page.get("textos") or []
        page_number = int(page.get("numero") or page_index)
        page_id = str(page.get("page_id") or f"page_{page_number:03d}")
        scanlation_credit_bands = {
            (
                _clean_string(candidate.get("band_id"))
                or _band_id_from_identity(_clean_string(candidate.get("trace_id")) or "")
                or ""
            )
            for candidate in layers
            if isinstance(candidate, dict) and _is_strong_scanlation_credit_layer(candidate)
        }
        scanlation_credit_bands.discard("")
        page_has_scanlation_credit = any(
            isinstance(candidate, dict) and _is_strong_scanlation_credit_layer(candidate)
            for candidate in layers
        )
        for layer_index, layer in enumerate(layers, start=1):
            if not isinstance(layer, dict):
                continue
            if not _layer_is_export_gate_candidate(layer):
                continue
            text_id = str(layer.get("text_id") or layer.get("id") or f"t{layer_index}")
            raw_trace_id = _clean_string(layer.get("trace_id") or layer.get("text_instance_id"))
            resolved_page_id = (
                layer.get("page_id")
                or _page_id_from_identity(raw_trace_id or "")
                or page_id
            )
            band_id = (
                _clean_string(layer.get("band_id"))
                or _band_id_from_identity(raw_trace_id or "")
                or _synthetic_band_id(str(resolved_page_id), layer_index)
            )
            policy_layer = layer
            if (
                band_id in scanlation_credit_bands
                and not _is_strong_scanlation_credit_layer(layer)
                and SCANLATION_CREDIT_RE.search(_layer_text_blob(layer))
            ):
                policy_layer = {**layer, "_scanlation_credit_context": True}
            elif (
                page_has_scanlation_credit
                and not _is_strong_scanlation_credit_layer(layer)
                and SCANLATION_CONTEXTUAL_RE.search(_layer_text_blob(layer))
            ):
                policy_layer = {**layer, "_scanlation_credit_context": True}
            trace_id = _resolve_trace_id(layer, text_id, band_id)
            translated = str(layer.get("translated") or layer.get("traduzido") or "")
            flags = _layer_qa_flags(layer)
            if cjk_source and translated and SOURCE_SCRIPT_RE.search(translated):
                flags.add("speech_cjk_preserved_inside_balloon")
            critical_flags = {flag for flag in flags if severity_for_flag(flag) == "critical"}
            demoted_critical_flags = {
                flag
                for flag in critical_flags
                if _critical_flag_can_be_review_only(flag, flags, policy_layer)
            }
            critical_flags -= demoted_critical_flags
            warning_flags = {flag for flag in flags if severity_for_flag(flag) == "high"}
            warning_flags.update(demoted_critical_flags)
            base_issue = {
                "page": page_number,
                "page_id": resolved_page_id,
                "band_id": band_id,
                "layer": layer.get("id") or text_id,
                "text_id": text_id,
                "text_instance_id": layer.get("text_instance_id")
                or (f"{band_id}_{text_id}" if band_id else None),
                "trace_id": trace_id,
                "coordinate_space": layer.get("coordinate_space") or "page",
                "text": translated[:160],
                "bbox": layer.get("bbox") or layer.get("layout_bbox") or layer.get("source_bbox"),
                "source_bbox": layer.get("source_bbox") or layer.get("bbox"),
                "balloon_bbox": layer.get("balloon_bbox"),
                "safe_text_box": layer.get("safe_text_box") or layer.get("_debug_safe_text_box"),
                "render_bbox": layer.get("render_bbox"),
                "qa_metrics": dict(layer.get("qa_metrics") or {}),
            }
            if _sfx_inpaint_requires_review(layer):
                issues.append(
                    {
                        **base_issue,
                        "type": "sfx_inpaint_review",
                        "severity": "warning",
                        "flags": _sfx_review_flags(layer),
                        "blocks_export": False,
                    }
                )
            if critical_flags:
                artifact_links = _artifact_links_for_issue(
                    critical_flags,
                    page=page,
                    page_id=str(base_issue.get("page_id") or ""),
                    band_id=str(band_id or ""),
                    trace_id=str(trace_id or ""),
                )
                issues.append(
                    {
                        **base_issue,
                        "type": "p0_render_blocker",
                        "severity": "critical",
                        "flags": sorted(critical_flags),
                        "blocks_export": True,
                        **({"artifact_links": artifact_links} if artifact_links else {}),
                    }
                )
            if warning_flags:
                blocks_export = bool(_warning_flags_blocking_export(warning_flags, policy_layer))
                artifact_links = _artifact_links_for_issue(
                    warning_flags,
                    page=page,
                    page_id=str(base_issue.get("page_id") or ""),
                    band_id=str(band_id or ""),
                    trace_id=str(trace_id or ""),
                )
                issues.append(
                    {
                        **base_issue,
                        "type": "needs_review",
                        "severity": "warning",
                        "flags": sorted(warning_flags),
                        "blocks_export": blocks_export,
                        **({"artifact_links": artifact_links} if artifact_links else {}),
                    }
                )
    issues.extend(_collect_final_visual_contract_issues(project, issues))
    qa = project.get("qa") if isinstance(project.get("qa"), dict) else {}
    propagation_audit = qa.get("flag_propagation_audit") if isinstance(qa, dict) else None
    if isinstance(propagation_audit, dict):
        for missing in propagation_audit.get("missing_in_project") or []:
            if not isinstance(missing, dict):
                continue
            identity = str(missing.get("identity") or missing.get("text_id") or "").strip()
            trace_id = _trace_id_from_identity(identity)
            text_id = missing.get("text_id") or _text_id_from_identity(identity)
            page_id = _page_id_from_identity(identity) or "unresolved"
            band_id = _band_id_from_identity(identity) or "unresolved"
            artifact_links = [
                "11_qa_export_gate/qa_flag_propagation_audit.json",
                "11_qa_export_gate/qa_issues.jsonl",
            ]
            if _traceability_missing_entry_is_review_only(missing):
                issues.append(
                    {
                        "page": None,
                        "page_id": page_id,
                        "band_id": band_id,
                        "layer": identity or "unresolved",
                        "text_id": text_id or "unresolved",
                        "text_instance_id": identity or None,
                        "trace_id": trace_id,
                        "coordinate_space": "debug_identity",
                        "text": f"QA flag not propagated from {missing.get('source') or 'debug'}",
                        "type": "needs_review",
                        "issue_scope": "run",
                        "severity": "warning",
                        "flags": ["qa_flag_not_propagated"],
                        "missing_flag": missing.get("flag"),
                        "missing_identity": identity or None,
                        "source": missing.get("source"),
                        "artifact_links": artifact_links,
                        "linked_artifacts": artifact_links,
                        "blocks_export": False,
                    }
                )
                continue
            issues.append(
                {
                    "page": None,
                    "page_id": page_id,
                    "band_id": band_id,
                    "layer": identity or "unresolved",
                    "text_id": text_id or "unresolved",
                    "text_instance_id": identity or None,
                    "trace_id": trace_id,
                    "coordinate_space": "debug_identity",
                    "text": f"QA flag not propagated from {missing.get('source') or 'debug'}",
                    "type": "p0_traceability_blocker",
                    "issue_scope": "run",
                    "severity": "critical",
                    "flags": ["qa_flag_not_propagated"],
                    "missing_flag": missing.get("flag"),
                    "missing_identity": identity or None,
                    "source": missing.get("source"),
                    "artifact_links": artifact_links,
                    "linked_artifacts": artifact_links,
                }
            )
        for candidate in propagation_audit.get("unmatched_detect_candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            band_id = str(candidate.get("band_id") or _band_id_from_identity(candidate_id) or "unresolved")
            page_id = str(candidate.get("page_id") or _page_id_from_identity(candidate_id) or _page_id_from_identity(band_id) or "unresolved")
            artifact_links = [
                "02_strip_detect/detect_candidates.jsonl",
                "02_strip_detect/candidate_text_matching.jsonl",
                "11_qa_export_gate/qa_flag_propagation_audit.json",
                "11_qa_export_gate/qa_issues.jsonl",
            ]
            issues.append(
                {
                    "page": None,
                    "page_id": page_id,
                    "band_id": band_id,
                    "layer": candidate_id or "unresolved",
                    "text_id": "unresolved",
                    "text_instance_id": candidate_id or None,
                    "trace_id": None,
                    "coordinate_space": "debug_candidate",
                    "text": "Accepted detect candidate has no OCR text layer",
                    "type": "p0_traceability_blocker",
                    "issue_scope": "run",
                    "severity": "critical",
                    "flags": ["detect_candidate_without_ocr_text"],
                    "candidate_id": candidate_id or None,
                    "bbox": candidate.get("bbox_page"),
                    "bbox_strip": candidate.get("bbox_strip"),
                    "match_reason": candidate.get("match_reason"),
                    "artifact_links": artifact_links,
                    "linked_artifacts": artifact_links,
                }
            )
    issues.extend(_collect_final_pixel_report_issues(project))
    issues.extend(_collect_owner_functional_contract_issues(project))
    return issues
