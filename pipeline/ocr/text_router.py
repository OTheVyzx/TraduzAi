from __future__ import annotations

import re
from typing import Any

from debug_tools.detectors import has_sfx_marker

from .postprocess import split_sfx_inline


TN_PATTERN = re.compile(r"^\s*T\s*/\s*N\s*:", re.IGNORECASE)
SIGN_PATTERN = re.compile(r"^\s*TEXT\s*:\s*(.+)$", re.IGNORECASE)
URL_OR_HANDLE_PATTERN = re.compile(
    r"https?://"
    r"|www\."
    r"|\b[A-Za-z0-9._-]+\.(?:com|net|org|gg|io|co)\b"
    r"|@\w{2,}"
    r"|\bread\s+at\b"
    r"|\bdiscord\.gg\b",
    re.IGNORECASE,
)
SCANLATOR_ROLE_PATTERN = re.compile(r"\b(?:TL|PR|TS|CL|QC|RD|RAW|TYPESET|CLEAN|REDRAW)\b", re.IGNORECASE)
SCANLATOR_CREDIT_PATTERN = re.compile(
    r"\b(?:scanlator|scans?|translation|translator|proofreader|typesetter|cleaner|redrawer)\b",
    re.IGNORECASE,
)
DIALOGUE_WORD_PATTERN = re.compile(
    r"\b(?:i|you|he|she|we|they|me|my|your|our|this|that|there|what|why|how|when|where|"
    r"am|are|is|was|were|be|been|do|did|does|have|has|had|will|would|can|could|"
    r"should|just|not|no|yes|please|here|there|come|came|go|going|think|know|believe)\b",
    re.IGNORECASE,
)
HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")

ROUTE_ACTIONS = {
    "translate_inpaint_render",
    "translate_sfx_inpaint_render",
    "translate_render_only",
    "inpaint_only",
    "review_required",
}
TRANSLATE_ROUTE_ACTIONS = {"translate_inpaint_render", "translate_sfx_inpaint_render", "translate_render_only"}
RENDER_ROUTE_ACTIONS = {"translate_inpaint_render", "translate_sfx_inpaint_render", "translate_render_only"}
INPAINT_ROUTE_ACTIONS = {"translate_inpaint_render", "translate_sfx_inpaint_render", "inpaint_only"}


def route_action_requires_translation(route_action: str | None) -> bool:
    return str(route_action or "").strip().lower() in TRANSLATE_ROUTE_ACTIONS


def route_action_requires_render(route_action: str | None) -> bool:
    return str(route_action or "").strip().lower() in RENDER_ROUTE_ACTIONS


def route_action_requires_inpaint(route_action: str | None) -> bool:
    return str(route_action or "").strip().lower() in INPAINT_ROUTE_ACTIONS


def apply_route_action(result: dict[str, Any], *, route_action: str | None = None, route_reason: str | None = None) -> dict[str, Any]:
    action, reason = _resolve_route_action(result, route_action=route_action, route_reason=route_reason)
    result["route_action"] = action
    result["route_reason"] = reason
    result["skip_processing"] = False
    result.pop("preserve_original", None)
    return result


def _resolve_route_action(
    result: dict[str, Any],
    *,
    route_action: str | None = None,
    route_reason: str | None = None,
) -> tuple[str, str]:
    render_policy = str(result.get("render_policy") or "").strip().lower()
    translate_policy = str(result.get("translate_policy") or "").strip().lower()
    needs_review = bool(result.get("needs_review"))

    if translate_policy == "skip_translation" or render_policy in {"preserve", "preserve_original"}:
        return "review_required", _route_reason(result, route_reason, "review_required")
    explicit_action = str(route_action or result.get("route_action") or "").strip().lower()
    if explicit_action in ROUTE_ACTIONS:
        return explicit_action, _route_reason(result, route_reason, explicit_action)
    if needs_review:
        return "review_required", _route_reason(result, route_reason, "review_required")
    if render_policy == "render_in_sign_bbox":
        return "translate_render_only", _route_reason(result, route_reason or "sign_bbox_available", "translate_render_only")
    return "translate_inpaint_render", _route_reason(result, route_reason, "translate_inpaint_render")


def _route_reason(result: dict[str, Any], explicit: str | None, action: str) -> str:
    for value in (
        explicit,
        result.get("reason"),
        (result.get("rules_applied") or [None])[0] if isinstance(result.get("rules_applied"), list) else None,
    ):
        reason = str(value or "").strip()
        if reason:
            return reason
    if action == "translate_inpaint_render":
        return "dialogue_balloon_with_english_text"
    return action


def route_text(
    text: str,
    *,
    text_id: str | None = None,
    tipo: str = "texto",
    bbox: list[int] | tuple[int, int, int, int] | None = None,
    page_number: int | None = None,
    page_height: int | None = None,
    sign_bbox: list[int] | tuple[int, int, int, int] | None = None,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict[str, Any]:
    value = _normalize_spacing(str(text or ""))

    def finalize(result: dict[str, Any]) -> dict[str, Any]:
        return apply_route_action(result)

    base = {
        "text_id": text_id,
        "input": value,
        "rules_applied": [],
        "needs_review": False,
        "skip_processing": False,
    }

    if not value:
        return finalize({
            **base,
            "route": "text",
            "content_class": "text",
            "tipo": "text",
            "translate_policy": "translate",
            "render_policy": "normal",
            "reason": "empty_text",
        })

    if TN_PATTERN.match(value):
        return finalize({
            **base,
            "route": "text",
            "content_class": "text",
            "tipo": "text",
            "render_policy": "normal",
            "translate_policy": "translate",
            "needs_review": False,
            "reason": "translator_note_marker",
            "rules_applied": ["translator_note_marker"],
        })

    if URL_OR_HANDLE_PATTERN.search(value):
        return finalize({
            **base,
            "route": "text",
            "content_class": "text",
            "tipo": "text",
            "render_policy": "normal",
            "translate_policy": "translate",
            "rules_applied": ["url_or_handle_watermark"],
            "reason": "url_or_handle_watermark",
        })

    sign_match = SIGN_PATTERN.match(value)
    if sign_match:
        clean_sign = _normalize_spacing(sign_match.group(1) or value)
        resolved_sign_bbox = _valid_bbox(sign_bbox)
        if resolved_sign_bbox:
            return finalize({
                **base,
                "route": "text",
                "content_class": "text",
                "tipo": "text",
                "text": clean_sign,
                "sign_bbox": resolved_sign_bbox,
                "render_policy": "normal",
                "translate_policy": "translate",
                "rules_applied": ["text_marker_sign"],
            })
        return finalize({
            **base,
            "route": "text",
            "content_class": "text",
            "tipo": "text",
            "text": clean_sign,
            "render_policy": "normal",
            "translate_policy": "translate",
            "rules_applied": ["text_marker_sign"],
        })

    cover_route = _route_cover_region_text(
        value,
        bbox=bbox,
        page_number=page_number,
        page_height=page_height,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    if cover_route:
        return finalize({**base, **cover_route})

    if has_sfx_marker(value):
        speech_text, sfx_text = split_sfx_inline(value)
        speech_word_count = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+", str(speech_text or "")))
        if sfx_text and speech_word_count >= 2:
            return finalize({
                **base,
                "route": "speech_with_sfx_split",
                "content_class": "text",
                "tipo": "text",
                "parts": [
                    {
                        "class": "speech",
                        "text": speech_text,
                        "text_id_synthetic": f"{text_id}_a" if text_id else None,
                    },
                    {
                        "class": "sfx",
                        "text": sfx_text,
                        "text_id_synthetic": f"{text_id}_b" if text_id else None,
                    },
                ],
                "translate_policy": "translate_speech_only",
                "render_policy": "split_layers",
                "rules_applied": ["split_sfx_marker_inside_dialogue"],
                "needs_review": False,
            })

    if _looks_like_hangul_sfx(value, tipo):
        return finalize({
            **base,
            "route": "manhwa_sfx",
            "content_class": "sfx",
            "tipo": "sfx",
            "script": "hangul",
            "route_action": "translate_sfx_inpaint_render",
            "translate_policy": "adapt_sfx",
            "render_policy": "sfx_style",
            "rules_applied": ["hangul_sfx_candidate"],
            "reason": "hangul_sfx_candidate",
        })

    return finalize({
        **base,
        "route": "text",
        "content_class": "text",
        "tipo": "text",
        "translate_policy": "translate",
        "render_policy": "normal",
    })


def route_text_record(
    record: dict[str, Any],
    *,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict[str, Any]:
    result = dict(record)
    routed = route_text(
        str(record.get("text") or record.get("raw_ocr") or record.get("original") or ""),
        text_id=record.get("id") or record.get("text_id"),
        tipo=str(record.get("tipo") or "texto"),
        bbox=record.get("bbox"),
        page_number=record.get("page_number") or record.get("page"),
        page_height=record.get("page_height"),
        sign_bbox=record.get("sign_bbox"),
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    )
    result.update(routed)
    result["text"] = str(record.get("text") or record.get("raw_ocr") or record.get("original") or "")
    return result


def _route_from_tipo(tipo: str) -> str:
    return "text"


def _content_class_from_route(route: str) -> str:
    return "text"


def _tipo_from_route(route: str) -> str:
    return "text"


def _normalize_spacing(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_hangul_sfx(value: str, tipo: str = "") -> bool:
    if not HANGUL_RE.search(value):
        return False
    if str(tipo or "").strip().lower() in {"sfx", "sound_effect", "sound"}:
        return True
    return False


def _valid_bbox(value: list[int] | tuple[int, int, int, int] | None) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _is_cover_region(
    bbox: list[int] | tuple[int, int, int, int] | None,
    *,
    page_number: int | None,
    page_height: int | None,
) -> bool:
    box = _valid_bbox(bbox)
    if not box:
        return False
    try:
        page = int(page_number or 0)
        height = int(page_height or 0)
    except (TypeError, ValueError):
        return False
    if page != 1 or height <= 0:
        return False
    return box[1] <= int(height * 0.15)


def _route_cover_region_text(
    value: str,
    *,
    bbox: list[int] | tuple[int, int, int, int] | None,
    page_number: int | None,
    page_height: int | None,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> dict[str, Any] | None:
    if not _is_cover_region(bbox, page_number=page_number, page_height=page_height):
        return None
    if not _matches_user_provided_title(
        value,
        work_title=work_title,
        work_title_aliases=work_title_aliases,
        work_title_user_provided=work_title_user_provided,
    ):
        return None

    return {
        "route": "title_text",
        "content_class": "text",
        "tipo": "text",
        "render_policy": "normal",
        "translate_policy": "translate",
        "skip_processing": False,
        "needs_review": False,
        "rules_applied": ["user_title_cover_logo_match"],
        "reason": "user_title_cover_logo_match",
    }


def _matches_user_provided_title(
    value: str,
    *,
    work_title: str = "",
    work_title_aliases: list[str] | tuple[str, ...] | None = None,
    work_title_user_provided: bool = False,
) -> bool:
    if not work_title_user_provided:
        return False
    text_key = _title_match_key(value)
    if len(text_key) < 4:
        return False
    for candidate in [work_title, *(work_title_aliases or [])]:
        title_key = _title_match_key(str(candidate or ""))
        if len(title_key) >= 4 and (title_key == text_key or title_key in text_key or text_key in title_key):
            return True
    return False


def _title_match_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _looks_like_latin_dialogue_or_narration(value: str) -> bool:
    normalized = _normalize_spacing(value)
    if not normalized:
        return False
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", normalized):
        return False

    words = re.findall(r"[A-Za-z][A-Za-z']*", normalized)
    if len(words) < 3:
        return False

    letters = re.findall(r"[A-Za-z]", normalized)
    if len(letters) < 8:
        return False

    letter_ratio = len(letters) / max(1, len(re.sub(r"\s+", "", normalized)))
    if letter_ratio < 0.55:
        return False

    has_sentence_signal = bool(re.search(r"[.!?]$", normalized)) or any(
        ch.islower() for ch in normalized
    )
    if not has_sentence_signal:
        return False

    return bool(DIALOGUE_WORD_PATTERN.search(normalized))
