"""OCR normalization before translation."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

try:
    from ocr.text_router import ROUTE_ACTIONS, apply_route_action
    from ocr.postprocess import (
        is_ocr_truncated_or_joined,
        normalize_rotated_text_metadata,
        should_retain_low_confidence_dialogue_ocr,
    )
except ImportError:  # pragma: no cover - supports package imports
    from .text_router import ROUTE_ACTIONS, apply_route_action
    from .postprocess import (
        is_ocr_truncated_or_joined,
        normalize_rotated_text_metadata,
        should_retain_low_confidence_dialogue_ocr,
    )


MANDATORY_CORRECTIONS = {
    "RAID SOUAD": "RAID SQUAD",
    "DRCS": "ORCS",
    "RDC": "ORCS",
    "CARBAGE": "GARBAGE",
    "TRAE": "TRAP",
    "FENRISNOW": "FENRIS NOW",
}

TRUNCATED_JOINED_REPAIRS = {
    "WEDO": "WE DO",
    "ITTOUS": "IT TO US",
    "LYINGIL": "LYING. I'LL",
    "NOTECPR": "NOTE CPR",
}

SAME_BALLOON_JOINED_WORD_REPAIRS: list[tuple[str, str]] = [
    (r"\bBITCHIS\b", "BITCH IS"),
    (r"\bAREAL(?=\s+[A-Z][A-Z']{2,})", "A REAL"),
    (r"\bREDUCEDBY\b", "REDUCED BY"),
    (r"\bWHYBOTHER\b", "WHY BOTHER"),
    (r"\bAPRIVATE\b", "A PRIVATE"),
    (r"\bFRUSTRATINGTOO\b", "FRUSTRATING TOO"),
    (r"\bCANYOUFIND\b", "CAN YOU FIND"),
    (r"\bAGOOD\b", "A GOOD"),
    (r"\bTHREEMONTHS\b", "THREE MONTHS"),
    (r"\bTHREEMONTH'SWORTH\b", "THREE MONTH'S WORTH"),
    (r"\?YOUR\b", "? YOUR"),
    (r"\bTHISIS\b", "THIS IS"),
    (r"\bTHATIS\b", "THAT IS"),
    (r"\bITIS\b", "IT IS"),
    (r"\bHEIS\b", "HE IS"),
    (r"\bSHEIS\b", "SHE IS"),
    (r"\bTHEREIS\b", "THERE IS"),
    (r"\bWHATIS\b", "WHAT IS"),
    (r"\bWHOIS\b", "WHO IS"),
    (r"\bWHEREIS\b", "WHERE IS"),
    (r"\bWHYIS\b", "WHY IS"),
]

REMOVED_LEGACY_FILTER_FLAGS = {
    "low_confidence_visual_noise",
    "cover_title_logo",
    "mask_density_high",
}
REMOVED_LEGACY_ROUTE_REASONS = REMOVED_LEGACY_FILTER_FLAGS | {
    "duplicate_lower_confidence",
    "low_confidence_noise",
    "manual_preserve",
    "ocr_artifact",
    "ocr_gibberish",
    "preserve_original",
    "suspicious_low_confidence",
}

INLINE_MANDATORY_CORRECTIONS: list[tuple[str, str, int]] = [
    (r"\bDWAS\s+UNABLE\s+TO\s+HIRHSTAND\s+TRHE\s+SRIIGSMIANDRANAWAY\s+FROM\s+HOME\b", "WAS UNABLE TO WITHSTAND THE STIGMA AND RAN AWAY FROM HOME", re.IGNORECASE),
    (r"\bSURVIVED\s+COUNTLESS\s+BAUES\.\s+BUTUP\s+MY\s+SADLAND\s+MADE\s+A\s+NAME\s+FOR\s+MYSELF\b", "SURVIVED COUNTLESS BATTLES. BUILT UP MY SKILLS AND MADE A NAME FOR MYSELF", re.IGNORECASE),
    (r"\bALMDST\s+ALL\s+OF\s+US\s+ENDED\s+IP\s+DYNG\b", "ALMOST ALL OF US ENDED UP DYING", re.IGNORECASE),
    (r"\bGHISLAIN\s+PERDIUM,\s+THEMERCENARYKING,\s+AND\s+ONE\s+OF\s+THE\s+CONTINENT'S\s+SEVENSTRONGESTMEN\.", "GHISLAIN PERDIUM, THE MERCENARY KING, AND ONE OF THE CONTINENT'S SEVEN STRONGEST MEN.", re.IGNORECASE),
    (r"\bWELL,\s+THERE'SNOPOINTIN\s+EXPLAINING\s+ITFURTHER\b", "WELL, THERE'S NO POINT IN EXPLAINING IT FURTHER", re.IGNORECASE),
    (r"\bIT'S\s+JUSTA\s+SHAME\s+THAT\s+I\s+WAS\s+UNABLE\s+TOFULFILL\b", "IT'S JUST A SHAME THAT I WAS UNABLE TO FULFILL", re.IGNORECASE),
    (r"\bBACK\s+TTRAVELED\s+TO\s+THE\s+PAST\?", "HAVE I TRAVELED BACK TO THE PAST?", re.IGNORECASE),
    (r"\bRAID\s+SOUAD\b", "RAID SQUAD", re.IGNORECASE),
    (r"\bSOUAD\b", "SQUAD", re.IGNORECASE),
    (r"\bDRCS\b", "ORCS", re.IGNORECASE),
    (r"\bRDC\b", "ORCS", re.IGNORECASE),
    (r"\bDRC\b", "ORC", re.IGNORECASE),
    (r"\bCARBAGE\b", "GARBAGE", re.IGNORECASE),
    (r"\bTRA[Ee]\b", "TRAP", re.IGNORECASE),
    (r"\bOEFENSE\b", "DEFENSE", re.IGNORECASE),
    (r"\bOOWN\b", "DOWN", re.IGNORECASE),
    (r"\bKINGDOME\b", "KINGDOM", re.IGNORECASE),
    (r"\bAGOE\b", "AGO", re.IGNORECASE),
    (r"\bHOUSEHOID\b", "HOUSEHOLD", re.IGNORECASE),
    (r"\bHOUSEHOLDAUOIDEDME\b", "HOUSEHOLD AVOIDED ME", re.IGNORECASE),
    (r"\bREJDICE\b", "REJOICE", re.IGNORECASE),
    (r"(?<![A-Za-z0-9])%NIGHT\b", "KNIGHT", re.IGNORECASE),
    (r"\bTEDQNG\b", "TELLING", re.IGNORECASE),
    (r"\bTS\s+TO\s+EARLY\b", "ITS TOO EARLY", re.IGNORECASE),
    (r"\bTHS\b", "THIS", re.IGNORECASE),
    (r"\bPEOPLE\b", "PEOPLE", re.IGNORECASE),
    (r"\bPEPLE\b", "PEOPLE", re.IGNORECASE),
    (r"\bSTLL\b", "STILL", re.IGNORECASE),
    (r"\bNTO\b", "INTO", re.IGNORECASE),
    (r"\bQUE\$TIONS\b", "QUESTIONS", re.IGNORECASE),
    (r"\bTIO\b", "TO", re.IGNORECASE),
    (r"\bANDAS\b", "AND AS", re.IGNORECASE),
    (r"\bMORETIME\b", "MORE TIME", re.IGNORECASE),
    (r"\bIDIDN'T\b", "I DIDN'T", re.IGNORECASE),
    (r"\bRICARDOE\b", "RICARDO", re.IGNORECASE),
    (r"\bHIRHSTAND\b", "WITHSTAND", re.IGNORECASE),
    (r"\bTRHE\b", "THE", re.IGNORECASE),
    (r"\bIMMATIURITYBORNE\b", "IMMATURITY BORNE", re.IGNORECASE),
    (r"\bFEEUING\b", "FEELING", re.IGNORECASE),
    (r"\bBAUES\b", "BATTLES", re.IGNORECASE),
    (r"\bDYNG\b", "DYING", re.IGNORECASE),
    (r"\bMASTER\.P\b", "MASTER.", re.IGNORECASE),
    (r"\bRALD\b", "RAID", re.IGNORECASE),
    (r"\bRGHT\b", "RIGHT", re.IGNORECASE),
    (r"\bSTHRT\b", "START", re.IGNORECASE),
    (r"\brt\b", "it", re.IGNORECASE),
    (r"\bMSTAES\b", "MISTAKES", re.IGNORECASE),
    (r"\blfe\b", "LIFE", re.IGNORECASE),
    (r"\bln\b", "IN", re.IGNORECASE),
    (r"\bSHOUID\b", "SHOULD", re.IGNORECASE),
    (r"\bSINCETHOSE\b", "SINCE THOSE", re.IGNORECASE),
    (r"\bHAVEA\b", "HAVE A", re.IGNORECASE),
    (r"\bWlll\b", "WILL", re.IGNORECASE),
    (r"(?<![A-Za-z0-9])THIN%(?![A-Za-z0-9])", "THINK", re.IGNORECASE),
    (r"\bldun\b", "Idun", re.IGNORECASE),
    (r"\bBUMIANS\b", "HUMANS", re.IGNORECASE),
]

COMMON_WORDS = {
    "THE",
    "AND",
    "YOU",
    "FOR",
    "ARE",
    "IS",
    "A",
    "I",
    "TO",
    "WE",
    "NO",
    "YES",
    "OK",
    "WHY",
    "WHAT",
    "HUH",
}
SHORT_QUOTED_DIALOGUE_WORDS = {"I", "WE", "YOU", "NO", "YES", "OK"}
KNOWN_LATIN_OCR_ARTIFACTS = {"HFOR"}
CJK_LETTER_PATTERN = re.compile(
    r"[\u1100-\u11FF\u3000-\u303F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)


@dataclass
class OcrCorrection:
    from_text: str
    to: str
    reason: str
    confidence: float

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["from"] = data.pop("from_text")
        return data


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).upper()


def _is_gibberish(text: str) -> bool:
    if _is_short_quoted_dialogue(text):
        return False
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if CJK_LETTER_PATTERN.search(compact):
        return False
    alpha = sum(ch.isalpha() for ch in compact)
    if alpha / max(1, len(compact)) < 0.45:
        return True
    if re.search(r"([A-Z])\1{5,}", compact.upper()):
        return True
    return False


def _apply_mandatory(text: str) -> tuple[str, list[OcrCorrection]]:
    key = _norm_key(text)
    if key in MANDATORY_CORRECTIONS:
        target = MANDATORY_CORRECTIONS[key]
        return target, [OcrCorrection(text, target, "mandatory_ocr_correction", 1.0)]
    return text, []


def _apply_inline_mandatory(text: str) -> tuple[str, list[OcrCorrection]]:
    result = text
    corrections: list[OcrCorrection] = []
    for pattern, replacement, flags in INLINE_MANDATORY_CORRECTIONS:
        updated = re.sub(pattern, replacement, result, flags=flags)
        if updated != result:
            corrections.append(
                OcrCorrection(result, updated, "mandatory_ocr_inline_correction", 1.0)
            )
            result = updated
    return result, corrections


def _apply_punctuation_spacing(text: str) -> tuple[str, list[OcrCorrection]]:
    result = str(text or "")
    updated = re.sub(r"([!?]+)([\"']?)(?=[A-Za-z])", r"\1\2 ", result)
    updated = re.sub(r",(?=[A-Za-z])", ", ", updated)
    if updated == result:
        return result, []
    return updated, [OcrCorrection(result, updated, "repair_missing_punctuation_spacing", 0.98)]


def _repair_truncated_or_joined_text(text: str) -> tuple[str, list[OcrCorrection]]:
    result = str(text or "")
    corrections: list[OcrCorrection] = []
    updated = result
    for source, replacement in TRUNCATED_JOINED_REPAIRS.items():
        updated = re.sub(
            rf"\b{re.escape(source)}\b",
            replacement,
            updated,
            flags=re.IGNORECASE,
        )
    updated = re.sub(r"([!?]+)([\"']?)(?=[A-Za-z])", r"\1\2 ", updated)
    updated = re.sub(r",(?=[A-Za-z])", ", ", updated)
    if updated != result:
        corrections.append(OcrCorrection(result, updated, "repair_truncated_or_joined_ocr", 0.9))
    return updated, corrections


def repair_ocr_truncated_or_joined(record: dict[str, Any]) -> dict[str, Any]:
    """Try deterministic OCR join repair before leaving a record in review."""

    updated = dict(record)
    original = str(updated.get("text") or updated.get("raw_ocr") or updated.get("original") or "")
    repaired, spacing_corrections = _apply_punctuation_spacing(original)
    repaired, joined_corrections = _repair_truncated_or_joined_text(repaired)
    corrections = [*spacing_corrections, *joined_corrections]

    if repaired != original:
        updated["text"] = repaired
        updated["normalized_ocr"] = repaired
        updated["normalized_text_final"] = repaired
        updated["ocr_repair_status"] = "repaired"
        flags = [flag for flag in list(updated.get("qa_flags") or []) if flag != "ocr_truncated_or_joined"]
        if "ocr_joined_repaired" not in flags:
            flags.append("ocr_joined_repaired")
        updated["qa_flags"] = flags
        updated["needs_review"] = False
        if str(updated.get("route_reason") or "").strip().lower() == "ocr_truncated_or_joined":
            updated.pop("route_action", None)
            updated.pop("route_reason", None)
        updated.pop("preserve_original", None)
        updated["normalization"] = {
            "changed": True,
            "corrections": [item.to_json() for item in corrections],
            "is_gibberish": _is_gibberish(repaired),
        }
        updated["content_class"] = "text"
        updated["route"] = "text"
        updated["skip_processing"] = False
        apply_route_action(updated, route_reason=updated.get("route_reason"))
        return updated

    updated["ocr_repair_status"] = "unrepaired" if is_ocr_truncated_or_joined(original) else "not_needed"
    if updated["ocr_repair_status"] == "unrepaired":
        updated["needs_review"] = True
        flags = list(updated.get("qa_flags") or [])
        if "ocr_truncated_or_joined" not in flags:
            flags.append("ocr_truncated_or_joined")
        updated["qa_flags"] = flags
        apply_route_action(updated, route_action="review_required", route_reason="ocr_truncated_or_joined")
    return updated


def _is_scanlation_credit(text: str) -> bool:
    normalized = _norm_key(text)
    compact = re.sub(r"[^A-Z0-9]+", "", normalized)
    if not normalized:
        return False
    if any(
        token in compact
        for token in (
            "ASURA",
            "ASURASCANS",
            "ASURASOANS",
            "ILEAFSKY",
            "SECRETSCANS",
            "SUPPORTUS",
            "KOFI",
            "DISCORDGG",
            "JOINUSATDISCORD",
            "READFIRST",
        )
    ):
        return True
    if any(
        phrase in normalized
        for phrase in (
            "FASTEST RELEASES",
            "SECRET SCANS",
            "SUPPORT US",
            "JOIN US AT DISCORD",
            "WE ARE RECRUITING",
            "WE ARE LOOKING FOR",
            "JOIN OUR DISCORD",
            "JOIN OUR TEAM",
            "PART OF OUR TEAM",
            "HELP US OUT",
            "READ FIRST",
        )
    ):
        return True
    role_tokens = re.findall(r"\b(?:TL|PR|TS|QC|RP|CI|CL|RD|RAW)\b", normalized)
    if len(role_tokens) >= 3:
        return True
    if re.search(r"\b(?:REDRAWERS?|CLEANERS?|TYPESETTERS?|TRANSLATORS?|PROOFREADERS?|QUALITY\s+CHECKERS?)\b", normalized):
        return True
    if (
        ". COM" in normalized
        or normalized.endswith(".COM")
        or "DISCORD" in normalized
        or "ISCORD.GG" in normalized
        or "ISCORD GG" in normalized
        or "ISCORDGG" in compact
    ):
        return True
    if "@" in text and any(token in normalized for token in (".COM", "NAVER", "GMAIL", "MAIL")):
        return True
    if re.search(r"[A-Z0-9_+-]+@[A-Z0-9.-]+", normalized):
        return True
    if compact.endswith("COM") and any(ch.isdigit() for ch in compact):
        return True
    if _is_hyphenated_credit_name_list(normalized):
        return True
    return False


def _mark_scanlation_credit_suppressed(updated: dict[str, Any], *, text: str) -> dict[str, Any]:
    flags = list(updated.get("qa_flags") or [])
    if "scanlation_credit_suppressed" not in flags:
        flags.append("scanlation_credit_suppressed")
    updated["qa_flags"] = flags
    updated["text"] = text
    updated["route"] = "text"
    updated["content_class"] = "text"
    updated["needs_review"] = False
    updated["skip_processing"] = True
    updated["skip_reason"] = "scanlation_credit_suppressed"
    updated["route_action"] = "review_required"
    updated["route_reason"] = "scanlation_credit_suppressed"
    return updated


def _is_hyphenated_credit_name_list(normalized: str) -> bool:
    tokens = re.findall(r"-?[A-Z0-9][A-Z0-9_]{2,}", normalized)
    hyphenated = [token for token in tokens if token.startswith("-")]
    if len(hyphenated) < 2:
        return False
    if not tokens:
        return False
    hyphen_ratio = len(hyphenated) / float(max(1, len(tokens)))
    return hyphen_ratio >= 0.66 or (len(hyphenated) >= 3 and hyphen_ratio >= 0.50)


def _is_short_quoted_dialogue(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped or not re.search(r"[?!]", stripped):
        return False
    words = re.findall(r"[A-Za-z]+", stripped)
    if len(words) != 1:
        return False
    return words[0].upper() in SHORT_QUOTED_DIALOGUE_WORDS


def _is_known_latin_ocr_artifact(text: str) -> bool:
    normalized = re.sub(r"[^A-Z]", "", str(text or "").upper())
    return normalized in KNOWN_LATIN_OCR_ARTIFACTS


def _bbox_area(value: Any) -> int:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0
    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return 0
    return max(0, x2 - x1) * max(0, y2 - y1)


def _line_polygon_bbox_area(record: dict[str, Any]) -> int:
    xs: list[int] = []
    ys: list[int] = []
    for polygon in record.get("line_polygons") or []:
        if not isinstance(polygon, (list, tuple)):
            continue
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                xs.append(int(point[0]))
                ys.append(int(point[1]))
            except (TypeError, ValueError):
                continue
    if not xs or not ys:
        return 0
    return max(0, max(xs) - min(xs)) * max(0, max(ys) - min(ys))


def _background_looks_like_art(record: dict[str, Any]) -> bool:
    value = record.get("background_rgb")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return False
    try:
        rgb = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return False
    luma = sum(rgb) / 3.0
    chroma = max(rgb) - min(rgb)
    return luma < 226.0 or chroma > 22.0


def _looks_like_short_art_ocr_fragment(record: dict[str, Any], text: str) -> bool:
    compact = re.sub(r"[^A-Z]", "", str(text or "").upper())
    if not compact or len(compact) > 6:
        return False
    if compact in COMMON_WORDS or _is_short_quoted_dialogue(text):
        return False
    if CJK_LETTER_PATTERN.search(str(text or "")):
        return False
    vowels = sum(ch in "AEIOU" for ch in compact)
    vowel_ratio = vowels / float(max(1, len(compact)))
    bbox_area = max(
        _bbox_area(record.get("text_pixel_bbox")),
        _bbox_area(record.get("bbox")),
        _line_polygon_bbox_area(record),
    )
    if len(compact) <= 1:
        return bbox_area >= 12000
    if _line_polygon_bbox_area(record) <= 0 and _background_looks_like_art(record):
        return bbox_area >= 4000
    if vowel_ratio > 0.25:
        return False
    return bbox_area >= 30000 and _background_looks_like_art(record)


def _looks_like_short_dark_visual_reocr_art_fragment(record: dict[str, Any], text: str) -> bool:
    flags = {
        str(flag).strip().lower()
        for flag in record.get("qa_flags") or []
        if str(flag).strip()
    }
    dark_visual_reocr = bool(
        {
            "candidate_crop_direct_paddle_reocr",
            "dark_bubble_oval_reocr",
            "partial_dark_bubble_lobe_reocr",
            "detected_dark_bubble_without_text_reocr",
            "dark_bubble_visual_glyph_mask_replaced_geometry",
        }
        & flags
    )
    if not dark_visual_reocr:
        return False
    token_source = str(text or record.get("raw_ocr") or record.get("original") or "").strip()
    if re.search(r"[?!.,]", token_source):
        return False
    tokens = re.findall(r"[A-Za-z]+", token_source)
    if len(tokens) != 1:
        return False
    compact = re.sub(r"[^A-Z]", "", tokens[0].upper())
    if not (2 <= len(compact) <= 4):
        return False
    if compact in COMMON_WORDS or _is_short_quoted_dialogue(token_source):
        return False
    if CJK_LETTER_PATTERN.search(token_source):
        return False
    bbox_area = max(
        _bbox_area(record.get("text_pixel_bbox")),
        _bbox_area(record.get("bbox")),
    )
    return bbox_area >= 800


def _looks_like_gibberish_art_ocr_fragment(record: dict[str, Any], text: str) -> bool:
    if not _is_gibberish(text):
        return False
    if CJK_LETTER_PATTERN.search(str(text or "")):
        return False
    if not _background_looks_like_art(record):
        return False
    compact = re.sub(r"\s+", "", str(text or ""))
    numeric_fragment = bool(re.fullmatch(r"[\d.,:/\\-]+", compact))
    repeated_noise_fragment = False
    for token in re.findall(r"[A-Za-z]{4,}", compact.upper()):
        if len(set(token)) <= 2:
            repeated_noise_fragment = True
            break
    line_polygon_area = _line_polygon_bbox_area(record)
    if line_polygon_area > 0 and not numeric_fragment and not repeated_noise_fragment:
        return False
    bbox_area = max(
        _bbox_area(record.get("text_pixel_bbox")),
        _bbox_area(record.get("bbox")),
    )
    if repeated_noise_fragment and bbox_area < 2000:
        return False
    if numeric_fragment:
        return bbox_area >= 200
    if repeated_noise_fragment:
        return bbox_area >= 2000
    return bbox_area >= 3500


def _visual_evidence_review_reason(record: dict[str, Any], text: str) -> str | None:
    flags = {
        str(flag).strip().lower()
        for flag in record.get("qa_flags") or []
        if str(flag).strip()
    }
    if "ocr_art_fragment_suspected" in flags:
        return "ocr_art_fragment_suspected"
    if "cjk_visual_misread_in_english_source" in flags:
        return "cjk_visual_misread_in_english_source"
    if "raw_text_evidence_missing" in flags and (
        "fast_fill_no_glyph_evidence" in flags
        or "render_on_art_suspected" in flags
    ):
        return "ocr_visual_evidence_missing"
    if "render_on_art_suspected" in flags and _is_scanlation_credit(text):
        return "ocr_visual_art_suspected"
    return None


def _has_real_bubble_evidence(record: dict[str, Any]) -> bool:
    route_reason = str(record.get("route_reason") or "").strip().lower()
    if "dialogue_balloon" in route_reason or "speech_balloon" in route_reason:
        return True
    bubble_bbox = record.get("bubble_mask_bbox")
    inner_bbox = record.get("bubble_inner_bbox")
    if not str(record.get("bubble_id") or "").strip():
        return False
    if not _record_has_valid_bbox4(bubble_bbox) or not _record_has_valid_bbox4(inner_bbox):
        return False
    try:
        inner = [int(v) for v in inner_bbox[:4]]
    except (TypeError, ValueError):
        return False
    if inner == [0, 0, 32, 32]:
        return False
    return True


def _loose_scene_text_review_reason(record: dict[str, Any], text: str) -> str | None:
    if _has_real_bubble_evidence(record):
        return None
    profiles = {
        str(record.get("layout_profile") or "").strip().lower(),
        str(record.get("block_profile") or "").strip().lower(),
        str(record.get("background_type") or "").strip().lower(),
    }
    if profiles & {"ui_form", "dark_panel", "colored_status_panel"}:
        return None
    token_source = str(text or record.get("raw_ocr") or record.get("original") or "").strip()
    tokens = re.findall(r"[A-Za-z0-9]+", token_source)
    single_token = len(tokens) == 1 and re.fullmatch(r"[A-Z0-9]{4,14}", tokens[0]) is not None
    has_speech_punctuation = bool(re.search(r"[?!.,]", token_source))
    if "standard" in profiles and not has_speech_punctuation:
        return "non_balloon_scene_text"
    if single_token and not has_speech_punctuation and (profiles & {"white_balloon", "speech_balloon"}):
        return "non_balloon_scene_text"
    return None


def _mark_visual_evidence_review(updated: dict[str, Any], reason: str) -> dict[str, Any]:
    updated["route"] = "text"
    updated["content_class"] = "text"
    updated["needs_review"] = True
    updated["skip_processing"] = False
    updated.pop("preserve_original", None)
    updated["skip_reason"] = None
    apply_route_action(updated, route_action="review_required", route_reason=reason)
    return updated


def _apply_glossary_fuzzy(text: str, glossary: dict[str, str] | None) -> tuple[str, list[OcrCorrection]]:
    if not glossary:
        return text, []
    key = _norm_key(text)
    if key in COMMON_WORDS or len(key) < 4:
        return text, []
    best: tuple[str, float] | None = None
    for source in glossary:
        source_key = _norm_key(source)
        if source_key in COMMON_WORDS or len(source_key) < 4:
            continue
        score = _similarity(key, source_key)
        if score >= 0.84 and (best is None or score > best[1]):
            best = (source, score)
    if best is None:
        return text, []
    return best[0], [OcrCorrection(text, best[0], "glossary_fuzzy_match", best[1])]


def _apply_stutter(text: str, glossary: dict[str, str] | None) -> tuple[str, list[OcrCorrection]]:
    match = re.match(r"^([A-Za-z])-([A-Za-z][A-Za-z ]+)([?!。！？.]*)$", text.strip())
    if not match or not glossary:
        return text, []
    _, term, punctuation = match.groups()
    target = glossary.get(term.upper()) or glossary.get(term.title()) or glossary.get(term)
    if not target:
        return text, []
    normalized = f"{target[0]}-{target}{punctuation}"
    return normalized, [OcrCorrection(text, normalized, "stutter_glossary_translation", 0.95)]


def normalize_ocr_text(text: str, glossary: dict[str, str] | None = None) -> dict[str, Any]:
    raw = text or ""
    normalized = raw
    corrections: list[OcrCorrection] = []

    normalized, new = _apply_stutter(normalized, glossary)
    corrections.extend(new)
    if not corrections:
        normalized, new = _apply_mandatory(normalized)
        corrections.extend(new)
    if not corrections:
        normalized, new = _apply_inline_mandatory(normalized)
        corrections.extend(new)
    normalized, new = _apply_punctuation_spacing(normalized)
    corrections.extend(new)
    normalized, new = _repair_truncated_or_joined_text(normalized)
    corrections.extend(new)
    if not corrections:
        normalized, new = _apply_glossary_fuzzy(normalized, glossary)
        corrections.extend(new)

    is_gibberish = _is_gibberish(normalized)
    return {
        "raw_ocr": raw,
        "normalized_ocr": normalized,
        "normalization": {
            "changed": normalized != raw,
            "corrections": [item.to_json() for item in corrections],
            "is_gibberish": is_gibberish,
        },
    }


def normalize_ocr_record(record: dict[str, Any], glossary: dict[str, str] | None = None) -> dict[str, Any]:
    raw = str(record.get("raw_ocr") or record.get("text") or record.get("original") or "")
    normalized = normalize_ocr_text(raw, glossary)
    provided_final = str(record.get("normalized_text_final") or "").strip()
    provided_normalization = record.get("normalization") if isinstance(record.get("normalization"), dict) else {}
    try:
        provided_confidence = float(provided_normalization.get("confidence_after_estimate") or 0.0)
    except (TypeError, ValueError):
        provided_confidence = 0.0
    if provided_final and provided_final != raw and provided_confidence >= 0.7:
        normalized["normalized_ocr"] = provided_final
        normalized["normalization"] = {
            **normalized.get("normalization", {}),
            **provided_normalization,
            "changed": True,
        }
    updated = dict(record)
    updated.update(normalized)
    updated["normalized_text_final"] = normalized["normalized_ocr"]
    normalize_rotated_text_metadata(updated)
    _strip_removed_legacy_decision_metadata(updated)
    visual_review_reason = _visual_evidence_review_reason(updated, normalized["normalized_ocr"])
    if visual_review_reason:
        updated["text"] = normalized["normalized_ocr"] if normalized["normalized_ocr"] else raw
        return _mark_visual_evidence_review(updated, visual_review_reason)
    loose_scene_review_reason = _loose_scene_text_review_reason(updated, normalized["normalized_ocr"])
    if loose_scene_review_reason:
        flags = list(updated.get("qa_flags") or [])
        if "non_balloon_scene_text_review" not in flags:
            flags.append("non_balloon_scene_text_review")
        updated["qa_flags"] = flags
        updated["text"] = normalized["normalized_ocr"] if normalized["normalized_ocr"] else raw
        return _mark_visual_evidence_review(updated, loose_scene_review_reason)
    if _looks_like_short_dark_visual_reocr_art_fragment(updated, normalized["normalized_ocr"]):
        flags = list(updated.get("qa_flags") or [])
        if "ocr_art_fragment_suspected" not in flags:
            flags.append("ocr_art_fragment_suspected")
        updated["qa_flags"] = flags
        updated["text"] = raw
        return _mark_visual_evidence_review(updated, "ocr_art_fragment_suspected")
    explicit_route_action = str(updated.get("route_action") or "").strip().lower()
    has_explicit_route_action = explicit_route_action in ROUTE_ACTIONS
    retain_low_confidence_dialogue = (
        not has_explicit_route_action
        and should_retain_low_confidence_dialogue_ocr(updated)
    )
    if _is_known_latin_ocr_artifact(raw):
        flags = list(updated.get("qa_flags") or [])
        if "suspected_ocr_error" not in flags:
            flags.append("suspected_ocr_error")
        updated["qa_flags"] = flags
        updated["text"] = raw
        updated["route"] = "text"
        updated["content_class"] = "text"
        route_action = explicit_route_action if has_explicit_route_action else "translate_inpaint_render"
        apply_route_action(updated, route_action=route_action, route_reason=updated.get("route_reason") or "ocr_artifact")
        return updated
    if _looks_like_short_art_ocr_fragment(updated, normalized["normalized_ocr"]):
        flags = list(updated.get("qa_flags") or [])
        if "ocr_art_fragment_suspected" not in flags:
            flags.append("ocr_art_fragment_suspected")
        updated["qa_flags"] = flags
        updated["text"] = raw
        updated["route"] = "text"
        updated["content_class"] = "text"
        updated["needs_review"] = True
        updated["skip_processing"] = False
        updated.pop("preserve_original", None)
        apply_route_action(updated, route_action="review_required", route_reason="ocr_art_fragment_suspected")
        return updated
    if _looks_like_gibberish_art_ocr_fragment(updated, normalized["normalized_ocr"]):
        flags = list(updated.get("qa_flags") or [])
        for flag in ("ocr_gibberish", "ocr_art_fragment_suspected"):
            if flag not in flags:
                flags.append(flag)
        updated["qa_flags"] = flags
        updated["text"] = raw
        updated["route"] = "text"
        updated["content_class"] = "text"
        updated["needs_review"] = True
        updated["skip_processing"] = False
        updated.pop("preserve_original", None)
        apply_route_action(updated, route_action="review_required", route_reason="ocr_art_fragment_suspected")
        return updated
    stale_truncated_review = (
        explicit_route_action == "review_required"
        and str(updated.get("route_reason") or "").strip().lower() == "ocr_truncated_or_joined"
    )
    if not stale_truncated_review and (_is_scanlation_credit(raw) or _is_scanlation_credit(normalized["normalized_ocr"])):
        return _mark_scanlation_credit_suppressed(updated, text=raw)
    if not normalized["normalization"]["is_gibberish"]:
        updated["text"] = normalized["normalized_ocr"]
    else:
        flags = list(updated.get("qa_flags") or [])
        if "ocr_gibberish" not in flags:
            flags.append("ocr_gibberish")
        updated["qa_flags"] = flags
        updated["route"] = "text"
        updated["content_class"] = "text"
        if not has_explicit_route_action:
            apply_route_action(updated, route_action="translate_inpaint_render", route_reason="dialogue_balloon_with_english_text")
    flags = list(updated.get("qa_flags") or [])
    if retain_low_confidence_dialogue:
        updated["skip_processing"] = False
        updated["skip_reason"] = None
        if not has_explicit_route_action:
            apply_route_action(
                updated,
                route_action="translate_inpaint_render",
                route_reason="ocr_retention_low_confidence_dialogue",
            )
    truncated_or_joined = is_ocr_truncated_or_joined(str(updated.get("text") or raw))
    if truncated_or_joined:
        if "ocr_truncated_or_joined" not in flags:
            flags.append("ocr_truncated_or_joined")
        updated["needs_review"] = True
        if _should_translate_unrepaired_truncated_or_joined(updated):
            apply_route_action(
                updated,
                route_action="translate_inpaint_render",
                route_reason="ocr_truncated_or_joined_retained_for_translation",
            )
        else:
            apply_route_action(
                updated,
                route_action="review_required",
                route_reason="ocr_truncated_or_joined",
            )
    elif is_ocr_truncated_or_joined(raw):
        flags = [flag for flag in flags if flag != "ocr_truncated_or_joined"]
        if "ocr_joined_repaired" not in flags:
            flags.append("ocr_joined_repaired")
        updated["needs_review"] = False
        if explicit_route_action == "review_required" and str(updated.get("route_reason") or "").strip().lower() == "ocr_truncated_or_joined":
            updated.pop("route_action", None)
            updated.pop("route_reason", None)
            explicit_route_action = ""
            has_explicit_route_action = False
    if flags:
        updated["qa_flags"] = flags
    if has_explicit_route_action and not truncated_or_joined:
        apply_route_action(updated, route_action=explicit_route_action, route_reason=updated.get("route_reason"))
    elif not updated.get("route_action"):
        updated["content_class"] = "text"
        apply_route_action(updated, route_reason=updated.get("route_reason"))
    updated["content_class"] = "text"
    updated["route"] = "text"
    updated["skip_processing"] = False
    updated.pop("preserve_original", None)
    updated["skip_reason"] = None
    return updated


def merge_same_balloon_fragments_before_translation(texts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge OCR fragments that are inside the same balloon before translation.

    This keeps machine translation from seeing sentence fragments like
    "BITCHIS AREAL" and "ACTRESS..." as independent prompts.
    """

    records = [dict(item or {}) for item in texts or []]
    if len(records) < 2:
        return records

    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, record in enumerate(records):
        key = _same_balloon_fragment_group_key(record)
        if key is None or _record_should_not_merge_for_translation(record):
            continue
        grouped.setdefault(key, []).append(index)

    consumed: set[int] = set()
    merged_records: dict[int, dict[str, Any]] = {}
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        ordered = sorted(indexes, key=lambda item: _record_reading_order(records[item]))
        group = [records[index] for index in ordered]
        if not _same_balloon_fragment_group_should_merge(group):
            continue
        merged = _merge_same_balloon_fragment_group(group)
        if merged is None:
            continue
        merged_records[ordered[0]] = merged
        consumed.update(ordered[1:])

    _merge_same_band_joined_word_fragments(records, consumed, merged_records)
    _merge_same_band_dependent_fragments(records, consumed, merged_records)

    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if index in consumed:
            continue
        result.append(merged_records.get(index, record))
    return result


def _same_balloon_fragment_group_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    band_id = str(record.get("band_id") or record.get("_band_id") or "").strip()
    if not band_id:
        trace_id = str(record.get("trace_id") or "").strip()
        match = re.search(r"@(page_\d{3}_band_\d{3})$", trace_id)
        if match:
            band_id = match.group(1)
    if not band_id:
        return None

    for key in ("bubble_mask_bbox", "balloon_bbox"):
        bbox = _record_bbox4(record.get(key))
        if bbox is not None:
            return (key, band_id, *bbox)
    bubble_id = str(record.get("bubble_id") or record.get("bubbleId") or "").strip()
    if bubble_id:
        return ("bubble_id", band_id, bubble_id)
    return None


def _record_band_id(record: dict[str, Any]) -> str:
    band_id = str(record.get("band_id") or record.get("_band_id") or "").strip()
    if band_id:
        return band_id
    trace_id = str(record.get("trace_id") or "").strip()
    match = re.search(r"@(page_\d{3}_band_\d{3})$", trace_id)
    return match.group(1) if match else ""


def _record_should_not_merge_for_translation(record: dict[str, Any]) -> bool:
    action = str(record.get("route_action") or "").strip().lower()
    if action in {"preserve", "merged_into_primary", "suppress"}:
        return True
    if str(record.get("translate_policy") or "").strip().lower() == "skip_translation":
        return True
    text = _record_source_text_for_merge(record)
    if not text or CJK_LETTER_PATTERN.search(text):
        return True
    return not bool(re.search(r"[A-Za-z]", text))


def _bbox_area_for_merge(bbox: list[int] | None) -> int:
    if not bbox:
        return 0
    return max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))


def _bbox_iou_for_merge(left: list[int], right: list[int]) -> float:
    inter = _bbox_overlap_area(left, right)
    if inter <= 0:
        return 0.0
    union = _bbox_area_for_merge(left) + _bbox_area_for_merge(right) - inter
    return inter / float(max(1, union))


def _record_stable_text_bbox(record: dict[str, Any]) -> list[int] | None:
    text_bbox = _record_bbox4(record.get("text_pixel_bbox"))
    layout_bbox = _record_bbox4(record.get("layout_bbox"))
    bbox = _record_bbox4(record.get("bbox"))
    peer = layout_bbox or bbox
    if text_bbox is not None and peer is not None:
        min_area = max(1, min(_bbox_area_for_merge(text_bbox), _bbox_area_for_merge(peer)))
        inter = _bbox_overlap_area(text_bbox, peer)
        if inter / float(min_area) >= 0.20 or _bbox_iou_for_merge(text_bbox, peer) >= 0.12:
            return text_bbox
        return peer
    return text_bbox or peer


def _record_is_dark_bubble(record: dict[str, Any]) -> bool:
    flags = {str(flag).strip().lower() for flag in record.get("qa_flags") or [] if str(flag).strip()}
    source = str(record.get("bubble_mask_source") or record.get("balloon_mask_source") or "").strip().lower()
    profile = str(record.get("layout_profile") or record.get("block_profile") or "").strip().lower()
    bubble_id = str(record.get("bubble_id") or record.get("bubbleId") or "").strip().lower()
    return bool(
        source in {"image_dark_bubble_mask", "image_dark_panel_mask"}
        or profile in {"dark_bubble", "dark_panel"}
        or "partial_dark_lobe" in bubble_id
        or "dark_lobe" in bubble_id
        or any(flag.startswith("dark_bubble") for flag in flags)
    )


def _records_are_distinct_dark_bubble_lobes(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not (_record_is_dark_bubble(left) and _record_is_dark_bubble(right)):
        return False
    left_bbox = _record_stable_text_bbox(left)
    right_bbox = _record_stable_text_bbox(right)
    if left_bbox is None or right_bbox is None:
        return False
    left_id = str(left.get("bubble_id") or left.get("bubbleId") or "").strip()
    right_id = str(right.get("bubble_id") or right.get("bubbleId") or "").strip()
    if left_id and right_id and left_id != right_id:
        return True
    left_w = max(1, left_bbox[2] - left_bbox[0])
    right_w = max(1, right_bbox[2] - right_bbox[0])
    left_h = max(1, left_bbox[3] - left_bbox[1])
    right_h = max(1, right_bbox[3] - right_bbox[1])
    left_cx = (left_bbox[0] + left_bbox[2]) / 2.0
    right_cx = (right_bbox[0] + right_bbox[2]) / 2.0
    left_cy = (left_bbox[1] + left_bbox[3]) / 2.0
    right_cy = (right_bbox[1] + right_bbox[3]) / 2.0
    dx = abs(left_cx - right_cx)
    dy = abs(left_cy - right_cy)
    inter = _bbox_overlap_area(left_bbox, right_bbox)
    min_area = max(1, min(_bbox_area_for_merge(left_bbox), _bbox_area_for_merge(right_bbox)))
    if inter / float(min_area) >= 0.20:
        return False
    same_bubble_region = False
    left_bubble = _record_bbox4(left.get("bubble_mask_bbox") or left.get("balloon_bbox"))
    right_bubble = _record_bbox4(right.get("bubble_mask_bbox") or right.get("balloon_bbox"))
    if left_bubble is not None and right_bubble is not None:
        same_bubble_region = _bbox_iou_for_merge(left_bubble, right_bubble) >= 0.50
    side_by_side = dx >= max(96, int(max(left_w, right_w) * 0.75)) and dy <= max(180, int(max(left_h, right_h) * 1.8))
    return same_bubble_region and side_by_side


def _group_has_distinct_dark_bubble_lobes(group: list[dict[str, Any]]) -> bool:
    if len(group) < 2:
        return False
    for index, left in enumerate(group):
        for right in group[index + 1 :]:
            if _records_are_distinct_dark_bubble_lobes(left, right):
                return True
    return False


def _same_balloon_fragment_group_should_merge(group: list[dict[str, Any]]) -> bool:
    if len(group) < 2:
        return False
    if _group_has_distinct_dark_bubble_lobes(group):
        return False
    boxes = [_record_bbox4(item.get("text_pixel_bbox")) or _record_bbox4(item.get("bbox")) for item in group]
    if any(box is None for box in boxes):
        return True
    ordered_boxes = [box for box in boxes if box is not None]
    for prev, nxt in zip(ordered_boxes, ordered_boxes[1:]):
        prev_h = max(1, prev[3] - prev[1])
        next_h = max(1, nxt[3] - nxt[1])
        vertical_gap = max(0, nxt[1] - prev[3])
        horizontal_overlap = min(prev[2], nxt[2]) - max(prev[0], nxt[0])
        min_width = max(1, min(prev[2] - prev[0], nxt[2] - nxt[0]))
        if vertical_gap <= max(52, int(max(prev_h, next_h) * 1.35)):
            continue
        if horizontal_overlap >= int(min_width * 0.25) and vertical_gap <= 92:
            continue
        return False
    return True


def _merge_same_band_joined_word_fragments(
    records: list[dict[str, Any]],
    consumed: set[int],
    merged_records: dict[int, dict[str, Any]],
) -> None:
    by_band: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if index in consumed or index in merged_records:
            continue
        if _record_should_not_merge_for_translation(record):
            continue
        band_id = _record_band_id(record)
        if band_id:
            by_band.setdefault(band_id, []).append(index)

    for indexes in by_band.values():
        if len(indexes) < 2:
            continue
        cursor = 0
        while cursor < len(indexes):
            start = indexes[cursor]
            if start in consumed or start in merged_records:
                cursor += 1
                continue
            group_indexes = [start]
            probe = cursor + 1
            while probe < len(indexes):
                next_index = indexes[probe]
                if next_index in consumed or next_index in merged_records:
                    break
                candidate_indexes = [*group_indexes, next_index]
                candidate_group = [records[index] for index in candidate_indexes]
                if not _same_band_joined_word_fragment_group_should_merge(candidate_group):
                    break
                group_indexes.append(next_index)
                probe += 1
            if len(group_indexes) > 1:
                group = [records[index] for index in group_indexes]
                merged = _merge_same_balloon_fragment_group(group)
                if merged is not None:
                    flags = list(merged.get("qa_flags") or [])
                    if "same_band_joined_word_fragment_merged" not in flags:
                        flags.append("same_band_joined_word_fragment_merged")
                    merged["qa_flags"] = flags
                    merged_records[group_indexes[0]] = merged
                    consumed.update(group_indexes[1:])
                    cursor = probe
                    continue
            cursor += 1


def _same_band_joined_word_fragment_group_should_merge(group: list[dict[str, Any]]) -> bool:
    if len(group) < 2:
        return False
    if _group_has_distinct_dark_bubble_lobes(group):
        return False
    if not _same_band_fragment_geometry_is_close(group):
        return False
    raw_joined = _normalize_spaces(" ".join(_record_source_text_for_merge(item) for item in group))
    if not raw_joined:
        return False
    _repaired, repair_rules = _repair_same_balloon_joined_source(raw_joined)
    if not repair_rules:
        return False
    if any("AREAL" in rule or "THISIS" in rule or "THATIS" in rule for rule in repair_rules):
        return True
    first = _record_source_text_for_merge(group[0]).strip().upper()
    tail = _record_source_text_for_merge(group[-1]).strip()
    return bool(
        first.endswith(("BITCHIS", "AREAL", "A REAL", "THISIS", "THATIS"))
        and re.fullmatch(r"[A-Za-z][A-Za-z' .!?-]{2,36}", tail or "")
    )


def _merge_same_band_dependent_fragments(
    records: list[dict[str, Any]],
    consumed: set[int],
    merged_records: dict[int, dict[str, Any]],
) -> None:
    by_band: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if index in consumed or index in merged_records:
            continue
        if _record_should_not_merge_for_translation(record):
            continue
        band_id = _record_band_id(record)
        if band_id:
            by_band.setdefault(band_id, []).append(index)

    for indexes in by_band.values():
        ordered_indexes = sorted(indexes, key=lambda item: _record_reading_order(records[item]))
        cursor = 0
        while cursor < len(ordered_indexes):
            group_indexes = [ordered_indexes[cursor]]
            probe = cursor + 1
            while probe < len(ordered_indexes):
                previous = records[group_indexes[-1]]
                candidate = records[ordered_indexes[probe]]
                if not _same_band_dependent_fragment_pair_should_merge(previous, candidate):
                    break
                group_indexes.append(ordered_indexes[probe])
                probe += 1
            if len(group_indexes) > 1:
                group = [records[index] for index in group_indexes]
                merged = _merge_same_balloon_fragment_group(group)
                if merged is not None:
                    flags = list(merged.get("qa_flags") or [])
                    if "same_band_dependent_fragment_merged" not in flags:
                        flags.append("same_band_dependent_fragment_merged")
                    merged["qa_flags"] = flags
                    merged_records[group_indexes[0]] = merged
                    consumed.update(group_indexes[1:])
                    cursor = probe
                    continue
            cursor += 1


def _same_band_dependent_fragment_pair_should_merge(prev: dict[str, Any], nxt: dict[str, Any]) -> bool:
    if _record_band_id(prev) != _record_band_id(nxt):
        return False
    if _records_are_distinct_dark_bubble_lobes(prev, nxt):
        return False
    if not _same_band_dependent_fragment_geometry_is_close(prev, nxt):
        return False
    prev_text = _normalize_spaces(_record_source_text_for_merge(prev))
    next_text = _normalize_spaces(_record_source_text_for_merge(nxt))
    if not prev_text or not next_text:
        return False
    prev_up = prev_text.upper()
    next_up = next_text.upper().strip()
    next_words = re.findall(r"[A-Z][A-Z']*", next_up)
    if len(next_words) > 6:
        return False

    if next_up in {"THE PRINCIPAL", "PRINCIPAL"}:
        return bool("INTEREST" in prev_up and re.search(r"\b(TIME|TIMES)\b", prev_up))
    if next_up in {"THE CHILD'S SAKE.", "THE CHILD'S SAKE", "CHILD'S SAKE.", "CHILD'S SAKE"}:
        return bool(prev_up.rstrip().endswith((" FOR", ", FOR", "PLEASE, FOR", "PLEASE FOR")))
    if next_up in {"FRUSTRATINGTOO", "FRUSTRATING TOO"}:
        return bool(prev_up.rstrip().endswith("IS SO"))
    if next_up.startswith("WHY ARE YOU") and re.search(r"\b(OPPA|HEY|HUH|AISH)[,!?.~]*$", prev_up):
        return True
    return _records_share_candidate_merge_evidence(prev, nxt)


def _same_band_dependent_fragment_geometry_is_close(prev: dict[str, Any], nxt: dict[str, Any]) -> bool:
    prev_box = _record_bbox4(prev.get("text_pixel_bbox")) or _record_bbox4(prev.get("bbox"))
    next_box = _record_bbox4(nxt.get("text_pixel_bbox")) or _record_bbox4(nxt.get("bbox"))
    if prev_box is None or next_box is None:
        return True
    prev_h = max(1, prev_box[3] - prev_box[1])
    next_h = max(1, next_box[3] - next_box[1])
    vertical_gap = max(0, next_box[1] - prev_box[3])
    vertical_overlap = min(prev_box[3], next_box[3]) - max(prev_box[1], next_box[1])
    if vertical_overlap >= -max(8, int(min(prev_h, next_h) * 0.25)):
        return True
    if vertical_gap <= max(96, int(max(prev_h, next_h) * 1.8)):
        return True
    prev_bubble = _record_bbox4(prev.get("bubble_mask_bbox")) or _record_bbox4(prev.get("balloon_bbox"))
    next_bubble = _record_bbox4(nxt.get("bubble_mask_bbox")) or _record_bbox4(nxt.get("balloon_bbox"))
    if prev_bubble is not None and next_bubble is not None:
        return _bbox_overlap_area(prev_bubble, next_bubble) > 0
    return False


def _records_share_candidate_merge_evidence(prev: dict[str, Any], nxt: dict[str, Any]) -> bool:
    prev_ids = _record_candidate_ids(prev)
    next_ids = _record_candidate_ids(nxt)
    return bool(prev_ids and next_ids and prev_ids.intersection(next_ids))


def _record_candidate_ids(record: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in (
        "candidate_id",
        "matched_candidate_id",
        "bubble_candidate_id",
        "detect_candidate_id",
        "source_candidate_id",
    ):
        value = str(record.get(key) or "").strip()
        if value:
            result.add(value)
    for key in (
        "candidate_ids",
        "matched_candidate_ids",
        "source_candidate_ids",
        "bubble_candidate_ids",
        "detect_candidate_ids",
    ):
        values = record.get(key)
        if isinstance(values, (list, tuple, set)):
            for value in values:
                text = str(value or "").strip()
                if text:
                    result.add(text)
    return result


def _same_band_fragment_geometry_is_close(group: list[dict[str, Any]]) -> bool:
    boxes = [_record_bbox4(item.get("text_pixel_bbox")) or _record_bbox4(item.get("bbox")) for item in group]
    if any(box is None for box in boxes):
        return True
    valid_boxes = [box for box in boxes if box is not None]
    for prev, nxt in zip(valid_boxes, valid_boxes[1:]):
        prev_w = max(1, prev[2] - prev[0])
        next_w = max(1, nxt[2] - nxt[0])
        prev_h = max(1, prev[3] - prev[1])
        next_h = max(1, nxt[3] - nxt[1])
        vertical_gap = max(0, nxt[1] - prev[3])
        vertical_overlap = min(prev[3], nxt[3]) - max(prev[1], nxt[1])
        horizontal_overlap = min(prev[2], nxt[2]) - max(prev[0], nxt[0])
        prev_cx = (prev[0] + prev[2]) / 2
        next_cx = (nxt[0] + nxt[2]) / 2
        centers_close = abs(prev_cx - next_cx) <= max(prev_w, next_w) * 0.45
        rows_close = vertical_gap <= max(18, int(max(prev_h, next_h) * 0.9))
        same_line = vertical_overlap >= min(prev_h, next_h) * 0.4
        if horizontal_overlap <= 0 and not centers_close:
            return False
        if not rows_close and not same_line:
            return False
    return True


def _merge_same_balloon_fragment_group(group: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not group:
        return None
    primary = dict(group[0])
    raw_parts = [_record_source_text_for_merge(item) for item in group]
    raw_parts_before_duplicate_cleanup = list(raw_parts)
    raw_parts = _drop_leading_duplicate_fragment_parts(raw_parts)
    raw_joined = _normalize_spaces(" ".join(part for part in raw_parts if part))
    if not raw_joined:
        return None
    repaired, repair_rules = _repair_same_balloon_joined_source(raw_joined)
    if raw_parts != raw_parts_before_duplicate_cleanup and "leading_duplicate_sentence_fragment_removed" not in repair_rules:
        repair_rules.append("leading_duplicate_sentence_fragment_removed")

    for key in ("text", "original", "raw_ocr", "normalized_ocr", "normalized_text_final"):
        primary[key] = repaired
    primary["_raw_ocr_before_same_balloon_merge_repair"] = raw_joined
    primary["same_balloon_fragment_source_texts"] = list(raw_parts)
    primary["ocr_merged_source_count"] = len(group)
    primary["merge_reason"] = "same_balloon_pretranslation_ocr_fragment"
    primary["source_text_ids"] = _unique_strings(
        [
            *(primary.get("source_text_ids") or []),
            *(primary.get("_source_text_ids") or []),
            *[item.get("id") or item.get("text_id") for item in group],
            *[
                source_id
                for item in group
                for source_id in (item.get("source_text_ids") or item.get("_source_text_ids") or [])
            ],
        ]
    )
    primary["_source_text_ids"] = list(primary["source_text_ids"])
    primary["source_trace_ids"] = _unique_strings(
        [
            *(primary.get("source_trace_ids") or []),
            *(primary.get("_source_trace_ids") or []),
            *[item.get("trace_id") for item in group],
            *[
                source_id
                for item in group
                for source_id in (item.get("source_trace_ids") or item.get("_source_trace_ids") or [])
            ],
        ]
    )
    primary["_source_trace_ids"] = list(primary["source_trace_ids"])

    merged_bboxes = [
        bbox
        for item in group
        for bbox in [_record_bbox4(item.get("text_pixel_bbox")) or _record_bbox4(item.get("bbox"))]
        if bbox is not None
    ]
    if merged_bboxes:
        primary["merged_source_bboxes"] = [list(bbox) for bbox in merged_bboxes]
        primary["_merged_source_bboxes"] = [list(bbox) for bbox in merged_bboxes]
        union = _bbox_union(merged_bboxes)
        if union is not None:
            primary["bbox"] = list(union)
            primary["source_bbox"] = list(union)
            primary["text_pixel_bbox"] = list(union)

    line_polygons: list[Any] = []
    for item in group:
        for polygon in item.get("line_polygons") or []:
            line_polygons.append(polygon)
    if line_polygons:
        primary["line_polygons"] = line_polygons

    flags = _unique_strings([flag for item in group for flag in (item.get("qa_flags") or [])])
    if "same_balloon_fragment_merged" not in flags:
        flags.append("same_balloon_fragment_merged")
    if repair_rules and "ocr_joined_repaired" not in flags:
        flags.append("ocr_joined_repaired")
    flags = [flag for flag in flags if flag != "ocr_truncated_or_joined"]
    primary["qa_flags"] = flags
    primary["needs_review"] = False
    primary["skip_processing"] = False
    primary.pop("preserve_original", None)
    primary["content_class"] = "text"
    primary["route"] = "text"
    primary["route_action"] = "translate_inpaint_render"
    primary["route_reason"] = "same_balloon_pretranslation_ocr_fragment"
    confidence = _merged_confidence_estimate(group)
    primary["confidence"] = confidence
    primary["ocr_confidence"] = confidence
    primary["normalization"] = {
        "changed": repaired != raw_joined,
        "corrections": [
            {"from": raw_joined, "to": repaired, "reason": "same_balloon_joined_word_repair", "rule": rule}
            for rule in repair_rules
        ],
        "is_gibberish": _is_gibberish(repaired),
        "confidence_after_estimate": max(0.7, confidence),
    }
    return primary


def _drop_leading_duplicate_fragment_parts(raw_parts: list[str]) -> list[str]:
    repaired: list[str] = []
    for part in raw_parts:
        current = _normalize_spaces(part)
        if repaired and current:
            stripped = _strip_leading_duplicate_sentence_fragment(repaired[-1], current)
            current = stripped
        repaired.append(current)
    return repaired


def _strip_leading_duplicate_sentence_fragment(previous: str, current: str) -> str:
    prev_tokens = re.findall(r"[A-Za-z0-9']+", _normalize_spaces(previous).lower())
    if len(prev_tokens) < 5:
        return current
    match = re.match(r"^(?P<head>[A-Za-z0-9' -]{6,48}[.!?])\s*(?P<tail>.*)$", _normalize_spaces(current))
    if not match:
        return current
    head = match.group("head").strip()
    tail = match.group("tail").strip()
    head_tokens = re.findall(r"[A-Za-z0-9']+", head.lower())
    if not (3 <= len(head_tokens) <= 6):
        return current
    fuzzy_matches = 0
    for token in head_tokens:
        if len(token) <= 1:
            continue
        if any(token == prev or token in prev or prev in token for prev in prev_tokens if len(prev) > 1):
            fuzzy_matches += 1
    if fuzzy_matches < max(3, int(round(len(head_tokens) * 0.75))):
        return current
    return tail


def _repair_same_balloon_joined_source(text: str) -> tuple[str, list[str]]:
    repaired = _normalize_spaces(text)
    rules: list[str] = []
    for pattern, replacement in SAME_BALLOON_JOINED_WORD_REPAIRS:
        updated = re.sub(pattern, replacement, repaired, flags=re.IGNORECASE)
        if updated != repaired:
            repaired = updated
            rules.append(pattern)
    updated = _drop_duplicate_sentence_fragments(repaired)
    if updated != repaired:
        repaired = updated
        rules.append("duplicate_sentence_fragment_removed")
    return _normalize_spaces(repaired), rules


def _drop_duplicate_sentence_fragments(text: str) -> str:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", _normalize_spaces(text)) if part.strip()]
    if len(parts) < 2:
        return _normalize_spaces(text)
    kept: list[str] = []
    for part in parts:
        tokens = re.findall(r"[A-Za-z0-9']+", part.lower())
        if kept and 3 <= len(tokens) <= 6:
            previous_tokens = re.findall(r"[A-Za-z0-9']+", kept[-1].lower())
            if len(previous_tokens) >= 5:
                fuzzy_matches = 0
                for token in tokens:
                    if len(token) <= 1:
                        continue
                    if any(token == prev or token in prev or prev in token for prev in previous_tokens if len(prev) > 1):
                        fuzzy_matches += 1
                if fuzzy_matches >= max(3, int(round(len(tokens) * 0.75))):
                    continue
        kept.append(part)
    return _normalize_spaces(" ".join(kept))


def _record_source_text_for_merge(record: dict[str, Any]) -> str:
    return _normalize_spaces(
        str(
            record.get("normalized_text_final")
            or record.get("normalized_ocr")
            or record.get("text")
            or record.get("raw_ocr")
            or record.get("original")
            or ""
        )
    )


def _record_reading_order(record: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = _record_bbox4(record.get("text_pixel_bbox")) or _record_bbox4(record.get("bbox")) or [0, 0, 0, 0]
    return (bbox[1], bbox[0], bbox[3], bbox[2])


def _record_bbox4(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _bbox_union(boxes: list[list[int]]) -> list[int] | None:
    valid = [box for box in boxes if _record_bbox4(box) is not None]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _bbox_overlap_area(a: list[int], b: list[int]) -> int:
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _merged_confidence_estimate(group: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for item in group:
        try:
            values.append(float(item.get("confidence", item.get("ocr_confidence", 0.0)) or 0.0))
        except (TypeError, ValueError):
            pass
    if not values:
        return 0.7
    return round(max(values), 3)


def _should_translate_unrepaired_truncated_or_joined(record: dict[str, Any]) -> bool:
    if not _record_has_valid_bbox4(record.get("balloon_bbox")):
        return False
    text = str(record.get("text") or record.get("raw_ocr") or record.get("original") or "").strip()
    words = re.findall(r"[A-Za-z][A-Za-z']*", text)
    if len(words) < 6:
        return False
    if _is_scanlation_credit(text):
        return False
    return True


def _record_has_valid_bbox4(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def _strip_removed_legacy_decision_metadata(record: dict[str, Any]) -> None:
    flags = list(record.get("qa_flags") or [])
    if flags:
        record["qa_flags"] = [
            flag for flag in flags
            if str(flag or "").strip().lower() not in REMOVED_LEGACY_FILTER_FLAGS
        ]
    record["skip_processing"] = False
    record["skip_reason"] = None
    record.pop("preserve_original", None)
    record["content_class"] = "text"
    record["route"] = "text"
    action = str(record.get("route_action") or "").strip().lower()
    reason = str(record.get("route_reason") or "").strip().lower()
    if action and action not in {"translate_inpaint_render", "translate_render_only"}:
        if action != "review_required" or reason != "ocr_truncated_or_joined":
            record["needs_review"] = False
            record.pop("route_action", None)
            record.pop("route_reason", None)
    elif reason in REMOVED_LEGACY_ROUTE_REASONS:
        record.pop("route_action", None)
        record.pop("route_reason", None)
    if str(record.get("render_policy") or "").strip().lower() in {"skip", "preserve", "preserve_original"}:
        record.pop("render_policy", None)
    if str(record.get("translate_policy") or "").strip().lower() == "skip_translation":
        record.pop("translate_policy", None)


def _similarity(left: str, right: str) -> float:
    max_len = max(len(left), len(right))
    if max_len == 0:
        return 1.0
    return 1.0 - (_levenshtein(left, right) / max_len)


def _levenshtein(left: str, right: str) -> int:
    prev = list(range(len(right) + 1))
    for i, lc in enumerate(left, 1):
        cur = [i]
        for j, rc in enumerate(right, 1):
            cur.append(prev[j - 1] if lc == rc else 1 + min(prev[j], cur[j - 1], prev[j - 1]))
        prev = cur
    return prev[-1]

