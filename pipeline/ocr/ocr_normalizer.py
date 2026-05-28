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

_LEGACY_SKIP_REASONS_TO_KEEP = {
    "duplicate_lower_confidence",
    "ocr_artifact",
    "ocr_gibberish",
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
]

COMMON_WORDS = {"THE", "AND", "YOU", "FOR", "ARE", "IS", "A", "I", "TO", "WE", "NO", "YES", "OK"}
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
    if any(token in compact for token in ("ASURA", "ASURASCANS", "ASURASOANS", "ILEAFSKY")):
        return True
    if "FASTEST RELEASES" in normalized:
        return True
    if ". COM" in normalized or normalized.endswith(".COM") or "DISCORD" in normalized:
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
    updated = dict(record)
    updated.update(normalized)
    normalize_rotated_text_metadata(updated)
    _neutralize_low_confidence_visual_noise_filter(updated)
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
    stale_truncated_review = (
        explicit_route_action == "review_required"
        and str(updated.get("route_reason") or "").strip().lower() == "ocr_truncated_or_joined"
    )
    if not stale_truncated_review and (_is_scanlation_credit(raw) or _is_scanlation_credit(normalized["normalized_ocr"])):
        flags = list(updated.get("qa_flags") or [])
        updated["qa_flags"] = flags
        updated["text"] = raw
        updated["route"] = "text"
        updated["content_class"] = "text"
        updated["needs_review"] = False
        apply_route_action(updated, route_action="translate_inpaint_render", route_reason="dialogue_balloon_with_english_text")
        updated["skip_reason"] = None
        return updated
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
    elif bool(record.get("skip_processing")) and _should_preserve_legacy_skip_route(updated):
        updated["route"] = "text"
        updated["content_class"] = "text"
        apply_route_action(updated, route_action="translate_inpaint_render", route_reason="dialogue_balloon_with_english_text")
    elif not updated.get("route_action"):
        updated["content_class"] = "text"
        apply_route_action(updated, route_reason=updated.get("route_reason"))
    updated["content_class"] = "text"
    updated["route"] = "text"
    updated["skip_processing"] = False
    updated.pop("preserve_original", None)
    updated["skip_reason"] = None
    return updated


def _neutralize_low_confidence_visual_noise_filter(record: dict[str, Any]) -> None:
    flags = list(record.get("qa_flags") or [])
    had_filter_flag = "low_confidence_visual_noise" in flags
    if had_filter_flag:
        record["qa_flags"] = [flag for flag in flags if flag != "low_confidence_visual_noise"]

    reasons = {
        str(record.get("skip_reason") or "").strip().lower(),
        str(record.get("route_reason") or "").strip().lower(),
        str(record.get("reason") or "").strip().lower(),
    }
    rules = record.get("rules_applied") or []
    if isinstance(rules, list):
        reasons.update(str(rule or "").strip().lower() for rule in rules)
    if not had_filter_flag and "low_confidence_visual_noise" not in reasons:
        return

    record["skip_processing"] = False
    record["skip_reason"] = None
    record.pop("preserve_original", None)
    record["content_class"] = "text"
    record["route"] = "text"
    action = str(record.get("route_action") or "").strip().lower()
    if action in {"skip", "preserve", "review_required"}:
        record.pop("route_action", None)
        record.pop("route_reason", None)
    if str(record.get("render_policy") or "").strip().lower() in {"skip", "preserve", "preserve_original"}:
        record.pop("render_policy", None)
    if str(record.get("translate_policy") or "").strip().lower() == "skip_translation":
        record.pop("translate_policy", None)


def _should_preserve_legacy_skip_route(record: dict[str, Any]) -> bool:
    reason = str(record.get("skip_reason") or "").strip().lower()
    if "duplicate" in reason:
        return True
    if reason in _LEGACY_SKIP_REASONS_TO_KEEP:
        return True
    if bool(record.get("has_better_duplicate") or record.get("better_duplicate")):
        return True
    return bool(record.get("duplicate_of") or record.get("duplicate_replaced_by"))


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

