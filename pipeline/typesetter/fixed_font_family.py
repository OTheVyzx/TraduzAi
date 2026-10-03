"""User-selected fixed family for automatic typesetting, with file provenance.

Only the family is selected here. Size, weight, effects and geometry remain
owned by their existing contracts. Missing assets/glyphs never switch families.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unicodedata
from fontTools.ttLib import TTFont
from typesetter.font_identity import resolve_font_identity

FIXED_FONT_NAME = "CCTotallyAwesome W00 Bold.ttf"
FIXED_FONT_SHA256 = "944e441c6468b55fbafb4b268e483b90baf256e8280ff65efe3773a42a863972"
POLICY_REASON = "user_fixed_base_font_family"
POLICY = {"schema": "fixed_font_family_policy_v1", "mode": "fixed",
          "family": FIXED_FONT_NAME, "relative_path": "commercial/" + FIXED_FONT_NAME,
          "font_sha256": FIXED_FONT_SHA256, "automatic_family_selection": False,
          "source_family_fidelity": False, "rendered_fixed_family_fidelity": True,
          "scope": "automatic_typesetting_family_only"}

def _digest(payload):
    return sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()

def fixed_font_path(fonts_root: Path | None = None) -> Path:
    root = fonts_root or Path(__file__).resolve().parents[2] / "fonts"
    path = Path(root) / POLICY["relative_path"]
    if not path.is_file():
        raise ValueError(f"FixedFontUnavailable: {path}")
    if sha256(path.read_bytes()).hexdigest() != FIXED_FONT_SHA256:
        raise ValueError(f"FixedFontHashMismatch: {path}")
    return path.resolve()

def fixed_font_decision(text: str, *, fonts_root: Path | None = None) -> dict:
    path = fixed_font_path(fonts_root)
    normalized = unicodedata.normalize("NFC", text)
    font = TTFont(path, lazy=True)
    try:
        coverage = font.getBestCmap() or {}
        missing = sorted({ord(c) for c in normalized if not c.isspace() and ord(c) not in coverage})
    finally:
        font.close()
    if missing:
        raise ValueError("FixedFontMissingGlyph: " + ",".join(f"U+{cp:04X}" for cp in missing))
    record = {**POLICY,"reason":POLICY_REASON,"policy_sha256":_digest(POLICY),
              "text_sha256":sha256(normalized.encode("utf-8")).hexdigest(),
              "identity":resolve_font_identity(path).to_dict()}
    record["decision_sha256"] = _digest(record)
    return record

def validate_fixed_font_decision(text: str, decision: dict, *, fonts_root: Path | None = None) -> dict:
    expected = fixed_font_decision(text, fonts_root=fonts_root)
    if decision != expected:
        raise ValueError("FixedFontPolicyBindingMismatch")
    return expected

def apply_fixed_font_family(text_data: dict) -> bool:
    if str(text_data.get("style_origin") or "").lower() in {"manual","user_override"}:
        return False
    target = str(text_data.get("translated_payload") or text_data.get("translated") or text_data.get("traduzido") or "")
    decision = fixed_font_decision(target)
    for key in ("estilo","style","visual_profile"):
        style = text_data.get(key)
        if isinstance(style, dict):
            text_data[key] = {**style,"fonte":FIXED_FONT_NAME}
    if not isinstance(text_data.get("estilo"),dict):
        text_data["estilo"] = {"fonte":FIXED_FONT_NAME}
    if not isinstance(text_data.get("style"),dict):
        text_data["style"] = deepcopy(text_data["estilo"])
    text_data["font_family_policy_v1"] = decision
    return True
