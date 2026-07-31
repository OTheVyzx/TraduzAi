"""Owner-scoped source-to-final style fidelity audit and rollout gate."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any


STYLE_MODES = frozenset({"shadow", "render", "enforce"})


def _hex_rgb(value: Any) -> tuple[int, int, int] | None:
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return None


def _rgb_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    values = []
    for channel in rgb:
        value = channel / 255.0
        values.append(((value + 0.055) / 1.055) ** 2.4 if value > 0.04045 else value / 12.92)
    r, g, b = values
    x, y, z = (r * .4124 + g * .3576 + b * .1805) / .95047, (r * .2126 + g * .7152 + b * .0722), (r * .0193 + g * .1192 + b * .9505) / 1.08883
    f = lambda value: value ** (1 / 3) if value > .008856 else 7.787 * value + 16 / 116
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e_2000(first: Any, second: Any) -> float | None:
    """Return CIEDE2000 distance for two #RRGGBB values."""
    rgb1, rgb2 = _hex_rgb(first), _hex_rgb(second)
    if rgb1 is None or rgb2 is None:
        return None
    l1, a1, b1 = _rgb_lab(rgb1); l2, a2, b2 = _rgb_lab(rgb2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    mean_c = (c1 + c2) / 2; g = .5 * (1 - math.sqrt(mean_c ** 7 / (mean_c ** 7 + 25 ** 7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = math.hypot(ap1, b1), math.hypot(ap2, b2)
    hp = lambda a, b: (math.degrees(math.atan2(b, a)) + 360) % 360 if a or b else 0.0
    hp1, hp2 = hp(ap1, b1), hp(ap2, b2)
    dl, dc = l2 - l1, cp2 - cp1
    dh_raw = hp2 - hp1
    dh = dh_raw - 360 if dh_raw > 180 else dh_raw + 360 if dh_raw < -180 else dh_raw
    dh_term = 2 * math.sqrt(cp1 * cp2) * math.sin(math.radians(dh / 2))
    mean_l, mean_cp = (l1 + l2) / 2, (cp1 + cp2) / 2
    mean_h = hp1 + hp2 if cp1 * cp2 == 0 else (hp1 + hp2 + (360 if abs(hp1 - hp2) > 180 and hp1 + hp2 < 360 else -360 if abs(hp1 - hp2) > 180 else 0)) / 2
    t = 1 - .17 * math.cos(math.radians(mean_h - 30)) + .24 * math.cos(math.radians(2 * mean_h)) + .32 * math.cos(math.radians(3 * mean_h + 6)) - .20 * math.cos(math.radians(4 * mean_h - 63))
    sl = 1 + .015 * (mean_l - 50) ** 2 / math.sqrt(20 + (mean_l - 50) ** 2)
    sc, sh = 1 + .045 * mean_cp, 1 + .015 * mean_cp * t
    rt = -2 * math.sqrt(mean_cp ** 7 / (mean_cp ** 7 + 25 ** 7)) * math.sin(math.radians(60 * math.exp(-((mean_h - 275) / 25) ** 2)))
    return round(math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh_term / sh) ** 2 + rt * (dc / sc) * (dh_term / sh)), 4)


def resolve_original_path(run_dir: Path, page: dict[str, Any], page_number: int) -> Path | None:
    root = Path(run_dir).resolve()
    for key in ("original_path", "source_path", "image_path", "arquivo", "path"):
        raw = page.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        candidate = Path(raw)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file():
            return candidate.resolve()
    for folder in ("originals", "images"):
        for stem in (f"{page_number:03d}", f"page_{page_number:03d}", str(page_number)):
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = root / folder / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate.resolve()
    return None


def _same_value(expected: Any, observed: Any) -> bool:
    if isinstance(expected, str) and isinstance(observed, str):
        return expected.strip().upper() == observed.strip().upper()
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(expected) - float(observed)) <= max(0.05, abs(float(expected)) * 0.08)
    return expected == observed


def _attribute_confidence(profile: dict[str, Any], name: str) -> float:
    evidence = profile.get("style_evidence_v2") if isinstance(profile.get("style_evidence_v2"), dict) else {}
    attributes = evidence.get("attributes") if isinstance(evidence.get("attributes"), dict) else {}
    item = attributes.get(name) if isinstance(attributes.get(name), dict) else {}
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def audit_style_fidelity(project: dict[str, Any], run_dir: Path, *, mode: str = "shadow") -> dict[str, Any]:
    rollout = str(mode or "shadow").strip().lower()
    if rollout not in STYLE_MODES:
        raise ValueError(f"unsupported style fidelity mode: {mode}")
    pages = project.get("paginas") or project.get("pages") or []
    owners: list[dict[str, Any]] = []
    blocking: list[str] = []
    for page_number, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            continue
        source_path = resolve_original_path(Path(run_dir), page, page_number)
        for layer in page.get("text_layers") or page.get("texts") or []:
            if not isinstance(layer, dict):
                continue
            owner_id = str(layer.get("owner_id") or "").strip()
            profile = layer.get("visual_profile_v2")
            if not owner_id or not isinstance(profile, dict) or str(profile.get("owner_id") or "") != owner_id:
                continue
            expected = profile.get("applied_style") if isinstance(profile.get("applied_style"), dict) else {}
            raster = layer.get("style_v2_raster_contract") if isinstance(layer.get("style_v2_raster_contract"), dict) else {}
            observed = raster.get("applied_attributes") if isinstance(raster.get("applied_attributes"), dict) else {}
            profile_status = str(profile.get("status") or "not_scanned")
            fields: dict[str, Any] = {}
            owner_blocking = False
            for name, value in expected.items():
                confidence = _attribute_confidence(profile, name)
                if value in (None, "", [], {}):
                    status = "not_applicable"
                elif profile_status == "fallback":
                    status = "fallback"
                elif name not in observed:
                    status = "mismatch"
                else:
                    status = "applied" if _same_value(value, observed[name]) else "mismatch"
                catastrophic = status == "mismatch" and name in {"font_name", "fill", "stroke", "gradient", "glow"}
                high_confidence_failure = status == "mismatch" and (confidence >= 0.80 or catastrophic)
                owner_blocking = owner_blocking or high_confidence_failure
                fields[name] = {
                    "status": status, "expected": copy.deepcopy(value),
                    "observed": copy.deepcopy(observed.get(name)), "confidence": confidence,
                    "high_confidence_failure": high_confidence_failure,
                }
                color_distance = delta_e_2000(value, observed.get(name))
                if color_distance is not None:
                    fields[name]["delta_e_2000"] = color_distance
            if rollout == "enforce" and owner_blocking:
                blocking.append(owner_id)
            owners.append({
                "page": page_number, "owner_id": owner_id,
                "source_path": str(source_path) if source_path else None,
                "source_sha256": profile.get("source_sha256"),
                "glyph_mask_sha256": profile.get("glyph_mask_sha256"),
                "visual_profile_sha256": profile.get("visual_profile_sha256"),
                "profile_status": profile_status, "fields": fields,
            })
    return {
        "schema_version": 1, "mode": rollout, "owners": owners,
        "summary": {"owner_count": len(owners), "blocking_owner_count": len(set(blocking))},
        "gate": {"status": "BLOCK" if blocking else "PASS", "blocking_owner_ids": sorted(set(blocking))},
    }


def merge_style_and_functional_gates(functional_gate: dict[str, Any], style_gate: dict[str, Any]) -> dict[str, Any]:
    functional = copy.deepcopy(functional_gate or {"status": "PASS"})
    style = copy.deepcopy(style_gate or {"status": "PASS"})
    blocked = str(functional.get("status") or "PASS").upper() == "BLOCK" or str(style.get("status") or "PASS").upper() == "BLOCK"
    return {"status": "BLOCK" if blocked else "PASS", "functional_gate": functional, "style_gate": style}
