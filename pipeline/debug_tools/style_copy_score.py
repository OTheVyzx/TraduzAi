"""Score synthetic and real style-copy evidence reports."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2

from typesetter.style_extractor import IMPACT_FONT_CHOICES, extract_text_style_evidence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATLAS_DIR = ROOT / "pipeline" / "tests" / "fixtures" / "style_copy_atlas"


REAL_CH3_EXPECTATIONS = [
    {
        "id": "p05_large_light_contour",
        "page": 5,
        "kind": "large_light_contour",
    },
    {
        "id": "p06_keeps_solid_dark_text_without_gradient",
        "page": 6,
        "kind": "solid_dark_text",
    },
    {
        "id": "p07_has_dark_text_gradient",
        "page": 7,
        "kind": "dark_text_gradient",
    },
    {
        "id": "p08_has_dark_text_gradient",
        "page": 8,
        "kind": "dark_text_gradient",
    },
    {
        "id": "p02_has_cyan_card_glow",
        "page": 2,
        "kind": "cyan_card_glow",
    },
]


def _load_generator(atlas_dir: Path):
    generator_path = atlas_dir / "generate_style_copy_atlas.py"
    spec = importlib.util.spec_from_file_location("style_copy_atlas_generator", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load atlas generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_expected(case_id: str, expected: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    expected_font = expected.get("font_name")
    if expected_font and evidence.get("font_name") != expected_font:
        failures.append(f"font {evidence.get('font_name')!r} != {expected_font!r}")
    if expected.get("font_family") == "impact" and evidence.get("font_name") not in IMPACT_FONT_CHOICES:
        failures.append(f"font {evidence.get('font_name')!r} is not impact")

    expected_color = expected.get("text_color")
    if expected_color and str(evidence.get("text_color") or "").upper() != expected_color.upper():
        failures.append(f"text_color {evidence.get('text_color')} != {expected_color}")

    expected_stroke_color = expected.get("stroke_color")
    if expected_stroke_color and str(evidence.get("stroke_color") or "").upper() != expected_stroke_color.upper():
        failures.append(f"stroke_color {evidence.get('stroke_color')} != {expected_stroke_color}")

    stroke_min = expected.get("stroke_width_px_min")
    if stroke_min is not None and int(evidence.get("stroke_width_px") or 0) < int(stroke_min):
        failures.append(f"stroke_width {evidence.get('stroke_width_px')} < {stroke_min}")

    if expected.get("stroke") is False and evidence.get("stroke_color"):
        failures.append(f"unexpected stroke {evidence.get('stroke_color')}:{evidence.get('stroke_width_px')}")
    if expected.get("stroke") is True and not evidence.get("stroke_color"):
        failures.append("expected stroke")

    for key in ("gradient", "glow", "shadow"):
        if expected.get(key) is True and evidence.get(key) is not True:
            failures.append(f"expected {key}")
        if expected.get(key) is False and evidence.get(key) is True:
            failures.append(f"unexpected {key}")

    if evidence.get("curved"):
        failures.append("unexpected curve")
    return [f"{case_id}: {failure}" for failure in failures]


def score_synthetic(atlas_dir: Path = DEFAULT_ATLAS_DIR) -> dict[str, Any]:
    generator = _load_generator(atlas_dir)
    generator.generate()
    manifest = json.loads((atlas_dir / "style_copy_manifest.json").read_text(encoding="utf-8"))
    atlas_bgr = cv2.imread(str(atlas_dir / manifest["image"]), cv2.IMREAD_COLOR)
    if atlas_bgr is None:
        raise FileNotFoundError(atlas_dir / manifest["image"])
    atlas_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)

    failures: list[str] = []
    passed = 0
    cases = manifest.get("cases") or []
    for case in cases:
        x1, y1, x2, y2 = [int(v) for v in case["bbox"]]
        evidence = extract_text_style_evidence(atlas_rgb[y1:y2, x1:x2, :3]).to_dict()
        case_failures = _score_expected(str(case["id"]), case.get("expected") or {}, evidence)
        if case_failures:
            failures.extend(case_failures)
        else:
            passed += 1

    return {
        "cases": len(cases),
        "passed": passed,
        "failed": failures,
        "pass_rate": round(passed / max(1, len(cases)), 4),
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _hex_luma(value: str) -> float:
    value = str(value or "").lstrip("#")
    if len(value) < 6:
        return 0.0
    try:
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return 0.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _bbox_size(rec: dict[str, Any]) -> tuple[int, int]:
    bbox = rec.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) < 4:
        return 0, 0
    try:
        return max(0, int(bbox[2]) - int(bbox[0])), max(0, int(bbox[3]) - int(bbox[1]))
    except (TypeError, ValueError):
        return 0, 0


def _has_applied_fields(rec: dict[str, Any]) -> bool:
    return any(str(key).startswith("applied_") for key in rec)


def _style_value(rec: dict[str, Any], applied_key: str, detected_key: str, default: Any = None) -> Any:
    if _has_applied_fields(rec):
        return rec.get(applied_key, default)
    return rec.get(detected_key, default)


def _matches_real_kind(rec: dict[str, Any], kind: str) -> bool:
    fill = str(_style_value(rec, "applied_text_color", "text_color", "") or "")
    stroke = str(_style_value(rec, "applied_stroke_color", "stroke_color", "") or "")
    glow = bool(_style_value(rec, "applied_glow", "glow", False))
    gradient = bool(_style_value(rec, "applied_gradient", "gradient", False))
    gradient_colors = _style_value(rec, "applied_gradient_colors", "gradient_colors", []) or []
    width, height = _bbox_size(rec)

    if kind == "large_light_contour":
        return (
            bool(stroke)
            and glow is not True
            and width >= 250
            and height >= 70
            and _hex_luma(fill) <= 45.0
            and _hex_luma(stroke) >= 210.0
        )

    if kind == "dark_text_gradient":
        return (
            gradient is True
            and glow is not True
            and len(gradient_colors) >= 2
            and _hex_luma(fill) <= 95.0
        )

    if kind == "solid_dark_text":
        return (
            gradient is not True
            and glow is not True
            and not stroke
            and width >= 180
            and height >= 60
            and _hex_luma(fill) <= 45.0
        )

    if kind == "cyan_card_glow":
        return (
            glow is True
            and not stroke
            and _hex_luma(fill) >= 210.0
        )

    return False


def score_real_records(records_path: Path) -> dict[str, Any]:
    records = _read_records(records_path)
    failed: list[dict[str, Any]] = []
    passed = 0
    for item in REAL_CH3_EXPECTATIONS:
        page_records = [rec for rec in records if int(rec.get("page") or 0) == int(item["page"])]
        matches = [rec for rec in page_records if _matches_real_kind(rec, str(item["kind"]))]
        if matches:
            passed += 1
        else:
            failed.append(
                {
                    "id": item["id"],
                    "reason": f"no page {item['page']} record matched {item['kind']}",
                    "page_record_count": len(page_records),
                }
            )

    counts = {
        "records": len(records),
        "stroke": sum(1 for rec in records if _style_value(rec, "applied_stroke_color", "stroke_color")),
        "glow": sum(1 for rec in records if _style_value(rec, "applied_glow", "glow", False)),
        "gradient": sum(1 for rec in records if _style_value(rec, "applied_gradient", "gradient", False)),
        "curved": sum(1 for rec in records if rec.get("curved")),
    }
    return {
        "checked": len(REAL_CH3_EXPECTATIONS),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / max(1, len(REAL_CH3_EXPECTATIONS)), 4),
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--records", type=Path, help="style_audit_records.jsonl from a real visual report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {"synthetic": score_synthetic(args.atlas_dir)}
    if args.records:
        result["real_ch3"] = score_real_records(args.records)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
