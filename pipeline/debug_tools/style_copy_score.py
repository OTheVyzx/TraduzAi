"""Score synthetic and real style-copy evidence reports."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PIPELINE_DIR))

import cv2

from typesetter.style_extractor import IMPACT_FONT_CHOICES, extract_text_style_evidence


ROOT = PIPELINE_DIR.parent
DEFAULT_ATLAS_DIR = ROOT / "pipeline" / "tests" / "fixtures" / "style_copy_atlas"
DEFAULT_RUNTIME_LOCK = Path(__file__).with_name("style_copy_benchmark_runtime.lock.json")
FREETYPE_PROVIDER = "matplotlib.ft2font"
RUNTIME_KEYS = (
    "python",
    "pillow",
    "opencv",
    "freetype",
    "freetype_provider",
    "freetype_provider_version",
)

_MISSING_MODULE = object()
_DIRECT_FT2FONT_MODULE: Any | None = None
_FREETYPE_RUNTIME: tuple[str, str] | None = None


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


def _ft2font_extension_path(distribution: importlib.metadata.Distribution) -> Path:
    extension_names = {
        f"ft2font{suffix}" for suffix in importlib.machinery.EXTENSION_SUFFIXES
    }
    candidates = [
        file
        for file in distribution.files or ()
        if Path(str(file)).parent.as_posix() == "matplotlib"
        and Path(str(file)).name in extension_names
    ]
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            "cannot identify exactly one matplotlib.ft2font extension; "
            f"found: {rendered}"
        )
    return Path(distribution.locate_file(candidates[0])).resolve()


def _restore_module(name: str, previous: object) -> None:
    if previous is _MISSING_MODULE:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous  # type: ignore[assignment]


def _load_ft2font_without_matplotlib() -> tuple[Any, str]:
    global _DIRECT_FT2FONT_MODULE

    try:
        distribution = importlib.metadata.distribution("matplotlib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("matplotlib distribution is required for FreeType metadata") from exc

    loaded = sys.modules.get(FREETYPE_PROVIDER)
    if loaded is not None and getattr(loaded, "__freetype_version__", None):
        return loaded, distribution.version
    if _DIRECT_FT2FONT_MODULE is not None:
        return _DIRECT_FT2FONT_MODULE, distribution.version

    extension_path = _ft2font_extension_path(distribution)
    previous_parent = sys.modules.get("matplotlib", _MISSING_MODULE)
    previous_child = sys.modules.get(FREETYPE_PROVIDER, _MISSING_MODULE)
    if previous_parent is _MISSING_MODULE:
        dummy_package = ModuleType("matplotlib")
        dummy_package.__package__ = "matplotlib"
        dummy_package.__path__ = [str(extension_path.parent)]
        dummy_package.__spec__ = importlib.machinery.ModuleSpec(
            "matplotlib",
            loader=None,
            is_package=True,
        )
        sys.modules["matplotlib"] = dummy_package

    try:
        spec = importlib.util.spec_from_file_location(FREETYPE_PROVIDER, extension_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load FreeType provider extension: {extension_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[FREETYPE_PROVIDER] = module
        spec.loader.exec_module(module)
    finally:
        _restore_module(FREETYPE_PROVIDER, previous_child)
        _restore_module("matplotlib", previous_parent)

    _DIRECT_FT2FONT_MODULE = module
    return module, distribution.version


def _freetype_runtime_metadata() -> dict[str, str]:
    global _FREETYPE_RUNTIME

    if _FREETYPE_RUNTIME is None:
        ft2font, provider_version = _load_ft2font_without_matplotlib()
        freetype_version = str(getattr(ft2font, "__freetype_version__", "") or "")
        if not freetype_version or freetype_version == "unavailable":
            raise RuntimeError(
                f"{FREETYPE_PROVIDER} did not expose a usable __freetype_version__"
            )
        _FREETYPE_RUNTIME = freetype_version, str(provider_version)

    freetype_version, provider_version = _FREETYPE_RUNTIME
    return {
        "freetype": freetype_version,
        "freetype_provider": FREETYPE_PROVIDER,
        "freetype_provider_version": provider_version,
    }


def _runtime_metadata() -> dict[str, str]:
    try:
        pillow_version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError:
        pillow_version = "unavailable"
    runtime = {
        "python": platform.python_version(),
        "pillow": pillow_version,
        "opencv": cv2.__version__,
    }
    runtime.update(_freetype_runtime_metadata())
    return runtime


def _runtime_contract(lock_path: Path = DEFAULT_RUNTIME_LOCK) -> dict[str, str]:
    lock_path = Path(lock_path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"style-copy benchmark runtime lock not found: {lock_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid style-copy benchmark runtime lock: {lock_path}: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported style-copy benchmark runtime lock: {lock_path}")
    expected = payload.get("runtime")
    if not isinstance(expected, dict) or set(expected) != set(RUNTIME_KEYS):
        raise RuntimeError(
            f"style-copy benchmark runtime lock must define exactly: {', '.join(RUNTIME_KEYS)}"
        )
    if not all(isinstance(value, str) and value for value in expected.values()):
        raise RuntimeError("style-copy benchmark runtime lock values must be non-empty strings")
    return {key: expected[key] for key in RUNTIME_KEYS}


def validate_runtime_contract(
    runtime: dict[str, str],
    lock_path: Path = DEFAULT_RUNTIME_LOCK,
) -> None:
    expected = _runtime_contract(lock_path)
    mismatches = []
    for key in RUNTIME_KEYS:
        actual_value = runtime.get(key, "<missing>")
        expected_value = expected[key]
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            "style-copy benchmark runtime mismatch against "
            f"{Path(lock_path)}: {'; '.join(mismatches)}"
        )


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--records", type=Path, help="style_audit_records.jsonl from a real visual report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "runtime": _runtime_metadata(),
        "synthetic": score_synthetic(args.atlas_dir),
    }
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
