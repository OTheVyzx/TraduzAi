"""Generate and measure an isolated Style Atlas v2 benchmark run."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab

from debug_tools import generate_style_benchmark_v2, style_benchmark_report
from typesetter.style_contract import style_evidence_v2_from_v1
from typesetter.style_policy import style_evidence_v2_shadow_policy
from typesetter.style_extractor import extract_text_style_evidence
from typesetter.style_extractor import extract_text_style_evidence_v2
from typesetter.font_matcher import FontShapeMatcher, load_font_catalog


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "style_benchmark_v2" / "benchmark_spec.json"
FONT_DIR = Path(__file__).resolve().parents[2] / "fonts"
FONT_MAP_PATH = FONT_DIR / "font-map.json"
BENCHMARK_FONT_NAMES = (
    "ComicNeue-Bold.ttf",
    "LeagueGothic-Regular-VariableFont_wdth.ttf",
)


def _detected_style_payload(image_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    evidence = extract_text_style_evidence(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).to_dict()
    evidence_v2 = style_evidence_v2_from_v1(evidence)
    stroke_color = evidence.get("stroke_color")
    return {
        "font_name": {"value": evidence.get("font_name", "unknown"), "confidence": evidence.get("font_confidence")},
        "font_weight": {"value": "unknown"},
        "font_width": {"value": "unknown"},
        "font_size_px": {"value": "unknown"},
        "alignment": {"value": "unknown"},
        "fill": {"value": evidence.get("text_color", "unknown"), "confidence": evidence.get("text_color_confidence")},
        "stroke": {
            "value": {"color": stroke_color, "width_px": evidence.get("stroke_width_px")}
            if stroke_color
            else None,
        },
        "shadow": {"value": evidence.get("shadow", "unknown")},
        "gradient": {"value": evidence.get("gradient", "unknown")},
        "rotation_deg": {"value": "unknown"},
        "container": {"value": "unknown"},
    }, evidence, evidence_v2.to_dict()


def _measure_current_engine(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for case in manifest["cases"]:
        for variant, text_key, image_key in (("a", "text_a", "image_a"), ("b", "text_b", "image_b")):
            attributes, evidence_v1, evidence_v2 = _detected_style_payload(run_dir / case[image_key])
            records.append(
                {
                    "attributes": attributes,
                    "case_id": case["id"],
                    "detector": "typesetter.style_extractor.current",
                    "image": case[image_key],
                    "level": case["level"],
                    "seed": case["seed"],
                    "source_text": case[text_key],
                    "style_evidence_v1": evidence_v1,
                    "style_evidence_v2": evidence_v2,
                    "style_evidence_v2_shadow_policy": style_evidence_v2_shadow_policy(
                        style_evidence_v2_from_v1(evidence_v1)
                    ),
                    "style_spec": {key: value for key, value in case.items() if key not in {"image_a", "image_b"}},
                    "threshold": None,
                    "variant": variant,
                }
            )
    return records


def _hex_rgb(value: str) -> np.ndarray:
    token = str(value).strip().lstrip("#")
    if len(token) != 6:
        raise ValueError(f"invalid benchmark color: {value!r}")
    return np.asarray([int(token[index:index + 2], 16) for index in (0, 2, 4)], dtype=np.uint8)


def _fill_delta_e(image_rgb: np.ndarray, core_mask: np.ndarray, expected_fill: str) -> float | None:
    pixels = image_rgb[np.asarray(core_mask) > 0]
    if len(pixels) < 8:
        return None
    observed = np.median(pixels.astype(np.float32), axis=0).round().astype(np.uint8)
    pair = np.stack((observed, _hex_rgb(expected_fill)), axis=0).reshape(1, 2, 3).astype(np.float32) / 255.0
    lab = rgb2lab(pair)
    return round(float(deltaE_ciede2000(lab[:, :1], lab[:, 1:])[0, 0]), 6)


def _measure_mask_backed_v2(run_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = load_font_catalog(FONT_DIR, FONT_MAP_PATH)
    matcher = FontShapeMatcher(catalog)
    records: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        for variant in ("a", "b"):
            image = cv2.imread(str(run_dir / case[f"image_{variant}"]), cv2.IMREAD_COLOR)
            core = cv2.imread(str(run_dir / case[f"glyph_core_mask_{variant}"]), cv2.IMREAD_GRAYSCALE)
            effect = cv2.imread(str(run_dir / case[f"effect_mask_{variant}"]), cv2.IMREAD_GRAYSCALE)
            safe = cv2.imread(str(run_dir / case["safe_mask"]), cv2.IMREAD_GRAYSCALE)
            if image is None or core is None or effect is None or safe is None:
                raise FileNotFoundError(f"incomplete mask-backed case: {case['id']}:{variant}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text = str(case[f"text_{variant}"])
            if not np.any(core):
                records.append({
                    "case_id": case["id"], "variant": variant, "source_text": text,
                    "level": case["level"], "owner_id": case["owner_id"],
                    "detector": "v2_mask_backed", "abstained": True,
                })
                continue
            effect_only = cv2.bitwise_and(effect, cv2.bitwise_not(core))
            exclusion = cv2.dilate(effect, np.ones((5, 5), np.uint8), iterations=1)
            context = cv2.bitwise_and(safe, cv2.bitwise_not(exclusion))
            evidence = extract_text_style_evidence_v2(
                image_rgb, core, context,
                stroke_ring_mask=effect_only,
                effect_region_mask=effect_only,
                owner_id=str(case["owner_id"]), semantic_role=str(case["semantic_role"]),
                source_phase="synthetic_source",
            )
            profile = {
                "rotation_deg": float(case["rotation_deg"]),
                "width_scale": float(case["font_width"]) / 100.0,
                "scale_y": 1.0,
                "slant_tangent": 0.0,
            }
            font_match = matcher.match(
                core, source_text=text, profile=profile,
                semantic_role=str(case["semantic_role"]), shortlist=BENCHMARK_FONT_NAMES,
            ).to_dict()
            top_k = [str(item["font_name"]) for item in font_match["top_k"]]
            evidence_dict = evidence.to_dict()
            records.append({
                "case_id": case["id"], "variant": variant, "source_text": text,
                "level": case["level"], "owner_id": case["owner_id"],
                "detector": "v2_mask_backed", "abstained": False,
                "style_evidence_v2": evidence_dict,
                "attributes": evidence_dict["attributes"],
                "font_top_k": top_k,
                "fill_delta_e_2000": None if case.get("gradient") else _fill_delta_e(image_rgb, core, case["fill"]),
                "geometry": evidence_dict.get("attribute_provenance", {}).get("typographic_metrics", {}),
            })
    return records


def run_benchmark(
    *,
    spec_path: Path,
    level: str,
    output_root: Path,
    run_id: str,
    seed: int,
    runtime_lock_path: Path = generate_style_benchmark_v2.DEFAULT_RUNTIME_LOCK,
) -> Path:
    """Generate a run and measure the current engine without changing its behavior."""
    run_dir = generate_style_benchmark_v2.generate_benchmark(
        spec_path=spec_path,
        level=level,
        output_root=output_root,
        run_id=run_id,
        seed=seed,
        runtime_lock_path=runtime_lock_path,
    )
    manifest = json.loads((run_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8-sig"))
    records = _measure_mask_backed_v2(run_dir, manifest)
    legacy_records = _measure_current_engine(run_dir, manifest)
    style_benchmark_report.write_run_reports(
        run_dir,
        manifest,
        records,
        legacy_records=legacy_records,
        validation_thresholds=spec.get("validation_thresholds"),
    )
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--level", choices=("all", *generate_style_benchmark_v2.REQUIRED_LEVELS), default="smoke")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-lock", type=Path, default=generate_style_benchmark_v2.DEFAULT_RUNTIME_LOCK)
    parser.add_argument("--mode", choices=("shadow", "enforce"), default="shadow")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = run_benchmark(
            spec_path=args.spec,
            level=args.level,
            output_root=args.output_root,
            run_id=args.run_id,
            seed=args.seed,
            runtime_lock_path=args.runtime_lock,
        )
    print(run_dir)
    summary = json.loads((run_dir / "style_benchmark_summary.json").read_text(encoding="utf-8"))
    blocked = str((summary.get("validation") or {}).get("status") or "BLOCK").upper() == "BLOCK"
    return 2 if args.mode == "enforce" and blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
