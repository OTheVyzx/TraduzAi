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

from debug_tools import generate_style_benchmark_v2, style_benchmark_report
from typesetter.style_contract import style_evidence_v2_from_v1
from typesetter.style_policy import style_evidence_v2_shadow_policy
from typesetter.style_extractor import extract_text_style_evidence


DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "style_benchmark_v2" / "benchmark_spec.json"


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
    records = _measure_current_engine(run_dir, manifest)
    style_benchmark_report.write_run_reports(run_dir, manifest, records)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--level", choices=("all", *generate_style_benchmark_v2.REQUIRED_LEVELS), default="smoke")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-lock", type=Path, default=generate_style_benchmark_v2.DEFAULT_RUNTIME_LOCK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        run_benchmark(
            spec_path=args.spec,
            level=args.level,
            output_root=args.output_root,
            run_id=args.run_id,
            seed=args.seed,
            runtime_lock_path=args.runtime_lock,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
