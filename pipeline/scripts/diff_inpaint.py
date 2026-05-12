"""Compare visual inpaint bands against a baseline with SSIM.

Usage:
    python pipeline/scripts/diff_inpaint.py --baseline pipeline/tests/visual/baseline_inpaint --candidate pipeline/scratch/current_inpaint
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _iter_images(path: Path) -> Iterable[Path]:
    for candidate in sorted(path.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate


def _load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float64)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b

    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    score = numerator / np.maximum(denominator, 1e-9)
    return float(np.mean(score))


def _write_diff_image(baseline: np.ndarray, candidate: np.ndarray, output_path: Path) -> None:
    if baseline.shape != candidate.shape:
        candidate = cv2.resize(candidate, (baseline.shape[1], baseline.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(baseline.astype(np.uint8), candidate.astype(np.uint8))
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), heat)


def compare_dirs(baseline_dir: Path, candidate_dir: Path, threshold: float, diff_dir: Path | None) -> dict:
    baseline_files = list(_iter_images(baseline_dir))
    candidate_by_name = {path.name: path for path in _iter_images(candidate_dir)}
    results = []

    for baseline_path in baseline_files:
        candidate_path = candidate_by_name.get(baseline_path.name)
        if candidate_path is None:
            results.append(
                {
                    "name": baseline_path.name,
                    "status": "missing_candidate",
                    "ssim": 0.0,
                    "passed": False,
                }
            )
            continue

        baseline = _load_gray(baseline_path)
        candidate = _load_gray(candidate_path)
        score = _ssim(baseline, candidate)
        passed = score >= threshold
        row = {
            "name": baseline_path.name,
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "ssim": round(score, 6),
            "passed": passed,
        }
        if diff_dir is not None and not passed:
            diff_path = diff_dir / f"{baseline_path.stem}_diff.png"
            _write_diff_image(baseline, candidate, diff_path)
            row["diff"] = str(diff_path)
        results.append(row)

    passed_count = sum(1 for item in results if item["passed"])
    return {
        "threshold": threshold,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare inpaint baseline images with SSIM.")
    parser.add_argument("--baseline", required=True, type=Path, help="Directory with approved baseline bands.")
    parser.add_argument("--candidate", required=True, type=Path, help="Directory with candidate bands.")
    parser.add_argument("--threshold", type=float, default=0.97, help="Minimum SSIM per image.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--diff-dir", type=Path, default=None, help="Optional directory for failed diff heatmaps.")
    args = parser.parse_args(argv)

    if not args.baseline.is_dir():
        parser.error(f"baseline directory not found: {args.baseline}")
    if not args.candidate.is_dir():
        parser.error(f"candidate directory not found: {args.candidate}")

    report = compare_dirs(args.baseline, args.candidate, args.threshold, args.diff_dir)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
