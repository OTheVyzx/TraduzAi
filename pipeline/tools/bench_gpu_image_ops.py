from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import cv2
import numpy as np

from vision_stack.gpu_image_ops import (
    apply_white_fill,
    connected_components_with_stats,
    expand_mask,
    probe_gpu_image_ops,
    resize_crops_batch,
)


def _bench(fn, *, repeats: int, warmup: int) -> dict:
    for _ in range(max(0, warmup)):
        fn()
    samples: list[float] = []
    last = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        last = fn()
        samples.append(time.perf_counter() - started)
    return {
        "mean_ms": round(statistics.mean(samples) * 1000.0, 3),
        "min_ms": round(min(samples) * 1000.0, 3),
        "max_ms": round(max(samples) * 1000.0, 3),
        "result_type": type(last).__name__,
    }


def _synthetic_inputs(size: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    step = max(32, size // 16)
    for y in range(step, size - step, step * 2):
        for x in range(step, size - step, step * 2):
            cv2.ellipse(mask, (x, y), (step // 2, step // 4), 0, 0, 360, 255, -1)
            cv2.putText(image, "TXT", (x - step // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    crops = [
        image[y : y + step, x : x + step].copy()
        for y in range(0, min(size - step, step * 12), step)
        for x in range(0, min(size - step, step * 12), step)
    ][:64]
    return image, mask, crops


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark experimental GPU image ops.")
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "torch", "cv2cuda"])
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    image, mask, crops = _synthetic_inputs(max(128, args.size))
    report = {
        "probe": probe_gpu_image_ops(),
        "backend_requested": args.backend,
        "size": args.size,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "benchmarks": {
            "white_fill": _bench(lambda: apply_white_fill(image, mask, backend=args.backend), repeats=args.repeats, warmup=args.warmup),
            "expand_mask": _bench(lambda: expand_mask(mask, kernel_size=5, iterations=2, backend=args.backend), repeats=args.repeats, warmup=args.warmup),
            "connected_components": _bench(lambda: connected_components_with_stats(mask, backend=args.backend), repeats=args.repeats, warmup=args.warmup),
            "resize_crops_batch": _bench(lambda: resize_crops_batch(crops, size=(96, 96), backend=args.backend), repeats=args.repeats, warmup=args.warmup),
        },
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
