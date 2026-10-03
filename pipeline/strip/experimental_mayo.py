"""Opt-in Mayo balloon proposals for a single local experiment.

Masks are evidence for candidate geometry only.  They never authorize erasing
the balloon, and the ordinary OCR, owner and pixel gates remain authoritative.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

WEIGHT_SHA256 = "c881d96771755fa628a94bb5f4b18301a0728ae4ffe8f14b2e9dde55e1b40552"
MODEL_ID = "mayocream/speech-bubble-segmentation@387bc1e93f3d24702bc8609798b6a13b37420edc"
_model = None


def enabled() -> bool:
    return os.getenv("TRADUZAI_EXPERIMENTAL_MAYO", "0").strip().lower() in {"1", "true", "yes", "on"}


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = _area(a) + _area(b) - intersection
    return intersection / union if union else 0.0


def _intersection_over_smaller(a: dict, b: dict) -> float:
    ab, bb = a["mask_bbox"], b["mask_bbox"]
    x1, y1 = max(ab[0], bb[0]), max(ab[1], bb[1])
    x2, y2 = min(ab[2], bb[2]), min(ab[3], bb[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    am = a["mask"][y1-ab[1]:y2-ab[1], x1-ab[0]:x2-ab[0]]
    bm = b["mask"][y1-bb[1]:y2-bb[1], x1-bb[0]:x2-bb[0]]
    smaller = min(int(a["mask"].sum()), int(b["mask"].sum()))
    return float(np.logical_and(am, bm).sum()) / smaller if smaller else 0.0


def deduplicate_tiles(candidates: list[dict]) -> tuple[list[dict], int]:
    """Keep the larger nearly contained cross-tile mask (p20 failure mode)."""
    kept: list[dict] = []
    larger_preserved = 0
    for candidate in sorted(candidates, key=lambda row: -float(row["confidence"])):
        duplicate_at = next((i for i, row in enumerate(kept)
                             if _iou(candidate["box"], row["box"]) >= 0.5
                             or (candidate["tile_y"] != row["tile_y"]
                                 and _intersection_over_smaller(candidate, row) >= 0.90)), None)
        if duplicate_at is None:
            kept.append(candidate)
            continue
        prior = kept[duplicate_at]
        contained_cross_tile = (candidate["tile_y"] != prior["tile_y"]
                                and _iou(candidate["box"], prior["box"]) < 0.5
                                and _intersection_over_smaller(candidate, prior) >= 0.90)
        if contained_cross_tile and int(candidate["mask"].sum()) > int(prior["mask"].sum()):
            candidate["repair"] = "larger_cross_tile_mask"
            kept[duplicate_at] = candidate
            larger_preserved += 1
    return kept, larger_preserved


def _closed_white_interior(image_rgb: np.ndarray, pair: tuple[dict, dict]) -> np.ndarray | None:
    """Conservative automatic contour gate from the p14 pilot."""
    height, width = image_rgb.shape[:2]
    boxes = [row["box"] for row in pair]
    left, top = min(box[0] for box in boxes), min(box[1] for box in boxes)
    right, bottom = max(box[2] for box in boxes), max(box[3] for box in boxes)
    px, py = max(32, math.ceil((right-left)*0.08)), max(32, math.ceil((bottom-top)*0.30))
    x1, y1 = max(0, math.floor(left)-px), max(0, math.floor(top)-py)
    x2, y2 = min(width, math.ceil(right)+px), min(height, math.ceil(bottom)+py)
    crop = image_rgb[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    chroma = crop.max(2).astype(np.int16) - crop.min(2).astype(np.int16)
    bright = ((gray >= 225) & (chroma <= 28)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    best = None
    for label in range(1, count):
        component = labels == label
        shares = []
        for row in pair:
            bbox = row["mask_bbox"]
            ax1, ay1, ax2, ay2 = max(x1,bbox[0]), max(y1,bbox[1]), min(x2,bbox[2]), min(y2,bbox[3])
            overlap = 0
            if ax2 > ax1 and ay2 > ay1:
                overlap = int(np.logical_and(component[ay1-y1:ay2-y1, ax1-x1:ax2-x1],
                                             row["mask"][ay1-bbox[1]:ay2-bbox[1], ax1-bbox[0]:ax2-bbox[0]]).sum())
            shares.append(overlap / max(1, int(row["mask"].sum())))
        score = min(shares)
        if score >= 0.50 and (best is None or score > best[0]):
            best = (score, label)
    if best is None:
        return None
    cx, cy, cw, ch, area = map(int, stats[best[1]])
    if area < 100 or cx == 0 or cy == 0 or cx+cw >= crop.shape[1] or cy+ch >= crop.shape[0]:
        return None
    contours, _ = cv2.findContours((labels == best[1]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    result = np.zeros((height, width), np.uint8)
    cv2.drawContours(result[y1:y2,x1:x2], contours, 0, 1, cv2.FILLED)
    return result.astype(bool)


def merge_intratile_pairs(image_rgb: np.ndarray, candidates: list[dict]) -> tuple[list[dict], int]:
    """Union only paired proposals inside a closed bright region."""
    remaining = list(candidates)
    merged = 0
    for first in list(candidates):
        if not any(row is first for row in remaining):
            continue
        for second in list(remaining):
            if first is second or first["tile_y"] != second["tile_y"]:
                continue
            box_iou = _iou(first["box"], second["box"])
            if not 0.45 <= box_iou < 0.50 or _intersection_over_smaller(first, second) < 0.50:
                continue
            interior = _closed_white_interior(image_rgb, (first, second))
            if interior is None:
                continue
            x1 = min(first["mask_bbox"][0], second["mask_bbox"][0])
            y1 = min(first["mask_bbox"][1], second["mask_bbox"][1])
            x2 = max(first["mask_bbox"][2], second["mask_bbox"][2])
            y2 = max(first["mask_bbox"][3], second["mask_bbox"][3])
            union = np.zeros((y2-y1,x2-x1), bool)
            for row in (first, second):
                b = row["mask_bbox"]
                union[b[1]-y1:b[3]-y1,b[0]-x1:b[2]-x1] |= row["mask"]
            union &= interior[y1:y2,x1:x2]
            if not union.any():
                continue
            ys, xs = np.nonzero(union)
            repaired = dict(first)
            repaired.update(mask=union, mask_bbox=[x1,y1,x2,y2],
                            box=[int(xs.min()+x1),int(ys.min()+y1),int(xs.max()+x1+1),int(ys.max()+y1+1)],
                            repair="closed_white_intratile_union",
                            confidence=max(first["confidence"],second["confidence"]))
            remaining = [row for row in remaining if row is not first and row is not second]
            remaining.append(repaired)
            merged += 1
            break
    return remaining, merged


def _load_model():
    global _model
    if _model is not None:
        return _model
    model_dir = Path(os.environ["TRADUZAI_EXPERIMENTAL_MAYO_MODEL_DIR"])
    weights = model_dir / "model.safetensors"
    yaml_path = model_dir / "yolov8m-seg-local.yaml"
    if not weights.is_file() or not yaml_path.is_file():
        raise FileNotFoundError("Mayo local SafeTensors/config missing")
    with weights.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != WEIGHT_SHA256:
        raise ValueError("Mayo model weight hash mismatch")
    os.environ.setdefault("YOLO_OFFLINE", "True")
    os.environ.setdefault("ULTRALYTICS_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from safetensors.torch import load_file
    from ultralytics import YOLO
    wrapper = YOLO(str(yaml_path), task="segment")
    state = wrapper.model.state_dict()
    safe = load_file(str(weights), device="cpu")
    missing = set(state) - set(safe)
    if any(not name.endswith(".num_batches_tracked") for name in missing) or set(safe)-set(state):
        raise ValueError("Mayo checkpoint keys differ from architecture")
    if any(tuple(state[name].shape) != tuple(safe[name].shape) for name in safe):
        raise ValueError("Mayo checkpoint shape mismatch")
    state.update(safe)
    wrapper.model.load_state_dict(state, strict=True)
    wrapper.model.names = {0: "speech bubble"}
    wrapper.model.eval()
    _model = wrapper
    return wrapper


def detect(image_rgb: np.ndarray) -> tuple[list[dict], dict]:
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Mayo expects an RGB uint8 page")
    started = time.perf_counter()
    model = _load_model()
    height, width = image_rgb.shape[:2]
    device = os.getenv("TRADUZAI_EXPERIMENTAL_MAYO_DEVICE", "cpu")
    candidates = []
    tile_count = 0
    tile_starts = [0] if height <= 1400 else range(0, height, 1200)
    for y0 in tile_starts:
        visible = image_rgb[y0:min(height,y0+1400)]
        if visible.size == 0:
            break
        tile = np.full((1400,width,3),114,np.uint8)
        tile[:visible.shape[0]] = visible
        result = model.predict(source=tile[:,:,::-1], imgsz=640, conf=0.15, iou=0.50,
                               device=device, batch=1, retina_masks=True, verbose=False, save=False)[0]
        tile_count += 1
        if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        masks = result.masks.data.detach().cpu().numpy() > 0
        for box, score, mask in zip(boxes,scores,masks):
            mask = mask[:visible.shape[0]]
            if not mask.any():
                continue
            ys,xs = np.nonzero(mask)
            x1,y1,x2,y2 = int(xs.min()),int(ys.min()+y0),int(xs.max()+1),int(ys.max()+y0+1)
            candidates.append({"box":[float(box[0]),float(box[1]+y0),float(box[2]),float(box[3]+y0)],
                               "mask_bbox":[x1,y1,x2,y2], "mask":mask[ys.min():ys.max()+1,xs.min():xs.max()+1].copy(),
                               "confidence":float(score),"tile_y":y0,"repair":"none"})
    candidates, unions = merge_intratile_pairs(image_rgb,candidates)
    candidates, larger = deduplicate_tiles(candidates)
    public = []
    for row in candidates:
        b = row["mask_bbox"]
        public.append({"bbox":b,"confidence":row["confidence"],"repair":row["repair"],
                       "mask_sha256":hashlib.sha256(np.ascontiguousarray(row["mask"]).tobytes()).hexdigest(),
                       "model":MODEL_ID,"mask_role":"ownership_candidate_only"})
    return public, {"tiles":tile_count,"candidates":len(public),"intratile_unions":unions,
                    "larger_cross_tile_preserved":larger,"elapsed_s":time.perf_counter()-started}
