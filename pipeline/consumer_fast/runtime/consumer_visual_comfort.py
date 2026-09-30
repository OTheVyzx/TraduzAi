"""Source-bound white-balloon comfort measurement for opt-in focal recipes.

The source crop is retained in the trusted local recipe. The clean canvas is
never allowed to invent an interior after cleanup has erased a contour.
Other balloon/background kinds must use a different policy or remain pending.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCHEMA = "source_closed_white_comfort_v1"
SEAM_SCHEMA = "source_closed_white_comfort_seam_v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _closed_white_interior(crop: np.ndarray, seed_xy: tuple[int, int],
                           threshold: int) -> np.ndarray:
    if crop.ndim != 3 or crop.shape[2] != 3 or not 220 <= threshold <= 252:
        raise ValueError("invalid source white-interior input")
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    chroma = crop.max(2).astype(np.int16) - crop.min(2).astype(np.int16)
    white = ((gray >= threshold) & (chroma <= 10)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    sx, sy = seed_xy
    if not (0 <= sx < crop.shape[1] and 0 <= sy < crop.shape[0]):
        raise ValueError("source anchor outside contour crop")
    label = int(labels[sy, sx])
    if label == 0:
        radius = 12
        x1, x2 = max(0, sx-radius), min(crop.shape[1], sx+radius+1)
        y1, y2 = max(0, sy-radius), min(crop.shape[0], sy+radius+1)
        local = labels[y1:y2, x1:x2]
        candidates = [int(value) for value in np.unique(local) if value]
        label = max(candidates, key=lambda value: int(stats[value, cv2.CC_STAT_AREA]),
                    default=0)
    if not label:
        raise ValueError("source anchor has no white body")
    x, y, width, height, area = map(int, stats[label])
    if x <= 0 or y <= 0 or x+width >= crop.shape[1] or y+height >= crop.shape[0]:
        raise ValueError("source white component leaks through crop edge")
    if area < 100:
        raise ValueError("source white component too small")
    component = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        raise ValueError("source contour is ambiguous")
    interior = np.zeros_like(component)
    cv2.drawContours(interior, contours, 0, 1, cv2.FILLED)
    return interior.astype(bool)


def build_source_white_policy(source_path: Path, roi_bbox: list[int],
                              anchor_bbox: list[int], minimum_px: int,
                              ideal_px: int, *, threshold: int = 245,
                              source_member_bytes: bytes | None = None) -> dict:
    """Create a source-linked policy; caller must select a single closed body."""
    source_path = Path(source_path)
    source_png_hash = _sha(source_path.read_bytes())
    with Image.open(source_path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    source_hash = source_png_hash
    if source_member_bytes is not None:
        with Image.open(BytesIO(source_member_bytes)) as image:
            decoded_member = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(decoded_member, original):
            raise ValueError("source member and project original pixels differ")
        source_hash = _sha(source_member_bytes)
    x1, y1, x2, y2 = map(int, roi_bbox)
    if not (0 <= x1 < x2 <= original.shape[1] and 0 <= y1 < y2 <= original.shape[0]):
        raise ValueError("invalid source contour crop")
    crop = original[y1:y2, x1:x2].copy()
    sx = round((anchor_bbox[0]+anchor_bbox[2])/2)-x1
    sy = round((anchor_bbox[1]+anchor_bbox[3])/2)-y1
    _closed_white_interior(crop, (sx, sy), threshold)
    if not (1 <= minimum_px <= ideal_px <= 128):
        raise ValueError("invalid comfort margins")
    success, png = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("source crop encoding failed")
    return {"schema": SCHEMA, "source_sha256": source_hash,
            "project_original_sha256": source_png_hash,
            "source_roi_bbox": list(map(int, roi_bbox)),
            "anchor_bbox": list(map(int, anchor_bbox)),
            "source_crop_png": png.tobytes(),
            "source_crop_pixels_sha256": _sha(crop.tobytes()),
            "white_threshold": threshold,
            "minimum_px": minimum_px, "ideal_px": ideal_px,
            "margin_units": "source_pixels",
            "boundary_kind": "physical_source_contour",
            "already_eroded_px": 0}


def build_source_white_policy_across_previous(
    previous_path: Path, current_path: Path, *,
    previous_index: int, current_index: int,
    previous_member: str, current_member: str,
    previous_member_bytes: bytes, current_member_bytes: bytes,
    previous_rows: int, current_rows: int,
    anchor_bbox: list[int], observation_bboxes: list[list[int]],
    minimum_px: int, ideal_px: int, threshold: int = 245,
) -> dict:
    """Observe a white body across one authenticated source-member seam.

    Only the local rows required to close the contour are retained. Coordinates
    in the returned distance map are relative to the current member, so a row
    from its predecessor has a negative y coordinate. The predecessor supplies
    geometry, never translated pixels or a target-language decision.
    """
    if previous_index + 1 != current_index or previous_member == current_member:
        raise ValueError("source context members are not adjacent")
    if not (1 <= minimum_px <= ideal_px <= 128):
        raise ValueError("invalid comfort margins")
    previous_path, current_path = Path(previous_path), Path(current_path)
    with Image.open(previous_path) as image:
        previous = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(current_path) as image:
        current = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(BytesIO(previous_member_bytes)) as image:
        previous_member_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(BytesIO(current_member_bytes)) as image:
        current_member_pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if not (np.array_equal(previous, previous_member_pixels) and
            np.array_equal(current, current_member_pixels)):
        raise ValueError("source member and project original pixels differ")
    if (previous.shape[1] != current.shape[1] or
            not 1 <= previous_rows <= previous.shape[0] or
            not 1 <= current_rows <= current.shape[0]):
        raise ValueError("source context dimensions or rows invalid")
    width = current.shape[1]
    if (len(anchor_bbox) != 4 or not observation_bboxes or
            any(len(box) != 4 or not (0 <= box[0] < box[2] <= width and
                0 <= box[1] < box[3] <= current_rows)
                for box in [anchor_bbox, *observation_bboxes])):
        raise ValueError("source context observations outside current member")
    crop = np.concatenate((previous[-previous_rows:], current[:current_rows]), axis=0)
    seed = (round((anchor_bbox[0] + anchor_bbox[2]) / 2),
            previous_rows + round((anchor_bbox[1] + anchor_bbox[3]) / 2))
    interior = _closed_white_interior(crop, seed, threshold)
    coverage = []
    for x1, y1, x2, y2 in observation_bboxes:
        patch = interior[previous_rows+y1:previous_rows+y2, x1:x2]
        coverage.append(float(np.mean(patch)))
    if min(coverage) < .55 or np.count_nonzero(interior) > .5 * crop.shape[0] * width:
        raise ValueError("source context body association ambiguous")
    success, png = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError("source context crop encoding failed")
    return {
        "schema": SEAM_SCHEMA,
        "source_sha256": _sha(current_member_bytes),
        "project_original_sha256": _sha(current_path.read_bytes()),
        "source_roi_bbox": [0, -previous_rows, width, current_rows],
        "anchor_bbox": list(map(int, anchor_bbox)),
        "source_crop_png": png.tobytes(),
        "source_crop_pixels_sha256": _sha(crop.tobytes()),
        "white_threshold": threshold,
        "minimum_px": minimum_px, "ideal_px": ideal_px,
        "margin_units": "source_pixels",
        "boundary_kind": "physical_source_contour_across_member_seam",
        "already_eroded_px": 0,
        "context_members": [
            dict(source_index=previous_index, source_member=previous_member,
                 source_sha256=_sha(previous_member_bytes),
                 project_original_sha256=_sha(previous_path.read_bytes()),
                 crop_bbox=[0, previous.shape[0]-previous_rows, width, previous.shape[0]],
                 virtual_bbox=[0, -previous_rows, width, 0]),
            dict(source_index=current_index, source_member=current_member,
                 source_sha256=_sha(current_member_bytes),
                 project_original_sha256=_sha(current_path.read_bytes()),
                 crop_bbox=[0, 0, width, current_rows],
                 virtual_bbox=[0, 0, width, current_rows]),
        ],
        "observation_bboxes": observation_bboxes,
        "observation_coverage": coverage,
    }


def build_source_white_policy_adaptive(
    source_path: Path, body_hint_bbox: list[int], anchor_bbox: list[int],
    observation_bboxes: list[list[int]], minimum_px: int, ideal_px: int,
    *, source_member_bytes: bytes | None = None, initial_pad_px: int | None = None,
    max_attempts: int = 6, max_total_roi_pixels: int = 8_000_000,
    threshold: int = 245,
) -> dict:
    """Expand only the *observation crop* until a source body is enclosed.

    This never expands the layout area. A component which remains connected to
    the image border is unresolved, not accepted by a larger padding value.
    """
    source_path = Path(source_path)
    with Image.open(source_path) as image:
        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = original.shape[:2]
    boxes = [body_hint_bbox, anchor_bbox, *observation_bboxes]
    if not observation_bboxes or any(len(b) != 4 or not (0 <= b[0] < b[2] <= width and
            0 <= b[1] < b[3] <= height) for b in boxes):
        raise ValueError("invalid source observation geometry")
    base = [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]
    heights = [b[3]-b[1] for b in observation_bboxes]
    pad = max(32, int(round(2 * float(np.median(heights))))) if initial_pad_px is None else int(initial_pad_px)
    if not (8 <= pad <= 1024 and 1 <= max_attempts <= 10 and max_total_roi_pixels > 0):
        raise ValueError("invalid adaptive observation budget")
    attempts = []
    cost = 0
    previous_roi = None
    for index in range(max_attempts):
        roi = [max(0, base[0]-pad), max(0, base[1]-pad),
               min(width, base[2]+pad), min(height, base[3]+pad)]
        if roi == previous_roi:
            break
        previous_roi = roi
        x1, y1, x2, y2 = roi
        cost += (x2-x1)*(y2-y1)
        if cost > max_total_roi_pixels:
            attempts.append(dict(roi_bbox=roi, status="cost_limit"))
            return dict(status="cost_limit", attempts=attempts, observed_roi_pixels=cost)
        crop = original[y1:y2, x1:x2]
        seed = (round((anchor_bbox[0]+anchor_bbox[2])/2)-x1,
                round((anchor_bbox[1]+anchor_bbox[3])/2)-y1)
        try:
            interior = _closed_white_interior(crop, seed, threshold)
        except ValueError as error:
            reason = str(error)
            if reason == "source white component leaks through crop edge":
                attempts.append(dict(roi_bbox=roi, status="crop_incomplete"))
                pad = int(np.ceil(pad * 1.75))
                continue
            attempts.append(dict(roi_bbox=roi, status="source_body_unresolved", reason=reason))
            return dict(status="source_body_unresolved", attempts=attempts,
                        observed_roi_pixels=cost)
        # A completed component must explain the selected source observations.
        coverage = []
        for bx1, by1, bx2, by2 in observation_bboxes:
            patch = interior[by1-y1:by2-y1, bx1-x1:bx2-x1]
            coverage.append(float(np.mean(patch)) if patch.size else 0.0)
        if min(coverage) < 0.55:
            attempts.append(dict(roi_bbox=roi, status="association_ambiguous",
                                 observation_coverage=coverage))
            return dict(status="association_ambiguous", attempts=attempts,
                        observed_roi_pixels=cost)
        if np.count_nonzero(interior) > 0.5 * width * height:
            attempts.append(dict(roi_bbox=roi, status="background_leak",
                                 observation_coverage=coverage))
            return dict(status="background_leak", attempts=attempts,
                        observed_roi_pixels=cost)
        policy = build_source_white_policy(source_path, roi, anchor_bbox,
                                           minimum_px, ideal_px, threshold=threshold,
                                           source_member_bytes=source_member_bytes)
        policy["observation_expansion"] = dict(schema="adaptive_source_crop_v1",
                                                 body_hint_bbox=body_hint_bbox,
                                                 observation_bboxes=observation_bboxes,
                                                 attempts=attempts + [dict(
                                                     roi_bbox=roi, status="body_validated",
                                                     observation_coverage=coverage)],
                                                 observed_roi_pixels=cost)
        return dict(status="body_validated", policy=policy,
                    attempts=policy["observation_expansion"]["attempts"],
                    observed_roi_pixels=cost)
    return dict(status="open_contour_or_background_leak",
                attempts=attempts, observed_roi_pixels=cost)


def prepare_source_white_comfort(policy: dict, source_sha256: str) -> dict:
    """Decode and measure one authenticated contour once per bounded search."""
    if policy.get("schema") not in {SCHEMA, SEAM_SCHEMA} or policy.get("source_sha256") != source_sha256:
        raise ValueError("source comfort policy or hash mismatch")
    if policy.get("margin_units") != "source_pixels" or policy.get("already_eroded_px") != 0:
        raise ValueError("unsupported comfort margin accounting")
    minimum, ideal = int(policy["minimum_px"]), int(policy["ideal_px"])
    if not 1 <= minimum <= ideal <= 128:
        raise ValueError("invalid comfort margins")
    crop_bgr = cv2.imdecode(np.frombuffer(policy["source_crop_png"], np.uint8), cv2.IMREAD_COLOR)
    if crop_bgr is None:
        raise ValueError("source contour crop is unreadable")
    crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if _sha(crop.tobytes()) != policy["source_crop_pixels_sha256"]:
        raise ValueError("source contour crop changed")
    rx1, ry1, rx2, ry2 = map(int, policy["source_roi_bbox"])
    if crop.shape[:2] != (ry2-ry1, rx2-rx1):
        raise ValueError("source contour crop size changed")
    ax1, ay1, ax2, ay2 = map(int, policy["anchor_bbox"])
    interior = _closed_white_interior(
        crop, (round((ax1+ax2)/2)-rx1, round((ay1+ay2)/2)-ry1),
        int(policy["white_threshold"]))
    distance = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, 5)
    yy, xx = np.nonzero(interior)
    return {"schema": policy["schema"], "source_sha256": source_sha256,
            "source_roi_bbox": [rx1, ry1, rx2, ry2], "minimum_px": minimum,
            "ideal_px": ideal, "distance": distance,
            "boundary_kind": policy["boundary_kind"],
            "interior_bbox": [int(xx.min()+rx1), int(yy.min()+ry1),
                              int(xx.max()+rx1+1), int(yy.max()+ry1+1)]}


def measure_prepared_source_white_comfort(prepared: dict,
                                           ink_bbox: tuple[int, int, int, int],
                                           alpha: np.ndarray) -> dict:
    """Measure final alpha against the prepared source contour."""
    rx1, ry1, rx2, ry2 = prepared["source_roi_bbox"]
    minimum, ideal = prepared["minimum_px"], prepared["ideal_px"]
    distance = prepared["distance"]
    x1, y1, x2, y2 = ink_bbox
    if not (rx1 <= x1 < x2 <= rx2 and ry1 <= y1 < y2 <= ry2 and
            alpha.shape == (y2-y1, x2-x1)):
        raise ValueError("rendered ink outside source contour crop")
    ink = alpha > 0
    if not ink.any():
        raise ValueError("empty rendered ink")
    values = distance[y1-ry1:y2-ry1, x1-rx1:x2-rx1][ink]
    shortest = float(values.min())
    unsafe = int(np.count_nonzero(values == 0))
    below = int(np.count_nonzero(values < minimum))
    result = {"schema": SCHEMA, "source_sha256": prepared["source_sha256"],
              "minimum_required_px": minimum, "ideal_px": ideal,
              "minimum_actual_px": shortest,
              "p10_actual_px": float(np.percentile(values, 10)),
              "median_actual_px": float(np.median(values)),
              "unsafe_ink_pixels": unsafe, "below_minimum_ink_pixels": below,
              "safety_pass": unsafe == 0, "comfort_minimum_pass": below == 0,
              "comfort_ideal_reached": shortest >= ideal,
              "boundary_kind": prepared["boundary_kind"]}
    return result


def measure_source_white_comfort(policy: dict, source_sha256: str,
                                  ink_bbox: tuple[int, int, int, int],
                                  alpha: np.ndarray) -> dict:
    """Measure final alpha against source contour, independently of clean."""
    prepared = prepare_source_white_comfort(policy, source_sha256)
    return measure_prepared_source_white_comfort(prepared, ink_bbox, alpha)
