"""Render a deterministic source-to-materialization directional-gradient matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from typesetter.glyph_rasterizer import rasterize_v2_glyph_layers
from typesetter.gradient_model import (
    canonicalize_linear_gradient,
    render_linear_gradient_rgb,
)
from typesetter.owner_style import _materialize_style
from typesetter.style_extractor import extract_text_style_evidence_v2
from typesetter.style_materialization import (
    build_materialization_observation,
    build_materialization_plan,
    build_resolved_style_intent,
    compare_materialization,
)
from typesetter.style_policy import decide_style_copy_v2


CANVAS_SIZE = (120, 240)
ROW_HEIGHT = 180
SHEET_WIDTH = 960


def _gradient(
    colors: Sequence[str],
    start: Sequence[float],
    end: Sequence[float],
) -> dict[str, object]:
    value = canonicalize_linear_gradient(
        {
            "kind": "linear",
            "colors": list(colors),
            "stops": [0.0, 1.0],
            "start": list(start),
            "end": list(end),
            "coordinate_space": "glyph_bbox_normalized",
        }
    )
    if value is None:
        raise ValueError("matrix contains an invalid gradient case")
    return value


DIRECTIONAL_GRADIENT_CASES: list[dict[str, Any]] = [
    {"id": "purple_black_diagonal", "gradient": _gradient(["#6633CC", "#08080A"], [0, 0], [1, 1])},
    {"id": "red_yellow_horizontal", "gradient": _gradient(["#D71932", "#FFD447"], [0, 0.5], [1, 0.5])},
    {"id": "blue_green_diagonal", "gradient": _gradient(["#1749D1", "#32D779"], [0, 0], [1, 1])},
    {"id": "white_cyan_vertical", "gradient": _gradient(["#F4F8FF", "#16BFD3"], [0.5, 0], [0.5, 1])},
    {"id": "dark_light_reverse_vertical", "gradient": _gradient(["#EADFA8", "#19152E"], [0.5, 1], [0.5, 0])},
    {"id": "magenta_orange_opposite_diagonal", "gradient": _gradient(["#D52CB8", "#F28A28"], [1, 0], [0, 1])},
    {"id": "solid_black_negative", "gradient": None, "solid": "#111111"},
    {
        "id": "outlined_shadowed_blue_gold",
        "gradient": _gradient(["#244CC8", "#F2B83F"], [0.1, 0], [0.9, 1]),
        "outline": "#FFFFFF",
        "shadow": "#202020",
    },
]


def _text_mask(lines: Sequence[str], *, translated: bool = False) -> tuple[np.ndarray, list[int]]:
    height, width = CANVAS_SIZE
    mask = np.zeros((height, width), dtype=np.uint8)
    if translated:
        origins = [(18, 26), (48, 51), (24, 76), (78, 101)]
        scale = 0.52
    else:
        origins = [(12, 23), (35, 45), (8, 67), (56, 89), (28, 111)]
        scale = 0.46
    widths: list[int] = []
    for line, (x, baseline_y) in zip(lines, origins, strict=True):
        (text_width, _text_height), _baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )
        widths.append(int(text_width))
        cv2.putText(
            mask,
            line,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            255,
            2,
            cv2.LINE_AA,
        )
    if not translated:
        for point in ((7, 7), (232, 7), (7, 112), (232, 112)):
            cv2.circle(mask, point, 7, 255, -1, cv2.LINE_AA)
    return mask, widths


def _source_artifact(case: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    mask, widths = _text_mask(
        ["HONESTLY", "THERE IS NO", "ONE AMONG", "THE PLAYERS", "AT ALL"]
    )
    source = np.full((*CANVAS_SIZE, 3), 244, dtype=np.uint8)
    gradient = canonicalize_linear_gradient(case.get("gradient"))
    if case.get("shadow"):
        shifted = np.zeros_like(mask)
        shifted[3:, 3:] = mask[:-3, :-3]
        source[shifted > 32] = np.asarray([32, 32, 32], dtype=np.uint8)
    if case.get("outline"):
        outline = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        ring = (outline > 0) & (mask == 0)
        source[ring] = np.asarray([255, 255, 255], dtype=np.uint8)
    if gradient is None:
        solid = str(case.get("solid") or "#111111")
        rgb = np.asarray([int(solid[index : index + 2], 16) for index in (1, 3, 5)])
        source[mask > 0] = rgb
    else:
        painted = render_linear_gradient_rgb(mask, gradient)
        source[mask > 0] = painted[mask > 0]
    return source, mask, widths


def _mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def _axis(gradient: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(gradient["end"], dtype=np.float64) - np.asarray(
        gradient["start"], dtype=np.float64
    )


def _rgb(color: str) -> np.ndarray:
    return np.asarray([int(color[index : index + 2], 16) for index in (1, 3, 5)], dtype=np.float64)


def _direction_and_endpoint_metrics(
    expected: Mapping[str, Any], detected: Mapping[str, Any]
) -> tuple[float, float]:
    expected_axis = _axis(expected)
    detected_axis = _axis(detected)
    cosine = float(
        np.dot(expected_axis, detected_axis)
        / max(1e-9, np.linalg.norm(expected_axis) * np.linalg.norm(detected_axis))
    )
    expected_colors = list(expected["colors"])
    detected_colors = list(detected["colors"])
    if cosine < 0:
        cosine = -cosine
        detected_colors.reverse()
    errors = [
        float(np.linalg.norm(_rgb(expected_color) - _rgb(detected_color)))
        for expected_color, detected_color in zip(expected_colors, detected_colors, strict=True)
    ]
    return round(cosine, 6), round(max(errors), 6)


def evaluate_directional_gradient_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Run one synthetic source through extraction, policy, owner style and raster audit."""

    case_id = str(case["id"])
    source_rgb, source_mask, source_widths = _source_artifact(case)
    translated_mask, translated_widths = _text_mask(
        ["HONESTAMENTE", "NINGUEM", "ENTRE TODOS", "ME VENCE"],
        translated=True,
    )
    context = np.full(source_mask.shape, 255, dtype=np.uint8)
    evidence = extract_text_style_evidence_v2(
        source_rgb,
        source_mask,
        context,
        owner_id=f"owner_{case_id}",
        semantic_role="dialogue_body",
    )
    gradient_attribute = evidence.attributes.get("gradient")
    detected = canonicalize_linear_gradient(
        gradient_attribute.value if gradient_attribute is not None else None
    )
    expected = canonicalize_linear_gradient(case.get("gradient"))
    base: dict[str, Any] = {
        "case_id": case_id,
        "expected_gradient": expected,
        "detected_gradient": detected,
        "gradient_confidence": round(
            float(gradient_attribute.confidence if gradient_attribute is not None else 0.0),
            6,
        ),
        "source_mask_sha256": _mask_sha256(source_mask),
        "translated_mask_sha256": _mask_sha256(translated_mask),
        "source_line_widths": source_widths,
        "translated_line_widths": translated_widths,
        "_source_rgb": source_rgb,
        "_source_mask": source_mask,
    }
    if expected is None:
        return {
            **base,
            "decision_status": "fallback",
            "approved_gradient": None,
            "owner_gradient": None,
            "materialization_status": "not_applicable",
            "direction_cosine": None,
            "endpoint_error_rgb_max": None,
            "_translated_rgba": np.zeros((*translated_mask.shape, 4), dtype=np.uint8),
        }

    decision = decide_style_copy_v2(
        {
            "confidence": 0.98,
            "route_action": "translate_inpaint_render",
            "semantic_role": "dialogue_body",
            "background_rgb": [244, 244, 244],
        },
        evidence,
    )
    approved = canonicalize_linear_gradient(decision.applied_attributes.get("gradient"))
    owner_style = _materialize_style(
        {"background_rgb": [244, 244, 244], "semantic_role": "dialogue_body"},
        decision.to_dict(),
    )
    owner_gradient = canonicalize_linear_gradient(owner_style.get("cor_gradiente"))
    safe = np.full(translated_mask.shape, 255, dtype=np.uint8)
    raster = rasterize_v2_glyph_layers(
        translated_mask,
        safe,
        {"fill": owner_style.get("cor") or "#000000", "gradient": owner_gradient},
        rendered_x_height_px=18,
    )
    intent = build_resolved_style_intent(
        owner_id=f"owner_{case_id}",
        page_id="page_matrix",
        visual_profile_sha256="a" * 64,
        decision_sha256="b" * 64,
        group_resolution_sha256="c" * 64,
        approved={"gradient": approved},
        approved_abstentions={},
    )
    plan = build_materialization_plan(
        intent=intent,
        render_layout_contract_sha256="d" * 64,
        targets={"gradient": owner_gradient},
        resolution_kinds={"gradient": "exact"},
        resolution_reasons={},
        rendered_x_height_px=18,
    )
    observation = build_materialization_observation(
        plan=plan,
        domain_observations={
            "raster": {
                "gradient": {
                    "value": raster.observed_attributes.get("gradient"),
                    "evidence_kind": "layer_pixels_and_mask",
                    "evidence_sha256": raster.attribute_evidence_sha256["gradient"],
                }
            }
        },
        render_completed=True,
    )
    comparison = compare_materialization(plan, observation)
    direction_cosine, endpoint_error = _direction_and_endpoint_metrics(expected, detected)
    return {
        **base,
        "decision_status": decision.status,
        "approved_gradient": approved,
        "owner_gradient": owner_gradient,
        "materialization_status": comparison.status,
        "direction_cosine": direction_cosine,
        "endpoint_error_rgb_max": endpoint_error,
        "_translated_rgba": raster.rgba,
    }


def _public_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items() if not str(key).startswith("_")
    }


def _diagnostic_panel(result: Mapping[str, Any]) -> np.ndarray:
    panel = np.full((*CANVAS_SIZE, 3), 250, dtype=np.uint8)
    lines = [
        str(result["case_id"]),
        f"decision: {result['decision_status']}",
        f"material: {result['materialization_status']}",
        f"cos: {result['direction_cosine']}",
        f"rgb err: {result['endpoint_error_rgb_max']}",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line[:34],
            (8, 18 + index * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    return panel


def render_directional_gradient_matrix(output_dir: Path | str) -> dict[str, Any]:
    """Write the deterministic contact sheet and its JSON evidence."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = [evaluate_directional_gradient_case(case) for case in DIRECTIONAL_GRADIENT_CASES]
    rows: list[np.ndarray] = []
    for result in results:
        source = np.asarray(result["_source_rgb"], dtype=np.uint8)
        mask = cv2.cvtColor(np.asarray(result["_source_mask"], dtype=np.uint8), cv2.COLOR_GRAY2RGB)
        rgba = np.asarray(result["_translated_rgba"], dtype=np.uint8)
        translated = np.full((*CANVAS_SIZE, 3), 244, dtype=np.uint8)
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        translated = np.clip(
            rgba[:, :, :3].astype(np.float32) * alpha
            + translated.astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        row = np.concatenate((source, mask, translated, _diagnostic_panel(result)), axis=1)
        row_canvas = np.full((ROW_HEIGHT, SHEET_WIDTH, 3), 230, dtype=np.uint8)
        row_canvas[: CANVAS_SIZE[0], : row.shape[1]] = row
        rows.append(row_canvas)
    sheet = np.concatenate(rows, axis=0)
    sheet_path = target / "directional_gradient_matrix.png"
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    public_results = [_public_result(result) for result in results]
    positive = [row for row in public_results if row["expected_gradient"] is not None]
    visual_go = all(
        row["decision_status"] == "applied"
        and row["materialization_status"] == "match"
        and float(row["direction_cosine"] if row["direction_cosine"] is not None else 0.0) >= 0.90
        and float(
            row["endpoint_error_rgb_max"]
            if row["endpoint_error_rgb_max"] is not None
            else math.inf
        ) <= 30.0
        for row in positive
    ) and all(
        row["detected_gradient"] is None
        for row in public_results
        if row["expected_gradient"] is None
    )
    summary = {
        "schema_version": 1,
        "case_ids": [row["case_id"] for row in public_results],
        "visual_verdict": "GO" if visual_go else "NO-GO",
        "cases": public_results,
    }
    (target / "directional_gradient_matrix.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = render_directional_gradient_matrix(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["visual_verdict"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
