from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from debug_tools.render_directional_gradient_matrix import (
    DIRECTIONAL_GRADIENT_CASES,
    evaluate_directional_gradient_case,
    render_directional_gradient_matrix,
)


EXPECTED_CASE_IDS = [
    "purple_black_diagonal",
    "red_yellow_horizontal",
    "blue_green_diagonal",
    "white_cyan_vertical",
    "dark_light_reverse_vertical",
    "magenta_orange_opposite_diagonal",
    "solid_black_negative",
    "outlined_shadowed_blue_gold",
]


def test_directional_gradient_matrix_cases_are_deterministic_and_systemic():
    assert [case["id"] for case in DIRECTIONAL_GRADIENT_CASES] == EXPECTED_CASE_IDS

    results = [evaluate_directional_gradient_case(case) for case in DIRECTIONAL_GRADIENT_CASES]

    for result in results:
        if result["expected_gradient"] is None:
            assert result["detected_gradient"] is None
            assert result["materialization_status"] == "not_applicable"
            continue
        assert result["decision_status"] == "applied"
        assert result["owner_gradient"] == result["approved_gradient"]
        assert result["materialization_status"] == "match"
        assert result["direction_cosine"] >= 0.90
        assert result["endpoint_error_rgb_max"] <= 30.0
        assert result["source_mask_sha256"] != result["translated_mask_sha256"]
        assert result["translated_line_widths"] != result["source_line_widths"]


def test_directional_gradient_matrix_tool_writes_only_declared_output(tmp_path: Path):
    before = set(tmp_path.rglob("*"))
    output_dir = tmp_path / "matrix"

    summary = render_directional_gradient_matrix(output_dir)

    after = set(tmp_path.rglob("*"))
    assert after - before == {
        output_dir,
        output_dir / "directional_gradient_matrix.png",
        output_dir / "directional_gradient_matrix.json",
    }
    assert summary["case_ids"] == EXPECTED_CASE_IDS
    assert summary["visual_verdict"] == "GO"
    evidence = json.loads(
        (output_dir / "directional_gradient_matrix.json").read_text(encoding="utf-8")
    )
    assert evidence == summary
    sheet = cv2.imread(str(output_dir / "directional_gradient_matrix.png"), cv2.IMREAD_COLOR)
    assert sheet is not None
    assert sheet.shape == (len(EXPECTED_CASE_IDS) * 180, 960, 3)
    assert int(np.std(sheet)) > 10
