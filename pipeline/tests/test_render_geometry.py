import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_check_render_inside_balloon_flags_low_containment():
    from qa.render_geometry import check_render_inside_balloon

    result = check_render_inside_balloon(
        render_bbox=[0, 0, 100, 100],
        balloon_bbox=[0, 0, 80, 80],
    )

    assert result["containment"] == 0.64
    assert result["flags"] == ["render_outside_balloon"]


def test_check_render_background_flags_dark_sample_in_white_balloon():
    from qa.render_geometry import check_render_background

    image = np.full((80, 120, 3), 245, dtype=np.uint8)
    image[20:40, 30:70] = 42

    result = check_render_background(
        image,
        render_bbox=[30, 20, 70, 40],
        balloon_bbox=[10, 10, 100, 70],
        balloon_type="white",
    )

    assert result["background_luma"] == 42.0
    assert result["flags"] == ["render_on_art_suspected"]


def test_check_render_background_does_not_flag_flat_colored_balloon():
    from qa.render_geometry import check_render_background

    image = np.full((120, 180, 3), [183, 202, 232], dtype=np.uint8)

    result = check_render_background(
        image,
        render_bbox=[55, 40, 125, 70],
        balloon_bbox=[20, 15, 160, 105],
        balloon_type="",
    )

    assert result["background_luma"] < 215.0
    assert result["flat_balloon_background"] is True
    assert result["flags"] == []


def test_check_sfx_render_geometry_flags_review_only_sfx_risks():
    from qa.render_geometry import check_sfx_render_geometry

    result = check_sfx_render_geometry(
        {
            "tipo": "sfx",
            "source_bbox": [10, 10, 90, 90],
            "render_bbox": [80, 80, 130, 130],
            "sfx": {
                "inpaint_allowed": False,
                "review_required": True,
                "style_confidence": 0.31,
            },
        }
    )

    assert result["flags"] == [
        "sfx_render_outside_source_region",
        "sfx_inpaint_damaged_art_risk",
        "sfx_translation_unknown",
        "sfx_style_low_confidence",
    ]


def test_check_sfx_render_geometry_flags_missing_render_bbox():
    from qa.render_geometry import check_sfx_render_geometry

    result = check_sfx_render_geometry(
        {
            "content_class": "sfx",
            "bbox": [10, 10, 60, 60],
            "sfx": {"adapted_text": "PAM"},
        }
    )

    assert result["flags"] == ["sfx_render_missing"]


def test_check_sfx_render_geometry_flags_visual_unknown_review_candidate():
    from qa.render_geometry import check_sfx_render_geometry

    result = check_sfx_render_geometry(
        {
            "content_class": "sfx",
            "route_action": "review_required",
            "script": "unknown",
            "bbox": [10, 10, 80, 80],
            "qa_flags": ["sfx_visual_candidate", "sfx_script_unknown"],
            "sfx": {
                "visual_detector": "sfx_visual",
                "inpaint_allowed": False,
                "qa_flags": ["sfx_script_unknown"],
            },
        }
    )

    assert "sfx_render_missing" in result["flags"]
    assert "sfx_inpaint_damaged_art_risk" in result["flags"]
    assert "sfx_translation_unknown" in result["flags"]
