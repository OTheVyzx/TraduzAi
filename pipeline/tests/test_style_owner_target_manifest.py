from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from test_final_pixel_qa import _graph
from tools.build_style_owner_target_manifest import (
    OwnerTargetError,
    build_effective_style_config,
    discover_owner_targets,
    selected_owner_target,
    source_crop_contract,
    verify_matrix_target,
)


def _project() -> dict:
    graph = _graph().to_dict()
    return {
        "paginas": [{"page_id": "page_001", "numero": 1}],
        "page_owner_graphs": [graph],
    }


def test_target_discovery_is_deterministic_and_owner_exact():
    first = discover_owner_targets(_project())
    second = discover_owner_targets(copy.deepcopy(_project()))

    assert first == second
    assert first[0]["owner_id"] == "owner_a"
    assert first[0]["component_ids"] == ["component_a"]
    assert len(first[0]["target_sha256"]) == 64


def test_matrix_rejects_out_of_range_page_instead_of_clamping():
    target = {"page_id": "page_055", "owner_id": "owner_a", "component_ids": ["component_a"]}

    with pytest.raises(OwnerTargetError, match="page target not found"):
        selected_owner_target(target, _project())


def test_matrix_rejects_missing_owner():
    target = {"page_id": "page_001", "owner_id": "missing", "component_ids": ["component_a"]}

    with pytest.raises(OwnerTargetError, match="owner target not found"):
        selected_owner_target(target, _project())


def test_style_matrix_forces_enforce_without_mutating_functional_config():
    shared = {"style_copy_mode": "shadow", "strict": True}

    effective = build_effective_style_config(shared, required_categories=["sfx", "dark_panel"])

    assert effective["style_copy_mode"] == "enforce"
    assert effective["style_inspection_required"] is True
    assert shared["style_copy_mode"] == "shadow"


def test_verify_matrix_target_binds_logical_crop_to_framed_artifact(tmp_path):
    from PIL import Image
    from strip.page_surface_geometry import PageSurfaceGeometry

    geometry = PageSurfaceGeometry.build(
        logical_width=40,
        logical_height=24,
        frame_width=50,
        frame_height=24,
        content_origin_xy=(5, 0),
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (50, 24), "white").save(source)
    project = _project()
    project["paginas"][0].update({
        "page_surface_geometry": geometry.to_dict(),
        "page_surface_geometry_sha256": geometry.geometry_sha256,
        "image_layers": {"base": {"path": "source.png"}},
    })
    logical_bbox = [5, 5, 30, 17]
    frame_bbox = list(geometry.logical_bbox_to_frame(tuple(logical_bbox)))
    source_crop = source_crop_contract(
        source,
        logical_bbox,
        artifact_bbox_frame=frame_bbox,
        coordinate_space="logical_page",
    )
    source_crop.pop("artifact_bbox_frame")
    target = {
        "page_id": "page_001",
        "owner_id": "owner_a",
        "component_ids": ["component_a"],
        "expected_artifact_space": "framed_page",
        "source_crop": source_crop,
    }

    result = verify_matrix_target(target, project, project_root=tmp_path)

    assert result["verified"] is True
    assert result["artifact_bbox_frame"] == [10, 5, 35, 17]
    assert result["page_surface_geometry_sha256"] == geometry.geometry_sha256


def test_exact_nine_owner_targets_and_source_hashes_are_unchanged():
    matrix_path = Path(__file__).parent / "fixtures" / "style_copy_corpus" / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    expected = {
        ("style_colored_cards_calibration", "page_001", "owner_p001_024cf5ddf5ea"): ([246, 10391, 628, 10542], "2a2f55262ed94479e8ee1819a8eb5e932e7a1a5af5b56b5a6f984da0ba6b2404"),
        ("style_colored_cards_calibration", "page_001", "owner_p001_06c2c46ee27f"): ([340, 4052, 602, 4180], "8f17bc734c4946c6eea066e001a30e7900cb1fa54e4d0971f2bc4cd786ede261"),
        ("style_colored_cards_calibration", "page_001", "owner_p001_0dc05817313b"): ([198, 6057, 604, 6242], "4f145ec82428d6deb3c6b427baa450411e0af75774bdde0c3bdf26523c0b40dc"),
        ("style_colored_cards_calibration", "page_001", "owner_p001_0f39ce3adba0"): ([169, 11133, 542, 11247], "00e896e3eabc1003c611cfd8fb5ac847534c1847c10a8bf99f5506229bdf4f23"),
        ("style_cross_tile_holdout", "page_002", "owner_p002_43752a8f8c08"): ([391, 484, 618, 595], "2c2147610a183eeb8e7500ce5d44c5aba4d0f43fc7f09682646345eb89b2ee10"),
        ("style_cross_tile_holdout", "page_002", "owner_p002_7227e479da0d"): ([76, 54, 243, 133], "23aa55bd1525fab675a524e9856f5beea464a0eea07c87ab77508180c7048dbe"),
        ("style_cross_tile_holdout", "page_002", "owner_p002_d9afb32ef935"): ([109, 985, 318, 1104], "e4c60408223dfef9a4f0236b41b07c603b983420e9ca5a3715a030ccb1532c31"),
        ("style_dark_panels_holdout", "page_002", "owner_p002_5a6976dbbf65"): ([107, 1915, 424, 2114], "3596b34910f6cb295478af3daa875224b742e8ebc250735e4c7db86e6b0aeeab"),
        ("style_dark_panels_holdout", "page_003", "owner_p003_98e72af0685a"): ([183, 572, 514, 682], "054fe2c77ba95ef21ccc6c4f9d90975f2293938e10649419304a19b1efcfc5c3"),
    }
    actual = {}
    for entry in matrix["entries"]:
        for target in entry["targets"]:
            key = (entry["entry_id"], target["page_id"], target["owner_id"])
            actual[key] = (target["source_crop"]["bbox_page"], target["source_crop"]["sha256"])
            assert target["source_crop"]["coordinate_space"] == "logical_page"
    assert actual == expected
