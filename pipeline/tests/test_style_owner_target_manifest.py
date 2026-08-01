from __future__ import annotations

import copy

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
