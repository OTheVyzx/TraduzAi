from __future__ import annotations

import copy

import pytest

from test_final_pixel_qa import _graph
from tools.build_style_owner_target_manifest import (
    OwnerTargetError,
    build_effective_style_config,
    discover_owner_targets,
    selected_owner_target,
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
