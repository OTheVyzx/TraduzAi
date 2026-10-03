import json
from copy import deepcopy
from hashlib import sha256
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _save_project_json
from project_writer import (
    neutralize_project_compatibility_metadata,
    validate_project_consistency,
    write_project_json_atomic,
)
from style_v2_fixtures import valid_owner_style_raster_contract
from ownership.model import OWNER_GRAPH_SCHEMA_VERSION
from ownership.project import (
    build_owner_invariant_summary,
    owner_project_validation_errors,
)


def _project():
    return {
        "paginas": [{"text_layers": [{"qa_flags": ["low_ocr_confidence"]}]}],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 1}},
    }


def _verified_owner_project():
    graph = {
        "schema_version": OWNER_GRAPH_SCHEMA_VERSION,
        "page_id": "page_001",
        "run_id": "run-project-writer",
        "origin_execution_id": "execution-project-writer",
        "page_source_sha256": "a" * 64,
        "verification_status": "verified",
        "components": [
            {
                "component_id": "cmp_page_001_body",
                "page_id": "page_001",
                "bbox_page": [10, 20, 110, 80],
                "polygon_page": [[10, 20], [110, 20], [110, 80], [10, 80]],
                "detector_sources": ["fixture"],
            }
        ],
        "observations": [
            {
                "observation_id": "obs_page_001_body",
                "page_id": "page_001",
                "component_ids": ["cmp_page_001_body"],
                "text": "HELLO THERE",
                "confidence": 0.97,
                "provider": "fixture",
                "bbox_page": [10, 20, 110, 80],
                "run_id": "run-project-writer",
                "origin_execution_id": "execution-project-writer",
                "invocation_id": "invocation-project-writer",
                "attempt_id": "attempt-project-writer",
                "provider_family": "fixture",
                "page_source_sha256": "a" * 64,
                "root_input_pixel_sha256": "a" * 64,
                "input_pixel_sha256": "b" * 64,
                "payload_sha256": sha256(b"HELLO THERE").hexdigest(),
            }
        ],
        "owners": [
            {
                "owner_id": "own_page_001_body",
                "page_id": "page_001",
                "component_ids": ["cmp_page_001_body"],
                "observation_ids": ["obs_page_001_body"],
                "selected_observation_ids": ["obs_page_001_body"],
                "semantic_role": "dialogue_body",
                "source_payload": "HELLO THERE",
                "translated_payload": None,
                "disposition": "owned",
                "state": "owned",
                "route_action": "translate_inpaint_render",
                "execution_tile_id": None,
                "action_mask_ref": None,
            }
        ],
        "projections": [],
        "component_dispositions": [
            {
                "component_id": "cmp_page_001_body",
                "decision": "owned",
                "owner_id": "own_page_001_body",
                "reason": "fixture",
            }
        ],
        "violations": [],
    }
    return {
        "paginas": [
            {
                "numero": 1,
                "text_layers": [
                    {
                        "id": "own_page_001_body",
                        "owner_id": "own_page_001_body",
                        "page_id": "page_001",
                        "component_ids": ["cmp_page_001_body"],
                        "observation_ids": ["obs_page_001_body"],
                        "semantic_role": "dialogue_body",
                        "route_action": "translate_inpaint_render",
                        "action_mask_ref": None,
                        "layout_region_ids": [],
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 0}},
        "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
        "owner_graph_status": "verified",
        "page_owner_graphs": [graph],
        "owner_invariant_summary": {
            "page_count": 1,
            "component_count": 1,
            "observation_count": 1,
            "owner_count": 1,
            "projection_count": 0,
            "violation_count": 0,
            "critical_violation_count": 0,
        },
    }


def _two_page_retired_review_project():
    """A merged review candidate on page 1 and split candidates on page 2."""
    def component(component_id, page_id, x):
        return {
            "component_id": component_id,
            "page_id": page_id,
            "bbox_page": [x, 20, x + 20, 40],
            "polygon_page": [[x, 20], [x + 20, 20], [x + 20, 40], [x, 40]],
            "detector_sources": ["test_fixture"],
        }

    def observation(observation_id, page_id, component_id, text):
        return {
            "observation_id": observation_id,
            "page_id": page_id,
            "component_ids": [component_id],
            "text": text,
            "confidence": 0.95,
            "provider": "test_fixture",
            "bbox_page": [10, 20, 80, 40],
            "run_id": f"run-{page_id}",
            "origin_execution_id": f"execution-{page_id}",
            "invocation_id": f"invocation-{observation_id}",
            "attempt_id": f"attempt-{observation_id}",
            "provider_family": "test_fixture",
            "page_source_sha256": "a" * 64 if page_id == "page_001" else "b" * 64,
            "root_input_pixel_sha256": "a" * 64 if page_id == "page_001" else "b" * 64,
            "input_pixel_sha256": "c" * 64,
            "payload_sha256": sha256(text.encode("utf-8")).hexdigest(),
        }

    def retired_disposition(component_id, observation_id, x):
        return {
            "component_id": component_id,
            "decision": "uncertain",
            "owner_id": None,
            "reason": "owner_execution_rejected",
            "policy_id": "coverage_ambiguous_candidate",
            "policy_bbox_page": [x, 20, x + 20, 40],
            "policy_evidence_ids": [observation_id],
            "policy_reason": "source pixels preserved after owner execution rejection",
        }

    def graph(page_id, components, observations, owners, dispositions):
        page_sha = "a" * 64 if page_id == "page_001" else "b" * 64
        return {
            "schema_version": OWNER_GRAPH_SCHEMA_VERSION,
            "page_id": page_id,
            "run_id": f"run-{page_id}",
            "origin_execution_id": f"execution-{page_id}",
            "page_source_sha256": page_sha,
            "verification_status": "verified",
            "components": components,
            "observations": observations,
            "owners": owners,
            "projections": [],
            "component_dispositions": dispositions,
            "violations": [],
        }

    def review_layer(candidate_id, page_id, component_ids, observation_ids, source, translated):
        return {
            "id": candidate_id,
            "owner_id": None,
            "candidate_owner_id": candidate_id,
            "page_id": page_id,
            "component_ids": component_ids,
            "observation_ids": observation_ids,
            "selected_observation_ids": observation_ids,
            "semantic_role": "dialogue_body",
            "route_action": "review_required",
            "action_mask_ref": None,
            "layout_region_ids": [],
            "disposition": "review",
            "state": "review_required",
            "execution_tile_id": None,
            "owner_execution_rejection_reason": "unverified_full_page_balloon_interior",
            "owner_graph_run_id": f"run-{page_id}",
            "owner_graph_origin_execution_id": f"execution-{page_id}",
            "owner_graph_page_source_sha256": "a" * 64 if page_id == "page_001" else "b" * 64,
            "execution_rejected": True,
            "derived_qa_status": "review_required",
            "write_authority": "revoked",
            "source_pixels_preserved": True,
            "committed": False,
            "blocking": True,
            "qa_action": "BLOCK",
            "visible": False,
            "render_policy": "review_required",
            "text": source,
            "original": source,
            "source_payload": source,
            "translated": translated,
            "translated_payload": translated,
            "qa_flags": ["owner_render_geometry_review"],
        }

    page1_id = "page_001"
    p1_keep = "component_page_001_kept"
    p1_a, p1_b = "component_page_001_split_a", "component_page_001_split_b"
    o1_keep = "observation_page_001_kept"
    o1_a, o1_b = "observation_page_001_split_a", "observation_page_001_split_b"
    kept_owner_id = "owner_page_001_kept"
    graph1 = graph(
        page1_id,
        [component(p1_keep, page1_id, 10), component(p1_a, page1_id, 40), component(p1_b, page1_id, 70)],
        [observation(o1_keep, page1_id, p1_keep, "HELLO"),
         observation(o1_a, page1_id, p1_a, "LEFT HALF"),
         observation(o1_b, page1_id, p1_b, "RIGHT HALF")],
        [{
            "owner_id": kept_owner_id, "page_id": page1_id,
            "component_ids": [p1_keep], "observation_ids": [o1_keep],
            "selected_observation_ids": [o1_keep], "semantic_role": "dialogue_body",
            "source_payload": "HELLO", "translated_payload": "OLÁ",
            "disposition": "owned", "state": "owned",
            "route_action": "translate_inpaint_render", "execution_tile_id": None,
            "action_mask_ref": None,
        }],
        [{"component_id": p1_keep, "decision": "owned", "owner_id": kept_owner_id,
          "reason": "fixture"}, retired_disposition(p1_a, o1_a, 40),
         retired_disposition(p1_b, o1_b, 70)],
    )
    page1_keep_layer = {
        "id": kept_owner_id, "owner_id": kept_owner_id, "page_id": page1_id,
        "component_ids": [p1_keep], "observation_ids": [o1_keep],
        "semantic_role": "dialogue_body", "route_action": "translate_inpaint_render",
        "action_mask_ref": None, "layout_region_ids": [], "qa_flags": [],
    }
    page1_review = review_layer(
        "retired_merge_candidate_page_001", page1_id, [p1_a, p1_b], [o1_a, o1_b],
        "LEFT HALF RIGHT HALF", "METADE ESQUERDA METADE DIREITA",
    )

    page2_id = "page_002"
    p2_a, p2_b = "component_page_002_split_a", "component_page_002_split_b"
    o2_a, o2_b = "observation_page_002_split_a", "observation_page_002_split_b"
    graph2 = graph(
        page2_id,
        [component(p2_a, page2_id, 10), component(p2_b, page2_id, 40)],
        [observation(o2_a, page2_id, p2_a, "FIRST LINE"),
         observation(o2_b, page2_id, p2_b, "SECOND LINE")],
        [],
        [retired_disposition(p2_a, o2_a, 10), retired_disposition(p2_b, o2_b, 40)],
    )
    page2_reviews = [
        review_layer("retired_split_candidate_page_002_a", page2_id, [p2_a], [o2_a], "FIRST LINE", "PRIMEIRA LINHA"),
        review_layer("retired_split_candidate_page_002_b", page2_id, [p2_b], [o2_b], "SECOND LINE", "SEGUNDA LINHA"),
    ]
    pages = [
        {"numero": 1, "text_layers": [page1_keep_layer, page1_review],
         "textos": deepcopy([page1_keep_layer, page1_review])},
        {"numero": 2, "text_layers": deepcopy(page2_reviews),
         "textos": deepcopy(page2_reviews)},
    ]
    graphs = [graph1, graph2]
    return {
        "paginas": pages,
        "estatisticas": {"total_paginas": 2},
        "qa": {"summary": {"total": 0}},
        "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
        "owner_graph_status": "verified",
        "page_owner_graphs": graphs,
        "owner_invariant_summary": build_owner_invariant_summary(graphs),
    }


def test_verified_project_persists_split_and_merged_retired_reviews():
    project = _two_page_retired_review_project()

    assert owner_project_validation_errors(project) == []
    canonical_owner_ids = {
        owner["owner_id"]
        for graph in project["page_owner_graphs"]
        for owner in graph["owners"]
    }
    review_layers = [
        layer
        for page in project["paginas"]
        for layer in page["text_layers"]
        if layer.get("candidate_owner_id")
    ]
    assert len(review_layers) == 3
    assert len(canonical_owner_ids) == 1
    assert all(layer["owner_id"] is None for layer in review_layers)
    assert all(layer["candidate_owner_id"] not in canonical_owner_ids for layer in review_layers)
    assert review_layers[0]["source_payload"] == "LEFT HALF RIGHT HALF"
    assert review_layers[0]["translated_payload"] == "METADE ESQUERDA METADE DIREITA"
    assert {layer["source_payload"] for layer in review_layers[1:]} == {
        "FIRST LINE",
        "SECOND LINE",
    }
    assert {layer["translated_payload"] for layer in review_layers[1:]} == {
        "PRIMEIRA LINHA",
        "SEGUNDA LINHA",
    }


def test_verified_project_writer_neutralization_preserves_review_candidate_contract():
    project = _two_page_retired_review_project()
    expected = {
        layer["candidate_owner_id"]: (
            layer["route_action"],
            layer["render_policy"],
            layer["text"],
            layer["source_payload"],
        )
        for page in project["paginas"]
        for layer in page["text_layers"]
        if layer.get("candidate_owner_id")
    }

    neutralize_project_compatibility_metadata(project)

    for page in project["paginas"]:
        for alias in ("text_layers", "textos"):
            for layer in page[alias]:
                candidate_id = layer.get("candidate_owner_id")
                if not candidate_id:
                    continue
                self_contract = (
                    layer["route_action"],
                    layer["render_policy"],
                    layer["text"],
                    layer["source_payload"],
                )
                assert self_contract == expected[candidate_id]
    assert owner_project_validation_errors(project) == []


def test_verified_project_rejects_unknown_owner_even_with_review_candidate_id():
    project = _two_page_retired_review_project()
    for page in project["paginas"]:
        for alias in ("text_layers", "textos"):
            for layer in page[alias]:
                if layer.get("candidate_owner_id"):
                    layer["owner_id"] = "owner_not_in_graph"

    errors = owner_project_validation_errors(project)

    assert any("unknown owner_id" in error for error in errors)


def test_verified_project_rejects_stale_review_candidate_snapshot_binding():
    project = _two_page_retired_review_project()
    for page in project["paginas"]:
        for alias in ("text_layers", "textos"):
            for layer in page[alias]:
                if layer.get("candidate_owner_id"):
                    layer["owner_graph_page_source_sha256"] = "f" * 64

    errors = owner_project_validation_errors(project)

    assert any("stale owner_graph_page_source_sha256" in error for error in errors)


def test_verified_project_rejects_review_candidate_with_write_authority():
    project = _two_page_retired_review_project()
    for page in project["paginas"]:
        for alias in ("text_layers", "textos"):
            for layer in page[alias]:
                if layer.get("candidate_owner_id"):
                    layer["write_authority"] = "granted"

    errors = owner_project_validation_errors(project)

    assert any("unsafe write_authority" in error for error in errors)


def test_verified_project_rejects_review_candidate_from_another_page():
    project = _two_page_retired_review_project()
    for alias in ("text_layers", "textos"):
        layer = project["paginas"][0][alias][1]
        layer["component_ids"] = ["component_page_002_split_a"]
        layer["observation_ids"] = ["observation_page_002_split_a"]
        layer["selected_observation_ids"] = ["observation_page_002_split_a"]

    errors = owner_project_validation_errors(project)

    assert any("unknown components" in error for error in errors)
    assert any("unknown observations" in error for error in errors)


def test_verified_project_persists_translation_rejection_attempt_evidence():
    project = _two_page_retired_review_project()
    graph = project["page_owner_graphs"][1]
    layer_id = "retired_split_candidate_page_002_a"
    layer_component = "component_page_002_split_a"
    attempt_id = "translation-attempt:0123456789abcdef01234567"
    attempt_sha256 = "d" * 64
    disposition = next(
        item
        for item in graph["component_dispositions"]
        if item["component_id"] == layer_component
    )
    disposition.update(
        {
            "reason": "owner_translation_rejected",
            "policy_evidence_ids": [attempt_id],
            "policy_reason": "translation validation exhausted; source pixels preserved for review",
        }
    )
    for alias in ("text_layers", "textos"):
        layer = next(
            item
            for item in project["paginas"][1][alias]
            if item["candidate_owner_id"] == layer_id
        )
        layer["translation_attempt_ids"] = [attempt_id]
        layer["translation_attempt_sha256s"] = [attempt_sha256]
    project["owner_invariant_summary"] = build_owner_invariant_summary(
        project["page_owner_graphs"]
    )

    assert owner_project_validation_errors(project) == []


def test_atomic_write_creates_project_json(tmp_path):
    path = tmp_path / "project.json"

    write_project_json_atomic(path, _project())

    assert (
        json.loads(path.read_text(encoding="utf-8"))["estatisticas"]["total_paginas"]
        == 1
    )
    assert not (tmp_path / "project.json.tmp").exists()


def test_project_writer_round_trips_style_raster_contract(tmp_path):
    path = tmp_path / "project.json"
    project = _project()
    contract = valid_owner_style_raster_contract().to_dict()
    project["paginas"][0]["text_layers"][0][
        "style_v2_raster_contract"
    ] = contract

    write_project_json_atomic(path, project)

    layer = json.loads(path.read_text(encoding="utf-8"))["paginas"][0][
        "text_layers"
    ][0]
    assert layer["style_v2_raster_contract"] == contract


def test_backup_created_before_overwrite(tmp_path):
    path = tmp_path / "project.json"
    write_project_json_atomic(path, _project())
    write_project_json_atomic(path, _project())

    assert list(tmp_path.glob("project.backup.*.json"))


def test_invalid_schema_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "project.json"
    write_project_json_atomic(path, _project())

    with pytest.raises(ValueError):
        write_project_json_atomic(path, {"paginas": "bad"})

    assert (
        json.loads(path.read_text(encoding="utf-8"))["estatisticas"]["total_paginas"]
        == 1
    )


def test_summary_mismatch_fails():
    project = _project()
    project["qa"]["summary"]["total"] = 2

    with pytest.raises(ValueError, match="qa.summary"):
        validate_project_consistency(project)


def test_page_count_mismatch_fails():
    project = _project()
    project["estatisticas"]["total_paginas"] = 2

    with pytest.raises(ValueError, match="total_paginas"):
        validate_project_consistency(project)


def test_log_summary_mismatch_fails():
    project = _project()
    project["log"] = {"summary": {"actual_pages": 99}}

    with pytest.raises(ValueError, match="log.summary"):
        validate_project_consistency(project)


def test_project_writer_rejects_layer_owner_not_in_graph():
    project = _project()
    project.update(
        {
            "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
            "owner_graph_status": "verified",
            "page_owner_graphs": [
                {
                    "schema_version": OWNER_GRAPH_SCHEMA_VERSION,
                    "page_id": "page_001",
                    "run_id": "run-project-writer-empty",
                    "origin_execution_id": "execution-project-writer-empty",
                    "page_source_sha256": "c" * 64,
                    "verification_status": "verified",
                    "components": [],
                    "observations": [],
                    "owners": [],
                    "projections": [],
                    "component_dispositions": [],
                    "violations": [],
                }
            ],
            "owner_invariant_summary": {
                "page_count": 1,
                "component_count": 0,
                "observation_count": 0,
                "owner_count": 0,
                "projection_count": 0,
                "violation_count": 0,
                "critical_violation_count": 0,
            },
        }
    )
    project["paginas"][0]["text_layers"][0].update(
        {
            "owner_id": "own_missing",
            "component_ids": ["cmp_missing"],
            "observation_ids": ["obs_missing"],
        }
    )

    with pytest.raises(ValueError, match="owner"):
        validate_project_consistency(project)


def test_project_writer_validates_every_present_text_alias():
    project = _verified_owner_project()
    project["paginas"][0]["text_layers"] = []
    project["paginas"][0]["textos"] = [
        {
            "id": "own_missing",
            "owner_id": "own_missing",
            "page_id": "page_001",
            "component_ids": ["cmp_missing"],
            "observation_ids": ["obs_missing"],
            "semantic_role": "dialogue_body",
            "route_action": "translate_inpaint_render",
            "qa_flags": [],
        }
    ]

    with pytest.raises(ValueError, match="owner"):
        validate_project_consistency(project)


def test_project_writer_rejects_owner_page_id_that_disagrees_with_container():
    project = _verified_owner_project()
    project["paginas"][0]["numero"] = 99

    with pytest.raises(ValueError, match="page|owner"):
        validate_project_consistency(project)


@pytest.mark.parametrize("invalid_count", ["1", 1.9, True])
def test_project_writer_rejects_non_integer_owner_summary_counts(invalid_count):
    project = _verified_owner_project()
    project["owner_invariant_summary"]["owner_count"] = invalid_count

    with pytest.raises(ValueError, match="summary|owner|integer"):
        validate_project_consistency(project)


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_owner_id",
        "duplicate_component_id",
        "duplicate_observation_id",
        "component_page_mismatch",
        "observation_page_mismatch",
        "owner_page_mismatch",
        "dangling_projection_owner",
        "dangling_observation_component",
        "missing_owner_semantic_role",
        "missing_owner_route_action",
        "missing_selected_observation_ids",
        "duplicate_context_projection",
        "disposition_owner_component_mismatch",
        "selected_observation_component_contamination",
        "selected_observation_incomplete_coverage",
        "whitespace_owner_identity",
        "numeric_component_identity",
        "whitespace_component_page_identity",
        "whitespace_observation_page_identity",
        "whitespace_owner_page_identity",
        "owned_owner_without_evidence",
    ],
)
def test_project_writer_rejects_verified_graph_identity_corruption(corruption):
    project = _verified_owner_project()
    graph = project["page_owner_graphs"][0]

    if corruption == "duplicate_owner_id":
        graph["owners"].append(json.loads(json.dumps(graph["owners"][0])))
    elif corruption == "duplicate_component_id":
        graph["components"].append(json.loads(json.dumps(graph["components"][0])))
    elif corruption == "duplicate_observation_id":
        graph["observations"].append(json.loads(json.dumps(graph["observations"][0])))
    elif corruption == "component_page_mismatch":
        graph["components"][0]["page_id"] = "page_999"
    elif corruption == "observation_page_mismatch":
        graph["observations"][0]["page_id"] = "page_999"
    elif corruption == "owner_page_mismatch":
        graph["owners"][0]["page_id"] = "page_999"
    elif corruption == "dangling_projection_owner":
        graph["projections"].append(
            {
                "owner_id": "own_missing",
                "tile_id": "tile_page_001_context",
                "role": "context_only",
                "bbox_page": [10, 20, 110, 80],
                "bbox_tile": [10, 20, 110, 80],
                "offset_xy": [0, 0],
            }
        )
        project["owner_invariant_summary"]["projection_count"] = 1
    elif corruption == "dangling_observation_component":
        graph["observations"][0]["component_ids"] = ["cmp_missing"]
    elif corruption == "missing_owner_semantic_role":
        graph["owners"][0].pop("semantic_role")
    elif corruption == "missing_owner_route_action":
        graph["owners"][0].pop("route_action")
    elif corruption == "missing_selected_observation_ids":
        graph["owners"][0].pop("selected_observation_ids")
    elif corruption == "duplicate_context_projection":
        projection = {
            "owner_id": "own_page_001_body",
            "tile_id": "tile_page_001_context",
            "role": "context_only",
            "bbox_page": [10, 20, 110, 80],
            "bbox_tile": [10, 20, 110, 80],
            "offset_xy": [0, 0],
        }
        graph["projections"].extend(
            [json.loads(json.dumps(projection)), json.loads(json.dumps(projection))]
        )
        project["owner_invariant_summary"]["projection_count"] = 2
    elif corruption == "disposition_owner_component_mismatch":
        graph["owners"][0]["component_ids"] = []
        project["paginas"][0]["text_layers"][0]["component_ids"] = []
    elif corruption == "selected_observation_component_contamination":
        graph["components"].append(
            {
                "component_id": "cmp_page_001_preserved",
                "page_id": "page_001",
                "bbox_page": [130, 20, 180, 80],
                "polygon_page": [[130, 20], [180, 20], [180, 80], [130, 80]],
                "detector_sources": ["fixture"],
            }
        )
        graph["component_dispositions"].append(
            {
                "component_id": "cmp_page_001_preserved",
                "decision": "preserve",
                "owner_id": None,
                "reason": "fixture",
            }
        )
        graph["observations"][0]["component_ids"].append("cmp_page_001_preserved")
        project["owner_invariant_summary"]["component_count"] = 2
    elif corruption == "selected_observation_incomplete_coverage":
        graph["components"].append(
            {
                "component_id": "cmp_page_001_body_2",
                "page_id": "page_001",
                "bbox_page": [10, 90, 110, 140],
                "polygon_page": [[10, 90], [110, 90], [110, 140], [10, 140]],
                "detector_sources": ["fixture"],
            }
        )
        graph["owners"][0]["component_ids"].append("cmp_page_001_body_2")
        graph["component_dispositions"].append(
            {
                "component_id": "cmp_page_001_body_2",
                "decision": "owned",
                "owner_id": "own_page_001_body",
                "reason": "fixture",
            }
        )
        project["paginas"][0]["text_layers"][0]["component_ids"].append(
            "cmp_page_001_body_2"
        )
        project["owner_invariant_summary"]["component_count"] = 2
    elif corruption == "whitespace_owner_identity":
        graph["owners"][0]["owner_id"] = " own_page_001_body "
        graph["component_dispositions"][0]["owner_id"] = " own_page_001_body "
        project["paginas"][0]["text_layers"][0]["owner_id"] = " own_page_001_body "
    elif corruption == "numeric_component_identity":
        graph["components"][0]["component_id"] = 7
        graph["observations"][0]["component_ids"] = [7]
        graph["owners"][0]["component_ids"] = [7]
        graph["component_dispositions"][0]["component_id"] = 7
        project["paginas"][0]["text_layers"][0]["component_ids"] = [7]
    elif corruption == "whitespace_component_page_identity":
        graph["components"][0]["page_id"] = " page_001 "
    elif corruption == "whitespace_observation_page_identity":
        graph["observations"][0]["page_id"] = " page_001 "
    elif corruption == "whitespace_owner_page_identity":
        graph["owners"][0]["page_id"] = " page_001 "
    elif corruption == "owned_owner_without_evidence":
        graph["owners"][0]["component_ids"] = []
        graph["owners"][0]["observation_ids"] = []
        graph["owners"][0]["selected_observation_ids"] = []
        graph["component_dispositions"][0].update(
            {"decision": "preserve", "owner_id": None}
        )
        project["paginas"][0]["text_layers"][0]["component_ids"] = []
        project["paginas"][0]["text_layers"][0]["observation_ids"] = []
    else:  # pragma: no cover - guards the parametrized fixture itself
        raise AssertionError(f"unknown corruption fixture: {corruption}")

    with pytest.raises(ValueError):
        validate_project_consistency(project)


def test_project_writer_does_not_let_empty_paginas_hide_v12_pages():
    project = _verified_owner_project()
    project["paginas"] = []
    project["pages"] = [{"page": 2, "regions": []}]
    project["source"] = {"page_count": 1}
    project["estatisticas"]["total_paginas"] = 0

    with pytest.raises(ValueError, match="page|container|owner"):
        validate_project_consistency(project)


def test_project_writer_rejects_verified_graph_without_materialized_page():
    project = _verified_owner_project()
    project["paginas"] = []
    project["source"] = {"page_count": 1}
    project["estatisticas"]["total_paginas"] = 0

    with pytest.raises(ValueError, match="page|container|owner"):
        validate_project_consistency(project)


def test_project_writer_rejects_two_populated_page_container_families():
    project = _verified_owner_project()
    project["pages"] = [
        {
            "page": 1,
            "regions": json.loads(json.dumps(project["paginas"][0]["text_layers"])),
        }
    ]

    with pytest.raises(ValueError, match="page|container"):
        validate_project_consistency(project)


def test_legacy_unverified_project_cannot_publish_owner_summary_claims():
    project = _project()
    project.update(
        {
            "owner_graph_schema_version": OWNER_GRAPH_SCHEMA_VERSION,
            "owner_graph_status": "legacy_unverified",
            "page_owner_graphs": [],
            "owner_invariant_summary": {"owner_count": 1},
        }
    )

    with pytest.raises(ValueError, match="legacy|summary|owner"):
        validate_project_consistency(project)


def test_project_writer_rejects_duplicate_page_owner_graphs():
    project = _verified_owner_project()
    project["page_owner_graphs"].append(
        json.loads(json.dumps(project["page_owner_graphs"][0]))
    )
    project["owner_invariant_summary"].update(
        {
            "page_count": 2,
            "component_count": 2,
            "observation_count": 2,
            "owner_count": 2,
        }
    )

    with pytest.raises(ValueError):
        validate_project_consistency(project)


def test_project_writer_rejects_verified_text_layer_without_owner_id():
    project = _verified_owner_project()
    project["paginas"][0]["text_layers"][0].pop("owner_id")

    with pytest.raises(ValueError, match="owner"):
        validate_project_consistency(project)


def test_project_writer_rejects_verified_owner_without_text_layer():
    project = _verified_owner_project()
    project["paginas"][0]["text_layers"] = []

    with pytest.raises(ValueError, match="owner|layer"):
        validate_project_consistency(project)


def test_project_writer_rejects_verified_owner_summary_mismatch():
    project = _verified_owner_project()
    project["owner_invariant_summary"]["owner_count"] = 99

    with pytest.raises(ValueError, match="summary|owner"):
        validate_project_consistency(project)


def test_project_writer_rejects_verified_graph_with_critical_violation():
    project = _verified_owner_project()
    project["page_owner_graphs"][0]["violations"] = [
        {
            "code": "source_text_unowned",
            "severity": "critical",
            "message": "fixture corruption",
            "offenders": ["cmp_page_001_body"],
        }
    ]
    project["owner_invariant_summary"].update(
        {"violation_count": 1, "critical_violation_count": 1}
    )

    with pytest.raises(ValueError, match="owner|graph|critical|violation"):
        validate_project_consistency(project)


def test_project_writer_rejects_verified_status_without_page_graphs():
    project = _verified_owner_project()
    project["page_owner_graphs"] = []
    project["owner_invariant_summary"].update(
        {
            "page_count": 0,
            "component_count": 0,
            "observation_count": 0,
            "owner_count": 0,
        }
    )

    with pytest.raises(ValueError, match="owner|graph|verified"):
        validate_project_consistency(project)


def test_editor_save_refreshes_stale_log_summary(tmp_path):
    path = tmp_path / "project.json"
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "layer-a",
                        "traduzido": "Ola",
                        "translated": "Ola",
                        "qa_flags": [],
                    }
                ]
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "log": {
            "summary": {
                "actual_pages": 1,
                "processed_pages": 1,
                "translated_regions": 0,
                "qa_flags": 0,
                "critical_flags": 0,
            }
        },
    }

    _save_project_json(path, project)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["log"]["summary"]["translated_regions"] == 1


def test_project_writer_normalizes_sfx_policies_without_losing_sfx_metadata(tmp_path):
    path = tmp_path / "project.json"
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "sfx-1",
                        "tipo": "text",
                        "content_class": "sfx",
                        "route_action": "translate_sfx_inpaint_render",
                        "translate_policy": "translate",
                        "render_policy": "normal",
                        "sfx": {"source_text": "\ucff5", "adapted_text": "TUM"},
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 0}},
    }

    write_project_json_atomic(path, project)

    layer = json.loads(path.read_text(encoding="utf-8"))["paginas"][0]["text_layers"][0]
    assert layer["tipo"] == "sfx"
    assert layer["content_class"] == "sfx"
    assert layer["translate_policy"] == "adapt_sfx"
    assert layer["render_policy"] == "sfx_style"
    assert layer["sfx"]["adapted_text"] == "TUM"


def test_project_writer_removes_stale_sfx_policy_from_normal_text(tmp_path):
    path = tmp_path / "project.json"
    project = {
        "paginas": [
            {
                "text_layers": [
                    {
                        "id": "text-1",
                        "content_class": "text",
                        "route_action": "translate_inpaint_render",
                        "translate_policy": "adapt_sfx",
                        "render_policy": "sfx_style",
                        "qa_flags": [],
                    }
                ],
            }
        ],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 0}},
    }

    write_project_json_atomic(path, project)

    layer = json.loads(path.read_text(encoding="utf-8"))["paginas"][0]["text_layers"][0]
    assert layer["tipo"] == "text"
    assert layer["content_class"] == "text"
    assert layer["translate_policy"] == "translate"
    assert layer["render_policy"] == "normal"


def test_owner_render_quality_round_trips_through_project_writer(tmp_path):
    path = tmp_path / "project.json"
    quality = {
        "schema_version": 1,
        "status": "ok",
        "font_size_final": 28,
        "minimum_legible_font_px": 14,
        "source_ink_height_px": 22.0,
        "render_ink_height_px": 22,
        "source_x_height_px": 15.4,
        "render_x_height_px": 15.4,
        "source_scale_ratio": 1.0,
        "x_height_ratio": 1.0,
        "rendered_line_core_heights_px": [22],
        "safe_height_occupancy": 0.40,
        "safe_area_occupancy": 0.16,
        "wrapped_line_count": 1,
        "containment_status": "ok",
        "outside_safe_pixels": 0,
        "page_width": 360,
        "page_height": 280,
        "reasons": [],
    }
    project = {
        "paginas": [{
            "text_layers": [{
                "id": "owner-1",
                "owner_render_quality": quality,
                "render_layout_contract": {"owner_render_quality": quality},
                "qa_flags": [],
            }],
        }],
        "estatisticas": {"total_paginas": 1},
        "qa": {"summary": {"total": 0}},
    }

    write_project_json_atomic(path, project)

    layer = json.loads(path.read_text(encoding="utf-8"))["paginas"][0]["text_layers"][0]
    assert layer["owner_render_quality"] == quality
    assert layer["render_layout_contract"]["owner_render_quality"] == quality
