"""Persistence boundaries for owners and retired candidates, without inference."""
import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

import main
from ownership.project import (
    OWNER_LAYER_CONTRACT_FIELDS,
    owner_project_validation_errors,
)
from project_writer import validate_project_consistency
from test_project_writer import _two_page_retired_review_project


def _build(graphs, layers, *, config=None):
    ocr = [{"_owner_graph_mode": "enforce", "_owner_graph_snapshot": g} for g in graphs]
    return main.build_project_json(
        config or {}, {}, ocr, [{"texts": p} for p in layers],
        [Path(f"{i:03}.png") for i in range(1, len(graphs) + 1)], len(graphs), 0.0,
    )


def _exercise(project, path):
    for index, page in enumerate(project["paginas"], 1):
        page["text_layers"] = [main._normalize_text_layer_for_renderer(l, index, i)
                               for i, l in enumerate(page["text_layers"])]
    main._normalize_final_project_page_space_layers(project)
    main._ensure_project_route_action_contract(project)
    assert owner_project_validation_errors(project) == []
    expected_input = deepcopy(project)
    main._save_project_json(path, project)
    saved = json.loads(path.read_text(encoding="utf-8"))
    validate_project_consistency(saved)
    assert owner_project_validation_errors(saved) == []
    for page in saved["paginas"]:
        for layer, alias in zip(page["text_layers"], page["textos"], strict=True):
            for field in OWNER_LAYER_CONTRACT_FIELDS:
                assert layer.get(field) == alias.get(field), field
            if layer.get("candidate_owner_id") is not None:
                assert layer["owner_id"] is None
                assert layer["write_authority"] == "revoked"
                assert layer["action_mask_ref"] is None
                assert layer["layout_region_ids"] == []
                assert layer["committed"] is False
                assert layer["visible"] is False
    capture = json.loads((path.parent / "debug" / "project_save_input.json").read_text(encoding="utf-8"))
    assert capture == expected_input
    assert not path.with_suffix(".json.tmp").exists()
    return saved


def _fixture_project():
    seed = _two_page_retired_review_project()
    layers = deepcopy([p["text_layers"] for p in seed["paginas"]])
    for graph, page_layers in zip(seed["page_owner_graphs"], layers, strict=True):
        components = {c["component_id"]: c for c in graph["components"]}
        owners = {o["owner_id"]: o for o in graph["owners"]}
        for layer in page_layers:
            boxes = [components[c]["bbox_page"] for c in layer["component_ids"]]
            box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes)]
            layer.update(bbox=box, source_bbox=box, text_pixel_bbox=box)
            if layer.get("candidate_owner_id"):
                layer["source_payload_sha256"] = sha256(layer["source_payload"].encode("utf-8")).hexdigest()
                layer["target_payload_sha256"] = sha256(layer["translated_payload"].encode("utf-8")).hexdigest()
            if layer.get("owner_id"):
                owner = owners[layer["owner_id"]]
                layer.update(text=owner["source_payload"], original=owner["source_payload"],
                             source_payload=owner["source_payload"], visible=False,
                             render_policy="preserve_original")
    return _build(seed["page_owner_graphs"], layers)


def test_build_renderer_alias_writer_atomic_readback_preserves_contract(tmp_path):
    project = _fixture_project()
    saved = _exercise(project, tmp_path / "project.json")
    assert len(saved["paginas"]) == 2


@pytest.mark.parametrize("field,value", [("write_authority", "granted"),
                                         ("visible", True), ("render_policy", "normal"),
                                         ("source_payload_sha256", "0" * 64),
                                         ("target_payload_sha256", "0" * 64)])
def test_invalid_candidate_remains_invalid_across_boundaries(tmp_path, field, value):
    project = _fixture_project()
    layer = next(l for p in project["paginas"] for l in p["text_layers"]
                 if l.get("candidate_owner_id"))
    layer[field] = value
    with pytest.raises((ValueError, AssertionError)):
        _exercise(project, tmp_path / "project.json")
    assert not (tmp_path / "project.json").exists()


def test_null_candidate_marker_does_not_change_legacy_layer_routing():
    from project_writer import neutralize_project_compatibility_metadata
    layer = {"id": "legacy", "candidate_owner_id": None, "route_action": "translate_inpaint_render",
             "render_policy": "sfx_style", "translate_policy": "adapt_sfx", "content_class": "text"}
    project = {"paginas": [{"text_layers": [layer]}]}
    neutralize_project_compatibility_metadata(project)
    assert layer["render_policy"] == "normal"
    assert layer["translate_policy"] == "translate"
    assert layer["skip_processing"] is False


def test_real_three_page_save_boundary_reconstruction(tmp_path):
    raw_root = os.environ.get("TRADUZAI_SAVE_REPLAY_SOURCE")
    if not raw_root:
        pytest.skip("requires authorized existing three-page evidence directory")
    source = Path(raw_root)
    from ownership.execution import recover_page_candidate_transaction

    private_roots = list((source / ".owner-private").iterdir())
    assert len(private_roots) == 1
    private = private_roots[0]
    graphs, layers, snapshots, source_hashes = [], [], [], {}
    for page_id in sorted(p.name for p in (private / ".page-current").iterdir()):
        ref = recover_page_candidate_transaction(private, page_id=page_id)
        assert ref is not None
        result = ref.read_verified(private)
        payload = result.to_canonical_dict()
        assert result.final_page is not None and result.terminal_proof is not None
        graphs.append(payload["owner_graph"])
        layers.append(payload["text_layers"]["texts"])
        snapshots.append(payload)
        evidence_path = private / ref.page_execution_evidence_relative_path
        source_hashes[str(evidence_path)] = sha256(evidence_path.read_bytes()).hexdigest()
    assert len(graphs) == 3
    config = json.loads((source / "debug/e2e/00_run/config_snapshot.json").read_text(encoding="utf-8"))
    # Reconstruction from recorded owner evidence; optional runtime OCR metadata
    # and the original in-memory project_data are not available in this journal.
    config.pop("work_dir", None)
    project = _build(graphs, deepcopy(layers), config=config)
    gate_path = source / "debug/e2e/11_qa_export_gate/export_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "BLOCK" and gate["allowed"] is False and gate["override"] is False
    project["qa"]["export_gate"] = deepcopy(gate)
    project["needs_review"] = True
    project["output_review_state"] = "blocked_preview"
    artifact_hashes = {}
    for i, page in enumerate(project["paginas"], 1):
        for key, relative in (("base", f"originals/{i:03}.png"), ("rendered", f"translated/{i:03}.png")):
            artifact = source / relative
            page["image_layers"][key]["path"] = str(artifact)
            artifact_hashes[str(artifact)] = sha256(artifact.read_bytes()).hexdigest()
    saved = _exercise(project, tmp_path / "project.json")
    assert saved["qa"]["export_gate"] == gate
    assert saved["needs_review"] is True
    assert saved["output_review_state"] == "blocked_preview"
    for path, digest in {**source_hashes, **artifact_hashes}.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == digest
    report = {"kind": "save_boundary_reconstruction_from_recorded_evidence",
              "exact_original_project_data_replay": False,
              "pages": 3, "candidate_count": sum(l.get("candidate_owner_id") is not None for p in layers for l in p),
              "evidence_sha256": source_hashes, "artifact_sha256": artifact_hashes,
              "project_sha256": sha256((tmp_path / "project.json").read_bytes()).hexdigest(),
              "gate_status": gate["status"], "allowed": gate["allowed"], "override": gate["override"]}
    (tmp_path / "replay_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
