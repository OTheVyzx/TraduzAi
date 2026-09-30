"""Publish an explicit legacy-operational AnalysisRecord and renderer lineage.

This adapter does not claim fresh Vision inference.  It binds an authenticated
operational project, its existing renderer raster, and Vision-owned authority
masks into one immutable revision for an integration smoke.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from integration_v1.contracts import AnalysisRecord
from ownership.hash_contract import canonical_json_sha256
from typesetter.analysis_lineage import RendererAnalysisLineageReceipt
from typesetter.preference_contract import PreferenceCandidate
from typesetter.raster_safety import assess_raster_safety
from typesetter.recipe_contract import ExactLinePlan, RendererRecipe


def _bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    encoded = _bytes(value)
    path.write_bytes(encoded)
    return encoded


def _artifact_ref(kind: str, path: Path, root: Path) -> dict[str, object]:
    return {
        "kind": kind,
        "artifact_ref": path.relative_to(root).as_posix(),
        "sha256": _file_sha(path),
    }


def publish(
    *,
    operational_root: Path,
    vision_receipt_path: Path,
    output_dir: Path,
    font_path: Path,
) -> dict[str, object]:
    vision = json.loads(vision_receipt_path.read_text(encoding="utf-8"))
    case = vision["case"]
    owner_id = str(case["owner_id"])
    source_index = int(case["source_index"])
    page_id = str(case["page_id"])
    page_dir = operational_root / "by_source" / f"{source_index:03d}"
    layer_path = page_dir / "text_layers" / f"{page_id}.json"
    layer_doc = json.loads(layer_path.read_text(encoding="utf-8"))
    matches = [row for row in layer_doc["texts"] if row.get("owner_id") == owner_id]
    if len(matches) != 1:
        raise ValueError("operational owner must resolve to exactly one text layer")
    layer = matches[0]
    geometry = layer["owner_render_geometry"]
    if geometry.get("geometry_sha256") != vision["binding"]["owner_render_geometry_sha256"]:
        raise ValueError("Vision authority belongs to another owner geometry revision")

    project_path = operational_root / "project.json"
    source_path = operational_root / "source_members" / f"{source_index:03d}.webp"
    if _file_sha(source_path) != vision["binding"]["source_member_sha256"]:
        raise ValueError("operational source differs from Vision authority")
    overlay_rel = next(
        row["path"] for row in layer_doc["raster_cache"] if row["owner_id"] == owner_id
    )
    overlay_path = operational_root / overlay_rel
    authorized_path = Path(vision["artifacts"]["authorized_body_mask"]["path"])
    protected_path = Path(vision["artifacts"]["protected_art_mask"]["path"])
    for key, path in (("authorized_body_mask", authorized_path), ("protected_art_mask", protected_path)):
        if _file_sha(path) != vision["artifacts"][key]["file_sha256"]:
            raise ValueError(f"Vision {key} file hash mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    copied = {
        "source": artifacts_dir / "source-041.webp",
        "raster": artifacts_dir / "renderer-rgba-041.png",
        "authorized": artifacts_dir / "authorized-body-041.png",
        "protected": artifacts_dir / "protected-art-041.png",
    }
    for source, destination in (
        (source_path, copied["source"]),
        (overlay_path, copied["raster"]),
        (authorized_path, copied["authorized"]),
        (protected_path, copied["protected"]),
    ):
        shutil.copyfile(source, destination)

    logical_id = f"logical:{owner_id}"
    component_ids = tuple(str(value) for value in layer["component_ids"])
    physical_ids = tuple(f"physical:{component_id}" for component_id in component_ids)
    logical = {
        "logical_units": [{
            "logical_unit_id": logical_id,
            "owner_id": owner_id,
            "component_ids": list(component_ids),
            "source_payload": layer["source_payload"],
            "target_payload": layer["translated_payload"],
            "adapter": "legacy_operational_adapter",
        }]
    }
    physical = {
        "physical_subblocks": [
            {
                "physical_subblock_id": physical_id,
                "logical_unit_id": logical_id,
                "component_id": component_id,
                "bbox": next(
                    row["bbox_page"]
                    for row in geometry["components"]
                    if row["component_id"] == component_id
                ),
            }
            for physical_id, component_id in zip(physical_ids, component_ids)
        ]
    }
    observations = {
        "observations": geometry["selected_observations"],
        "adapter": "legacy_operational_adapter",
    }
    selection = {
        "kind": "legacy_operational_owner_selection",
        "owner_id": owner_id,
        "selected_observation_ids": layer["selected_observation_ids"],
        "source": layer_path.as_posix(),
        "source_file_sha256": _file_sha(layer_path),
    }
    transform = {
        "kind": "identity",
        "from_space": "source",
        "to_space": "logical_page",
        "inverse": "identity",
    }
    relations = {"relations": []}
    reading_order = {"reading_order": list(physical_ids)}
    contour = {
        "coordinate_space": "logical_page",
        "polygon": geometry["layout_container_polygon_page"],
        "source": geometry["layout_container_source"],
    }
    artifact_values = {
        "logical-units.json": logical,
        "physical-subblocks.json": physical,
        "ocr-observations.json": observations,
        "selection.json": selection,
        "transform.json": transform,
        "relations.json": relations,
        "reading-order.json": reading_order,
        "container-contour.json": contour,
    }
    artifact_paths: dict[str, Path] = {}
    for name, value in artifact_values.items():
        path = artifacts_dir / name
        _write_json(path, value)
        artifact_paths[name] = path

    analysis_config = {
        "adapter": "legacy_operational_adapter",
        "schema_version": 1,
        "owner_id": owner_id,
        "project_sha256": _file_sha(project_path),
        "vision_authority_receipt_sha256": _file_sha(vision_receipt_path),
    }
    refs = [
        _artifact_ref(name.removesuffix(".json").replace("-", "_"), path, output_dir)
        for name, path in artifact_paths.items()
    ] + [
        _artifact_ref("writing_body_mask", copied["authorized"], output_dir),
        _artifact_ref("protected_art_mask", copied["protected"], output_dir),
        _artifact_ref("renderer_rgba", copied["raster"], output_dir),
    ]
    record = AnalysisRecord.build({
        "status": "complete",
        "source_sha256": _file_sha(copied["source"]),
        "authenticated_neighbor_sha256s": [],
        "region": {
            "bbox": geometry["layout_container_bbox_page"],
            "coordinate_space": "logical_page",
        },
        "coordinate_space": "logical_page",
        "transform_sha256": _file_sha(artifact_paths["transform.json"]),
        "source_language": "en",
        "analysis_config_sha256": canonical_json_sha256(analysis_config),
        "provider_family": "legacy_operational_adapter",
        "provider_name": "TraduzAI operational project adapter",
        "provider_model": "project-v12-owner-evidence",
        "provider_version": _file_sha(project_path),
        "capability_version": "legacy-operational-adapter-v1",
        "artifact_refs": refs,
        "transform_ref": {
            "artifact_ref": artifact_paths["transform.json"].relative_to(output_dir).as_posix(),
            "kind": "identity",
            "sha256": _file_sha(artifact_paths["transform.json"]),
            "inverse_sha256": _file_sha(artifact_paths["transform.json"]),
        },
        "ocr_observations": {
            "artifact_ref": artifact_paths["ocr-observations.json"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(artifact_paths["ocr-observations.json"]),
            "count": len(observations["observations"]),
        },
        "selected_observation_id": layer["selected_observation_ids"][0],
        "selection_provenance": {
            "artifact_ref": artifact_paths["selection.json"].relative_to(output_dir).as_posix(),
            "kind": "legacy_operational_owner_selection",
            "sha256": _file_sha(artifact_paths["selection.json"]),
        },
        "logical_units": {
            "artifact_ref": artifact_paths["logical-units.json"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(artifact_paths["logical-units.json"]),
            "count": 1,
        },
        "physical_subblocks": {
            "artifact_ref": artifact_paths["physical-subblocks.json"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(artifact_paths["physical-subblocks.json"]),
            "count": len(physical_ids),
        },
        "relations": {
            "artifact_ref": artifact_paths["relations.json"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(artifact_paths["relations.json"]),
            "count": 0,
        },
        "reading_order": {
            "artifact_ref": artifact_paths["reading-order.json"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(artifact_paths["reading-order.json"]),
            "count": len(physical_ids),
        },
        "container_contour_ref": {
            "artifact_ref": artifact_paths["container-contour.json"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(artifact_paths["container-contour.json"]),
        },
        "writing_body_ref": {
            "artifact_ref": copied["authorized"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(copied["authorized"]),
        },
        "tail_ref": None,
        "dependency_hashes": {
            "analysis_config": canonical_json_sha256(analysis_config),
            "operational_project": _file_sha(project_path),
            "source_pixels": _file_sha(copied["source"]),
            "vision_authority": _file_sha(vision_receipt_path),
        },
        "mask_channels": {
            name: {"state": "unknown", "reason_code": "legacy_adapter_did_not_infer_channel"}
            for name in ("glyph", "outline", "shadow", "glow", "ignore_or_uncertain")
        },
    })
    analysis_path = output_dir / "analysis-record.json"
    _write_json(analysis_path, record.to_dict())

    lines = tuple(str(value) for value in layer["render_layout_contract"]["lines"])
    target_text = " ".join(lines)
    plan_body = {
        "owner_id": owner_id,
        "lines": list(lines),
        "metrics": {
            "font_size_px": int(layer["render_layout_contract"]["font_size"]),
            "line_advance_px": int(layer["render_layout_contract"]["line_advance"]),
        },
        "usable_body": {
            "bbox": geometry["layout_container_bbox_page"],
            "coordinate_space": "logical_page",
        },
        "dependency_hashes": {"analysis_record": record.analysis_record_sha256},
        "adapter": "legacy_operational_adapter",
    }
    plan = plan_body | {"plan_sha256": canonical_json_sha256(plan_body)}
    rgba = np.asarray(Image.open(copied["raster"]).convert("RGBA"), dtype=np.uint8)
    authorized = np.asarray(Image.open(copied["authorized"]).convert("L"), dtype=np.uint8)
    protected = np.asarray(Image.open(copied["protected"]).convert("L"), dtype=np.uint8)
    bbox = tuple(int(value) for value in layer["render_bbox"])
    safety = assess_raster_safety(
        alpha=rgba[:, :, 3], bbox=bbox,
        authorized_body_mask=authorized, protected_art_mask=protected,
    )
    font = {
        "family": font_path.stem,
        "relative_path": "fonts/commercial/" + font_path.name,
        "sha256": _file_sha(font_path),
    }
    effects = {
        "fill": layer["visual_profile_v2"]["applied_style"]["cor"],
        "adapter": "legacy_operational_adapter",
    }
    recipe = RendererRecipe.build(
        owner_id=owner_id,
        source_sha256=record.payload["source_sha256"],
        output_sha256=sha256(np.ascontiguousarray(rgba).tobytes()).hexdigest(),
        target_text=target_text,
        font=font,
        rasterizer={
            "runtime_id": "legacy-operational-raster-adapter-v1",
            "runtime_sha256": _file_sha(Path(__file__)),
            "config_sha256": canonical_json_sha256(layer["render_layout_contract"]),
        },
        line_plan=ExactLinePlan.build(
            target_text=target_text,
            lines=lines,
            separators=(" ",) * (len(lines) - 1),
        ),
        bbox=bbox,
        font_size_px=plan["metrics"]["font_size_px"],
        line_advance_px=plan["metrics"]["line_advance_px"],
        effects=effects,
        anchors={"mode": "legacy_operational_bbox", "owner_geometry_sha256": geometry["geometry_sha256"]},
        geometry={
            "coordinate_space": "logical_page",
            "usable_body_sha256": canonical_json_sha256(plan["usable_body"]),
            "alpha_sha256": safety["alpha_sha256"],
        },
        policy_versions={"layout": "legacy-operational-adapter-v1"},
        dependency_hashes={
            "analysis_record": record.analysis_record_sha256,
            "layout_plan": plan["plan_sha256"],
        },
    )
    candidate = PreferenceCandidate.build(
        owner_id=owner_id,
        target_text=target_text,
        source_sha256=record.payload["source_sha256"],
        style_sha256=canonical_json_sha256({"font": font, "effects": effects}),
        layout_plan_sha256=plan["plan_sha256"],
        recipe_sha256=recipe.recipe_sha256,
        output_sha256=recipe.output_sha256,
        preview_ref={
            "relative_path": copied["raster"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(copied["raster"]),
        },
        context_ref={
            "relative_path": copied["source"].relative_to(output_dir).as_posix(),
            "sha256": _file_sha(copied["source"]),
        },
        metrics={
            "analysis_record_sha256": record.analysis_record_sha256,
            "raster_safety": safety,
        },
        hard_safety_passed=safety["status"] == "pass",
    )

    route = {
        "schema": "traduzai.consumer-owner-route.v1",
        "analysis_record_sha256": record.analysis_record_sha256,
        "units": [{
            "owner_id": owner_id,
            "logical_unit_id": logical_id,
            "status": "translation_ready",
            "target_text": target_text,
        }],
    }
    route_bytes = _write_json(output_dir / "owner-route.json", route)
    binding_body = {
        "owner_id": owner_id,
        "analysis_record_sha256": record.analysis_record_sha256,
        "logical_unit_id": logical_id,
        "physical_subblock_ids": list(physical_ids),
        "route_state": "ready_for_layout",
        "route_evidence_sha256": sha256(route_bytes).hexdigest(),
        "producer": {
            "runtime_id": "integration-legacy-operational-adapter-v1",
            "runtime_sha256": _file_sha(Path(__file__)),
        },
    }
    binding = binding_body | {"binding_sha256": canonical_json_sha256(binding_body)}
    event_log = {
        "schema": "traduzai.renderer-provider-events.v1",
        "analysis_record_sha256": record.analysis_record_sha256,
        "events": [
            {"kind": "stage_started", "stage": "renderer"},
            {"kind": "stage_completed", "stage": "renderer"},
        ],
    }
    event_bytes = _write_json(output_dir / "renderer-events.json", event_log)
    trace_body = {
        "schema": "traduzai.renderer-provider-trace.v1",
        "analysis_record_sha256": record.analysis_record_sha256,
        "scope": "renderer",
        "provider_calls": {"ocr": 0, "analysis_discovery": 0},
        "event_log_sha256": sha256(event_bytes).hexdigest(),
        "producer": {
            "runtime_id": "renderer-lineage-adapter-v1",
            "runtime_sha256": _file_sha(Path(__file__)),
        },
    }
    trace = trace_body | {"trace_sha256": canonical_json_sha256(trace_body)}
    receipt = RendererAnalysisLineageReceipt.build(
        analysis_record=record.to_dict(),
        owner_binding=binding,
        provider_trace=trace,
        logical_units_artifact_bytes=artifact_paths["logical-units.json"].read_bytes(),
        physical_subblocks_artifact_bytes=artifact_paths["physical-subblocks.json"].read_bytes(),
        owner_route_evidence_bytes=route_bytes,
        renderer_event_log_bytes=event_bytes,
        layout_plan=plan,
        recipe=recipe.to_dict(),
        raster_candidate=candidate.to_dict(),
        raster_rgba=rgba,
        authorized_body_mask=authorized,
        protected_art_mask=protected,
    )
    for name, value in (
        ("layout-plan.json", plan),
        ("renderer-recipe.json", recipe.to_dict()),
        ("raster-candidate.json", candidate.to_dict()),
        ("owner-binding.json", binding),
        ("provider-trace.json", trace),
        ("lineage-receipt.json", receipt.to_dict()),
    ):
        _write_json(output_dir / name, value)
    manifest = {
        "schema": "traduzai.integration.legacy-operational-lineage.v1",
        "adapter": "legacy_operational_adapter",
        "source_index": source_index,
        "owner_id": owner_id,
        "analysis_record_sha256": record.analysis_record_sha256,
        "lineage_receipt_sha256": receipt.receipt_sha256,
        "state": receipt.state,
        "publication_allowed": receipt.publication_allowed,
        "raster_safety_status": receipt.raster_safety_status,
        "outside_authorized_body_px": safety["outside_authorized_body_px"],
        "protected_art_overlap_px": safety["protected_art_overlap_px"],
        "inputs": {
            "operational_project_sha256": _file_sha(project_path),
            "vision_authority_receipt_sha256": _file_sha(vision_receipt_path),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operational-root", type=Path, required=True)
    parser.add_argument("--vision-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(
        operational_root=args.operational_root,
        vision_receipt_path=args.vision_receipt,
        output_dir=args.output_dir,
        font_path=args.font,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
