"""Discover deterministic owner candidates for manual Style V2 target curation."""

from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


class OwnerTargetError(ValueError):
    """Raised when a curated owner target cannot be proven against its graph."""


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def source_crop_contract(
    image_path: Path,
    bbox_page: list[int],
    *,
    artifact_bbox_frame: list[int] | None = None,
    coordinate_space: str | None = None,
) -> dict[str, Any]:
    """Hash native source pixels and dimensions for an immutable target crop."""

    from PIL import Image

    with Image.open(image_path) as image:
        source = image.convert("RGB")
        crop_bbox = artifact_bbox_frame or bbox_page
        crop = source.crop(tuple(int(value) for value in crop_bbox))
        digest = sha256()
        digest.update(f"RGB:{crop.width}x{crop.height}\0".encode("ascii"))
        digest.update(crop.tobytes())
        payload = {
            "bbox_page": list(bbox_page),
            "width": crop.width,
            "height": crop.height,
            "sha256": digest.hexdigest(),
        }
        if coordinate_space is not None:
            payload["coordinate_space"] = str(coordinate_space)
            payload["artifact_bbox_frame"] = list(crop_bbox)
        return payload


def _page_by_id(project: dict[str, Any], page_id: str) -> dict[str, Any]:
    for index, page in enumerate(project.get("paginas") or project.get("pages") or [], start=1):
        if not isinstance(page, dict):
            continue
        candidate = str(page.get("page_id") or f"page_{int(page.get('numero') or index):03d}")
        if candidate == page_id:
            return page
    raise OwnerTargetError(f"page target not found: {page_id}")


def selected_owner_target(target: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    """Resolve one exact page/owner/component target, never clamping or guessing."""

    page_id = str(target.get("page_id") or "").strip()
    owner_id = str(target.get("owner_id") or "").strip()
    _page_by_id(project, page_id)
    graph = next(
        (
            graph
            for graph in project.get("page_owner_graphs") or []
            if isinstance(graph, dict) and str(graph.get("page_id") or "") == page_id
        ),
        None,
    )
    if graph is None:
        raise OwnerTargetError(f"page owner graph not found: {page_id}")
    owner = next(
        (
            row
            for row in graph.get("owners") or []
            if isinstance(row, dict) and str(row.get("owner_id") or "") == owner_id
        ),
        None,
    )
    if owner is None:
        raise OwnerTargetError(f"owner target not found: {page_id}/{owner_id}")
    expected_components = sorted(str(value) for value in target.get("component_ids") or [])
    actual_components = sorted(str(value) for value in owner.get("component_ids") or [])
    if not expected_components or expected_components != actual_components:
        raise OwnerTargetError(f"owner component target mismatch: {page_id}/{owner_id}")
    return {"page": _page_by_id(project, page_id), "graph": graph, "owner": owner}


def _owner_bbox(graph: dict[str, Any], owner: dict[str, Any]) -> list[int]:
    component_ids = {str(value) for value in owner.get("component_ids") or []}
    boxes = [
        list(component.get("bbox_page") or [])
        for component in graph.get("components") or []
        if isinstance(component, dict) and str(component.get("component_id") or "") in component_ids
    ]
    if not boxes or any(len(box) != 4 for box in boxes):
        raise OwnerTargetError(f"owner geometry missing: {owner.get('owner_id')}")
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def discover_owner_targets(
    project: dict[str, Any], *, project_root: Path | None = None
) -> list[dict[str, Any]]:
    """Return stable candidates; humans still assign category and inspection verdict."""

    targets: list[dict[str, Any]] = []
    for graph in project.get("page_owner_graphs") or []:
        if not isinstance(graph, dict):
            continue
        page_id = str(graph.get("page_id") or "")
        for owner in graph.get("owners") or []:
            if not isinstance(owner, dict):
                continue
            target = {
                "page_id": page_id,
                "owner_id": str(owner.get("owner_id") or ""),
                "component_ids": sorted(str(value) for value in owner.get("component_ids") or []),
                "bbox_page": _owner_bbox(graph, owner),
                "semantic_role": str(owner.get("semantic_role") or "unknown"),
                "state": str(owner.get("state") or "unknown"),
            }
            if project_root is not None:
                page = _page_by_id(project, page_id)
                base_path = ((page.get("image_layers") or {}).get("base") or {}).get("path")
                if not isinstance(base_path, str) or not base_path.strip():
                    raise OwnerTargetError(f"source artifact missing: {page_id}")
                target["source_crop"] = source_crop_contract(
                    (Path(project_root) / base_path).resolve(),
                    target["bbox_page"],
                )
            target["target_sha256"] = _canonical_hash(target)
            targets.append(target)
    return sorted(targets, key=lambda row: (row["page_id"], row["owner_id"]))


def build_effective_style_config(
    shared_config: dict[str, Any],
    *,
    required_categories: list[str],
) -> dict[str, Any]:
    """Create a style-only overlay without mutating functional matrix config."""

    effective = copy.deepcopy(shared_config)
    effective.update(
        {
            "style_copy_mode": "enforce",
            "style_inspection_required": True,
            "style_fidelity_required_categories": sorted(set(required_categories)),
        }
    )
    return effective


def verify_matrix_target(
    target: dict[str, Any],
    project: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Prove owner identity, logical crop and framed artifact binding."""

    from strip.page_surface_geometry import PageSurfaceGeometry

    resolved = selected_owner_target(target, project)
    page = resolved["page"]
    geometry_payload = page.get("page_surface_geometry")
    if not isinstance(geometry_payload, dict):
        raise OwnerTargetError("missing_page_surface_geometry")
    try:
        geometry = PageSurfaceGeometry.from_dict(geometry_payload)
    except (TypeError, ValueError) as exc:
        raise OwnerTargetError("invalid_page_surface_geometry") from exc
    if str(page.get("page_surface_geometry_sha256") or "") != geometry.geometry_sha256:
        raise OwnerTargetError("page_surface_geometry_hash_mismatch")
    source_crop = target.get("source_crop")
    if not isinstance(source_crop, dict):
        raise OwnerTargetError("source_crop_missing")
    if str(source_crop.get("coordinate_space") or "") != "logical_page":
        raise OwnerTargetError("source_crop_coordinate_space_invalid")
    if str(target.get("expected_artifact_space") or "") != "framed_page":
        raise OwnerTargetError("expected_artifact_space_invalid")
    logical_bbox = [int(value) for value in source_crop.get("bbox_page") or []]
    if len(logical_bbox) != 4:
        raise OwnerTargetError("source_crop_bbox_invalid")
    frame_bbox = list(geometry.logical_bbox_to_frame(tuple(logical_bbox)))
    base_path = ((page.get("image_layers") or {}).get("base") or {}).get("path")
    source_path = (
        (project_root / base_path).resolve()
        if isinstance(base_path, str) and base_path.strip()
        else None
    )
    if source_path is not None and source_path.is_file():
        actual = source_crop_contract(
            source_path,
            logical_bbox,
            artifact_bbox_frame=frame_bbox,
            coordinate_space="logical_page",
        )
        actual.pop("artifact_bbox_frame", None)
    else:
        authenticated = next(
            (
                row for row in page.get("authenticated_source_crops") or []
                if isinstance(row, dict)
                and str(row.get("owner_id") or "") == str(target.get("owner_id") or "")
            ),
            None,
        )
        if not isinstance(authenticated, dict):
            raise OwnerTargetError("source_artifact_missing")
        if str(authenticated.get("page_surface_geometry_sha256") or "") != geometry.geometry_sha256:
            raise OwnerTargetError("authenticated_source_geometry_mismatch")
        actual = dict(authenticated.get("source_crop") or {})
    for key in ("bbox_page", "width", "height", "sha256", "coordinate_space"):
        if actual.get(key) != source_crop.get(key):
            raise OwnerTargetError(f"source_crop_{key}_mismatch")
    return {
        "page_id": str(target.get("page_id") or ""),
        "owner_id": str(target.get("owner_id") or ""),
        "component_ids": sorted(str(value) for value in target.get("component_ids") or []),
        "source_crop": dict(source_crop),
        "artifact_bbox_frame": frame_bbox,
        "expected_artifact_space": "framed_page",
        "page_surface_geometry_sha256": geometry.geometry_sha256,
        "verified": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--entry-id")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    project = json.loads(args.project.read_text(encoding="utf-8-sig"))
    if args.verify_only:
        if args.matrix is None or not str(args.entry_id or "").strip():
            parser.error("--verify-only requires --matrix and --entry-id")
        matrix = json.loads(args.matrix.read_text(encoding="utf-8-sig"))
        entry = next(
            (
                row for row in matrix.get("entries") or []
                if isinstance(row, dict) and str(row.get("entry_id") or "") == args.entry_id
            ),
            None,
        )
        if entry is None:
            raise OwnerTargetError(f"matrix entry not found: {args.entry_id}")
        payload = {
            "schema_version": 1,
            "entry_id": args.entry_id,
            "verified_targets": [
                verify_matrix_target(
                    target,
                    project,
                    project_root=args.project.resolve().parent,
                )
                for target in entry.get("targets") or []
            ],
        }
    else:
        payload = {
            "schema_version": 1,
            "candidates": discover_owner_targets(project, project_root=args.project.resolve().parent),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
