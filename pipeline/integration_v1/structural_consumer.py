"""Consume a hash-bound Vision structural handoff without page-specific answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from integration_v1.providers import ProviderUnavailable, translate_complete_unit
from ownership.hash_contract import canonical_json_sha256, sha256_text


def _is_sha256(value: Any) -> bool:
    candidate = str(value or "")
    return len(candidate) == 64 and all(char in "0123456789abcdef" for char in candidate)


def _safe_relative_path(value: Any, field: str) -> Path:
    candidate = str(value or "")
    if (
        not candidate
        or "\\" in candidate
        or candidate.startswith("/")
        or ":" in candidate
        or any(part in {"", ".", ".."} for part in candidate.split("/"))
    ):
        raise ValueError(f"{field} must be a safe relative path")
    return Path(candidate)


def _read_verified_json(root: Path, reference: Mapping[str, Any], field: str) -> Any:
    relative = _safe_relative_path(reference.get("artifact_ref"), f"{field}.artifact_ref")
    expected = str(reference.get("sha256") or "")
    if not _is_sha256(expected):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256")
    root = root.resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"{field} escaped the handoff root")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{field} SHA-256 mismatch")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} is not valid JSON") from exc


def _observations(value: Any) -> list[dict[str, Any]]:
    rows = value.get("observations") if isinstance(value, Mapping) else value
    if not isinstance(rows, list):
        raise ValueError("OCR observations artifact is incomplete")
    normalized = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("OCR observation must be an object")
        identity = str(row.get("observation_id") or "")
        if not identity or identity in seen:
            raise ValueError("OCR observation identity is missing or duplicated")
        seen.add(identity)
        normalized.append(dict(row))
    return normalized


def _validate_structure(structure: Any, observation_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if not isinstance(structure, Mapping) or structure.get("schema") != "traduzai.vision.structural-analysis.v1":
        raise ValueError("structural analysis schema is unsupported")
    units = structure.get("logical_units")
    subblocks = structure.get("physical_subblocks")
    if not isinstance(units, list) or not isinstance(subblocks, list):
        raise ValueError("structural analysis is incomplete")
    by_unit: dict[str, list[tuple[int, str]]] = {}
    assigned_observations: set[str] = set()
    for block in subblocks:
        if not isinstance(block, Mapping):
            raise ValueError("physical subblock must be an object")
        unit_id = str(block.get("logical_unit_id") or "")
        block_id = str(block.get("physical_subblock_id") or "")
        if not unit_id or not block_id:
            raise ValueError("physical subblock identity is incomplete")
        for observation_id in block.get("observation_ids") or []:
            if observation_id not in observation_ids:
                raise ValueError("physical subblock references an unknown observation")
            if observation_id in assigned_observations:
                raise ValueError("physical observation was assigned more than once")
            assigned_observations.add(observation_id)
        by_unit.setdefault(unit_id, []).append((int(block.get("order") or 0), block_id))
    destinations = {
        unit_id: [block_id for _, block_id in sorted(rows)]
        for unit_id, rows in by_unit.items()
    }
    normalized_units = []
    unit_ids = set()
    for row in units:
        if not isinstance(row, Mapping):
            raise ValueError("logical unit must be an object")
        unit = dict(row)
        unit_id = str(unit.get("logical_unit_id") or "")
        if not unit_id or unit_id in unit_ids:
            raise ValueError("logical unit identity is missing or duplicated")
        unit_ids.add(unit_id)
        references = [str(value) for value in unit.get("observation_ids") or []]
        if not references or any(value not in observation_ids for value in references):
            raise ValueError("logical unit references unknown OCR evidence")
        normalized_units.append(unit)
    if any(unit_id not in unit_ids for unit_id in destinations):
        raise ValueError("physical subblock references an unknown logical unit")
    return normalized_units, destinations


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def consume_structural_handoff(
    handoff_path: str | Path,
    output_dir: str | Path,
    *,
    translator: Callable[..., Mapping[str, Any]] = translate_complete_unit,
    context: Mapping[str, Any] | None = None,
    glossary: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Translate supported logical units once and preserve unsupported units fail-closed."""

    handoff_path = Path(handoff_path).resolve(strict=True)
    handoff_bytes = handoff_path.read_bytes()
    handoff = json.loads(handoff_bytes)
    if (
        handoff.get("schema") != "traduzai.vision.structural-handoff.v1"
        or handoff.get("status") != "complete"
        or not _is_sha256(handoff.get("source_sha256"))
    ):
        raise ValueError("Vision structural handoff is incomplete")
    root = handoff_path.parent
    observation_payload = _read_verified_json(root, handoff.get("provider_evidence") or {}, "provider_evidence")
    _read_verified_json(root, handoff.get("containers") or {}, "containers")
    structure = _read_verified_json(root, handoff.get("structure") or {}, "structure")
    observations = _observations(observation_payload)
    units, destinations = _validate_structure(
        structure, {row["observation_id"] for row in observations},
    )

    consumed = []
    translation_provider_calls = 0
    review_required = 0
    for unit in units:
        unit_id = str(unit["logical_unit_id"])
        source = " ".join(str(unit.get("text") or "").split())
        uncertain = list(unit.get("uncertainty_reasons") or [])
        classification = str(unit.get("classification") or "unknown_text")
        destination_ids = destinations.get(unit_id, [unit_id])
        if classification == "unknown_text" or uncertain or not source:
            review_required += 1
            consumed.append({
                "owner_id": unit_id,
                "source": source,
                "target": None,
                "destination_ids": destination_ids,
                "status": "review_required",
                "reason_code": uncertain[0] if uncertain else "unsupported_or_empty_visual_unit",
                "source_preserved": True,
            })
            continue
        try:
            translated = dict(translator(
                source=source,
                owner_id=unit_id,
                page_id=str(handoff.get("source_member") or handoff.get("case") or "page"),
                context=dict(context or {}),
                glossary=dict(glossary or {}),
            ))
            if translated.get("provider_called") is not True or translated.get("cache_hit") is not False:
                raise ProviderUnavailable("fresh physical translation evidence is required")
            target = " ".join(str(translated.get("target") or "").split())
            if not target:
                raise ProviderUnavailable("empty translated unit")
            translation_provider_calls += 1
            consumed.append({
                "owner_id": unit_id,
                "source": source,
                "source_sha256": sha256_text(source),
                "target": target,
                "target_sha256": sha256_text(target),
                "destination_ids": destination_ids,
                "status": "translation_ready",
                "translation": translated,
                "source_preserved": True,
            })
        except ProviderUnavailable as exc:
            review_required += 1
            consumed.append({
                "owner_id": unit_id,
                "source": source,
                "target": None,
                "destination_ids": destination_ids,
                "status": "review_required",
                "reason_code": str(exc),
                "source_preserved": True,
            })

    body = {
        "schema": "traduzai.integration-structural-consumer-trace.v1",
        "handoff_path": str(handoff_path).replace("\\", "/"),
        "handoff_file_sha256": hashlib.sha256(handoff_bytes).hexdigest(),
        "source_member": handoff.get("source_member"),
        "source_sha256": handoff["source_sha256"],
        "vision_commit": handoff.get("vision_commit"),
        "answer_by_id": False,
        "provider_evidence": {
            "ocr_provider_called": handoff["provider_evidence"].get("provider_called") is True,
            "ocr_cache_hit": handoff["provider_evidence"].get("cache_hit") is True,
            "translation_provider_calls": translation_provider_calls,
        },
        "units": consumed,
        "stages": {
            "analysis": {"status": "consumed", "artifact_sha256": handoff["structure"]["sha256"]},
            "ocr": {"status": "consumed", "artifact_sha256": handoff["provider_evidence"]["sha256"]},
            "logical_units": {"status": "complete", "count": len(units)},
            "translate": {
                "status": "complete_with_review" if review_required else "complete",
                "provider_calls": translation_provider_calls,
                "review_required": review_required,
            },
            "restore": {
                "status": "review_required",
                "reason_code": "authenticated_action_masks_not_in_structural_handoff",
                "source_preserved": True,
            },
            "layout": {"status": "not_started"},
            "rasterize": {"status": "not_started"},
            "review": {"status": "awaiting_review"},
        },
        "publication_status": "not_publishable",
        "human_review": False,
    }
    result = body | {"trace_sha256": canonical_json_sha256(body)}
    output = Path(output_dir)
    _atomic_write_json(output / "consumer-trace.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = consume_structural_handoff(args.handoff, args.output_dir)
    print(json.dumps({
        "status": "complete_with_review",
        "trace_sha256": result["trace_sha256"],
        "output": str(Path(args.output_dir) / "consumer-trace.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
