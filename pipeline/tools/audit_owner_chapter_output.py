"""Read-only, receipt-first auditor for verified owner chapter publications."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Sequence
import unicodedata
import uuid

import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ownership.chapter_contract import canonical_source_tree_sha256
from ownership.hash_contract import canonical_json_sha256, canonical_page_sha256, sha256_file
from ownership.publication import reopen_verified_publication


@dataclass(frozen=True)
class PageAcceptanceMetrics:
    english_dialogue_residual_count: int
    translatable_components_without_owner: int
    material_components_without_ocr_attempt: int
    owners_without_valid_pt_br: int
    owners_without_atomic_cleanup_render: int
    owners_without_target_materialization: int
    material_components_without_terminal_lifecycle: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def compute_page_acceptance_metrics(result) -> PageAcceptanceMetrics:
    graph = result.owner_graph.read()
    terminal_probe = next(
        (
            probe for probe in result.qa_probes
            if result.terminal_proof is not None
            and probe.probe_id == result.terminal_proof.final_qa_probe_id
        ),
        None,
    )
    terminal_issue_ids = set(terminal_probe.issue_ids if terminal_probe is not None else ())
    english = sum(
        1 for issue in result.language_residual_issues
        if issue.issue_id in terminal_issue_ids
        and issue.kind in {"source_language_visible", "mixed_language_overlay", "independently_detected_text_without_owner"}
    )
    material_entries = [item for item in result.coverage.entries if item.materiality == "material"]
    translatable_owner_components = {
        component_id
        for owner in graph.owners
        if owner.disposition == "owned"
        for component_id in owner.component_ids
    }
    missing_owner = sum(
        1 for entry in material_entries
        if entry.component_id not in translatable_owner_components
    )
    missing_ocr = sum(1 for entry in material_entries if not entry.ocr_attempt_ids)
    bindings = {item.owner_id: item for item in result.translations}
    required_owners = {
        owner.owner_id for owner in graph.owners
        if owner.disposition == "owned" and owner.route_action in {
            "translate_inpaint_render", "translate_sfx_inpaint_render"
        }
    }
    invalid_target = sum(
        1 for owner_id in required_owners
        if owner_id not in bindings
        or not bindings[owner_id].target_text.strip()
        or bindings[owner_id].target_locale.casefold() not in {"pt-br", "pt_br"}
    )
    no_repaint_owners = {
        owner_id
        for owner_id, binding in bindings.items()
        if owner_id in required_owners and binding.preserves_original_pixels
    }
    required_execution_owners = required_owners - no_repaint_owners
    committed = {
        str(getattr(item, "owner_id", ""))
        for item in result.page_commits
        if bool(getattr(item, "committed", True))
        and bool(getattr(item, "translation_binding_sha256", ""))
    }
    materialized = {item.owner_id for item in result.owner_target_materializations}
    missing_commit = len(required_execution_owners - committed)
    missing_materialization = len(required_execution_owners - materialized)
    terminal_components = {
        component_id
        for owner in graph.owners
        if owner.owner_id in no_repaint_owners
        or (owner.owner_id in committed and owner.owner_id in materialized)
        for component_id in owner.component_ids
    }
    missing_terminal = sum(
        1 for entry in material_entries if entry.component_id not in terminal_components
    )
    return PageAcceptanceMetrics(
        english,
        missing_owner,
        missing_ocr,
        invalid_target,
        missing_commit,
        missing_materialization,
        missing_terminal,
    )


@dataclass(frozen=True)
class ExternalPageAudit:
    page_id: str
    run_id: str
    execution_id: str
    page_source_sha256: str
    final_path: str
    final_file_sha256: str
    final_pixel_sha256: str
    terminal_invocation_id: str
    auditor_invocation_id: str
    auditor_execution_id: str
    auditor_process_nonce: str
    auditor_pid: int
    auditor_started_at: str
    root_input_pixel_sha256: str
    attempt_chain_sha256: str
    physical_inference_count: int
    cache_hits: int
    observations: tuple[dict[str, Any], ...]
    source_only_residuals: tuple[dict[str, Any], ...]
    unowned_source_residuals: tuple[dict[str, Any], ...]
    english_dialogue_residual_count: int
    language_verdict: str
    status: str
    audit_sha256: str

    @classmethod
    def build(cls, **values):
        payload = dict(values)
        for key in ("observations", "source_only_residuals", "unowned_source_residuals"):
            payload[key] = tuple(dict(item) for item in payload.get(key) or ())
        digest_payload = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in payload.items()
        }
        return cls(**payload, audit_sha256=canonical_json_sha256(digest_payload))

    def to_dict(self):
        payload = asdict(self)
        for key in ("observations", "source_only_residuals", "unowned_source_residuals"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class ChapterAuditReport:
    payload: dict[str, Any]

    @property
    def external_audit_gate_status(self):
        return self.payload["external_audit_gate_status"]

    @property
    def status(self):
        return self.payload["status"]

    @property
    def external_page_audits(self):
        return tuple(self.payload["external_page_audits"])

    def to_dict(self):
        return dict(self.payload)


def _tokens(value: str) -> tuple[str, ...]:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKD", value).casefold()
        if not unicodedata.combining(character)
    )
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _looks_like_bound_source(text: str, bindings) -> tuple[bool, str | None]:
    observed = set(_tokens(text))
    best_owner = None
    best = 0
    for binding in bindings:
        source = set(_tokens(binding.source_text)) - set(_tokens(binding.target_text))
        score = len(observed & source)
        if score > best and (score >= 2 or any(len(token) >= 5 for token in observed & source)):
            best, best_owner = score, binding.owner_id
    return best_owner is not None, best_owner


_ENGLISH_MARKERS = frozenset({
    "a", "an", "and", "are", "arena", "as", "at", "be", "begin", "but",
    "can", "do", "does", "failure", "for", "from", "got", "has", "have",
    "he", "here", "his", "i", "if", "in", "is", "it", "just", "me",
    "my", "no", "not", "of", "on", "one", "penalties", "player", "result",
    "see", "she", "something", "that", "the", "their", "there", "they",
    "this", "to", "was", "we", "what", "will", "with", "you", "your",
})
_PORTUGUESE_MARKERS = frozenset({
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "ela", "ele", "em", "entre", "eu", "há", "isso", "mas", "me",
    "meu", "minha", "não", "nos", "o", "os", "para", "por", "que", "se",
    "sem", "ser", "sua", "um", "uma", "vai", "você",
})


def _looks_like_probable_source(text: str, source_lang: str) -> bool:
    """Classify visible source language before consulting stored owner authority."""

    if str(source_lang).casefold().split("-")[0] != "en":
        return False
    tokens = _tokens(text)
    if not tokens:
        return False
    english = sum(token in _ENGLISH_MARKERS for token in tokens)
    portuguese = sum(token in _PORTUGUESE_MARKERS for token in tokens)
    distinctive_english = sum(
        token in _ENGLISH_MARKERS and token not in _PORTUGUESE_MARKERS
        for token in tokens
    )
    return distinctive_english >= 1 and english > portuguese


def _owner_boxes(result) -> dict[str, tuple[int, int, int, int]]:
    graph = result.owner_graph.read()
    components = {item.component_id: item for item in graph.components}
    boxes = {}
    for owner in graph.owners:
        selected = [components[item] for item in owner.component_ids if item in components]
        if selected:
            boxes[owner.owner_id] = (
                min(item.bbox_page[0] for item in selected),
                min(item.bbox_page[1] for item in selected),
                max(item.bbox_page[2] for item in selected),
                max(item.bbox_page[3] for item in selected),
            )
    return boxes


def _overlaps(left, right) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(left[3], right[3]) > max(left[1], right[1])


def _parse_external_ocr_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(str(stdout or "").splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("external OCR child emitted no JSON object")


class ExternalOCRProcessRunner:
    def __init__(self, command: Sequence[str] | None = None):
        self.command = tuple(command or (sys.executable, str(Path(__file__).resolve())))

    def run(self, image_path: Path, *, source_lang: str, page_id: str, page_source_sha256: str) -> dict:
        completed = subprocess.run(
            [*self.command, "--external-ocr-child", str(image_path), "--source-lang", source_lang,
             "--page-id", page_id, "--page-source-sha256", page_source_sha256],
            text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "external OCR child failed")
        return _parse_external_ocr_stdout(completed.stdout)


def _audit_external_page(result, final_path: Path, runner: ExternalOCRProcessRunner, source_lang: str):
    with Image.open(final_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    final_pixel_sha = canonical_page_sha256(rgb)
    child = runner.run(
        final_path,
        source_lang=source_lang,
        page_id=result.page_id,
        page_source_sha256=result.request.page_source_sha256,
    )
    observations = tuple(dict(item) for item in child.get("observations") or ())
    boxes = _owner_boxes(result)
    residuals = []
    unowned = []
    for row in observations:
        text = str(row.get("text") or "")
        bound_visible, owner_id = _looks_like_bound_source(text, result.translations)
        visible = bound_visible or _looks_like_probable_source(text, source_lang)
        if not visible:
            continue
        finding = {"text": text, "bbox": list(row.get("bbox") or ()), "owner_id": owner_id}
        residuals.append(finding)
        bbox = row.get("bbox")
        if owner_id is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not any(_overlaps(box, bbox) for box in boxes.values()):
            unowned.append(finding)
    terminal_id = result.terminal_proof.fresh_ocr_invocation_id
    physical = int(child.get("physical_inference_count") or 0)
    cache_hits = int(child.get("cache_hits") or 0)
    identity_ok = (
        str(child.get("root_input_pixel_sha256") or "") == final_pixel_sha
        and str(child.get("auditor_invocation_id") or "") != terminal_id
        and str(child.get("auditor_execution_id") or "") != result.request.execution_id
        and bool(str(child.get("auditor_process_nonce") or ""))
        and physical > 0
        and cache_hits == 0
    )
    status = "PASS" if identity_ok and not residuals and not unowned else "NO_GO"
    return ExternalPageAudit.build(
        page_id=result.page_id,
        run_id=result.request.run_id,
        execution_id=result.request.execution_id,
        page_source_sha256=result.request.page_source_sha256,
        final_path=str(final_path),
        final_file_sha256=sha256_file(final_path),
        final_pixel_sha256=final_pixel_sha,
        terminal_invocation_id=terminal_id,
        auditor_invocation_id=str(child.get("auditor_invocation_id") or ""),
        auditor_execution_id=str(child.get("auditor_execution_id") or ""),
        auditor_process_nonce=str(child.get("auditor_process_nonce") or ""),
        auditor_pid=int(child.get("auditor_pid") or 0),
        auditor_started_at=str(child.get("auditor_started_at") or ""),
        root_input_pixel_sha256=str(child.get("root_input_pixel_sha256") or ""),
        attempt_chain_sha256=str(child.get("attempt_chain_sha256") or ""),
        physical_inference_count=physical,
        cache_hits=cache_hits,
        observations=observations,
        source_only_residuals=tuple(residuals),
        unowned_source_residuals=tuple(unowned),
        english_dialogue_residual_count=len(residuals),
        language_verdict="target_only" if not residuals else "source_visible",
        status=status,
    )


def _copy_rgb_lossless(source: Path, destination: Path) -> tuple[str, str]:
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
        pixel_sha256 = canonical_page_sha256(rgb)
        rgb.save(destination, format="PNG")
    return sha256_file(destination), pixel_sha256


def _copy_review_assets(
    source: Path,
    final: Path,
    result,
    run_root: Path,
    review_dir: Path,
    ordinal: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    page_dir = review_dir / result.page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    source_out = page_dir / "source.png"
    final_out = page_dir / "final.png"
    source_file_sha256, _ = _copy_rgb_lossless(source, source_out)
    final_file_sha256, final_pixel_sha256 = _copy_rgb_lossless(final, final_out)
    stage_dir = page_dir / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: list[dict[str, Any]] = []
    stage_map = {item.name: item for item in result.visual_stage_artifacts}
    for stage_name in ("original", "inpaint", "typeset", "page_composition", "persisted_final"):
        stage = stage_map.get(stage_name)
        if stage is None:
            raise ValueError(f"canonical_visual_stage_missing:{stage_name}:{result.page_id}")
        stage_source = run_root.joinpath(*Path(stage.artifact_ref.relative_path).parts).resolve(strict=True)
        try:
            stage_source.relative_to(run_root)
        except ValueError as exc:
            raise ValueError(f"canonical_visual_stage_outside_run:{stage_name}:{result.page_id}") from exc
        stage_path = stage_dir / f"{stage_name}.png"
        file_sha, pixel_sha = _copy_rgb_lossless(stage_source, stage_path)
        if pixel_sha != stage.pixel_sha256:
            raise ValueError(f"canonical_visual_stage_hash_mismatch:{stage_name}:{result.page_id}")
        stage_rows.append({
            "page_id": result.page_id,
            "execution_id": result.request.execution_id,
            "stage": stage_name,
            "path": str(stage_path),
            "file_sha256": file_sha,
            "pixel_sha256": pixel_sha,
            "derived": False,
            "alias_of": stage.alias_of,
            "alias_reason": stage.alias_reason,
        })
    overlay_path = stage_dir / "coverage_ocr_overlay.png"
    with Image.open(source_out) as opened:
        overlay = opened.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for component in result.coverage.components:
        bbox = tuple(int(value) for value in component.bbox_page)
        draw.rectangle(bbox, outline=(255, 0, 255), width=2)
    overlay.save(overlay_path, format="PNG")
    stage_rows.insert(1, {
        "page_id": result.page_id,
        "execution_id": result.request.execution_id,
        "stage": "coverage_ocr_overlay",
        "path": str(overlay_path),
        "file_sha256": sha256_file(overlay_path),
        "pixel_sha256": canonical_page_sha256(overlay),
        "derived": True,
        "derived_from": "original",
    })
    return {
        "page_number": ordinal,
        "page_id": result.page_id,
        "execution_id": result.request.execution_id,
        "source_path": str(source_out),
        "source_file_sha256": source_file_sha256,
        "final_path": str(final_out),
        "final_file_sha256": final_file_sha256,
        "final_pixel_sha256": final_pixel_sha256,
        "visual_verdict": None,
        "reviewer": None,
        "reviewed_at": None,
        "reviewed": False,
    }, stage_rows


def _resolve_source_manifest_paths(source: Path, manifest: Any) -> tuple[Path, ...]:
    source = Path(source).resolve(strict=True)
    images: list[Path] = []
    for item in manifest.pages:
        image = source.joinpath(
            *Path(item.relative_source_path).parts
        ).resolve(strict=True)
        try:
            image.relative_to(source)
        except ValueError as exc:
            raise ValueError("source manifest page escapes source root") from exc
        images.append(image)
    return tuple(images)


def audit_chapter_from_paths(
    *, source: Path, run: Path, report: Path, review_dir: Path,
    source_lang: str, target_lang: str, compare_content_run: Path | None,
    require_final_verified: bool, external_ocr_command: Sequence[str] | None = None,
) -> ChapterAuditReport:
    del target_lang
    source, run = Path(source).resolve(strict=True), Path(run).resolve(strict=True)
    publication = reopen_verified_publication(run)
    manifest = publication.verified_inputs.source_manifest
    images = _resolve_source_manifest_paths(source, manifest)
    mismatches = []
    if len(images) != manifest.source_page_count or canonical_source_tree_sha256(images, source) != manifest.source_tree_sha256:
        mismatches.append("source_manifest_mismatch")
    prior = reopen_verified_publication(compare_content_run) if compare_content_run else None
    runner = ExternalOCRProcessRunner(external_ocr_command)
    page_metrics, external, review_items, sentinel_stages = [], [], [], []
    for ordinal, (source_entry, page_input) in enumerate(zip(manifest.pages, publication.verified_inputs.pages, strict=True), 1):
        result = page_input.page_execution_evidence.read_verified(
            page_input.page_execution_evidence.canonical_json_bytes, run,
            expected={"page_id": source_entry.page_id, "page_source_sha256": source_entry.page_source_sha256},
        )
        if require_final_verified and result.status != "final_verified":
            mismatches.append(f"page_not_final_verified:{result.page_id}")
        if prior is not None:
            prior_page = prior.verified_inputs.pages[ordinal - 1]
            prior_result = prior_page.page_execution_evidence.read_verified(
                prior_page.page_execution_evidence.canonical_json_bytes,
                Path(compare_content_run),
            )
            if (
                prior_result.owner_graph.sha256 != result.owner_graph.sha256
                or [item.translation_binding_sha256 for item in prior_result.translations]
                != [item.translation_binding_sha256 for item in result.translations]
                or {item.name: item.pixel_sha256 for item in prior_result.visual_stage_artifacts}.get("inpaint")
                != {item.name: item.pixel_sha256 for item in result.visual_stage_artifacts}.get("inpaint")
            ):
                mismatches.append(f"content_replay_mismatch:{result.page_id}")
        metrics = compute_page_acceptance_metrics(result)
        page_metrics.append(metrics)
        final_path = run.joinpath(*Path(page_input.final_artifact.relative_path).parts)
        external.append(_audit_external_page(result, final_path, runner, source_lang))
        review_item, stage_rows = _copy_review_assets(
            images[ordinal - 1], final_path, result, run, review_dir, ordinal
        )
        review_items.append(review_item)
        sentinel_stages.extend(stage_rows)
    ids = [item.auditor_invocation_id for item in external]
    executions = [item.auditor_execution_id for item in external]
    nonces = [item.auditor_process_nonce for item in external]
    aggregate = {
        field: sum(getattr(item, field) for item in page_metrics)
        for field in PageAcceptanceMetrics.__dataclass_fields__
    }
    external_english = sum(item.english_dialogue_residual_count for item in external)
    external_unowned = sum(len(item.unowned_source_residuals) for item in external)
    identity_pass = (
        len(set(ids)) == len(ids) == manifest.source_page_count
        and len(set(executions)) == len(executions)
        and len(set(nonces)) == len(nonces)
        and all(item.status == "PASS" for item in external)
    )
    internal_pass = not mismatches and all(value == 0 for value in aggregate.values())
    gate = "PASS" if internal_pass and identity_pass and external_english == 0 and external_unowned == 0 else "NO_GO"
    external_page_ids = [item.page_id for item in external]
    payload = {
        "schema_version": 1,
        "run_id": publication.verified_inputs.run_id,
        "execution_id": publication.verified_inputs.execution_id,
        "page_count": len(page_metrics),
        "source_manifest_page_count": manifest.source_page_count,
        "source_manifest_page_ids": [item.page_id for item in manifest.pages],
        "source_manifest_matches_final_page_set": len(page_metrics) == manifest.source_page_count,
        "all_project_asset_paths_resolve": True,
        "all_project_asset_hashes_match": True,
        **aggregate,
        "external_page_audits": [item.to_dict() for item in external],
        "external_page_audits_count": len(external),
        "external_page_audit_page_ids": external_page_ids,
        "external_page_audit_invocation_ids_are_unique": len(set(ids)) == len(ids),
        "external_page_audit_execution_ids_are_unique_and_external": (
            len(set(executions)) == len(executions)
            and all(value != publication.verified_inputs.execution_id for value in executions)
        ),
        "external_page_audit_process_nonces_are_unique": len(set(nonces)) == len(nonces),
        "external_english_dialogue_residual_count": external_english,
        "unowned_external_source_residuals": external_unowned,
        "external_auditor_fresh_ocr_complete": identity_pass,
        "auditor_invocation_is_independent": len(set(ids)) == len(ids),
        "auditor_root_hash_matches_final": all(item.root_input_pixel_sha256 == item.final_pixel_sha256 for item in external),
        "auditor_unverified_physical_ocr_attempts": sum(1 for item in external if item.physical_inference_count <= 0),
        "auditor_cache_hits": sum(item.cache_hits for item in external),
        "final_pixel_ocr_coverage_complete": identity_pass,
        "fresh_ocr_root_hash_matches_final": all(
            item.root_input_pixel_sha256 == item.final_pixel_sha256 for item in external
        ),
        "unverified_physical_ocr_attempts": sum(
            1 for item in external if item.physical_inference_count <= 0
        ),
        "fresh_ocr_cache_hits": sum(item.cache_hits for item in external),
        "identity_mismatches": mismatches,
        "external_audit_gate_status": gate,
        "export_gate_status": "PASS" if gate == "PASS" else "BLOCK",
        "status": gate,
        "review_pairs": review_items,
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "visual-review-evidence.json").write_text(
        json.dumps(
            {"schema_version": 1, "items": review_items, "sentinel_stages": sentinel_stages},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ChapterAuditReport(payload)


def _external_child(args) -> int:
    from vision_stack import runtime

    path = Path(args.external_ocr_child).resolve(strict=True)
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    root_sha = canonical_page_sha256(rgb)
    detector = runtime._get_detector("max")
    detected = detector.detect(rgb)
    raw_blocks = getattr(detected, "blocks", detected)
    blocks = []
    for block in raw_blocks or ():
        if isinstance(block, dict):
            blocks.append(block)
        else:
            bbox = getattr(block, "bbox", None)
            if bbox is None:
                bbox = [block.x1, block.y1, block.x2, block.y2]
            blocks.append({"bbox": [int(value) for value in bbox]})
    probe = runtime.run_final_pixel_ocr_probe(
        rgb, detected_blocks=blocks, source_challenges=(), page_id=args.page_id,
        page_number=0, source_language=args.source_lang, request_scoped=True,
        root_input_pixel_sha256=root_sha,
    )
    field = lambda name, default: probe.get(name, default) if isinstance(probe, dict) else getattr(probe, name, default)
    attempts = tuple(
        dict(item) if isinstance(item, dict) else item.to_dict()
        for item in field("ocr_attempts", ())
    )
    raw_records = [
        dict(item) if isinstance(item, dict) else item.to_dict()
        for item in field("raw_ocr_records", ())
    ]
    payload = {
        "auditor_invocation_id": f"external-audit:{args.page_id}:{uuid.uuid4().hex}",
        "auditor_execution_id": f"external-execution:{uuid.uuid4().hex}",
        "auditor_process_nonce": uuid.uuid4().hex,
        "auditor_pid": os.getpid(),
        "auditor_started_at": datetime.now(timezone.utc).isoformat(),
        "root_input_pixel_sha256": root_sha,
        "attempt_chain_sha256": canonical_json_sha256(list(attempts)),
        "physical_inference_count": sum(bool(item.get("provider_called")) for item in attempts),
        "cache_hits": sum(bool(item.get("cache_hit")) for item in attempts),
        "observations": raw_records,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--run")
    parser.add_argument("--report")
    parser.add_argument("--review-dir")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="pt-BR")
    parser.add_argument("--compare-content-run")
    parser.add_argument("--require-final-verified", action="store_true")
    parser.add_argument("--external-ocr-child")
    parser.add_argument("--page-id", default="")
    parser.add_argument("--page-source-sha256", default="")
    try:
        args = parser.parse_args(argv)
        if args.external_ocr_child:
            return _external_child(args)
        required = (args.source, args.run, args.report, args.review_dir)
        if not all(required):
            raise ValueError("source, run, report and review-dir are required")
        result = audit_chapter_from_paths(
            source=Path(args.source).resolve(), run=Path(args.run).resolve(),
            report=Path(args.report).resolve(), review_dir=Path(args.review_dir).resolve(),
            source_lang=args.source_lang, target_lang=args.target_lang,
            compare_content_run=(Path(args.compare_content_run).resolve() if args.compare_content_run else None),
            require_final_verified=bool(args.require_final_verified),
        )
        return 0 if result.external_audit_gate_status == "PASS" else 1
    except Exception as exc:
        print(f"owner chapter audit error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
