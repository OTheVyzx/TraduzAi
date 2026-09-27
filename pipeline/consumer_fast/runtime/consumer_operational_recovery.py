"""Source-driven Consumer Fast split/merge recovery, with fail-closed stages.

The caller selects source members. This module discovers units and does not
contain page IDs, owner IDs, source text or Portuguese target tables.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import copy
from pathlib import Path

import numpy as np
from PIL import Image

from consumer_case_policy import apply as apply_source_case
from consumer_fast_core import RasterLayer, flatten_layers
from consumer_layout_search import preferred_size, search_source_centered_layout, source_line_ink_height
from consumer_operational_discovery import discover_recovery_proposals
from consumer_visual_comfort import (build_source_white_policy_adaptive,
                                     build_source_white_policy_across_previous,
                                     prepare_source_white_comfort)
from consumer_white_glyph_cleanup import clear_residual_white_glyphs
from integration_v1.providers import ProviderUnavailable, recover_source_with_ocr, translate_complete_unit


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha(value: dict) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def _runtime_cache_path(root: Path) -> Path:
    return root / "integration_v1_runtime_cache.json"


def _read_runtime_cache(root: Path) -> dict:
    path = _runtime_cache_path(root)
    if not path.is_file():
        return {"schema": "traduzai.integration-runtime-cache.v1",
                "ocr_entries": {}, "translation_entries": {}}
    payload = read(path)
    if payload.get("schema") != "traduzai.integration-runtime-cache.v1":
        raise ValueError("integration runtime cache schema mismatch")
    return payload


def _write_runtime_cache(root: Path, payload: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = _runtime_cache_path(root)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
               + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_key(kind: str, identity: dict) -> str:
    return f"{kind}:{_canonical_sha(identity)}"


def pixels(path: Path, mode: str = "RGB") -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert(mode), dtype=np.uint8).copy()


def union(boxes: list[list[int]]) -> list[int]:
    return [min(row[0] for row in boxes), min(row[1] for row in boxes),
            max(row[2] for row in boxes), max(row[3] for row in boxes)]


def crop_for_observations(boxes: list[list[int]], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = union(boxes)
    pad = max(18, round(.12 * max(x2-x1, y2-y1)))
    return [max(0, x1-pad), max(0, y1-pad), min(width, x2+pad), min(height, y2+pad)]


def _letters(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _cached_ocr(crop: np.ndarray, bbox: list[int], member: str,
                source_sha256: str, cache_root: Path | None) -> tuple[dict, dict]:
    if cache_root is None:
        raise ValueError("local OCR cache missing; direct OCR stage not configured")
    evidence = read(cache_root / "ocr_evidence.json")
    matches = [row for row in evidence["jobs"]
               if row["source_member"] == member and row["source_sha256"] == source_sha256 and
               row["crop_bbox_page"] == bbox and row["crop_source_pixel_sha256"] == sha(crop.tobytes())]
    if len(matches) != 1:
        raise ValueError("authenticated local OCR observation cache missing or ambiguous")
    job = matches[0]
    if sha((cache_root / job["crop_path"]).read_bytes()) != job["crop_png_sha256"]:
        raise ValueError("OCR crop bytes changed")
    reviews = read(cache_root / "source_review_model_r001.json")
    rows = [row for row in reviews["units"] if row["crop_png_sha256"] == job["crop_png_sha256"]]
    if len(rows) != 1 or reviews.get("human_review") is not False:
        raise ValueError("source OCR correction lacks model provenance")
    review = rows[0]
    by_id = {row["observation_id"]: row for row in job["observations"]}
    ids = review["selected_observation_ids"]
    if len(ids) != len(set(ids)) or any(identity not in by_id for identity in ids):
        raise ValueError("selected OCR identities changed")
    selected = [by_id[identity] for identity in ids]
    if _letters(review["selected_source"]) != _letters(" ".join(row["text"] for row in selected)):
        raise ValueError("model source correction changes visible letters")
    return job, dict(review=review, selected=selected,
                     source_review_sha256=sha((cache_root / "source_review_model_r001.json").read_bytes()))


def _cached_or_fresh_ocr(*, page_rgb: np.ndarray, crop: np.ndarray, bbox: list[int],
                         member: str, page_id: str, source_sha256: str,
                         cache_root: Path | None, run_id: str,
                         origin_execution_id: str, ocr_invocation=None,
                         context: dict | None = None,
                         allow_cache_read: bool = True) -> tuple[dict, dict]:
    """Reuse authenticated observations or execute the hash-bound Vision interface."""
    runtime_identity = {
        "member": member, "page_id": page_id,
        "source_sha256": source_sha256, "bbox": list(bbox),
        "crop_pixel_sha256": sha(crop.tobytes()),
        "provider_policy": "vision-variant-consensus-v1",
    }
    runtime_key = _runtime_key("ocr", runtime_identity)
    if cache_root is not None and allow_cache_read:
        runtime_cache = _read_runtime_cache(cache_root)
        cached = runtime_cache["ocr_entries"].get(runtime_key)
        if cached is not None:
            if cached.get("identity") != runtime_identity:
                raise ValueError("integration OCR cache identity mismatch")
            job = copy.deepcopy(cached["job"])
            job.update(provider_called=False, cache_hit=True,
                       origin_provider_called=True)
            return job, copy.deepcopy(cached["adjudicated"])
    if allow_cache_read and cache_root is not None and (cache_root / "ocr_evidence.json").is_file() and (
            cache_root / "source_review_model_r001.json").is_file():
        try:
            return _cached_ocr(crop, bbox, member, source_sha256, cache_root)
        except ValueError as exc:
            if str(exc) != "authenticated local OCR observation cache missing or ambiguous":
                raise
            evidence = read(cache_root / "ocr_evidence.json")
            candidates = [row for row in evidence.get("jobs") or []
                          if row.get("source_member") == member]
            if len(candidates) > 1:
                raise
    recovered = recover_source_with_ocr(
        page_rgb=page_rgb, bbox_page=tuple(bbox), page_id=page_id,
        page_source_sha256=source_sha256, run_id=run_id,
        origin_execution_id=origin_execution_id, invocation=ocr_invocation,
        context=context,
    )
    job = {
        "schema": "traduzai.cold-ocr-job.v1",
        "source_member": member,
        "source_sha256": source_sha256,
        "crop_bbox_page": bbox,
        "crop_source_pixel_sha256": sha(crop.tobytes()),
        "provider_called": True,
        "cache_hit": False,
        "attempt_chain_sha256": recovered["attempt_chain_sha256"],
        "observations": recovered["selected"],
    }
    review = {
        "schema": "traduzai.cold-ocr-selection.v1",
        "selected_source": recovered["source"],
        "selected_observation_ids": recovered["selected_observation_ids"],
        "human_review": False,
        "provenance": "fresh_physical_ocr",
    }
    adjudicated = {
        "review": review,
        "selected": recovered["selected"],
        "source_review_sha256": _canonical_sha(review),
    }
    if cache_root is not None:
        runtime_cache = _read_runtime_cache(cache_root)
        runtime_cache["ocr_entries"][runtime_key] = {
            "identity": runtime_identity,
            "job": job,
            "adjudicated": adjudicated,
        }
        _write_runtime_cache(cache_root, runtime_cache)
    return job, adjudicated


def _accepted_translation(control_root: Path, control: dict, member: str,
                          source_sha256: str, source: str, owners: list[str]) -> dict:
    matches = []
    for relative in control.get("complete_balloon_reconciliations") or []:
        path = control_root / relative
        group = read(path)
        if group["source_member"] != member:
            continue
        if (group["source_sha256"] != source_sha256 or group["source"] != source or
                {group["current_owner_id"], group["superseded_owner_id"]} != set(owners)):
            raise ValueError("accepted translation lineage differs from recovered source")
        recipe_path = control_root / group["recipe_path"]
        if sha(recipe_path.read_bytes()) != group["recipe_sha256"]:
            raise ValueError("accepted translation recipe hash changed")
        matches.append(dict(target=group["target"], group_path=relative,
                            group_sha256=sha(path.read_bytes()),
                            recipe_sha256=group["recipe_sha256"],
                            provenance="accepted_existing_translation_reuse"))
    if len(matches) != 1:
        raise ValueError("accepted translation absent or ambiguous")
    return matches[0]


def _accepted_or_fresh_translation(*, control_root: Path, control: dict,
                                   member: str, source_sha256: str, source: str,
                                   owners: list[str], page_id: str,
                                   context: dict, glossary: dict,
                                   translation_invocation=None,
                                   allow_accepted: bool = True,
                                   runtime_cache_root: Path | None = None,
                                   allow_cache_read: bool = True) -> dict:
    identity = {
        "member": member, "page_id": page_id,
        "source_sha256": source_sha256, "source": source,
        "owners": list(owners), "source_language": "en",
        "target_locale": "pt-BR", "provider_policy": "google-cold-complete-unit-v1",
    }
    runtime_key = _runtime_key("translation", identity)
    if runtime_cache_root is not None and allow_cache_read:
        runtime_cache = _read_runtime_cache(runtime_cache_root)
        cached = runtime_cache["translation_entries"].get(runtime_key)
        if cached is not None:
            if cached.get("identity") != identity:
                raise ValueError("integration translation cache identity mismatch")
            binding = copy.deepcopy(cached["binding"])
            binding.update(provider_called=False, cache_hit=True,
                           origin_provider_called=True,
                           provenance="warm_complete_unit_translation_cache")
            return binding
    if allow_accepted:
        try:
            return _accepted_translation(control_root, control, member, source_sha256, source, owners)
        except ValueError as exc:
            if str(exc) != "accepted translation absent or ambiguous":
                raise
    fresh = translate_complete_unit(
        source=source, owner_id=owners[0], page_id=page_id, context=context,
        glossary=glossary, invoke=translation_invocation,
    )
    binding = {
        **fresh,
        "group_path": None,
        "group_sha256": fresh["provider_metadata_sha256"],
    }
    if runtime_cache_root is not None:
        runtime_cache = _read_runtime_cache(runtime_cache_root)
        runtime_cache["translation_entries"][runtime_key] = {
            "identity": identity, "binding": binding,
        }
        _write_runtime_cache(runtime_cache_root, runtime_cache)
    return binding


def _reviewed_translation(binding: dict, review_path: Path | None,
                          member: str, source_sha256: str, source: str) -> dict:
    if review_path is None:
        return binding
    payload = read(review_path)
    if payload.get("schema") != "consumer_model_editorial_translation_v1" or (
            payload.get("reviewer_type") != "model" or
            payload.get("human_review") is not False):
        raise ValueError("editorial candidate lacks model provenance")
    matches = [row for row in payload.get("units") or [] if row.get("source_member") == member]
    if not matches:
        return binding
    if len(matches) != 1:
        raise ValueError("ambiguous editorial candidate")
    row = matches[0]
    if (row.get("source_sha256") != source_sha256 or row.get("source") != source or
            binding["target"] not in {row.get("previous_target"), row.get("target")} or
            not str(row.get("target") or "").strip()):
        raise ValueError("editorial candidate lineage differs")
    return dict(binding, target=row["target"],
                provenance="model_editorial_translation_review",
                editorial_review_sha256=sha(review_path.read_bytes()),
                editorial_reason=row.get("reason"))


def _accepted_split(control_root: Path, control: dict, member: str,
                    source_sha256: str, parent: str, sources: list[str]) -> dict:
    matches = []
    for relative in control.get("local_subblock_groups") or []:
        path = control_root / relative
        group = read(path)
        if group.get("source_member") != member:
            continue
        if (group.get("source_sha256") != source_sha256 or
                group.get("original_owner_id") != parent):
            raise ValueError("accepted subblock lineage differs from recovered source")
        members = group.get("members") or []
        if len(members) != len(sources) or any(
                item.get("source") != source for item, source in zip(members, sources)):
            raise ValueError("accepted subblock sources differ from local OCR")
        for item in members:
            if sha((control_root / item["recipe_path"]).read_bytes()) != item["recipe_sha256"]:
                raise ValueError("accepted subblock recipe hash changed")
        matches.append(dict(group=group, group_path=relative,
                            group_sha256=sha(path.read_bytes())))
    if len(matches) != 1:
        raise ValueError("accepted subblock translation absent or ambiguous")
    return matches[0]


def _prepare_split(*, source_root: Path, control_root: Path, control: dict,
                   page: dict, meta: dict, control_meta: dict, proposal: dict, original: np.ndarray,
                   member: str, cache_root: Path | None, font_path: Path,
                   run_id: str, origin_execution_id: str, context: dict,
                   glossary: dict, ocr_invocation=None,
                   translation_invocation=None,
                   allow_prepared_responses: bool = True) -> dict:
    parent = proposal["owner_ids"][0]
    units = proposal["plan"]["units"]
    observations = []
    for unit in units:
        bbox = unit["ocr_crop_bbox_page"]
        x1, y1, x2, y2 = bbox
        crop = original[y1:y2, x1:x2]
        job, adjudicated = _cached_or_fresh_ocr(
            page_rgb=original, crop=crop, bbox=bbox, member=member,
            page_id=page["page_id"], source_sha256=page["source_sha256"],
            cache_root=cache_root, run_id=run_id,
            origin_execution_id=origin_execution_id,
            ocr_invocation=ocr_invocation, context=context,
            allow_cache_read=allow_prepared_responses)
        observations.append(dict(unit=unit, bbox=bbox, job=job, adjudicated=adjudicated,
                                 source=adjudicated["review"]["selected_source"]))
    sources = [row["source"] for row in observations]
    binding = None
    if allow_prepared_responses:
        try:
            binding = _accepted_split(control_root, control, member, page["source_sha256"],
                                      parent, sources)
        except ValueError as exc:
            if str(exc) != "accepted subblock translation absent or ambiguous":
                raise
    if binding is None:
        members = []
        for order, source in enumerate(sources):
            fresh = translate_complete_unit(
                source=source, owner_id=f"{parent}:subblock:{order}",
                page_id=page["page_id"], context=context, glossary=glossary,
                invoke=translation_invocation)
            members.append({"owner_id": f"{parent}:subblock:{order}",
                            "source": source, "target": fresh["target"],
                            "translation_provenance": fresh["provenance"],
                            "provider_metadata_sha256": fresh["provider_metadata_sha256"]})
        group = {"schema": "traduzai.cold-subblock-translation.v1",
                 "source_member": member, "source_sha256": page["source_sha256"],
                 "original_owner_id": parent, "members": members}
        binding = {"group": group, "group_path": None,
                   "group_sha256": sha(json.dumps(group, sort_keys=True,
                                                   ensure_ascii=False).encode())}
    if binding["group"].get("source_recovery_kind") == "operational_source_observations_v1":
        records = {row["owner_id"]: row for row in control_meta["texts"]}
        if all((row := records.get(item["owner_id"])) is not None and
               row.get("source_payload") == observed["source"] and
               row.get("translated_payload") == item["target"] and
               (row.get("model_recovery") or {}).get("source_review_sha256") ==
                   observed["adjudicated"]["source_review_sha256"]
               for observed, item in zip(observations, binding["group"]["members"])):
            return dict(status="already_published", proposal=proposal,
                        source_member=member, group_sha256=binding["group_sha256"])
    clean = pixels(source_root / page["clean_base"])
    prepared_units = []
    for order, (observed, accepted) in enumerate(zip(observations, binding["group"]["members"])):
        unit = observed["unit"]
        boxes = [row["bbox_page"] for row in observed["adjudicated"]["selected"]]
        anchor = union(boxes)
        hint = unit["safe_bbox_page"]
        rough_height = float(np.median([box[3]-box[1] for box in boxes]))
        def policy_for(minimum: int, ideal: int) -> dict:
            return build_source_white_policy_adaptive(
                source_root / page["original"], hint, anchor, boxes, minimum, ideal,
                source_member_bytes=(source_root / "source_members" / member).read_bytes())
        rough = policy_for(max(3, round(.14*rough_height)),
                           max(6, round(.30*rough_height)))
        if rough["status"] != "body_validated":
            raise ValueError("subblock source white body unproven: "+rough["status"])
        visual, visual_evidence = source_line_ink_height(
            original, prepare_source_white_comfort(rough["policy"], page["source_sha256"]), boxes)
        policy_result = policy_for(max(3, round(.14*visual)), max(6, round(.30*visual)))
        if policy_result["status"] != "body_validated":
            raise ValueError("subblock measured source white body unproven")
        policy = policy_result["policy"]
        target, case = apply_source_case(observed["source"], accepted["target"])
        font_size, preference = preferred_size(font_path, visual)
        recipe = dict(schema="consumer_focal_text_v1", page_id=page["page_id"],
            owner_id=f"{parent}:subblock:{order}", source_member=member,
            source_sha256=page["source_sha256"], source=observed["source"], target=target,
            clean_pixels_sha256=sha(clean.tobytes()),
            adjudication=dict(observer="model_visual_review", human_review=False,
                source_sha256=page["source_sha256"],
                source_review_sha256=observed["adjudicated"]["source_review_sha256"],
                translation_binding_sha256=binding["group_sha256"]),
            font_path=str(font_path), font_sha256=sha(font_path.read_bytes()),
            font_size=font_size, minimum_font_size=max(18, round(.50*visual)),
            line_advance=font_size, line_spacing_profile="compact_safe_leading_v1",
            safe_bbox=prepare_source_white_comfort(policy, page["source_sha256"])["interior_bbox"],
            anchor_bbox=anchor, visual_comfort_policy=policy,
            source_case_profile=case["source_case_profile"])
        result = search_source_centered_layout(clean, recipe,
            source_visual_height_px=visual, preferred_font_size=font_size,
            minimum_font_size=recipe["minimum_font_size"],
            maximum_font_size=min(56, font_size+6), maximum_seconds=90)
        if result["status"] != "rendered_candidate":
            return dict(status="review_required", reason=result["reason"], proposal=proposal,
                        failed_subblock_order=order, search_attempts=result["attempts"])
        prepared_units.append(dict(order=order, source=observed["source"], target=target,
            crop_bbox=observed["bbox"],
            crop_pixel_sha256=sha(original[observed["bbox"][1]:observed["bbox"][3],
                                           observed["bbox"][0]:observed["bbox"][2]].tobytes()),
            selected_observation_ids=observed["adjudicated"]["review"]["selected_observation_ids"],
            source_review_sha256=observed["adjudicated"]["source_review_sha256"],
            ocr_job_sha256=sha(json.dumps(observed["job"],sort_keys=True,
                                         ensure_ascii=False,default=str).encode()),
            layer=result["layer"], recipe=result["recipe"], fit=result["evidence"],
            source_visual_evidence=visual_evidence, font_preference=preference,
            candidate_count=result["candidate_count"],
            metric_cache=result["metric_cache"]))
    if len([row for row in meta["raster_cache"] if row["owner_id"] == parent]) != 1:
        raise ValueError("split parent raster absent or ambiguous")
    layers = [RasterLayer(page["page_id"], row["owner_id"], tuple(row["bbox"]),
                          pixels(source_root / row["path"], "RGBA"))
              for row in meta["raster_cache"] if row["owner_id"] != parent]
    layers += [row["layer"] for row in prepared_units]
    return dict(status="rendered_candidate", proposal=proposal, page=page,
                binding=binding, subunits=prepared_units, clean=clean,
                final=flatten_layers(clean, layers))


def prepare_member(*, source_root: Path, control_root: Path, member: str,
                   graph_observations: dict[str, dict], cache_root: Path | None,
                   font_path: Path, translation_review: Path | None = None,
                   run_id: str | None = None,
                   origin_execution_id: str | None = None,
                   ocr_invocation=None, translation_invocation=None,
                   allow_prepared_responses: bool = True) -> list[dict]:
    """Discover and execute merge candidates through source/OCR/target/layout."""
    source_project, control = read(source_root / "project.json"), read(control_root / "project.json")
    pages = [row for row in source_project["pages"] if row.get("source_member") == member]
    if len(pages) != 1:
        raise ValueError("source member absent or duplicated")
    page = pages[0]
    control_pages = [row for row in control["pages"] if row.get("source_member") == member]
    if len(control_pages) != 1 or control_pages[0]["source_sha256"] != page["source_sha256"]:
        raise ValueError("control source member changed")
    control_meta = read(control_root / control_pages[0]["text_layers"])
    if sha((source_root / "source_members" / member).read_bytes()) != page["source_sha256"]:
        raise ValueError("source member hash changed")
    original = pixels(source_root / page["original"])
    if not np.array_equal(original, pixels(source_root / "source_members" / member)):
        raise ValueError("original/source pixel mismatch")
    meta = read(source_root / page["text_layers"])
    run_id = run_id or f"consumer-fast-{page['source_sha256'][:16]}"
    origin_execution_id = origin_execution_id or f"consumer-fast-{member}"
    context = source_project.get("context") or {}
    glossary = source_project.get("glossary") or source_project.get("glossario") or {}
    proposals = discover_recovery_proposals(meta["texts"], page["width"], page["height"],
                                             source_observations=graph_observations)
    outcomes = []
    for proposal in proposals:
        if proposal["kind"] == "split_connected_bodies":
            try:
                outcomes.append(_prepare_split(source_root=source_root,
                    control_root=control_root, control=control, page=page, meta=meta,
                    control_meta=control_meta,
                    proposal=proposal, original=original, member=member,
                    cache_root=cache_root if allow_prepared_responses else None,
                    font_path=font_path, run_id=run_id,
                    origin_execution_id=origin_execution_id, context=context,
                    glossary=glossary, ocr_invocation=ocr_invocation,
                    translation_invocation=translation_invocation,
                    allow_prepared_responses=allow_prepared_responses))
            except ProviderUnavailable as error:
                outcomes.append(dict(status="review_required", reason=str(error),
                                     proposal=proposal, source_member=member,
                                     source_preserved=True))
            continue
        if proposal["kind"] != "possible_fragmented_body":
            continue
        owners = [next(row for row in meta["texts"] if row["owner_id"] == identity)
                  for identity in proposal["owner_ids"]]
        all_observations = [item for owner in owners for item in
                            owner["owner_render_geometry"]["selected_observations"]]
        rough_boxes = [item["bbox_page"] for item in all_observations]
        crop_bbox = crop_for_observations(rough_boxes, page["width"], page["height"])
        x1, y1, x2, y2 = crop_bbox
        crop = original[y1:y2, x1:x2]
        try:
            job, adjudicated = _cached_or_fresh_ocr(
                page_rgb=original, crop=crop, bbox=crop_bbox, member=member,
                page_id=page["page_id"], source_sha256=page["source_sha256"],
                cache_root=cache_root, run_id=run_id,
                origin_execution_id=origin_execution_id,
                ocr_invocation=ocr_invocation, context=context,
                allow_cache_read=allow_prepared_responses)
        except ProviderUnavailable as error:
            outcomes.append(dict(status="review_required", reason=str(error),
                                 proposal=proposal, source_member=member,
                                 source_preserved=True))
            continue
        selected = adjudicated["selected"]
        boxes = [item["bbox_page"] for item in selected]
        source = adjudicated["review"]["selected_source"]
        try:
            target_binding = _accepted_or_fresh_translation(
                control_root=control_root, control=control, member=member,
                source_sha256=page["source_sha256"], source=source,
                owners=proposal["owner_ids"], page_id=page["page_id"],
                context=context, glossary=glossary,
                translation_invocation=translation_invocation,
                allow_accepted=allow_prepared_responses,
                runtime_cache_root=cache_root,
                allow_cache_read=allow_prepared_responses)
        except ProviderUnavailable as error:
            outcomes.append(dict(status="review_required", reason=str(error),
                                 proposal=proposal, source_member=member,
                                 source=source, source_preserved=True,
                                 selected_observation_ids=adjudicated["review"]["selected_observation_ids"]))
            continue
        target_binding = _reviewed_translation(target_binding, translation_review,
                                               member, page["source_sha256"], source)
        target, case = apply_source_case(source, target_binding["target"])
        active_group = (read(control_root / target_binding["group_path"])
                        if target_binding.get("group_path") else {})
        current_record = next((row for row in control_meta["texts"] if row["owner_id"] ==
                               proposal["owner_ids"][0]), None)
        if (active_group.get("source_recovery_kind") == "operational_source_observations_v1" and
                current_record is not None and current_record.get("source_payload") == source and
                current_record.get("translated_payload") == target and
                current_record.get("selected_observation_ids") ==
                    adjudicated["review"]["selected_observation_ids"] and
                (current_record.get("model_recovery") or {}).get("source_review_sha256") ==
                    adjudicated["source_review_sha256"]):
            outcomes.append(dict(status="already_published", proposal=proposal,
                                 source_member=member, group_sha256=target_binding["group_sha256"]))
            continue
        components = [item["bbox_page"] for owner in owners for item in
                      owner["owner_render_geometry"]["components"]]
        body_hint, anchor = union(components), union(boxes)
        rough_height = float(np.median([box[3]-box[1] for box in boxes]))
        def policy_for(minimum: int, ideal: int) -> dict:
            return build_source_white_policy_adaptive(
                source_root / page["original"], body_hint, anchor, boxes, minimum, ideal,
                source_member_bytes=(source_root / "source_members" / member).read_bytes())
        rough_policy = policy_for(max(3, round(.14*rough_height)),
                                  max(6, round(.30*rough_height)))
        if rough_policy["status"] != "body_validated":
            raise ValueError("source white body unproven: "+rough_policy["status"])
        prepared = prepare_source_white_comfort(rough_policy["policy"], page["source_sha256"])
        visual, visual_evidence = source_line_ink_height(original, prepared, boxes)
        policy_result = policy_for(max(3, round(.14*visual)), max(6, round(.30*visual)))
        if policy_result["status"] != "body_validated":
            raise ValueError("measured source white body unproven")
        policy = policy_result["policy"]
        prepared = prepare_source_white_comfort(policy, page["source_sha256"])
        old_clean = pixels(source_root / page["clean_base"])
        clean = old_clean.copy()
        cleanup = []
        for box in boxes:
            try:
                clean, delta = clear_residual_white_glyphs(clean, original, prepared, box)
                cleanup.append(delta)
            except ValueError as error:
                if str(error) != "no source-bound residual ink in observation":
                    raise
        font_size, preference = preferred_size(font_path, visual)
        safe = prepared["interior_bbox"]
        recipe = dict(schema="consumer_focal_text_v1", page_id=page["page_id"],
            owner_id=proposal["owner_ids"][0], source_member=member,
            source_sha256=page["source_sha256"], source=source, target=target,
            clean_pixels_sha256=sha(clean.tobytes()),
            adjudication=dict(observer="model_visual_review", human_review=False,
                source_sha256=page["source_sha256"],
                source_review_sha256=adjudicated["source_review_sha256"],
                translation_binding_sha256=target_binding["group_sha256"],
                target_provenance=target_binding["provenance"],
                editorial_review_sha256=target_binding.get("editorial_review_sha256")),
            font_path=str(font_path), font_sha256=sha(font_path.read_bytes()),
            font_size=font_size, minimum_font_size=max(18, round(.50*visual)),
            line_advance=font_size, line_spacing_profile="compact_safe_leading_v1",
            safe_bbox=safe, anchor_bbox=anchor, visual_comfort_policy=policy,
            source_case_profile=case["source_case_profile"])
        result = search_source_centered_layout(clean, recipe,
            source_visual_height_px=visual, preferred_font_size=font_size,
            minimum_font_size=recipe["minimum_font_size"],
            maximum_font_size=min(56, font_size+6), maximum_seconds=90)
        if result["status"] != "rendered_candidate":
            outcomes.append(dict(status="review_required", reason=result["reason"],
                                 proposal=proposal, source_member=member,
                                 source=source, target=target,
                                 target_binding=target_binding,
                                 source_preserved=True,
                                 source_review_sha256=adjudicated["source_review_sha256"],
                                 ocr_job_sha256=sha(json.dumps(
                                     job, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()),
                                 selected_observation_ids=
                                     adjudicated["review"]["selected_observation_ids"],
                                 crop_bbox=crop_bbox,
                                 crop_pixel_sha256=sha(crop.tobytes()),
                                 search_attempts=result["attempts"]))
            continue
        primary = proposal["owner_ids"][0]
        if primary not in {row["owner_id"] for row in meta["raster_cache"]}:
            raise ValueError("primary source raster absent")
        if any(row["owner_id"] == proposal["owner_ids"][1] for row in meta["raster_cache"]):
            raise ValueError("continuation has independent raster")
        layers = [result["layer"] if cache["owner_id"] == primary else RasterLayer(
            page["page_id"], cache["owner_id"], tuple(cache["bbox"]),
            pixels(source_root / cache["path"], "RGBA")) for cache in meta["raster_cache"]]
        final = flatten_layers(clean, layers)
        outcomes.append(dict(status="rendered_candidate", proposal=proposal,
            page=page, source=source, target=target, target_binding=target_binding,
            source_review_sha256=adjudicated["source_review_sha256"],
            ocr_job_sha256=sha(json.dumps(job,sort_keys=True,ensure_ascii=False,default=str).encode()),
            selected_observation_ids=adjudicated["review"]["selected_observation_ids"],
            crop_bbox=crop_bbox, crop_pixel_sha256=sha(crop.tobytes()),
            clean=clean, final=final, layer=result["layer"], recipe=result["recipe"],
            fit=result["evidence"], cleanup=cleanup,
            source_visual_evidence=visual_evidence, font_preference=preference,
            candidate_count=result["candidate_count"],
            metric_cache=result["metric_cache"]))
    return outcomes


def prepare_seam_member(*, control_root: Path, member: str, seam_evidence: Path,
                        translation_binding: Path, editorial_candidate: Path,
                        font_path: Path) -> dict:
    """Recover one seam-spanning source unit with both adjacent source hashes."""
    project = read(control_root / "project.json")
    pages = sorted(project["pages"], key=lambda row: row["source_index"])
    selected = [i for i, row in enumerate(pages) if row.get("source_member") == member]
    if len(selected) != 1 or selected[0] == 0:
        raise ValueError("seam source member absent or first")
    index = selected[0]
    prior, page = pages[index-1:index+1]
    if prior["source_index"] + 1 != page["source_index"]:
        raise ValueError("seam members not adjacent in full project")
    evidence = read(seam_evidence)
    members = evidence.get("source_members") or []
    if len(members) != 2 or [row["source_member"] for row in members] != [
            prior["source_member"], member]:
        raise ValueError("seam context differs from full project order")
    for row, observed in zip((prior, page), members):
        if row["source_sha256"] != observed["source_sha256"] or sha(
                (control_root / "source_members" / row["source_member"]).read_bytes()) != row["source_sha256"] or sha(
                (control_root / row["original"]).read_bytes()) != observed["project_original_sha256"]:
            raise ValueError("seam source hashes changed")
    meta = read(control_root / page["text_layers"])
    observation_ids = set(evidence["observation_ids"])
    rows = [row for row in meta["texts"] if {item["observation_id"] for item in
            (row.get("owner_render_geometry") or {}).get("selected_observations") or []} == observation_ids]
    if len(rows) != 1:
        raise ValueError("seam source owner absent or ambiguous")
    owner = rows[0]
    boxes = [item["bbox_page"] for item in owner["owner_render_geometry"]["selected_observations"]]
    anchor = union(boxes)
    if anchor != evidence["anchor_bbox"]:
        raise ValueError("seam anchor changed")
    bindings = read(translation_binding)
    accepted = [row for row in bindings if row.get("unit_id") == owner["owner_id"] and
                row.get("accepted") is True]
    if len(accepted) != 1:
        raise ValueError("corrected seam translation binding absent")
    candidate = read(editorial_candidate)
    if (candidate.get("source") != accepted[0]["source"] or
            candidate.get("provenance") != "model_editorial_candidate" or
            candidate.get("human_review") is not False or
            not all(candidate.get("fidelity_review", {}).values())):
        raise ValueError("seam editorial candidate unbound or incomplete")
    source, target = accepted[0]["source"], candidate["target"]
    target, case = apply_source_case(source, target)
    previous_recovery = owner.get("model_recovery") or {}
    if (owner.get("source_payload") == source and owner.get("translated_payload") == target and
            previous_recovery.get("seam_evidence_sha256") == sha(seam_evidence.read_bytes()) and
            previous_recovery.get("editorial_candidate_sha256") == sha(editorial_candidate.read_bytes()) and
            any(read(control_root / relative).get("owner_id") == owner["owner_id"]
                for relative in project.get("source_context_bindings") or [])):
        return dict(status="already_published", source_member=member,
                    owner_id=owner["owner_id"],
                    seam_evidence_sha256=previous_recovery["seam_evidence_sha256"])
    previous_rows = members[0]["crop_bbox"][3] - members[0]["crop_bbox"][1]
    current_rows = members[1]["crop_bbox"][3] - members[1]["crop_bbox"][1]
    def policy_for(minimum: int, ideal: int) -> dict:
        return build_source_white_policy_across_previous(
            control_root / prior["original"], control_root / page["original"],
            previous_index=prior["source_index"], current_index=page["source_index"],
            previous_member=prior["source_member"], current_member=member,
            previous_member_bytes=(control_root / "source_members" / prior["source_member"]).read_bytes(),
            current_member_bytes=(control_root / "source_members" / member).read_bytes(),
            previous_rows=previous_rows, current_rows=current_rows,
            anchor_bbox=anchor, observation_bboxes=boxes,
            minimum_px=minimum, ideal_px=ideal)
    rough = policy_for(4, 9)
    original, clean = pixels(control_root / page["original"]), pixels(control_root / page["clean_base"])
    visual, visual_evidence = source_line_ink_height(original,
        prepare_source_white_comfort(rough, page["source_sha256"]), boxes)
    policy = policy_for(max(3, round(.14*visual)), max(6, round(.30*visual)))
    if policy["source_crop_pixels_sha256"] != evidence["source_crop_sha256"]:
        raise ValueError("seam source crop changed")
    prepared = prepare_source_white_comfort(policy, page["source_sha256"])
    interior = prepared["interior_bbox"]
    safe = [max(0, interior[0]), max(0, interior[1]),
            min(clean.shape[1], interior[2]), min(clean.shape[0], interior[3])]
    font_size, preference = preferred_size(font_path, visual)
    recipe = dict(schema="consumer_focal_text_v1", page_id=page["page_id"],
        owner_id=owner["owner_id"], source_member=member,
        source_sha256=page["source_sha256"], source=source, target=target,
        clean_pixels_sha256=sha(clean.tobytes()),
        adjudication=dict(observer="model_visual_review", human_review=False,
            source_sha256=page["source_sha256"],
            seam_evidence_sha256=sha(seam_evidence.read_bytes()),
            translation_binding_sha256=sha(translation_binding.read_bytes()),
            editorial_candidate_sha256=sha(editorial_candidate.read_bytes())),
        font_path=str(font_path), font_sha256=sha(font_path.read_bytes()),
        font_size=font_size, minimum_font_size=20, line_advance=font_size,
        line_spacing_profile="compact_safe_leading_v1", safe_bbox=safe,
        anchor_bbox=anchor, visual_comfort_policy=policy,
        source_case_profile=case["source_case_profile"])
    result = search_source_centered_layout(clean, recipe,
        source_visual_height_px=visual, preferred_font_size=font_size,
        minimum_font_size=20, maximum_font_size=min(56, font_size+2),
        alternative_breaks=True, width_ratios=(1.0, .84, .68),
        maximum_candidates=30, maximum_seconds=90)
    if result["status"] != "rendered_candidate":
        return dict(status="review_required", reason=result["reason"],
                    source_member=member, search_attempts=result["attempts"])
    layers = [result["layer"] if row["owner_id"] == owner["owner_id"] else RasterLayer(
        page["page_id"], row["owner_id"], tuple(row["bbox"]),
        pixels(control_root / row["path"], "RGBA")) for row in meta["raster_cache"]]
    return dict(status="rendered_candidate", proposal=dict(kind="seam_single_owner",
        owner_ids=[owner["owner_id"]], context_members=policy["context_members"]),
        page=page, prior_page=prior, source=source, target=target, clean=clean,
        final=flatten_layers(clean, layers), layer=result["layer"], recipe=result["recipe"],
        fit=result["evidence"], source_visual_evidence=visual_evidence,
        font_preference=preference,
        seam_evidence_sha256=sha(seam_evidence.read_bytes()),
        translation_binding_sha256=sha(translation_binding.read_bytes()),
        editorial_candidate_sha256=sha(editorial_candidate.read_bytes()),
        candidate_count=result["candidate_count"],
        metric_cache=result["metric_cache"])
