"""Atomic, append-only publication of source-driven Consumer Fast recoveries."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from consumer_fast_core import RasterLayer, flatten_layers
from consumer_fast_project import open_project
from consumer_fast_recipe import load_recipe, rerender_recipe, save_recipe


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def pixels(path: Path, mode: str = "RGB") -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert(mode), dtype=np.uint8).copy()


def _next_ordinal(stage: Path, page: dict, meta: dict, refs: dict) -> int:
    """Allocate beyond active overlays and every historical recipe on disk."""
    used = {int(Path(row["path"]).stem) for row in meta["raster_cache"]}
    used.update(int(Path(row["path"]).name.split(".")[0]) for row in refs["recipes"])
    used.update(int(path.name.split(".")[0]) for path in
                (stage / "text_layers" / page["page_id"]).glob("[0-9]*.recipe*.pickle"))
    return max(used, default=-1) + 1


def _publish_split(stage: Path, project: dict, outcome: dict) -> dict:
    proposal, page_source = outcome["proposal"], outcome["page"]
    parent = proposal["owner_ids"][0]
    pages = [row for row in project["pages"] if row.get("source_member") == page_source["source_member"]]
    if len(pages) != 1 or pages[0]["source_sha256"] != page_source["source_sha256"]:
        raise ValueError("split source member changed")
    page = pages[0]
    meta, refs = read(stage / page["text_layers"]), read(stage / page["rerender_metadata"])
    groups = [(i, rel, read(stage / rel)) for i, rel in enumerate(
        project.get("local_subblock_groups") or [])
        if read(stage / rel).get("original_owner_id") == parent]
    if len(groups) != 1:
        raise ValueError("current split group absent or ambiguous")
    group_index, old_group_rel, group = groups[0]
    if (group["source_member"] != page["source_member"] or
            group["source_sha256"] != page["source_sha256"] or
            sha((stage / old_group_rel).read_bytes()) != outcome["binding"]["group_sha256"]):
        raise ValueError("current split group lineage changed")
    if len(group["members"]) != len(outcome["subunits"]):
        raise ValueError("split member count changed")
    clean = pixels(stage / page["clean_base"])
    if not np.array_equal(clean, outcome["clean"]):
        raise ValueError("split clean differs from published clean")
    old_final = pixels(stage / page["final"])
    allowed = np.zeros(clean.shape[:2], dtype=bool)
    replacements = {}
    saved_members = []
    previous_refs = []
    ordinal = _next_ordinal(stage, page, meta, refs)
    for unit, member in zip(outcome["subunits"], group["members"]):
        owner = unit["recipe"]["owner_id"]
        if owner != member["owner_id"] or unit["source"] != member["source"]:
            raise ValueError("split owner/source changed")
        cache_indices = [i for i, row in enumerate(meta["raster_cache"])
                         if row["owner_id"] == owner]
        if len(cache_indices) != 1:
            raise ValueError("split raster missing")
        index = cache_indices[0]
        old_cache, old_ref = meta["raster_cache"][index], refs["recipes"][index]
        if old_ref["owner_id"] != owner or old_ref["sha256"] != member["recipe_sha256"]:
            raise ValueError("split recipe lineage changed")
        layer = unit["layer"]
        if (unit["recipe"]["page_id"] != page["page_id"] or
                unit["recipe"]["source_sha256"] != page["source_sha256"] or
                unit["recipe"]["clean_pixels_sha256"] != sha(clean.tobytes())):
            raise ValueError("split recipe source changed")
        replay = rerender_recipe(clean, unit["recipe"])["text_layers"][0]
        if replay.bbox != layer.bbox or not np.array_equal(replay.rgba, layer.rgba):
            raise ValueError("split candidate recipe differs")
        for x1, y1, x2, y2 in (old_cache["bbox"], layer.bbox):
            allowed[y1:y2, x1:x2] = True
        new_ref = dict(owner_id=owner, **save_recipe(stage, page["page_id"],
                                                      ordinal, unit["recipe"]))
        cache_rel = f"by_source/{page['source_index']:03d}/overlay/{page['page_id']}/{ordinal:04d}.png"
        if (stage / cache_rel).exists():
            raise FileExistsError(cache_rel)
        (stage / cache_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(layer.rgba).save(stage / cache_rel)
        refs["recipes"][index] = new_ref
        meta["raster_cache"][index] = dict(owner_id=owner, bbox=list(layer.bbox), path=cache_rel)
        records = [row for row in meta["texts"] if row["owner_id"] == owner]
        if len(records) != 1:
            raise ValueError("split text record absent")
        record = records[0]
        record["prior_operational_record"] = {key: value for key, value in record.items()
                                              if key != "prior_operational_record"}
        record.update(source_payload=unit["source"], translated_payload=unit["target"],
                      selected_observation_ids=unit["selected_observation_ids"],
                      source_text_anchor_bbox=unit["recipe"]["anchor_bbox"],
                      layout_safe_bbox=unit["recipe"]["safe_bbox"],
                      render_layout_contract=unit["fit"], render_bbox=list(layer.bbox),
                      source_case_profile=unit["recipe"]["source_case_profile"],
                      model_recovery=dict(source_review_sha256=unit["source_review_sha256"],
                          ocr_job_sha256=unit["ocr_job_sha256"],
                          accepted_translation_group_sha256=outcome["binding"]["group_sha256"],
                          previous_recipe_sha256=old_ref["sha256"],
                          recipe_sha256=new_ref["sha256"], human_review=False))
        replacements[owner] = layer
        saved_members.append(dict(member, recipe_path=new_ref["path"],
                                  recipe_sha256=new_ref["sha256"],
                                  source=unit["source"], target=unit["target"],
                                  observation_ids=unit["selected_observation_ids"]))
        previous_refs.append(old_ref["sha256"])
        ordinal += 1
    layers = [replacements.get(row["owner_id"]) or RasterLayer(
        page["page_id"], row["owner_id"], tuple(row["bbox"]),
        pixels(stage / row["path"], "RGBA")) for row in meta["raster_cache"]]
    final = flatten_layers(clean, layers)
    if np.any(np.any(final != outcome["final"], axis=2) & allowed):
        raise ValueError("split candidate differs inside recovered owners")
    if np.any(np.any(old_final != final, axis=2) & ~allowed):
        raise ValueError("split recovery changed pixels outside current owners")
    Image.fromarray(final).save(stage / page["final"])
    write(stage / page["text_layers"], meta)
    write(stage / page["rerender_metadata"], refs)
    new_group = dict(group, members=saved_members,
                     prior_group_sha256=sha((stage / old_group_rel).read_bytes()),
                     source_recovery_kind="operational_source_observations_v1")
    new_group_rel = str(Path(old_group_rel).with_name("local_subblocks_operational_r001.json")).replace("\\", "/")
    if (stage / new_group_rel).exists():
        raise FileExistsError(new_group_rel)
    write(stage / new_group_rel, new_group)
    project["local_subblock_groups"][group_index] = new_group_rel
    for member in saved_members:
        ref = next((row for row in refs["recipes"] if row["owner_id"] == member["owner_id"]
                    and row["path"] == member["recipe_path"]), None)
        if ref is None:
            raise ValueError("saved split recipe reference missing")
        saved = rerender_recipe(clean, load_recipe(stage, ref))["text_layers"][0]
        layer = replacements[member["owner_id"]]
        if saved.bbox != layer.bbox or not np.array_equal(saved.rgba, layer.rgba):
            raise ValueError("saved split recipe differs")
    return dict(source_member=page["source_member"], page_id=page["page_id"],
                parent_owner=parent, previous_recipe_sha256=previous_refs,
                recipe_sha256=[row["recipe_sha256"] for row in saved_members],
                prior_group=old_group_rel, current_group=new_group_rel,
                old_final_sha256=sha(old_final.tobytes()), new_final_sha256=sha(final.tobytes()),
                changed_pixels=int(np.count_nonzero(np.any(old_final != final, axis=2))),
                old_clean_equals_new=True,
                new_recipe_rerender="exact_bbox_alpha_pixels")


def _publish_seam(stage: Path, project: dict, outcome: dict) -> dict:
    page_source, prior_source = outcome["page"], outcome["prior_page"]
    pages = {row["source_member"]: row for row in project["pages"]}
    page, prior = pages.get(page_source["source_member"]), pages.get(prior_source["source_member"])
    if (page is None or prior is None or page["source_sha256"] != page_source["source_sha256"] or
            prior["source_sha256"] != prior_source["source_sha256"] or
            prior["source_index"] + 1 != page["source_index"]):
        raise ValueError("seam source context changed")
    owner = outcome["proposal"]["owner_ids"][0]
    meta, refs = read(stage / page["text_layers"]), read(stage / page["rerender_metadata"])
    indices = [i for i, row in enumerate(meta["raster_cache"]) if row["owner_id"] == owner]
    if len(indices) != 1:
        raise ValueError("seam owner raster absent")
    index = indices[0]
    old_cache, old_ref = meta["raster_cache"][index], refs["recipes"][index]
    if old_ref["owner_id"] != owner:
        raise ValueError("seam old recipe owner changed")
    clean = pixels(stage / page["clean_base"])
    if not np.array_equal(clean, outcome["clean"]):
        raise ValueError("seam clean differs from published clean")
    recipe, layer = outcome["recipe"], outcome["layer"]
    if (recipe["source_sha256"] != page["source_sha256"] or
            recipe["clean_pixels_sha256"] != sha(clean.tobytes()) or
            recipe["owner_id"] != owner):
        raise ValueError("seam recipe source changed")
    replay = rerender_recipe(clean, recipe)["text_layers"][0]
    if replay.bbox != layer.bbox or not np.array_equal(replay.rgba, layer.rgba):
        raise ValueError("seam candidate recipe differs")
    old_final = pixels(stage / page["final"])
    layers = [layer if i == index else RasterLayer(page["page_id"], row["owner_id"],
              tuple(row["bbox"]), pixels(stage / row["path"], "RGBA"))
              for i, row in enumerate(meta["raster_cache"])]
    final = flatten_layers(clean, layers)
    if not np.array_equal(final, outcome["final"]):
        raise ValueError("seam candidate final differs from published composition")
    allowed = np.zeros(clean.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in (old_cache["bbox"], layer.bbox):
        allowed[y1:y2, x1:x2] = True
    if np.any(np.any(old_final != final, axis=2) & ~allowed):
        raise ValueError("seam recovery changed pixels outside source owner")
    ordinal = _next_ordinal(stage, page, meta, refs)
    new_ref = dict(owner_id=owner, **save_recipe(stage, page["page_id"], ordinal, recipe))
    cache_rel = f"by_source/{page['source_index']:03d}/overlay/{page['page_id']}/{ordinal:04d}.png"
    if (stage / cache_rel).exists():
        raise FileExistsError(cache_rel)
    (stage / cache_rel).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(layer.rgba).save(stage / cache_rel)
    refs["recipes"][index] = new_ref
    meta["raster_cache"][index] = dict(owner_id=owner, bbox=list(layer.bbox), path=cache_rel)
    rows = [row for row in meta["texts"] if row["owner_id"] == owner]
    if len(rows) != 1:
        raise ValueError("seam text record absent")
    row = rows[0]
    row["prior_operational_record"] = {key: value for key, value in row.items()
                                       if key != "prior_operational_record"}
    row.update(source_payload=outcome["source"], translated_payload=outcome["target"],
               source_text_anchor_bbox=recipe["anchor_bbox"],
               layout_safe_bbox=recipe["safe_bbox"],
               render_layout_contract=outcome["fit"], render_bbox=list(layer.bbox),
               source_case_profile=recipe["source_case_profile"],
               model_recovery=dict(seam_evidence_sha256=outcome["seam_evidence_sha256"],
                   translation_binding_sha256=outcome["translation_binding_sha256"],
                   editorial_candidate_sha256=outcome["editorial_candidate_sha256"],
                   previous_recipe_sha256=old_ref["sha256"],
                   recipe_sha256=new_ref["sha256"], human_review=False))
    Image.fromarray(final).save(stage / page["final"])
    write(stage / page["text_layers"], meta)
    write(stage / page["rerender_metadata"], refs)
    context_rel = f"by_source/{page['source_index']:03d}/text_layers/source_context_operational_r001.json"
    if (stage / context_rel).exists():
        raise FileExistsError(context_rel)
    context = dict(schema="consumer_source_context_binding_v1", page_id=page["page_id"],
                   owner_id=owner, recipe_path=new_ref["path"],
                   recipe_sha256=new_ref["sha256"],
                   members=recipe["visual_comfort_policy"]["context_members"],
                   seam_evidence_sha256=outcome["seam_evidence_sha256"],
                   editorial_candidate_sha256=outcome["editorial_candidate_sha256"])
    write(stage / context_rel, context)
    project.setdefault("source_context_bindings", []).append(context_rel)
    saved = rerender_recipe(clean, load_recipe(stage, new_ref))["text_layers"][0]
    if saved.bbox != layer.bbox or not np.array_equal(saved.rgba, layer.rgba):
        raise ValueError("saved seam recipe differs")
    return dict(source_member=page["source_member"], page_id=page["page_id"],
                owner_id=owner, previous_recipe_sha256=old_ref["sha256"],
                recipe_sha256=new_ref["sha256"], context_binding=context_rel,
                previous_member=prior["source_member"],
                old_final_sha256=sha(old_final.tobytes()), new_final_sha256=sha(final.tobytes()),
                changed_pixels=int(np.count_nonzero(np.any(old_final != final, axis=2))),
                old_clean_equals_new=True,
                new_recipe_rerender="exact_bbox_alpha_pixels")


def _publish_one(stage: Path, project: dict, outcome: dict) -> dict:
    if outcome["status"] != "rendered_candidate":
        raise ValueError("unrendered unit cannot publish")
    if outcome["proposal"]["kind"] == "split_connected_bodies":
        return _publish_split(stage, project, outcome)
    if outcome["proposal"]["kind"] == "seam_single_owner":
        return _publish_seam(stage, project, outcome)
    proposal, page_source = outcome["proposal"], outcome["page"]
    if proposal["kind"] != "possible_fragmented_body" or len(proposal["owner_ids"]) != 2:
        raise ValueError("unsupported physical recovery unit")
    pages = [row for row in project["pages"] if row["source_member"] == page_source["source_member"]]
    if len(pages) != 1:
        raise ValueError("control source member absent or ambiguous")
    page = pages[0]
    if page["source_sha256"] != page_source["source_sha256"]:
        raise ValueError("control source member changed")
    meta, refs = read(stage / page["text_layers"]), read(stage / page["rerender_metadata"])
    primary, continuation = proposal["owner_ids"]
    if continuation in {row["owner_id"] for row in meta["texts"] + meta["raster_cache"]}:
        raise ValueError("continuation still active in control project")
    indices = [i for i, row in enumerate(meta["raster_cache"]) if row["owner_id"] == primary]
    if len(indices) != 1 or len(refs["recipes"]) != len(meta["raster_cache"]):
        raise ValueError("current owner recipe/raster absent or ambiguous")
    index = indices[0]
    old_ref, old_cache = refs["recipes"][index], meta["raster_cache"][index]
    if old_ref["owner_id"] != primary:
        raise ValueError("current owner recipe differs")
    group_hits = [(i, rel, read(stage / rel)) for i, rel in enumerate(
        project.get("complete_balloon_reconciliations") or [])
        if read(stage / rel).get("source_member") == page["source_member"]]
    if len(group_hits) != 1:
        raise ValueError("current complete-balloon reconciliation missing")
    group_index, old_group_rel, group = group_hits[0]
    if (group["current_owner_id"], group["superseded_owner_id"], group["recipe_sha256"]) != (
            primary, continuation, old_ref["sha256"]):
        raise ValueError("current group does not bind current recipe and predecessors")
    clean = pixels(stage / page["clean_base"])
    if not np.array_equal(clean, outcome["clean"]):
        raise ValueError("new cleanup differs from published clean; retained layers need rebind")
    recipe = outcome["recipe"]
    layer = outcome["layer"]
    if (recipe["owner_id"] != primary or recipe["page_id"] != page["page_id"] or
            recipe["source_sha256"] != page["source_sha256"] or
            recipe["clean_pixels_sha256"] != sha(clean.tobytes())):
        raise ValueError("new recipe identity differs from current source")
    replay = rerender_recipe(clean, recipe)["text_layers"][0]
    if replay.bbox != layer.bbox or not np.array_equal(replay.rgba, layer.rgba):
        raise ValueError("new recipe does not reproduce candidate")
    old_final = pixels(stage / page["final"])
    layers = [layer if i == index else RasterLayer(page["page_id"], cache["owner_id"],
              tuple(cache["bbox"]), pixels(stage / cache["path"], "RGBA"))
              for i, cache in enumerate(meta["raster_cache"])]
    final = flatten_layers(clean, layers)
    allowed = np.zeros(clean.shape[:2], dtype=bool)
    for x1, y1, x2, y2 in (old_cache["bbox"], layer.bbox):
        allowed[y1:y2, x1:x2] = True
    if np.any(np.any(final != outcome["final"], axis=2) & allowed):
        raise ValueError("candidate differs inside recovered owner")
    if np.any(np.any(old_final != final, axis=2) & ~allowed):
        raise ValueError("recovery changed pixels outside current owner")
    ordinal = _next_ordinal(stage, page, meta, refs)
    new_ref = dict(owner_id=primary, **save_recipe(stage, page["page_id"], ordinal, recipe))
    cache_rel = f"by_source/{page['source_index']:03d}/overlay/{page['page_id']}/{ordinal:04d}.png"
    if (stage / cache_rel).exists():
        raise FileExistsError(cache_rel)
    (stage / cache_rel).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(layer.rgba).save(stage / cache_rel)
    refs["recipes"][index] = new_ref
    meta["raster_cache"][index] = dict(owner_id=primary, bbox=list(layer.bbox), path=cache_rel)
    current_rows = [row for row in meta["texts"] if row["owner_id"] == primary]
    if len(current_rows) != 1:
        raise ValueError("current text record absent")
    current = current_rows[0]
    current["prior_operational_record"] = {key: value for key, value in current.items()
                                            if key != "prior_operational_record"}
    current.update(source_payload=outcome["source"], translated_payload=outcome["target"],
                   selected_observation_ids=outcome["selected_observation_ids"],
                   source_text_anchor_bbox=recipe["anchor_bbox"], layout_safe_bbox=recipe["safe_bbox"],
                   render_layout_contract=outcome["fit"], render_bbox=list(layer.bbox),
                   source_case_profile=recipe["source_case_profile"],
                   model_recovery=dict(source_review_sha256=outcome["source_review_sha256"],
                       ocr_job_sha256=outcome["ocr_job_sha256"],
                       accepted_translation_group_sha256=outcome["target_binding"]["group_sha256"],
                       target_provenance=outcome["target_binding"]["provenance"],
                       editorial_review_sha256=outcome["target_binding"].get("editorial_review_sha256"),
                       previous_recipe_sha256=old_ref["sha256"],
                       recipe_sha256=new_ref["sha256"], human_review=False))
    Image.fromarray(final).save(stage / page["final"])
    write(stage / page["text_layers"], meta)
    write(stage / page["rerender_metadata"], refs)
    new_group = dict(group, recipe_path=new_ref["path"], recipe_sha256=new_ref["sha256"],
                     source=outcome["source"], target=outcome["target"],
                     selected_observation_ids=outcome["selected_observation_ids"],
                     prior_group_sha256=sha((stage / old_group_rel).read_bytes()),
                     target_provenance=outcome["target_binding"]["provenance"],
                     editorial_review_sha256=outcome["target_binding"].get("editorial_review_sha256"),
                     source_recovery_kind="operational_source_observations_v1")
    new_group_rel = str(Path(old_group_rel).with_name("complete_balloon_operational_r001.json")).replace("\\", "/")
    if (stage / new_group_rel).exists():
        raise FileExistsError(new_group_rel)
    write(stage / new_group_rel, new_group)
    project["complete_balloon_reconciliations"][group_index] = new_group_rel
    retired = [block for block in page.get("blockers") or []
               if block.get("owner_id") in {primary, continuation}]
    page["blockers"] = [block for block in page.get("blockers") or [] if block not in retired]
    saved = rerender_recipe(clean, load_recipe(stage, new_ref))["text_layers"][0]
    if saved.bbox != layer.bbox or not np.array_equal(saved.rgba, layer.rgba):
        raise ValueError("saved recipe raster differs")
    return dict(source_member=page["source_member"], page_id=page["page_id"],
                primary_owner=primary, retired_fragment=continuation,
                previous_recipe_sha256=old_ref["sha256"], recipe_sha256=new_ref["sha256"],
                prior_group=old_group_rel, current_group=new_group_rel,
                old_final_sha256=sha(old_final.tobytes()), new_final_sha256=sha(final.tobytes()),
                changed_pixels=int(np.count_nonzero(np.any(old_final != final, axis=2))),
                old_clean_equals_new=True, retired_blockers=retired,
                new_recipe_rerender="exact_bbox_alpha_pixels")


def publish(*, control_root: Path, destination: Path, outcomes: list[dict]) -> dict:
    control_root, destination = Path(control_root), Path(destination)
    if destination.exists():
        raise FileExistsError("append-only recovery destination already exists")
    if not outcomes or any(row["status"] != "rendered_candidate" for row in outcomes):
        raise ValueError("all publishing units require validated rendered candidates")
    parent_hash = sha((control_root / "project.json").read_bytes())
    stage = destination.with_name(destination.name+".building-"+uuid.uuid4().hex[:8])
    shutil.copytree(control_root, stage)
    try:
        project = read(stage / "project.json")
        changes = [_publish_one(stage, project, outcome) for outcome in outcomes]
        project.update(revision_kind="consumer_operational_source_recovery_v1",
                       parent_project_sha256=parent_hash,
                       qa_status="failed", verified=False, export_gate="BLOCK",
                       blocker_count=sum(len(page.get("blockers") or []) for page in project["pages"]))
        write(stage / "project.json", project)
        opened = open_project(stage, rerender=False)
        if opened["status"] != "PASS":
            raise ValueError("normal persisted loader rejected operational revision")
        audit = dict(schema="consumer_operational_publication_v1",
                     parent_project_sha256=parent_hash, changes=changes,
                     persisted_open_status=opened["status"],
                     human_review=False, export_gate="BLOCK")
        write(stage / "operational_recovery_audit.json", audit)
        os.replace(stage, destination)
        return audit
    except Exception as error:
        (stage / "failure.txt").write_text(repr(error), encoding="utf-8")
        raise
