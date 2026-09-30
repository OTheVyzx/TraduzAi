"""Normal Consumer Fast project open/reopen and persisted rerender gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from consumer_fast_core import RasterLayer, flatten_layers
from consumer_fast_recipe import load_recipe, rerender_recipe
from consumer_recipe_runtime import MAP_SHA, load_bindings, require_binding, rerender_historical, rerender_quality_case
from consumer_visual_comfort import SEAM_SCHEMA


def _asset(root: Path, relative: str) -> Path:
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('project asset missing or outside project: '+relative)
    return path


def _pixels(path: Path, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert(mode),dtype=np.uint8).copy()


def validate_local_subblock_groups(root: Path, project: dict) -> list[str]:
    """Require every physical member of a same-page logical utterance."""
    accepted=[]
    for relative in project.get('local_subblock_groups') or []:
        group=json.loads(_asset(root,relative).read_text(encoding='utf-8'))
        if group.get('schema')!='consumer_local_subblock_group_v1':
            raise ValueError('unsupported local subblock group')
        page=next((row for row in project['pages'] if row['page_id']==group.get('page_id')),None)
        if page is None or page['source_member']!=group.get('source_member') or page['source_sha256']!=group.get('source_sha256'):
            raise ValueError('local subblock source/page binding changed')
        source=_asset(root,'source_members/'+page['source_member'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=page['source_sha256']:
            raise ValueError('local subblock source hash changed')
        members=group.get('members') or []
        if len(members)<2 or [item.get('order') for item in members]!=list(range(len(members))):
            raise ValueError('local subblock requires ordered physical members')
        refs=json.loads(_asset(root,page['rerender_metadata']).read_text(encoding='utf-8'))['recipes']
        cache=json.loads(_asset(root,page['text_layers']).read_text(encoding='utf-8'))['raster_cache']
        ref_by_owner={item['owner_id']:item for item in refs}
        cache_owners={item['owner_id'] for item in cache}
        if len(ref_by_owner)!=len(refs) or len(cache_owners)!=len(cache):
            raise ValueError('duplicate local subblock owner')
        for item in members:
            owner=item.get('owner_id')
            ref=ref_by_owner.get(owner)
            if owner not in cache_owners or ref is None or ref['path']!=item.get('recipe_path') or ref['sha256']!=item.get('recipe_sha256'):
                raise ValueError('local subblock is only partially published')
        accepted.append(group['group_id'])
    return accepted


def validate_complete_balloon_reconciliations(root: Path, project: dict) -> list[str]:
    """A recovered whole balloon must not retain an active source fragment."""
    accepted=[]
    for relative in project.get('complete_balloon_reconciliations') or []:
        group=json.loads(_asset(root,relative).read_text(encoding='utf-8'))
        if group.get('schema')!='consumer_complete_balloon_reconciliation_v1':
            raise ValueError('unsupported complete balloon reconciliation')
        page=next((row for row in project['pages'] if row['page_id']==group.get('page_id')),None)
        if page is None or page['source_member']!=group.get('source_member') or page['source_sha256']!=group.get('source_sha256'):
            raise ValueError('complete balloon source/page binding changed')
        refs=json.loads(_asset(root,page['rerender_metadata']).read_text(encoding='utf-8'))['recipes']
        meta=json.loads(_asset(root,page['text_layers']).read_text(encoding='utf-8'))
        active={row['owner_id'] for row in meta['texts']}
        raster={row['owner_id'] for row in meta['raster_cache']}
        primary=group.get('current_owner_id')
        continuation=group.get('superseded_owner_id')
        if not primary or not continuation or primary==continuation:
            raise ValueError('complete balloon owner identity invalid')
        current=[row for row in refs if row['owner_id']==primary]
        if (len(current)!=1 or current[0]['path']!=group.get('recipe_path') or
            current[0]['sha256']!=group.get('recipe_sha256') or primary not in active or
            primary not in raster or continuation in active or continuation in raster or
            any(row['owner_id']==continuation for row in refs)):
            raise ValueError('complete balloon retains a fragment or loses current recipe')
        current_row=next(row for row in meta['texts'] if row['owner_id']==primary)
        if current_row.get('source_payload')!=group.get('source') or current_row.get('translated_payload')!=group.get('target'):
            raise ValueError('complete balloon text binding changed')
        accepted.append(group['group_id'])
    return accepted


def validate_source_context_bindings(root: Path, project: dict) -> list[str]:
    """Authenticate both physical sources behind a seam-bound focal recipe."""
    accepted = []
    by_index = {row['source_index']: row for row in project['pages']}
    for relative in project.get('source_context_bindings') or []:
        binding = json.loads(_asset(root, relative).read_text(encoding='utf-8'))
        if binding.get('schema') != 'consumer_source_context_binding_v1':
            raise ValueError('unsupported source context binding')
        members = binding.get('members') or []
        if len(members) != 2 or members[0]['source_index'] + 1 != members[1]['source_index']:
            raise ValueError('source context members are not adjacent')
        page = by_index.get(members[1]['source_index'])
        if page is None or page['page_id'] != binding.get('page_id'):
            raise ValueError('source context current page changed')
        refs = json.loads(_asset(root, page['rerender_metadata']).read_text(encoding='utf-8'))['recipes']
        ref = next((row for row in refs if row['path'] == binding.get('recipe_path') and
                    row['sha256'] == binding.get('recipe_sha256') and
                    row['owner_id'] == binding.get('owner_id')), None)
        if ref is None:
            raise ValueError('source context recipe binding changed')
        recipe = load_recipe(root, ref)
        policy = recipe.get('visual_comfort_policy') or {}
        if policy.get('schema') != SEAM_SCHEMA or policy.get('context_members') != members:
            raise ValueError('source context policy binding changed')
        previous_virtual, current_virtual = [member.get('virtual_bbox') for member in members]
        if (not previous_virtual or not current_virtual or
                previous_virtual[1] >= 0 or previous_virtual[3] != 0 or
                current_virtual[1] != 0 or
                previous_virtual[0] != current_virtual[0] or
                previous_virtual[2] != current_virtual[2] or
                policy.get('source_roi_bbox') != [previous_virtual[0], previous_virtual[1],
                                                  current_virtual[2], current_virtual[3]]):
            raise ValueError('source context reversible offsets changed')
        chunks = []
        for member in members:
            row = by_index.get(member['source_index'])
            if row is None or (row['source_member'], row['source_sha256']) != (
                    member['source_member'], member['source_sha256']):
                raise ValueError('source context member identity changed')
            source = _asset(root, 'source_members/' + row['source_member'])
            original = _asset(root, row['original'])
            if (hashlib.sha256(source.read_bytes()).hexdigest() != member['source_sha256'] or
                    hashlib.sha256(original.read_bytes()).hexdigest() != member['project_original_sha256']):
                raise ValueError('source context member hash changed')
            pixels = _pixels(original, 'RGB')
            if not np.array_equal(_pixels(source, 'RGB'), pixels):
                raise ValueError('source context member pixels changed')
            x1, y1, x2, y2 = member['crop_bbox']
            if not (0 <= x1 < x2 <= pixels.shape[1] and 0 <= y1 < y2 <= pixels.shape[0]):
                raise ValueError('source context crop outside member')
            virtual = member['virtual_bbox']
            if virtual[2] - virtual[0] != x2 - x1 or virtual[3] - virtual[1] != y2 - y1:
                raise ValueError('source context virtual crop dimensions changed')
            chunks.append(pixels[y1:y2, x1:x2])
        import cv2
        crop_bgr = cv2.imdecode(np.frombuffer(policy['source_crop_png'], np.uint8), cv2.IMREAD_COLOR)
        if crop_bgr is None or not np.array_equal(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                                                  np.concatenate(chunks, axis=0)):
            raise ValueError('source context crop differs from authenticated members')
        accepted.append(binding['recipe_sha256'])
    return accepted


def open_project(root: Path, *, rerender: bool = False) -> dict:
    root=Path(root).resolve()
    project=json.loads(_asset(root,'project.json').read_text(encoding='utf-8'))
    if project.get('runtime_profile')!='consumer_fast' or not isinstance(project.get('pages'),list):
        raise ValueError('not a Consumer Fast project')
    bindings=load_bindings(root,project) if rerender else {}
    source_context_bindings=validate_source_context_bindings(root,project)
    used_bindings=set()
    groups={}
    for rel in project.get('connected_groups') or []:
        group=json.loads(_asset(root,rel).read_text(encoding='utf-8'))
        if len(group['members'])!=2 or len(group['recipes'])!=2:
            raise ValueError('connected group requires both members and recipes')
        for member in group['members']:
            if hashlib.sha256(_asset(root,member['path']).read_bytes()).hexdigest()!=member['sha256']:
                raise ValueError('connected source hash changed')
        if group['members'][1]['source_index']!=group['members'][0]['source_index']+1:
            raise ValueError('connected source order is not adjacent')
        groups[group['group_id']]=group
    seen=[];pages=[]
    for page in project['pages']:
        pid=page['page_id']
        if pid in seen:raise ValueError('duplicate page identity')
        seen.append(pid)
        clean=_pixels(_asset(root,page['clean_base']),'RGB')
        original=_pixels(_asset(root,page['original']),'RGB')
        final=_pixels(_asset(root,page['final']),'RGB')
        source=_asset(root,'source_members/'+page['source_member'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=page['source_sha256']:
            raise ValueError('page source member hash changed')
        if not np.array_equal(_pixels(source,'RGB'),original):
            raise ValueError('project original differs from source member')
        if clean.shape!=original.shape or clean.shape!=final.shape:
            raise ValueError('page dimension mismatch')
        meta=json.loads(_asset(root,page['text_layers']).read_text(encoding='utf-8'))
        refs=json.loads(_asset(root,page['rerender_metadata']).read_text(encoding='utf-8'))
        if meta['page_id']!=pid or refs['page_id']!=pid:
            raise ValueError('page metadata identity mismatch')
        layers=[]
        runtime_rows=[]
        for cache in meta['raster_cache']:
            layers.append(RasterLayer(pid,cache['owner_id'],tuple(cache['bbox']),
                                      _pixels(_asset(root,cache['path']),'RGBA')))
        if not np.array_equal(flatten_layers(clean,layers),final):
            raise ValueError('persisted layers disagree with published final: '+pid)
        if len(refs['recipes'])!=len(layers):
            raise ValueError('recipe/layer count mismatch: '+pid)
        for index,(ref,layer) in enumerate(zip(refs['recipes'],layers)):
            if ref['owner_id']!=layer.owner_id:
                raise ValueError('recipe/layer owner mismatch')
            if rerender:
                recipe=load_recipe(root,ref)
                if isinstance(recipe,dict) and recipe.get('schema')=='consumer_focal_text_v1':
                    if (recipe.get('page_id')!=pid or recipe.get('owner_id')!=ref['owner_id'] or
                        recipe.get('source_member')!=page['source_member'] or
                        recipe.get('source_sha256')!=page['source_sha256']):
                        raise ValueError('focal recipe source or owner binding changed')
                if isinstance(recipe,dict) and 'layout' in recipe:
                    policy=((recipe.get('layout') or {}).get('texts') or [{}])[0].get('font_policy_v1') or {}
                    binding=bindings.get(ref['sha256'])
                    if not binding:
                        raise ValueError('editable recipe has no pinned runtime: '+pid+'/'+str(index))
                    if binding['family']=='font12_r003':
                        require_binding(recipe,ref,binding,pid)
                        if policy.get('map_sha256')!=MAP_SHA:
                            raise ValueError('historical map family unsupported')
                        actual_layer,runtime=rerender_historical(root,ref,binding,clean)
                    elif binding['family']=='quality_case_r1_focal':
                        if binding['page_id']!=pid or binding['owner_id']!=ref['owner_id']:
                            raise ValueError('focal runtime owner/page mismatch')
                        actual_layer,runtime=rerender_quality_case(root,ref,binding,clean)
                    else:
                        raise ValueError('unknown editable recipe runtime')
                    value=dict(exception=None,text_layers=[actual_layer])
                    used_bindings.add(ref['sha256'])
                    runtime_rows.append(dict(index=index,owner_id=ref['owner_id'],
                                             family=binding['family'],recipe_sha256=ref['sha256'],
                                             snapshot_sha256=binding.get('snapshot_sha256'),
                                             rasterized=True,face_sha256=runtime['face_sha256']))
                else:
                    value=rerender_recipe(clean,recipe)
                    schema=recipe.get('schema') if isinstance(recipe,dict) else None
                    runtime_rows.append(dict(index=index,owner_id=ref['owner_id'],
                        family=schema,rasterized=schema!='consumer_held_raster_v1',
                        held_raster=schema=='consumer_held_raster_v1'))
                actual=value['text_layers']
                if (value.get('exception') is not None or len(actual)!=1 or
                    (actual[0].page_id,actual[0].owner_id,actual[0].bbox)!=
                    (layer.page_id,layer.owner_id,layer.bbox) or
                    not np.array_equal(actual[0].rgba,layer.rgba)):
                    raise ValueError(f'rerender mismatch: {pid}/{index}/{ref["owner_id"]}')
        if page.get('preservation_decision'):
            evidence=json.loads(_asset(root,page['preservation_decision']).read_text(encoding='utf-8'))
            decisions=evidence.get('decisions') or []
            if len(decisions)!=1 or decisions[0]['visual_evidence']['source_sha256']!=page['source_sha256']:
                raise ValueError('vocalization decision source changed')
            decision=decisions[0]
            if decision['visual_evidence'].get('observer')!='model_visual_review' or decision['visual_evidence'].get('human_review') is not False:
                raise ValueError('vocalization decision provenance changed')
            bbox=decision['visual_evidence']['crop_bbox_page']
            crop_path=_asset(root,'evidence/vocalization_crop.png')
            if hashlib.sha256(crop_path.read_bytes()).hexdigest()!=decision['visual_evidence']['crop_sha256']:
                raise ValueError('vocalization crop hash changed')
            x1,y1,x2,y2=map(int,bbox)
            if not np.array_equal(original[y1:y2,x1:x2],_pixels(crop_path,'RGB')):
                raise ValueError('vocalization crop pixels changed')
            records=[r for r in meta['texts'] if r.get('owner_id')==decision['owner_id']]
            if len(records)!=1 or records[0].get('preserve_original') is not True:
                raise ValueError('vocalization preservation missing from normal text layer')
        pages.append(dict(page_id=pid,source_member=page.get('source_member'),
                          final_sha256=hashlib.sha256(_asset(root,page['final']).read_bytes()).hexdigest(),
                          layers=len(layers),rerendered=rerender,
                          runtime_rows=runtime_rows if rerender else [],
                          blockers=page.get('blockers') or []))
    if rerender and used_bindings!=set(bindings):
        raise ValueError('runtime manifest contains unused or absent recipes')
    for group in groups.values():
        if any(m['page_id'] not in seen for m in group['members']):
            raise ValueError('connected group member absent from project')
        linked=[r for page in project['pages'] for r in
                json.loads(_asset(root,page['rerender_metadata']).read_text(encoding='utf-8'))['recipes']
                if r['path'] in {x['path'] for x in group['recipes']}]
        if len(linked)!=2:
            raise ValueError('connected group is only partially published')
        if [m['page_id'] for m in group['members']]!=[
                next(p['page_id'] for p in project['pages'] if p['source_member']==m['member'])
                for m in group['members']]:
            raise ValueError('connected member/page mapping changed')
    if project['blocker_count']!=sum(len(p['blockers']) for p in project['pages']):
        raise ValueError('project blocker count diverges from current pages')
    local_groups=validate_local_subblock_groups(root,project)
    complete_groups=validate_complete_balloon_reconciliations(root,project)
    return dict(status='PASS',project=str(root/'project.json'),pages=pages,
                connected_groups=list(groups),rerendered=rerender,
                local_subblock_groups=local_groups,
                complete_balloon_reconciliations=complete_groups,
                source_context_bindings=source_context_bindings,
                export_gate=project['export_gate'],qa_status=project['qa_status'],
                verified=project['verified'])
