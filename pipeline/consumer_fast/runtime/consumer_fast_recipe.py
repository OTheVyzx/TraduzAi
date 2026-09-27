"""Exact local rerender recipes, without a duplicated RGB page.

The pickle is a trusted local runtime asset, NOT an untrusted project importer.
Its integrity hash does not authenticate an arbitrary external author.
"""
from pathlib import Path
import hashlib
import os
import pickle

_render_initialized = False


def save_recipe(root,page_id,index,value):
    if not page_id.startswith('page_') or not page_id[5:].isdigit():
        raise ValueError('invalid recipe page')
    path=Path('text_layers')/page_id/f'{index:04d}.recipe.pickle'
    target=Path(root)/path
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    payload=pickle.dumps(value,protocol=5)
    pending=target.with_suffix('.pending')
    pending.write_bytes(payload);os.replace(pending,target)
    return dict(path=path.as_posix(),sha256=hashlib.sha256(payload).hexdigest(),
                nbytes=len(payload),schema='consumer_local_render_recipe_v1',
                input_canvas='clean_base',trust='local_generated_only')


def load_recipe(root,reference):
    root=Path(root).resolve();target=(root/reference['path']).resolve()
    if not target.is_relative_to(root):raise ValueError('recipe path escapes project')
    payload=target.read_bytes()
    if len(payload)!=reference['nbytes'] or hashlib.sha256(payload).hexdigest()!=reference['sha256']:
        raise ValueError('recipe integrity mismatch')
    import owner_snapshot_codec
    recipe=pickle.loads(payload)
    if isinstance(recipe,dict) and recipe.get('schema')=='consumer_connected_fragment_v1':
        # A half cannot be independently reused after its neighbor, source or
        # other recipe is changed. This check is part of the ordinary loader.
        import json
        group_path=root/'connected_groups'/f"{recipe['group_id']}.json"
        group=json.loads(group_path.read_text(encoding='utf-8'))
        if group['group_id']!=recipe['group_id'] or len(group['members'])!=2:
            raise ValueError('connected group identity mismatch')
        refs=group['recipes']
        if len(refs)!=2 or not any(r['path']==reference['path'] and
              r['sha256']==reference['sha256'] for r in refs):
            raise ValueError('connected recipe is not a member of its group')
        for item in (*group['members'],*refs):
            asset=(root/item['path']).resolve()
            if not asset.is_relative_to(root) or not asset.is_file():
                raise ValueError('connected source or peer is absent')
            if hashlib.sha256(asset.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('connected source or peer changed')
    return recipe


def rerender_recipe(clean,recipe):
    global _render_initialized
    if isinstance(recipe,dict) and recipe.get('schema')=='consumer_focal_text_v1':
        from consumer_focal_text import render_reviewed_text
        layer, evidence = render_reviewed_text(clean, recipe)
        return dict(exception=None,text_layers=[layer],value=evidence)
    if isinstance(recipe,dict) and recipe.get('schema')=='consumer_held_raster_v1':
        # Source text outside the focal approval remains an exact, explicit
        # raster hold. Editing it requires a new review/revision, not a silent
        # rerender with a changed renderer.
        import numpy as np
        from consumer_fast_core import RasterLayer
        layer=RasterLayer(recipe['page_id'],recipe['owner_id'],
                          tuple(recipe['bbox']),recipe['rgba'])
        if recipe['clean_pixels_sha256']!=hashlib.sha256(clean.tobytes()).hexdigest():
            raise ValueError('held raster clean base changed')
        return dict(exception=None,text_layers=[layer],value=dict(
            status='held_outside_focal_scope',editable_metadata=True,
            source_recipe_sha256=recipe['source_recipe_sha256']))
    if isinstance(recipe,dict) and recipe.get('schema')=='consumer_connected_fragment_v1':
        from consumer_connected import fragment_layer
        layer,block=fragment_layer(clean,recipe)
        return dict(exception=None,text_layers=[layer],value=block)
    from integrated_render import RenderPacket, initialize_render
    from consumer_fast_render import render_job
    if not _render_initialized:
        initialize_render()
        _render_initialized = True
    packet=RenderPacket.create(clean,recipe['layout'],recipe['graph'],recipe['state'])
    value=render_job(packet)
    if value['exception'] is not None:raise value['exception']
    return value
