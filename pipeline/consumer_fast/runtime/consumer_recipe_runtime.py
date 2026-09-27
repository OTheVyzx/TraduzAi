"""Hash-bound, process-isolated rerender for a proven historical recipe family."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from consumer_fast_core import RasterLayer

RUNTIME_ROOT=Path(__file__).resolve().parent
PIPELINE=Path(__file__).resolve().parents[2]
ROOT=PIPELINE.parent
SNAPSHOT=PIPELINE/'consumer_fast/historical_runtime/font12_r003'
CHILD=Path(__file__).with_name('consumer_historical_child.py')
CURRENT_OVERLAY=RUNTIME_ROOT
CURRENT_PIPELINE_OVERLAY=PIPELINE
MAP_SHA='0832d735103c242bfd1bd7807079b3ca8148ab52c08e5a4401bdaa684a88b83c'
POLICY_SHA='a5ec1672357cdfb73aa9151e258ae27fcae3f471a2c5d582ef2b0c17af09b03d'


def _sha(path: Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_snapshot_root() -> Path:
    configured=os.environ.get('TRADUZAI_HISTORICAL_RUNTIME_ROOT')
    root=Path(configured).expanduser() if configured else SNAPSHOT
    manifest=root/'SNAPSHOT.json'
    if not manifest.is_file():
        raise FileNotFoundError(
            'historical recipe runtime is unavailable; set '
            'TRADUZAI_HISTORICAL_RUNTIME_ROOT to the read-only font12_r003 snapshot'
        )
    return root.resolve()


def verify_snapshot() -> dict:
    snapshot=historical_snapshot_root()
    data=json.loads((snapshot/'SNAPSHOT.json').read_text(encoding='utf8'))
    if (data.get('runtime')!='font12_r003' or
            _sha(Path(data['source_zip']))!=data['source_zip_sha256']):
        raise ValueError('font12-r003 source archive changed')
    for row in data['files']:
        path=snapshot/row['path']
        if _sha(path)!=row['sha256']:raise ValueError('font12-r003 module changed: '+row['path'])
    return data


def load_bindings(root: Path,project: dict) -> dict[str,dict]:
    declaration=project.get('recipe_runtime_manifest')
    if declaration is None:return {}
    path=(root/declaration['path']).resolve()
    if not path.is_relative_to(root) or _sha(path)!=declaration['sha256']:
        raise ValueError('recipe runtime manifest integrity mismatch')
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('schema')!='consumer_recipe_runtime_manifest_v1':
        raise ValueError('unknown recipe runtime manifest')
    snapshot=historical_snapshot_root() if any(
        item.get('family')=='font12_r003' for item in data['bindings']
    ) else None
    snapshot_sha=_sha(snapshot/'SNAPSHOT.json') if snapshot is not None else None
    bindings={}
    for item in data['bindings']:
        key=item['recipe_sha256']
        if key in bindings:raise ValueError('ambiguous recipe runtime binding')
        if item['family']=='font12_r003':
            if item['snapshot_sha256']!=snapshot_sha or item['font_map_sha256']!=MAP_SHA or item['font_policy_sha256']!=POLICY_SHA:
                raise ValueError('historical font contract mismatch')
        elif item['family']=='quality_case_r1_focal':
            required={'renderer','stable_baseline','font_policy','integrated_render',
                      'consumer_fast_render','consumer_fast_core','consumer_fast_recipe',
                      'ownership.execution','strip.process_bands','worker_bootstrap'}
            if {module['name'] for module in item['modules']}!=required:
                raise ValueError('focal recipe runtime module inventory incomplete')
            for module in item['modules']:
                path=Path(module['path'])
                if not path.is_file() or _sha(path)!=module['sha256']:
                    raise ValueError('focal recipe runtime module changed: '+str(path))
            map_path=Path(item['font_map_path'])
            policy_path=Path(item['font_policy_path'])
            if (_sha(map_path)!=item['font_map_sha256'] or
                    _sha(policy_path)!=item['font_policy_sha256']):
                raise ValueError('focal recipe font contract changed')
        else:
            raise ValueError('unknown recipe runtime family')
        bindings[key]=item
    if any(item['family']=='font12_r003' for item in bindings.values()):verify_snapshot()
    return bindings


def require_binding(recipe: dict,reference: dict,binding: dict|None,page_id: str) -> None:
    block=(recipe.get('layout') or {}).get('texts',[{}])[0]
    policy=block.get('font_policy_v1') or {}
    if not binding or (binding['recipe_sha256']!=reference['sha256'] or
            binding['recipe_path']!=reference['path'] or binding['page_id']!=page_id or
            binding['owner_id']!=reference['owner_id'] or
            policy.get('map_sha256')!=binding['font_map_sha256'] or
            policy.get('policy_sha256')!=binding['font_policy_sha256'] or
            policy.get('font_sha256')!=binding['font_sha256'] or
            binding['baseline_policy_sha256']!=_sha(historical_snapshot_root()/'typesetter/baseline-policy.json')):
        raise ValueError('historical recipe lacks exact font12-r003 binding')


def rerender_historical(root: Path,reference: dict,binding: dict,clean: np.ndarray) -> tuple[RasterLayer,dict]:
    from consumer_fast_recipe import load_recipe
    recipe=load_recipe(root,reference)
    require_binding(recipe,reference,binding,binding['page_id'])
    verify_snapshot()
    snapshot=historical_snapshot_root()
    env=os.environ.copy()
    removed={name:env.pop(name) for name in ('TRADUZAI_QUALITY_FONT_MAP',)
             if name in env}
    env['TRADUZAI_QUALITY_CLOSED_FONTS']='1'
    env['PYTHONPATH']=os.pathsep.join(map(str,[snapshot,RUNTIME_ROOT,PIPELINE]))
    with tempfile.TemporaryDirectory(prefix='font12-r003-') as temp:
        work=Path(temp)
        completed=subprocess.run([sys.executable,str(CHILD),
            str(root),reference['path'],reference['sha256'],str(work)],
            env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=600)
        if completed.returncode:
            raise ValueError('isolated historical render failed: '+completed.stderr[-1500:])
        result=json.loads((work/'result.json').read_text(encoding='utf8'))
        array=np.load(work/'rgba.npy',allow_pickle=False)
        if result['rgba_sha256']!=hashlib.sha256(array.tobytes()).hexdigest():
            raise ValueError('historical render transport integrity mismatch')
        if result['recipe_sha256']!=reference['sha256'] or result['owner_id']!=binding['owner_id']:
            raise ValueError('historical render identity mismatch')
        if result['clean_pixels_sha256']!=hashlib.sha256(clean.tobytes()).hexdigest():
            raise ValueError('historical render clean base mismatch')
        layer=RasterLayer(binding['page_id'],binding['owner_id'],tuple(result['bbox']),array)
        result['removed_overrides']=removed
        return layer,result


def rerender_quality_case(root: Path,reference: dict,binding: dict,clean: np.ndarray) -> tuple[RasterLayer,dict]:
    """Rerasterize a new focal recipe in a hash-pinned clean subprocess."""
    from consumer_fast_recipe import load_recipe
    if (binding['family']!='quality_case_r1_focal' or binding['recipe_sha256']!=reference['sha256']
            or binding['recipe_path']!=reference['path']):
        raise ValueError('focal recipe runtime binding mismatch')
    load_recipe(root,reference)
    for module in binding['modules']:
        if _sha(Path(module['path']))!=module['sha256']:
            raise ValueError('focal recipe runtime module changed: '+module['path'])
    env=os.environ.copy()
    env['TRADUZAI_QUALITY_CLOSED_FONTS']='1'
    env['TRADUZAI_QUALITY_FONT_MAP']=binding['font_map_path']
    env['PYTHONPATH']=os.pathsep.join(map(str,[CURRENT_OVERLAY,CURRENT_PIPELINE_OVERLAY]))
    with tempfile.TemporaryDirectory(prefix='quality-case-r1-') as temp:
        work=Path(temp)
        completed=subprocess.run([sys.executable,str(CHILD),
            str(root),reference['path'],reference['sha256'],str(work)],
            env=env,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=600)
        if completed.returncode:
            raise ValueError('isolated focal rerender failed: '+completed.stderr[-1500:])
        result=json.loads((work/'result.json').read_text(encoding='utf8'))
        array=np.load(work/'rgba.npy',allow_pickle=False)
        if result['rgba_sha256']!=hashlib.sha256(array.tobytes()).hexdigest():
            raise ValueError('focal render transport integrity mismatch')
        if (result['recipe_sha256']!=reference['sha256'] or
                result['owner_id']!=binding['owner_id'] or
                result['clean_pixels_sha256']!=hashlib.sha256(clean.tobytes()).hexdigest() or
                result['font_map_sha256']!=binding['font_map_sha256'] or
                result['font_policy_sha256']!=binding['font_policy_sha256']):
            raise ValueError('focal rerender contract changed')
        indexed={module['name']:module for module in binding['modules']}
        if not {'renderer','stable_baseline','font_policy','integrated_render','consumer_fast_render'} <= set(result['modules']):
            raise ValueError('focal rerender module audit incomplete')
        for name, module in result['modules'].items():
            if name in indexed and (module['sha256']!=indexed[name]['sha256'] or
                                    module['path']!=indexed[name]['path']):
                raise ValueError('focal rerender loaded a different '+name)
        layer=RasterLayer(binding['page_id'],binding['owner_id'],tuple(result['bbox']),array)
        return layer,result
