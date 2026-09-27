"""Private structural-only restart checkpoints, never benchmark inputs.

Only load trusted checkpoints created locally by this module: pickle is not an
untrusted interchange format. A digest detects corruption, not a hostile author.
"""
from pathlib import Path
import hashlib
import json
import os
import pickle
from consumer_fast_core import atomic_json


def save(root,value):
    root=Path(root)
    payload=pickle.dumps(value,protocol=5)
    digest=hashlib.sha256(payload).hexdigest()
    root.mkdir(parents=True,exist_ok=False)
    pending=root/'payload.pending'
    pending.write_bytes(payload)
    os.replace(pending,root/'payload.pickle')
    atomic_json(root/'manifest.json',dict(schema='consumer_structural_frontier_v1',
        purpose='structural_only',sha256=digest,nbytes=len(payload)))


def load(root,*,structural):
    if not structural:raise ValueError('benchmark must not restore a frontier')
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='consumer_structural_frontier_v1' or manifest.get('purpose')!='structural_only':
        raise ValueError('unsupported structural checkpoint')
    payload=(root/'payload.pickle').read_bytes()
    if len(payload)!=manifest['nbytes'] or hashlib.sha256(payload).hexdigest()!=manifest['sha256']:
        raise ValueError('frontier digest mismatch')
    import owner_snapshot_codec  # original immutable-value pickle transport
    return pickle.loads(payload)
