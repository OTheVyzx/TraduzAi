"""Consumer-only compact outputs. No import or modification of STRICT runtime."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import contextvars
import json
import os
import threading
import time

import numpy as np


@dataclass(frozen=True)
class ConsumerProfile:
    runtime_profile: str = 'consumer_fast'
    qa_status: str = 'not_run'
    verified: bool = False
    window: int = 3

    def __post_init__(self):
        if (self.runtime_profile != 'consumer_fast' or self.qa_status != 'not_run'
                or self.verified or not 1 <= self.window <= 3):
            raise ValueError('invalid isolated consumer profile')


@dataclass(frozen=True)
class RasterLayer:
    page_id: str
    owner_id: str
    bbox: tuple
    rgba: np.ndarray

    def __post_init__(self):
        x1,y1,x2,y2=self.bbox
        if (not self.page_id or not self.owner_id or min(x1,y1)<0 or x2<=x1 or y2<=y1
                or self.rgba.dtype != np.uint8 or self.rgba.shape != (y2-y1,x2-x1,4)):
            raise ValueError('invalid text layer identity/geometry/RGBA')
        # Immutable independent storage: a caller cannot mutate the layer through an alias.
        pixels=np.frombuffer(self.rgba.tobytes(order='C'),np.uint8).reshape(self.rgba.shape)
        object.__setattr__(self,'rgba',pixels)


def flatten_layers(clean, layers):
    """Export/validation only. Match legacy float32 truncation, not PIL rounding.

    Keep owner layers ordered: collapsing several semi-transparent owners into a
    single uint8 PNG may change intermediate rounding. Individual layers remain
    the authoritative raster cache; editable text metadata remains canonical.
    """
    if clean.dtype != np.uint8 or clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError('clean base must be uint8 RGB')
    result=clean.copy()
    for layer in layers:
        x1,y1,x2,y2=layer.bbox
        if x2>result.shape[1] or y2>result.shape[0]:
            raise ValueError('layer exceeds page')
        alpha=layer.rgba[...,3:4].astype(np.float32)/255.0
        result[y1:y2,x1:x2]=np.clip(layer.rgba[...,:3].astype(np.float32)*alpha
            +result[y1:y2,x1:x2].astype(np.float32)*(1-alpha),0,255).astype(np.uint8)
    return result


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(pending,path)


def publish_page(root,page_id,clean,layers,texts):
    """Only the permanent base, editable metadata and art-free ROI caches."""
    from PIL import Image
    root=Path(root)
    if not page_id.startswith('page_') or not page_id[5:].isdigit():
        raise ValueError('unsafe page identity')
    if clean.dtype != np.uint8 or clean.ndim!=3 or clean.shape[2]!=3:
        raise ValueError('invalid clean RGB')
    base_path=root/'clean'/f'{page_id}_clean.png'
    base_path.parent.mkdir(parents=True,exist_ok=True)
    if base_path.exists():raise FileExistsError('clean base is immutable; do not overwrite')
    Image.fromarray(clean).save(base_path)
    caches=[]
    for i,layer in enumerate(layers):
        if layer.page_id!=page_id:raise ValueError('cross-page layer')
        rel=Path('overlay')/page_id/f'{i:04d}.png'
        (root/rel).parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(layer.rgba).save(root/rel)
        caches.append(dict(owner_id=layer.owner_id,bbox=list(layer.bbox),path=rel.as_posix()))
    metadata=dict(page_id=page_id,composition='ordered_float32_truncate_v1',
                  texts=texts,raster_cache=caches)
    atomic_json(root/'text_layers'/f'{page_id}.json',metadata)
    return dict(page_id=page_id,width=clean.shape[1],height=clean.shape[0],
                clean_base=base_path.relative_to(root).as_posix(),
                text_layers=f'text_layers/{page_id}.json')


class Metrics:
    """In-memory recorder; no per-span persistence, no worker time added to wall."""
    def __init__(self):
        self.start=time.perf_counter();self.rows=[];self.owner_tid=threading.get_ident()
        self.page=contextvars.ContextVar('consumer_metric_page',default=None)

    def event(self,kind,**values):
        self.rows.append(dict(kind=kind,monotonic=time.perf_counter(),pid=os.getpid(),
                              tid=threading.get_ident(),**values))

    def flush(self):
        pass  # Intentional: the consumer writes metrics once after shutdown.

    @contextmanager
    def page_scope(self,page):
        token=self.page.set(page)
        try:yield
        finally:self.page.reset(token)

    @contextmanager
    def span(self,phase,page=None,callsite=None):
        start=time.perf_counter();error=None
        try:yield
        except BaseException as exc:error=type(exc).__name__;raise
        finally:self.event('span',start=start,end=time.perf_counter(),phase=phase,
                           page=page or self.page.get(),callsite=callsite,error=error)

    def report(self):
        end=time.perf_counter()
        spans=[r for r in self.rows if r['kind']=='span' and r['tid']==self.owner_tid]
        cuts=sorted({self.start,end,*[max(self.start,min(end,r[k])) for r in spans for k in ('start','end')]})
        totals={}
        for left,right in zip(cuts,cuts[1:]):
            covering=[r for r in spans if r['start']<=left and r['end']>=right]
            phase=max(covering,key=lambda r:(r['start'],-r['end']))['phase'] if covering else 'unattributed'
            totals[phase]=totals.get(phase,0.)+right-left
        return dict(internal_wall=end-self.start,exclusive=totals,events=self.rows,
                    qa_status='not_run',verified=False,qa_seconds=0.,authenticated_compaction_seconds=0.)
