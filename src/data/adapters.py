"""Discover paired images without guessing that all datasets share labels."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import PurePosixPath

import numpy as np

from .storage import Store

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
FIRST = {'a': 'B', 'image1': 'Image2', 'time1': 'time2', 'im1': 'im2'}
PALETTE = {(255,255,255):0, (0,0,255):1, (128,128,128):2,
           (0,128,0):3, (0,255,0):4, (128,0,0):5, (255,0,0):6}


def scalar_mask(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        if a.shape[-1] != 3 or not np.all(a == a[..., :1]):
            raise ValueError('Expected replicated grayscale categorical channels')
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError('Expected a 2-D categorical mask')
    return a


def binary_mask(a: np.ndarray) -> np.ndarray:
    a = scalar_mask(a)
    if not set(np.unique(a)).issubset({0, 1, 255}):
        raise ValueError('Unexpected binary mask values; refuse grayscale colour coercion')
    return (a > 0).astype(np.uint8)


def semantic_mask(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        if not set(np.unique(a)).issubset(set(range(7))):
            raise ValueError('Unexpected SECOND class indices')
        return a.astype(np.uint8)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('SECOND requires RGB palette or class indices')
    result = np.full(a.shape[:2], 255, dtype=np.uint8)
    for color, value in PALETTE.items():
        result[np.all(a == color, -1)] = value
    if np.any(result == 255):
        raise ValueError('Unknown semantic palette colour')
    return result


def directional_mask(a: np.ndarray, color: tuple[int,int,int]) -> np.ndarray:
    """S2Looking source masks encode their foreground in blue/red, not gray."""
    if a.ndim==2 or (a.ndim==3 and np.all(a==a[...,:1])):
        return binary_mask(a)
    if a.ndim!=3 or a.shape[-1]!=3:
        raise ValueError('Expected RGB S2Looking mask')
    background=np.all(a==(0,0,0),-1);positive=np.all(a==color,-1)
    if not np.all(background|positive):
        raise ValueError('Unknown S2Looking label colour')
    return positive.astype(np.uint8)


def decode_targets(store: Store, row: dict, kind: str) -> dict[str, np.ndarray]:
    arrays = [store.image(p)[0] for p in row['labels']]
    if kind == 'mci':
        m = scalar_mask(arrays[0])
        if not set(np.unique(m)).issubset({0,128,255}):
            raise ValueError('Unknown LEVIR-MCI target value')
        return {'target': np.where(m == 128, 1, np.where(m == 255, 2, 0)).astype(np.uint8)}
    if kind == 'binary':
        return {'target': binary_mask(arrays[0])}
    if kind == 'directional_binary':
        first=directional_mask(arrays[0],(0,0,255));second=directional_mask(arrays[1],(255,0,0))
        return {'target':np.maximum(first,second),'source_label1':first,'source_label2':second}
    if kind == 'semantic':
        a, b = (semantic_mask(x) for x in arrays)
        return {'target': ((a != 0) | (b != 0)).astype(np.uint8), 'semantic_t1': a, 'semantic_t2': b,
                'building_change': ((a == 5) | (b == 5)).astype(np.uint8)}
    raise ValueError(f'Unsupported target kind: {kind}')


def discover_pairs(store: Store, dataset: dict) -> tuple[list[dict], list[dict]]:
    names = store.names()
    by_lower = {n.lower(): n for n in names}
    issues = []
    references = {}
    for n in names:
        if n.lower().endswith('captions.json') or n.endswith('SECOND-CC-AUG.json'):
            data = json.loads(store.read(n))
            for r in data.get('images', []):
                references[(r['filepath'], r['filename'])] = {'captions': [s['raw'].strip() for s in r['sentences']], 'changeflag': r.get('changeflag')}
    records = []
    for name in names:
        p = PurePosixPath(name)
        if p.suffix.lower() not in IMAGE_SUFFIXES or p.parent.name.lower() not in FIRST:
            continue
        if dataset['kind'] == 'semantic' and p.parent.parent.name.lower() == 'sem':
            continue
        if '_random_augment' in p.name:
            continue
        first_dir = p.parent.name
        base = p.parent.parent
        second = FIRST[first_dir.lower()]
        t2 = by_lower.get((base / second / p.name).as_posix().lower())
        parts = [s.lower() for s in p.parts]
        split = next((s for s in reversed(parts[:-1]) if s in {'train','val','test','training','validation','testing'}), 'unspecified')
        split = {'training':'train','validation':'val','testing':'test'}.get(split, split)
        kind = dataset['kind']
        if kind == 'directional_binary':
            candidates = [base/'label1'/p.name, base/'label2'/p.name]
        elif kind == 'semantic':
            # SECOND-CC: split/rgb/A with split/sem/A,B. Original SECOND: split/im1 with split/label1,label2.
            candidates = [base.parent/'sem'/'A'/p.name, base.parent/'sem'/'B'/p.name] if base.name.lower() == 'rgb' else [base/'label1'/p.name, base/'label2'/p.name]
        elif kind == 'captions':
            candidates = []
        else:
            options = [base/d/p.name for d in ['label','labels','mask','masks','label1']]
            candidates = [next((q for q in options if q.as_posix().lower() in by_lower), options[0])]
        labels = [by_lower.get(q.as_posix().lower()) for q in candidates]
        ref = references.get((split, p.name), {})
        if not t2 or any(x is None for x in labels):
            issues.append({'file': name, 'reason': 'missing_pair_or_target', 't2_found': bool(t2), 'labels_found': sum(x is not None for x in labels)})
            continue
        records.append({'pair_id': f"{dataset['id']}/{split}/{p.stem}", 'dataset': dataset['id'], 'official_split': split,
                        't1': name, 't2': t2, 'labels': labels, 'captions': ref.get('captions', []),
                        'source_changeflag': ref.get('changeflag'), 'kind': kind,
                        'image_family': dataset.get('image_family', dataset['id']), 'parent_scene_id': None})
    if len({r['pair_id'] for r in records}) != len(records):
        raise ValueError('Ambiguous pair IDs: nested release copies or stem collisions detected')
    return sorted(records, key=lambda r: r['pair_id']), issues
