"""Framework-neutral model inputs with explicit target/reference opt-in."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .storage import Store,sha256
from .preprocess import read_jsonl


class PairDataset:
    def __init__(self,manifest: str|Path,processed_root: str|Path='data/processed',include_targets: bool=True,include_references: bool=False,verify: bool=True):
        self.rows=read_jsonl(Path(manifest));self.root=Path(processed_root);self.targets=include_targets;self.verify=verify;self.stores={}
        self.refs=json.loads((self.root/'references.json').read_text(encoding='utf-8')) if include_references else None

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        r=self.rows[index];path=r['source_path']
        if path not in self.stores:self.stores[path]=Store(path)
        store=self.stores[path];out={'pair_id':r['pair_id']}
        for key in ['t1','t2']:
            a,h=store.image(r[key])
            if self.verify and h!=r[key+'_sha256']:raise ValueError('Source image changed after preparation')
            if a.ndim!=3 or a.shape[-1]!=3:raise ValueError('Expected RGB')
            out[key]=np.array(Image.fromarray(a).resize(tuple(r['prepared_size']),Image.Resampling.BILINEAR))
        if self.targets:
            out['targets']={}
            for key,relative in r.get('target_paths',{}).items():
                p=self.root/relative;b=p.read_bytes()
                if self.verify and sha256(b)!=r['target_hashes'][key]:raise ValueError('Prepared target changed')
                out['targets'][key]=np.array(Image.open(p)).astype(np.int64)
            if r.get('target_view')=='mci_building':out['targets']['target']=(out['targets']['target']==2).astype(np.int64)
            elif r.get('target_view')=='mci_road':out['targets']['target']=(out['targets']['target']==1).astype(np.int64)
            elif r.get('target_view')=='mci_any':out['targets']['target']=(out['targets']['target']>0).astype(np.int64)
            elif r.get('target_view')=='semantic_pair':
                out['targets']={'semantic_t1':out['targets']['semantic_t1'],'semantic_t2':out['targets']['semantic_t2'],'change':out['targets']['target']}
        if self.refs is not None:out['references']=self.refs[r['pair_id']]['captions']
        return out

    def batch(self,indices):
        samples=[self[i] for i in indices]
        images=np.stack([np.stack([s['t1'],s['t2']]).transpose(0,3,1,2) for s in samples]).astype(np.float32)/255
        result={'pair_ids':[s['pair_id'] for s in samples],'images':images}
        if self.targets and samples and 'target' in samples[0]['targets']:result['targets']=np.stack([s['targets']['target'] for s in samples])
        elif self.targets and samples and 'semantic_t1' in samples[0]['targets']:
            result['semantic_t1']=np.stack([s['targets']['semantic_t1'] for s in samples])
            result['semantic_t2']=np.stack([s['targets']['semantic_t2'] for s in samples])
            result['change_targets']=np.stack([s['targets']['change'] for s in samples])
        return result

    def close(self):
        for store in self.stores.values():store.close()
        self.stores.clear()


def paired_augmentation(t1,t2,target,operation,has_caption=False):
    """Geometry transforms share one operation; spatial captions are protected."""
    if has_caption:
        raise ValueError('Geometric augmentation requires separately transformed caption/region references')
    if operation=='flip_horizontal':fn=lambda a:np.flip(a,axis=1).copy()
    elif operation=='flip_vertical':fn=lambda a:np.flip(a,axis=0).copy()
    elif operation=='rotate_90':fn=lambda a:np.rot90(a,axes=(0,1)).copy()
    else:raise ValueError('Unknown paired augmentation')
    return fn(t1),fn(t2),fn(target)


class InstructionDataset:
    """Image-grouped conversations; generation mode never returns answers."""
    def __init__(self,manifest,processed_root='data/processed',include_targets=False,instruction_root=None):
        self.rows=read_jsonl(Path(manifest));self.targets=include_targets;self.instruction_root=Path(instruction_root) if instruction_root else None
        self.parents={}
        root=Path(processed_root)
        for split in ['train','val','test']:
            for r in read_jsonl(root/(split+'.jsonl')):self.parents[r['pair_id']]=r
        self.images=PairDataset(root/'train.jsonl',root,include_targets=False)
        self.images.rows=list(self.parents.values());self.parent_index={r['pair_id']:i for i,r in enumerate(self.images.rows)};self.files={}

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        r=self.rows[index];parent=r['parent_pair_id']
        path=(self.instruction_root or Path(r['source_directory']))/r['source_file']
        if path not in self.files:self.files[path]=json.loads(path.read_text(encoding='utf-8'))
        raw=self.files[path][r['source_index']]
        if sha256(json.dumps(raw,sort_keys=True).encode())!=r['content_sha256']:raise ValueError('Instruction changed after audit')
        conversations=raw['conversations'];prompt=next(t['value'] for t in conversations if t['from']=='human')
        result=self.images[self.parent_index[parent]]|{'instruction_id':r['instruction_id'],'prompt':prompt}
        if self.targets:
            result['messages']=[{'role':'user' if t['from']=='human' else 'assistant','content':t['value']} for t in conversations]
            result['supervised_roles']=['assistant']
        return result

    def close(self):self.images.close();self.files.clear()


def balanced_training_indices(rows,batch_size,weights=None,seed=42):
    """Draw source-balanced fitting examples, never validation/test records."""
    import random
    from collections import defaultdict
    if any(r['official_split']!='train' for r in rows):raise ValueError('Sampler accepts fitting records only')
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['dataset']].append(i)
    if not groups:raise ValueError('No training records')
    selected=weights or {k:1 for k in groups}
    if any(k not in groups and w>0 for k,w in selected.items()):raise ValueError('Requested source has no eligible training records')
    if any(w<0 for w in selected.values()) or sum(selected.values())<=0:raise ValueError('Invalid source weights')
    keys=[k for k in groups if selected.get(k,0)>0];rng=random.Random(seed)
    return [rng.choice(groups[k]) for k in rng.choices(keys,weights=[selected[k] for k in keys],k=batch_size)]
