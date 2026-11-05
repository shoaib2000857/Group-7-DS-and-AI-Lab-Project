"""Deterministic cleaning, leakage protection and task-specific preparation."""
from __future__ import annotations
from collections import defaultdict, Counter
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .adapters import decode_targets
from .catalog import load_catalog, write_json, write_jsonl
from .storage import Store, sha256

RANK = {'test':0,'val':1,'train':2,'unspecified':3}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def perceptual_candidates(profiles: list[dict], threshold: int=4) -> list[dict]:
    """Pairwise dHash candidates, explicitly distinct from confirmed identity."""
    valid=[p for p in profiles if p['valid'] and 't1_dhash' in p]
    bands=defaultdict(list);results=[]
    for idx,p in enumerate(valid):
        h=p['t1_dhash'];candidates=set()
        for band,(offset,width) in enumerate([(0,13),(13,13),(26,13),(39,13),(52,12)]):
            key=(band,(h>>offset)&((1<<width)-1));candidates.update(bands[key]);bands[key].append(idx)
        for j in candidates:
            other=valid[j]
            if p['official_split']==other['official_split']:continue
            a=(h^other['t1_dhash']).bit_count();b=(p['t2_dhash']^other['t2_dhash']).bit_count()
            if a<=threshold and b<=threshold:
                exact=sorted([p['t1_pixel_sha256'],p['t2_pixel_sha256']])==sorted([other['t1_pixel_sha256'],other['t2_pixel_sha256']])
                results.append({'pair_a':p['pair_id'],'pair_b':other['pair_id'],'t1_distance':a,'t2_distance':b,'exact_pair':exact})
    return results


def exclusion_policy(records: list[dict], profiles: list[dict], instructions: list[dict]) -> tuple[dict,list[dict]]:
    """Preserve official evaluation splits, including across related datasets."""
    reasons=defaultdict(set);pmap={p['pair_id']:p for p in profiles};rmap={r['pair_id']:r for r in records}
    exact=defaultdict(list);fingerprints=defaultdict(set)
    for r in records:
        p=pmap[r['pair_id']]
        if not p['valid']:
            reasons[r['pair_id']].add('invalid_input');continue
        if p.get('mask_caption_conflict'):
            reasons[r['pair_id']].add('mask_caption_changeflag_conflict')
        if (r['kind']=='mci' and len(r['captions'])!=5) or (r['captions'] and p['empty_captions']):
            reasons[r['pair_id']].add('invalid_caption_references')
        sig=tuple(sorted([p['t1_pixel_sha256'],p['t2_pixel_sha256']]))
        exact[(r['dataset'],r['kind'],sig)].append(r['pair_id'])
        for h in p['fingerprints']:
            fingerprints[h].add(r['pair_id'])
    groups=[]
    for members in exact.values():
        if len(members)<2:continue
        order=sorted(members,key=lambda x:(RANK[rmap[x]['official_split']],x))
        for pid in order[1:]:reasons[pid].add('exact_repeated_pair')
        targets={pmap[x].get('target_pixel_sha256') for x in members}
        if len(targets)>1:
            for pid in members:reasons[pid].add('inconsistent_duplicate_targets')
        groups.append({'members':order,'different_targets':len(targets)>1})
    # Any exact frame/tile reused across partitions conservatively excludes the
    # lower-priority record. The match may come from another related dataset.
    for members in fingerprints.values():
        best=min(RANK[rmap[x]['official_split']] for x in members)
        for pid in members:
            if RANK[rmap[pid]['official_split']]>best:
                reasons[pid].add('exact_frame_or_tile_cross_split')
    for inst in instructions:
        if inst.get('annotation_split')=='test' and inst['parent_pair_id'] in rmap:
            pid=inst['parent_pair_id']
            if rmap[pid]['official_split']=='train':reasons[pid].add('instruction_benchmark_parent_holdout')
    return {k:sorted(v) for k,v in reasons.items() if v},groups


def prepare(audit_dir: str|Path='outputs/data', output: str|Path='data/processed',
            config_path: str|Path|None=None, image_size: int|None=256) -> dict:
    audit_dir,output=Path(audit_dir),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    records=read_jsonl(audit_dir/'source_records.jsonl');profiles=read_jsonl(audit_dir/'profiles.jsonl')
    inst=read_jsonl(audit_dir/'instruction_records.jsonl')
    reasons,groups=exclusion_policy(records,profiles,inst)
    pmap={p['pair_id']:p for p in profiles}
    near=perceptual_candidates(profiles)
    for candidate in near:
        if candidate['exact_pair']:continue
        ids=[candidate['pair_a'],candidate['pair_b']]
        loser=max(ids,key=lambda pid:RANK[pmap[pid]['official_split']])
        reasons.setdefault(loser,[]).append('perceptual_cross_split_candidate')
    reasons={k:sorted(set(v)) for k,v in reasons.items()}
    write_json(audit_dir/'perceptual_candidates.json',near)
    stores={};prepared=[];refs={};lost=0
    try:
        for row in records:
            pid=row['pair_id']
            if pid in reasons:continue
            if row['official_split'] not in {'train','val','test'}:
                reasons[pid]=['missing_official_split'];continue
            source=row['source_path']
            if source not in stores:stores[source]=Store(source)
            store=stores[source]
            targets=decode_targets(store,row,row['kind']) if row['labels'] else {}
            size=(image_size,image_size) if image_size else (pmap[pid]['width'],pmap[pid]['height'])
            target_paths={};target_hashes={}
            for kind,array in targets.items():
                dst=output/'targets'/row['dataset']/row['official_split']/kind/(pid.rsplit('/',1)[-1]+'.png')
                dst.parent.mkdir(parents=True,exist_ok=True)
                scaled=np.array(Image.fromarray(array).resize(size,Image.Resampling.NEAREST))
                if kind=='target' and array.any() and not scaled.any():lost+=1
                Image.fromarray(scaled).save(dst)
                target_paths[kind]=dst.relative_to(output).as_posix();target_hashes[kind]=sha256(dst.read_bytes())
            prepared.append({k:v for k,v in row.items() if k not in {'captions','source_changeflag'}} | {
                't1_sha256':pmap[pid]['t1_sha256'],'t2_sha256':pmap[pid]['t2_sha256'],
                'target_paths':target_paths,'target_hashes':target_hashes,'original_size':[pmap[pid]['width'],pmap[pid]['height']],
                'prepared_size':list(size),'scale_xy':[size[0]/pmap[pid]['width'],size[1]/pmap[pid]['height']],
                'resize':'RGB bilinear, categorical nearest; full frame; no crop',
                'image_storage':'immutable original; model loader applies declared resize',
                'preparation_version':'main-data-v1'})
            refs[pid]={'captions':row['captions'],'source_changeflag':row['source_changeflag']}
    finally:
        for store in stores.values():store.close()
    for split in ['train','val','test']:
        write_jsonl(output/(split+'.jsonl'),[r for r in prepared if r['official_split']==split])
        safe=[{k:r[k] for k in ['pair_id','dataset','source_path','t1','t2','t1_sha256','t2_sha256','prepared_size']} for r in prepared if r['official_split']==split]
        write_jsonl(output/(split+'_inference.jsonl'),safe)
    write_json(output/'references.json',refs)
    write_json(output/'exclusions.json',reasons)
    write_json(output/'duplicate_groups.json',groups)
    tasks=build_task_manifests(prepared,refs,inst,output,load_catalog(config_path))
    result={'source_records':len(records),'prepared_records':len(prepared),'excluded_records':len(reasons),
            'by_dataset_split':dict(Counter(r['dataset']+'/'+r['official_split'] for r in prepared)),
            'exclusion_reasons_nonexclusive':dict(Counter(x for xs in reasons.values() for x in xs)),
            'duplicate_groups':len(groups),'whole_changed_masks_lost_after_resize':lost,'image_size':image_size,
            'tasks':tasks,'manifest_sha256':{p.name:sha256(p.read_bytes()) for p in output.glob('*.jsonl')}}
    write_json(audit_dir/'preparation_summary.json',result)
    return result


def build_task_manifests(records, refs, instructions, output, catalog):
    definitions={d['id']:d for d in catalog['datasets']}
    summary={};by_id={r['pair_id']:r for r in records}
    for task,spec in catalog['training_tasks'].items():
        if task=='instruction_tuning':continue
        rows=[]
        for r in records:
            if r['dataset'] not in spec['datasets']:continue
            if not definitions[r['dataset']].get('training_eligible',False):continue
            if task=='captioning' and not refs[r['pair_id']]['captions']:continue
            view=spec.get('views',{}).get(r['dataset'],'target')
            if task!='captioning' and view not in r['target_paths'] and view not in {'mci_building','mci_road','mci_any','semantic_pair'}:continue
            rows.append(dict(r,task=task,target_view=view))
        for split in ['train','val','test']:
            sub=[r for r in rows if r['official_split']==split]
            write_jsonl(output/'tasks'/task/(split+'.jsonl'),sub)
        summary[task]=dict(Counter(r['official_split'] for r in rows))
    accepted=[];seen=set()
    variant=catalog['training_tasks'].get('instruction_tuning',{}).get('variant')
    for r in instructions:
        parent=by_id.get(r['parent_pair_id'])
        if parent is None or r['role']!='instruction' or not r['has_source_pair']:continue
        split=parent['official_split']
        if split=='train' and variant and r['source_file']!=variant:continue
        if r.get('annotation_split')=='test' and split!='test':continue
        sig=(r['content_sha256'],split)
        if sig in seen:continue
        seen.add(sig);accepted.append(dict(r,prepared_split=split))
    for split in ['train','val','test']:
        write_jsonl(output/'tasks/instruction_tuning'/(split+'.jsonl'),[r for r in accepted if r['prepared_split']==split])
    summary['instruction_tuning']=dict(Counter(r['prepared_split'] for r in accepted))
    return summary
