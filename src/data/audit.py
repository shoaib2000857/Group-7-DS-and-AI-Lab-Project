"""Source-backed, task-aware EDA for paired remote-sensing datasets."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from .adapters import decode_targets, discover_pairs
from .catalog import load_catalog, resolve_source, write_json, write_jsonl
from .storage import Store, pixel_hash, sha256


def image_fingerprints(a: np.ndarray) -> list[str]:
    """Full frames and aligned 256px tiles help detect cross-release reuse.

    This detects exact pixel matches only; it cannot establish scene independence
    under resampling, radiometric changes or arbitrary overlapping crops.
    """
    result = [pixel_hash(a)]
    h, w = a.shape[:2]
    if h > 256 or w > 256:
        for y in range(0, h-255, 256):
            for x in range(0, w-255, 256):
                tile = a[y:y+256, x:x+256]
                if np.std(tile) > 0:
                    result.append(pixel_hash(tile))
    return sorted(set(result))


def profile_pair(store: Store, row: dict) -> dict:
    result = {k: row[k] for k in ['pair_id','dataset','official_split','kind','image_family','source_changeflag']}
    try:
        a, h1 = store.image(row['t1'])
        b, h2 = store.image(row['t2'])
        if a.ndim != 3 or a.shape[-1] != 3 or a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValueError('Expected matching RGB uint8 input shapes')
        targets = decode_targets(store, row, row['kind']) if row['labels'] else {}
        if any(m.shape != a.shape[:2] for m in targets.values()):
            raise ValueError('Target/input dimensions disagree')
        mask = targets.get('target')
        result.update(valid=True, width=a.shape[1], height=a.shape[0], t1_sha256=h1, t2_sha256=h2,
                      t1_pixel_sha256=pixel_hash(a), t2_pixel_sha256=pixel_hash(b), identical_dates=bool(np.array_equal(a,b)),
                      fingerprints=image_fingerprints(a)+image_fingerprints(b))
        for key,arr in [('t1',a),('t2',b)]:
            small=np.array(Image.fromarray(arr).convert('L').resize((9,8),Image.Resampling.LANCZOS))
            result[key+'_dhash']=int.from_bytes(np.packbits(small[:,1:]>small[:,:-1]).tobytes(),'big')
        for key, arr in [('t1',a),('t2',b)]:
            lum = arr @ np.array([.2126,.7152,.0722])
            result[key+'_brightness'] = float(lum.mean())
            result[key+'_contrast'] = float(lum.std())
            result[key+'_saturated_fraction'] = float(((arr == 0) | (arr == 255)).mean())
            result[key+'_rgb_mean'] = arr.mean(axis=(0,1)).tolist()
        result['brightness_delta'] = result['t2_brightness']-result['t1_brightness']
        result['pixel_mae'] = float(np.abs(a.astype(float)-b).mean())
        if mask is not None:
            counts = dict(zip(*np.unique(mask, return_counts=True)))
            result['label_counts'] = {str(int(k)):int(v) for k,v in counts.items()}
            result['target_pixel_sha256'] = pixel_hash(mask)
            result['changed_fraction'] = float((mask>0).mean())
            result['mask_class'] = ('both' if np.any(mask==1) and np.any(mask==2) else 'road_only' if np.any(mask==1) else 'building_only' if np.any(mask==2) else 'unchanged') if row['kind']=='mci' else 'changed' if mask.any() else 'unchanged'
            result['mask_caption_conflict'] = row['source_changeflag'] is not None and bool(row['source_changeflag']) != bool(mask.any())
            result['components_8'] = int(ndimage.label(mask>0,structure=np.ones((3,3)))[1])
            result['grid_changed_fraction'] = [[float((block>0).mean()) for block in np.array_split(strip,3,axis=1)] for strip in np.array_split(mask,3,axis=0)]
            result['target_sha256'] = [sha256(store.read(p)) for p in row['labels']]
        if 'semantic_t1' in targets:
            c = Counter((targets['semantic_t1'].astype(int)*7+targets['semantic_t2']).ravel().tolist())
            result['transitions'] = {f'{k//7}->{k%7}':int(v) for k,v in c.items()}
            aligned=sorted([(result['t1_pixel_sha256'],pixel_hash(targets['semantic_t1'])),
                            (result['t2_pixel_sha256'],pixel_hash(targets['semantic_t2']))])
            result['target_pixel_sha256']=sha256(json.dumps(aligned).encode())
        caps = row['captions']
        result['caption_count'] = len(caps)
        result['empty_captions'] = sum(not c.strip() for c in caps)
        result['caption_tokens_mean'] = float(np.mean([len(re.findall(r'\b\w+\b',c)) for c in caps])) if caps else None
        result['direction_caption_count'] = sum(bool(re.search(r'\b(left|right|top|bottom|upper|lower|centre|center|middle)\b',c.lower())) for c in caps)
    except Exception as exc:
        result.update(valid=False,error=f'{type(exc).__name__}: {exc}')
    return result


def aggregate(profiles: list[dict]) -> dict:
    good = [r for r in profiles if r['valid']]
    labels, transitions = Counter(), Counter()
    for r in good:
        labels.update(r.get('label_counts',{}))
        transitions.update(r.get('transitions',{}))
    total = sum(labels.values())
    numeric = ['changed_fraction','brightness_delta','pixel_mae','caption_tokens_mean','components_8']
    frame = pd.DataFrame(good)
    available = [k for k in numeric if k in frame and frame[k].notna().any()]
    outliers = {}
    for k in available:
        s = frame[k].dropna()
        q1,q3 = s.quantile([.25,.75]);lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
        outliers[k] = {'lower':float(lo),'upper':float(hi),'flagged':int(((s<lo)|(s>hi)).sum())}
    correlations = frame[available].corr().replace({np.nan:None}).to_dict() if available else {}
    return {'records_audited':len(profiles),'valid_records':len(good),'invalid_records':len(profiles)-len(good),
            'split_counts':dict(Counter(r['official_split'] for r in profiles)),
            'mask_classes':dict(Counter(r.get('mask_class','unlabelled') for r in good)),
            'label_pixels':dict(labels),'label_pixel_percent':{k:100*v/total for k,v in labels.items()} if total else {},
            'mask_caption_conflicts':sum(r.get('mask_caption_conflict',False) for r in good),
            'identical_dates':sum(r.get('identical_dates',False) for r in good),
            'small_changes_below_one_percent':sum(0<r.get('changed_fraction',0)<.01 for r in good),
            'caption_references':sum(r['caption_count'] for r in good),'empty_caption_references':sum(r['empty_captions'] for r in good),
            'shapes':dict(Counter(f"{r['height']}x{r['width']}" for r in good)),
            'numeric_summary':{k:{str(q):float(v) for q,v in frame[k].dropna().quantile([0,.25,.5,.75,1]).items()} for k in available},
            'outliers_iqr':outliers,'correlations':correlations,'transition_pixels':dict(transitions)}


def complete_reference_checks(records: list[dict],profiles: list[dict]) -> dict:
    """Additional source-caption and optional colour-copy consistency checks."""
    pmap={p['pair_id']:p for p in profiles};stores={};names={};diagnostics=defaultdict(Counter);templates=defaultdict(Counter)
    try:
        for row in records:
            p=pmap[row['pair_id']];did=row['dataset'];caps=row['captions']
            if caps:
                flags=[bool(re.search(r'no (?:change|difference)|nothing (?:has )?changed|same as before|seem identical|seem the same|almost nothing',c.lower())) for c in caps]
                p['mixed_reference_flags']=any(flags) and not all(flags)
                diagnostics[did]['lexically_mixed_reference_pairs']+=int(p['mixed_reference_flags'])
                p['caption_set_sha256']=sha256(json.dumps(sorted(caps)).encode())
                templates[did][p['caption_set_sha256']]+=1
            if row['kind']!='mci' or not p['valid']:continue
            source=row['source_path']
            if source not in stores:
                stores[source]=Store(source);names[source]=set(stores[source].names())
            member=row['labels'][0].replace('/label/','/label_rgb/')
            if member not in names[source]:
                diagnostics[did]['missing_optional_colour_copies']+=1;continue
            target=decode_targets(stores[source],row,'mci')['target']
            visual=stores[source].image(member)[0]
            palette=np.array([[0,0,0],[255,255,0],[255,0,0]],dtype=np.uint8)
            equal=visual.shape==(*target.shape,3) and np.array_equal(visual,palette[target])
            p['palette_equivalent']=bool(equal);diagnostics[did]['colour_copies_checked']+=1
            if not equal:
                p['valid']=False;p['error']='Colour-copy/class-mask inconsistency';diagnostics[did]['palette_mismatches']+=1
    finally:
        for store in stores.values():store.close()
    result={}
    for did in {r['dataset'] for r in records}:
        result[did]=dict(diagnostics[did])|{'distinct_caption_sets':len(templates[did]),'largest_repeated_caption_set':max(templates[did].values(),default=0)}
    return result


def audit_instructions(path: Path, dataset: dict, image_index: dict) -> tuple[list[dict],dict]:
    """Join every instruction to its underlying LEVIR pair before any splitting."""
    files = sorted(path.glob('*.json')) if path.is_dir() else [path]
    rows = []
    malformed = Counter()
    for file in files:
        data = json.loads(file.read_text(encoding='utf-8'))
        for i,r in enumerate(data):
            images = r.get('image') or [r.get('image_A'),r.get('image_B')]
            conversations = r.get('conversations',[])
            if len(images)!=2 or not all(isinstance(s,str) for s in images):
                malformed['invalid_image_pair']+=1;continue
            # Source paths define split and pair; synthetic row IDs never define it.
            parts = Path(images[0].replace('\\','/')).parts
            split = next((p for p in parts if p in {'train','val','test'}),'unspecified')
            stem = Path(images[0]).stem
            parent = f'levir_mci/{split}/{stem}'
            if not conversations and 'captions' in r:
                role='evaluation_references'
            elif conversations and all('from' in t and 'value' in t for t in conversations):
                role='instruction'
            else:
                malformed['invalid_conversation']+=1;continue
            rows.append({'instruction_id':f"{dataset['id']}/{file.stem}/{i}",'dataset':dataset['id'],'source_file':file.name,'source_directory':str(file.parent.resolve()),
                         'source_index':i,'official_split':split,'parent_pair_id':parent,'image_family':'levir_cc_mci',
                         'role':role,'turns':len(conversations),'tokens':sum(len(t['value'].split()) for t in conversations),
                         'annotation_split':'test' if '_test' in file.stem else 'train' if '_train' in file.stem else split,
                         'has_source_pair':parent in image_index,'content_sha256':sha256(json.dumps(r,sort_keys=True).encode()),
                         'location_task':bool(re.search(r'loc|position|where|coordinate',json.dumps(conversations).lower()))})
    counts=Counter(r['content_sha256'] for r in rows)
    return rows,{'records_audited':len(rows),'invalid_records':sum(malformed.values()),'malformed':dict(malformed),
                 'unique_parent_pairs':len({r['parent_pair_id'] for r in rows}),
                 'split_counts':dict(Counter(r['official_split'] for r in rows)),
                 'roles':dict(Counter(r['role'] for r in rows)),'missing_parent_rows':sum(not r['has_source_pair'] for r in rows),
                 'duplicate_instruction_rows':sum(v-1 for v in counts.values()),
                 'conversation_turns':dict(Counter(r['turns'] for r in rows)),
                 'annotation_file_rows':dict(Counter(r['source_file'] for r in rows))}


def run_audit(config_path: str|Path|None=None, data_root: str|Path|None=None,
              output: str|Path='outputs/data', datasets: list[str]|None=None,
              max_pairs: int|None=None, workers: int=4) -> dict:
    catalog = load_catalog(config_path)
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    summary, all_records, all_profiles, instructions = {}, [], [], []
    definitions = {d['id']:d for d in catalog['datasets']}
    selected = datasets or list(definitions)
    for did in selected:
        d=definitions[did]
        if d.get('alias_of'):
            summary[did]={'status':'shared_image_alias','alias_of':d['alias_of'],'scope':'not counted as independent imagery'};continue
        source=resolve_source(d,data_root)
        if source is None or d['kind']=='context':
            summary[did]={'status':'not_available_locally' if d['kind']!='context' else 'requires_geographic_query','scope':'not executed','source':d['source']};continue
        if d['kind']=='instructions':
            rows,s = audit_instructions(source,d,{r['pair_id']:r for r in all_records})
            instructions.extend(rows);s.update(status='executed',source_path=str(source),kind=d['kind'],scope='all local annotation files');summary[did]=s
            print(did,s['records_audited'],'instruction records',flush=True);continue
        with Store(source) as store:
            print(did,'indexing source records',flush=True)
            verified_revision=None
            if store.archive and d.get('archive_checksum'):
                algorithm,expected=d['archive_checksum'].split(':',1)
                with store.archive_stream() as f:actual=hashlib.file_digest(f,algorithm).hexdigest()
                if actual!=expected:raise ValueError(f'Publisher archive checksum mismatch: {did}')
                verified_revision=d.get('revision')
            rows,issues=discover_pairs(store,d)
            discovered=len(rows)
            print(did,'discovered',discovered,'pairs;',len(issues),'missing joins',flush=True)
            if max_pairs is not None:
                # Seedless stable content-key ranking avoids filesystem listing bias.
                rows=sorted(rows,key=lambda r:sha256(r['pair_id'].encode()))[:max_pairs]
            if not rows:
                summary[did]={'status':'no_pairs_found','source_path':str(source),'issues':issues};continue
            with ThreadPoolExecutor(max_workers=workers) as pool:
                profiles=[]
                for i,p in enumerate(pool.map(lambda r:profile_pair(store,r),rows),1):
                    profiles.append(p)
                    if i%1000==0:print(did,'profiled',i,'/',len(rows),flush=True)
            all_records.extend([dict(r,source_path=str(source),source_revision=verified_revision or 'local_member_snapshot',expected_release=d.get('revision'),supervised_classes=d.get('supervised_classes',[])) for r in rows])
            all_profiles.extend(profiles)
            local_scope=('convenience Drive probe; not a full-release estimate' if 'drive_samples' in str(source) else '100-pair original-validation transfer probe' if 'second_cc_probe' in str(source) else 'all discovered local records')
            snapshot=sha256(json.dumps([(p['pair_id'],p.get('t1_sha256'),p.get('t2_sha256'),p.get('target_sha256')) for p in profiles],sort_keys=True).encode())
            s=aggregate(profiles);s.update(status='executed',source_path=str(source),kind=d['kind'],source_revision=verified_revision or 'local_member_snapshot',expected_release=d.get('revision'),source_snapshot_sha256=snapshot,
                 records_discovered=discovered,missing_joins=len(issues),scope=local_scope if max_pairs is None else f'stable subset capped at {max_pairs}; discovered {discovered}',
                 licence_disposition=d['licence_disposition'])
            summary[did]=s
            write_json(output/'issues'/f'{did}.json',issues)
            print(did,len(profiles),'audited;',s['invalid_records'],'invalid',flush=True)
    diagnostics=complete_reference_checks(all_records,all_profiles)
    for did,stats in diagnostics.items():
        if stats.get('palette_mismatches'):summary[did].update(aggregate([p for p in all_profiles if p['dataset']==did]))
        summary[did].update(stats)
    write_json(output/'audit_summary.json',summary)
    write_jsonl(output/'source_records.jsonl',all_records)
    write_jsonl(output/'profiles.jsonl',all_profiles)
    write_jsonl(output/'instruction_records.jsonl',instructions)
    return summary
