"""Image pair -> model workers -> claims -> crop evidence -> gated JSON report."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from datetime import datetime,timezone

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from POC.evidence.pipeline import StageCEvidencePipeline


def parse_claims(caption,pair_id):
    spec=importlib.util.spec_from_file_location('tw_claim_parser',ROOT/'POC/execution/scripts/claims.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return [(c,module.to_contract(c)) for c in module.extract_claims(caption,pair_id=pair_id,run_id='pipeline')]


def candidate_box(mask,claim):
    h,w=mask.shape;cls={'road':1,'building':2}.get(claim['claim']['object'])
    foreground=mask==cls if cls else mask>0
    n,_,stats,_=cv2.connectedComponentsWithStats(foreground.astype('uint8'),8)
    location=claim['claim'].get('location');rows=set(location['rows']) if location else set(range(3));cols=set(location['cols']) if location else set(range(3))
    matches=[]
    for x,y,bw,bh,area in stats[1:]:
        r=min(2,int((y+bh/2)*3/h));c=min(2,int((x+bw/2)*3/w))
        if r in rows and c in cols:matches.append((int(area),[int(x),int(y),int(x+bw),int(y+bh)]))
    return max(matches)[1] if matches else [0,0,w,h], bool(matches)


def gated_decision(raw,threshold,forced_reasons=()):
    reasons=list(forced_reasons);outcome=raw.get('outcome','defer');probs=raw.get('probabilities')
    if outcome not in {'accept','reject','defer'}:reasons.append('invalid_decision');outcome='defer'
    if outcome=='accept':
        valid=isinstance(probs,dict) and set(probs)=={'accept','reject','defer'}
        valid=valid and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in probs.values())
        valid=valid and math.isclose(sum(probs.values()),1,abs_tol=.02)
        if not valid:reasons.append('unverified_accept_probability')
        elif probs['accept']<threshold:reasons.append('accept_below_operating_threshold')
    return {'outcome':'defer' if reasons else outcome,'raw_outcome':raw.get('outcome'),
            'probabilities':probs,'reason_codes':reasons,'raw_response':raw.get('raw')}


def model_worker(stage,request,run,config):
    request['spec']=dict(config[stage]);request['device']=config.get('device','cuda');request['seed']=config.get('seed',42)
    for key in ['source_dir','checkpoint','vocab']:
        if request['spec'].get(key):request['spec'][key]=str((ROOT/request['spec'][key]).resolve())
    inp=run/(stage+'_request.json');out=run/(stage+'_response.json');inp.write_text(json.dumps(request,indent=2),encoding='utf-8')
    command=[config[stage]['python'],str(ROOT/'POC/pipeline/workers.py'),stage,str(inp),str(out)]
    with (run/(stage+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=config.get('worker_timeout_seconds',600),check=True)
    return json.loads(out.read_text(encoding='utf-8'))


def run_pipeline(t1,t2,config,output,pair_id='input_pair',worker=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    images=[str(Path(p).resolve()) for p in [t1,t2]];arrays=[np.array(Image.open(p).convert('RGB')) for p in images]
    if arrays[0].shape!=arrays[1].shape:raise ValueError('Paired images must be aligned and have matching dimensions')
    invoke=worker or (lambda stage,req:model_worker(stage,req,output,config))
    result={'pair_id':pair_id,'execution_mode':'test_double' if worker else 'model_workers',
            'input_sha256':[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images],
            'claims':[],'errors':[],'generated_caption':None}
    try:
        detector=invoke('detector',{'pair_id':pair_id,'images':images})
        mask=np.array(Image.open(detector['mask_path']))
        if mask.ndim!=2 or mask.shape!=arrays[0].shape[:2] or not set(np.unique(mask)).issubset({0,1,2}):raise ValueError('Detector mask must contain original-frame 0/1/2 class indices')
        caption=invoke('captioner',{'pair_id':pair_id,'images':images})
        result['generated_caption']=caption['caption'];parsed=parse_claims(caption['caption'],pair_id)
        witness=StageCEvidencePipeline(output_dir=str(output/'crops'),witness_run_id='structural_crop_witness_v3')
        tasks=[];records=[]
        for claim,contract in parsed:
            bbox,localized=candidate_box(mask,claim);content=contract['claim'].copy()
            if not isinstance(content.get('quantity'),int):content['quantity']=None
            content['mask_run_id']=detector.get('model','BIT-MCI-3class')
            safe_id=hashlib.sha256(contract['claim_id'].encode()).hexdigest()[:20]
            rec=witness.process_claim(pair_id,safe_id,contract['source_span'],content,bbox,*arrays).model_dump()
            rec['claim_id']=contract['claim_id'];rec['parser_provenance']=contract.get('b_source');forced=[]
            if rec['evidence']['visible_change'] is None:forced.append('insufficient_visual_evidence')
            if not localized and content.get('action') not in {'unchanged','no_change'}:forced.append('no_matching_predicted_region')
            if content.get('quantity') is not None:forced.append('exact_count_without_independent_instance_evidence')
            if re.search(r'\billegal\b|\bunauthori[sz]ed\b|\bownership\b',contract['source_span'],re.I):forced.append('nonvisual_legal_or_ownership_claim')
            rec['forced_review_reasons']=forced;records.append(rec)
            tasks.append({'claim_id':rec['claim_id'],'state':{'claim':rec['source_span'],'structured_claim':rec['claim'],
                          'predicted_mask_summary':{'source':detector.get('model'),'visible_change':bool(mask.any()),'region_localized':localized},
                          'fallible_witness_observation':rec['evidence'],'temporal_order':'T1_earlier_T2_later'}})
        try:decisions=invoke('referee',{'pair_id':pair_id,'images':images,'cases':tasks}) if tasks else {'decisions':[]}
        except Exception as error:result['errors'].append({'stage':'referee','error':str(error)});decisions={'decisions':[]}
        by_id={r['claim_id']:r for r in decisions['decisions']}
        for rec in records:
            raw=by_id.get(rec['claim_id'],{});reasons=rec.pop('forced_review_reasons')
            if not raw:reasons.append('referee_output_unavailable')
            rec['decision']=gated_decision(raw,config['referee']['accept_threshold'],reasons);result['claims'].append(rec)
    except Exception as error:result['errors'].append({'stage':'upstream','error':str(error)})
    accepted=list(dict.fromkeys(r['source_span'] for r in result['claims'] if r['decision']['outcome']=='accept'))
    result['verified_description']='; '.join(accepted) if accepted else 'The image pair requires review before a change description can be confirmed.'
    result['review_required']=bool(result['errors']) or not result['claims'] or any(r['decision']['outcome']=='defer' for r in result['claims'])
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--t1',required=True);ap.add_argument('--t2',required=True);ap.add_argument('--config',default=str(ROOT/'POC/pipeline/pipeline.json'));ap.add_argument('--output');ap.add_argument('--pair-id',default='input_pair');a=ap.parse_args()
    config=json.loads(Path(a.config).read_text());out=a.output or str(ROOT/'POC/pipeline/runs'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    run_pipeline(a.t1,a.t2,config,out,a.pair_id);print(str(Path(out)/'report.json'))


if __name__=='__main__':main()
