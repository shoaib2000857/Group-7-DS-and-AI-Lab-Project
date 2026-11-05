"""CPU contract tests; test doubles are never presented as model inference."""
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from POC.pipeline.run_pipeline import candidate_box,gated_decision,run_pipeline


def test_localization_preserves_original_coordinates():
    m=np.zeros((90,120),dtype='uint8');m[60:80,80:110]=2
    claim={'claim':{'object':'building','location':{'rows':[2],'cols':[2]}}}
    assert candidate_box(m,claim)==([80,60,110,80],True)
    claim['claim']['location']={'rows':[0],'cols':[0]}
    assert candidate_box(m,claim)==([0,0,120,90],False)


def test_acceptance_gate_rejects_unverified_confidence():
    assert gated_decision({'outcome':'accept'},.8)['outcome']=='defer'
    assert gated_decision({'outcome':'accept','probabilities':{'accept':.6,'reject':.2,'defer':.2}},.8)['outcome']=='defer'
    assert gated_decision({'outcome':'accept','probabilities':{'accept':float('nan'),'reject':0,'defer':0}},.8)['outcome']=='defer'
    assert gated_decision({'outcome':'accept','probabilities':{'accept':.9,'reject':.05,'defer':.05}},.8)['outcome']=='accept'


def test_full_contract_route_and_existing_run_preservation(tmp_path):
    rng=np.random.default_rng(7);a=rng.integers(40,160,(64,64,3),dtype='uint8');b=a.copy();b[20:45,20:45]=230
    for key,im in [('t1',a),('t2',b)]:Image.fromarray(im).save(tmp_path/(key+'.png'))
    m=np.zeros((64,64),dtype='uint8');m[20:45,20:45]=2;Image.fromarray(m).save(tmp_path/'mask.png')
    seen=[]
    def worker(stage,request):
        seen.append(stage);assert not any(k in request for k in ['references','ground_truth','human_reference','captions'])
        if stage=='detector':return {'mask_path':str(tmp_path/'mask.png'),'model':'test_double'}
        if stage=='captioner':return {'caption':'buildings appeared in the center.'}
        return {'decisions':[{'claim_id':c['claim_id'],'outcome':'accept','probabilities':{'accept':.95,'reject':.02,'defer':.03}} for c in request['cases']]}
    out=tmp_path/'run';r=run_pipeline(tmp_path/'t1.png',tmp_path/'t2.png',{'referee':{'accept_threshold':.8}},out,worker=worker)
    assert seen==['detector','captioner','referee'];assert r['execution_mode']=='test_double';assert not r['errors'];assert r['claims'];assert (out/'report.json').is_file()
    with pytest.raises(FileExistsError):run_pipeline(tmp_path/'t1.png',tmp_path/'t2.png',{'referee':{'accept_threshold':.8}},out,worker=worker)


def test_model_failure_produces_review_record(tmp_path):
    im=np.full((32,32,3),100,dtype='uint8');Image.fromarray(im).save(tmp_path/'a.png')
    def failed(*args):raise RuntimeError('model unavailable')
    r=run_pipeline(tmp_path/'a.png',tmp_path/'a.png',{'referee':{'accept_threshold':.8}},tmp_path/'run',worker=failed)
    assert r['review_required'] and r['errors'] and not r['claims']
