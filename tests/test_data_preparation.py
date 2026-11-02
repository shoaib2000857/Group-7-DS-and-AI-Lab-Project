"""Meaningful adapter, split-leakage and model-input contract checks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.data.adapters import scalar_mask,binary_mask,semantic_mask,directional_mask,decode_targets,discover_pairs
from src.data.storage import Store,pixel_hash,SplitZipReader
from src.data.audit import image_fingerprints,audit_instructions,run_audit
from src.data.catalog import resolve_source
from src.data.preprocess import exclusion_policy,prepare,unique_pairs
from src.data.dataset import paired_augmentation,balanced_training_indices,PairDataset


def test_replicated_rgb_target_and_unknown_colours():
    mask=np.array([[0,128,255]],dtype=np.uint8)
    assert np.array_equal(scalar_mask(np.repeat(mask[...,None],3,axis=2)),mask)
    with pytest.raises(ValueError):scalar_mask(np.array([[[255,0,0]]],dtype=np.uint8))
    with pytest.raises(ValueError):binary_mask(mask)


def test_s2looking_blue_red_and_unknown_palette():
    blue=np.array([[[0,0,0],[0,0,255]]],dtype=np.uint8)
    red=np.array([[[255,0,0],[0,0,0]]],dtype=np.uint8)
    assert directional_mask(blue,(0,0,255)).tolist()==[[0,1]]
    assert directional_mask(red,(255,0,0)).tolist()==[[1,0]]
    with pytest.raises(ValueError):directional_mask(red,(0,0,255))


def test_semantic_white_is_unchanged_and_red_is_not_binary_building():
    colors=np.array([[[255,255,255],[128,0,0],[255,0,0],[0,0,255]]],dtype=np.uint8)
    assert semantic_mask(colors).tolist()==[[0,5,6,1]]
    with pytest.raises(ValueError):semantic_mask(np.array([[[12,17,8]]],dtype=np.uint8))


def test_semantic_discovery_does_not_treat_labels_as_rgb(tmp_path):
    for sub in ['rgb/A','rgb/B','sem/A','sem/B']:
        p=tmp_path/'val'/sub/'x.png';p.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.zeros((3,3,3),dtype=np.uint8)).save(p)
    with Store(tmp_path) as store:
        rows,issues=discover_pairs(store,{'id':'second_cc','kind':'semantic'})
    assert len(rows)==1 and not issues and '/rgb/A/' in rows[0]['t1']


def test_missing_join_is_explicit(tmp_path):
    p=tmp_path/'train/A/x.png';p.parent.mkdir(parents=True);Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(p)
    with Store(tmp_path) as store:rows,issues=discover_pairs(store,{'id':'levir_cd','kind':'binary'})
    assert not rows and len(issues)==1


def test_storage_prevents_parent_escape(tmp_path):
    with Store(tmp_path) as store:
        with pytest.raises(ValueError):store.read('../outside.png')


def fixture_row(pid,split,dataset='d',target='target',fingerprints=None):
    row={'pair_id':pid,'dataset':dataset,'kind':'mci','official_split':split,'captions':['a']*5}
    profile={'pair_id':pid,'valid':True,'t1_pixel_sha256':'a','t2_pixel_sha256':'b','target_pixel_sha256':target,
             'empty_captions':0,'fingerprints':fingerprints or ['a','b'],'mask_caption_conflict':False}
    return row,profile


def test_test_priority_and_inconsistent_targets_quarantine():
    r1,p1=fixture_row('d/train/x','train');r2,p2=fixture_row('d/test/y','test')
    reasons,groups=exclusion_policy([r1,r2],[p1,p2],[])
    assert 'd/train/x' in reasons and 'd/test/y' not in reasons
    p2['target_pixel_sha256']='different'
    reasons,_=exclusion_policy([r1,r2],[p1,p2],[])
    assert all('inconsistent_duplicate_targets' in reasons[r['pair_id']] for r in [r1,r2])


def test_shared_tile_protects_partitions_across_datasets():
    r1,p1=fixture_row('a/train/x','train','a',fingerprints=['tile'])
    r2,p2=fixture_row('b/test/y','test','b',fingerprints=['tile'])
    reasons,_=exclusion_policy([r1,r2],[p1,p2],[])
    assert 'exact_frame_or_tile_cross_split' in reasons[r1['pair_id']]


def test_aligned_tile_hash_detects_native_and_cropped_reuse():
    a=np.random.default_rng(1).integers(0,255,(512,512,3),dtype=np.uint8)
    assert pixel_hash(a[:256,:256]) in image_fingerprints(a)


def test_instruction_benchmark_cannot_reenter_image_training():
    r,p=fixture_row('levir_mci/train/x','train')
    reasons,_=exclusion_policy([r],[p],[{'parent_pair_id':r['pair_id'],'annotation_split':'test'}])
    assert 'instruction_benchmark_parent_holdout' in reasons[r['pair_id']]


def test_paired_augmentation_and_caption_guard():
    a=np.arange(18,dtype=np.uint8).reshape(2,3,3);mask=np.arange(6).reshape(2,3)
    x,y,m=paired_augmentation(a,a,mask,'flip_horizontal')
    assert np.array_equal(x,y) and np.array_equal(m,mask[:,::-1])
    with pytest.raises(ValueError):paired_augmentation(a,a,mask,'rotate_90',has_caption=True)


def test_source_sampler_refuses_evaluation_and_absent_sources():
    rows=[{'dataset':'primary','official_split':'train'},{'dataset':'extra','official_split':'train'}]
    assert set(balanced_training_indices(rows,20,seed=1)).issubset({0,1})
    with pytest.raises(ValueError):balanced_training_indices(rows+[{'dataset':'x','official_split':'test'}],2)
    with pytest.raises(ValueError):balanced_training_indices(rows,2,{'unavailable_source':1})


def test_split_zip_seeks_across_part_boundaries(tmp_path):
    import io,zipfile,hashlib
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('train/A/value.txt',b'original archive bytes'*70)
    data=stream.getvalue();parts=[]
    for i,start in enumerate(range(0,len(data),31)):
        b=data[start:start+31];name=f'part{i:03}';(tmp_path/name).write_bytes(b);parts.append({'name':name,'size':len(b)})
    manifest=tmp_path/'example.zip.parts.json';manifest.write_text(json.dumps({'format':'terrawatch-split-zip-v1','original_size':len(data),'parts':parts}))
    with SplitZipReader(manifest) as f:
        f.seek(25);assert f.read(60)==data[25:85]
        f.seek(-20,2);assert f.read()==data[-20:]
    with Store(manifest) as s:assert s.read('train/A/value.txt')==b'original archive bytes'*70


def test_clean_copy_survives_when_preferred_copy_is_flagged():
    r1,p1=fixture_row('d/test/x','test');r2,p2=fixture_row('d/train/y','train')
    p1['mask_caption_conflict']=True
    reasons,_=exclusion_policy([r1,r2],[p1,p2],[])
    assert 'd/test/x' in reasons and 'd/train/y' not in reasons


def test_excluded_record_cannot_displace_shared_tile():
    r1,p1=fixture_row('a/test/x','test','a',fingerprints=['tile']);r2,p2=fixture_row('b/train/y','train','b',fingerprints=['tile'])
    p1['t1_pixel_sha256'],p1['mask_caption_conflict']='other',True
    reasons,_=exclusion_policy([r1,r2],[p1,p2],[])
    assert 'b/train/y' not in reasons


def test_task_keeps_one_copy_across_related_datasets():
    rows=[{'pair_id':'b/test/y','dataset':'b','pixel_pair_signature':'s'},{'pair_id':'a/test/x','dataset':'a','pixel_pair_signature':'s'},
          {'pair_id':'b/test/z','dataset':'b','pixel_pair_signature':'t'}]
    assert [r['pair_id'] for r in unique_pairs(rows,['a','b'])]==['a/test/x','b/test/z']


def test_ineligible_source_stays_out_of_training_but_not_evaluation(tmp_path):
    for split in ['train','val']:
        for sub in ['A','B']:
            p=tmp_path/'src'/split/sub/'x.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(p)
    rows=[];profiles=[]
    for split,value in [('train','1'),('val','2')]:
        pid=f'whu/{split}/x'
        rows.append({'pair_id':pid,'dataset':'whu','kind':'binary','official_split':split,'captions':[],'source_changeflag':None,'labels':[],
                     'source_path':str(tmp_path/'src'),'t1':f'{split}/A/x.png','t2':f'{split}/B/x.png'})
        profiles.append({'pair_id':pid,'valid':True,'t1_pixel_sha256':value+'a','t2_pixel_sha256':value+'b','t1_sha256':'a','t2_sha256':'b','target_pixel_sha256':None,
                         'empty_captions':0,'fingerprints':[value],'mask_caption_conflict':False,'width':4,'height':4})
    audit=tmp_path/'audit';audit.mkdir()
    (audit/'source_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (audit/'profiles.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in profiles))
    (audit/'instruction_records.jsonl').write_text('')
    config=tmp_path/'c.yaml';config.write_text('datasets:\n- id: whu\n  training_eligible: false\ntraining_tasks: {}\n')
    result=prepare(audit,tmp_path/'out',config)
    assert result['by_dataset_split']=={'whu/val':1}
    assert json.loads((tmp_path/'out/exclusions.json').read_text())=={'whu/train/x':['not_training_eligible']}


def test_windows_instruction_paths_keep_their_parent(tmp_path):
    p=tmp_path/'changechat_train.json'
    p.write_text(json.dumps([{'image':['train\\\\A\\\\train_1.png','train\\\\B\\\\train_1.png'],'conversations':[{'from':'human','value':'q'}]}]))
    rows,_=audit_instructions(p,{'id':'changechat'},{'levir_mci/train/train_1':{}})
    assert rows[0]['parent_pair_id']=='levir_mci/train/train_1' and rows[0]['has_source_pair']


def test_subset_audit_keeps_other_datasets_and_runs_instructions_last(tmp_path):
    out=tmp_path/'out';out.mkdir()
    keep={'pair_id':'levir_mci/train/train_1','dataset':'levir_mci','kind':'binary','captions':[]}
    (out/'audit_summary.json').write_text(json.dumps({'levir_mci':{'status':'executed'}}))
    (out/'source_records.jsonl').write_text(json.dumps(keep)+'\n')
    (out/'profiles.jsonl').write_text(json.dumps({'pair_id':keep['pair_id'],'dataset':'levir_mci','valid':False})+'\n')
    (out/'instruction_records.jsonl').write_text('')
    src=tmp_path/'raw/changechat';src.mkdir(parents=True)
    (src/'a_train.json').write_text(json.dumps([{'image':['train/A/train_1.png','train/B/train_1.png'],'conversations':[{'from':'human','value':'q'}]}]))
    config=tmp_path/'c.yaml'
    config.write_text('datasets:\n- {id: levir_mci, kind: binary, paths: [none]}\n- {id: changechat, kind: instructions, paths: [changechat], source: x}\n')
    summary=run_audit(config,tmp_path/'raw',out,['changechat'])
    assert summary['changechat']['missing_parent_rows']==0 and 'levir_mci' in summary
    assert (out/'source_records.jsonl').read_text().count('train_1')==1


def test_empty_folder_does_not_hide_later_source(tmp_path,monkeypatch):
    monkeypatch.delenv('TERRAWATCH_X',raising=False)
    (tmp_path/'empty').mkdir();(tmp_path/'full').mkdir();(tmp_path/'full/x.png').write_bytes(b'x')
    assert resolve_source({'id':'x','paths':['empty','full']},tmp_path)==(tmp_path/'full').resolve()


def test_pair_dataset_reopens_stores_outside_the_owning_process(tmp_path):
    import pickle
    (tmp_path/'processed').mkdir();manifest=tmp_path/'m.jsonl';manifest.write_text('')
    ds=PairDataset(manifest,tmp_path/'processed');ds.stores={'x':object()}
    assert pickle.loads(pickle.dumps(ds.__getstate__()))['stores']=={}
