"""Meaningful adapter, split-leakage and model-input contract checks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.data.adapters import scalar_mask,binary_mask,semantic_mask,directional_mask,decode_targets,discover_pairs
from src.data.storage import Store,pixel_hash,SplitZipReader
from src.data.audit import image_fingerprints
from src.data.preprocess import exclusion_policy
from src.data.dataset import paired_augmentation,balanced_training_indices


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
