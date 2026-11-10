import json
import numpy as np
import pytest
from src.evaluation.segmentation import confusion_matrix,metrics_from_confusion
from src.models.claims import extract_claims
from src.evaluation.claim_metrics import check_claim
from src.inference.preflight import inspect_config
from scripts.package_model_run import package_run


def test_binary_change_counts_semantic_class_confusion_as_change():
    truth=np.array([[0,1],[2,2]],dtype='uint8')
    predicted=np.array([[0,2],[1,0]],dtype='uint8')
    result=metrics_from_confusion(confusion_matrix(truth,predicted))
    assert result['binary_f1']==pytest.approx(.8)
    assert result['binary_iou']==pytest.approx(2/3)
    assert result['change_class_mean_iou']==0


def test_empty_change_population_is_undefined_not_perfect():
    result=metrics_from_confusion(np.array([[16,0],[0,0]]))
    assert result['binary_f1'] is None
    assert result['change_class_mean_iou'] is None


def test_masks_reject_bad_dimensions_and_unknown_classes():
    with pytest.raises(ValueError):confusion_matrix(np.zeros((2,2),dtype='uint8'),np.zeros((4,4),dtype='uint8'))
    with pytest.raises(ValueError):confusion_matrix(np.zeros((2,2),dtype='uint8'),np.full((2,2),255,dtype='uint8'))


def test_reference_proxy_rejects_change_on_unchanged_target():
    class Truth:
        in_scope={'road':1,'building':2}
        ref_claims={'p':extract_claims('there is no difference.')}
        refs={'p':{'changeflag':0}}
        def class_changed(self,*args):return False
    claim=extract_claims('buildings appeared.')[0]
    assert check_claim(claim,'p',Truth())['label']=='contradicted'


def test_preflight_does_not_accept_missing_assets(tmp_path):
    spec={'detector':{'python':'python','source_dir':'missing','checkpoint':'missing.pt'},
          'captioner':{'python':'python','source_dir':'missing','checkpoint':'missing.pth'},
          'referee':{'python':'python','revision':'abc','accept_threshold':.8}}
    assert not inspect_config(spec,tmp_path)['ready']


def test_export_preserves_files_and_detectable_hashes(tmp_path):
    run=tmp_path/'run';run.mkdir();(run/'report.json').write_text('{"value":1}')
    archive=package_run(run,tmp_path/'run.zip')
    import zipfile,hashlib
    with zipfile.ZipFile(archive) as exported:
        receipt=json.loads(exported.read('_export_manifest.json'))
        assert receipt['files'][0]['sha256']==hashlib.sha256(exported.read('report.json')).hexdigest()
    with pytest.raises(ValueError):package_run(run,run/'recursive.zip')
    with pytest.raises(FileExistsError):package_run(run,archive)
