"""CPU integration control using image differences, not learned model predictions."""
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def control_worker(output):
    output = Path(output)
    def invoke(stage, request):
        if stage == 'detector':
            arrays = [np.array(Image.open(p).convert('RGB')).astype('float32') for p in request['images']]
            difference = np.abs(arrays[1]-arrays[0]).mean(-1).astype('uint8')
            _, mask = cv2.threshold(difference, 0, 2, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            path = output/'control_mask.png'; Image.fromarray(mask).save(path)
            return {'mask_path': str(path.resolve()), 'model': 'Otsu-control-assumed-building-class', 'source_kind': 'deterministic_control'}
        if stage == 'captioner':
            changed = np.any(np.array(Image.open(output/'control_mask.png')))
            return {'caption': 'buildings changed.' if changed else 'there is no difference.', 'model': 'template-control'}
        # No model or human judgement is imputed to a deterministic routing control.
        return {'decisions': [{'claim_id': case['claim_id'], 'outcome': 'defer',
                'probability_semantics': 'not_applicable_control', 'raw': 'integration control; no learned referee'} for case in request['cases']]}
    invoke.execution_mode = "deterministic_integration_control"
    return invoke


def make_fixture(folder, seed=7):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    first = np.random.default_rng(seed).integers(50, 130, (96, 96, 3), dtype='uint8')
    second = first.copy(); second[45:78, 40:76] = 220
    paths = []
    for name, array in [('t1', first), ('t2', second)]:
        path = folder/(name+'.png'); Image.fromarray(array).save(path); paths.append(path)
    return paths
