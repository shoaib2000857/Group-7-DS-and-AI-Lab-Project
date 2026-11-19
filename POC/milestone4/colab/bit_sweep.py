"""Bounded BIT learning-rate comparison on the strict M2 data contract."""
import hashlib
import io
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_pairs(archive, rows):
    pairs=[]
    for row in rows:
        decoded=[]
        for key in ('t1','t2','target'):
            payload=archive.read(row[key+'_member'])
            if hashlib.sha256(payload).hexdigest()!=row[key+'_sha256']:
                raise ValueError(f"Archive member hash mismatch: {row[key+'_member']}")
            with Image.open(io.BytesIO(payload)) as image:
                decoded.append(np.array(image.convert('RGB')))
        raw=decoded[2][...,0]
        if not set(np.unique(raw)).issubset({0,128,255}):raise ValueError('Unexpected source mask encoding')
        pairs.append((np.stack(decoded[:2]),((raw==128)+2*(raw==255)).astype('uint8')))
    return pairs


def loss_fn(logits, target, weights):
    import torch
    ce = torch.nn.functional.cross_entropy(logits, target, weight=weights, reduction="none")
    pt = logits.softmax(1).gather(1, target[:, None]).squeeze(1)
    focal = ((1 - pt) ** 2 * ce).mean()
    probs = logits.softmax(1)
    truth = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = 1 - ((2 * (probs * truth).sum(dims) + 1) /
                (probs.sum(dims) + truth.sum(dims) + 1)).mean()
    return 0.5 * focal + 0.5 * dice

def run(archive_path, repo, bit_source, output, minutes=36, *, device="cuda", train_count=192, val_count=64, epochs=4):
    import torch
    import torchvision
    from src.models.workers import build_detector
    from src.evaluation.segmentation import metrics_from_confusion
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Select a GPU runtime before starting')
    if device=='cpu':torch.set_num_threads(4)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    manifest_dir=Path(repo)/'POC/milestone2/evidence/full_audit'
    cohorts={}
    for split,count in [('train',train_count),('val',val_count)]:
        source=[json.loads(line) for line in (manifest_dir/f'{split}_manifest.jsonl').read_text().splitlines()]
        cohorts[split]=random.Random(42+(split=='val')).sample(source,count)
    (output/'cohort.json').write_text(json.dumps(cohorts,indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        data={split:load_pairs(archive,rows) for split,rows in cohorts.items()}
    # One set of initial weights, one data order and one augmentation sequence for both arms.
    torch.manual_seed(42)
    # The pinned BIT source imports a torchvision path removed in recent releases.
    # Redirect only the downloader import to its supported PyTorch implementation.
    import sys
    from types import ModuleType
    legacy_utils=ModuleType('torchvision.models.utils')
    legacy_utils.load_state_dict_from_url=torch.hub.load_state_dict_from_url
    sys.modules.setdefault(legacy_utils.__name__,legacy_utils)
    model=build_detector({'source_dir':str(bit_source)},device)
    initial={key:tensor.detach().cpu().clone() for key,tensor in model.state_dict().items()}
    for parameter in model.resnet.parameters():parameter.requires_grad_(False)
    weights=torch.tensor([.1,.45,.45],device=device)
    batch_size=4
    (output/'run.json').write_text(json.dumps({'seed':42,'batch_size':batch_size,'epochs_cap':epochs,
        'learning_rates':[1e-4,3e-4],'optimizer':'AdamW','weight_decay':.01,'scheduler':'none',
        'criterion':'validation change-class mean IoU','loss':'equal weighted focal and soft Dice',
        'augmentation':'paired horizontal flips','backbone':'ImageNet ResNet18, frozen',
        'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'device':device,
        'train_pairs':train_count,'validation_pairs':val_count,
        'vram_bytes':torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None},indent=2))

    def batches(split,indices,epoch):
        flip=random.Random(500+epoch)
        for start in range(0,len(indices),batch_size):
            pairs=[data[split][index] for index in indices[start:start+batch_size]]
            images=np.stack([pair[0] for pair in pairs]);masks=np.stack([pair[1] for pair in pairs])
            if split=='train' and flip.random()<.5:
                images=images[:,:,:,::-1,:].copy();masks=masks[:,:,::-1].copy()
            yield torch.from_numpy(images).permute(0,1,4,2,3).to(device).float()/127.5-1, torch.from_numpy(masks).to(device).long()

    best={};histories={}
    for arm,lr in [('lr_1e-4',1e-4),('lr_3e-4',3e-4)]:
        deadline=time.monotonic()+minutes*30
        model.load_state_dict(initial);torch.manual_seed(42)
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=.01)
        history=[];histories[arm]=history;best_score=-1
        for epoch in range(1,epochs+1):
            model.train();model.resnet.eval();train_loss=[];finished=True
            indices=np.random.default_rng(42+epoch).permutation(len(data['train']))
            for images,target in batches('train',indices,epoch):
                if time.monotonic()>=deadline:finished=False;break
                optimizer.zero_grad(set_to_none=True)
                loss=loss_fn(model(*images.unbind(1)),target,weights)
                loss.backward();optimizer.step();train_loss.append(float(loss.detach()))
            if not finished:break
            model.eval();val_loss=[];confusion=np.zeros((3,3),dtype='int64')
            with torch.inference_mode():
                for images,target in batches('val',list(range(len(data['val']))),epoch):
                    logits=model(*images.unbind(1));val_loss.append(float(loss_fn(logits,target,weights)))
                    confusion+=torch.bincount((target*3+logits.argmax(1)).flatten(),minlength=9).reshape(3,3).cpu().numpy()
            metrics=metrics_from_confusion(confusion)
            row={'epoch':epoch,'train_loss':float(np.mean(train_loss)),'validation_loss':float(np.mean(val_loss)),**metrics}
            history.append(row);(output/f'{arm}_history.json').write_text(json.dumps(history,indent=2));print(arm,row,flush=True)
            score=row['change_class_mean_iou']
            if score is not None and score>best_score:
                best_score=score;best[arm]=row
                checkpoint=output/f'{arm}.pt'
                torch.save({'model':model.state_dict(),'epoch':epoch,'learning_rate':lr,'validation':row},checkpoint)
                best[arm]={**row,'checkpoint':checkpoint.name}
        del optimizer
    common_epochs=min(map(len,histories.values()))
    # Compare at the last epoch completed by both arms. Checkpoint selection
    # is restricted to matched schedules, otherwise report both arms separately.
    selected=None
    if common_epochs and len({len(rows) for rows in histories.values()})==1 and len(best)==2:
        arm=max(best,key=lambda key:best[key]['change_class_mean_iou']);selected={'arm':arm,**best[arm]}
    hashes={}
    for path in output.glob('*.pt'):
        with path.open('rb') as stream:hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
    result={'best_by_arm':best,'common_epochs':common_epochs,'selected':selected,'checkpoint_sha256':hashes}
    (output/'selection.json').write_text(json.dumps(result,indent=2))
    return result
