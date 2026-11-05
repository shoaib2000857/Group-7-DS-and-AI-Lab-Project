"""Train the three-class BIT design on prepared manifests; preserve all checkpoints."""
from pathlib import Path
import argparse
import json
import math
import sys
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import torch
    from src.data.dataset import PairDataset
    from POC.pipeline.workers import build_detector
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',required=True);ap.add_argument('--processed',default='data/processed');ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--accum',type=int,default=4);ap.add_argument('--patience',type=int,default=8);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--device',default='cuda');ap.add_argument('--output');a=ap.parse_args()
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    output=Path(a.output or ROOT/'outputs/detector_training'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'));output.mkdir(parents=True,exist_ok=False)
    root=Path(a.processed);train=PairDataset(root/'train.jsonl',root);val=PairDataset(root/'val.jsonl',root)
    # Main target is MCI's road/building ontology, not unrelated binary targets.
    train.rows=[r for r in train.rows if r['dataset']=='levir_mci'];val.rows=[r for r in val.rows if r['dataset']=='levir_mci']
    if not train.rows or not val.rows:raise ValueError('MCI train and validation records are required')
    assert not {r['pair_id'] for r in train.rows}&{r['pair_id'] for r in val.rows}
    model=build_detector({'source_dir':a.source_dir},a.device)
    backbone=list(model.resnet.parameters());ids={id(p) for p in backbone}
    other=[p for p in model.parameters() if id(p) not in ids]
    opt=torch.optim.AdamW([{'params':backbone,'lr':1e-5},{'params':other,'lr':1e-4}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    weights=torch.tensor([.1,.45,.45],device=a.device)
    def loss_fn(logits,target):
        ce=torch.nn.functional.cross_entropy(logits,target,weight=weights,reduction='none')
        pt=logits.softmax(1).gather(1,target[:,None]).squeeze(1)
        focal=((1-pt)**2*ce).mean();prob=logits.softmax(1)
        truth=torch.nn.functional.one_hot(target,3).permute(0,3,1,2).float()
        dims=(0,2,3);dice=1-((2*(prob*truth).sum(dims)+1)/(prob.sum(dims)+truth.sum(dims)+1)).mean()
        return .5*focal+.5*dice
    best=-1.;stale=0
    config=vars(a)|{'classes':['background','road','building'],'loss':'0.5 focal(gamma=2,alpha=[.1,.45,.45])+0.5 soft Dice','normalisation':'RGB /127.5 -1','train_pairs':len(train),'val_pairs':len(val)}
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    try:
        for epoch in range(a.epochs):
            freeze=epoch<5
            for p in backbone:p.requires_grad=not freeze
            model.train()
            if freeze:model.resnet.eval()
            order=np.random.default_rng(a.seed+epoch).permutation(len(train));steps=math.ceil(len(order)/a.batch);opt.zero_grad();total=0.
            for step,start in enumerate(range(0,len(order),a.batch)):
                batch=train.batch(order[start:start+a.batch]);x=torch.from_numpy(batch['images']).to(a.device)*2-1;y=torch.from_numpy(batch['targets']).long().to(a.device)
                if np.random.random()<.5:x=x.flip(-1);y=y.flip(-1)
                divisor=min(a.accum,steps-(step//a.accum)*a.accum)
                loss=loss_fn(model(x[:,0],x[:,1]),y);(loss/divisor).backward();total+=float(loss.detach())
                if (step+1)%a.accum==0 or step+1==steps:opt.step();opt.zero_grad()
            model.eval();cm=torch.zeros((3,3),dtype=torch.int64,device=a.device)
            with torch.inference_mode():
                for start in range(0,len(val),a.batch):
                    b=val.batch(range(start,min(start+a.batch,len(val))));x=torch.from_numpy(b['images']).to(a.device)*2-1;y=torch.from_numpy(b['targets']).long().to(a.device)
                    pred=model(x[:,0],x[:,1]).argmax(1);cm+=torch.bincount((y*3+pred).reshape(-1),minlength=9).reshape(3,3)
            tp=cm.diag().float();union=cm.sum(0)+cm.sum(1)-tp;iou=tp/union.clamp_min(1);score=float(iou[union>0].mean());scheduler.step()
            row={'epoch':epoch,'train_loss':total/steps,'val_macro_iou':score,'val_class_iou':iou.tolist(),'backbone_frozen':freeze}
            with (output/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True)
            if score>best:
                best=score;stale=0;target=output/f'best_epoch_{epoch:03d}.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':config,'val_macro_iou':score},target)
                (output/'best.json').write_text(json.dumps({'checkpoint':str(target.resolve()),'val_macro_iou':score}),encoding='utf-8')
            else:stale+=1
            if stale>=a.patience:break
    finally:train.close();val.close()


if __name__=='__main__':main()
