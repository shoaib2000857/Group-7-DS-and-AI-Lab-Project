"""Command-line entry points shared by the notebooks and scheduled runs.

python -m src.data.cli audit --output outputs/data
python -m src.data.cli prepare --audit-dir outputs/data --output data/processed
python -m src.data.cli verify --audit-dir outputs/data --processed data/processed
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .audit import run_audit
from .catalog import write_json
from .preprocess import prepare,read_jsonl
from .dataset import PairDataset


def verify(processed='data/processed',audit_dir='outputs/data',limit=None):
    root=Path(processed);seen=set();results={};hashes={}
    for split in ['train','val','test']:
        ds=PairDataset(root/(split+'.jsonl'),root,include_references=True)
        ids={r['pair_id'] for r in ds.rows};assert not seen.intersection(ids);seen|=ids
        n=min(limit,len(ds)) if limit is not None else len(ds)
        for i in range(n):
            sample=ds[i];assert sample['t1'].shape==sample['t2'].shape
            assert all(v.shape==sample['t1'].shape[:2] for v in sample['targets'].values())
        if len(ds):
            batch=ds.batch(range(min(4,len(ds))));assert batch['images'].dtype==np.float32 and 0<=batch['images'].min()<=batch['images'].max()<=1
            infer=PairDataset(root/(split+'_inference.jsonl'),root,include_targets=False)
            assert set(infer[0])=={'pair_id','t1','t2'};infer.close()
        results[split]={'records':len(ds),'records_verified':n,'scope':'all retained records' if n==len(ds) else 'bounded verification sample'};ds.close()
    result={'splits':results,'reference_separation':'passed'}
    write_json(Path(audit_dir)/'loader_validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--config');a.add_argument('--data-root');a.add_argument('--output',default='outputs/data');a.add_argument('--datasets',nargs='+');a.add_argument('--max-pairs',type=int);a.add_argument('--workers',type=int,default=4)
    a=sub.add_parser('prepare');a.add_argument('--config');a.add_argument('--audit-dir',default='outputs/data');a.add_argument('--output',default='data/processed');a.add_argument('--image-size',type=int,default=256)
    a=sub.add_parser('verify');a.add_argument('--audit-dir',default='outputs/data');a.add_argument('--processed',default='data/processed');a.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.command=='audit':result=run_audit(args.config,args.data_root,args.output,args.datasets,args.max_pairs,args.workers)
    elif args.command=='prepare':result=prepare(args.audit_dir,args.output,args.config,args.image_size or None)
    else:result=verify(args.processed,args.audit_dir,args.limit)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
