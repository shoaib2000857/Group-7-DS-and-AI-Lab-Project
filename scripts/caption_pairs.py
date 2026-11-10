"""Batch caption prepared pairs, loading RSCaMa once and keeping references out of inference."""
from pathlib import Path
import argparse,json,sys,time
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',default='data/processed')
    parser.add_argument('--config',default='configs/pipeline.json')
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--dataset',default='levir_mci')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from PIL import Image
    from src.data.dataset import PairDataset
    from src.models.captioner import RSCaMaCaptioner
    config=json.loads(Path(args.config).read_text());model=RSCaMaCaptioner(config['captioner'],config['device'])
    root=Path(args.processed);dataset=PairDataset(root/(args.split+'.jsonl'),root,include_targets=False)
    dataset.rows=[row for row in dataset.rows if row['dataset']==args.dataset]
    if args.limit is not None:
        if args.limit<=0:parser.error('--limit must be positive')
        dataset.rows=dataset.rows[:args.limit]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with output.open('x',encoding='utf-8') as stream,tempfile.TemporaryDirectory() as temporary:
            for index in range(len(dataset)):
                sample=dataset[index];paths=[Path(temporary)/f'{key}.png' for key in ['t1','t2']]
                for key,path in zip(['t1','t2'],paths):Image.fromarray(sample[key]).save(path)
                start=time.perf_counter();caption=model.describe(paths)
                stream.write(json.dumps({'pair_id':sample['pair_id'],'raw_text':caption,'status':'ok',
                    'run_id':output.stem,'latency_s':time.perf_counter()-start})+'\n');stream.flush()
    finally:dataset.close()

if __name__=='__main__':main()
