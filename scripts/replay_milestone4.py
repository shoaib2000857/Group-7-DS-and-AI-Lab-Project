"""Create a replayable CPU review demo using the archived LEVIR image pairs."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.controls import control_worker
from src.inference.pipeline import model_worker, run_pipeline


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='Use a trained three-class BIT checkpoint for the mask stage')
    parser.add_argument('--bit-source',type=Path,default=ROOT/'models/external/BIT_CD')
    parser.add_argument('--detector-python',default=sys.executable)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    root=args.output or ROOT/'outputs/pipeline_runs'/datetime.now(timezone.utc).strftime('m4_review_%Y%m%d_%H%M%S_%f')
    config=json.loads((ROOT/'configs/pipeline.json').read_text())
    if args.checkpoint:
        config['device']=args.device
        config['detector'].update(checkpoint=str(args.checkpoint.resolve()),source_dir=str(args.bit_source.resolve()),python=args.detector_python)
    assets=ROOT/'src/app/assets'
    for pair in ('train_000010','val_000005'):
        folder=root/pair
        images=[assets/f'{pair}_{date}.png' for date in ('T1','T2')]
        control=control_worker(folder)
        def worker(stage, request):
            if stage=='detector' and args.checkpoint:
                return model_worker(stage,request,folder.resolve(),config)
            if stage=='captioner':
                caption=('buildings appeared at the bottom.' if pair=='train_000010'
                         else 'two houses were built in the center.')
                return {'caption':caption,'model':'authored-review-prompt','source_kind':'authored_demo'}
            return control(stage,request)
        worker.execution_mode=('trained_bit_with_authored_claims_and_decision_control' if args.checkpoint
                               else 'real_imagery_cpu_control_with_authored_claims')
        result=run_pipeline(*images,config,folder,f'LEVIR-MCI/{pair.split("_")[0]}/{pair}',worker=worker)
        print(result['pair_id'],result['execution_mode'],[(r['source_span'],r['decision']['outcome']) for r in result['claims']],result['errors'])
        if result['errors']:raise RuntimeError(f'Replay failed: {result["errors"]}')
        mask='detector_response.mask.png' if args.checkpoint else 'control_mask.png'
        (folder/'review_inputs.json').write_text(json.dumps({'images':[str(p) for p in images],'mask':str(folder.resolve()/mask)},indent=2))
    print(root.resolve())

if __name__=='__main__':main()
