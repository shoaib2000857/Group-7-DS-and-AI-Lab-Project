"""Evaluate detector masks or captions on a selected prepared manifest."""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
from src.data.dataset import PairDataset
from src.evaluation.segmentation import confusion_matrix,metrics_from_confusion


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['detector','captions'])
    parser.add_argument('--processed',default='data/processed')
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--dataset',default='levir_mci')
    parser.add_argument('--predictions',required=True,help='JSONL pair_id/mask_path, or pair_id/raw_text/status/run_id')
    parser.add_argument('--output',required=True)
    parser.add_argument('--text-metrics',action='store_true',help='COCO-caption metrics; requires Java and requirements-model-evaluation.txt')
    args=parser.parse_args()
    predictions_path=Path(args.predictions).resolve()
    rows=[json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['pair_id'] for r in rows})!=len(rows):raise ValueError('Nonempty unique prediction pair IDs required')
    root=Path(args.processed)
    dataset=PairDataset(root/(args.split+'.jsonl'),root,include_references=args.task=='captions')
    try:
        dataset.rows=[row for row in dataset.rows if row['dataset']==args.dataset]
        index={row['pair_id']:i for i,row in enumerate(dataset.rows)}
        if any(row['pair_id'] not in index for row in rows):raise ValueError('Predictions outside selected source/split')
        if args.task=='captions':
            from src.evaluation.caption import PreparedTruth,evaluate_predictions
            scope={'road':1,'building':2} if args.dataset=='levir_mci' else {'building':1}
            summary=evaluate_predictions(rows,PreparedTruth(dataset,scope),args.output,args.text_metrics)
        else:
            classes=3 if args.dataset=='levir_mci' else 2
            cm=np.zeros((classes,classes),dtype='int64');unchanged=alarms=0
            for row in rows:
                target=dataset[index[row['pair_id']]]['targets']['target']
                path=Path(row['mask_path']);path=path if path.is_absolute() else predictions_path.parent/path
                with Image.open(path) as image:prediction=np.array(image)
                cm+=confusion_matrix(target,prediction,classes)
                if not target.any():unchanged+=1;alarms+=int(prediction.any())
            summary=metrics_from_confusion(cm)|{'evaluated_pairs':len(rows),'manifest_pairs':len(dataset),
                'unchanged_pairs':unchanged,'false_alarm_pairs':alarms,'false_alarm_rate':alarms/unchanged if unchanged else None,
                'split':args.split,'dataset':args.dataset}
            out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
            (out/'eval_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))
    finally:dataset.close()
