"""Validate and render an M4 BIT comparison from a local run or Colab ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parents[1]/'docs/milestone4'


def ingest(source, target):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    arms={'lr_1e-4':.0001,'lr_3e-4':.0003}
    zipper=ZipFile(source) if source.is_file() else None
    def read(name):
        return zipper.read(name) if zipper else (source/name).read_bytes()
    try:
        config=json.loads(read('run.json'));cohort=json.loads(read('cohort.json'));selection=json.loads(read('selection.json'))
        assert set(cohort)=={'train','val'}
        assert len(cohort['train'])==config['train_pairs'] and len(cohort['val'])==config['validation_pairs']
        for split,rows in cohort.items():
            assert all(row['official_split']==split for row in rows)
            assert len({row['pair_id'] for row in rows})==len(rows)
        histories={arm:json.loads(read(arm+'_history.json')) for arm in arms}
        assert all(rows and [row['epoch'] for row in rows]==list(range(1,len(rows)+1)) for rows in histories.values())
        assert selection['common_epochs']==min(map(len,histories.values()))
        for arm,rows in histories.items():
            best=max(rows,key=lambda row:row['change_class_mean_iou'])
            saved=selection['best_by_arm'][arm]
            assert saved['epoch']==best['epoch'] and saved['change_class_mean_iou']==best['change_class_mean_iou']
            assert saved['checkpoint']==arm+'.pt'
            assert hashlib.sha256(read(saved['checkpoint'])).hexdigest()==selection['checkpoint_sha256'][saved['checkpoint']]
        selected=selection['selected']
        if selected:
            assert len({len(rows) for rows in histories.values()})==1
            assert selected['arm']==max(arms,key=lambda arm:selection['best_by_arm'][arm]['change_class_mean_iou'])
        target.mkdir(parents=True,exist_ok=False)
        for name in ['run.json','selection.json','cohort.json',*(arm+'_history.json' for arm in arms)]:
            (target/name).write_bytes(read(name))
    finally:
        if zipper:zipper.close()
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.0))
    for ax,metric,title in zip(axes,['train_loss','validation_loss','change_class_mean_iou'],['Train loss','Validation loss','Change-class mean IoU']):
        for arm,lr in arms.items():
            rows=histories[arm];ax.plot([r['epoch'] for r in rows],[r[metric] for r in rows],marker='o',label=f'LR {lr:g}')
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_xticks(range(1,max(map(len,histories.values()))+1));ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(target/'learning_curves.png',dpi=180);plt.close(fig)
    return config,selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('--target',type=Path,default=HERE/'evidence/bit_comparison')
    args=parser.parse_args()
    config,selection=ingest(args.source,args.target.resolve())
    print(json.dumps(selection.get('selected'),indent=2))
