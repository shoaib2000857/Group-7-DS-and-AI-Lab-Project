"""Validate and render an M4 BIT comparison from a local run or Colab ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent


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


def update_report(config,selection,target):
    report=HERE/'Milestone4_Report.md';text=report.read_text(encoding='utf-8-sig')
    begin='<!-- M4_COMPARISON_BEGIN -->';end='<!-- M4_COMPARISON_END -->'
    old='<!-- M4_COMPARISON -->' if begin not in text else text[text.index(begin):text.index(end)+len(end)]
    device=config['gpu'] or 'CPU';section=f'### Matched learning-rate comparison\n\nBoth arms used {config["train_pairs"]} strict training pairs and {config["validation_pairs"]} strict validation pairs on {device}. The comparison completed {selection["common_epochs"]} common epochs with identical initial weights, batch size and paired flips.\n\n'
    section+='| Head learning rate | Best epoch | Validation change-class mean IoU |\n|---|---:|---:|\n'
    for arm,row in selection['best_by_arm'].items():section+=f'| {arm.removeprefix("lr_")} | {row["epoch"]} | {100*row["change_class_mean_iou"]:.2f}% |\n'
    winner=selection['selected']
    if winner:
        section+=f'\nThe comparison selects learning rate {winner["arm"].removeprefix("lr_")} at epoch {winner["epoch"]}, with {100*winner["change_class_mean_iou"]:.2f}% validation change-class mean IoU. This result characterises the bounded fitting population.\n'
    else:section+='\nThe run reports both arms separately because their completed epoch budgets differ.\n'
    section+=f'\n![Matched BIT train loss, validation loss and validation change-class mean IoU]({(target/"learning_curves.png").relative_to(HERE).as_posix()})\n\n'
    report.write_text(text.replace(old,f'{begin}\n\n{section}{end}'),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('--target',type=Path,default=HERE/'evidence/bit_comparison')
    parser.add_argument('--update-report',action='store_true');args=parser.parse_args()
    config,selection=ingest(args.source,args.target.resolve())
    if args.update_report:update_report(config,selection,args.target.resolve())
    print(json.dumps(selection.get('selected'),indent=2))
