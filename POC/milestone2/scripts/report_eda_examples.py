"""Render compact report EDA and two traceable source-annotation examples."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from src.data.preprocess import read_jsonl
from src.data.storage import Store
from src.data.adapters import decode_targets

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, default=PROJECT/'outputs/data')
    args = parser.parse_args()
    profiles = read_jsonl(args.audit_dir/'profiles.jsonl')
    records = {r['pair_id']: r for r in read_jsonl(args.audit_dir/'source_records.jsonl')}
    summary = json.loads((args.audit_dir/'audit_summary.json').read_text())
    assets = ROOT/'assets'
    plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':11,
                         'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2,2.65), layout='constrained')
    for dataset, label, color in [('levir_mci','LEVIR-MCI (10,077)','#1F5FAE'),
                                   ('second_cc','SECOND-CC (6,041)','#526A86')]:
        rows = [p for p in profiles if p['dataset']==dataset and p['valid']]
        assert len(rows)==summary[dataset]['records_audited']
        axes[0].hist([p['changed_fraction']*100 for p in rows], bins=np.arange(0,101,5),
                     weights=np.full(len(rows),100/len(rows)),histtype='step',linewidth=1.8,color=color,label=label)
        axes[1].hist([p['brightness_delta'] for p in rows],bins=np.arange(-150,151,10),
                     weights=np.full(len(rows),100/len(rows)),histtype='step',linewidth=1.8,color=color,label=label)
    axes[0].set(title='Annotated changed area',xlabel='Changed pixels per image pair (%)',ylabel='Share of source pairs (%)')
    axes[1].set(title='Temporal brightness difference',xlabel='Mean luminance: T2 minus T1 (0-255)',ylabel='Share of source pairs (%)')
    axes[0].legend(fontsize=8,frameon=False)
    fig.savefig(assets/'eda_source_comparison.png',dpi=240);plt.close(fig)
    examples=[('levir_mci/train/train_000014',0),('levir_mci/train/train_000013',4)]
    fig, axes=plt.subplots(2,3,figsize=(9.2,5.4))
    fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.065,wspace=.035,hspace=.33)
    provenance=[]
    for i,(pid,caption_index) in enumerate(examples):
        row=records[pid];store=Store(row['source_path'])
        try:
            a=store.image(row['t1'])[0];b=store.image(row['t2'])[0]
            mask=decode_targets(store,row,row['kind'])['target']
        finally:store.close()
        for ax,im in zip(axes[i],[a,b,mask]):
            if im.ndim==2:ax.imshow(im,cmap=ListedColormap(['#FFFFFF','#E3AD22','#CC3344']),vmin=0,vmax=2,interpolation='nearest')
            else:ax.imshow(im)
            ax.axis('off')
        if i==0:
            for ax,title in zip(axes[0],['Earlier image (T1)','Later image (T2)','Source change mask']):ax.set_title(title,fontsize=12)
        y=axes[i,1].get_position().y0-.022
        caption=row['captions'][caption_index]
        fig.text(.5,y,f"{pid.rsplit('/',1)[-1]} - Source reference: \"{caption}\"",ha='center',va='top',fontsize=11)
        provenance.append({'pair_id':pid,'source_revision':summary['levir_mci']['source_revision'],
                           'caption_index_zero_based':caption_index,'source_reference':caption,
                           't1_sha256':next(p['t1_sha256'] for p in profiles if p['pair_id']==pid),
                           't2_sha256':next(p['t2_sha256'] for p in profiles if p['pair_id']==pid)})
    fig.savefig(assets/'eda_image_mask_captions.png',dpi=240);plt.close(fig)
    (ROOT/'evidence/report_eda_examples.json').write_text(json.dumps({'sources':{d:{k:summary[d][k] for k in ['records_audited','source_revision','label_pixel_percent']} for d in ['levir_mci','second_cc']},'examples':provenance},indent=2),encoding='utf-8')


if __name__=='__main__':main()
