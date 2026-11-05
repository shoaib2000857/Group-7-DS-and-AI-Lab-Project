"""Readable notebook charts built directly from executed per-pair profiles."""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .preprocess import read_jsonl

BLUE,NAVY,GRAY='#1F5FAE','#12202B','#9DA7AF'


def set_style():
    plt.rcParams.update({'figure.dpi':110,'font.size':10,'axes.titlesize':11,'axes.titleweight':'bold',
                         'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False})


def profile_frame(audit_dir='outputs/data'):
    return pd.DataFrame(read_jsonl(Path(audit_dir)/'profiles.jsonl'))


def availability_table(summary):
    return pd.DataFrame([{'dataset':k,'status':v['status'],'audited_records':v.get('records_audited'),
                          'valid_records':v.get('valid_records'),'scope':v.get('scope',''),
                          'missing_joins':v.get('missing_joins')} for k,v in summary.items()])


def plot_dataset(profiles,dataset):
    set_style();df=profiles[(profiles.dataset==dataset)&(profiles.valid==True)].copy()
    if df.empty:raise ValueError('No executed valid profiles for this dataset')
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    fig.suptitle(f'{dataset} - {len(df):,} inspected pairs',fontsize=15,weight='bold')
    shapes=(df.height.astype(str)+' x '+df.width.astype(str)).value_counts()
    axes[0,0].barh(shapes.index,shapes.values,color=BLUE);axes[0,0].set(title='Native image dimensions',xlabel='Pairs')
    counts=Counter()
    for c in df.get('label_counts',[]):counts.update(c)
    keys=sorted(counts);total=sum(counts.values())
    names={'0':'Background','1':'Road change','2':'Building change'} if df.iloc[0]['kind']=='mci' else {'0':'No annotated change','1':'Annotated change'}
    axes[0,1].barh([names.get(k,k) for k in keys],[counts[k]/total*100 for k in keys],color=[GRAY,BLUE,NAVY][:len(keys)])
    axes[0,1].set(title='Target pixel distribution',xlabel='Inspected target pixels (%)',xlim=(0,100))
    axes[0,2].hist(df.changed_fraction*100,bins=25,color=BLUE,edgecolor='white',linewidth=.4)
    axes[0,2].set(title='Changed-area distribution',xlabel='Changed pixels per pair (%)',ylabel='Pairs')
    axes[1,0].hist(df.brightness_delta,bins=30,color=BLUE,edgecolor='white',linewidth=.4)
    axes[1,0].set(title='Temporal brightness difference',xlabel='Mean luminance T2 minus T1 (0-255)',ylabel='Pairs')
    axes[1,1].scatter(df.changed_fraction*100,df.pixel_mae,s=8,alpha=.25,color=BLUE,rasterized=True)
    axes[1,1].set(title='Pixel difference is not a change label',xlabel='Annotated changed area (%)',ylabel='Mean absolute RGB difference (0-255)')
    grid=np.mean(np.stack(df.grid_changed_fraction.to_list()),axis=0)*100
    axes[1,2].imshow(grid,cmap='Blues',vmin=0,vmax=max(15,grid.max()))
    axes[1,2].set_xticks([0,1,2],['Left','Centre','Right']);axes[1,2].set_yticks([0,1,2],['Top','Middle','Bottom'])
    axes[1,2].set_title('Image-relative changed-pixel density')
    for y in range(3):
        for x in range(3):axes[1,2].text(x,y,f'{grid[y,x]:.1f}%',ha='center',va='center',color='white' if grid[y,x]>12 else NAVY)
    return fig


def plot_correlations(profiles,dataset):
    set_style();df=profiles[(profiles.dataset==dataset)&(profiles.valid==True)]
    cols=[c for c in ['changed_fraction','brightness_delta','pixel_mae','caption_tokens_mean','components_8'] if c in df and df[c].notna().any()]
    corr=df[cols].corr();fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    im=ax.imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1);fig.colorbar(im,ax=ax,label='Pearson r')
    ax.set_xticks(range(len(cols)),[c.replace('_',' ') for c in cols],rotation=35,ha='right')
    ax.set_yticks(range(len(cols)),[c.replace('_',' ') for c in cols]);ax.set_title(dataset+' - descriptive correlations')
    for y in range(len(cols)):
        for x in range(len(cols)):
            v=corr.iloc[y,x];ax.text(x,y,'n/a' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',color='white' if pd.notna(v) and abs(v)>.7 else NAVY)
    return fig


def plot_transitions(summary,dataset):
    transitions=summary[dataset].get('transition_pixels',{})
    if not transitions:raise ValueError('No semantic transition targets for this dataset')
    matrix=np.zeros((7,7),dtype=np.int64)
    for k,n in transitions.items():
        a,b=map(int,k.split('->'));matrix[a,b]+=n
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');im=ax.imshow(np.log1p(matrix),cmap='Blues')
    labels=['Unchanged','Water','Ground','Low vegetation','Trees','Buildings','Sports fields']
    ax.set_xticks(range(7),labels,rotation=40,ha='right');ax.set_yticks(range(7),labels)
    ax.set(xlabel='T2 semantic change class',ylabel='T1 semantic change class',title=dataset+' - semantic target transitions')
    fig.colorbar(im,ax=ax,label='log(1 + labelled pixels)')
    return fig
