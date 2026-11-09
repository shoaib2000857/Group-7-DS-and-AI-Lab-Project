"""Render point estimates transcribed from the preserved team benchmark summary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 data=json.loads((ROOT/'evidence/provided_local_benchmarks.json').read_text())
 labels=['RSCaMa','RSCaMa retrain','Qwen3-VL','SmolVLM2','Pixel template','No-change']
 strict=[r['strict_precision'][1] for r in data['caption_comparison']]
 alarm=[r['metrics'][-1] for r in data['supplementary']]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(10,3.1),sharey=True)
 y=np.arange(6)
 for ax,vals,title,color in zip(axes,[strict,alarm],['Strict claim precision (%)','Unchanged-pair false alarms (%)'],['#1f5fae','#8b6e4e']):
  ax.barh(y,vals,color=color,height=.6);ax.set_xlim(0,108);ax.set_title(title,loc='left',fontsize=11,fontweight='bold');ax.set_yticks(y,labels);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  for yy,v in zip(y,vals):ax.text(v+1.4,yy,f'{v:.1f}',va='center',fontsize=9)
 axes[0].invert_yaxis();fig.tight_layout(w_pad=2);fig.savefig(ROOT/'assets/local_caption_benchmarks.png',dpi=220,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
