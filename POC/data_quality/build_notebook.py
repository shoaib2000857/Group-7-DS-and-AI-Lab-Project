"""Build the optional POC preprocessing experiments, separate from submission notebooks."""
from pathlib import Path
import sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.build_data_notebooks import SETUP,md,code

def build():
    cells=[md('# Data-quality experiments\n\n## Summary\nControlled experiments on real source images measure preprocessing effects. These are not model benchmarks or evidence of generalisation.'),code(SETUP),
           md('## Context and methods\nUse one fixed real pair for controlled comparisons. Photometric perturbations are synthetic stress cases and are labelled as such. Mask components are not treated as object counts.'),
           md('## Data'),code("from src.data.dataset import PairDataset, paired_augmentation\nds=PairDataset(PROCESSED_DIR/'train.jsonl',PROCESSED_DIR)\nprofiles={r['pair_id']:r for r in read_jsonl(AUDIT_DIR/'profiles.jsonl')}\nidx=next(i for i,r in enumerate(ds.rows) if r['dataset']=='levir_mci' and 0<profiles[r['pair_id']].get('changed_fraction',0)<.01)\nsample=ds[idx]; a,b=sample['t1'],sample['t2'];mask=sample['targets']['target']\nprint('Pair:',sample['pair_id'])"),
           md('## Results\n### Resize sensitivity'),
           code("from PIL import Image\nrows=[]\nfig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')\nfor ax,size in zip(axes,[64,128,256]):\n    m=np.array(Image.fromarray(mask.astype(np.uint8)).resize((size,size),Image.Resampling.NEAREST))\n    rows.append({'size':size,'changed_fraction':float((m>0).mean()),'classes':str(np.unique(m).tolist())})\n    ax.imshow(m,vmin=0,vmax=2);ax.set_title(f'{size} x {size}');ax.axis('off')\nplt.show();display(pd.DataFrame(rows))"),
           md('### Brightness perturbations can change pixel differences without construction'),
           code("rows=[];fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')\nfor ax,gain in zip(axes,[.7,1.,1.3]):\n    altered=np.clip(a.astype(float)*gain,0,255).astype(np.uint8)\n    rows.append({'synthetic_brightness_gain':gain,'mae_against_same_source':float(np.abs(a.astype(float)-altered).mean())})\n    ax.imshow(altered);ax.set_title(f'Same T1, gain {gain}');ax.axis('off')\nplt.show();display(pd.DataFrame(rows))"),
           md('### Paired geometry and caption safeguards'),
           code("ta,tb,tm=paired_augmentation(a,b,mask,'flip_horizontal',has_caption=False)\nassert np.array_equal(tm,np.flip(mask,axis=1))\ntry:\n    paired_augmentation(a,b,mask,'flip_horizontal',has_caption=True)\nexcept ValueError as exc:\n    print('Caption safeguard:',exc)\nprint('Label histogram preserved:',np.array_equal(np.bincount(mask.ravel()),np.bincount(tm.ravel())))\nds.close()"),
           md('## Takeaways\nResolution can remove small regions; brightness differences alone are not structural-change labels. Geometric transforms need matching target transforms, and spatial captions require their own verified transformation rules. These experiments establish preprocessing behaviour only.')]
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.validate(notebook)
    nbf.write(notebook,Path(__file__).parent/'03_Data_Quality_Experiments.ipynb')

if __name__=='__main__':build()
