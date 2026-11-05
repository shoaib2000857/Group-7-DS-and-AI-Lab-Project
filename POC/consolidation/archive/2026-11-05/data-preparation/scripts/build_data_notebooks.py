"""Build readable main-project notebooks from the same reusable Python pipeline."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
SETUP='''from pathlib import Path
import os, sys, json, importlib.util, subprocess
try:
    IN_COLAB = importlib.util.find_spec('google.colab') is not None
except ModuleNotFoundError:
    IN_COLAB = False
roots = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((p for p in roots if (p / 'configs/datasets.yaml').exists()), None)
if PROJECT_ROOT is None and IN_COLAB:
    PROJECT_ROOT = Path('/content/TerraWatch')
    if not (PROJECT_ROOT / '.git').exists():
        subprocess.run(['git','clone','--branch','data-preparation',
            'https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project.git',str(PROJECT_ROOT)],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','-r',str(PROJECT_ROOT/'requirements-data.txt')],check=True)
if PROJECT_ROOT is None:
    raise RuntimeError('Clone the project first, then open this notebook in its notebooks folder. See notebooks/README.md.')
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.data.catalog import load_catalog
from src.data.preprocess import read_jsonl
AUDIT_DIR = PROJECT_ROOT / 'outputs/data'
PROCESSED_DIR = PROJECT_ROOT / 'data/processed'
DATA_ROOT = os.getenv('TERRAWATCH_DATA_ROOT')  # e.g. /content/drive/MyDrive/TerraWatch_Extracted
CONFIG = PROJECT_ROOT / 'configs/datasets.yaml'
'''
COLAB='''# Optional: run only when the datasets are in Google Drive and you are in Colab.
MOUNT_DRIVE = IN_COLAB
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = '/content/drive/MyDrive/TerraWatch_Extracted'
'''


def save(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':name}})
    nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/name)


def build():
    cells=[md('# Dataset EDA\n\n## Summary\nThis notebook audits the data actually available for this run. The source-coverage table distinguishes full local releases, convenience probes, shared-image aliases and unavailable sources. Saved results are measurements, not invented examples.'),
           md('## Context and methods\nImages, masks and captions have dataset-specific meanings. Labels are never coerced into a universal ontology. Exact frames and 256-pixel tiles help detect reuse; no exact-match test proves geographic independence. Brightness, outliers and correlations are descriptive, not change truth.\n\n### Key assumptions\n- T1/T2 order follows the release.\n- One image pair stays in one partition across all its captions/instructions.\n- Missing data is unavailable, not a zero-valued observation.\n- HLS requires a chosen area/date/product and is not individual-building supervision.'),md('### Set up the project'),code(SETUP),code(COLAB),
           md('## Data\nSet `RERUN_AUDIT=True` for a fresh scan. With mounted Drive, the same adapters scan the complete available folders. A local subset is labelled as such; results from different scopes must not be pooled as a common benchmark.'),
           code("from src.data.audit import run_audit\nRERUN_AUDIT = not (AUDIT_DIR / 'audit_summary.json').exists()\nMAX_PAIRS = None  # None scans all available records; an integer is a clearly labelled cap\nif RERUN_AUDIT:\n    run_audit(CONFIG, DATA_ROOT, AUDIT_DIR, max_pairs=MAX_PAIRS, workers=4)\nsummary = json.loads((AUDIT_DIR / 'audit_summary.json').read_text())\nfrom src.data.plots import availability_table, profile_frame, plot_dataset, plot_correlations, plot_transitions\ndisplay(availability_table(summary))"),
           md('### Dataset roles and label coverage'),code("catalog=load_catalog(CONFIG)\ndisplay(pd.DataFrame([{'dataset':d['id'],'target_kind':d['kind'],'supervision':', '.join(d['supervised_classes']),'training_eligible':d['training_eligible'],'source':d['source']} for d in catalog['datasets']]))"),
           md('## Results\nEach panel describes only the inspected population listed above. Native dimensions and label proportions are measured before standardisation.'),
           code("profiles=profile_frame(AUDIT_DIR)\nexecuted=[k for k,v in summary.items() if v['status']=='executed' and v.get('kind')!='instructions']\nfor dataset in executed:\n    if summary[dataset]['valid_records']:\n        display(Markdown('### '+dataset+' â€” '+summary[dataset]['scope']))\n        plot_dataset(profiles,dataset); plt.show()\n    else:\n        display(profiles.loc[profiles.dataset==dataset,['pair_id','error']].head(10))"),
           md('### Completeness, small changes and conflicts'),code("display(pd.DataFrame([{'dataset':d,'audited':summary[d]['records_audited'],'invalid':summary[d]['invalid_records'],'missing_joins':summary[d]['missing_joins'],'small_changes_under_1pct':summary[d]['small_changes_below_one_percent'],'mask_caption_flags_disagree':summary[d]['mask_caption_conflicts'],'empty_captions':summary[d]['empty_caption_references']} for d in executed]))"),
           md('### Duplicate and leakage evidence\nRun the preprocessing notebook for the full exclusion ledger. An exact-frame or tile match can expose cross-release leakage even when filenames differ. Parent-scene independence under arbitrary resampling is outside exact hashing.'),
           code("from collections import defaultdict\nseen=defaultdict(set)\nfor row in profiles.to_dict('records'):\n    if row['valid']:\n        for h in row['fingerprints']: seen[h].add(row['pair_id'])\nshared=[sorted(ids) for ids in seen.values() if len(ids)>1]\ndisplay(pd.DataFrame({'exact_shared_frame_or_tile_groups':[len(shared)]}))\ndisplay(pd.DataFrame({'example_pair_ids':shared[:8]}))"),
           md('### Outliers and descriptive correlations'),code("outliers=[]\nfor d in executed:\n    for feature,v in summary[d]['outliers_iqr'].items(): outliers.append(dict(dataset=d,feature=feature,**v))\ndisplay(pd.DataFrame(outliers))\nplot_correlations(profiles,'levir_mci'); plt.show()"),
           md('### Semantic transitions'),code("for d in executed:\n    if summary[d].get('transition_pixels'):\n        plot_transitions(summary,d); plt.show()"),
           md('### Caption references and instruction variants\nChangeChat rows are derived supervision, not new independent image pairs or human factuality judgements. Train/train-loc files are alternative variants; their raw row counts must not be added as unique examples. Training selects one variant in configuration.'),
           code("display(pd.DataFrame([{'dataset':d,'reference_captions':summary[d].get('caption_references',0),'largest_repeated_caption_set':summary[d].get('largest_repeated_caption_set',0),'lexically_mixed_reference_pairs':summary[d].get('lexically_mixed_reference_pairs',0)} for d in executed if summary[d].get('caption_references')]))\nif 'changechat' in summary and summary['changechat']['status']=='executed':\n    s=summary['changechat']\n    display(pd.DataFrame([{'annotation_rows':s['records_audited'],'unique_image_pairs':s['unique_parent_pairs'],'exact_duplicate_rows_across_files':s['duplicate_instruction_rows'],'missing_image_pairs':s['missing_parent_rows']}]))\n    counts=pd.Series(s['annotation_file_rows'],name='rows')\n    display(counts.to_frame())\n    ax=counts.plot.barh(figsize=(10,4),color='#1F5FAE',title='Rows per annotation file (variants overlap)'); ax.set_xlabel('Rows, not independent image pairs'); plt.tight_layout(); plt.show()"),
           md('## Takeaways'),code("for d in executed:\n    s=summary[d]\n    display(Markdown(f\"- **{d}:** {s['records_audited']:,} inspected pairs; {s['invalid_records']} invalid; {s['mask_caption_conflicts']} coarse mask/caption conflicts. Scope: {s['scope']}.\"))\ndisplay(Markdown('Use the measured label coverage and leakage checks when choosing task-specific training data; a larger pooled row count does not establish better generalisation.'))")]
    save('01_EDA.ipynb',cells)
    cells=[md('# Dataset preprocessing and training handover\n\n## Summary\nThis notebook applies the recorded exclusion policy, prepares class targets and task manifests, and verifies real model-input batches. It uses the EDA notebook\'s source records; it does not run neural-model training.'),md('## Context and methods\nExact duplicates retain test before validation before training. Conflicting repeated targets and incompatible joint mask/caption flags are quarantined, not relabelled. Exact frame/tile collisions protect partitions across datasets. Original imagery stays immutable. All geometric changes are recorded.'),code(SETUP),code(COLAB),
           md('## Data\nRun the EDA notebook first. This notebook deliberately fails rather than making up a missing source audit.'),code("assert (AUDIT_DIR/'source_records.jsonl').exists(), 'Run 01_EDA first'\nsource_rows=read_jsonl(AUDIT_DIR/'source_records.jsonl')\ndisplay(pd.DataFrame(source_rows).groupby(['dataset','official_split']).size().rename('source_pairs').to_frame())"),
           md('## Results\n### Prepare targets and partitions\nRGB is resized bilinearly and categorical targets with nearest neighbour. Captions stay attached to full frames; no crop or rotation silently changes their location words. Source images remain available at native resolution for VLM processors.'),
           code("from src.data.preprocess import prepare\nRERUN_PREPARATION = not (AUDIT_DIR/'preparation_summary.json').exists()\nIMAGE_SIZE = 256  # set None to retain native sizes; batching then requires a task-specific shape policy\nif RERUN_PREPARATION:\n    prepare(AUDIT_DIR, PROCESSED_DIR, CONFIG, IMAGE_SIZE)\nprepared=json.loads((AUDIT_DIR/'preparation_summary.json').read_text())\ndisplay(pd.Series({k:prepared[k] for k in ['source_records','prepared_records','excluded_records','duplicate_groups','whole_changed_masks_lost_after_resize']}).to_frame('value'))\ndisplay(pd.Series(prepared['by_dataset_split'],name='prepared_pairs').to_frame())"),
           md('### Inspect exclusions without rewriting ground truth'),code("exclusions=json.loads((PROCESSED_DIR/'exclusions.json').read_text())\ndisplay(pd.Series(prepared['exclusion_reasons_nonexclusive'],name='reason_count').to_frame())\ndisplay(pd.DataFrame([{'pair_id':k,'reasons':', '.join(v)} for k,v in list(exclusions.items())[:10]]))"),
           md('### Task-dependent training data\nSeparate datasets support separate objectives. Binary building labels do not teach road negatives; broad SYSU change does not equal building-only change. The curriculum is described in docs/data/Training_Data_Plan.md.'),
           code("display(pd.DataFrame(prepared['tasks']).T.fillna(0).astype(int))"),
           md('### Verify every retained sample\nSet a limit only for a clearly labelled smoke check. The saved validation output records whether all records or a bounded subset were visited.'),
           code("from src.data.cli import verify\nRERUN_VERIFICATION = not (AUDIT_DIR/'loader_validation.json').exists()\nif RERUN_VERIFICATION: verify(PROCESSED_DIR, AUDIT_DIR, limit=None)\nvalidation=json.loads((AUDIT_DIR/'loader_validation.json').read_text())\ndisplay(pd.DataFrame(validation['splits']).T)"),
           md('### Inspect a real batch and reference separation'),
           code("from src.data.dataset import PairDataset\nds=PairDataset(PROCESSED_DIR/'train.jsonl',PROCESSED_DIR,include_references=True)\nbatch=ds.batch(range(min(4,len(ds))))\nprint('Images:',batch['images'].shape,batch['images'].dtype,'range',float(batch['images'].min()),float(batch['images'].max()))\nprint('Targets:',batch['targets'].shape,'classes',np.unique(batch['targets']))\nprofile_map={r['pair_id']:r for r in read_jsonl(AUDIT_DIR/'profiles.jsonl')}\nexample=next(i for i,r in enumerate(ds.rows) if r['dataset']=='levir_mci' and profile_map[r['pair_id']].get('changed_fraction',0)>.1)\nsample=ds[example]\nfig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')\nfor ax,key in zip(axes[:2],['t1','t2']): ax.imshow(sample[key]); ax.set_title(key.upper()); ax.axis('off')\naxes[2].imshow(sample['targets']['target'],vmin=0,vmax=2,cmap='viridis'); axes[2].set_title('Target: 0 background, 1 road, 2 building'); axes[2].axis('off'); plt.show()\nprint('Reference count:',len(sample['references']))\ninfer=PairDataset(PROCESSED_DIR/'test_inference.jsonl',PROCESSED_DIR,include_targets=False)\nassert set(infer[0])=={'pair_id','t1','t2'}\nprint('Inference fields:',sorted(infer[0]))\nds.close(); infer.close()"),
           md('## Takeaways'),code("display(Markdown(f\"Prepared **{prepared['prepared_records']:,}** pairs with **{prepared['excluded_records']:,}** recorded exclusions across the available sources. Validation scope is shown above. Training-ready data is not a model-performance or human-factuality result.\"))")]
    save('02_Preprocessing.ipynb',cells)
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
    save('03_Data_Quality_Experiments.ipynb',cells)
    print('Built three main-project notebooks')


if __name__=='__main__':build()
