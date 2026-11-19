from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
SETUP='''from pathlib import Path
import os, sys, json, subprocess, importlib.util
try:
    IN_COLAB = importlib.util.find_spec('google.colab') is not None
except ModuleNotFoundError:
    IN_COLAB = False
PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/pipeline.json').exists()), None)
if PROJECT_ROOT is None and IN_COLAB:
    PROJECT_ROOT = Path('/content/TerraWatch')
    subprocess.run(['git','clone','--branch','model-pipeline','https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project.git',str(PROJECT_ROOT)],check=True)
if PROJECT_ROOT is None:
    raise RuntimeError('Open this notebook from a clone of the model-pipeline branch.')
os.chdir(PROJECT_ROOT)
sys.path.insert(0,str(PROJECT_ROOT))
if IN_COLAB:
    subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-pipeline.txt'],check=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from datetime import datetime, timezone
plt.rcParams.update({'figure.figsize':(10,3.4),'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
RUN = PROJECT_ROOT/'outputs/model_notebooks'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
RUN.mkdir(parents=True,exist_ok=False)
PROCESSED = Path(os.environ.get('TERRAWATCH_PROCESSED',str(PROJECT_ROOT/'data/processed')))
CONFIG_PATH = PROJECT_ROOT/'configs/pipeline.json'
CONFIG = json.loads(CONFIG_PATH.read_text())
print('Run:',RUN)
'''
nb3=[md('''# Model Experiments and Pipeline Walkthrough

## Goal
Inspect the selected M3 stack, replay the recorded comparison, and pass one image pair through the reusable pipeline. The default CPU execution uses an **Otsu/template integration control** and a defer-only decision control. Choose `RUN_LEARNED=True` to use BIT, RSCaMa and Qwen after supplying checkpoints and worker interpreters. Historical model measurements and fresh control outputs have separate labels.

The main implementations live in `src/`. The POC originals remain available for reproducing their original protocols.'''),md('## 1. Setup and run parameters'),code(SETUP),md('## 2. Recorded model comparison\nSource: `docs/milestone3/evidence/caption_benchmark_proof.json`, supplied local run dated 9 November 2026. Percentages use automatic mask/reference checks on the frozen 100-pair screen.'),code('''proof = json.loads(Path('docs/milestone3/evidence/caption_benchmark_proof.json').read_text())
comparison = pd.DataFrame([{'Model':r['configuration'], 'Strict precision (%)':100*r['metrics']['strict_precision']['local_rerun'], 'In-scope precision (%)':100*r['metrics']['in_scope_precision']['local_rerun'], 'BLEU-4':r['metrics']['BLEU-4']['local_rerun']} for r in proof['comparisons']])
display(comparison.round(4))
fig, ax = plt.subplots(figsize=(10,3.6))
ax.barh(comparison['Model'],comparison['Strict precision (%)'],color='#1f5fae')
ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Strict automatic claim precision (%)');ax.set_title('Recorded local caption screen: 100 LEVIR pairs')
fig.tight_layout();plt.show()
'''),md('## 3. Inspect configured model assets\nThe preflight checks paths and the acceptance threshold without downloading weights. Set the three interpreter paths in `configs/pipeline.json` to the separately installed model environments.'),code('''from src.inference.preflight import inspect_config
setup = inspect_config(CONFIG)
display(pd.DataFrame(setup['checks']))
'''),md('## 4. Select a temporal image pair\nUse the first prepared validation MCI pair when the data handover is present. A clearly labelled synthetic routing fixture provides a portable CPU walkthrough in a fresh clone. Training and benchmark results are never calculated from that fixture.'),code('''from PIL import Image
from src.inference.controls import make_fixture
from src.data.dataset import PairDataset
manifest = PROCESSED/'val.jsonl'
if manifest.exists():
    dataset=PairDataset(manifest,PROCESSED,include_targets=False)
    try:
        dataset.rows=[r for r in dataset.rows if r['dataset']=='levir_mci']
        sample=dataset[0];pair_id=sample['pair_id'];paths=[]
        for key in ['t1','t2']:
            path=RUN/(key+'.png');Image.fromarray(sample[key]).save(path);paths.append(path)
        input_source='prepared validation MCI pair'
    finally:dataset.close()
else:
    paths=make_fixture(RUN/'synthetic_input');pair_id='synthetic_routing_fixture';input_source='synthetic routing fixture'
print('Input source:',input_source,'; pair:',pair_id)
fig, axes=plt.subplots(1,2,figsize=(9,4))
for ax,path,title in zip(axes,paths,['Earlier T1','Later T2']):ax.imshow(Image.open(path));ax.set_title(title);ax.axis('off')
fig.tight_layout();plt.show()
'''),md('## 5. Execute the pipeline\nThe CPU control exercises parsing, localisation, crop inspection, decision routing and report composition. Its mask assumes that changed pixels represent the building class solely for integration testing. The learned mode uses the independent configured model workers.'),code('''RUN_LEARNED = False
from src.inference.pipeline import run_pipeline
from src.inference.controls import control_worker
pipeline_run=RUN/'pipeline'
if RUN_LEARNED:
    if input_source=='synthetic routing fixture':raise ValueError('Select real images for learned model evaluation')
    checked=inspect_config(CONFIG,environments=True)
    if not checked['ready']:raise RuntimeError('Resolve the preflight checks before launching the learned stack')
    result=run_pipeline(*paths,CONFIG,pipeline_run,pair_id=pair_id)
else:
    result=run_pipeline(*paths,CONFIG,pipeline_run,pair_id=pair_id,worker=control_worker(pipeline_run))
print('Execution mode:',result['execution_mode'])
print('Description:',result['verified_description'])
print('Errors:',result['errors'])
claims=pd.DataFrame([{'Claim':c['source_span'],'Decision':c['decision']['outcome'],'Reasons':', '.join(c['decision']['reason_codes'])} for c in result['claims']])
display(claims)
'''),md('## 6. Inspect intermediate evidence'),code('''mask_path=pipeline_run/('detector_response.mask.png' if RUN_LEARNED else 'control_mask.png')
if mask_path.exists():
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(Image.open(mask_path),vmin=0,vmax=2,cmap='viridis');ax.set_title('Predicted/control class mask');ax.axis('off');plt.show()
if result['claims']:
    evidence=result['claims'][0]['evidence'];display(pd.DataFrame([{'Visible change':evidence['visible_change'],'Observation':evidence['observations'],'Frame':result['claims'][0]['coordinate_frame']}]))
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,path in zip(axes,evidence['crop_paths']):ax.imshow(Image.open(path));ax.axis('off')
    fig.suptitle('Matched T1/T2 crop evidence');plt.show()
'''),md('## 7. Training entry points\nThese commands call the same modules as the CLI. The JSON profiles define the full task-training configuration. The recorded two-epoch experiment keeps its separate raw-archive protocol in the preserved POC source.'),code('''RUN_TRAINING = False
training_commands = [
    [sys.executable,'scripts/train_models.py','detector','--config','configs/detector_training.json','--print-command'],
    [sys.executable,'scripts/train_models.py','captioner','--config','configs/captioner_training.json','--print-command']]
for command in training_commands:
    subprocess.run(command,check=True)
if RUN_TRAINING:
    # Use each task's own interpreter from CONFIG; remove --print-command to execute.
    for command,stage in zip(training_commands,['detector','captioner']):
        launch=command[:-1]+['--python',CONFIG[stage]['python']]
        subprocess.run(launch,check=True)
'''),md('## 8. Downloadable handover\nThe ZIP includes this run\'s images, crops, responses and report with per-file hashes. Model weights and source datasets remain separate assets.'),code('''from scripts.package_model_run import package_run
archive=package_run(RUN,RUN.parent/(RUN.name+'_results.zip'))
print(archive)
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
''')]
nb4=[md('''# Model Evaluation and Error Analysis

## Scope
Replay the recorded M3 results, inspect transfer behaviour, calculate new pixel metrics and demonstrate reference-isolated claim evaluation. Recorded learned scores are labelled separately from the fresh CPU controls. Model predictions generated by `scripts/caption_pairs.py` or pipeline workers can be passed to the same evaluation modules.'''),md('## 1. Setup'),code(SETUP),md('## 2. Replay the local and historical caption screen'),code('''proof=json.loads(Path('docs/milestone3/evidence/caption_benchmark_proof.json').read_text())
records=[]
for row in proof['comparisons']:
    metric=row['metrics']['strict_precision']
    records.append({'Model':row['configuration'],'Historical (%)':100*metric['historical_b0_json'],'Local (%)':100*metric['local_rerun'],'Delta (pp)':metric['local_minus_historical_exact']})
table=pd.DataFrame(records);display(table.round(2))
fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(table));ax.bar(x-.18,table['Historical (%)'],.36,label='Historical B0',color='#8da2ad');ax.bar(x+.18,table['Local (%)'],.36,label='Local rerun',color='#1f5fae');ax.set_xticks(x,['RSCaMa','Retrain','SmolVLM2','Qwen3-VL','Pixel','No-change']);ax.set_ylabel('Strict claim precision (%)');ax.set_ylim(0,100);ax.legend();fig.tight_layout();plt.show()
'''),md('## 3. Transfer and detector evidence'),code('''transfer=pd.DataFrame([{'Dataset':r['dataset'],'Pairs':r['sample_count_local'],'Strict (%)':100*r['metrics']['strict_precision']['local_rerun'],'BLEU-4':r['metrics']['BLEU-4']['local_rerun']} for r in proof['transfer_comparisons']]);display(transfer)
reported=json.loads(Path('docs/milestone3/evidence/reported_component_results.json').read_text())
detector=reported['detector'];display(pd.DataFrame([{'Detector':'BIT','Binary F1 (%)':detector['BIT_binary_f1_percent'],'False alarms':f"{detector['BIT_false_alarms']}/45"},{'Detector':'Otsu','Binary F1 (%)':detector['Otsu_binary_f1_percent'],'False alarms':f"{detector['Otsu_false_alarms']}/45"}]))
print('BIT class-mean IoU and Otsu binary IoU retain separate target definitions.')
'''),md('## 4. Fresh control evaluation on prepared data\nThis cell runs Otsu on up to three prepared validation MCI pairs and compares its binary change output to their targets. It is a new control sample, separate from the frozen 100-pair local benchmark. In a clone without prepared data, an explicitly synthetic mask checks metric arithmetic.'),code('''from src.evaluation.segmentation import confusion_matrix,metrics_from_confusion
from src.data.dataset import PairDataset
from PIL import Image
import cv2
manifest=PROCESSED/'val.jsonl';cm=np.zeros((2,2),dtype='int64');evaluation_rows=[]
if manifest.exists():
    dataset=PairDataset(manifest,PROCESSED)
    try:
        dataset.rows=[r for r in dataset.rows if r['dataset']=='levir_mci'][:3]
        for i in range(len(dataset)):
            sample=dataset[i];difference=np.abs(sample['t2'].astype(float)-sample['t1'].astype(float)).mean(-1).astype('uint8')
            _,prediction=cv2.threshold(difference,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            target=(sample['targets']['target']>0).astype('uint8');cm+=confusion_matrix(target,prediction,2)
            evaluation_rows.append({'pair_id':sample['pair_id'],'changed_pixels':int(target.sum()),'predicted_pixels':int(prediction.sum())})
        metric_scope=f'fresh Otsu control on {len(dataset)} prepared validation pairs'
    finally:dataset.close()
else:
    target=np.array([[0,0,1],[0,1,1]],dtype='uint8');prediction=np.array([[0,1,1],[0,0,1]],dtype='uint8');cm=confusion_matrix(target,prediction,2);metric_scope='synthetic metric arithmetic fixture'
new_metrics=metrics_from_confusion(cm)
print(metric_scope);display(pd.DataFrame(evaluation_rows));display(pd.DataFrame([new_metrics]).drop(columns=['confusion_matrix','class_iou']))
fig,ax=plt.subplots(figsize=(4,3.5));ax.imshow(cm,cmap='Blues')
for (i,j),value in np.ndenumerate(cm):ax.text(j,i,str(value),ha='center',va='center',color='black')
ax.set_xticks([0,1],['Background','Change']);ax.set_yticks([0,1],['Background','Change']);ax.set_xlabel('Prediction');ax.set_ylabel('Reference');ax.set_title('Pooled binary pixel confusion');fig.tight_layout();plt.show()
(RUN/'new_control_metrics.json').write_text(json.dumps({'scope':metric_scope,**new_metrics},indent=2))
'''),md('## 5. Fresh automatic caption checks\nOnly this evaluation driver reads source masks and reference captions. The inference worker receives images alone. The default predictions below are the deterministic no-change control, not RSCaMa outputs.'),code('''if manifest.exists():
    from src.evaluation.caption import PreparedTruth,evaluate_predictions
    dataset=PairDataset(manifest,PROCESSED,include_references=True)
    try:
        dataset.rows=[r for r in dataset.rows if r['dataset']=='levir_mci'][:3]
        predictions=[{'pair_id':r['pair_id'],'run_id':'fresh_no_change_control','raw_text':'there is no difference.','status':'ok'} for r in dataset.rows]
        summary=evaluate_predictions(predictions,PreparedTruth(dataset),RUN/'caption_control')
        display(pd.DataFrame([{k:summary[k] for k in ['pairs','supported_claim_precision_strict','reference_fact_recall','no_change_false_alarm_rate']}]))
    finally:dataset.close()
else:
    print('Prepared reference evaluation: run notebooks 01 and 02 to create the manifest and targets.')
'''),md('## 6. Human judgement and agreement\nThe protocol uses SUPPORTED, CONTRADICTED, UNSUPPORTED and UNVERIFIABLE. Two raters score independently; Cohen\'s kappa is calculated before third-rater adjudication. This notebook creates the input template and never fills human labels from model outputs.'),code('''template=pd.DataFrame(columns=['pair_id','claim_id','claim_text','rater_1','rater_2','adjudicated_label','adjudicator_notes'])
template.to_csv(RUN/'human_audit_template.csv',index=False)
print('Audit template:',RUN/'human_audit_template.csv')
'''),md('## 7. Evaluate your own model outputs\nThe detector CLI expects JSONL rows with `pair_id` and `mask_path` (relative to that JSONL or absolute). Caption rows use `pair_id`, `raw_text`, `status` and `run_id`. Prediction IDs must belong to the chosen dataset/split. Mask dimensions and class indices are checked.'),code('''commands=[
'python scripts/caption_pairs.py --split val --limit 100 --output outputs/model_evaluation/captions.jsonl',
'python scripts/evaluate_models.py captions --split val --predictions outputs/model_evaluation/captions.jsonl --output outputs/model_evaluation/caption_metrics',
'python scripts/evaluate_models.py detector --split val --predictions outputs/model_evaluation/masks.jsonl --output outputs/model_evaluation/detector_metrics']
print('\\n'.join(commands))
'''),md('## 8. Export evaluation results'),code('''from scripts.package_model_run import package_run
archive=package_run(RUN,RUN.parent/(RUN.name+'_results.zip'));print(archive)
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
''')]
for name,cells in [('03_Model_Experiments.ipynb',nb3),('04_Evaluation.ipynb',nb4)]:
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
 nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/name)
print('Built model and evaluation notebooks from reusable main modules.')
