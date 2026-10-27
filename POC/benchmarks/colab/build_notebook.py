"""Build the self-contained Colab deliverable from the reviewable runtime."""
from pathlib import Path
import base64
import hashlib
import textwrap
import zlib
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
runtime=(ROOT/'benchmark_runtime.py').read_bytes()
encoded=base64.b64encode(zlib.compress(runtime,9)).decode()
chunks='\n'.join('    '+repr(encoded[i:i+110]) for i in range(0,len(encoded),110))
cells=[]

def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

def code(text, **metadata):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip(),metadata=metadata))

md('''
# TerraWatch: Shoaib's decision-model benchmark
**Owner: Shoaib | Review initials: SSSM | Work package D | 27 October 2026**

Run this notebook in **Google Colab with a GPU**. It tests your assigned claim checkers, displays the important results here, and produces **one downloadable results ZIP**. The notebook contains its own runtime; you do not need to clone the project or upload another Python file.

**Start:** `Runtime > Change runtime type > GPU`, then `Runtime > Run all`. An L4/A100-class session can attempt the BF16-only d3 model. On a T4, compatible models continue and d3 is recorded as skipped. Actual fit and speed are measured, not guaranteed by Pro+.

The default is a **10-pair smoke run**, with four judge candidates attempted sequentially. Start here. After checking its output, use `screen` for 10 development + 30 validation pairs. No model training, new paid APIs, compute purchases or hosted Jev calls occur. Optional local Kev/Clef runs are disabled initially.

**What to send back:** the final `SSSM_..._results.zip`, even if some candidates fail. It contains raw decisions, metrics, charts, selected image pairs, settings, revisions, logs, blockers and human review sheets. It excludes model weights and runtime environments.
''')
md('''
## 1. What this run can establish

| Comparison | Input and interpretation |
|---|---|
| **Common evidence** | Every judge gets the same claim, diagnostic mask summary and frozen fallible witness observation. Text-only models are evaluated here. |
| **Image backed** | Compatible judges additionally get the original T1/T2 images, in that order. Keep this table separate: more information can improve decisions. |
| **Controls** | R00 accepts everything; R01 uses mask-only rules; R02 uses structured witness rules. Each VLM's direct image run is the R03-style control. |
| **Logic fixtures** | Constructed count/location/scope errors test the decision contract. These are **not satellite accuracy measurements**. |
| **Satellite pilot** | Real LEVIR-MCI pairs, change flags, reference captions, injected logical negations and unverifiable claims. Scores are **annotation proxies**, pending independent human review. Natural generated captions are exported unrated. |

The reference captions and change flags stay outside inference inputs. They are not substituted for missing witness evidence. The generator and checker can still share errors; J06 also supplies the default witness, so its checker is not an independent observer.

Caption-level checks are identified separately from atomic propositions. This first notebook does not claim instance-count accuracy, reference-fact recall, detector F1/IoU, a validated custom metric, or a completed held-out factuality study.
''')
md('''
## 2. Install lightweight notebook dependencies
Judge dependencies are installed in separate worker environments later. This avoids replacing the Colab kernel's Transformers package or mixing InternVL's version with d3's.
''')
code('''
import importlib.util, os, subprocess, sys
required = {"numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<4",
            "matplotlib": "matplotlib>=3.8,<4", "requests": "requests>=2.32,<3", "PIL": "Pillow>=10,<13"}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", *missing])
print("Notebook dependencies available. GPU libraries are checked per worker.")
''')
md('''
### Embedded runtime
This collapsed cell writes the runtime included inside this notebook and verifies its SHA-256. The readable source is `POC/benchmarks/colab/benchmark_runtime.py` in the repository and is copied into your results ZIP.
''')
code('''# @title Load the self-contained benchmark runtime
import base64, hashlib, zlib
from pathlib import Path
RUNTIME_DATA = (
'''+chunks+'''
)
runtime_bytes = zlib.decompress(base64.b64decode(RUNTIME_DATA))
assert hashlib.sha256(runtime_bytes).hexdigest() == '''+repr(hashlib.sha256(runtime).hexdigest())+'''
LOCAL_ROOT = Path("/content/terrawatch_decisions") if Path("/content").exists() else Path.cwd()/"qa/notebook_runtime"
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
runtime_path = LOCAL_ROOT/"benchmark_runtime.py"
runtime_path.write_bytes(runtime_bytes)
sys.path.insert(0, str(LOCAL_ROOT.resolve()))
import benchmark_runtime as tw
import pandas as pd
from IPython.display import display, Markdown, FileLink
print("Runtime:", tw.VERSION, "| Owner: Shoaib (SSSM)")
''',cellView='form')
md('''
## 3. Choose the run
Keep the defaults for the first attempt. Models run one at a time. GPU memory gates are conservative scheduling choices, not published minimum VRAM claims. Install/load/inference failures are saved and do not remove the candidate from the register.

`MODEL_IDS` may include J02/J03/J04 only after enabling optional local runtimes. J08 always stays deferred because it needs a separate access/cost decision. A CPU session runs the data and rule checks but **cannot establish learned-model results**.
''')
code('''# @title Run settings
PROFILE = "smoke" # @param ["smoke", "screen", "heldout"]
DATA_MODE = "levir_mci" # @param ["levir_mci", "team_bundle"]
MODEL_IDS = ["J05", "J01", "J06", "J07"] # @param {type:"raw"}
ALLOW_OPTIONAL_RUNTIMES = False # @param {type:"boolean"}
USE_DRIVE_BACKUP = False # @param {type:"boolean"}
TEAM_MANIFEST_PATH = "" # @param {type:"string"}
TEAM_BUNDLE_PATH = "" # @param {type:"string"}
LOCAL_DATASET_ZIP = "" # @param {type:"string"}
RESUME_RESULTS_ZIP = "" # @param {type:"string"}
FREEZE_PATH = "" # @param {type:"string"}

CONFIG = tw.default_config()
CONFIG.update(PROFILE=PROFILE, DATA_MODE=DATA_MODE, MODEL_IDS=MODEL_IDS,
              ALLOW_OPTIONAL_RUNTIMES=ALLOW_OPTIONAL_RUNTIMES,
              TEAM_MANIFEST_PATH=TEAM_MANIFEST_PATH, TEAM_BUNDLE_PATH=TEAM_BUNDLE_PATH,
              LOCAL_DATASET_ZIP=LOCAL_DATASET_ZIP, FREEZE_PATH=FREEZE_PATH)
if PROFILE == "screen":
    CONFIG.update(N_DEV=10, N_VAL=30)
# Advanced options remain explicit and are exported. No threshold is chosen using test labels.
CONFIG.update(MAX_MODEL_MINUTES=30, TOTAL_MINUTES=120, ACCEPT_THRESHOLD=0.80,
              MAX_IMAGE_SIDE=512, MAX_DATA_MB=256, OPTION_ORDER_DIAGNOSTIC_CASES=6)
if os.environ.get("TW_NOTEBOOK_TEST") == "1":
    CONFIG.update(N_DEV=1, N_VAL=1, MODEL_IDS=[], DOWNLOAD_ZIP=False,
                  EXECUTION_MODE="local_cpu_validation_no_learned_model_results")
display(pd.DataFrame([{"ID": key, **value} for key, value in tw.REGISTRY.items()]))
''')
md('''
### Optional Drive backup and resume
Local outputs are saved after every decision; a checkpoint ZIP is built after every candidate. If you enable Drive, checkpoint ZIPs are also copied to `MyDrive/TerraWatch/SSSM_benchmarks`. The selected working images stay on local disk for inference timing. Existing Colab/Drive authorisation is used; there are no new purchases.

To resume after a disconnect, mount Drive, set `RESUME_RESULTS_ZIP` to the previous ZIP path, keep the same configuration, and run again. Completed outputs are reused by content/revision keys; failed rows are only retried if you explicitly set `CONFIG['RETRY_ERRORS'] = True` before starting. Partial timings do not become fresh inference timings.
''')
code('''
if USE_DRIVE_BACKUP:
    from google.colab import drive
    drive.mount("/content/drive")
    CONFIG["BACKUP_DIR"] = "/content/drive/MyDrive/TerraWatch/SSSM_benchmarks"
if RESUME_RESULTS_ZIP:
    name = Path(RESUME_RESULTS_ZIP).name.removesuffix("_results.zip")
    if not name.startswith("SSSM_"):
        raise ValueError("Choose a results ZIP produced by this notebook.")
    resume_dir = LOCAL_ROOT/"runs"/name
    tw.safe_unpack(RESUME_RESULTS_ZIP, resume_dir)
    CONFIG["RESUME_RUN"] = str(resume_dir.resolve())
RUN, CONFIG = tw.start_run(CONFIG, LOCAL_ROOT/"runs")
display(pd.DataFrame([tw.environment()["gpu"]]))
if not tw.environment()["gpu"]["name"]:
    display(Markdown("**No GPU detected. Only data/rule validation can run; learned candidates will be recorded as blocked.**"))
''')
md('''
## 4. Load the selected data

| Source | Use in this notebook |
|---|---|
| **LEVIR-MCI** | Default real-image pilot; fetch only selected train/validation pairs from the author's release, not the full 2.7 GB archive. |
| **LEVIR-CC** | Caption annotations packaged with MCI. Shared imagery is not another independent dataset. |
| **LEVIR-CD / CD+, S2Looking, SYSU-CD, SECOND, OSCD** | Detection/transfer candidates. They lack ready-made adjudicated claim decisions; use the team-bundle route after compatible claims/references are prepared. Do not invent caption or decision labels from binary masks. |
| **ChangeChat-105k** | Optional instruction data; source overlap and labels require the team audit before benchmark use. |

If the central manifest is available, set `TEAM_MANIFEST_PATH`. Otherwise the seeded selection is **provisional**: share the exported `shared_manifest.json` with all four owners so nobody draws a fresh subset. Within-split POC ceilings are checked. Parent-scene overlap and pretrained exposure remain audit items; no unseen-domain claim follows from this pilot.

For your own reviewed bundle, upload a ZIP using the Colab Files panel and set `TEAM_BUNDLE_PATH`. Its root must contain `manifest.json`, `cases.jsonl`, `references.jsonl` and any referenced images. The repository README documents the exact schema. Sealed evaluation requires that route, frozen settings and two-rater adjudicated references; the default downloader never opens the official test split.
''')
code('''
if CONFIG["DATA_MODE"] == "team_bundle":
    CASES, REFERENCES = tw.load_team_bundle(CONFIG, RUN)
    PAIRS, ANNOTATIONS = [], []
elif CONFIG["DATA_MODE"] == "levir_mci":
    PAIRS, ANNOTATIONS = tw.prepare_levir(CONFIG, RUN)
    CASES, REFERENCES = None, None
    display(pd.DataFrame(PAIRS)[["pair_id", "official_split", "phase", "scene_grouping"]])
else:
    raise ValueError("Unknown DATA_MODE")
print("Selected real pairs:", len(PAIRS) if PAIRS else len({c["pair_id"] for c in CASES}))
''')
md('''
## 5. Freeze the witness evidence
The default J06 witness sees the original images **without reference captions, change flags or human targets**. Its JSON observation is saved once and reused across all judges. A failed observation stays unknown; the rules and text judges can still run. The mask control is deliberately labelled **absolute difference + Otsu diagnostic**, not a trained detector or verified change mask.

This also makes the first integrated baseline available without waiting for the other three work packages. To compare actual detector/captioner/witness outputs later, use their frozen team bundle.
''')
code('''
if CASES is None:
    WITNESSES = tw.collect_witnesses(PAIRS, CONFIG, RUN)
    CASES, REFERENCES = tw.make_real_cases(PAIRS, ANNOTATIONS, WITNESSES, RUN)
display(pd.DataFrame(CASES).groupby(["benchmark_group", "phase"]).size().rename("claims").reset_index())
print("Reference targets are stored separately and are never passed to the judge adapters.")
tw.archive_results(RUN, announce=False)
''')
md('''
## 6. Run the controls and judges
Models use native supported precision; the notebook does not silently quantise, change checkpoints or call a cloud service. d3 uses its publisher's readout and current BF16 requirement. Qwen judges score the three allowed decision tokens; their probabilities are **conditional on those tokens**, not proven probabilities of correctness. InternVL generates a decision code and therefore has **no fabricated confidence score**. Optional Kev preserves the publisher's adapter/pointer-head server; Clef uses a probed local System One endpoint.

The candidate time limit includes installation, loading and inference. The overall judge loop is capped at 120 minutes; the witness stage has its separate 30-minute cap. It may finish much sooner. Progress and exceptions are logged. After an interruption, you can run the results/export cells below on whatever completed.
''')
code('''
tw.run_benchmarks(CASES, CONFIG, RUN)
''')
md('''
## 7. View the important results
Compare candidates **within the same track, phase and label source**. Missing measurements stay empty. All-defer does not receive perfect precision, and failed/missing outputs reduce completion and coverage. Do not promote smoke or annotation-proxy results into human factuality claims.

The tables include supported-target acceptance, false-target rejection, false rejection, deferral, useful coverage, three-class Brier score where real probability vectors exist, fresh median/p95 latency and measured PyTorch peak memory. Server-process GPU peaks remain unmeasured. The Brier definition is the mean sum of three squared class errors, with range 0-2.

Main tables use the fixed acceptance gate: a probabilistic accept below 0.80 becomes defer. Rules and generated labels without probabilities keep their raw decisions. Raw answers and ungated metrics are also exported, so this policy change is inspectable. Calibration diagrams use the original top-option answer, not the thresholded decision.
''')
code('''
DECISIONS, METRICS, REGISTER = tw.evaluate(CASES, REFERENCES, CONFIG, RUN)
display(REGISTER[[c for c in ["model_id", "name", "status", "successful", "requested"] if c in REGISTER.columns]].fillna("—"))
selected = METRICS[METRICS["phase"].isin(["val", "test"])]
for population, group in selected.groupby("benchmark_group", sort=True):
    display(Markdown("### " + population.replace("_", " ").title()))
    quality = group[["model_id", "track", "completed_labelled_claims", "accepted_precision_proxy",
                     "true_claim_acceptance", "false_claim_rejection", "decision_coverage"]].rename(columns={
        "model_id":"Judge", "track":"Input", "completed_labelled_claims":"Labelled N",
        "accepted_precision_proxy":"Accepted precision", "true_claim_acceptance":"True accepted",
        "false_claim_rejection":"False rejected", "decision_coverage":"Coverage"}).round(3)
    operational = group[["model_id", "track", "completion_rate", "deferral_rate", "brier_multiclass_0_to_2",
                         "median_latency_s", "peak_pytorch_gpu_gb"]].rename(columns={
        "model_id":"Judge", "track":"Input", "completion_rate":"Completion", "deferral_rate":"Defer",
        "brier_multiclass_0_to_2":"Brier (0-2)", "median_latency_s":"Median seconds",
        "peak_pytorch_gpu_gb":"Peak GPU GB"}).round(3)
    display(quality.astype(object).where(pd.notna(quality), "—"))
    display(operational.astype(object).where(pd.notna(operational), "—"))
learned_ok = DECISIONS[(DECISIONS["model_id"].str.startswith("J")) & (DECISIONS["status"] == "ok")]
if learned_ok.empty:
    display(Markdown("**No learned judge completed successfully. These are rule/data checks, not a learned-model benchmark. Send the ZIP so the blockers can be diagnosed.**"))
''')
code('''
tw.plot_results(DECISIONS, METRICS, CONFIG, RUN)
''')
md('''
### Option-order, threshold and paired diagnostics
Reversed choice order is tested only on a small validation diagnostic set. Threshold sensitivity uses validation only; the fixed 0.80 acceptance threshold is a starting setting, not a validated success target. Pair/scene-group resampling is used for paired differences; few independent groups mean broad, exploratory uncertainty. These outputs support a later prompt/threshold freeze, not tuning on sealed labels.
''')
code('''
def show_csv(name, limit=20):
    path = RUN/"tables"/name
    try:
        table = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(name + ": not available for this run.")
        return
    display(table.head(limit).round(3))
show_csv("option_order_diagnostics.csv")
show_csv("paired_differences.csv")
show_csv("validation_threshold_sensitivity.csv", 14)
''')
md('''
### Targeted versus random reinspection
This replays saved text/image decisions at equal **image-call counts** for targeted and random selection. Full-image checking is an additional reference with a larger budget. It is a policy diagnostic, **not fresh end-to-end latency or token-budget equivalence**. It cannot establish an optimal trigger or true deployment savings.
''')
code('''
REPLAY = tw.targeted_replay(DECISIONS, CONFIG, RUN)
if len(REPLAY):
    display(REPLAY[["model_id", "policy", "replayed_image_calls", "eligible_common_claims",
                    "accepted_precision_proxy", "true_claim_acceptance", "false_claim_rejection"]].round(3))
else:
    print("Replay requires successful common-evidence and image-backed outputs for the same model/cases.")
''')
md('''
## 8. Inspect pairs and prepare independent review
The examples show the earlier/later pair, claims and decisions. Proxy targets are displayed **after inference**, for error inspection. Natural captions still need human review. The export includes a system-anonymised review sheet and a separate identity key. Two raters should establish visible reference facts first, retain disagreements and adjudicate; model agreement does not establish truth.
''')
code('''
tw.show_examples(CASES, DECISIONS, RUN, limit=3)
tw.export_review(CASES, DECISIONS, CONFIG, RUN)
print("Human review:", RUN/"review/blinded_claim_review.csv")
''')
md('''
## 9. Download one combined results ZIP
The ZIP contains a readable `RESULTS.html`, all CSV/JSONL outputs, raw model responses, revisions, prompts, environment records, selected inputs, errors, human-review sheets and file checksums. Extract it before opening the HTML. Send back the **whole ZIP**, not just the best-looking table.

If the automatic download is blocked by your browser, use the Colab Files panel to download the printed ZIP path. If a runtime disconnects before this cell, recover the checkpoint ZIP from Drive when backup was enabled. Do not delete failed rows or replace missing values with zero.
''')
code('''
tw.make_summary_report(METRICS, REGISTER, CONFIG, RUN)
RESULT_ZIP = tw.archive_results(RUN)
if CONFIG["DOWNLOAD_ZIP"]:
    try:
        from google.colab import files
        files.download(str(RESULT_ZIP))
    except ImportError:
        display(FileLink(str(RESULT_ZIP)))
print("Keep this run ID:", RUN.name)
''')
md('''
### Recovery cell: download partial results
Run this cell manually if you stopped a model or a later result cell failed. It does not rerun models.
''')
code('''
# Uncomment to rescue partial outputs:
# partial_zip = tw.archive_results(RUN)
# from google.colab import files
# files.download(str(partial_zip))
''')
md('''
## Sources and next decision
This notebook implements the project's [Shoaib work package](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/blob/main/POC/benchmarks/D_decisions.md) and shared protocol. Publisher documentation was checked on 27 October 2026; each run records actual model revisions and installed versions.

- Dataset: [author LEVIR-MCI release](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI), [LEVIR-CC authors](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset).
- Native decisions: [d3-mini model/readout](https://huggingface.co/vllm-sr/d3-mini), [Kev model](https://huggingface.co/jaredpalmer/kev-4b), [Kev Linux server](https://github.com/jaredpalmer/kev), [Clef-flash](https://ollama.com/library/clef-flash), [Clef](https://ollama.com/library/clef).
- General judges: [Qwen3 text](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [InternVL3.5](https://huggingface.co/OpenGVLab/InternVL3_5-4B).
- Runtime: [Colab resource limits](https://research.google.com/colaboratory/faq.html).

**Next decision after reviewing the ZIP:** fix real load/format failures, pilot the independent annotations, then retain at most two finalists. Add the team's frozen detector/captioner/witness evidence before claiming the checker improves the full pipeline. Broader datasets and a custom score remain separate, justified follow-up work.
''')

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},
    'colab':{'name':'TerraWatch_Shoaib_Decision_Benchmarks.ipynb','provenance':[],'gpuType':'L4'},
    'accelerator':'GPU',
    'terrawatch':{'owner':'Shoaib','initials':'SSSM','runtime_sha256':hashlib.sha256(runtime).hexdigest(),
                 'validation':'GPU model inference must be completed in the user Colab session.'}})
nbf.validate(notebook)
path=ROOT/'TerraWatch_Shoaib_Decision_Benchmarks.ipynb'
nbf.write(notebook,path)
print(f'{path} | {len(cells)} cells | {len(runtime):,} embedded runtime bytes')
