# TerraWatch: Intelligent Satellite Image Change Detection and Analysis

**DS & AI Lab Project (BSDA4001) | Indian Institute of Technology Madras**

**Milestone 3: Model Selection and System Architecture | Group 7 | November 2026**

**Local benchmark validation edition | Local evaluation by Sarthak Saini**

<p align="center"><img src="figures/iitm_logo.png" alt="IIT Madras emblem" width="130"></p>

| Team member | Student email |
|---|---|
| Priyanshu Agarwal | 23f2003488@ds.study.iitm.ac.in |
| Sarthak Saini | 23f2005347@ds.study.iitm.ac.in |
| Sujal Raj Pradhan | 23f2004759@ds.study.iitm.ac.in |
| Shoaib SSM | 24f2000857@ds.study.iitm.ac.in |

## 1. Introduction

TerraWatch identifies visible building and road changes between two satellite images and produces descriptions linked to visual evidence. Milestone 3 establishes the model choices, module interfaces, training configuration and complete input-to-output architecture [1]. The system combines pixel-level detection, domain-specific captioning, structured claims, crop-based evidence and an accept/reject/defer referee.

The data contract uses the Milestone 2 strict LEVIR-MCI package: **9,544 pairs**, partitioned into **6,444 training, 1,242 validation and 1,858 test records**. Source masks and five reference captions support supervised learning and evaluation. Inference consumes ordered images and model-generated intermediates.

## 2. Overall System Architecture

![System architecture and reference separation](figures/m3_system_architecture.png)

Figure 1. Detection and captioning operate on the same image pair. Predicted regions and caption claims meet at the evidence stage. Evaluation references occupy a separate store.

| Layer | Technology and role |
|---|---|
| Data and persistence | Python, NumPy, Pillow; immutable image sources, split manifests, JSON/JSONL records and file hashes |
| Detection and description | PyTorch; BIT three-class detector and RSCaMa captioner in dedicated model environments |
| Evidence and decisions | OpenCV, Pydantic and isolated Transformers workers; explicit coordinates, typed records and decision gates |
| Execution and interaction | Colab/WSL/local CLI; two ordered image paths and a configuration file produce a JSON report and crop evidence |

Model weights are acquired from their author repositories or Hugging Face. The inference design runs through local workers with cached weights. Evidence retrieval uses run IDs and file paths; the task does not require a vector database or document-retrieval stage.

<!-- PAGEBREAK -->

## 3. End-to-End Workflow

| Stage | Input | Output and control |
|---|---|---|
| Validate the pair | Earlier T1 and later T2 RGB images | Matching dimensions, temporal-order declaration and SHA-256 provenance; incompatible pairs raise an input error |
| Detect changes | Normalised temporal tensors | Background/road/building mask; nearest-neighbour restoration to original image dimensions |
| Generate and parse | Full-frame images; captioner output | Caption plus atomic object/action/location/quantity claims, source spans and parser provenance |
| Localise and inspect | Claim, predicted class mask and original images | Matching connected region, padded T1/T2 crops, transform record, visibility flags and structural observations |
| Referee and gate | Claim, fallible evidence and original images | Raw accept/reject/defer decision; probability validity and operating-threshold checks |
| Compose and store | Gated claim records | Accepted-description text, review flag, per-claim reasons, raw responses and auditable report JSON |

Connected components identify candidate regions; they are not building-instance counts. Claim locations use the image-relative 3 x 3 grid. Matching regions retain original-T1 half-open pixel coordinates. Unlocalised claims use full-image context and receive a review reason.

The implementation executes each learned stage in a separate process, releases its memory on exit and records request, response and log files. A new run directory preserves earlier deliveries. Worker timeouts and missing referee outputs yield explicit error/review records. Insufficient visibility, an unsupported legal inference or an exact count without independent instance evidence produces a defer decision.

![Model pipeline and guarded output](figures/m3_workflow.png)

Figure 2. The operating pipeline retains intermediate evidence and raw decisions. Report composition uses accepted claim spans and carries review requirements into the final output.

<!-- PAGEBREAK -->

## 4. Model Architecture Selection

| Module | Selected architecture | Size, representation and integration |
|---|---|---|
| A - change detection | BIT with a shared ResNet18 backbone and three-class output head [2] | Four semantic tokens per temporal stream; one token-encoder layer and eight decoder layers. Class indices: 0 background, 1 road, 2 building. |
| B - change description | RSCaMa with CLIP ViT-B/32, spatial/temporal state-space fusion and a Transformer language decoder [3] | 7 x 7 patch features, 768-dimensional fusion, three fusion layers, eight attention heads and one decoder layer; greedy output capped at 42 tokens. |
| C - visual evidence | Custom crop and structural witness | Parameter-free OpenCV processing: LAB luminance mean shift, Sobel edges, rectilinear orientation, local anomalies, visibility and registration diagnostics. |
| D - claim referee | Qwen3-VL-4B-Instruct image-backed judge; Qwen3-4B-Instruct-2507 shared-evidence comparator [4,5] | Approximately 4B language-model scale; original images plus structured evidence for VL, evidence text for the text comparator. Closed accept/reject/defer outputs. |

![Detector and captioner representations](figures/m3_model_architectures.png)

Figure 3. BIT predicts dense class targets. RSCaMa translates paired visual features into text. These modules remain independently replaceable and trainable.

BIT is the detector architecture selection, justified by its compact shared backbone and temporal context mechanism. Detector training uses ImageNet visual initialisation and a new three-class task head. RSCaMa's tested interface consumes two RGB images; detector masks enter the downstream claim-localisation and verification path. This distinction preserves the actual captioner input contract.

The selected captioning comparison includes the released RSCaMa checkpoint and a subset-trained RSCaMa implementation initialised from public CLIP weights. Both referees use the exact checkpoint revisions recorded in the supplied run metadata.

<!-- PAGEBREAK -->

## 5. Justification and Experimental Evidence

The captioning screen compares a remote-sensing specialist with general vision-language models and deterministic controls. The table uses the **100-pair LEVIR screening population** and automatic mask/reference checks. Strict precision includes all factual claims in its denominator; in-scope precision restricts the evaluated claim population to the supported annotation scope.

The table below preserves the original report's values as the historical B0 results. The local evaluation reused the frozen 100-pair screen, automatic mask/reference evaluator and PTB/BLEU implementation. Differences are calculated against the stored B0 JSON at full precision, not against rounded table text. These are automated annotation-proxy scores, not human factuality ratings.

| Captioning configuration | Strict precision | In-scope precision | BLEU-4 |
|---|---:|---:|---:|
| Released RSCaMa | 78.7% | 88.9% | 0.6223 |
| RSCaMa, 20%-budget retrain | 79.2% | 88.0% | 0.5786 |
| Qwen3-VL-4B | 58.3% | 72.9% | 0.0267 |
| SmolVLM2-2.2B | 38.6% | 48.4% | 0.0102 |
| Pixel-difference template | 69.0% | 69.0% | 0.0000 |
| Always no change | 40.0% | 40.0% | 0.2798 |

In the original benchmark, released RSCaMa had a median caption latency of 0.139 seconds per pair and Qwen3-VL 7.605 seconds. Local timing is recorded separately because execution conditions differ. The two RSCaMa precision intervals overlap, so the original screen did not establish a reliable retraining improvement. The released checkpoint was selected using validation performance; the controlled retrain used **1,313 training pairs and 50 development pairs**.

### Local benchmark results (Sarthak)

Sarthak Saini reran all six configurations on the same 100 image pairs used in the original screen and scored them with the same claim parser and reference metrics. Each cell shows **original → local (difference)**; precision differences are percentage points (pp). The local screen came from the raw LEVIR-MCI archive rather than the cleaned 9,544-pair Milestone 2 handover set. This is a reproducibility check, not a new held-out test. Precision and recall are automatic annotation proxies, not human factuality ratings.

| Configuration | Strict precision: B0 → local (Δ) | In-scope precision: B0 → local (Δ) | BLEU-4: B0 → local (Δ) |
|---|---|---|---|
| Released RSCaMa | 78.69% → 78.69% (0.00 pp) | 88.89% → 88.89% (0.00 pp) | 0.6223 → 0.6223 (0.0000) |
| RSCaMa, 20%-budget retrain | 79.17% → 76.04% (−3.13 pp) | 87.96% → 82.02% (−5.94 pp) | 0.5786 → 0.5793 (+0.0007) |
| Qwen3-VL-4B | 58.33% → 55.70% (−2.63 pp) | 72.92% → 71.74% (−1.18 pp) | 0.0267 → 0.0183 (−0.0084) |
| SmolVLM2-2.2B | 38.58% → 37.76% (−0.82 pp) | 48.41% → 47.13% (−1.28 pp) | 0.0102 → 0.0000 (−0.0102) |
| Pixel-difference template | 69.00% → 69.00% (0.00 pp) | 69.00% → 69.00% (0.00 pp) | 0.0000 → 0.0000 (0.0000) |
| Always no change | 40.00% → 40.00% (0.00 pp) | 40.00% → 40.00% (0.00 pp) | 0.2798 → 0.2798 (0.0000) |

**What the comparison says:** the released RSCaMa checkpoint and both controls reproduce the saved B0 aggregates exactly. The report rounds RSCaMa precision to one decimal place, so the local values (78.69% and 88.89%) display as 78.7% and 88.9%. The retrain and both general vision-language models differ from their B0 results. Keep B0 values labelled as historical and use the local column as the implementation check.

![Historical aggregate compared with local caption evaluation](../POC/milestone3/sarthak/figures/historical-local-exact-parity.png)

Figure 4. Each marker compares one model's historical aggregate with the local result. The dashed diagonal marks exact parity.

![Display-rounding differences for exact matches](../POC/milestone3/sarthak/figures/m3-displayed-rounding-delta.png)

Figure 5. The released RSCaMa display differs by 0.01 percentage points because of one-decimal rounding; both controls and BLEU-4 match exactly.

![Additional local RSCaMa metrics and bootstrap intervals](../POC/milestone3/sarthak/figures/rscama-supplementary-evidence.png)

Figure 6. Local caption-overlap metrics and pair-bootstrap confidence intervals for the released RSCaMa rerun.

The local evaluation also calculated caption and claim metrics that were not included in the original summary table:

| Configuration | BLEU-1 | BLEU-2 | BLEU-3 | ROUGE-L | CIDEr-D | Fact recall | Unchanged-pair false alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Released RSCaMa | 0.8412 | 0.7548 | 0.6813 | 0.7464 | 1.3184 | 76.8% | 0.0% |
| RSCaMa, 20%-budget retrain | 0.7872 | 0.7087 | 0.6349 | 0.6810 | 1.0837 | 58.4% | 12.5% |
| Qwen3-VL-4B | 0.1739 | 0.0850 | 0.0380 | 0.1740 | 0.0003 | 72.8% | 27.5% |
| SmolVLM2-2.2B | 0.1513 | 0.0744 | 0.0220 | 0.1612 | 0.0022 | 56.0% | 82.5% |
| Pixel-difference template | 0.2458 | 0.1130 | 0.0246 | 0.1785 | 0.0426 | 76.0% | 22.5% |
| Always no change | 0.3133 | 0.2905 | 0.2833 | 0.4650 | 0.9370 | 32.0% | 0.0% |

These are automatic reference and mask checks. They help compare runs, but they do not replace a human review of caption factuality. The no-change control has relatively high text-overlap scores despite weak claim precision, showing why BLEU-4 alone cannot evaluate a captioner.

The run records preserve the model revision, data manifest, raw captions and evaluation summary. The old B0 archive contains aggregate scores but not the original captions, so we cannot compare old and new predictions pair by pair. We can confirm that their overall scores match where shown; we cannot claim the captions themselves were identical. The released RSCaMa vocabulary rebuild reads captions from official splits, including evaluation references. That upstream behavior was repeated for this parity check, so treat the score as a reproduction of the prior screen, not as a fresh leakage-free test.

The transfer screens use 100 records each from SECOND-CC and WHU-CDC. The original report prints released RSCaMa strict precision as **48.0%** and **55.6%**, respectively. Local reruns exactly reproduce the full B0 aggregates: SECOND-CC strict/in-scope/BLEU-4 **48.04% / 50.00% / 0.2426**; WHU-CDC **55.56% / 66.67% / 0.4717**. This source dependence supports explicit evidence checking and source-specific evaluation. WHU retains its separate usage-rights caveat.

| Referee configuration | Historical proxy claims | Historical agreement | Local status |
|---|---:|---:|---|
| Qwen3 text, common evidence | 150/150 | 41.3% | Local aggregate not completed |
| Qwen3-VL, common evidence | 150/150 | 45.3% | Local aggregate not completed |
| Qwen3-VL, image-backed | 150/150 | 56.7% | Local aggregate not completed |

The original referee figures use **30 validation pairs**, constructed annotation-proxy claims and raw choices before the 0.8 accept gate. Image-backed Qwen3-VL is the selected visual-referee configuration; the text model supplies a controlled shared-evidence comparison. These measurements are distinct from independently human-rated caption factuality. The local referee dataset and runner are prepared, but no local agreement aggregate is included in this edition; retain these three figures as historical until that run completes.

BIT complements these choices through dense localisation and a smaller shared backbone; detector selection is architectural rather than a claimed segmentation leaderboard result. Alternative transformer detectors and Siamese CNNs remain represented in the benchmark registry. The selected stack balances domain-specific generation, explicit visual checks and serial GPU execution.

<!-- PAGEBREAK -->

## 6. Model Inputs and Outputs

| Contract | Representation |
|---|---|
| Detector input/output | Paired B x 2 x 3 x 256 x 256 tensors; BIT scaling to [-1,1]; B x 3 x 256 x 256 logits and categorical mask |
| Captioner features | Author CLIP processor: 224-pixel RGB views, CLIP mean/std; temporal patch features and checkpoint-matched word vocabulary |
| Structured claim | Object, action, optional exact quantity, image-relative location and exact caption source span |
| Evidence | Original-frame box, padded crops, transform metadata, mask/witness producer IDs, quality flags and fallible observations |
| Referee/report | Checkpoint chat template, closed decisions, raw response, probability semantics, gated outcome, reason codes and accepted-description text |

The RSCaMa retrain builds its vocabulary from training-subset captions and preserves start/end/padding tokens. The released checkpoint uses its matching vocabulary. Qwen uses the publisher tokenizer/processor and temporal-order prompt. Reference captions, human labels and source masks are excluded from inference requests.

## 7. Training Strategy

| Component | Training configuration and checkpoint rule |
|---|---|
| BIT task training | ImageNet visual initialisation; backbone frozen for five epochs, then trained at 1e-5; temporal/head parameters at 1e-4. AdamW, weight decay 0.01; cosine schedule, 50-epoch cap; batch eight with four-step accumulation. |
| Detector loss/selection | Equal mixture of class-weighted focal loss (gamma 2; background/road/building weights 0.10/0.45/0.45) and soft Dice. Validation macro-IoU selects checkpoints; patience eight. Paired detector flips preserve masks. |
| RSCaMa retraining | Public CLIP visual initialisation; temporal fusion and caption decoder trained through the upstream trainability policy. Token cross-entropy, Adam 1e-4, batch 12 pairs, 40-epoch cap, patience eight and seed zero. Development BLEU-4 selects best weights. |
| Witness and referee | Structural witness uses fixed image-processing rules. Qwen weights remain frozen; prompt, image budget, model revision and the 0.8 operating gate are recorded. |

The detector row specifies the executable task-training configuration. The captioner row records the implemented subset-retraining script. Training and validation retain their fixed source partitions; test records are reserved for evaluation. Checkpoint files and histories preserve model configuration, epoch and selection score.

Training and inference can run in Colab or in local environments. Model workers use separate environments because captioner and referee library versions differ. Sequential loading and stored intermediates avoid keeping the complete model stack active at once.

<!-- PAGEBREAK -->

## 8. Pipeline Implementation and Verification

The pipeline entrypoint accepts two image paths and a model configuration. Separate workers execute BIT, RSCaMa and the referee; the coordinator performs claim extraction, predicted-region matching, crop inspection and final composition. Model checkpoints, upstream source directories and worker interpreters are explicit configuration fields. This keeps the architecture reproducible across Colab, WSL and local environments.

Acceptance requires a valid decision record and the configured operating gate. Missing evidence and invalid probability vectors produce review reasons. Exact counts require independent instance evidence; connected-region counts do not satisfy that requirement. Output descriptions concern visible changes and preserve uncertainty through the defer state.

| Verification evidence | Executed scope |
|---|---|
| Consolidated CPU suite | Original report: 81 passed, one Torch-dependent shim skipped. Local suite: **83 passed, zero skipped** |
| Visual witness | Original record: 50 synthetic cases and three real-image demonstrations. Local check: 50/50 synthetic outcomes; 2/3 coarse mask-region presence checks matched |
| Captioning | Six configurations rerun locally on the 100-pair screen; retrain, Qwen3-VL and SmolVLM2 differ from their historical aggregates as shown above |
| Transfer | SECOND-CC and WHU-CDC, 100 records each; strict/in-scope/BLEU-4 aggregates locally reproduced exactly |
| Referee | Original 150-proxy-claim agreements remain historical; no local aggregate is available in this edition |
| Orchestration | CPU test doubles exercise localisation, reference isolation, decision gating, failure review and preservation of existing run directories |

The 50-case witness record reports 50/50 synthetic conditions. The real-image cases use authored claims and regions; their evidence records retain separate referee fields. Decision-session exports use a frozen Qwen visual witness. Those component protocols remain identifiable through their run IDs and are not pooled into a single accuracy score.

Human assessment uses SUPPORTED, CONTRADICTED, UNSUPPORTED and UNVERIFIABLE judgements from two independent raters. Agreement and Cohen's kappa precede third-rater adjudication using the original images and contextual crops. SUPPORTED maps to accept, CONTRADICTED/UNSUPPORTED to reject and UNVERIFIABLE to defer. This measures claim factuality separately from mask overlap, reference-caption similarity and automated proxy decisions.

The architecture fixes the complete data flow, model choices and training controls. The delivered scripts provide model-worker interfaces, detector training, evidence records and auditable output generation, supported by the consolidated experiment register.

The local BIT result is a two-epoch pilot, not the planned converged detector benchmark: 100 training pairs, 25 validation pairs and a separate 100-pair evaluation subset. On that subset, BIT-R18 reached 45.8% binary any-change F1 and 26.8% mean road/building IoU; 31/45 unchanged pairs had a predicted change alarm. The Otsu control reached 16.7% binary F1 and 9.1% IoU, with alarms on all 45 unchanged pairs. These are pilot results and do not establish the final detector ranking.

The locally executed caption chain is not a complete learned end-to-end validation. BIT and RSCaMa were run and measured separately; the full coordinator-to-learned-referee chain was not exercised on a single real run. Referee agreement and human factuality remain unvalidated locally.

### Local evaluation and benchmark code

| Purpose | File |
|---|---|
| RSCaMa caption generation | [`caption_rscama.py`](../POC/execution/scripts/caption_rscama.py) |
| VLM caption generation (Qwen3-VL / SmolVLM2) | [`caption_vlm.py`](../POC/execution/scripts/caption_vlm.py) |
| Pixel-template and constant controls | [`controls.py`](../POC/execution/scripts/controls.py) |
| Shared automatic claim/reference and text evaluator | [`evaluate.py`](../POC/execution/scripts/evaluate.py) |
| Local caption screen scoring | [`evaluate_caption_screen.py`](../POC/milestone3/sarthak/scripts/evaluate_caption_screen.py) |
| Frozen local screen preparation | [`prepare_local_rscama_screen.py`](../POC/milestone3/sarthak/scripts/prepare_local_rscama_screen.py) |
| Local RSCaMa 20%-budget training and captioning | [`rscama_colab.py`](../POC/b-captioning/colab/rscama_colab.py) |
| BIT pilot training and scoring | [`train_bit_pilot.py`](../POC/milestone3/sarthak/scripts/train_bit_pilot.py) |
| Otsu pixel-difference baseline | [`benchmark_pixel_control.py`](../POC/milestone3/sarthak/scripts/benchmark_pixel_control.py) |
| Referee screen runner (no local aggregate yet) | [`run_local_referee_screen.py`](../POC/milestone3/sarthak/scripts/run_local_referee_screen.py) |
| Reconciliation JSON and artifact hashes | [`build_evaluation_proof.py`](../POC/milestone3/sarthak/scripts/build_evaluation_proof.py) |
| Figure generation | [`plot_evaluation_proof.py`](../POC/milestone3/sarthak/scripts/plot_evaluation_proof.py) |

The full [evaluation proof JSON](../POC/milestone3/sarthak/evaluation_proof.json) records historical and local aggregates, exact differences, data/model/run hashes and scope limitations. Raw local predictions and checkpoints are stored under the ignored `POC/execution/runs/` and `POC/execution/data/` folders; the evaluation scripts and summary tables remain in the repository. Local latency is not compared with the historical timing because the execution conditions differed.

<!-- PAGEBREAK -->

## References

1. IIT Madras DS & AI Lab. *Milestone 3: Model selection and pipeline design*. https://bsc-iitm.github.io/dsai-lab-project/milestones.html
2. Chen, H., Qi, Z. and Shi, Z. (2022). *Remote Sensing Image Change Detection with Transformers*. IEEE TGRS. https://arxiv.org/abs/2103.00208 ; official implementation: https://github.com/justchenhao/BIT_CD
3. Liu, C. et al. (2024). *RSCaMa: Remote Sensing Image Change Captioning With State Space Model*. IEEE GRSL. https://doi.org/10.1109/LGRS.2024.3404604 ; https://github.com/Chen-Yang-Liu/RSCaMa
4. Qwen Team. *Qwen3-4B-Instruct-2507: model card and configuration*. https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
5. Qwen Team. *Qwen3-VL-4B-Instruct: model card and configuration*. https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
6. TerraWatch Group 7. *Captioning implementation and aggregate screens*. Commit 8328ba7. https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/tree/8328ba7/POC/execution
7. TerraWatch Group 7. *Visual evidence inspector and component tests*. Commit 54b713a. https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/54b713a
8. TerraWatch Group 7. *Consolidated experiment register and preservation evidence*. https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/tree/poc-consolidation/POC/consolidation
