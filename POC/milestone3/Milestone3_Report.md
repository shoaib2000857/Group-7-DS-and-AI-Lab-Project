# TerraWatch: Intelligent Satellite Image Change Detection and Analysis

**DS & AI Lab Project (BSDA4001) | Indian Institute of Technology Madras**

**Milestone 3: Model Selection and System Architecture | Group 7 | November 2026**

<p align="center"><img src="assets/iitm_logo.png" alt="IIT Madras emblem" width="130"></p>

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

![System architecture and reference separation](assets/system_architecture.png)

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

![Model pipeline and guarded output](assets/workflow.png)

Figure 2. The operating pipeline retains intermediate evidence and raw decisions. Report composition uses accepted claim spans and carries review requirements into the final output.

<!-- PAGEBREAK -->

## 4. Model Architecture Selection

| Module | Selected architecture | Size, representation and integration |
|---|---|---|
| A - change detection | BIT with a shared ResNet18 backbone and three-class output head [2] | Four semantic tokens per temporal stream; one token-encoder layer and eight decoder layers. Class indices: 0 background, 1 road, 2 building. |
| B - change description | RSCaMa with CLIP ViT-B/32, spatial/temporal state-space fusion and a Transformer language decoder [3] | 7 x 7 patch features, 768-dimensional fusion, three fusion layers, eight attention heads and one decoder layer; greedy output capped at 42 tokens. |
| C - visual evidence | Custom crop and structural witness | Parameter-free OpenCV processing: LAB luminance mean shift, Sobel edges, rectilinear orientation, local anomalies, visibility and registration diagnostics. |
| D - claim referee | Qwen3-VL-4B-Instruct image-backed judge; Qwen3-4B-Instruct-2507 shared-evidence comparator [4,5] | Approximately 4B language-model scale; original images plus structured evidence for VL, evidence text for the text comparator. Closed accept/reject/defer outputs. |

![Detector and captioner representations](assets/model_architectures.png)

Figure 3. BIT predicts dense class targets. RSCaMa translates paired visual features into text. These modules remain independently replaceable and trainable.

BIT is the detector architecture selection, justified by its compact shared backbone and temporal context mechanism. Detector training uses ImageNet visual initialisation and a new three-class task head. RSCaMa's tested interface consumes two RGB images; detector masks enter the downstream claim-localisation and verification path. This distinction preserves the actual captioner input contract.

The selected captioning comparison includes the released RSCaMa checkpoint and a subset-trained RSCaMa implementation initialised from public CLIP weights. Both referees use the exact checkpoint revisions recorded in the supplied run metadata.

<!-- PAGEBREAK -->

## 5. Justification and Experimental Evidence

The captioning screen compares a remote-sensing specialist with general vision-language models and deterministic controls. The table uses the **100-pair LEVIR screening population** and automatic mask/reference checks. Strict precision includes all factual claims in its denominator; in-scope precision restricts the evaluated claim population to the supported annotation scope.

| Captioning configuration | Strict precision | In-scope precision | BLEU-4 |
|---|---:|---:|---:|
| Released RSCaMa | 78.7% | 88.9% | 0.6223 |
| RSCaMa, 20%-budget retrain | 79.2% | 88.0% | 0.5786 |
| Qwen3-VL-4B | 58.3% | 72.9% | 0.0267 |
| SmolVLM2-2.2B | 38.6% | 48.4% | 0.0102 |
| Pixel-difference template | 69.0% | 69.0% | 0.0000 |
| Always no change | 40.0% | 40.0% | 0.2798 |

The specialist provides the strongest overall captioning baseline in this screen. On Apple M4/MPS at batch one, median caption latency is 0.139 s for RSCaMa and 7.605 s for Qwen3-VL. The two RSCaMa precision intervals overlap; their difference does not establish an improvement from retraining. The released checkpoint uses the original train split and validation-based checkpoint selection. The controlled retrain uses **1,313 train pairs plus 50 development pairs**, separating that comparison from the released caption checkpoint.

The transfer screens use 100 records each from SECOND-CC and WHU-CDC. Released RSCaMa strict precision is **48.0%** and **55.6%**, respectively. This source dependence supports explicit evidence checking and source-specific evaluation. WHU remains a historical transfer screen with its separate usage-rights disposition.

| Referee configuration | Completed proxy claims | Raw decision agreement |
|---|---:|---:|
| Qwen3 text, common evidence | 150/150 | 41.3% |
| Qwen3-VL, common evidence | 150/150 | 45.3% |
| Qwen3-VL, image-backed | 150/150 | 56.7% |

The referee figures use **30 validation pairs**, constructed annotation-proxy claims and raw choices before the 0.8 accept gate. Image-backed Qwen3-VL is the selected visual-referee configuration; the text model supplies a controlled shared-evidence comparison. These measurements are distinct from independently human-rated caption factuality.

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

Colab supplies GPU training/inference, with L4-class evidence from the returned decision sessions. Model workers use separate environments because captioner and referee library versions differ. Sequential residency and stored intermediates control memory without loading the complete model stack simultaneously.

<!-- PAGEBREAK -->

## 8. Pipeline Implementation and Verification

The pipeline entrypoint accepts two image paths and a model configuration. Separate workers execute BIT, RSCaMa and the referee; the coordinator performs claim extraction, predicted-region matching, crop inspection and final composition. Model checkpoints, upstream source directories and worker interpreters are explicit configuration fields. This keeps the architecture reproducible across Colab, WSL and local environments.

Acceptance requires a valid decision record and the configured operating gate. Missing evidence and invalid probability vectors produce review reasons. Exact counts require independent instance evidence; connected-region counts do not satisfy that requirement. Output descriptions concern visible changes and preserve uncertainty through the defer state.

| Verification evidence | Executed scope |
|---|---|
| Consolidated CPU suite | **81 passing tests**; one Torch-dependent shim module skipped in the CPU environment |
| Visual witness | Nine unit tests; supplied 50-case synthetic regression record and three real-image cases over two source pairs |
| Captioning | Six configurations; 50 development/100 screening LEVIR records; two 100-record transfer screens |
| Decision exports | Six checksum-verified ZIPs, five session IDs and 40 distinct image pairs; successful Qwen text/VL outputs and preserved failure records |
| Orchestration | CPU test doubles exercise localisation, reference isolation, decision gating, failure review and preservation of existing run directories |

The 50-case witness record reports 50/50 synthetic conditions. The real-image cases use authored claims and regions; their evidence records retain separate referee fields. Decision-session exports use a frozen Qwen visual witness. Those component protocols remain identifiable through their run IDs and are not pooled into a single accuracy score.

Human assessment uses SUPPORTED, CONTRADICTED, UNSUPPORTED and UNVERIFIABLE judgements from two independent raters. Agreement and Cohen's kappa precede third-rater adjudication using the original images and contextual crops. SUPPORTED maps to accept, CONTRADICTED/UNSUPPORTED to reject and UNVERIFIABLE to defer. This measures claim factuality separately from mask overlap, reference-caption similarity and automated proxy decisions.

The architecture fixes the complete data flow, model choices and training controls. The delivered scripts provide model-worker interfaces, detector training, evidence records and auditable output generation, supported by the consolidated experiment register.

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
