# TerraWatch

**Milestone 4 | Training, model selection and evidence review**

DS & AI Lab Project (BSDA4001) | IIT Madras | Group 7

## 1. Introduction

TerraWatch describes visible building and road changes between aligned satellite images. Milestone 3 selected BIT-ResNet18 for dense localisation, RSCaMa for captioning, an OpenCV witness for crop inspection, and Qwen3-VL for claim decisions. Training concerns BIT and RSCaMa. The visual witness uses fixed rules and the referee uses frozen weights.

## 2. Training data

The strict Milestone 2 LEVIR-MCI package contains **9,544 pairs**: 6,444 for training, 1,242 for validation and 1,858 for test. The prepared pixel shares are **91.24% background, 3.65% road and 5.11% building**. The earlier BIT pilot and RSCaMa subset retraining use distinct raw-release subsets. The M4 comparison reads the strict training and validation manifests and checks selected source bytes against the recorded SHA-256 hashes. Source annotations serve as supervision and evaluation references.

## 3. Model configuration and training

| Component | Training and selection |
|---|---|
| BIT-ResNet18 | ImageNet-initialised shared encoder, four tokens per date, one encoder layer, eight decoder layers and three pixel classes. Inputs are RGB pairs scaled to [-1,1]. Focal and soft Dice losses address change-class imbalance. |
| RSCaMa | CLIP ViT-B/32 features, temporal state-space fusion and Transformer decoder. The subset retrain builds its vocabulary from 1,313 fitting pairs, uses 50 development pairs for checkpoint selection, and trains on reference captions with token cross-entropy. |
| Witness and referee | The witness inspects crop differences, edges, lighting and alignment. The Qwen referee receives a claim, both images and structured evidence, and produces an accept, reject or defer decision. |

## 4. Training environment

The recorded local BIT pilot used an RTX 4060 Ti with 8 GB VRAM and PyTorch 2.9.0 with CUDA 12.8. The RSCaMa and Qwen workers use separate Transformers environments. Sequential subprocesses release model memory between stages. The local M4 comparison ran on CPU with PyTorch 2.15.0 and torchvision 0.29.2. Each run writes its device, package versions and resolved settings to `run.json`.

<!-- PAGEBREAK -->

## 5. Methodology, optimisation and regularisation

The documented BIT pilot keeps ResNet18 frozen for two epochs, uses batches of two and optimises the temporal layers and prediction head with AdamW at 0.0001. Weight decay is 0.01. Its equal focalÃ¢â‚¬â€œDice objective gives the road and building classes larger focal weights. The M4 comparison holds data, initial weights, batch size and augmentation constant while varying the head learning rate. It uses paired horizontal flips and selects by road/building mean IoU on validation.

RSCaMa uses Adam at 0.0001 and development BLEU-4 for checkpoint selection in the recorded subset protocol. A training-only vocabulary constrains the retrain to its fitting data. Its fusion and decoder use dropout, and the image geometry stays consistent with spatial caption terms.

## 6. Experiments and checkpoint selection

| Run or benchmark | Measured result | Selection role |
|---|---|---|
| BIT pilot, epoch 1 | 47.1% validation all-class macro-IoU | Best of the two logged pilot epochs |
| BIT pilot, epoch 2 | 44.6% validation all-class macro-IoU | Lower than epoch 1 by 2.5 points |
| BIT vs Otsu, matched 100-pair screen | 45.8% vs 16.7% binary F1 | Detector pilot comparison. BIT false alarms were 31/45, versus 45/45 for Otsu. |
| Released vs subset-trained RSCaMa, local 100-pair screen | 78.69% vs 76.04% strict automatic claim precision. BLEU-4 of 0.6223 vs 0.5793. | The released checkpoint leads both measures on this shared screen. |

The BIT pilot's validation score peaks at epoch 1 within the observed two epochs. Its evaluation road/building mean IoU is 26.8%. Binary IoU of 29.7% compares with 9.1% for Otsu on the same binary target. The captioning comparison measures automatic reference and mask agreement. The released RSCaMa checkpoint retains its pinned revision and matching vocabulary.

<!-- M4_COMPARISON_BEGIN -->

### Matched learning-rate comparison

Both arms used 24 strict training pairs and 16 strict validation pairs on CPU. The comparison completed 3 common epochs with identical initial weights, batch size and paired flips.

| Head learning rate | Best epoch | Validation change-class mean IoU |
|---|---:|---:|
| 1e-4 | 3 | 16.17% |
| 3e-4 | 3 | 12.89% |

The comparison selects learning rate 1e-4 at epoch 3, with 16.17% validation change-class mean IoU. The higher rate reached a lower validation loss but lower change-class IoU at epoch 3. Selection therefore follows foreground overlap. Both curves cover three epochs and describe the early fit.

![Matched BIT train loss, validation loss and validation change-class mean IoU](evidence/bit_comparison/learning_curves.png)

<!-- M4_COMPARISON_END -->

<!-- PAGEBREAK -->

## 7. Challenges and trade-offs

Most pixels represent background, so the detector reports road/building overlap and binary change metrics alongside all-class scores. The measured false alarms motivate crop-level inspection and explicit deferral. The referee score is conditional on the three decision tokens. The acceptance policy uses its operating threshold and defers exact counts when independent instance evidence is absent.

## 8. Evidence review and demonstration

The review interface imports the coordinator's claim records and displays both source images, the detector mask, marked regions, corresponding crops, witness observations and the recorded decision. Source image hashes are checked before display. Records link to the report by its SHA-256 and the claim ID. Reviewers can flag a claim, revise its wording or count, add a reason and record a decision. SQLite retains successive review events and the original pipeline outcome. The viewer exports claim records and their review history as JSON.

| LEVIR-MCI pair | Claim used in walkthrough | Recorded path |
|---|---|---|
| train_000010 | Buildings appeared at the bottom. | An Otsu mask and authored claim go through the coordinator. The decision control defers. A scripted review accepts the claim and retains the original deferral in the audit. |
| val_000005 | Two houses were built in the centre. | The same control pipeline preserves the exact-count review reason and defers the claim. |

The replay exercises the evidence and review workflow on real satellite imagery. A second replay runs the selected BIT checkpoint on both pairs and records its predicted masks and crops. Its authored claims and decision control preserve the walkthrough conditions. The learned captioner and referee use the same worker contract. The review replay retrieved ten image assets and confirmed two successive review events after reopening the event store.

## 9. Milestone 5 handover

The evaluation entry points support checkpoint-specific detector and caption assessment. The evidence viewer provides a separate record of human decisions and corrections, which can be linked to the same run and claim identifiers for the M5 audit. Controlled training uses validation for selection and keeps the test partition outside the learning-rate comparison.

## References

1. [Milestone 2 data preparation and audited cohort](../../reports/Milestone2_Report.md).
2. [Milestone 3 model selection and architecture](../../reports/Milestone3_Report.md).
3. [Historical detector and caption training evidence](../milestone3/sarthak/Benchmark_Report.md).
4. [Model setup and reproducibility](../../docs/models/README.md).
5. [Milestone 4 run instructions and code walkthrough](Walkthrough.md).
