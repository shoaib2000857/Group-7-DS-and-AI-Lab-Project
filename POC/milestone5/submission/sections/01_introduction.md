## 1. Introduction

### 1.1 Project Objective

**TerraWatch** is an evidence-backed, bitemporal satellite change-detection, change-captioning, and claim-verification system designed to transform raw before-and-after aerial imagery pairs $(T_1, T_2)$ into spatially localized change masks, concise natural-language change descriptions, and auditable per-claim verification verdicts (`accept`, `reject`, or `defer`). Remote-sensing analysts monitoring urban expansion, road construction, and land-use change face two complementary bottlenecks: pixel-level change segmentation masks are precise but slow to summarize across large regions, whereas neural change-captioning models generate fluent descriptions that can hallucinate nonexistent buildings or roads, miss small structural changes, or state unverifiable object counts.

TerraWatch resolves this tension by coupling three modular stages under a single frozen-checkpoint provenance contract:

1. **Spatial Change Detection (`D03_bit_lr3e4`)** — a 3-class Bitemporal Image Transformer (BIT-ResNet18) predicting per-pixel background, road-change, and building-change masks at $256 \times 256$ resolution.
2. **Bitemporal Change Captioning (`C06_rscama_frozen_clip`)** — an RSCaMa state-space captioner combining a frozen CLIP ViT-B/32 visual encoder, Mamba-based spatial-temporal fusion (accelerated by an exact fused Triton selective-scan kernel), and a Transformer decoder trained on a strictly train-only 388-token vocabulary.
3. **Calibrated Mask-First Hybrid Claim Verification** — an atomic claim parser paired with a deterministic OpenCV structural witness (LAB luminance, Sobel edge density, block anomaly, SSIM, alignment jitter) and selective escalation to a frozen `Qwen3-VL-4B-Instruct` visual-language referee, surfacing every spatial mask, contextual crop, trigger code, and verdict inside an immutable Evidence Review Web UI.

### 1.2 Scope of Evaluation

Milestone 5 executes a **strictly locked, held-out evaluation** of all frozen Milestone 4 checkpoints without retraining, re-tuning, or post-hoc model selection:

- **In-Domain Held-Out Benchmark:** Full evaluation on all **1,858 held-out LEVIR-MCI Test pairs** (934 changed, 924 unchanged; 9,290 human reference captions; 121.77 million pixels), strictly unseen during training and Milestone 4 validation selection.
- **Cross-Dataset Zero-Shot Domain Transfer:** Evaluation on **SECOND-CC-AUG** (530 Validation + 1,142 Test pairs) with a complete 6-class semantic ontology dissection (`building`, `non_vegetated_ground`, `low_vegetation`, `tree`, `water`, `playground`) quantifying the exact ontology gap between LEVIR-MCI's 2-class supervision and multi-class land-cover benchmarks.
- **Extended Perturbation Stress Testing:** A 17-condition held-out Test corruption suite spanning radiometric shifts, optical blur, JPEG compression, additive Gaussian noise, sub-pixel/multi-pixel misregistration shifts, in-plane rotations, synthetic cloud haze, cross-sensor radiometric mismatch, and $0.5\times$ spatial downsampling, plus a 200-pair captioner robustness probe.
- **Blind End-to-End Verification & Human Groundedness Audit:** A 150-pair stratified held-out Test verification audit comparing seven verification policies (Unverified, Witness-Only, Mask-Only, Direct Qwen3-VL, 4-View Reinspection-All, Milestone 4 Triggered Reinspection, and the Validation-locked Milestone 5 Calibrated Mask-First Hybrid Verifier), paired with a blind dual-rater human-annotation study (`M5T_001`–`M5T_120`, 120 claims, Cohen's $\kappa$, and Wilson 95% confidence intervals).
- **Cross-Platform Computational Profiling:** End-to-end training cost accounting across all 33 Milestone 4 runs (16.63 GPU-hours) and batch-1 / batch-16 inference profiling across four compute platforms (NVIDIA RTX 4060 Laptop GPU, Google Colab Tesla T4, Apple M4 MPS, and Apple M4 CPU).

### 1.3 Models Evaluated

| Module | Model / Run ID | Architecture / Role | Selection Status |
|---|---|---|---|
| Change Detector | `D03_bit_lr3e4` | BIT-ResNet18 (11.94M params) | **Selected Production Detector** |
| Change Detector | `A03_siam_r18_unet` | Siamese ResNet18-UNet (15.63M) | Clean-Domain Challenger Baseline |
| Change Detector | `A01_ef_unet` | Early-Fusion UNet (7.76M) | Architectural Baseline |
| Change Detector | `A02_fc_siam_diff` | FC-Siam-Diff (7.76M) | Architectural Baseline |
| Change Detector | `A04_changeformer` | ChangeFormer-B2 (41.04M) | Transformer Baseline |
| Change Detector | `otsu_diff`, `always_background` | Non-learning pixel controls | Lower-Bound Sanity Controls |
| Change Captioner | `C06_rscama_frozen_clip` | RSCaMa + Frozen CLIP ViT-B/32 (898.81M total / 245.53M active) | **Selected Production Captioner** |
| Change Captioner | `C00_rscama_released` | Upstream Released RSCaMa (473-word vocab) | External Published Benchmark |
| Change Captioner | `C04_rscama_adamw_cos` | RSCaMa Unfrozen CLIP (AdamW + Cosine) | Controlled Fine-Tuning Baseline |
| Change Captioner | `C01_rscama_ref` | RSCaMa Reference (Adam + Constant LR) | Upstream Recipe Baseline |
| Change Captioner | `bit_mask_template`, `always_no_change` | Deterministic mask template & constant baseline | Groundedness & Trivality Controls |
| Claim Verifier | Calibrated Mask-First Hybrid (`D03` + OpenCV Witness + `Qwen3-VL-4B-Instruct`) | Selective VLM escalation ($\tau_A=0.60, \tau_R=0.75$, Val-locked) | **Selected Production Verifier** |
| Claim Verifier | 6 alternative verification policies | Unverified, Mask-Only, Witness-Only, Direct Qwen3-VL, 4-View All, M4 Triggered | Controlled Verification Ablations |

### 1.4 Objectives of Milestone 5

1. **Establish unbiased held-out Test generalization** with 1,000-resample bootstrap 95% confidence intervals and paired Wilcoxon / McNemar significance tests across every detector, captioner, and verification policy.
2. **Validate the Milestone 5 Calibrated Mask-First Hybrid Verifier** locked solely on Milestone 4 Validation data, confirming simultaneous gains in accepted-claim groundedness, hallucination rejection, and VLM compute savings on blind Test claims.
3. **Quantify failure modes and robustness boundaries** through systematic error mining, a 6-category failure taxonomy, 17-condition Test stress testing, and 6-class SECOND-CC transfer dissection.
4. **Deliver deployment-grade reproducibility and operational evidence** — verified SHA-256 checkpoint registry, automated end-to-end CLI tests, cross-platform latency/VRAM/cost profiles, and authentic held-out Test Evidence Review UI traces.
