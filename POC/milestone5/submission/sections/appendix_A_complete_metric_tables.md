## Appendix A. Complete Metric Tables

### A.1 Held-Out LEVIR-MCI Test Detector Metrics (1,858 Pairs, 121.77M Pixels)

| Model | Arch | Chg mIoU (%) | 95% Bootstrap CI | Road IoU (%) | Bldg IoU (%) | Bin F1 (%) | 95% Bootstrap CI | Small (<1%) F1 (%) | Pixel Acc (%) | ROC-AUC | PR-AUC (AP) | FA@16px (%) | FA@50px (%) | FA@200px (%) |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **`D03_bit_lr3e4` (Selected)** | `bit_resnet18` | **75.93** | `[74.79, 77.04]` | 74.53 | 77.33 | **86.83** | `[86.18, 87.42]` | 48.88 | 97.64 | 0.9901 | 0.9361 | 11.80 | 10.28 | 5.74 |
| `A03_siam_r18_unet` | `siam_resnet18_unet` | 76.73 | `[75.55, 77.84]` | 74.94 | 78.52 | 87.49 | `[86.84, 88.09]` | 49.10 | 97.75 | 0.9909 | 0.9411 | 9.09 | 8.23 | 5.09 |
| `A01_ef_unet` | `ef_unet` | **77.36** | `[76.19, 78.45]` | **75.62** | **79.11** | **87.73** | `[87.07, 88.30]` | 51.17 | **97.80** | **0.9913** | **0.9434** | 8.87 | 7.68 | 5.63 |
| `A02_fc_siam_diff` | `fc_siam_diff` | 74.48 | `[73.03, 75.83]` | 72.71 | 76.25 | 85.81 | `[84.96, 86.59]` | **54.77** | 97.59 | 0.9890 | 0.9335 | **5.84** | **4.55** | **3.14** |
| `A04_changeformer` | `changeformer` | 74.38 | `[73.08, 75.55]` | 72.26 | 76.50 | 85.99 | `[85.23, 86.69]` | 46.63 | 97.48 | 0.9889 | 0.9274 | 11.04 | 9.52 | 6.82 |
| `otsu_diff` (Control) | Non-learning | n/a | — | — | — | 14.20 | — | — | 63.65 | 0.5559 | 0.1148 | — | — | — |
| `always_background` (Control) | Constant 0 | 0.00 | — | 0.00 | 0.00 | 0.00 | — | 0.00 | 91.06 | 0.5000 | — | 0.00 | 0.00 | 0.00 |

### A.2 Paired Statistical Significance Tests vs `D03_bit_lr3e4` (1,858 Held-Out Test Pairs)

| Comparison (`D03` vs Baseline) | Mean Per-Pair F1 Diff | Wilcoxon $p$-value | Unchanged FA@50 Discordant (`D03` Better / Other Better) | McNemar $p$-value |
|---|---:|---:|---:|---:|
| `D03` vs `A03_siam_r18_unet` | -0.0173 | $2.82 \times 10^{-16}$ | 35 / 54 | $0.0558$ (n.s.) |
| `D03` vs `A01_ef_unet` | -0.0147 | $1.77 \times 10^{-19}$ | 27 / 51 | $0.0088$ |
| `D03` vs `A02_fc_siam_diff` | -0.0171 | $0.0883$ (n.s.) | 10 / 63 | $1.55 \times 10^{-10}$ |
| `D03` vs `A04_changeformer` | +0.0056 | $0.2245$ (n.s.) | 45 / 52 | $0.5426$ (n.s.) |

### A.3 Area-Stratified Held-Out Test Detector Performance (`D03_bit_lr3e4`)

| Change-Area Stratum | Test Pairs | `D03` Binary F1 (%) | `D03` Change mIoU (%) | Operational Note |
|---|---:|---:|---:|---|
| `unchanged_0` (0 px) | 924 | n/a (`FA@50 = 10.28%`) | n/a | 89.72% specificity at 50 px threshold |
| `tiny_1_250` (1–250 px) | 20 | 12.99 | 8.85 | Sub-0.38% tile area; primary FN regime |
| `small_251_1000` (251–1,000 px) | 69 | 62.92 | 31.59 | Partial small structures / narrow paths |
| `medium_1001_4000` (1k–4k px) | 186 | 73.80 | 54.79 | Individual buildings / short road segments |
| `large_gt_4000` (>4,000 px) | 659 | **87.68** | **77.26** | Multi-building & highway expansions |

### A.4 Held-Out LEVIR-MCI Test Captioner Complete Metrics & 95% Bootstrap CIs (1,858 Pairs)

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Chg B4 | ROUGE-L [95% CI] | METEOR | CIDEr-D [95% CI] | BERTScore-F1 [95% CI] | NC FA (%) | Miss (%) | ObjHall (%) |
|---|---:|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|
| **`C06_rscama_frozen_clip`** | 0.8421 | 0.7543 | 0.6778 | **0.6126** | 0.3686 | `0.7324 [0.718, 0.746]` | 0.7360 | `1.3377 [1.299, 1.376]` | `0.9490 [0.948, 0.950]` | **3.57** | 11.78 | **4.93** |
| `C00_rscama_released` | 0.8613 | 0.7819 | 0.7124 | 0.6551 | 0.4004 | `0.7556 [0.742, 0.768]` | 0.7560 | `1.3930 [1.355, 1.429]` | `0.9514 [0.950, 0.953]` | 1.19 | 12.31 | 2.73 |
| `C04_rscama_adamw_cos` | 0.8504 | 0.7614 | 0.6814 | 0.6101 | 0.4077 | `0.7381 [0.725, 0.751]` | 0.7430 | `1.3530 [1.315, 1.389]` | `0.9507 [0.949, 0.952]` | 5.74 | 6.53 | 6.22 |
| `C01_rscama_ref` | 0.8359 | 0.7492 | 0.6757 | 0.6135 | 0.3759 | `0.7152 [0.701, 0.730]` | 0.7199 | `1.2819 [1.240, 1.322]` | `0.9460 [0.945, 0.947]` | 9.42 | 5.14 | 8.93 |
| `bit_mask_template` | 0.7529 | 0.6314 | 0.5478 | 0.4893 | 0.1613 | `0.6427 [0.627, 0.658]` | 0.6435 | `1.0734 [1.032, 1.115]` | `0.9398 [0.938, 0.941]` | 10.28 | 3.00 | 11.66 |
| `always_no_change` | 0.6414 | 0.5615 | 0.5269 | 0.5101 | 0.0001 | `0.6019 [0.582, 0.621]` | 0.5623 | `1.0130 [0.963, 1.061]` | `0.9291 [0.927, 0.931]` | 0.00 | 100.00 | 0.00 |

### A.5 Blind Held-Out Test Verification Route Breakdown (185 Claims, 150 Pairs)

| M5 Hybrid Verifier Route | Claim Count | Share (%) | VLM Used? |
|---|---:|---:|---|
| `mask_first_accept` | 120 | 64.86% | No |
| `mask_first_reject` | 24 | 12.97% | No |
| `deterministic_defer_ontology_or_exact_count` | 23 | 12.43% | No |
| `vlm_escalate_no_change_residual_or_hedged -> accept` | 10 | 5.41% | Yes |
| `vlm_escalate_no_change_residual_or_hedged -> defer` | 4 | 2.16% | Yes |
| `vlm_escalate_no_change_residual_or_hedged -> reject` | 1 | 0.54% | Yes |
| `vlm_escalate_weak_mask_support -> accept` | 2 | 1.08% | Yes |
| `vlm_escalate_weak_mask_support -> defer` | 1 | 0.54% | Yes |
| **Total** | **185** | **100.00%** | **18 / 185 (9.73%)** |
