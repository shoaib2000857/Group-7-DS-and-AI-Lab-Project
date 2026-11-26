## 6. Experimental Results

All numbers in this section come from a **single locked pass over held-out LEVIR-MCI Test data** (1,858 image pairs; 121.77 million pixels; 9,290 human reference captions; 150-pair / 185-claim blind verification audit) using the frozen Milestone 4 checkpoints verified by SHA-256 (`POC/milestone5/configs/model_registry.json`).

### 6.1 Module 1 — Change Detection Results (1,858 Held-Out Test Pairs)

**Table 6.1. Held-out LEVIR-MCI Test detector performance (1,858 pairs, 1,000-resample 95% bootstrap CIs).**

| Model | Arch | Chg mIoU (%) | 95% CI | Road IoU (%) | Bldg IoU (%) | Bin F1 (%) | Small (<1%) F1 (%) | ROC-AUC | PR-AUC | FA@50px (%) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **`D03_bit_lr3e4` (Selected)** | `bit_resnet18` | **75.93** | `[74.79, 77.04]` | 74.53 | 77.33 | **86.83** | 48.88 | **0.9901** | **0.9361** | 10.28 |
| `A03_siam_r18_unet` | `siam_resnet18_unet` | 76.73 | `[75.55, 77.84]` | 74.94 | 78.52 | 87.49 | 49.10 | 0.9909 | 0.9411 | 8.23 |
| `A01_ef_unet` | `ef_unet` | **77.36** | `[76.19, 78.45]` | **75.62** | **79.11** | **87.73** | 51.17 | **0.9913** | **0.9434** | 7.68 |
| `A02_fc_siam_diff` | `fc_siam_diff` | 74.48 | `[73.03, 75.83]` | 72.71 | 76.25 | 85.81 | **54.77** | 0.9890 | 0.9335 | **4.55** |
| `A04_changeformer` | `changeformer` | 74.38 | `[73.08, 75.55]` | 72.26 | 76.50 | 85.99 | 46.63 | 0.9889 | 0.9274 | 9.52 |
| `otsu_diff` (Control) | Non-learning | n/a | — | — | — | 14.20 | — | 0.5559 | 0.1148 | — |
| `always_background` | Constant 0 | 0.00 | — | 0.00 | 0.00 | 0.00 | 0.00 | 0.5000 | — | 0.00 |

![Pixel-level ROC and Precision-Recall curves across all 5 neural detectors and Otsu pixel-difference control on 1,858 held-out LEVIR-MCI Test pairs (121.77M pixels).](figures/m5_roc_pr_curves.png)

![Held-out Test detector Binary F1 stratified by ground-truth change area (left) and unchanged-scene false-alarm operating curve across pixel thresholds 1–500 (right).](figures/m5_stratified_area_performance.png)

**Key Detector Findings:**
- **Exceeds All Target Criteria:** `D03_bit_lr3e4` achieves **`75.93%`** Change-Class mIoU (`+1.39` pts above its `74.54%` Validation score, confirming zero Validation overfitting), **`86.83%`** Binary Change F1, **`0.9901`** ROC-AUC, and **`0.9361`** PR-AUC across 121.77 million pixels.
- **Scale Stratification:** `D03` reaches **`87.68%` F1** (`77.26%` mIoU) on large changes (`>4,000 px`, 659 pairs) and **`73.80%` F1** on medium changes (`1k–4k px`, 186 pairs), tapering to `62.92%` on small (`251–1,000 px`, 69 pairs) and `12.99%` on tiny (`1–250 px`, 20 pairs) edits.
- **Clean-vs-Robust Trade-Off Validated:** Pure convolutional baselines `A01_ef_unet` (`77.36%`) and `A03_siam_r18_unet` (`76.73%`) edge `D03` by `0.80–1.43` mIoU points on pristine Test imagery, whereas `D03` decisively wins **12 of 17** held-out Test stress conditions (`+33.63` pts under heavy blur, `+12.03` pts under $0.5\times$ downsampling, `+4.57` pts under sensor noise) and SECOND-CC zero-shot transfer (`+5.46` F1 pts; Section 11).

### 6.2 Module 2 — Change Captioning Results (1,858 Held-Out Test Pairs)

**Table 6.2. Held-out LEVIR-MCI Test captioning quality, semantic similarity, and hallucination audit (1,858 pairs, 9,290 references).**

| Model | BLEU-1 | BLEU-4 | Changed BLEU-4 | ROUGE-L | METEOR | CIDEr-D | BERTScore-F1 | NC FA (%) | Changed Miss (%) | Obj Hall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`C06_rscama_frozen_clip` (Selected)** | 0.8421 | **0.6126** | 0.3686 | **0.7324** | **0.7360** | **1.3377** | **0.9490** | **3.57** | 11.78 | **4.93** |
| `C00_rscama_released` (External) | **0.8613** | **0.6551** | 0.4004 | **0.7556** | **0.7560** | **1.3930** | **0.9514** | **1.19** | 12.31 | **2.73** |
| `C04_rscama_adamw_cos` (Unfrozen) | 0.8504 | 0.6101 | **0.4077** | 0.7381 | 0.7430 | 1.3530 | 0.9507 | 5.74 | 6.53 | 6.22 |
| `C01_rscama_ref` (Upstream Recipe) | 0.8359 | 0.6135 | 0.3759 | 0.7152 | 0.7199 | 1.2819 | 0.9460 | 9.42 | 5.14 | 8.93 |
| `bit_mask_template` (Deterministic) | 0.7529 | 0.4893 | 0.1613 | 0.6427 | 0.6435 | 1.0734 | 0.9398 | 10.28 | **3.00** | 11.66 |
| `always_no_change` (Control) | 0.6414 | 0.5101 | 0.0001 | 0.6019 | 0.5623 | 1.0130 | 0.9291 | 0.00 | 100.00 | 0.00 |

![Held-out Test captioning quality across BLEU-4, ROUGE-L, METEOR, and BERTScore-F1 (left) alongside unchanged-scene hallucination, missed-change, and object-hallucination rates (right) across all 6 models.](figures/m5_captioner_semantic_comparison.png)

**Key Captioner Findings:**
- **Strong Semantic Fidelity & Lowest Hallucination Among Trained Models:** Selected `C06_rscama_frozen_clip` reaches **`BLEU-4 = 0.6126`**, **`ROUGE-L = 0.7324`** (`[0.718, 0.746]`), **`METEOR = 0.7360`**, **`CIDEr-D = 1.3377`** (`[1.299, 1.376]`), and **`BERTScore-F1 = 0.9490`** (`[0.948, 0.950]`) — exceeding every Milestone 5 captioner target while holding unchanged-scene hallucinations to **`3.57%`** (`33/924` pairs) and object hallucinations to **`4.93%`**, a **`2.64×` reduction in unchanged hallucinations** vs `C01_rscama_ref` (`9.42%`) and **`1.61×` reduction** vs `C04_rscama_adamw_cos` (`5.74%`).
- **Statistically Significant Dominance Over Non-Neural Baselines:** `C06` decisively outperforms `bit_mask_template` on ROUGE-L (`+0.0896`, Wilcoxon $p = 9.5 \times 10^{-63}$), CIDEr-D (`+0.2643`, $p = 3.3 \times 10^{-57}$), and BERTScore-F1 (`+0.0092`, $p = 3.9 \times 10^{-55}$), and outperforms `C01_rscama_ref` on BERTScore-F1 (`+0.0031`, $p = 5.7 \times 10^{-9}$).
- **Gap vs Upstream Released Checkpoint (`C00`):** `C00_rscama_released` (trained on the upstream 473-word vocabulary and longer schedule) leads `C06` by `+0.0425` BLEU-4 and `+0.0024` BERTScore-F1 (`0.9514` vs `0.9490`), demonstrating that our strictly train-only 388-word frozen-CLIP model already recovers **`99.75%`** of published semantic BERTScore similarity within 30.3 minutes on a single T4.

### 6.3 Module 3 & Overall System — Blind Held-Out Test Claim Verification (150 Pairs, 185 Claims)

Evaluating the exact Validation-locked verification rules (`tau_accept = 0.60, tau_reject = 0.75, mask_alarm_px = 50`) zero-shot on **150 stratified held-out Test pairs** (`50` unchanged, `35` building-only, `20` road-only, `25` mixed, `20` small-change; **185 extracted atomic claims**: `136` proxy-supported, `30` proxy-contradicted, `19` proxy-unverifiable) confirms complete generalization from Validation to blind Test:

**Table 6.3. Blind held-out Test comparison of all 7 claim-verification policies (150 pairs, 185 claims).**

| Verification Policy | Accept (%) | Defer (%) | Reject (%) | Accepted Precision (%) | Selective Risk (%) | Supported Recall (%) | Contradicted Catch (%) | VLM Call Rate (%) | ECE $\downarrow$ | Brier $\downarrow$ | AURC $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1. Unverified Captioner` | 100.0 | 0.0 | 0.0 | 73.51 | 26.49 | 100.00 | 0.00 | 0.0 | — | — | — |
| `3. Witness-Only Checker` | 51.9 | 10.3 | 37.8 | 73.96 | 26.04 | 52.21 | 16.67 | 0.0 | — | — | — |
| `4. Direct Qwen3-VL (τ=0.70)` | 46.0 | 43.2 | 10.8 | 78.82 | 21.18 | 49.26 | 3.33 | 100.0 | 0.3000 | 0.3142 | 0.2024 |
| `5. 4-View Reinspection-All` | 55.1 | 29.7 | 15.1 | 69.61 | 30.39 | 52.21 | 3.33 | 100.0 | 0.4464 | 0.4417 | 0.2298 |
| `6. M4 Triggered Reinspection` | 55.1 | 29.7 | 15.1 | 69.61 | 30.39 | 52.21 | 3.33 | 68.7 | — | — | — |
| `2. BIT Mask-Only Checker` | 76.8 | 10.3 | 13.0 | 93.66 | 6.34 | **97.79** | **70.00** | **0.0** | — | — | — |
| **`7. M5 Calibrated Mask-First Hybrid`** | **71.4** | **15.1** | **13.5** | **93.94** | **6.06** | **91.18** | **70.00** | **9.7** (**-90.3%**) | **0.0710** | **0.1041** | **0.0948** |

![Blind held-out Test verifier comparison (left), threshold risk-coverage sweep (center), and reliability/calibration diagram (right) across 185 claims.](figures/m5_hybrid_verifier_calibration.png)

![D03 pixel-level confusion matrix across 121.77M Test pixels (left) and M5 Calibrated Mask-First Hybrid decision matrix on 185 held-out Test claims (right).](figures/m5_confusion_matrices.png)

**Headline System Findings:**
1. **Lowest Selective Risk & Highest Precision Across All 7 Policies:** On 100% blind Test claims, the Validation-locked **M5 Calibrated Mask-First Hybrid Verifier** achieves **`93.94%` accepted-claim precision** (`124/132` supported, `0` unverifiable leaks, **`6.06%` selective risk** — cutting analyst error exposure by **`4.37×`** vs `26.49%` unverified captions and **`3.50×`** vs `21.18%` standalone Direct Qwen3-VL!).
2. **90.27% VLM Compute Reduction:** Only **18 of 185 claims (`9.73%`)** escalated to `Qwen3-VL-4B-Instruct` (`15` residual/hedged `no_change` checks + `3` weak `<200 px` mask checks), while `120` confident accepts, `24` strong mask rejects, and `23` out-of-ontology/exact-count defers resolved deterministically in `< 15 ms` with zero VLM cost.
3. **Dramatic Calibration Breakthrough:** Expected Calibration Error drops **`4.2×`** vs Direct Qwen3-VL (**`0.0710`** vs `0.3000`) and **`6.3×`** vs 4-View Reinspection (`0.4464`), Brier score drops **`3.0×`** (`0.1041` vs `0.3142`), and AURC drops **`2.1×`** (`0.0948` vs `0.2024`). In the highest confidence bin `[0.9, 1.0]` (`n = 123`), mean predicted confidence (`0.9517`) matches empirical support (`0.9512`) within **`0.0005`**.
4. **Authentic End-to-End Test Traces & Blind Human-Audit Packet:** Ten authentic held-out Test end-to-end Evidence Review bundles (`test_e2e_traces/`, **Figure 6.5**) and the 120-claim blind dual-rater packet (`M5T_001`–`M5T_120`, `POC/milestone5/evidence/human_audit/`, verified with zero verdict leakage) were exported directly from this run.

![Five representative authentic held-out Test end-to-end traces flowing through D03 mask overlay, C06 caption, OpenCV contextual crops, and Mask-First Hybrid verification.](figures/m5_test_e2e_traces.png)

### 6.4 Target Performance Attainment Summary

| Milestone 5 Target | Required Threshold | Measured Held-Out Test Result | Status |
|---|---|---|---|
| Detector Change-Class mIoU | $\ge 74.0\%$ | **`75.93%`** `[74.79, 77.04]` (`D03_bit_lr3e4`) | **Achieved (+1.93 pts)** |
| Detector Binary F1 & ROC-AUC | $\ge 85.0\%$ F1 / $\ge 0.98$ ROC | **`86.83%`** F1 / **`0.9901`** ROC / **`0.9361`** PR-AUC | **Achieved** |
| Captioner BLEU-4 & BERTScore-F1 | $\ge 0.60$ B4 / $\ge 0.94$ BERT | **`0.6126`** B4 / **`0.9490`** BERT / **`1.3377`** CIDEr | **Achieved** |
| Captioner Unchanged Hallucination | $\le 6.0\%$ | **`3.57%`** (`33/924` pairs) | **Achieved (-2.43 pts)** |
| Hybrid Verifier Precision & VLM Savings | $\ge 90\%$ prec / $>70\%$ VLM savings | **`93.94%`** prec (`6.06%` risk) / **`90.27%`** VLM savings | **Achieved (+20.3 pts savings)** |
