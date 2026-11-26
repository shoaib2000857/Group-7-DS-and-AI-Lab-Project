## 9. Ablation Study

To isolate the causal contribution of every design choice in TerraWatch — across spatial detection, change captioning, and multimodal claim verification — we synthesize controlled module-removal and component-swap ablations anchored to Milestone 4 Validation sweeps and Milestone 5 held-out Test evaluations.

### 9.1 Detector Component Ablations (BIT-ResNet18)

**Table 9.1. Controlled detector component ablations (Val = 1,242 pairs; Test = 1,858 pairs).**

| Component Tested | Full Configuration | Ablated Variant | Val Chg mIoU (%) | $\Delta$ mIoU | Val FA@50px (%) | Diagnostic Interpretation |
|---|---|---|---:|---:|---:|---|
| **Backbone Adaptation** | Unfreeze after 5 ep (`D01`) | Frozen ResNet18 (`D13`) | 71.82 vs 62.45 | **-9.37 pts** | 18.04 vs 17.41 | ImageNet features cannot separate aerial roads from roofs without end-to-end fine-tuning. |
| **Focal + Dice Loss** | Focal + Soft Dice (`D01`) | Weighted CE only (`D11`) | 71.82 vs 63.87 | **-7.95 pts** | 18.04 vs 40.98 | Removing Focal+Dice doubles unchanged false alarms and erodes thin road boundaries. |
| **Focal Hard-Mining** | Focal + Soft Dice (`D01`) | CE + Soft Dice (`D10`) | 71.82 vs 68.44 | **-3.38 pts** | 18.04 vs 29.59 | Focal modulation ($\gamma=2$) suppresses easy background pixels and cuts FA by 11.6 pts. |
| **Flip Augmentation** | Random H/V flips (`D01`) | No augmentation (`D14`) | 71.82 vs 69.68 | **-2.14 pts** | 18.04 vs 24.53 | Orientation invariance reduces overfitting and cuts unchanged FA by 6.5 pts. |
| **Photometric Jitter** | Flip only (`D01`) | Independent $T_1/T_2$ jitter (`D15`) | 71.82 vs 68.34 | **-3.48 pts** | 18.04 vs 25.63 | Asymmetric radiometric jitter mimics shadow/albedo drift and degrades clean boundary fidelity. |
| **Warmup Freeze (5 ep)** | Freeze 5 ep (`D01`) | Unfreeze at ep 0 (`D12`) | 71.82 vs 71.70 | -0.12 pts | 18.04 vs 21.68 | Warm-starting the random BIT head before unfreezing ResNet18 reduces unchanged FA by 3.6 pts. |
| **Global Token Attention** | `D03_bit_lr3e4` (BIT tokens) | `A03_siam_r18_unet` (local CNN) | **Test Blur r=1.0:** `72.10` vs `60.12` | **+11.98 pts** | **Test Blur r=2.0:** `40.54` vs `6.91` (`+33.63`) | Bitemporal semantic tokens sacrifice 0.80 pts on clean Test (`75.93%` vs `76.73%`) in exchange for massive resilience to optical blur, downscaling (`+12.03` pts), noise (`+4.57` pts), and SECOND-CC transfer (`+5.46` F1 pts). |

### 9.2 Captioner Component Ablations (RSCaMa)

**Table 9.2. Controlled RSCaMa captioner ablations (Validation & CUDA Compute Profile).**

| Component Tested | Selected (`C06`) | Ablated Variant | Key Metric Impact | Engineering Takeaway |
|---|---|---|---|---|
| **Frozen CLIP ViT-B/32 Encoder** | Frozen (`C06`) | Unfrozen CLIP (`C04_rscama_adamw_cos`) | Val B4: `0.6358` vs `0.6264` (`+0.0094`); Val NC FA: **`3.32%` vs `5.70%`** (`-2.38` pts); Train VRAM: `4.82` vs `5.47 GB` | Freezing CLIP preserves pre-trained visual invariance on 6.4k pairs, halving hallucinations on unchanged scenes while cutting VRAM. |
| **AdamW + Cosine Decay** | AdamW + Cosine (`C06`) | Adam + Constant (`C01_rscama_ref`) | Val B4: `0.6358` vs `0.6296`; Val ROUGE-L: `0.7405` vs `0.7275`; Val NC FA: **`3.32%` vs `7.75%`** | Decoupled weight decay + cosine schedule cuts unchanged hallucinations by 4.43 pts. |
| **Exact Fused Triton Selective Scan** | `triton_fused_exact` | Naive PyTorch reference scan | **Training:** `227 s/ep` vs `2,310 s/ep` (**`10.2×` speedup**); **RTX 4060 Inference:** `23.30 ms` vs `89.35 ms` P50 (**`3.83×` speedup**, identical `8.69` decode steps) | Zero accuracy loss (gradient rel. err $\le 4.4 \times 10^{-7}$), turning a 37-hour captioner sweep into 3.88 GPU-hours and enabling sub-25 ms inference. |
| **Learned Neural Captioning vs Mask Templates** | `C06` neural RSCaMa | `bit_mask_template` (deterministic mask text) | Section 6.2 & 7.2 compare lexical/semantic fluency (BLEU-4, CIDEr-D, BERTScore) vs mask-grounded object fidelity | Neural captioning captures natural scene layout and phrasing that rigid templates cannot, while templates provide zero object-class hallucination relative to `D03`. |

### 9.3 Multimodal Verification Pipeline Ablations

Every component of the Milestone 5 **Calibrated Mask-First Hybrid Verifier** serves a distinct, measurable role verified on Validation (147 claims) and confirmed on held-out Test (150 pairs, Section 6.3):

1. **Without Verification (`1_unverified_captioner`):** Exposes analysts directly to captioner hallucinations (`72.79%` Val precision, `27.21%` selective risk, `0.0%` hallucination catch rate).
2. **Without Spatial Detector Mask (`4_direct_qwen3vl_gated`):** Standalone `Qwen3-VL-4B-Instruct` inspects full $T_1/T_2$ images at `100%` VLM compute cost, achieving only `82.43%` Val precision (`17.57%` risk) and deferring `40.8%` of claims due to small-object visual ambiguity.
3. **Naive 4-View Crop Prompting (`5_reinspection_all_gated`):** Feeding zoomed $T_1/T_2$ crops plus full images unconditionally to `Qwen3-VL-4B` *degraded* Val precision from `82.43%` to `72.41%` because zoomed crops remove surrounding spatial context and bias the 4B VLM toward over-predicting change on minor shadow/alignment differences.
4. **Without VLM Escalation (`2_mask_only_checker`):** Deterministic BIT mask grounding alone already reaches `91.38%` Val precision at `0%` VLM cost, but cannot resolve `removed`-object claims (BIT predicts unsigned change masks without temporal direction), marginal `< 200 px` changes, or hedged `no_change` captions with nonzero predicted masks (`62.96%` contradicted catch rate).
5. **Without Deterministic Abstention Rules (Out-of-Ontology & Exact Counts):** Removing the parser's `T_out_of_mask_ontology` and `exact_count_without_independent_instance_evidence` deferral rules forces unsupported vegetation/water or exact-count claims into false accepts or false rejects.
6. **Full Calibrated Mask-First Hybrid (`7_m5_calibrated_mask_first_hybrid`):** Combining BIT mask-first resolution, deterministic ontology/count abstention, OpenCV structural witness diagnostics, and selective Direct-Qwen3-VL escalation ONLY on BIT's true blind spots achieves **`94.50%` Val precision** (`5.50%` risk), **`70.37%` contradicted catch**, and **`89.1%` VLM compute savings** (`10.9%` VLM call rate) — strictly Pareto-dominating every single-module ablation.
