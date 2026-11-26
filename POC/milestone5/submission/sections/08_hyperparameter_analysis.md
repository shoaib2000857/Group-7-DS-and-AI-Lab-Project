## 8. Hyperparameter Analysis

Milestone 4 conducted a systematic **one-factor-at-a-time (OFAT) controlled hyperparameter sweep** on the full 6,444-pair LEVIR-MCI training split (1,242-pair Validation evaluation) across both the BIT-ResNet18 detector (`D01`–`D16`, 16 controlled runs + 3-seed verification) and the RSCaMa captioner (`C01`–`C07`, 7 controlled runs + 3-seed verification). All parameters were locked strictly on Validation before opening the held-out Test split in Milestone 5.

### 8.1 Detector Hyperparameter Sweep (`D01`–`D16`, BIT-ResNet18)

Starting from the controlled reference arm `D01_bit_ref` (head LR `1e-4`, backbone LR `1e-5`, WD `0.01`, effective batch `32`, cosine schedule, Focal + Soft Dice loss, 5 frozen-backbone epochs, flip augmentation -> **`71.82%` Val Change mIoU**, `18.04%` FA@50px), each hyperparameter family was varied independently:

**Table 8.1. Controlled BIT-ResNet18 hyperparameter exploration on LEVIR-MCI Validation (1,242 pairs).**

| Factor Explored | Run ID | Setting Changed vs `D01` | Best Ep | Val Chg mIoU (%) | $\Delta$ vs `D01` | Val Bin F1 (%) | Val FA@50px (%) | Selection Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| *Reference baseline* | `D01_bit_ref` | Head `1e-4`, BB `1e-5`, WD `0.01`, EB `32`, Cos, Focal+Dice, Frz `5`, Flip | 27 | **71.82** | — | 84.57 | 18.04 | Control anchor |
| **Learning Rate** | `D02_bit_lr3e5` | Head `3e-5` / BB `3e-6` (too slow) | 20 | 64.55 | -7.27 | 79.87 | 33.39 | Underfits severely |
| **Learning Rate** | **`D03_bit_lr3e4`** | **Head `3e-4` / BB `3e-5` (3× higher)** | **24** | **74.54** | **+2.72** | **86.25** | **6.80** | **Selected (best mIoU & lowest FA!)** |
| **Weight Decay** | `D04_bit_wd1e3` | WD `0.001` (10× weaker) | 20 | 69.94 | -1.88 | 83.38 | 24.37 | Worse regularization |
| **Weight Decay** | `D05_bit_wd5e2` | WD `0.05` (5× stronger) | 20 | 69.94 | -1.88 | 83.38 | 24.53 | Over-regularized |
| **Effective Batch** | `D06_bit_eb16` | Effective batch `16` (`16×1`) | 20 | 71.66 | -0.16 | 84.43 | 18.51 | Near-parity, noisier grads |
| **Effective Batch** | `D07_bit_eb64` | Effective batch `64` (`16×4`) | 20 | 67.94 | -3.88 | 82.08 | 27.85 | Fewer steps/epoch hurts convergence |
| **LR Schedule** | `D08_bit_const` | Constant LR (`1e-4`) | 24 | 72.92 | +1.10 | 85.17 | 16.93 | Strong at 25 ep, lacks tail decay |
| **LR Schedule** | `D09_bit_step` | Step decay LR | 24 | 72.94 | +1.12 | 85.19 | 16.46 | Strong at 25 ep |
| **Loss Function** | `D10_bit_cedice` | Weighted CE + Soft Dice (no focal) | 22 | 68.44 | -3.38 | 82.47 | 29.59 | Focal hard-mining is critical |
| **Loss Function** | `D11_bit_wce` | Weighted CE only (no focal, no Dice) | 22 | 63.87 | -7.95 | 79.27 | 40.98 | Severe boundary collapse & FA spike |
| **Backbone Freeze** | `D12_bit_unfreeze0` | Unfrozen from epoch 0 | 20 | 71.70 | -0.12 | 84.39 | 21.68 | Early head noise raises FA (+3.6 pts) |
| **Backbone Freeze** | `D13_bit_frozen` | Permanently frozen ImageNet ResNet18 | 23 | 62.45 | -9.37 | 78.61 | 17.41 | Domain adaptation essential |
| **Augmentation** | `D14_bit_noaug` | No augmentation | 20 | 69.68 | -2.14 | 83.23 | 24.53 | Overfits spatial orientations |
| **Augmentation** | `D15_bit_photo` | Independent photometric jitter ($T_1 \neq T_2$) | 22 | 68.34 | -3.48 | 82.41 | 25.63 | Confuses real radiometric changes |
| **Augmentation** | `D16_bit_photoscale` | Photometric + random scale crop | 22 | 68.55 | -3.27 | 82.50 | 22.31 | Same photometric penalty |

**Key Detector Hyperparameter Findings:**
1. **Differential learning rate (`3e-4` head / `3e-5` backbone) was the single highest-leverage knob**, lifting Val Change mIoU by **`+2.72` points** over `D01` (`71.82% -> 74.54%`) while cutting unchanged-scene false alarms by **`2.65×`** (`18.04% -> 6.80%`). Too-low LR (`D02`, `3e-5`) stalled at `64.55%` with `33.39%` false alarms.
2. **Focal + Soft Dice loss outperformed Weighted CE by `+7.95` points (`71.82%` vs `63.87%`)** and cut false alarms by more than half (`18.04%` vs `40.98%`), directly confirming that hard-example focal weighting plus overlap-based Dice supervision is essential under LEVIR-MCI's 10:1 background imbalance.
3. **Effective batch size `32` (`16 × 2` accumulation)** hit the sweet spot between gradient stability and optimizer steps per epoch; doubling to `64` (`D07`) halved update frequency and cost `-3.88` mIoU points.
4. **Weight decay `0.01`** outperformed both `0.001` (`D04`, `-1.88` pts) and `0.05` (`D05`, `-1.88` pts).

### 8.2 Captioner Hyperparameter Sweep (`C01`–`C07`, RSCaMa)

**Table 8.2. Controlled RSCaMa hyperparameter exploration on LEVIR-MCI Validation (1,242 pairs, 388-token train-only vocab).**

| Factor Explored | Run ID | Setting Changed | Best Ep | Val BLEU-4 | Val ROUGE-L | Changed BLEU-4 | Val CE | NC FA (%) | Selection Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| *Upstream recipe* | `C01_rscama_ref` | Adam, LR `1e-4`, Constant, Unfrozen CLIP, EB `12` | 5 | 0.6296 | 0.7275 | 0.3916 | 1.2514 | 7.75 | High unchanged hallucination |
| **Learning Rate** | `C02_rscama_lr3e4` | LR `3e-4` (3× higher) | 0 | 0.4956 | 0.6765 | 0.2272 | 1.5260 | 0.79 | Diverges into mode collapse |
| **Learning Rate** | `C03_rscama_lr3e5` | LR `3e-5` (3× lower) | 0 | 0.6214 | 0.7177 | 0.2447 | 2.2182 | 0.47 | Underfits changed scenes (`0.2447`) |
| **Opt + Schedule** | `C04_rscama_adamw_cos` | AdamW (`WD=0.01`) + Cosine | 6 | 0.6264 | **0.7498** | **0.4380** | 1.2653 | 5.70 | Best changed-pair fluency, higher FA |
| **Label Smoothing** | `C05_rscama_ls01` | Label smoothing `0.1` + AdamW/Cos | 3 | 0.5963 | 0.7421 | 0.4151 | 2.1336 | 6.80 | Hurts corpus BLEU-4 (-0.030) |
| **CLIP Freezing** | **`C06_rscama_frozen_clip`** | **Frozen CLIP ViT-B/32 + AdamW/Cos** | **6** | **0.6358** | **0.7405** | **0.4047** | **1.3147** | **3.32** | **Selected (best B4 + lowest FA!)** |
| **Effective Batch** | `C07_rscama_eb24` | Effective batch `24` (`12×2`) | 3 | 0.6281 | 0.7400 | 0.4068 | 1.3791 | 6.49 | Marginal difference, 2× higher FA |

**Key Captioner Hyperparameter Findings:**
1. **Learning rate sensitivity is razor-sharp:** `1e-4` is the sole viable regime; raising to `3e-4` (`C02`) collapsed training immediately after epoch 0 (`0.4956` BLEU-4), while lowering to `3e-5` (`C03`) left changed-scene BLEU-4 stuck at `0.2447` (`Val CE = 2.218`).
2. **Freezing CLIP ViT-B/32 (`C06`)** prevented visual feature drift on the 6,444-pair dataset, delivering both the highest corpus BLEU-4 (`0.6358`) and a **`2.3×` reduction in unchanged-scene caption hallucinations** (`3.32%` vs `7.75%` for `C01` and `5.70%` for `C04`) while cutting training VRAM by `649 MB`.
3. **Label smoothing (`0.1`, `C05`)** softened token probabilities over the compact 388-word vocabulary and degraded BLEU-4 by `-0.0301`.

### 8.3 Hybrid Verifier Threshold and Routing Calibration (Validation-Locked)

On the 120-pair (147-claim) Milestone 4 Validation verification audit, we swept the acceptance threshold $\tau \in [0.50, 0.90]$ and calibrated the selective routing rule (`POC/milestone5/scripts/m5_eval_captioners_and_verifier.py`):

- Unconditional Direct `Qwen3-VL-4B-Instruct` at $\tau=0.70$ achieved `82.43%` proxy precision (`17.57%` selective risk) with `40.8%` deferral and `100%` VLM cost.
- Locking the **Calibrated Mask-First Hybrid Verifier** on Validation — resolving unambiguous mask/ontology claims deterministically and escalating only BIT blind spots (`action == 'removed'`, `no_change` with `any_px >= 10` or hedged phrasing, and marginal `added` claims with `< 200 px`) to Direct Qwen3-VL at $\tau_A = 0.60, \tau_R = 0.75$ — raised Validation accepted precision to **`94.50%`** (`5.50%` selective risk) and contradicted-claim catch rate to **`70.37%`** while cutting VLM calls to **`10.9%`** (**`89.1%` compute savings**). Section 6.3 reports how this exact Validation-locked policy generalizes zero-shot to the 150-pair blind held-out Test split.
