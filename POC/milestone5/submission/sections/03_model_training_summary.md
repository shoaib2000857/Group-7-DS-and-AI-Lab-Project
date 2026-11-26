## 3. Model Training Summary

All models evaluated in Milestone 5 are the **exact frozen checkpoints produced during Milestone 4** (zero retraining or test-set tuning). Across the full Milestone 4 experimental campaign, **33 full-data training runs** (24 detector runs and 9 captioner runs; 681 total epochs; 16.63 GPU-hours) were executed on the complete 6,444-pair LEVIR-MCI training split and validated on the 1,242-pair validation split.

### 3.1 Final Model Architectures

- **Selected Change Detector (`D03_bit_lr3e4` — BIT-ResNet18):** A Siamese ResNet-18 backbone (`resnet18`, ImageNet pretrained) extracts multi-scale spatial feature maps from $T_1$ and $T_2$ ($256 \times 256 \times 3$, normalized to $[-1, 1]$). A semantic tokenizer projects spatial features into $L = 4$ compact visual semantic tokens per timestamp; a Bitemporal Transformer Encoder models cross-temporal interactions globally across both images, and a Transformer Decoder projects refined tokens back to spatial resolution before elementwise absolute differencing and a lightweight 3-class segmentation head (`background`, `road`, `building`). Total parameters: **11,944,043** (head: 0.25M; disk size: **45.68 MiB**; **17.45 GFLOPs/pair**).
- **Selected Change Captioner (`C06_rscama_frozen_clip` — RSCaMa with Frozen CLIP ViT-B/32):** Dual-branch CLIP ViT-B/32 visual encoder (`151.28M` parameters, frozen) encodes $224 \times 224$ bitemporal frames into dense patch sequences. A Spatial-Temporal Mamba fusion module (`733.90M` recorded parameters, of which `502.0M` are unused inherited language-model embedding tables and `231.90M` are active state-space layers) fuses bitemporal differences via bidirectional selective state-space scans accelerated by an exact fused Triton kernel (`triton_fused_exact`, matching FP32 reference gradients within $1.2 \times 10^{-7}$–$4.4 \times 10^{-7}$ relative error). A 1-layer Transformer decoder (`13.63M` parameters) autoregressively decodes change captions over a **388-token train-only vocabulary** (frequency $\ge 5$, max length 42). Total recorded parameters: **898,805,894** (checkpoint disk size: **1,714.63 MiB**; **39.81 GFLOPs/pair**).
- **Frozen Visual Referee (`Qwen/Qwen3-VL-4B-Instruct`):** Pinned at Hugging Face revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` (~4.44B parameters, zero task-specific fine-tuning). Evaluates structured atomic claims alongside bitemporal imagery via restricted 3-class single-token scoring over `{A: accept, B: reject, C: defer}` with deterministic softmax normalization.

### 3.2 Final Training Hyperparameters and Schedules

**Table 3.1. Locked training configurations of selected production models and architectural challengers.**

| Parameter | Detector `D03_bit_lr3e4` (Selected) | Challenger `A03_siam_r18_unet` | Baseline `A01_ef_unet` | Baseline `A04_changeformer` | Captioner `C06_rscama_frozen_clip` (Selected) |
|---|---|---|---|---|---|
| **Architecture** | `bit_resnet18` | `siam_resnet18_unet` | `ef_unet` | `changeformer` (MiT-B2) | RSCaMa (Frozen CLIP ViT-B/32 + Mamba) |
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW |
| **Learning Rate** | Head `3e-4` / Backbone `3e-5` | Head `3e-4` / Backbone `3e-5` | `3e-4` (uniform) | `1e-4` (uniform) | `1e-4` (fusion + decoder) |
| **Weight Decay** | `0.01` | `0.01` | `0.01` | `0.01` | `0.01` |
| **LR Schedule** | Cosine annealing | Cosine annealing | Cosine annealing | Cosine annealing | Cosine annealing |
| **Effective Batch Size** | 32 (`16 × 2` grad accum) | 32 (`16 × 2`) | 32 (`16 × 2`) | 32 (`16 × 2`) | 12 (`12 × 1`) |
| **Loss Function** | Focal ($\gamma=2$, $w=[0.10, 0.45, 0.45]$) + Soft Dice | Focal + Soft Dice | Focal + Soft Dice | Focal + Soft Dice | Cross-Entropy (label smoothing `0.0`) |
| **Backbone Freeze** | First 5 epochs, then unfrozen | First 5 epochs | None (0 epochs) | None (0 epochs) | Permanently frozen (`freeze_encoder=True`) |
| **Data Augmentation** | Random horizontal/vertical flip | Random flip | Random flip | Random flip | None (CLIP preprocessing) |
| **Gradient Clipping** | Global norm `1.0` | Global norm `1.0` | Global norm `1.0` | Global norm `1.0` | Global norm `5.0` |
| **Mixed Precision** | `bf16` AMP (RTX 4060) | `fp16` AMP (T4) | `fp16` AMP (T4) | `bf16` AMP (RTX 4060) | `fp16` AMP + `triton_fused_exact` (T4) |
| **Epoch Cap / Executed** | 35 cap / **32 executed** | 25 cap / **25 executed** | 25 cap / **25 executed** | 25 cap / **25 executed** | 8 cap / **8 executed** |
| **Early Stopping** | `min_epochs=15`, `patience=8` | `min_epochs=12`, `patience=6` | `min_epochs=12`, `patience=6` | `min_epochs=12`, `patience=6` | `min_epochs=6`, `patience=3` |
| **Checkpoint Selection** | Max Val Change-Class mIoU (**Ep 24**: `74.54%`) | Max Val Change mIoU (**Ep 22**: `75.02%`) | Max Val Change mIoU (**Ep 23**: `73.60%`) | Max Val Change mIoU (**Ep 22**: `70.46%`) | Max Val Corpus BLEU-4 (**Ep 6**: `0.6358`) |
| **Training Duration** | **31.2 min** (58.6 s/ep, RTX 4060) | 35.1 min (84.3 s/ep, T4) | 38.9 min (93.3 s/ep, T4) | 69.0 min (165.6 s/ep, RTX 4060) | **30.3 min** (227.3 s/ep, T4) |
| **Peak Train VRAM** | 2,287 MB | 1,410 MB | 2,289 MB | 6,570 MB | 4,820 MB |

### 3.3 Multi-Seed Stability and Checkpoint Selection Rationale

To confirm that selected model rankings were not artifacts of a single random seed, both top detector families and the winning captioner were trained across three independent random seeds (`42`, `0`, `2026`):

- **`D03_bit_lr3e4` (3 seeds):** Val Change-Class mIoU = **`74.23% ± 0.28%`** (`74.54%` s42, `74.00%` s0, `74.15%` s2026), confirming tight convergence stability.
- **`A03_siam_r18_unet` (3 seeds):** Val Change-Class mIoU = **`74.73% ± 0.27%`** (`75.02%` s42, `74.49%` s0, `74.68%` s2026). Although `A03` edged `D03` by `+0.50` points on clean Validation imagery, **`D03_bit_lr3e4` was selected as the production detector** strictly on Validation evidence because it delivered dramatically superior resilience to blur (`+12.38` pts at $r=1.0$, `+30.36` pts at $r=2.0$), lower clean unchanged false alarms (`6.80%` vs `8.54%` Val FA@50px), and stronger cross-dataset zero-shot generalization on SECOND-CC.
- **`C06_rscama_frozen_clip` (3 seeds):** Val Corpus BLEU-4 = **`0.6297 ± 0.0183`** (`0.6358` s42, `0.6092` s0, `0.6442` s2026). Seed 42 achieved the best combined balance of Val BLEU-4 (`0.6358`), ROUGE-L (`0.7405`), changed-pair BLEU-4 (`0.4047`), and the lowest unchanged-scene hallucination rate (`3.32%` vs `5.70%–7.75%` for unfrozen CLIP runs) while reducing training peak VRAM by `649 MB`.

Full learning curves across all 33 runs and exact SHA-256 digests of all 12 frozen checkpoints are documented in **Section 2.3**, **Appendix D**, and **Appendix H**.
