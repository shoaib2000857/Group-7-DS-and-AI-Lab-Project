## Appendix E. Complete Hyperparameter Search Results (All 33 Milestone 4 Runs)

### E.1 All 24 Detector Runs (LEVIR-MCI Validation, 1,242 Pairs)

| Run ID | Arch | Head LR | WD | Eff Batch | Sched | Loss | Freeze Ep | Aug | Best Ep | Val Chg mIoU (%) | Road IoU (%) | Bldg IoU (%) | Val Bin F1 (%) | Val FA@50px (%) |
|---|---|---:|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `A03_siam_r18_unet` | `siam_resnet18_unet` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 22 | 75.02 | 72.61 | 77.42 | 86.42 | 8.54 |
| `A03_siam_r18_unet_s2026` | `siam_resnet18_unet` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 15 | 74.68 | 72.06 | 77.30 | 86.23 | 7.44 |
| **`D03_bit_lr3e4` (Selected)** | **`bit_resnet18`** | **3e-4** | **0.01** | **32** | **cosine** | **focal_dice** | **5** | **flip** | **24** | **74.54** | **72.38** | **76.71** | **86.25** | **6.80** |
| `A03_siam_r18_unet_s0` | `siam_resnet18_unet` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 24 | 74.49 | 71.86 | 77.13 | 86.20 | 8.07 |
| `D03_bit_lr3e4_s2026` | `bit_resnet18` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 23 | 74.15 | 72.12 | 76.19 | 85.93 | 9.97 |
| `D03_bit_lr3e4_s0` | `bit_resnet18` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 29 | 74.00 | 71.85 | 76.15 | 85.89 | 10.60 |
| `A01_ef_unet` | `ef_unet` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 0 | flip | 23 | 73.60 | 70.53 | 76.67 | 85.65 | 8.39 |
| `D09_bit_step` | `bit_resnet18` | 1e-4 | 0.01 | 32 | step | focal_dice | 5 | flip | 24 | 72.94 | 71.12 | 74.76 | 85.19 | 16.46 |
| `D08_bit_const` | `bit_resnet18` | 1e-4 | 0.01 | 32 | constant | focal_dice | 5 | flip | 24 | 72.92 | 71.11 | 74.73 | 85.17 | 16.93 |
| `D01_bit_ref` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 27 | 71.82 | 69.72 | 73.92 | 84.57 | 18.04 |
| `D12_bit_unfreeze0` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 0 | flip | 20 | 71.70 | 69.94 | 73.46 | 84.39 | 21.68 |
| `D06_bit_eb16` | `bit_resnet18` | 1e-4 | 0.01 | 16 | cosine | focal_dice | 5 | flip | 20 | 71.66 | 69.78 | 73.54 | 84.43 | 18.51 |
| `A02_fc_siam_diff` | `fc_siam_diff` | 3e-4 | 0.01 | 32 | cosine | focal_dice | 0 | flip | 17 | 71.60 | 68.75 | 74.45 | 84.16 | 4.11 |
| `A04_changeformer` | `changeformer` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 0 | flip | 22 | 70.46 | 66.44 | 74.49 | 83.99 | 10.60 |
| `D04_bit_wd1e3` | `bit_resnet18` | 1e-4 | 0.001 | 32 | cosine | focal_dice | 5 | flip | 20 | 69.94 | 67.90 | 71.99 | 83.38 | 24.37 |
| `D05_bit_wd5e2` | `bit_resnet18` | 1e-4 | 0.05 | 32 | cosine | focal_dice | 5 | flip | 20 | 69.94 | 67.91 | 71.97 | 83.38 | 24.53 |
| `D14_bit_noaug` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 5 | none | 20 | 69.68 | 67.54 | 71.81 | 83.23 | 24.53 |
| `D16_bit_photoscale` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip_photo_scale | 22 | 68.55 | 67.20 | 69.91 | 82.50 | 22.31 |
| `D10_bit_cedice` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | ce_dice | 5 | flip | 22 | 68.44 | 65.41 | 71.47 | 82.47 | 29.59 |
| `D15_bit_photo` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 5 | flip_photo | 22 | 68.34 | 67.10 | 69.59 | 82.41 | 25.63 |
| `D07_bit_eb64` | `bit_resnet18` | 1e-4 | 0.01 | 64 | cosine | focal_dice | 5 | flip | 20 | 67.94 | 65.78 | 70.09 | 82.08 | 27.85 |
| `D02_bit_lr3e5` | `bit_resnet18` | 3e-5 | 0.01 | 32 | cosine | focal_dice | 5 | flip | 20 | 64.55 | 62.43 | 66.67 | 79.87 | 33.39 |
| `D11_bit_wce` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | weighted_ce | 5 | flip | 22 | 63.87 | 60.34 | 67.40 | 79.27 | 40.98 |
| `D13_bit_frozen` | `bit_resnet18` | 1e-4 | 0.01 | 32 | cosine | focal_dice | 999 | flip | 23 | 62.45 | 60.13 | 64.77 | 78.61 | 17.41 |

### E.2 All 9 Captioner Runs (LEVIR-MCI Validation, 1,242 Pairs, 388-Word Train-Only Vocab)

| Run ID | LR | Opt | Sched | Label Smooth | Freeze CLIP | Eff Batch | Best Ep | Val BLEU-4 | Val ROUGE-L | Changed BLEU-4 | Val CE | Val NC FA (%) |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `C06_rscama_frozen_clip_s2026` | 1e-4 | adamw | cosine | 0.0 | True | 12 | 1 | 0.6442 | 0.7325 | 0.3720 | 1.4276 | 4.43 |
| **`C06_rscama_frozen_clip` (Selected)** | **1e-4** | **adamw** | **cosine** | **0.0** | **True** | **12** | **6** | **0.6358** | **0.7405** | **0.4047** | **1.3147** | **3.32** |
| `C01_rscama_ref` | 1e-4 | adam | constant | 0.0 | False | 12 | 5 | 0.6296 | 0.7275 | 0.3916 | 1.2514 | 7.75 |
| `C07_rscama_eb24` | 1e-4 | adam | constant | 0.0 | False | 24 | 3 | 0.6281 | 0.7400 | 0.4068 | 1.3791 | 6.49 |
| `C04_rscama_adamw_cos` | 1e-4 | adamw | cosine | 0.0 | False | 12 | 6 | 0.6264 | 0.7498 | 0.4380 | 1.2653 | 5.70 |
| `C03_rscama_lr3e5` | 3e-5 | adam | constant | 0.0 | False | 12 | 0 | 0.6214 | 0.7177 | 0.2447 | 2.2182 | 0.47 |
| `C06_rscama_frozen_clip_s0` | 1e-4 | adamw | cosine | 0.0 | True | 12 | 1 | 0.6092 | 0.7168 | 0.3619 | 1.4751 | 7.44 |
| `C05_rscama_ls01` | 1e-4 | adamw | cosine | 0.1 | False | 12 | 3 | 0.5963 | 0.7421 | 0.4151 | 2.1336 | 6.80 |
| `C02_rscama_lr3e4` | 3e-4 | adam | constant | 0.0 | False | 12 | 0 | 0.4956 | 0.6765 | 0.2272 | 1.5260 | 0.79 |
