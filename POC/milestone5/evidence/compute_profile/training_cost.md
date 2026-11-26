| Run | Family | Params total (M) | Trainable (M) | GPU | Epochs run / cap | Best epoch | Wall (min) | Sec/epoch | Peak train VRAM (MB) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| `A01_ef_unet` | ef_unet | 7.76 | 7.76 | Tesla T4 | 25 / 25 | 23 | 38.9 | 93.3 | 2,289 |
| `A02_fc_siam_diff` | fc_siam_diff | 7.76 | 7.76 | Tesla T4 | 24 / 25 | 17 | 49.1 | 122.7 | 3,083 |
| `A03_siam_r18_unet` | siam_resnet18_unet | 15.63 | 15.63 | Tesla T4 | 25 / 25 | 22 | 35.1 | 84.3 | 1,410 |
| `A03_siam_r18_unet_s0` | siam_resnet18_unet | 15.63 | 15.63 | NVIDIA L4 | 25 / 25 | 24 | 15.6 | 37.5 | 1,408 |
| `A03_siam_r18_unet_s2026` | siam_resnet18_unet | 15.63 | 15.63 | NVIDIA L4 | 22 / 25 | 15 | 14.2 | 38.8 | 1,408 |
| `A04_changeformer` | changeformer | 41.04 | 41.04 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 22 | 69.0 | 165.6 | 6,570 |
| `D01_bit_ref` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 35 / 35 | 27 | 34.2 | 58.6 | 2,287 |
| `D02_bit_lr3e5` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 20 | 24.0 | 57.6 | 2,287 |
| `D03_bit_lr3e4` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 32 / 35 | 24 | 31.2 | 58.6 | 2,287 |
| `D03_bit_lr3e4_s0` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 30 / 30 | 29 | 29.1 | 58.3 | 2,287 |
| `D03_bit_lr3e4_s2026` | bit_resnet18 | 11.94 | 11.94 | NVIDIA L4 | 30 / 30 | 23 | 23.8 | 47.7 | 2,492 |
| `D04_bit_wd1e3` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 20 | 45.6 | 109.4 | 2,285 |
| `D05_bit_wd5e2` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 20 | 24.2 | 58.0 | 2,287 |
| `D06_bit_eb16` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 20 | 24.2 | 58.0 | 2,274 |
| `D07_bit_eb64` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 20 | 45.5 | 109.2 | 2,286 |
| `D08_bit_const` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 24 | 24.0 | 57.6 | 2,287 |
| `D09_bit_step` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 24 | 45.3 | 108.7 | 2,287 |
| `D10_bit_cedice` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 22 | 23.8 | 57.1 | 2,287 |
| `D11_bit_wce` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 22 | 23.6 | 56.7 | 2,287 |
| `D12_bit_unfreeze0` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 20 | 25.2 | 60.5 | 2,745 |
| `D13_bit_frozen` | bit_resnet18 | 11.94 | 0.25 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 23 | 19.0 | 45.6 | 2,174 |
| `D14_bit_noaug` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 20 | 24.1 | 57.9 | 2,287 |
| `D15_bit_photo` | bit_resnet18 | 11.94 | 11.94 | NVIDIA GeForce RTX 4060 Laptop GPU | 25 / 25 | 22 | 24.0 | 57.5 | 2,287 |
| `D16_bit_photoscale` | bit_resnet18 | 11.94 | 11.94 | Tesla T4 | 25 / 25 | 22 | 52.1 | 125.1 | 2,287 |
| `C01_rscama_ref` | rscama | 898.81 | 288.06 | Tesla T4 | 8 / 8 | 5 | 31.3 | 234.9 | 5,469 |
| `C02_rscama_lr3e4` (diverged_nonfinite) | rscama | 898.81 | 288.06 | Tesla T4 | 2 / 8 | 0 | 7.6 | 228.4 | 5,469 |
| `C03_rscama_lr3e5` | rscama | 898.81 | 288.06 | Tesla T4 | 6 / 8 | 0 | 24.2 | 241.7 | 5,469 |
| `C04_rscama_adamw_cos` | rscama | 898.81 | 288.06 | Tesla T4 | 8 / 8 | 6 | 34.0 | 254.7 | 5,469 |
| `C05_rscama_ls01` | rscama | 898.81 | 288.06 | Tesla T4 | 7 / 8 | 3 | 31.2 | 267.8 | 5,469 |
| `C06_rscama_frozen_clip` | rscama | 898.81 | 245.53 | Tesla T4 | 8 / 8 | 6 | 30.3 | 227.3 | 4,820 |
| `C06_rscama_frozen_clip_s0` | rscama | 898.81 | 245.53 | Tesla T4 | 6 / 8 | 1 | 24.4 | 243.8 | 4,820 |
| `C06_rscama_frozen_clip_s2026` | rscama | 898.81 | 245.53 | Tesla T4 | 6 / 8 | 1 | 22.6 | 225.7 | 4,820 |
| `C07_rscama_eb24` | rscama | 898.81 | 288.06 | Tesla T4 | 7 / 8 | 3 | 27.4 | 234.8 | 5,888 |

| GPU | Runs | Epochs | Wall (min) | GPU-hours |
|---|---:|---:|---:|---:|
| NVIDIA GeForce RTX 4060 Laptop GPU | 14 | 372 | 399.6 | 6.66 |
| NVIDIA L4 | 3 | 77 | 53.7 | 0.90 |
| Tesla T4 | 16 | 232 | 544.5 | 9.07 |
| **Total** | **33** | **681** | **997.8** | **16.63** |
