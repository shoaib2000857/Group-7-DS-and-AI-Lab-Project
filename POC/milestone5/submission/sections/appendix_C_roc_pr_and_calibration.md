## Appendix C. ROC, Precision-Recall, and Calibration Curves

### C.1 Pixel-Level ROC and Precision-Recall Analysis (121.77M Held-Out Test Pixels)

All five neural change detectors and the non-learning Otsu pixel-difference control were scored continuously using non-saturating log-odds $s = \log(\exp(z_1) + \exp(z_2)) - z_0 \in [-15, 15]$ across 10,000 uniform histogram bins, yielding exact step-function Average Precision (PR-AUC) and trapezoidal ROC-AUC (`POC/milestone5/submission/figures/m5_roc_pr_curves.png`):

- Every trained neural detector achieves **ROC-AUC $\in [0.9889, 0.9913]$** and **PR-AUC $\in [0.9274, 0.9434]$**, vastly exceeding Otsu pixel differencing (`ROC-AUC = 0.5559`, `PR-AUC = 0.1148`).
- `D03_bit_lr3e4` achieves **`ROC-AUC = 0.9901`** and **`PR-AUC = 0.9361`**, tracking within `0.005` AP of `A03_siam_r18_unet` (`0.9411`) and `0.007` AP of `A01_ef_unet` (`0.9434`), while outperforming `A02_fc_siam_diff` (`0.9335`) and `A04_changeformer` (`0.9274`).

### C.2 Hybrid Verifier Risk-Coverage and Reliability Calibration (185 Blind Test Claims)

| Verifier Model | ECE $\downarrow$ | Brier Score $\downarrow$ | AURC $\downarrow$ | Top Confidence Bin `[0.9, 1.0]` Mean Conf vs Empirical Acc |
|---|---:|---:|---:|---|
| **`M5 Calibrated Mask-First Hybrid`** | **0.0710** | **0.1041** | **0.0948** | `n=123`: mean conf **`0.9517`** vs empirical **`0.9512`** (`Δ = 0.0005`) |
| `Direct Qwen3-VL-4B (full image)` | 0.3000 | 0.3142 | 0.2024 | `n=58`: mean conf `0.9655` vs empirical `0.8276` (`Δ = 0.1379`) |
| `4-View Reinspection-All` | 0.4464 | 0.4417 | 0.2298 | `n=90`: mean conf `0.9908` vs empirical `0.7000` (`Δ = 0.2908`) |

Full calibration curves and threshold risk-coverage sweeps across $\tau \in [0.50, 0.90]$ are plotted in `figures/m5_hybrid_verifier_calibration.png` (Section 6.3).
