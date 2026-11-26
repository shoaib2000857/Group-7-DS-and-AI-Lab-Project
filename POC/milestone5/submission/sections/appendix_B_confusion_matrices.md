## Appendix B. Confusion Matrices

### B.1 Pixel-Level 3-Class Confusion Matrix (`D03_bit_lr3e4`, 1,858 Held-Out Test Pairs)

Total evaluated pixels: **121,765,888** ($1,858 \times 256 \times 256$). Visualized alongside the Hybrid Verifier decision matrix in `POC/milestone5/submission/figures/m5_confusion_matrices.png`.

| Ground Truth \ Predicted | Background (`0`) | Road (`1`) | Building (`2`) | Row Total (GT Pixels) | Class Recall (%) |
|---|---:|---:|---:|---:|---:|
| **Background (`0`)** | **109,533,948** (`98.78%`) | 499,938 (`0.45%`) | 849,805 (`0.77%`) | 110,883,691 | 98.78% |
| **Road (`1`)** | 650,809 (`15.80%`) | **3,457,972** (`83.97%`) | 9,511 (`0.23%`) | 4,118,292 | 83.97% |
| **Building (`2`)** | 847,334 (`12.53%`) | 21,531 (`0.32%`) | **5,895,040** (`87.15%`) | 6,763,905 | 87.15% |

**Diagnostic Observation:** Cross-class confusion between `Road` and `Building` is remarkably low (**`0.23%`** of road pixels misclassified as building and **`0.32%`** of building pixels misclassified as road; `31,042` total pixels out of `10.88M` foreground pixels, or `< 0.29%`). Over `97%` of foreground errors are boundary/extent misses against `Background` (`15.80%` road-to-BG, `12.53%` building-to-BG), confirming that spatial resolution and small-structure recall — not semantic road-vs-building separation — dominate remaining detector error.

### B.2 Held-Out Test Hybrid Verifier Decision Matrix (150 Stratified Test Pairs, 185 Claims)

| Reference Proxy Ground Truth \ M5 Hybrid Decision | `accept` | `reject` | `defer` | Row Total |
|---|---:|---:|---:|---:|
| **`supported`** | **124** (`91.2%`) | 4 (`2.9%`) | 8 (`5.9%`) | 136 |
| **`contradicted`** | 8 (`26.7%`) | **21** (`70.0%`) | 1 (`3.3%`) | 30 |
| **`unverifiable`** | **0** (`0.0%`) | 0 (`0.0%`) | **19** (`100.0%`) | 19 |
| **Column Total** | **132** (`71.4%`) | **25** (`13.5%`) | **28** (`15.1%`) | **185** |

Zero unverifiable claims (`0/19`) leaked into accepted outputs, while accepted precision reached **`93.94%`** (`124/132`) and contradicted rejection reached **`70.00%`** (`21/30`).
