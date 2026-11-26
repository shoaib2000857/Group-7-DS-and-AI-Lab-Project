## 10. Error Analysis

To move beyond aggregate metrics and pinpoint actionable root causes, we systematically mined errors across all 1,858 held-out Test detector predictions, all 1,858 Test captions, and all 185 blind Test verification claims (`POC/milestone5/evidence/error_analysis/failure_taxonomy.csv`, `detector_error_mining.json`, `captioner_test_full.json`, `test_verification_claims.jsonl`).

### 10.1 Quantified 6-Category Failure Taxonomy

**Table 10.1. End-to-end Milestone 5 failure taxonomy quantified across all three TerraWatch modules on held-out Test.**

| ID | Failure Category | Module | Measured Scope | Quantified Rate / Count | Primary Root Cause | Representative Test Pair |
|---|---|---|---|---|---|---|
| **E1** | Unchanged-Scene Detector FP | Detector (`D03`) | 924 unchanged Test pairs | **95 / 924 (`10.28%` @50px)** | Seasonal foliage, cast rooftop shadows, off-nadir solar angle shift | `test_000977` (2,593 FP px) |
| **E2** | Tiny & Small Structure Miss | Detector (`D03`) | 89 tiny+small pairs ($\le 1,000$ px) | Tiny (1–250 px) **12.99% F1**; Small (<1%) **48.88% F1** | $256 \times 256$ tile resolution floor and dense tree canopy occlusion | `test_000226` (141 GT px, 0 pred) |
| **E3** | Large Low-Contrast / Glare Miss | Detector (`D03`) | 934 changed pairs (10.88M FG px) | Road$\to$BG **15.80%**; Bldg$\to$BG **12.53%** | Bright specular warehouse glare & subtle asphalt resurfacing | `test_001303` (14,009 FN px) |
| **E4** | Road-vs-Building Semantic Swap | Detector (`D03`) | 10.88M foreground Test pixels | **31,042 px (`0.29%` of FG)** | Flat concrete plazas/parking slabs adjacent to highways | `test_001217` (3,492 swap px) |
| **E5** | Captioner Hallucination & Miss | Captioner (`C06`) | 1,858 held-out Test pairs | NC FA **3.57%** (`33/924`); Miss **11.78%** (`110/934`); ObjHall **4.93%** | 388-word train-only vocabulary ceiling & conservative no-change prior | `test_000010` (FP), `test_001307` (FN) |
| **E6** | Out-of-Ontology / Exact-Count / Residual Verifier Error | Hybrid Verifier | 185 claims (150 blind Test pairs) | **23** deterministic defers; **8/132** accepted contradicted (`6.06%` risk); **4** rejected supported | Non-building/road mentions (*"woods"*, *"bareland"*), exact counts, upstream `D03` FPs/FNs | `test_000381`, `test_001729`, `test_001038` |

### 10.2 Detector Error Visual Forensics & Confusion Matrix Analysis

As quantified in **Appendix B** (`m5_confusion_matrices.png`), cross-class confusion between `Road` and `Building` is negligible (`0.23%` of road pixels and `0.32%` of building pixels). Over **97% of detector pixel errors** are foreground-vs-background boundary misses or false alarms. Visual inspection of the worst held-out Test failure cases (**Figure 10.1**, `m5_error_taxonomy_gallery.png`) exposes four distinct physical mechanisms:

![Color-coded TP/FP/FN detector failure gallery across the four dominant held-out Test error modes of D03_bit_lr3e4.](figures/m5_error_taxonomy_gallery.png)

1. **Seasonal Shadow & Roof-Angle Shift (`test_000977`, `test_002083` — Category E1):** Both $T_1$ and $T_2$ show identical residential neighborhoods, but deeper cast shadows and darker winter canopy in $T_2$ shift apparent roof edges, triggering `2,593` false-positive building pixels (`F1 = 0.0%`).
2. **High-Albedo Warehouse Glare & Road Resurfacing (`test_001303`, `test_001393` — Category E3):** In `test_001303`, bright white overexposed roof structures in $T_1$ are replaced by a parking lot and road realignment (`14,009` GT changed pixels), yet `D03` outputs zero changed pixels because the specular brightness saturation breaks ResNet-18 local contrast features. In `test_001393`, subtle highway lane widening under muted grey asphalt (`F1 = 0.0%`) escapes detection.
3. **Concrete Roof / Highway Deck Ambiguity (`test_001217` — Category E4):** Despite strong overall detection (`F1 = 82.5%`), a newly constructed flat grey industrial concrete slab adjacent to a new road is partially labeled as `road` (`3,492` swapped pixels, shown in yellow in Figure 10.1), illustrating why purely local texture cannot always separate flat roofs from wide paved lanes.
4. **Canopy-Occluded Tiny Structures (`test_000226` — Category E2):** A single 141-pixel shed (`0.22%` of tile area) appearing inside dense dark forest receives `0` predicted pixels because ResNet-18 stride-8 feature maps pool away isolated $12 \times 12$-pixel structures surrounded by foliage.

### 10.3 Captioner & Hybrid Verifier Residual Error Analysis

- **Captioner Hallucinations vs Missed Changes (Category E5):** On `33` of `924` unchanged pairs (`3.57%`), `C06` hallucinates small isolated structures (e.g., `test_000010`: *"a small house appears in the bareland"*; `test_000055`: *"a villa appears in the center of the scene"*). Conversely, on `110` of `934` changed pairs (`11.78%`), `C06` predicts *"the two scenes seem identical"* — including high-glare industrial redevelopments such as `test_001303` (`11,259` bldg / `2,750` road px) where *both* `D03` and `C06` are simultaneously blinded by specular glare, and `test_001307` (`22,812` bldg / `3,320` road px) where large construction occurs across an already dense urban grid.
- **Verifier Residual Errors (Category E6 — Why the Remaining `6.06%` Risk Occurs):** Inspecting all **8 accepted-but-contradicted** claims and **4 rejected-but-supported** claims across the 185 blind Test claims shows two cleanly separated failure causes:
  1. **Upstream `D03` Mask Propagation (6 of 8 false accepts, 3 of 4 false rejects):** Because the Mask-First Hybrid trusts confident `D03` masks to avoid unnecessary VLM calls, a `D03` false positive on an unchanged scene (`test_001422`, `test_001787`) or a `D03` false negative on a changed scene (`test_000092`, `test_001250`, `test_001760`) directly propagates into a confident `mask_first_accept` or `mask_first_reject` (`test_000299`, `test_000840`, `test_001417`).
  2. **Escalated VLM Boundary Ambiguity (2 of 8 false accepts, 1 of 4 false rejects):** On `test_001729` (*"the two scenes seem identical"* with residual mask pixels) and `test_001319` (*"a house appears on the bareland"* with `<200 px` mask), the escalated `Qwen3-VL-4B` referee erroneously accepted a contradicted claim; on `test_001038`, it rejected a supported `no_change` claim due to seasonal color drift. Importantly, **zero unverifiable claims were erroneously accepted (`0/19`)**, confirming that deterministic out-of-ontology and exact-count deferral gates (`23` claims deferred cleanly) work with `100%` reliability.
