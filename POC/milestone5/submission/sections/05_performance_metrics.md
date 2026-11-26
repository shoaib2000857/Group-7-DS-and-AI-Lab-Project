## 5. Performance Metrics

Because TerraWatch spans dense semantic segmentation, natural-language generation, and selective decision-making under uncertainty, no single scalar metric captures system quality. Each module is evaluated with domain-standard metrics selected explicitly to counter class imbalance, lexical-overlap blindness, and uncalibrated confidence scores.

### 5.1 Change Detection Metrics (Semantic Segmentation)

LEVIR-MCI Test contains **91.06% background pixels** (`110.88M` background vs `4.12M` road and `6.76M` building pixels). Raw pixel accuracy is therefore trivially inflated (`always_background` scores `91.06%` pixel accuracy with `0.00%` F1). We prioritize foreground-sensitive metrics:

- **Change-Class Mean IoU (Primary Selection Metric):** Macro average of road-change IoU and building-change IoU, explicitly excluding background:
  $$\text{Change mIoU} = \frac{1}{2}\left(\frac{\text{TP}_{\text{road}}}{\text{TP}_{\text{road}} + \text{FP}_{\text{road}} + \text{FN}_{\text{road}}} + \frac{\text{TP}_{\text{bldg}}}{\text{TP}_{\text{bldg}} + \text{FP}_{\text{bldg}} + \text{FN}_{\text{bldg}}}\right)$$
- **Binary Change F1 (Dice Score):** Harmonic mean of pixel precision and recall over the union of road and building change classes ($\text{class} \in \{1, 2\}$), computed globally and stratified across five ground-truth change-area buckets (`unchanged_0`, `tiny_1_250`, `small_251_1000`, `medium_1001_4000`, `large_gt_4000`) plus the standard `<1%` area threshold (`small_change_f1`, $<655$ pixels).
- **Pixel-Level ROC-AUC & PR-AUC (Average Precision):** Computed across all 121.77 million Test pixels using a 10,000-bin non-saturating log-odds score $s = \log(\exp(z_1) + \exp(z_2)) - z_0 \in [-15, 15]$ and exact step-function Average Precision $\sum_k (R_k - R_{k-1}) P_k$. PR-AUC is critical under the 10:1 background skew where ROC-AUC compresses near `0.99`.
- **Unchanged-Scene False Alarm Rate (`FA@T`):** Fraction of the 924 truly unchanged Test pairs where predicted changed pixels reach or exceed threshold $T \in \{1, 16, 50, 100, 200, 500\}$, with **`FA@50px`** as the operational alarm benchmark.

### 5.2 Change Captioning Metrics (Natural-Language Generation)

- **Corpus BLEU-1 through BLEU-4:** Precision-oriented $n$-gram overlap against all 5 human reference captions per pair with brevity penalty (primary Val selection metric matching upstream RSCaMa literature). Reported both globally and restricted to **changed pairs only** (`changed_bleu4`) so boilerplate *"the scene is the same as before"* matches cannot mask weak change descriptions.
- **ROUGE-L:** Longest-common-subsequence F1 capturing structural phrasing fidelity.
- **METEOR:** Unigram alignment with stemming and WordNet synonymy (`nltk` WordNet + OMW-1.4), penalizing fragmentation.
- **CIDEr-D:** TF-IDF weighted $n$-gram consensus across the 5 references per image, rewarding salient change terms (*"houses"*, *"villa"*, *"road"*) over stop words.
- **BERTScore-F1 (`roberta-large`):** Contextual embedding similarity capturing semantic equivalence even when lexical phrasing differs (e.g., *"a new residential area"* vs *"several houses were built"*).
- **Factual Hallucination & Miss Diagnostics:**
  - *Unchanged Hallucination Rate (`NC_FA%`):* Fraction of 924 truly unchanged pairs for which the caption asserts a change.
  - *Missed Change Rate (`Miss%`):* Fraction of 934 truly changed pairs for which the caption predicts *"no change"*.
  - *Object-Class Hallucination Rate (`ObjHall%`):* Fraction of pairs where the caption names `road` or `building` despite $< 50$ ground-truth pixels of that class.

### 5.3 Claim Verification, Calibration, and Human-Audit Metrics

- **Accepted-Claim Precision (Groundedness):** Among claims emitted as `accept`, the fraction supported by ground truth:
  $$\text{Precision}_{\text{accept}} = \frac{\text{Accepted \& Supported}}{\text{Total Accepted}}$$
  Selective Risk is its complement: $\text{Risk} = 1 - \text{Precision}_{\text{accept}}$ (error rate exposed to downstream analysts).
- **Supported Recall & Contradicted Catch Rate:** Fraction of truly supported claims accepted (`recall`) and fraction of truly contradicted (hallucinated) claims explicitly rejected (`catch_rate = Rejected & Contradicted / Total Contradicted`).
- **Coverage (`accept_rate`) and Deferral Rate (`defer_rate`):** Operational throughput vs human-review queue volume.
- **Expected Calibration Error (ECE), Brier Score, and AURC:**
  - $\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ (10-bin reliability gap between predicted support probability and empirical accuracy).
  - $\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2$ (proper scoring rule rewarding both accuracy and calibrated uncertainty).
  - **Area Under the Risk-Coverage Curve (AURC):** Threshold-free measure of selective classification ranking quality (lower is better).
- **VLM Call Rate & Compute Savings:** Fraction of claims escalating to `Qwen3-VL-4B-Instruct`, directly governing inference latency and cloud GPU cost per 1,000 image pairs.
- **Cohen's $\kappa$ (Inter-Rater & Proxy Alignment):** Chance-corrected agreement $\kappa = (p_o - p_e)/(1 - p_e)$ across the 3 blind annotation labels (`supported`, `contradicted`, `unverifiable`), with Wilson 95% confidence intervals on human-verified groundedness.
