# TerraWatch M1: team explanation and defence guide

**Internal preparation document | 25 October 2026 | Not part of the submission report**

## 1. What changed, and what stayed the same

The original problem, urban-change scope, stakeholders, detector, captioner and local demo remain. The main revision specifies how checking works and how to test its value. The report also adds a staged data portfolio, careful input adaptation and an optional decision model. These are proposals, not features already implemented.

The report retains the original M1 content areas while consolidating repeated sections. Arbitrary performance targets, universal robustness, “all tools stop at masks” and “no prior factual evaluation” claims are removed. This is a substantive clarification of the mechanism, not a claim that a novel architecture has already been proven.

## 2. The strongest defensible contribution

Say: **“We investigate whether targeted reinspection of the original image pair improves claim decisions when predicted change evidence is imperfect, compared with mask-only checks and a direct visual judge.”**

The proposed mechanism is: extract claims → link candidate regions → identify disagreement/insufficient evidence → reinspect both dates with context → accept/reject/defer → preserve evidence and decision history.

What makes this more than assembly is the controlled question: does the policy add measurable information beyond its components? Swapping model names or adding a fourth model is not enough. The key experiment fixes detector outputs and candidate captions, then varies only checking. A direct image-backed judge with the same evidence budget is a necessary baseline. Otherwise any gain could simply come from spending more inference compute.

Do not say “first” or “hallucination-free.” The contribution can legitimately be a useful adaptation and empirical study. Establishing stronger methodological novelty needs full comparison with the closest mechanisms and positive ablations.

## 3. Direct responses to the TA

| Likely question | Defensible answer / evidence required |
|---|---|
| Why is modularity novel? | It is not claimed as novel. It allows detector and checker interventions while holding other stages fixed. |
| Why isn't this Change-Agent? | Change-Agent integrates perception and tools. Our proposed experiment isolates claim acceptance under faulty predicted evidence; demonstrate the distinction with identical-input ablations, not a claim that it lacks all verification. |
| Doesn't CCExpert already ground text? | Yes. Generation-time difference features are established. We test an additional post-generation reinspection/decision policy. |
| What does ChangeChat already do? | Interactive change captioning, quantification and localisation. Those capabilities are not claimed as our invention. |
| Why not just compare the sentence to the mask? | Agreement can repeat a detector error. We evaluate against original-image human references, including naturally wrong masks. |
| What if the mask and caption are both confidently wrong? | A disagreement-only trigger can miss them. Add quality/uncertainty triggers and a validation-designed sample of apparently accepted claims for inspection. Measure missed-trigger errors explicitly. |
| Why should the decision model be better? | It might not be. Compare rules, a general VLM judge and the decision model on identical evidence. Keep it only if the tradeoff improves. |
| What is new about the metric? | No new metric is claimed as validated. The exploratory pipeline score is a development objective; primary evidence remains interpretable component metrics. |

<!-- PAGEBREAK -->

## 4. Metrics: relevance, goals and achievable evidence

### Detection and robustness

F1 = 2TP/(2TP+FP+FN); IoU = TP/(TP+FP+FN), for changed pixels within valid regions. Overall accuracy is misleading when unchanged pixels dominate. State micro versus scene-macro aggregation and empty-mask conventions. mIoU requires an audited class dictionary and explicit background inclusion.

The attainable M1 commitment is a reproducible fair baseline, not a full-data leaderboard score from a subset. Robustness goals are relative to the same detector trained without the proposed intervention: compare clean quality and degradation under a fixed corruption set, then a genuinely held-out domain. Freeze the acceptable clean-quality margin using validation repeatability and application needs, not test results. If the margin is not set by M3, do not present a formal non-inferiority claim.

### Factual quality and useful coverage

- **Supported-claim precision P:** supported claims divided by all emitted factual claims. Unsupported, contradicted and unverifiable claims remain in this strict denominator.
- **Reference-fact recall R:** unique in-scope observable reference facts correctly conveyed divided by the adjudicated reference inventory. Deduplicate paraphrases so verbosity cannot inflate recall.
- **Factual F1:** 2PR/(P+R), using a declared aggregation level. Set F1 to zero for an empty report on a changed pair; precision itself is undefined. Handle unchanged pairs separately rather than inventing positive reference facts.
- **Report coverage:** fraction of eligible pairs receiving an accepted informative description. Compare factual precision at matched coverage so rejecting almost everything cannot masquerade as progress.
- Also report claim retention, omission, contradiction, unverifiability, and false-change reports on unchanged pairs. Building counts require validated instances; connected components support only region counts.

**Primary success test:** paired improvement over mask-only checking and a direct visual judge, with matched useful coverage and no hidden loss of reference-fact recall. A confidence interval wholly above zero for the preregistered main contrast supports improvement; an interval spanning zero is inconclusive. Report effect sizes even if they are smaller than the old +10-point aspiration. Choose one primary contrast and measure at pair/scene level to avoid treating correlated claims as independent samples.

These goals are achievable as experiments because they compare the same images and captions with a small number of checker variants. Positive improvement is not guaranteed. A 100-pair audit is an initial plan, not a power calculation; use pilot variance, error prevalence and rater workload to estimate the required main sample before freezing it.

### Runtime and calibration

Measure cold load, warm median/p95 pipeline latency, peak memory and additional checker cost. Synchronise GPU timing. Cached upstream predictions are appropriate for checker comparisons but must not be called fresh end-to-end timing. Do not restore the five-second target without a measured basis.

For decision probabilities, examine reliability diagrams, Brier score and calibration error on an annotated validation set. ECE depends on binning and sample size; show counts. A confidence/concentration field is not automatically a probability of correctness. Thresholds must be tuned on validation and frozen before test. Repeat claims with paraphrases and reordered choices to test stability.

<!-- PAGEBREAK -->

## 5. How to develop the custom pipeline metric

Keep it exploratory until validated. A simple weighted average lets strong caption fluency hide bad detection, while a geometric product can double-count correlated errors or collapse for legitimate abstention. Different label sets cannot support the same dimensions.

Recommended development process:

1. Define an event as an object/type, action, location and evidence region. Determine which datasets genuinely annotate those fields.
2. On an independently annotated development set, match predicted/reference events one-to-one with declared spatial/semantic criteria. Penalise duplicates, wrong actions and false changes; measure missed events.
3. Consider a grounded-event precision/recall/F1 core with localisation as a match requirement, rather than averaging unrelated scores. State how ambiguous counts, no-change pairs and deferrals behave.
4. Evaluate robustness by reporting this score separately per corruption/domain. Consider worst-domain or macro-domain summaries only with sufficient support; do not let the largest dataset dominate.
5. Validate rankings against blinded human preferences on held-out development judgments. Stress-test empty output, verbose repetition, shifted boxes, correct words/wrong region, wrong action and corrupted evidence.
6. Freeze the formula, thresholds and weights before final test. Retain all component metrics. A custom score is a useful benchmark contribution only if it reliably exposes failures existing measures miss.

No numeric formula or score in this guide is a validated TerraWatch result. The report deliberately avoids promising that an undefined metric establishes success.

## 6. Data strategy: what “8–10 datasets” can honestly mean

The report lists nine candidate releases, not nine independent distributions. LEVIR-MCI/CC and instruction derivatives share imagery; LEVIR-CD+ also needs overlap checks. More releases can mean more labels for the same scenes rather than more geographic diversity.

Start with MCI. Add compatible building-change sources to detector training only after a parent-scene audit. Use task-specific losses/heads for semantic and text supervision; missing labels are masked out, not treated as absent change. Keep a whole domain unseen for transfer. SECOND's semantic labels and SYSU's broader change definitions cannot be silently collapsed into building truth. OSCD has 24 Sentinel-2 multispectral pairs and remains a different scale/modality track [report ref. 19].

Count the union of unique pairs across POC use. Freeze 20% initially, never exceed 30% per selected source, and keep original partition membership. A pretrained model trained on the full source is fine for an inference pilot, but not evidence of learning from only 20% of data.

Input adaptation can reconcile tensor shapes and documented radiometry, not recover roofs absent at the sensor's resolution. Never infer metres-per-pixel from width/height alone. Preserve raw imagery and compare adjusted/unadjusted inputs. Naive histogram matching, aggressive denoising or registration can erase true changes.

### Immediate team preparation

Agree on the primary contrast, one feasible VLM and a data-access owner. Verify checkpoint training exposure and rights. Inspect 30–50 development pairs and estimate annotation workload. By M3 freeze matching rules, metrics and validation-selected thresholds. Maintain contribution logs with actual artifacts; this M1 report does not contain completed experiments.

<!-- PAGEBREAK -->

## 7. Model selection and verified caveats

Source checks were performed on 25 October 2026. See the report references for primary links; model capabilities are candidate-selection evidence, not project results.

| Candidate | Appropriate role | Important qualification |
|---|---|---|
| Compact U-Net/Siamese; ChangeFormer | Trainable detection core and comparison | Fair split, ontology, preprocessing and compute budget matter more than adding model count |
| Chg2Cap / RSCaMa | Specialist caption baseline | RSCaMa uses state-space change modeling; measure runtime rather than accepting “lightweight” as a proven local property |
| Qwen3-VL 4B-class | Initial general multi-image caption/inspection candidate | Dynamic resolution increases token/memory cost; no satellite accuracy guarantee [report ref. 20] |
| Gemma-family multimodal model | Alternative visual inspector | Google's vision documentation describes model-specific resolution handling; pin a particular model and processor before comparing: https://ai.google.dev/gemma/docs/capabilities/vision |
| Clef / Clef-flash | Bounded decisions on supplied evidence | Ollama lists 27B/9B multimodal variants. Listed confidence measures distribution concentration, not guaranteed correctness [report refs. 21–22] |
| Jev 1.14.0 | Optional decision comparator | Official image support is experimental; it is not a guaranteed visual verifier. Access and costs need checking before any call [report ref. 23] |
| TEOChat | Domain-specific multi-task comparator | Temporal EO instruction model; its existence is a reason to qualify broad temporal novelty [report ref. 10] |

No models were downloaded, fine-tuned or benchmarked for this report. No paid calls were made. Ollama is a runtime/distribution route; vLLM is an inference engine, not a separate vision-language model. A vision transformer encoder is not by itself a captioning system.

The Gemini-generated model list is informal inspiration. Do not repeat “gold standard,” “exact structural changes,” “ignores seasonal noise,” “best out-of-the-box accuracy” or “completely eliminates hallucinations.” Those are unsupported generalisations.

## 8. Further defence and experiment details

- **Trigger baseline:** compare targeted inspection with random inspection and direct inspection of all claims at matched call/token budget. Attribute gains to the policy only if these controls support it.
- **Correlated errors:** separate models can share training data or architecture. A second opinion is corroboration, not independent truth. Use original-image human references and a distinct-backbone ablation where feasible.
- **Counterfactuals:** a shuffled/displaced mask tests evidence dependence, but synthetic error rates are not deployment error rates. Report natural failures separately.
- **Raters:** establish reference facts before viewing predictions; anonymise output conditions, randomise ordering, keep original disagreements and adjudications. Report raw agreement as well as kappa.
- **Compute:** three or four roles need not imply four giant resident models. Reuse frozen VLM weights initially, run stages sequentially and cache comparisons. Parameter-efficient fine-tuning still needs memory estimates.
- **Later prior art:** the search also identified TerraScope, CVPR 2026, which combines pixel grounding and EO reasoning. Its supplemental describes building-change evaluation. Review it before any stronger novelty claim: https://openaccess.thecvf.com/content/CVPR2026/supplemental/Shu_TerraScope_Pixel-Grounded_Visual_CVPR_2026_supplemental.pdf
