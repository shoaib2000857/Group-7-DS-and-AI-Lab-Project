# TerraWatch: Intelligent Satellite Image Change Detection and Analysis

**DS & AI Lab Project (BSDA4001) | Indian Institute of Technology Madras**

**Milestone 1: Problem Definition & Literature Review | Group 7 | Revised submission | 25 October 2026**

| Team member | Student email | GitHub |
|---|---|---|
| Priyanshu Agarwal | 23f2003488@ds.study.iitm.ac.in | 23f2003488 |
| Sarthak Saini | 23f2005347@ds.study.iitm.ac.in | Sarthak-Saini265 |
| Sujal Raj Pradhan | 23f2004759@ds.study.iitm.ac.in | SujalPradhan |
| Shoaib SSM | 24f2000857@ds.study.iitm.ac.in | shoaib2000857 |

## 1. Introduction and Problem Definition

Satellite image comparison reveals building and road changes. Masks localise differences, while captioning and interactive systems describe them [4–6]. However, descriptions can misstate objects, quantities or locations; lighting, viewpoint and alignment differences complicate interpretation.

**Problem statement.** How can two images of the same place be converted into a localised, plain-language description of visible urban change, while identifying claims that the available visual evidence does not support?

TerraWatch retains its focus on detection, grounded descriptions and reliability. A caption and checker may agree with the same incorrect mask. We will test whether inspecting both original images, followed by an accept/reject/defer decision, improves factual reliability without suppressing useful information.

## 2. Scope, Stakeholders and Practical Relevance

**In scope:** roughly co-registered, high-resolution RGB image pairs; labelled building/road change; masks, descriptions and traceable evidence; nuisance tests; and a local demo. Appearance/removal claims require time-specific evidence. Groundwork requires observable, appropriately labelled examples.

**Out of scope:** deciding legality, ownership, intent or cause; unverified damage/impact estimates; nationwide or real-time operation; exact geographic measurements without suitable metadata; and validated long-term trajectories from only two observations. Lower-resolution multispectral monitoring is a separate extension.

| Stakeholder | Need and proposed value |
|---|---|
| Municipal building/estate officers; primary | Review visible construction or encroachment candidates using a mask, concise description and uncertainty flag; determine legality through external records and field checks |
| Urban planners; secondary | Compare visible built-up changes across image pairs |
| Remote-sensing researchers and non-specialist users; secondary | Inspect reproducible outputs and understand what the images do and do not establish |

The intended value is better review support. Reduced officer workload is a potential benefit, not an established outcome; demonstrating it would require a separate user study.

## 3. Objectives

**O1:** train and evaluate a credible change-detection baseline. **O2:** assess sensitivity to nuisance variation and domain shift. **O3:** test whether image-backed claim checking improves factual correctness while retaining informative content. **O4:** deliver a reproducible local pipeline with measured runtime and memory. Section 9 defines how these objectives will be assessed.

<!-- PAGEBREAK -->

## 4. Literature Review: Study-Comparison Matrix

The matrix separates mechanisms from evaluation. A paper's use of conventional caption scores does not establish an absence of grounding or verification.

| Study | Objective | Datasets / setting | Experimental design | Main finding | Limitation relevant to TerraWatch |
|---|---|---|---|---|---|
| FC-Siam [1]; Reality Check [3] | Establish effective detection baselines | Urban change benchmarks | Early fusion/Siamese networks; controlled U-Net comparisons | Simple CNN baselines remain important | Detection scores alone do not validate a description |
| ChangeFormer [2] | Model multiscale change context | LEVIR-CD and DSIFN-CD | Siamese hierarchical transformer; detector comparisons | Strong published detection baseline | Label/domain dependence; no standalone claim verifier |
| Chg2Cap [7]; RSCaMa [8] | Describe image-pair changes | LEVIR-CC | Attentive captioning; spatial/temporal state-space features | Specialist change representations support captioning | Caption similarity does not fully establish factual correctness |
| Change-Agent [4] | Interactive comprehensive change analysis | LEVIR-MCI | Joint mask/caption perception, feature interaction and LLM tool use; ablations | Integrates pixel, text and object-statistic outputs | Shared perception errors can propagate; evaluate checking under faulty evidence |
| ChangeChat [5] | Interactive change questions | LEVIR-derived instructions (87k) | Rule/GPT-assisted instruction tuning; caption, count and localisation tasks | Supports task-specific interactive analysis | Generated supervision and shared imagery require provenance and error audits |
| CCExpert [6] | Improve MLLM change understanding | LEVIR-CC; CC-Foundation | Difference-aware integration, multiscale features and staged training | Explicit difference features improve reported caption metrics | Grounding is already prior art; gains do not establish calibration under domain shift |
| KCFI [9]; TEOChat [10] | Detection-guided captioning; temporal EO dialogue | LEVIR-CC; multi-task EO instruction data | Pixel-task-constrained features; temporal instruction tuning | Visual guidance and temporal multi-task modeling are established | Broad capabilities do not substitute for a controlled claim-checking experiment |
| Ricci et al. [11] | Zero-shot descriptions and fact matching | SECOND; LEVIR-CC | Pretrained LLM/VLM approaches; FMScore | Fact-oriented evaluation already exists in this task | Reference-fact matching and inference-time visual verification answer different questions |
| Spot-the-shift [12] | Evaluate grounded difference descriptions | Street-view structural-change pairs | Human-verified change units, spatial annotations and no-change evaluation | Atomic factual/coverage evaluation is established | Different domain; its principles inform, rather than establish novelty for, our protocol |
| TerraScope [26] | Grounded EO reasoning | EO reasoning; building change estimation | Visual reasoning coupled with segmentation; task comparisons | Pixel grounding is an established reasoning approach | Grounded reasoning alone is not novel; test the additional claim-decision policy |

**Published reference points.** Change-Agent reports 86.43 mIoU, 65.95 BLEU-4 and 140.29 CIDEr-D for its MCI model on LEVIR-MCI [4, Table II]. Within that table, Chg2Cap reaches 64.39 BLEU-4 and 136.61 CIDEr-D. These are literature results, not TerraWatch measurements or targets. Cross-paper scores are not treated as directly comparable without matching splits and evaluation scripts.

**Current practice.** Image differencing offers a transparent diagnostic but confounds appearance differences with change. Learned detectors improve spatial prediction; captioners and agents add semantic interpretation. The remaining question is how reliably an output can be checked when its own evidence source is wrong. Generic models' capability claims are not evidence of satellite-domain accuracy.

<!-- PAGEBREAK -->

## 5. Research Gap and Proposed Contribution

### 5.1 What prior work already addresses

Integration and tool use are established in Change-Agent; interactive counting/localisation in ChangeChat; difference-aware visual conditioning in CCExpert and KCFI; and factual evaluation in FMScore and grounded change-unit evaluation [4–6,9,11,12]. Modularity, mask conditioning and human evaluation are therefore design choices or established practices, not independent novelty claims.

### 5.2 Specific mechanism to investigate

We propose **disagreement-triggered visual reinspection**. Each description is decomposed into claims linked to candidate regions. Consistency, visibility and uncertainty checks identify conflicts or insufficient evidence. Triggered claims are queried on corresponding crops from both original images, with full-image context retained; a validation-designed audit of untriggered claims measures missed errors. A separate decision layer uses these observations, quality flags and mask evidence to accept, reject or defer each claim. Deferred claims are qualified or withheld, with the reason preserved.

A binary mask cannot establish object identity or addition/removal direction. Region counts are not building counts. The decision layer therefore cannot approve a claim solely because the caption matches the mask. Its thresholds are selected on validation data, and its visual observations remain fallible predictions.

### 5.3 Mechanism-level comparison

| System | How visual evidence informs output | Specific distinction examined here |
|---|---|---|
| Change-Agent [4] | Joint detection/caption features and agent tools | Hold perception fixed and test acceptance of claims when that perception is faulty |
| ChangeChat [5] | Instruction-tuned paired-image analysis | Compare free-form answers with explicit claim decisions and deferral |
| CCExpert [6] | Difference-aware features integrated before language decoding | Compare generation-time grounding with additional targeted reinspection after generation |
| TerraWatch; proposed | Claim-to-region links, disagreement triggers, original-image reinspection and bounded decisions | Test whether reinspection adds information beyond mask consistency at useful coverage |

This comparison does not assert that prior systems lack all checking. The experimental distinction is the explicitly isolated reinspection-and-decision policy and its response to perception errors.

### 5.4 Contribution and supporting evidence

**Primary hypothesis:** image-backed checking yields better human-judged factual precision and reference-fact coverage than mask-only checking on identical candidate descriptions, particularly for naturally incorrect masks. Targeted crops offer evidence unavailable to a mask-only checker; constrained decisions allow uncertainty to be represented rather than converted into confident prose. Prior work supports these ingredients, but their benefit in this proposed configuration must be demonstrated.

The intended contribution is a reliability mechanism evaluated through a detector-error-aware protocol. It is currently a proposed adaptation and empirical investigation, not an established first-of-kind architecture. A meaningful advance requires improvement over mask-only checking and a simple image-backed judge, while controlling retained content, compute and domain shift. If these comparisons show no gain, the contribution will be reported as an empirical analysis of failure modes.

<!-- PAGEBREAK -->

## 6. Proposed Approach and Technical Grounding

The existing detector-to-description pipeline is retained. Four learned roles are considered; preprocessing, schema validation and deterministic checks remain explicit software components.

**Image pair → quality/scale checks → change detector → evidence-linked caption → visual reinspection → accept/reject/defer → report.**

| Role | Candidate and training strategy | Output |
|---|---|---|
| Change perception | Compact early-fusion/Siamese U-Net trained as a baseline; ChangeFormer as a stronger comparison [1–3] | Change probabilities, class masks where supervised, candidate regions |
| Change description | Specialist baseline such as Chg2Cap/RSCaMa; Qwen3-VL as a candidate for multi-image prompting or adapter tuning [7,8,20] | Short caption and structured claim spans |
| Visual evidence inspection | Frozen VLM initially; inspect matched T1/T2 crops and context | Object, action, location observations and visibility/ambiguity flags |
| Decision policy | Rules baseline; Clef/Clef-flash or another bounded classifier; Jev as an optional comparator [21–23] | Claim acceptance, contradiction or deferral and logged decision scores |

**Input adaptation.** Inspect dimensions, channels, metadata, blur, saturation and alignment. Apply model-required normalisation, paired tiling and coordinate-preserving resizing. Use declared ground-sampling distance when available; do not infer physical resolution from image dimensions. Radiometric adjustment is optional and compared against unchanged inputs because histogram matching can remove genuine change. Retain raw images and transformation metadata; reject incompatible modalities rather than attempting universal automatic correction.

**Learning plan.** Train a compact detector and compare BCE-plus-Dice with a focal-loss alternative on audited class distributions. Use synchronised spatial augmentations and controlled one-date appearance perturbations. Fine-tune only selected caption/visual adapters if a frozen baseline proves inadequate and compute permits. The visual inspector may initially reuse frozen VLM weights; a different backbone is an ablation, not an assumption of independent errors. A decision model is retained only if it improves held-out decisions over rules and a general VLM judge.

**Decision-model rationale.** Clef is an open decision-model family; current Ollama listings include multimodal variants [21,22]. Jev's October release introduces experimental image support [23]. These make bounded claim decisions feasible candidates, but structured responses and confidence scores do not prove factual correctness. Recent Jev evaluation identifies wording/order sensitivity [24]; we will test calibration and decision stability on this task. No remote-sensing reliability gain is assumed from unrelated vendor benchmarks.

**Reporting safeguards.** Preserve original/revised captions, evidence links and deferral reasons. Exact counts require validated instance evidence; geographic area requires suitable spatial metadata. A crop's upper-right is image-relative, not geographic northeast unless orientation is known. Predicted evidence and human reference annotations remain separate.

<!-- PAGEBREAK -->

## 7. Data Availability, Alignment and Robustness Strategy

LEVIR-MCI is the preferred primary dataset because it extends LEVIR-CC with aligned change masks; the shared family has 10,077 image pairs and five captions per pair [4,13]. Dataset access, label dictionaries and asset-specific usage terms will be audited before training. Source links have been identified; complete payload/rights validation is not claimed.

The broader plan considers a portfolio of nine releases. It is not nine independent populations and will not be concatenated indiscriminately.

| Data group | Candidate releases and supervision | Use / boundary |
|---|---|---|
| Aligned detection and text | LEVIR-MCI, LEVIR-CC [4,13] | Core training/evaluation; same underlying pairs counted once |
| Building change | LEVIR-CD, LEVIR-CD+, S2Looking [14,15] | Compatible detection supervision after overlap audit; hold out a domain for transfer |
| Broader visual change | SYSU-CD [16] | Separate transfer evaluation; reconcile broader label coverage before joint training |
| Semantic transitions | SECOND [17] | Optional task-specific head; not interchangeable with binary building labels |
| Change instructions | ChangeChat-105k [18] | Optional adapter data; trace LEVIR-derived images and instruction provenance |
| Lower-resolution urban change | OSCD [19] | Separate multispectral extension; not pooled with high-resolution RGB as equivalent building evidence |

**Combination strategy.** Harmonise image ordering, modality, units, label ontology and parent-scene IDs first. A shared backbone may use task/domain-specific adapters or heads. Apply losses only where the corresponding annotation exists; an absent caption or semantic label is not a negative label. Sample domains explicitly to prevent large datasets dominating. Compare single-domain training, compatible mixed training and mixed training plus domain-aware adaptation. More data does not guarantee robustness to unseen imagery.

**Leakage control.** Preserve official train/validation/test membership and group related crops by parent scene where possible. Check hashes, near duplicates and shared provenance across releases; keep all captions/instructions of a pair together. Reserve an entire compatible dataset/domain for transfer testing rather than training on every source. Track pretrained checkpoint exposure to LEVIR-derived images.

**POC limit.** Use 20% of unique pairs by default and no more than 30% of any selected dataset, cumulatively across development, training, annotation and evaluation. Shared-image releases do not multiply independent samples. Prefer pretrained inference and short learning pilots; full training belongs to the subsequent project phase.

## 8. Feasibility and Key Risks

Start with one dataset, compact detection and one description baseline; demonstrate 5–10 pairs end to end before expanding. Select image size, batch size, quantisation and adapters after measuring device memory and runtime. Large multimodal models are optional, not prerequisites. Open/local baselines keep the core reproducible without a paid service.

Main risks are shared detector/verifier errors, ambiguous/touching buildings, incompatible labels, duplicate scenes, destructive normalisation and excessive model cost. Mitigations are original-image inspection, uncertainty/deferral, label-aware heads, provenance audits, preprocessing ablations and staged model selection. Real seasonal/viewpoint robustness requires suitable held-out examples; synthetic corruptions alone cannot establish it.

<!-- PAGEBREAK -->

## 9. Evaluation and Definition of Success

Use official splits, validation-only tuning, frozen test protocols and identical input pairs across comparisons. Report uncertainty by paired resampling at scene level where possible, otherwise at pair level with the dependence limitation stated. Published scores provide context, not fair full-data targets for a small POC.

| Objective | Measures | Evidence of success |
|---|---|---|
| O1: detection | Change-class precision/recall, F1 and IoU; class mIoU where applicable | Reproducible learned baseline and fair comparison on the same split; inspect small-object and no-change failures |
| O2: robustness | Clean-to-corrupted F1/claim-quality drop; separate held-out-domain scores | Smaller degradation than the matched baseline without material clean-quality loss; margins fixed from validation before test |
| O3: reliable descriptions; primary | Human supported-claim precision, reference-fact recall and their harmonic mean; precision at matched report coverage | Improvement over mask-only checking with retained information controlled; paired confidence intervals distinguish demonstrated improvement from inconclusive effects |
| O4: practicality | End-to-end median/p95 latency, cold load, peak memory and setup reproducibility | Runnable documented demo on stated hardware; added checking cost reported against its factual benefit |

**Factual audit.** Plan at least 100 held-out pairs with two independent annotators, subject to a pilot workload/power assessment. Establish reference facts from original images and available labels without predicted masks, then judge anonymised outputs. Label atomic claims supported, unsupported, contradicted or unverifiable; adjudicate disagreements and report agreement. Supported-claim precision uses all emitted factual claims as its denominator. Reference-fact recall measures observable, in-scope facts conveyed. Empty descriptions have zero recall and no defined precision, not perfect reliability. Report omission, abstention and false-change rates separately.

**Ablations.** Compare: (A) images-only caption; (B) predicted-mask conditioning; (C) B plus mask consistency; (D) C plus direct image-backed judgment; (E) C plus disagreement-triggered reinspection and decision policy. Hold detector and candidate captions fixed for C–E; compare targeted and full inspection at matched evidence budgets. Add an oracle-mask diagnostic, natural detector-error strata and separately labelled injected mask/claim errors. Compare rules with the decision model using identical evidence. Stress decision wording and option order. Nuisance tests cover brightness/colour, blur/compression and small misalignment with valid overlap defined.

**Standard and proposed overall measures.** BLEU-4, METEOR, ROUGE-L and CIDEr-D remain supplementary caption measures. Region-count MAE is distinct from building-count MAE, which requires instance annotations. We also plan an exploratory end-to-end grounded-change score combining localisation, correct object/action claims, retained coverage and robustness. Its formula, matching rules and weighting will be fixed on development data before final evaluation and validated against blinded human rankings. Until validated, it is not a primary success criterion and cannot replace the component metrics.

**Decision rule.** The research hypothesis is supported only if the proposed checker improves factual quality over the controlled baselines at comparable useful coverage; otherwise the result is inconclusive or negative. Arbitrary promises of a 10-point gain, 25% robustness improvement or five-second runtime are replaced by measured effects and explicit uncertainty.

## 10. Milestone Plan

**M2, Oct 29:** provenance, label/split audits and bounded subset. **M3, Nov 5:** justified architecture, contracts and end-to-end subset. **M4, Nov 19:** detector training, bounded adapter experiments and logs. **M5, Nov 26:** frozen ablations, factual audit and failures. **M6, Dec 10:** reproducible demo and documentation [25]. Individual contributions will be tracked per milestone. At M1, models, mixed-data training and reliability improvements remain proposed rather than measured results.

<!-- PAGEBREAK -->

## References

1. Daudt, R. C., Le Saux, B. & Boulch, A. (2018). *Fully Convolutional Siamese Networks for Change Detection*. ICIP. https://arxiv.org/abs/1810.08462
2. Bandara, W. G. C. & Patel, V. M. (2022). *A Transformer-Based Siamese Network for Change Detection*. IGARSS. https://arxiv.org/abs/2201.01293
3. Corley, I., Robinson, C. & Ortiz, A. (2024). *A Change Detection Reality Check*. https://arxiv.org/abs/2402.06994
4. Liu, C. et al. (2024). *Change-Agent: Toward Interactive Comprehensive Remote Sensing Change Interpretation and Analysis*. IEEE TGRS. https://arxiv.org/abs/2403.19646
5. Deng, P., Zhou, W. & Wu, H. (2024). *ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning*. https://arxiv.org/abs/2409.08582
6. Wang, Z. et al. (2024). *CCExpert: Advancing MLLM Capability in Remote Sensing Change Captioning with Difference-Aware Integration and a Foundational Dataset*. https://arxiv.org/abs/2411.11360
7. Chang, S. & Ghamisi, P. (2023). *Changes to Captions: An Attentive Network for Remote Sensing Change Captioning*. https://arxiv.org/abs/2304.01091
8. Liu, C. et al. (2024). *RSCaMa: Remote Sensing Image Change Captioning with State Space Model*. IEEE GRSL. https://arxiv.org/abs/2404.18895
9. Yang, C. et al. (2024). *Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning*. https://arxiv.org/abs/2409.12612
10. Irvin, J. A. et al. (2025). *TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data*. ICLR. https://arxiv.org/abs/2410.06234
11. Ricci, R., Bazi, Y. & Melgani, F. (2026). *Remote sensing change captioning meets large language and vision models*. ISPRS JPRS, 239, 793–807. https://doi.org/10.1016/j.isprsjprs.2026.06.003
12. Liberatori, B. et al. (2026). *Spot-the-shift: Evaluating Grounded Image Difference Captioning of Long-term Changes*. Preprint. https://arxiv.org/abs/2609.10356
13. Liu, C. et al. *LEVIR-CC / LEVIR-MCI author dataset sources*. https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset ; https://huggingface.co/datasets/lcybuaa/LEVIR-MCI
14. Chen, H. & Shi, Z. (2020). *LEVIR-CD dataset and associated study*. https://justchenhao.github.io/LEVIR/
15. S2Looking authors. *S2Looking and LEVIR-CD+ datasets*. https://github.com/S2Looking/Dataset
16. Shi, Q. et al. (2021). *SYSU-CD dataset and DSAMNet study*. https://github.com/liumency/SYSU-CD
17. Yang, K. et al. *SECOND: Semantic Change Detection with Asymmetric Siamese Networks*. https://captain-whu.github.io/SCD/
18. Wu, H. *ChangeChat-105k release*. https://huggingface.co/datasets/hlwu/changechat-105k
19. Daudt, R. C. et al. *Onera Satellite Change Detection dataset*. https://rcdaudt.github.io/oscd/
20. Qwen team. *Qwen3-VL-4B-Instruct model card*. https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
21. Cloudflare (2026). *Introducing Clef: our open-source decision models, and new RL fine-tuning platform*. https://blog.cloudflare.com/clef-decision-models/
22. Ollama (2026). *Clef / Clef-flash model documentation*. https://ollama.com/library/clef ; https://ollama.com/library/clef-flash
23. TypeSafe AI (2026). *Jev: For the builders AND dreamers*; Jev 1.14.0 and experimental image support. https://typesafe.ai/blog/jev-1-14
24. Imperatrice, I. et al. (2026). *Are AI Decision Models Biased? A Case Study of Jev*. Preprint. https://arxiv.org/abs/2610.19619
25. IITM DS & AI Lab. *Project Guidelines and Milestones*. https://bsc-iitm.github.io/dsai-lab-project/guidelines.html ; https://bsc-iitm.github.io/dsai-lab-project/milestones.html
26. Shu, Y. et al. (2026). *TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation*. CVPR, 16712–16722. https://openaccess.thecvf.com/content/CVPR2026/supplemental/Shu_TerraScope_Pixel-Grounded_Visual_CVPR_2026_supplemental.pdf

Online sources accessed 24–25 October 2026. Model variants, access conditions and dataset terms will be pinned before experimentation.
