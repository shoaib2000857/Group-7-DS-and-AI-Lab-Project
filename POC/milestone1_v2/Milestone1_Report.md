# TerraWatch: Intelligent Satellite Image Change Detection and Analysis

**DS & AI Lab Project (BSDA4001) | Indian Institute of Technology Madras**

**Milestone 1: Problem Definition & Literature Review | Group 7 | Revised submission | 26 October 2026**

<p align="center"><img src="assets/iitm_logo.png" alt="IIT Madras emblem" width="130"></p>

| Team member | Student email | GitHub |
|---|---|---|
| Priyanshu Agarwal | 23f2003488@ds.study.iitm.ac.in | 23f2003488 |
| Sarthak Saini | 23f2005347@ds.study.iitm.ac.in | Sarthak-Saini265 |
| Sujal Raj Pradhan | 23f2004759@ds.study.iitm.ac.in | SujalPradhan |
| Shoaib SSM | 24f2000857@ds.study.iitm.ac.in | shoaib2000857 |

## 1. Introduction and Problem Definition

Satellite images captured at different dates can reveal building construction, removal and road expansion. Change-detection models locate these differences, while captioning and interactive systems describe them [4–6]. However, a description may identify the wrong object, count or location. Differences in lighting, viewpoint and alignment can also be mistaken for physical change.

**Problem statement.** How can two images of the same place be converted into a localised, plain-language description of visible urban change, while identifying claims that the available visual evidence does not support?

TerraWatch combines change detection with evidence-based descriptions. Its central question is whether checking a description against both original images improves reliability beyond checking agreement with a predicted mask alone. This matters because a caption and a checker can agree with the same incorrect detection.

## 2. Scope, Stakeholders and Practical Relevance

**In scope**

- Two approximately aligned, high-resolution RGB images of the same location.
- Visible building and road changes, with a change mask and a short, evidence-linked description.
- Tests under lighting, blur and alignment variations, followed by a local demonstration.

**Out of scope**

- Determining legality, ownership, intent or the cause of a change.
- Real-time monitoring, damage estimates or complete development histories from two images.
- Geographic measurements without suitable spatial metadata.

| Stakeholder | Intended benefit |
|---|---|
| Municipal building and estate officers; primary | Prioritise visible changes for further review |
| Urban planners; secondary | Compare building and infrastructure changes |
| Researchers and non-specialist users; secondary | Inspect reproducible results and understandable descriptions |

The system is intended to help reviewers understand and prioritise change alerts. Any reduction in review workload would require a separate user study.

## 3. Objectives

- **O1 - Detection:** train and evaluate a standard change-detection model.
- **O2 - Robustness:** measure performance under image-quality variations and on a separate compatible dataset.
- **O3 - Reliable descriptions:** test whether image-based checking reduces incorrect statements without omitting relevant changes.
- **O4 - Practicality:** deliver a reproducible local demonstration and measure runtime and memory use.

<!-- PAGEBREAK -->

## 4. Literature Review: Study-Comparison Matrix

The following studies establish the detection, captioning and verification methods relevant to TerraWatch. Their reported evaluation measures are considered separately from their model mechanisms.

| Study | Objective | Datasets / setting | Experimental design | Main finding | Limitation relevant to TerraWatch |
|---|---|---|---|---|---|
| FC-Siam [1]; Reality Check [3] | Establish effective detection baselines | Urban change benchmarks | Early fusion/Siamese networks; controlled U-Net comparisons | Simple CNN baselines remain important | Detection scores alone do not validate a description |
| ChangeFormer [2] | Model multiscale change context | LEVIR-CD and DSIFN-CD | Siamese hierarchical transformer; detector comparisons | Strong published detection baseline | Label/domain dependence; no standalone claim verifier |
| Chg2Cap [7]; RSCaMa [8] | Describe image-pair changes | LEVIR-CC | Attentive captioning; spatial/temporal state-space features | Specialist change representations support captioning | Caption similarity does not fully establish factual correctness |
| Change-Agent [4] | Interactive change analysis | LEVIR-MCI | Joint mask/caption perception and LLM tools; component comparisons | Reports 86.43 mIoU and 140.29 CIDEr-D in Table II | Shared perception errors may affect downstream descriptions |
| ChangeChat [5] | Interactive change questions | LEVIR-derived instructions (87k) | Rule/GPT-assisted instruction tuning; caption, count and localisation tasks | Supports task-specific interactive analysis | Generated supervision and shared imagery require provenance and error audits |
| CCExpert [6] | Improve MLLM captions | LEVIR-CC; CC-Foundation | Difference-aware integration, multiscale features and staged training | Explicit difference features improve reported caption metrics | Grounding is already prior art; gains do not establish calibration under domain shift |
| KCFI [9]; TEOChat [10] | Guide captions; temporal EO dialogue | LEVIR-CC; multi-task EO instruction data | Pixel-task-constrained features; temporal instruction tuning | Visual guidance and temporal multi-task modeling are established | Broad capabilities do not substitute for a controlled claim-checking experiment |
| Ricci et al. [11] | Zero-shot descriptions and fact matching | SECOND; LEVIR-CC | Pretrained LLM/VLM approaches; FMScore | Fact-oriented evaluation already exists in this task | Reference-fact matching and inference-time visual verification answer different questions |
| Spot-the-shift [12] | Evaluate grounded difference descriptions | Street-view change pairs | Human-verified change units, spatial annotations and no-change evaluation | Atomic factual/coverage evaluation is established | Different domain; its principles inform, rather than establish novelty for, our protocol |
| TerraScope [25] | Grounded EO reasoning | EO reasoning; building change estimation | Visual reasoning coupled with segmentation; task comparisons | Pixel grounding is an established reasoning approach | Grounded reasoning alone is not novel; test the additional claim-decision policy |

**Current practice and opportunity.** Detectors locate changes; captioners and agents explain them. Existing systems already use visual evidence, so the contribution cannot simply be combining these components. TerraWatch will examine whether an additional check against the original images helps when predicted masks are wrong. Published scores in the matrix provide context, not TerraWatch results or directly comparable targets.

<!-- PAGEBREAK -->

## 5. Research Gap and Proposed Contribution

### 5.1 What prior work already addresses

Change-Agent combines detection, captioning and tools; ChangeChat supports interactive change questions; CCExpert and KCFI use change-sensitive visual features [4–6,9]. FMScore and grounded change-unit evaluation already address factual evaluation [11,12]. TerraWatch does not claim that integration, mask conditioning or human evaluation is new.

### 5.2 Specific mechanism to investigate

We propose **disagreement-triggered visual reinspection**. The system separates a description into individual claims and links them to candidate image regions. A consistency check flags conflicting or uncertain claims. It then examines the corresponding areas in both original images, retaining the surrounding scene as context. A decision layer accepts, rejects or defers each claim; claims with insufficient evidence are qualified or withheld.

A mask alone cannot establish object identity, addition/removal direction or an exact building count. Agreement with the mask is therefore insufficient to approve a claim. A sample of unflagged claims will also be reviewed to identify errors missed by the trigger.

### 5.3 Mechanism-level comparison

| System | Use of visual evidence | Comparison with TerraWatch |
|---|---|---|
| Change-Agent [4] | Joint detection/caption features and agent tools | Test claim checking while keeping the detector output fixed |
| ChangeChat [5] | Instruction-tuned analysis of image pairs | Compare generated answers with explicit accept/reject/defer decisions |
| CCExpert [6] | Difference-aware features guide caption generation | Compare generation-time evidence with an additional check after generation |
| TerraWatch; proposed | Region-linked claims and targeted inspection of original images | Test whether this check catches errors that mask agreement misses |

This comparison concerns the checking mechanism; it does not assume that earlier systems lack grounding or verification.

### 5.4 Contribution and supporting evidence

**Research hypothesis:** checking the original images reduces incorrect claims compared with mask-only checking, without leaving out more relevant changes. This will be tested on the same images and candidate descriptions, including cases with incorrect detections. The mechanism must also be compared with a direct image-based judge to establish whether targeted reinspection adds value.

The contribution is a proposed reliability method and its controlled evaluation. Its novelty and benefit remain to be established; unsuccessful comparisons will be reported alongside improvements.

<!-- PAGEBREAK -->

## 6. Proposed Approach and Technical Grounding

The pipeline retains four model roles: change detection, description, visual inspection and claim decisions. Input checks and rule-based comparisons support these stages.

![TerraWatch pipeline: paired images, change detection, description, verification and reporting.](assets/pipeline.png)

*Figure 1. Proposed pipeline. Verification uses both the predicted change evidence and the original image pair.*

| Role | Candidate and training strategy | Output |
|---|---|---|
| Change perception | Compact early-fusion/Siamese U-Net trained as a baseline; ChangeFormer as a stronger comparison [1–3] | Change probabilities, class masks where supervised, candidate regions |
| Change description | Specialist baseline such as Chg2Cap/RSCaMa; Qwen3-VL as a candidate for multi-image prompting or adapter tuning [7,8,20] | Short caption and structured claim spans |
| Visual evidence inspection | Frozen VLM initially; inspect matched T1/T2 crops and context | Object, action, location observations and visibility/ambiguity flags |
| Decision policy | Rules baseline; Clef/Clef-flash or another bounded classifier; Jev as an optional comparator [21–23] | Claim acceptance, contradiction or deferral and logged decision scores |

**Input preparation**

- Check image dimensions, channels, quality and approximate alignment.
- Apply the selected model's normalisation and paired resizing or tiling; preserve region coordinates.
- Use brightness or alignment adjustments only when they improve validation results, and retain the original images.

**Training approach.** Train a standard detector, then compare pretrained caption and inspection models. Fine-tune selected components where baseline results justify it. Compare the decision layer with simple rules and a direct visual judge.

Reports will retain links to supporting regions and explain deferred claims. Exact counts and geographic measurements will be included only when the evidence and metadata support them.

<!-- PAGEBREAK -->

## 7. Data Availability, Alignment and Robustness Strategy

LEVIR-MCI is the primary candidate because it provides paired images, change masks and descriptions. It extends LEVIR-CC, which contains 10,077 image pairs with five captions per pair [4,13]. Additional datasets support detection, semantic analysis and testing under different imaging conditions.

| Data group | Candidate releases and supervision | Use / boundary |
|---|---|---|
| Aligned detection and text | LEVIR-MCI, LEVIR-CC [4,13] | Core training/evaluation; same underlying pairs counted once |
| Building change | LEVIR-CD, LEVIR-CD+, S2Looking [14,15] | Compatible detection supervision after overlap audit; hold out a domain for transfer |
| Broader visual change | SYSU-CD [16] | Separate transfer evaluation; reconcile broader label coverage before joint training |
| Semantic transitions | SECOND [17] | Optional task-specific head; not interchangeable with binary building labels |
| Change instructions | ChangeChat-105k [18] | Optional adapter data; trace LEVIR-derived images and instruction provenance |
| Lower-resolution urban change | OSCD [19] | Separate multispectral extension; not pooled with high-resolution RGB as equivalent building evidence |

**Combining datasets.** Combine sources only when their image formats and labels are compatible; otherwise use separate task-specific components. Compare single-dataset and combined-data training, then test on a compatible dataset excluded from training. Verify access, usage terms and label definitions before use.

Preserve official splits and check shared scenes across releases so the same imagery does not enter both training and evaluation; record known overlap with pretrained models' training data.

## 8. Feasibility and Key Risks

Begin with one dataset and a compact detector-to-description baseline before expanding the model set. Choose model sizes and training settings after measuring memory and runtime.

The main risks are shared detection/checking errors, ambiguous structures, inconsistent labels and preprocessing that removes real changes. Address these through original-image review, deferred decisions, data checks and comparisons with unmodified inputs. Tests using artificial image variations will be reported separately from tests on real imaging differences.

<!-- PAGEBREAK -->

## 9. Evaluation and Definition of Success

Compare models on the same held-out images, with settings selected on validation data. Report measured improvements and uncertainty rather than promising a fixed gain.

| Objective | Main measures | What would count as improvement |
|---|---|---|
| O1: detection | Change-class F1 and IoU | Better detection than the selected baseline on the same labelled images |
| O2: robustness | Performance loss under image variations | Smaller losses while maintaining performance on clean images |
| O3: reliable descriptions | Claim correctness and coverage of visible changes | Fewer incorrect statements without leaving out more relevant changes |
| O4: practicality | Runtime, memory and reproducibility | A working demonstration with documented resource use and repeatable outputs |

### 9.1 Checking description quality

Two reviewers will independently assess at least 100 test pairs without seeing the system identity. They will compare claims with the original images and labels, mark them supported, unsupported, contradicted or unverifiable, and resolve disagreements.

**Supported-claim precision** measures the proportion of emitted claims supported by the images. **Reference-fact recall** measures the relevant visible changes correctly described. Both are needed: an empty description is not evidence of reliability.

### 9.2 Comparing the components

The evaluation will compare five settings:

1. Caption generated from the two images alone.
2. Caption generated with predicted change evidence.
3. The same caption checked for consistency with the predicted mask.
4. The same caption checked directly against the original images.
5. The proposed targeted reinspection and accept/reject/defer policy.

For settings 3–5, keep detections and candidate captions fixed. Include incorrect masks and unchanged scenes; compare methods at similar computational budgets. Test lighting, blur and alignment variations separately.

BLEU-4, METEOR, ROUGE-L and CIDEr-D provide supplementary caption comparisons. Any custom pipeline score must be defined and validated before final testing, alongside the component measures.

**The checker succeeds if it reduces incorrect claims without increasing missed changes.** Otherwise, report the comparison as inconclusive or negative.

## 10. Milestone Plan

- **M2 - October 29:** verify data sources, labels and training/evaluation splits.
- **M3 - November 5:** justify the architecture and demonstrate the complete pipeline on sample pairs.
- **M4 - November 19:** train the selected detector and compare training or fine-tuning settings.
- **M5 - November 26:** evaluate the models, compare checking methods and analyse errors.
- **M6 - December 10:** deliver the reproducible demonstration and final documentation [24].

Individual contributions will be recorded at each milestone. This report presents the proposed work; it does not claim completed training or measured improvements.

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
24. IITM DS & AI Lab. *Project Guidelines and Milestones*. https://bsc-iitm.github.io/dsai-lab-project/guidelines.html ; https://bsc-iitm.github.io/dsai-lab-project/milestones.html
25. Shu, Y. et al. (2026). *TerraScope: Pixel-Grounded Visual Reasoning for Earth Observation*. CVPR, 16712–16722. https://openaccess.thecvf.com/content/CVPR2026/supplemental/Shu_TerraScope_Pixel-Grounded_Visual_CVPR_2026_supplemental.pdf

Online sources accessed 24–25 October 2026. Model variants, access conditions and dataset terms will be pinned before experimentation.
