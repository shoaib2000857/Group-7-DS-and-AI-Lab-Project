# Literature comparison and research positioning

Read alongside [source reading depths](sources.md). Findings are qualitative summaries, not reproduced measurements. Rows based on abstracts are explicitly marked; do not present them as exhaustive full-paper reviews.

## Study-comparison matrix

| Study / source | Research objective | Data / setting | Experimental design or mechanism | Main finding relevant here | Limitation / implication for TerraWatch |
|---|---|---|---|---|---|
| [Daudt et al., 2018](https://arxiv.org/abs/1810.08462), abstract | Efficient dense change detection | Urban change benchmarks | Early fusion and fully convolutional Siamese comparisons | Establishes compact paired-image baselines | Produces change masks; supplies no factual-language verifier |
| [BIT, 2021](https://arxiv.org/abs/2103.00208), abstract | Capture spatial-temporal context | Remote-sensing CD benchmarks | Semantic tokens and transformer context modeling | Context is a useful detection mechanism | Detection improvement does not establish correct counts or narratives |
| [ChangeFormer, 2022](https://arxiv.org/abs/2201.01293), abstract | Hierarchical transformer change detection | LEVIR-CD and related benchmarks | Siamese transformer encoder with multiscale decoder | Credible stronger detector comparison | Not the first POC dependency; checkpoint/split compatibility must be audited |
| [A Change Detection Reality Check, 2024](https://arxiv.org/abs/2402.06994), abstract | Reassess architecture gains | Standard change-detection benchmarks | Compare a simple U-Net with proposed CD architectures | A straightforward U-Net remains competitive in the study | Do not generalize this into a claim that all detection is solved or saturated |
| [Segment Any Change, 2024/2025 revision](https://arxiv.org/abs/2402.01188), abstract | Change segmentation without task-specific training | Foundation segmentation / cross-domain setting | SAM-derived change analysis | Relevant pretrained alternative | Mask proposals are not semantic, directional, or legal evidence; local feasibility unknown |
| [Chg2Cap, 2023](https://arxiv.org/abs/2304.01091), abstract | Generate remote-sensing change captions | LEVIR-CC | Attentive change captioning | Useful specialist caption baseline | Reproduce preprocessing/tokenization before comparing caption scores |
| [Change-Agent / MCINet, 2024](https://arxiv.org/html/2403.19646v1), paper + repository | Comprehensive interactive change interpretation | LEVIR-MCI | Joint pixel/text perception; LLM calls change-analysis tools | Already integrates masks, captions, object statistics and interactive analysis | Tool use and replaceability already exist. Its mask-derived statistics can share perception errors; test rather than assume factual-verification absence |
| [ChangeChat, 2024](https://arxiv.org/html/2409.08582v1), paper + repository | Interactive paired-image analysis | LEVIR-derived ChangeChat-87k | Rule/GPT-assisted instructions, captioning/counting/localization, multimodal tuning | Task breadth goes beyond a one-line caption | Generated supervision is not independent factual truth; distinguish paper data from later 105k release |
| [CCExpert, 2024](https://arxiv.org/html/2411.11360v1), paper | Improve MLLM change perception | LEVIR-CC; CC-Foundation | Difference-aware multiscale integration and progressive training | Explicit difference features already guide language generation | Expensive training is unsuitable for the initial POC. Grounding is prior art; verification is a separate question |
| [KCFI, 2024 preprint](https://arxiv.org/abs/2409.12612), abstract | Focus language on key changes | LEVIR-CC | Pixel-level detection constrains key features; instruction-tuned language decoder | Detection-guided caption representations already exist | Refutes a broad claim that captioners never use detection evidence |
| [ChangeVG / ChangeIMTI, 2025](https://arxiv.org/html/2509.23105v2), selected full-text sections | Multi-task interactive change understanding | LEVIR-derived instruction data | Fine-grained and global visual guidance with multi-task evaluation | Counting/localization and visual guidance are established comparators | Not an independent new geography when derived from LEVIR; training contamination must be traced |
| [Ricci, Bazi & Melgani, 2026](https://www.sciencedirect.com/science/article/pii/S0924271626003035), publisher abstract/sections | Zero-shot change description and factual evaluation | SECOND and LEVIR-CC | Three pretrained LLM/VLM approaches; fact-matching FMScore | Fact-based evaluation already exists in remote-sensing change captioning | Read full FMScore implementation before claiming a distinct metric or mechanism; LLM judgments need validation |
| [RADAR / RSHBench, 2026](https://arxiv.org/abs/2603.02754), abstract + repository summary | Reduce remote-sensing VQA hallucinations | RS-VQA | Attention-guided localization and local reasoning without training | Grounding-focused inference and factual/logical hallucination tests are relevant | Single-image VQA differs from two-date change verification; adaptation is not automatically novel |
| [Spot-the-shift, September 2026](https://arxiv.org/html/2609.10356v1), selected full-text sections | Evaluate grounded descriptions of structural change | Real-world driving scenes | Human-verified change units, spatial annotations, unit matching and no-change tests | Atomic change evaluation and hallucination-aware comparison are already prior art | Different viewpoint/domain, but close conceptual overlap; cannot claim first claim-level change evaluation |
| [RSCC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/62867024377cac4233195949b9db0ebd-Abstract-Datasets_and_Benchmarks_Track.html), abstract | Disaster-oriented change caption data | 62,351 pre/post-disaster image pairs | Large paired image/caption benchmark | A meaningful later route into disaster interpretation | Different application/labels; not a reason to expand the first urban POC |
| [Lee et al., ICML 2025](https://proceedings.mlr.press/v267/lee25aj.html), abstract | Detailed caption factuality and coverage | General image captioning | Caption correction plus dual factuality/coverage evaluation | Factual precision must be considered with coverage | Adjacent-domain prior art; a terse or empty caption can otherwise look artificially safe |

## Mechanism-level comparison of the closest work

| Mechanism | Existing precedent | Consequence |
|---|---|---|
| Detector plus language plus tools | Change-Agent | Integration and modularity alone do not establish novelty |
| Difference features passed to language | CCExpert, KCFI, ChangeVG | Visual grounding alone is not new |
| Counts, localization, interactive questions | Change-Agent, ChangeChat, ChangeVG | These features are baseline capabilities, not a research gap |
| Fact matching against reference facts | Ricci et al. FMScore | Avoid claiming factuality evaluation is absent |
| Atomic change units and no-change hallucination testing | Spot-the-shift | Claim decomposition and grounded evaluation are not first-of-kind |
| Local evidence inspection | RADAR | A crop-backed checker needs comparison with established local reasoning |

The remaining **candidate question**, not a proven gap, is whether an inexpensive, explicitly uncertainty-aware checker can distinguish caption errors from detector errors on high-resolution change pairs, and retain useful content under corrupted evidence. That is an empirical hypothesis. This search cannot establish that no other publication addresses it.

## Contribution ladder

1. **Engineering outcome:** a reproducible local pipeline with inspectable region links and careful scope.
2. **Empirical outcome:** a controlled comparison showing where mask-only checks fail and whether image-backed checks help, including negative results.
3. **Possible methodological outcome:** a specific new verification or abstention rule with ablations and evidence beyond close prior art. Claim this only after full-method comparison and experiments.

Recommended provisional wording: “We study the reliability of change descriptions when predicted masks are imperfect, comparing mask consistency with image-backed claim checking and measuring factual errors together with retained information.”

## Follow-up reading before freezing the hypothesis

- Read Ricci et al.'s complete FMScore definition and evaluation protocol; determine whether it is reference-fact matching, image verification, or both in each setting.
- Inspect Change-Agent's counting/tool code and full agent prompts; document what is checked at inference rather than inferring from caption metrics.
- Inspect CCExpert and ChangeVG released weights/data provenance and inference requirements.
- Compare Spot-the-shift's unit matching, spatial evaluation, and distractor handling with the proposed annotation protocol.
- Trace references in these works for entailment, caption correction, visual verification, and selective prediction. Record overlapping mechanisms, not just paper titles.
- For every retained baseline extract exact dataset split, checkpoint training data, metric implementation, hardware, and limitations. No numerical leaderboard table is reproduced here because protocol-matched verification is unfinished.
