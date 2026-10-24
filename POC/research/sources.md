# Sources and verification register

Access/research date: **24 October 2026**. This is a targeted research pass, not a systematic or exhaustive review. Primary research and author repositories were prioritized. Publication year and arXiv revision year can differ.

## Local context

All supplied files are listed in `00_context_and_requirements.md`. Searchable PDF and saved-HTML extractions are in `source_extracts/`; originals remain in `../DOCS/`. Original PDFs were text-extracted and a first-page inventory contact sheet inspected; this is not a layout audit of every page. The timestamped transcript was reviewed for relevant discussions. Written TA feedback is more authoritative than its noisy summary.

Submitted repository materials inspected include root README, problem statement, M1 report, requirements, representative config/test/app files, and the module inventory. The root code was not executed or changed.

## Official course pages

| Source | Verification |
|---|---|
| [Guidelines](https://bsc-iitm.github.io/dsai-lab-project/guidelines.html) | Direct HTTP retrieval succeeded after browsing-tool access failure; headline schedule agrees with supplied snapshot |
| [Milestones](https://bsc-iitm.github.io/dsai-lab-project/milestones.html) | Direct HTTP retrieval succeeded; detailed M1–M5 content read |

Recheck before submissions. Relative dates in copied chat messages do not establish an exact 48-hour deadline.

## Research bibliography and reading depth

| ID | Source | Reading depth / use |
|---|---|---|
| R01 | [Daudt et al. — Fully Convolutional Siamese Networks for Change Detection, 2018](https://arxiv.org/abs/1810.08462) | Abstract; compact detector precedent |
| R02 | [Chen et al. — Remote Sensing Image Change Detection with Transformers, 2021](https://arxiv.org/abs/2103.00208) | Abstract; BIT context mechanism |
| R03 | [Bandara & Patel — A Transformer-Based Siamese Network for Change Detection, 2022](https://arxiv.org/abs/2201.01293) | Abstract; ChangeFormer comparison candidate |
| R04 | [Corley et al. — A Change Detection Reality Check, 2024](https://arxiv.org/abs/2402.06994) | Abstract; rationale for strong simple baselines |
| R05 | [Zheng et al. — Segment Any Change, revised 2025](https://arxiv.org/abs/2402.01188) | Abstract; pretrained segmentation candidate, not locally run |
| R06 | [Chang & Ghamisi — Changes to Captions, 2023](https://arxiv.org/abs/2304.01091) | Abstract; specialist caption baseline |
| R07 | [Liu et al. — Change-Agent, 2024 preprint](https://arxiv.org/html/2403.19646v1) | Paper text, particularly dataset/tool/agent sections; title differs slightly from published version |
| R08 | [Change-Agent author implementation](https://github.com/Chen-Yang-Liu/Change-Agent) | README data organization, environment, train/evaluate/inference and checkpoint links; code page inspected, not fully audited or executed |
| R09 | [Deng et al. — ChangeChat, 2024](https://arxiv.org/html/2409.08582v1) | Paper text; instruction data and task design |
| R10 | [ChangeChat author implementation](https://github.com/hanlinwu/ChangeChat) | README task descriptions and 87k dataset details |
| R11 | [Wang et al. — CCExpert, 2024](https://arxiv.org/html/2411.11360v1) | Paper text, difference-aware integration and data/training sections; no reproduction |
| R12 | [Yang et al. — Enhancing Perception of Key Changes, 2024](https://arxiv.org/abs/2409.12612) | Abstract; KCFI detection-guided features |
| R13 | [ChangeVG / ChangeIMTI, revised December 2025](https://arxiv.org/html/2509.23105v2) | Selected full-text sections: dataset origin, task design and visual guidance |
| R14 | [Ricci et al. — Remote sensing change captioning meets large language and vision models, 2026](https://www.sciencedirect.com/science/article/pii/S0924271626003035) | Publisher abstract and exposed method/data/result sections; full FMScore implementation not yet audited |
| R15 | [Liu et al. — Seeing Clearly without Training, 2026](https://arxiv.org/abs/2603.02754) | Abstract; RADAR/RSHBench relevance |
| R16 | [RADAR author repository](https://github.com/MiliLab/RADAR) | Repository summary; no local run or attention-code audit |
| R17 | [Liberatori et al. — Spot-the-shift, September 2026](https://arxiv.org/html/2609.10356v1) | Selected full-text sections on human verification, change units, matching and no-change evaluation; street-view domain |
| R18 | [Chen et al. — RSCC, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/62867024377cac4233195949b9db0ebd-Abstract-Datasets_and_Benchmarks_Track.html) | Conference abstract; later disaster-data candidate |
| R19 | [Lee et al. — Toward Robust Hyper-Detailed Image Captioning, ICML 2025](https://proceedings.mlr.press/v267/lee25aj.html) | Conference abstract; factuality and coverage precedent |

## Dataset / imagery sources

| ID | Source | Verified here / still unresolved |
|---|---|---|
| D01 | [LEVIR-MCI author hosting](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI) | Page reachable, card/preview limitations observed; archive and label values untested |
| D02 | [LEVIR-CC author source](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset) | Pair/caption counts and MCI relationship; original payload and complete rights audit pending |
| D03 | [LEVIR-CD author source](https://github.com/justchenhao/LEVIR) | Dataset dimensions, nominal resolution, binary labels and supplementary metadata notice |
| D04 | [S2Looking / LEVIR-CD+ source](https://github.com/S2Looking/Dataset) | Download links identified; downloads and access/rights not tested |
| D05 | [SYSU-CD author source](https://github.com/liumency/SYSU-CD) | Size, resolution, region and change types; payload untested |
| D06 | [SECOND author page](https://captain-whu.github.io/SCD/) | Semantic task and dataset link; available split counts/labels need archive verification |
| D07 | [ChangeChat-105k release](https://huggingface.co/datasets/hlwu/changechat-105k) | Release page reachable; exact instruction inventory/version and training provenance not audited |
| D08 | [NASA HLS products](https://hls.gsfc.nasa.gov/data-products/) | Official product spatial scale; no imagery downloaded |

## Search coverage and limits

Searches included remote-sensing change captioning, factual verification/hallucination benchmarks, 2025–2026 interactive change understanding, FMScore, RADAR and grounded image difference evaluation. Search-result aggregators were used only to discover primary sources; the comparison cites primary pages.

No claim of “first,” “none,” “all existing methods,” or current state of the art follows from this pass. Full code audits, systematic citation tracing, dataset rights resolution, baseline reproduction, and measured hardware feasibility remain next steps. Numerical performance values from the old report have deliberately not been promoted into new results.
