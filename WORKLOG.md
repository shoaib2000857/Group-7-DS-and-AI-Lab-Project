# Worklog: Group 7 (TerraWatch)

Each member keeps their own section and adds one row per week, with links to the work (commits, PRs, documents). Update it before every milestone submission.

Other members: add your section below using the same table.

---

## Sujal Raj Pradhan ([@SujalPradhan](https://github.com/SujalPradhan))

| Week | Dates | Milestone | Work done | Links | Hours |
|---|---|---|---|---|---|
| 1 | 15–22 Oct 2026 | M1 | Narrowed the use cases with short data stories; picked urban growth / unauthorised construction as the lead use case and grounded change descriptions as the contribution | TerraWatch Data Stories (Claude artifact) | |
| 1 | 15–22 Oct 2026 | M1 | Researched add-on ideas (photo → place history, change timing, robustness checks, decision models such as Clef/Jev, height/3D) and recorded feasibility; kept them out of M1 scope | TerraWatch Data Stories, "Going deeper" section | |
| 1 | 15–22 Oct 2026 | M1 / M2 prep | Shortlisted datasets; checked every download link, file sizes and licences (LEVIR-MCI, LEVIR-CC, ChangeChat-105k, LEVIR-CD, LEVIR-CD+, S2Looking, SYSU-CD, SECOND) | [Milestone 1 report §6, §8](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M2 prep | Compared storage options (Google Drive, Hugging Face, Kaggle, buckets); team chose a shared Google Drive folder | | |
| 1 | 15–22 Oct 2026 | M1 | Wrote the Milestone 1 report (problem, scope, stakeholders, objectives, literature review, benchmarks, gaps, contribution, feasibility, evaluation plan) | [reports/DSAI Lab_Group 7_Milestone 1.pdf](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Built the Milestone 1 review slides | [presentation/DSAI Lab_Group 7_Milestone 1_Slides.pdf](presentation/DSAI%20Lab_Group%207_Milestone%201_Slides.pdf) | |
| 1 | 15–22 Oct 2026 | Project setup | Created GitHub milestones M1–M6 with an epic issue for each, so tasks can be tracked as individual issues | [Milestones](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/milestones) | |

---

## Priyanshu Agarwal ([@23f2003488](https://github.com/23f2003488))

| Week | Dates | Milestone | Work done | Links | Hours |
|---|---|---|---|---|---|
| 1 | 15–22 Oct 2026 | M1 | Formulated dual-head grounded architecture: conditioned change captioning directly on predicted change mask events | [Report §8.1](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Designed the 3-stage factual consistency check (claim extraction → mask evidence check → hallucination mitigation) to prevent phantom object descriptions | [Report §8.1](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Formulated the loss and sampling strategy for severe class imbalance (composite Focal + Dice loss and 70:30 change-centric patch sampling) | [Report §8.1, §10](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Defined evaluation framework adding automated factual metrics (Count Absolute Error, Claim Grounding Precision) alongside human factual audit | [Report §9.2](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Audited Milestone 1 deliverables against IIT Madras BSDA4001 rubric; fixed mathematical expressions and restored Markdown report source | [PR #10 Review](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/10) | |

---

## Sarthak Saini ([@Sarthak-Saini265](https://github.com/Sarthak-Saini265))

| Week | Dates | Milestone | Work done | Links | Hours |
|---|---|---|---|---|---|
| 1 | 15–22 Oct 2026 | M1 | Reviewed dataset requirements and evaluation methodology for satellite image change detection, identifying image-pair alignment, change-mask quality, and class imbalance as key factors affecting model performance | [Report §8, §9](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Analysed the proposed evaluation metrics for change detection, focusing on precision, recall, F1-score, and IoU to assess the quality of predicted change masks | [Report §9](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 / M2 prep | Helped Shortlist datasets; checked every download link, file sizes and licences and created a Drive Space to gather all available data | [Milestone 1 report §6, §8](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 | Examined the role of ground-truth change masks in supervised model training and evaluation, focusing on how annotation quality can affect segmentation performance | [Report §8, §9](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |

---

## Shoaib SSM ([@shoaib2000857](https://github.com/shoaib2000857))

| Week | Dates | Milestone | Work done | Links | Hours |
|---|---|---|---|---|---|
| 0 | 10–14 Oct 2026 | Problem selection / M1 prep | Explored and screened potential DS/AI Lab problem statements across disaster response, environmental monitoring, satellite vision, educational AI and other deep-learning applications; discussed feasibility, novelty, public data availability and research scope with the team, contributing to the selection of TerraWatch. | [Approved problem statement](docs/Problem_Statement.md) | |
| 0 | 10–14 Oct 2026 | Problem definition | Helped frame and refine TerraWatch's initial broad problem statement around multi-temporal satellite image change detection, distinguishing real changes from seasonal/illumination artefacts, with optional multimodal interpretation; kept the proposal flexible until dataset and use-case feasibility could be established. | [Problem statement](docs/Problem_Statement.md) | |
| 1 | 15–22 Oct 2026 | M1 literature / research | Conducted initial research into relevant public datasets and task supervision: binary change masks, semantic land-cover transitions, paired captions and instruction-based change Q&A; compared potential uses and compatibility of LEVIR-CD/+, LEVIR-MCI/CC, S2Looking, SYSU-CD, SECOND and ChangeChat, including overlap and cross-dataset generalisation concerns. | [Dataset discussion in M1 report §8](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | M1 model feasibility | Reviewed possible model families for the multi-stage system: classical change detection; Siamese CNN/U-Net segmentation; temporal/bi-temporal attention and transformer architectures (e.g., BIT, ChangeFormer); semantic change models; change captioning and vision-language approaches. Proposed a modular detection → structured change events → grounded description workflow for further evaluation. | [Literature review and proposed approach, report §§6–8](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) | |
| 1 | 15–22 Oct 2026 | Research planning / M1 review | Prepared a comprehensive research and engineering planning blueprint covering candidate datasets, architecture alternatives, baseline comparisons, evaluation metrics, robustness experiments, feasible training workflows and upcoming milestone deliverables. Reviewed the group's scoped M1 material for consistency with the approved proposal and course requirements. | [M1 report](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) · [Course guidelines](https://bsc-iitm.github.io/dsai-lab-project/guidelines.html) | |
| 1 | 15–22 Oct 2026 | Project infrastructure | Created and configured the main public Group 7 GitHub project repository and its initial modular layout: documentation/reports, data staging and instructions, notebooks, reusable model/data/inference/evaluation code, configuration files, application, model checkpoints, outputs, tests, contribution guidelines, licence and project README. | [Initial repository structure commit](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/cace5f6cbf66afe40c0f5cc683e1129f6ff474cc) · [Repository](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project) | |
| 1 | 15–22 Oct 2026 | Repository documentation | Updated the README, problem statement and documentation index to describe TerraWatch's motivation, features, preliminary architecture, repository layout, planned model/dataset areas and milestone deliverables; helped establish a consistent location for future project artifacts. | [Documentation commit](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/63f8b04c7b40fc5931623d59f1e9de5fff24aa45) | |
| 1 | 15–22 Oct 2026 | M1 presentation | Reviewed the research report and the initial M1 slide deck, then designed a substantially revised visual presentation with an improved research narrative, satellite change visuals, model-pipeline infographic, dataset strategy, published-baseline charts, experimental comparisons and milestone roadmap; prepared editable PPTX and PDF deliverables for team review. | [Milestone 1 report](reports/DSAI%20Lab_Group%207_Milestone%201.pdf) · Redesigned PPTX/PDF (pending repository upload) | |

| 2 | 23-26 Oct 2026 | M1 review / POC preparation | Prepared the redesigned presentation; organised TA/course context and POC research; revised the M1 report with an analytical literature matrix, testable claim checking, clearer success criteria and a polished pipeline/layout; prepared four-person benchmark work packages and source/environment checks; expanded the internal defence handbook. Staged the report PDF/Markdown for team review with section sign-off (Shoaib: SSSM), requested reviews and consolidated the changelog. Model experiments remain unrun in these preparation artifacts. | [PR #12](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/12) · [Changelog](CHANGELOG.md) · [Report](reports/Milestone1_Report.md) · [POC research](POC/research/README.md) · [Benchmark plan](POC/benchmarks/README.md) · [Internal handbook](POC/milestone1_v2/Team_Defence_Guide.md) | |
| 2 | 27 Oct 2026 | POC decision benchmarking | Implemented Shoaib's (SSSM) self-contained Colab notebook, primary-data subset loader, frozen witness stage, decision adapters and checking controls, evidence/label separation, evaluation/plots, resume/checkpoint handling, independent review export and combined results ZIP. Verified CPU/data/export execution and protocol tests; actual GPU model inference remains for the Colab run. | [PR #13](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/13) · [Notebook](POC/benchmarks/colab/TerraWatch_Shoaib_Decision_Benchmarks.ipynb) · [Instructions](POC/benchmarks/colab/README.md) · [Validation](POC/benchmarks/colab/VALIDATION.md) | |
| 2 | 27 Oct 2026 | Colab smoke/screen evidence and repair | Ran two smoke sessions and one screen session on a Colab L4; archived and checksum-verified all three ZIPs. Reviewed the shared environment-creation failure and diagnostic controls, confirming zero learned outputs and forty distinct real pairs. Replaced the failing setup path with a private seeded bootstrap, added setup logs/early failure checks and explicit run-health reporting, and verified the bootstrap on a native Linux filesystem. | [Run inventory and findings](POC/benchmarks/results/2026-10-27/shoaib/README.md) · [Repaired notebook](POC/benchmarks/colab/TerraWatch_Shoaib_Decision_Benchmarks.ipynb) · [Validation](POC/benchmarks/colab/VALIDATION.md) | |
