# Changelog

All notable changes to **TerraWatch (Group 7 — DS & AI Lab Project)** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **27 October 2026 - Shoaib (SSSM) benchmark implementation:** self-contained 30-cell Colab notebook under `POC/benchmarks/colab/`, readable embedded runtime, notebook generator and protocol tests. Includes bounded selection/download of LEVIR-MCI train/validation pairs, frozen image-derived witness evidence, controls, separate common-evidence/image-backed judge tracks, default J05/J01/J06/J07 adapters, optional local Kev/Clef routes and an eight-candidate status register with hosted Jev deferred.
- Notebook results include raw and thresholded decisions, completion/failure coverage, label-agreement and acceptance/rejection measures, valid probability diagnostics, grouped comparisons, option-order checks, cached reinspection diagnostics, inline figures, independent review sheets and a combined checksummed results ZIP. Checkpoints support optional Drive backup and content/revision-keyed resume. Synthetic logic, annotation proxies and unreviewed natural captions remain separate; CPU/data/export validation does not establish GPU model accuracy.
- **26 October 2026 - [team report review (PR #12)](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/12):** section-by-section sign-off tables in the report Markdown and pull request for Priyanshu, Sarthak, Sujal and Shoaib (initials **SSSM**). All entries start pending; finalisation requires each member's own review. The report changes are proposed through a review branch, not merged automatically.
- **26 October 2026 - POC benchmark plan:** four work packages in `POC/benchmarks/`: nine change detectors (Sarthak), eight captioning candidates (Sujal), six visual-evidence candidates and six input comparisons (Priyanshu), and eight decision/judge candidates plus integration (Shoaib). The registry distinguishes 31 role entries from 25 distinct model identities.
- Shared benchmark protocol, first-run priorities, two-track dataset policy, split/overlap controls, smoke tests, compute budgets, error analysis, calibration and human review guidance; JSON templates for run records and claim decisions. Candidate metadata audit and environment-probe scripts, recorded source metadata, access restrictions and checkpoint caveats are included. These are preparation artifacts, not measured benchmarks ([benchmark commit](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/2763dd9)).
- **25-26 October 2026 - report sources and internal handbook:** `POC/milestone1_v2/` contains the report Markdown, reproducible PDF builder and dependencies, source checks, quality checks, original cover/logo assets, and a separate team guide. The guide now has 17 pages and navigation bookmarks covering evidence contracts, prior art, dataset compatibility, leakage, preprocessing, model assignments, training/fine-tuning limits, worked metric examples, evaluation controls, annotation, custom-metric development, compute, troubleshooting and defence questions. Further handbook changes are on hold while the submission report takes priority ([initial report](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/57b6232), [handbook expansion](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/0f7d58d)).
- **25 October 2026 - POC research workspace:** archived official course material, TA meeting transcript and summary, earlier reports, research blueprint and dataset notes in `POC/DOCS/`. Added context synthesis, analytical literature matrix, data/subset protocol, proposed system, experiments, execution roadmap, M1 revision notes and source register in `POC/research/`, with extracted source text and manifest/annotation/experiment/access templates ([research commit](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/b50a9ee)).
- **22 October 2026 - contribution records:** individual M1 worklogs for all four members, including research, architecture, data/evaluation planning, project setup, report preparation and presentation work. Markdown report source and supporting figures restored alongside the submitted PDF ([report-source commit](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/7846d95), [Shoaib worklog](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/0618b9f)).

### Changed
- **26 October 2026 - proposed submission replacement:** stage the reviewed POC report as `reports/DSAI Lab_Group 7_Milestone 1.pdf` and `reports/Milestone1_Report.md`, with matching pipeline assets. Remove revision labels from the report heading and PDF metadata. Keep the review-only signature table in Markdown and outside the submission PDF. The builder can publish matching report artifacts without rebuilding the paused handbook.
- **25-26 October 2026 - M1 technical revision:** respond to TA feedback with a study-comparison matrix, mechanism-level comparison of closest prior work, a more precise claim-checking contribution and controlled comparisons against mask-only checking. Separate proposed experiments from results; clarify dataset supervision, original-image verification, human factual audit, robustness tests and the exploratory status of any custom score. The core detection-to-captioning approach is retained.
- **25-26 October 2026 - M1 presentation and wording:** reuse the original searchable IITM cover and theme; correct the problem-statement container, align tables and lists, add an editable vector pipeline and simplify scope, objectives, risks, success criteria and milestones. Remove the submission's internal POC-limit and model-rationale paragraphs. The main report remains eight pages ([cover/theme](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/0ed550d), [wording/layout](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/9038a60)).
- **23-24 October 2026 - M1 presentation:** replace the original slide deck with the redesigned 18-slide deck and add `presentation/TerraWatch_Milestone_1_Redesigned_Pitch_Deck.pdf` ([deck replacement](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/fc76633), [additional presentation copy](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/commit/4c64706)).
- **21-22 October 2026 - report maintenance:** consolidate submitted artifacts under `reports/` and `presentation/`, remove stale build files/references and redundant directory READMEs, restore the editable Markdown and repair mathematical rendering before merging the M1 report review ([PR #10](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/10)).

### Validation status
- Document builds, source metadata checks, registry/protocol checks and rendered-page inspections have been performed. Full model training, pipeline benchmark runs, measured reliability improvements, universal robustness and validation of a custom score have not been established by these documentation changes. No paid model inference was performed.

## [0.2.0] - 2026-10-21
### Added
- Milestone 1 report `reports/DSAI Lab_Group 7_Milestone 1.pdf`: problem definition, scope, stakeholders, measurable objectives, literature review, benchmarks and baselines, gaps, proposed contribution, data feasibility and evaluation plan.
- Milestone 1 review slides `presentation/DSAI Lab_Group 7_Milestone 1_Slides.pdf`.
- `WORKLOG.md` for weekly individual contributions.
- GitHub milestones M1–M6 with one epic issue each.

## [0.1.1] - 2026-10-18
### Added
- Added approved brief Problem Statement for **TerraWatch — Intelligent Satellite Image Change Detection and Analysis** in `README.md` and `docs/Problem_Statement.md`.
- Added Milestone 1 research & feasibility framework in `docs/Problem_Statement.md` covering real-world problem scope, stakeholders, academic/commercial SOTA review, value proposition, and dataset feasibility.

## [0.1.0] - 2026-10-18
### Added
- Initialized repository structure following IITM DS & AI Lab guidelines.
- Added foundational documentation (`README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`).
- Created directory hierarchy for `docs/`, `data/`, `notebooks/`, `src/`, `app/`, `configs/`, `models/`, `outputs/`, and `tests/`.
- Added starter Jupyter notebooks for EDA, Preprocessing, Model Experiments, and Evaluation.
- Added YAML configuration templates (`model.yaml`, `train.yaml`, `inference.yaml`) and `src/config.py`.
- Added unit test placeholders in `tests/`.
