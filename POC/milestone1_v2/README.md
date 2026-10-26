# Revised Milestone 1 report

Prepared 25 October 2026; wording and layout polished 26 October 2026. The report is staged for team review in `reports/` on a pull-request branch. The defence handbook remains internal and further work on it is on hold.

- [Milestone1_Report.md](Milestone1_Report.md): submission text; editable source of truth.
- [output/pdf/TerraWatch_Milestone1_Revised.pdf](output/pdf/TerraWatch_Milestone1_Revised.pdf): formatted submission copy.
- [Team_Defence_Guide.md](Team_Defence_Guide.md): 17-chapter internal handbook covering the pipeline, experiments, datasets, model assignments, worked metric examples, implementation controls and defence questions.
- [output/pdf/TerraWatch_M1_Team_Defence_Guide.pdf](output/pdf/TerraWatch_M1_Team_Defence_Guide.pdf): formatted internal guide.
- [build_report.py](build_report.py): builds both PDFs directly from Markdown; no LaTeX required.
- [pipeline_figure.py](pipeline_figure.py): editable vector diagram, exported as [SVG](assets/pipeline.svg) and [PNG](assets/pipeline.png) for the Markdown copy.
- [SOURCE_CHECKS.md](SOURCE_CHECKS.md): verification limits and new-source register.

## Rebuild

From this folder, with Python 3:

```powershell
python -m pip install -r requirements-report.txt
python build_report.py
```

To rebuild only the submission report while preserving the existing internal guide PDF:

```powershell
python build_report.py --report-only
```

To rebuild only the internal handbook, preserving the submission PDF:

```powershell
python build_report.py --guide-only
```

The handbook PDF has a bookmark for each chapter. Its current expansion is saved for team reference; further handbook revisions are on hold while the submission report is reviewed.

To build the report and stage its PDF, Markdown and matching figures in the main `reports/` folder:

```powershell
python build_report.py --publish-report
```

Use this command on the report review branch. It leaves the handbook PDF unchanged. The Markdown's final team sign-off table is an internal approval record and is excluded from the submission PDF. Each member signs their own entries; Shoaib uses **SSSM**. Committing and pushing a branch does not merge or finalise the report.

The report reuses the original searchable cover from `assets/original_cover.pdf`, preserving its IITM logo, title, team details and placement. The body uses Arial, dark headings, blue accents, light-grey grid tables, left-aligned text, understated running headers and numbered footers. Windows Arial is used when available, otherwise Helvetica. The pipeline is rendered as native vectors in the PDF and an image in Markdown. Generated PDF previews and layout checks stay in ignored `qa/`. The PDFs are committed deliverables. Do not submit the internal guide as part of the M1 report.

This revision clarifies the original system and defines a testable reliability mechanism. It does not claim completed model training, full dataset access, validated novelty, a validated custom metric or universal robustness. Candidate model versions and data terms must be pinned before experiments.
