# Revised Milestone 1 report

Prepared 25 October 2026. The original submission outside `POC/` is preserved.

- [Milestone1_Report.md](Milestone1_Report.md): submission text; editable source of truth.
- [output/pdf/TerraWatch_Milestone1_Revised.pdf](output/pdf/TerraWatch_Milestone1_Revised.pdf): formatted submission copy.
- [Team_Defence_Guide.md](Team_Defence_Guide.md): internal explanation of novelty, metrics, tradeoffs and likely questions.
- [output/pdf/TerraWatch_M1_Team_Defence_Guide.pdf](output/pdf/TerraWatch_M1_Team_Defence_Guide.pdf): formatted internal guide.
- [build_report.py](build_report.py): builds both PDFs directly from Markdown; no LaTeX required.
- [SOURCE_CHECKS.md](SOURCE_CHECKS.md): verification limits and new-source register.

## Rebuild

From this folder, with Python 3:

```powershell
python -m pip install -r requirements-report.txt
python build_report.py
```

The builder uses Calibri when available on Windows, otherwise ReportLab's standard Helvetica family. Generated PDF previews and layout checks stay in ignored `qa/`. The PDFs are committed deliverables. Do not submit the internal guide as part of the M1 report.

This revision clarifies the original system and defines a testable reliability mechanism. It does not claim completed model training, full dataset access, validated novelty, a validated custom metric or universal robustness. Candidate model versions and data terms must be pinned before experiments.
