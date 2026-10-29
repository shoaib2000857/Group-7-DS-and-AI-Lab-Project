# Milestone 2 report

- [Submission Markdown](Milestone2_Report.md)
- [Submission PDF](output/pdf/TerraWatch_Milestone2_Report.pdf)
- [Internal review notes](REVIEW_NOTES.md)
- [Build script](build_m2_report.py)

Prepared from Sujal's supplied M2 draft, the approved M1 style, the live course M2 requirements and the captioning source/output snapshot `dc8f3ce`. The team has verified the report; the approved PDF, Markdown and figures are staged in `reports/` through a submission PR. Editable sources remain here. The Markdown includes the requested initials **PA · SS · SRP · SSSM**; this review record is excluded from the submission PDF.

## Build

```powershell
python -m pip install -r POC/milestone2/requirements-report.txt
python POC/milestone2/build_m2_report.py
```

Run from the repository. The builder imports the approved M1 typography and numbering helpers, updates only the milestone line on a copy of the original cover, and builds the body directly from Markdown. The IITM emblem and official team details are preserved. The data-flow diagram is native PDF vector artwork; its SVG/PNG and the preparation-audit chart are editable/reproducible assets. Preview images and temporary source checks stay in ignored `qa/`.

Raw supplied documents are preserved in `sources/`; input hashes and editorial decisions are recorded separately from the report. Figures use the provided preparation-audit aggregates, not a newly acquired dataset. The worklog update was pushed directly to main in `f91a7ad`.
