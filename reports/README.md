# Reports

Milestone reports as PDFs, each built from a Markdown source in this folder.

| Milestone | PDF | Source |
|---|---|---|
| M1: Problem Definition & Literature Review | [DSAI Lab_Group 7_Milestone 1.pdf](DSAI%20Lab_Group%207_Milestone%201.pdf) | [Milestone1_Report.md](Milestone1_Report.md) |

## Rebuilding a PDF

```bash
pip install markdown weasyprint
python reports/build_report.py reports/Milestone1_Report.md "reports/DSAI Lab_Group 7_Milestone 1.pdf"
```

`report.css` holds the page layout (A4, cover page, headers, "Page X / N" footers). Figures live in `figures/`.
