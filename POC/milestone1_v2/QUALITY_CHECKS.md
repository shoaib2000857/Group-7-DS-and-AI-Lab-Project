# Delivery checks

- Polished report, 26 October 2026: 8 A4 pages, including the original dedicated cover and references; approximately 2,740 words versus approximately 5,180 in the original Markdown report.
- Separate internal defence handbook: 17 A4 pages, 17 navigation bookmarks and approximately 5,730 Markdown words. All pages were rendered and visually inspected after the expansion.
- The revised report was rendered and inspected on every page, including full-size views of the problem statement, study matrix, pipeline and evaluation page. The handbook-only build left that report's SHA-256 unchanged: `ffafae082e34facef3428aa76ede46ec0b5bd0c200ccf0e2fbd35b5309fab56f`.
- The handbook includes worked segmentation and claim-evaluation arithmetic, evidence contracts, dataset/split controls, all four role assignments, candidate model pools, compute limits, annotation guidance, experiment comparisons, troubleshooting and defence questions. Hypotheses and unrun experiments remain explicitly distinguished from measured results.
- PDF text bounds checked; no text outside page margins found. Searchable text and clickable external reference links retained.
- Bibliography has 25 consecutive entries, all cited in the report. Markdown and PDF use the same submission text and diagram definition; no raw LaTeX or duplicated diagram text remains.
- Build ran with Python, ReportLab 5.1.0, Markdown 3.10.2 and pypdf 6.21.0. The requirements file records build dependencies.
- The original PDF cover is reused without reconstruction. Its rendered pixels were compared with the original and match exactly. Body typography, blue accents, grey grid tables, headers and page numbering follow the original theme.
- Replaced the overlapping paragraph background with a padded layout container. Text-block separation was checked around the problem statement. Objectives and milestones are properly indented lists, and all milestones fit on one page.
- Removed the submission's POC-limit paragraph, model-rationale paragraph and standalone published-score paragraph. The internal experiment data cap remains in the POC benchmark documents.
- Added the editable vector pipeline and a high-resolution PNG for ordinary Markdown readers. Figure captions, document margins and page footers were checked for alignment.
- No project training, model downloads or paid inference performed. Literature values are labeled as published results.
- All newly authored report files are under `POC/milestone1_v2/`; root submission artifacts are preserved. Local preview PNGs are ignored by Git.

The checks establish document consistency and layout, not model performance or experimental validity. The latter remain proposed M2–M5 work.
