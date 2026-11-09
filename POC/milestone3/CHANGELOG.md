# Milestone 3 document revision log

| Date | Revision | Evidence |
|---|---|---|
| 5 November 2026 | Fetched all remote heads; integrated current data fixes, captioning, visual evidence and decision benchmarking without discarding original variants | Consolidation branch inventory and archives |
| 5-6 November 2026 | Built the canonical experiment register; verified original exports and local-file preservation; executed the CPU checks | Six ZIP CRC/hash checks, preservation inventory and 81 passing tests |
| 6 November 2026 | Prepared the model-selection/system-architecture report, native diagrams, training configuration, model workers and inference coordinator | Report sections 1-8, pipeline scripts and four contract tests |
| 6 November 2026 | Rendered and inspected the eight-page PDF; checked body bounds, figure labels and corresponding Markdown | `qa/layout_checks.json` |
| 9 November 2026 | Added Sarthak's bounded local BIT-R18 pilot and Otsu comparison, reran the visual witness and 83-test suite, and clarified local versus historical evidence in the report | `sarthak/Benchmark_Report.md`, local-only run records and updated report PDF |
| 9 November 2026 | Rebuilt and visually inspected the eight-page PDF; refreshed local QA hashes and test count | `qa/layout_checks.json`, rendered pages under ignored `qa/pages/` |
| 9 November 2026 | Ran the 20%-budget RSCaMa training schedule locally, evaluated its best checkpoint on the frozen 100-pair screen, and documented its precision divergence from the historical aggregate; added hash-backed local evaluation proof and figures | `sarthak/Benchmark_Report.md`, `sarthak/evaluation_proof.json`, `sarthak/figures/` |

This file records report revisions separately from the technical narrative. Model selections, measured component results and implementation tests retain their source identities.


### 10 November 2026 - Local benchmark report integration

- Preserved Sarthak's supplied local report and previous report editions under POC, with source hash receipt and numeric transcription.
- Integrated six caption reruns, supplementary caption results, transfer screens and the 100/25/100 two-epoch BIT experiment. Checked baseline numbers and arithmetic; separated class-mean and binary IoU definitions.
- Replaced absent-source figure links with a reproducible point-estimate figure; retained the original cover, formal nine-page layout and component-specific metric scopes.
- Added PA, SS, SRP and SSSM sign-offs dated 10 November 2026 under Shoaib's explicit confirmation of team approval. Matched the submitted PDF/Markdown and kept experimental materials under POC.

- 10 November 2026: Pulled benchmark-evidence commit f6bc2ec; checked all six caption configurations and supplementary metrics against the supplied proof. Repaired committed conflict markers in the POC report/README and matched the POC PDF to the checked submission. Submission PDF/Markdown and team sign-offs remain unchanged. All new experiment files are retained.
