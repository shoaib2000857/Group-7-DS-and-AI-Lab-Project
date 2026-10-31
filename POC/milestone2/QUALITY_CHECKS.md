# Delivery checks - 31 October 2026

## Data and calculations

- All 10,077 primary pairs and 40,308 image/label files decoded; dimensions, channel equality, mask values and colour-map equivalence passed. Every source record has five non-empty captions.
- Acquired archive SHA-256 and size match the independent publisher LFS metadata (`evidence/primary_release_api.json`).
- Confirmed 16 exact repeated pair groups, including five spanning source training/test; separately recorded inconsistent targets and 511 mask/caption coarse-flag conflicts.
- Deterministic union of exclusions: 533. Strict package: 9,544 pairs (6,444/1,242/1,858). Class totals, pixel denominators and split ratios reconcile.
- Full loader validation visited all 9,544 retained records, checking PNG hashes, RGB/target decoding, five references, class-index conversion, batch dimensions and [0,1] float32 scaling. Inference-only records return no masks or captions.
- Separate evidence check finds no repeated retained unordered pair and no exact image frame shared across retained partitions. This does not certify geographic independence or undetected perceptual similarity.
- Original 150-pair seeded selection reproduced exactly: 65 unchanged, 28 building only, 11 road only, 46 both; eight mask/caption conflicts.
- Small POC cohorts are disjoint: 1,313 fitting / 50 development / 100 screening / 100 evaluation. New hashes distinguish them from older model experiment populations.
- SECOND-CC: completed 100 original-validation pairs, 400 source files checked against ZIP CRC/size, compatible 256 x 256 dimensions, derived masks/hashes and 500 reference captions. Source metadata and 389 cross-split CRC candidate groups are distinguished from confirmed pixel duplicates.
- No new neural-model training, model leaderboard or independently human-rated factuality score is claimed. Human annotations are explicitly deferred by Shoaib outside this report's completion claims.

## Report and visual verification

- Seven A4 pages, including original-style cover and final document-control page; approximately 2,565 Markdown words.
- Preserves IITM emblem, team cover details, Arial typography, navy/blue hierarchy, grey tables and page numbering. No revision badge is added.
- Three captioned report figures: full-release class/pixel distributions, change-area/location distributions and a native vector preparation diagram. A separate diagnostic figure accompanies detailed EDA.
- Final PDF pages were rendered and visually inspected. No short spill pages, clipped paragraphs, split sign-off table or overlapping elements remain. Programmatic text-bound checks report no issues.
- All 13 substantive sections appear. The main report contains no pending/TODO/TBD language, future-scope section or internal role-swapping/work-planning prose.
- Names, initials (including SSSM), contribution record, revision log and approval-date/signature fields are present in both formats. Date/signature fields remain empty; this revision has not been falsely signed for other people.
- Figures and relative Markdown image links exist. The final PDF contains no replacement glyphs or question-mark encoding artifacts. Hashes and checks are in `qa/layout_checks.json`.

## Repository scope

- The new `origin/execution` branch was inspected without merging. Its aggregate results duplicate the previously reviewed captioning results; folder changes are not counted as additional experiments.
- Publication is limited to the POC working branch. No main files, GitHub reviews or PR settings were changed. PR #14 was already merged before this work.
- Earlier report sources are preserved. Raw data, partial downloads and working imagery stay Git-ignored. Final data/scripts/evidence/report documentation remain separate.
