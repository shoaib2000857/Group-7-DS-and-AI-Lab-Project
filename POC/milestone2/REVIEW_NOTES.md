# Internal M2 review record

This file is for the team and is not part of the submission PDF or Markdown.

## Scope and source decisions

- The report follows the live Milestone 2 requirements retrieved on 29 October: source/ownership/format/terms; size and characteristics; missing records, inconsistencies, noise and duplicates; train/validation/test strategy and overlap; augmentation/synthetic data; subtask alignment; model-specific inputs; reproducibility.
- The supplied draft is the source for the detailed 150-pair preparation-audit aggregates, duplicate counts and supplementary-data preparation. The committed captioning scripts and aggregate outputs were inspected. Private manifests, raw training logs and human annotations were not independently reconstructed for this edit. No further dataset building was needed or completed.
- The submitted M1 PDF and source were used as the visual/style reference. Its core project approach is retained.

## Corrections made

1. ZIP-index CRC/size matches are duplicate candidates, not proof of content identity. Preserve the list and confirm with file hashes before final deduplication/exclusions.
2. The parser source discloses development from screening outputs, including v2/v3 revisions. The report therefore describes the current screen as exploratory; it does not call it untouched validation.
3. Automatic mask/reference checks are not human factual accuracy. The report retains the constant-caption CIDEr example to explain the data limitation without promoting a proxy leaderboard into the main M2 narrative.
4. Dataset licensing is separated from a repository's code restrictions. Missing terms and WHU-CDC release identity remain open; no new permission is inferred.
5. SECOND's label projection is described for semantic change annotations, whose unchanged-area convention matters. It is not a generic union of building presence from two full-scene class maps.
6. Human annotation uses supported/contradicted/unsupported/unverifiable; accept/reject/defer are decision outputs. Mask components are not accepted as building-instance ground truth.
7. Captioning and decision pilot manifests currently differ. A canonical cross-role join and consolidated usage ledger remain an explicit M3 handover item.
8. Constructed decision fixtures already exist. The report therefore says no synthetic satellite imagery is used, rather than saying no synthetic data exists anywhere in the project.
9. Removed the ambiguous supplied line that appeared to sign for all four members while also saying confirmation was pending. No signatures have been entered on anyone's behalf.
10. Removed internal command tables, abbreviated hashes and commentary from the main report. The code snapshot remains a formal reproducibility reference.

## Confirm before finalisation

- Sujal: confirm the 150-pair mask aggregates, duplicate-candidate list, split ledger and sealed-test status against the retained preparation records.
- Confirm the final shared manifest and training deduplication policy before integrating detector/captioner/checker comparisons.
- Confirm WHU-CDC caption provenance and terms; keep its results provisional meanwhile.
- Keep the private cohort/training records, parser exposure history and exact metric denominators available for TA questions. A four-hour versus approximately three-hour training-duration inconsistency in the supplied material was omitted from the report rather than guessed.
- Human factuality and count claims remain pending two-rater annotation. No unseen-domain guarantee, universal robustness or final model ranking is claimed.

## Team review

Each member signs their own cell with initials and date after reviewing the relevant section. All entries are pending. Shoaib uses **SSSM**.

| Section | Priyanshu | Sarthak | Sujal | Shoaib (SSSM) |
|---|---|---|---|---|
| 1. Sources and characteristics | Pending | Pending | Pending | Pending |
| 2. Preparation and alignment | Pending | Pending | Pending | Pending |
| 3. Splits and leakage | Pending | Pending | Pending | Pending |
| 4. Quality and suitability | Pending | Pending | Pending | Pending |
| 5. Readiness and reproducibility | Pending | Pending | Pending | Pending |
| References, figures and layout | Pending | Pending | Pending | Pending |
