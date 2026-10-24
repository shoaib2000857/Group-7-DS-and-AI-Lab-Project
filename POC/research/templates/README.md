# Tracking templates

These CSV files contain headers only. Populate actual measurements and evidence when implementation begins; blank fields are not zero-valued results.

- `data_access_register.csv`: distinguish identified URLs from acquired/audited assets and usage rights.
- `subset_manifest.csv`: one row per unique image pair; keep captions/instructions associated with that pair.
- `experiment_register.csv`: one row per immutable configuration/run; state pretrained training exposure.
- `claim_annotations.csv`: one row per rater and atomic claim; retain adjudication separately or as an explicitly named adjudicator record.

Store restricted imagery, credentials, personal data and bulky weights outside these registers. Hashes and relative paths suffice for most provenance needs. Human audit setup and sample selection are specified in `../04_experiments_and_evaluation.md`.
