# Plan validation

26 October 2026:

- Retrieved public metadata for 13 HF model repositories and six author code repositories; all 19 source records were retrieved successfully. No model payloads were downloaded.
- Checked all eight Open-CD config/checkpoint pairs in package A against the upstream config tree and model-zoo filename inventory.
- Validated registry counts: 31 unique role IDs, 25 distinct model identities; 9 detection, 8 captioning, 6 visual inspection and 8 decision entries. Reused weights are explicit.
- Parsed both JSON contract templates and checked local Markdown links.
- Executed the environment probe in the local Windows shell as a portability smoke check. This does not verify the WSL or Colab runtime. Its machine-specific output is kept in ignored `runs/`.
- No learning experiments, GPU inference jobs, paid API calls or benchmark measurements performed. The plan does not pretend that unimplemented adapters already run.

GPU tiers are scheduling estimates. A successful source audit is weaker evidence than a successful model load, and successful model loading is weaker evidence than benchmark correctness.
