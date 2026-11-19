# Milestone 4 working area

[19 November readiness review](audit/2026-11-19/Milestone4_Readiness_Review.md) compares fetched teammate work with the M2/M3 evidence and all TA training/interface requirements. Supporting asset checks and branch inventories remain next to the review. Original source branches, experiment variants and submitted reports are preserved.

This working area contains internal review/planning evidence, separate from the submission report.

Use the [M4 submission and review checklist](Submission_Checklist.md) to track training, the engineering enhancement, the presentation, the code walkthrough and the examiner-facing repository.


- [Colab BIT comparison](colab/TerraWatch_M4_BIT_Training.ipynb) and [learned demo notebook](colab/TerraWatch_M4_Learned_Demo.ipynb)
- [Execution walkthrough and review app quick start](Walkthrough.md)
- [Milestone 4 report source](Milestone4_Report.md)
- [Report PDF](deliverables/TerraWatch_Milestone4_Report.pdf) and [meeting slides](deliverables/TerraWatch_Milestone4_Slides.pdf)

Build both PDFs with `python POC/milestone4/build_deliverables.py`. To reconcile returned Colab results, run `python POC/milestone4/ingest_bit_results.py <run.zip> --target POC/milestone4/evidence/colab_comparison --update-report`, then rebuild. Keep the original run ZIP intact as the checkpoint bundle.
