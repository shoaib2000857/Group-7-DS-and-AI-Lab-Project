# TerraWatch proof of concept

This is a local exploration area for the IITM DS & AI Lab project. Start with [the research guide](research/README.md).

For the four-person model-screening assignments and Colab/WSL experiment protocol, use [the benchmark work packages](benchmarks/README.md).

## Working boundaries

- Research, feasibility checks, and future small experiments belong here. The submitted project outside `POC/` remains separate.
- Do not push, publish, or submit anything unless the user later requests it. No commits were made during this research pass.
- Use at most 30% of each selected dataset's unique image pairs for the entire POC, with a 20% default. Include training, validation, evaluation, annotation, and development examples in that accounting. This is the working interpretation of the user's cap; see the research data protocol.
- Do not train full models during the POC. Prefer pretrained inference and tightly bounded training pilots when implementation begins. This research pass performs neither training nor dataset/weight downloads.
- Preserve `DOCS/` as supplied context. Old plans are proposals; official requirements and newer written TA feedback take precedence.
- Keep future data, weights, caches, and generated experiment outputs local. Check ignore rules before staging any future work.

Research prepared: 24 October 2026. Recommendations are provisional until data and compute are audited.
