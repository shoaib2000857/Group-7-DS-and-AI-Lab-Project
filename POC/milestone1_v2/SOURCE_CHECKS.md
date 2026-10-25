# Source verification and editorial audit

Checked 25 October 2026. The bibliography is in the report. Earlier reading-depth records remain in `../research/sources.md`.

## Checks performed for this revision

- Reopened original report and close prior-art primary pages: Change-Agent, ChangeChat, CCExpert, Spot-the-shift; checked relevant feature/task/metric sections. Change-Agent Table II directly supports the numerical reference values. No local reproduction is claimed.
- Rechecked ChangeFormer, Reality Check, Chg2Cap and KCFI abstracts. Checked RSCaMa's author repository and arXiv abstract; TEOChat's author repository and paper/abstract.
- FMScore's publisher page was accessible in the previous research pass but returned 403 on this pass. Its bibliographic record and previously read publisher abstract support the narrow claim that fact matching exists; full metric implementation has not been audited. The report does not infer that other systems lack verification.
- Read official TypeSafe Jev 1.14.0 release information: image support is experimental. Checked Cloudflare Clef release information and current official Ollama Clef/Clef-flash pages. Checked Qwen's author model card and Google's vision documentation. These establish candidate capabilities, not remote-sensing benchmark performance.
- Read the primary Jev bias-study abstract; its results motivate option-order/wording tests, not a claim that every model or task has the same error pattern.
- Reopened dataset-author pages for LEVIR, S2Looking, SYSU, SECOND and OSCD, and the LEVIR caption-family source. No dataset payload, complete rights audit, checkpoint or model was downloaded.
- TerraScope was discovered through the optional model-shortlist follow-up. Its primary CVPR supplemental describes pixel grounding and building-change evaluation; it is included in the final matrix and team follow-up. Full method/code comparison remains pending, and the report makes no claim of being first.

## Primary links newly relevant to this revision

- https://typesafe.ai/blog/jev-1-14
- https://blog.cloudflare.com/clef-decision-models/
- https://blog.cloudflare.com/clef-faster-cheaper-multimodal/
- https://ollama.com/library/clef
- https://ollama.com/library/clef-flash
- https://arxiv.org/abs/2610.19619
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://ai.google.dev/gemma/docs/capabilities/vision
- https://arxiv.org/abs/2404.18895
- https://github.com/Chen-Yang-Liu/RSCaMa
- https://arxiv.org/abs/2410.06234
- https://github.com/ermongroup/TEOChat
- https://rcdaudt.github.io/oscd/
- https://openaccess.thecvf.com/content/CVPR2026/supplemental/Shu_TerraScope_Pixel-Grounded_Visual_CVPR_2026_supplemental.pdf

## Deliberate corrections

No claim of universal robustness, exact physical resolution inferred from dimensions, verified object counts from components, automatic legal assessment, guaranteed hallucination reduction from a decision model, or measured runtime remains. The custom pipeline metric is explicitly exploratory and must be defined and validated before final testing. Full training and main-repository promotion remain later steps.
