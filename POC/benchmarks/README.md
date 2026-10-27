# TerraWatch POC: four-person benchmark plan

**Runnable work package D:** [Shoaib's Colab notebook](colab/TerraWatch_Shoaib_Decision_Benchmarks.ipynb) · [Run instructions and data contract](colab/README.md). Includes incremental outputs and a combined downloadable ZIP. This is the first implementation; GPU benchmark findings are not yet established.

**Received run review:** [two smoke runs and one screen run](results/2026-10-27/shoaib/README.md). The original ZIPs are retained locally with verified hashes. They expose a setup failure before learned inference; runtime `.2` repairs the bootstrap and needs a fresh Colab smoke check.

**26 October 2026 | Planning and source verification, not completed model runs**

We have Colab Pro+ and WSL/CLI access, as confirmed by the team. Actual assigned GPU, VRAM, installed libraries and model compatibility remain to be measured. Start with inference; do not train every candidate.

## Work allocation

Assignments are practical starting roles, not assumptions about anyone's expertise. Owners build adapters and run comparisons; everyone contributes to the blinded factual audit.

| Owner | Work package | Candidate pool | Run first |
|---|---|---|---|
| **Sarthak Saini** | [A: change detection and shared image manifests](A_detection.md) | **9 detectors** | FC-EF, BIT-R18, ChangeFormer-B0; absdiff/Otsu control |
| **Sujal Raj Pradhan** | [B: captions and claim extraction](B_captioning.md) | **8 caption models** | RSCaMa, Qwen3-VL-4B, SmolVLM2-2.2B; evidence-template control |
| **Priyanshu Agarwal** | [C: original-image inspection and input adaptation](C_visual_evidence.md) | **6 visual inspectors**, plus **6 preprocessing approaches** | Qwen2.5-VL-3B, InternVL3.5-4B; unchanged-input control |
| **Shoaib SSM** | [D: claim decisions, calibration and integration](D_decisions.md) | **8 learned decision/judge candidates** | Qwen3-4B text judge, d3-mini; rules and no-check controls |

There are **31 model-role entries**, including reused VLMs across roles; these are not 31 independent model families. There are **25 distinct model identities** in the four pools. Preprocessing methods, rules, templates and different quantisations are not counted as extra learned models. Candidate pools meet the 5–10-per-learned-stage goal; the first wave is deliberately smaller.

## First-wave run order

1. **Shared setup:** Sarthak freezes dataset IDs/splits with Priyanshu; Shoaib freezes output contracts; each member records their actual Colab/WSL environment. Use [SHARED_PROTOCOL.md](SHARED_PROTOCOL.md).
2. **Independent smoke tests:** each owner uses the same 10 selected development pairs. B runs on images without waiting for A. C uses manually defined development crops. D uses a hand-checked development evidence bundle. These are diagnostic fixtures, not final evaluation data.
3. **Screening:** run first-wave candidates on the shared validation-screen set, then add remaining feasible candidates in the prescribed order. Log blocked candidates instead of spending days forcing legacy builds.
4. **Freeze finalists:** retain at most two candidates per role using quality, resource use, failure coverage and access. Do not choose only by one scalar score.
5. **Integrate:** connect one baseline chain first. For the reliability hypothesis hold candidate captions and detector outputs fixed, then vary checks. A full 9×8×6×8 Cartesian search is unnecessary.
6. **Held-out benchmark:** only the frozen shortlist sees the sealed POC evaluation set. Report stage results and end-to-end ablations separately. Negative or inconclusive findings remain valid outcomes.

## Files to use

- [SHARED_PROTOCOL.md](SHARED_PROTOCOL.md): data tracks, fair comparisons, metrics, artifact contracts, compute limits and seven-day sequence.
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md): primary sources, checkpoint status, known blockers and reading limits.
- [source_metadata.json](source_metadata.json): live metadata snapshot, revisions and model-file listings; not downloaded weights.
- [candidate_registry.json](candidate_registry.json): 31 role assignments, 25 unique model identities, owners, priority and explicit not-started status.
- [scripts/probe_environment.py](scripts/probe_environment.py): executable, read-only environment probe for Colab or WSL.
- [scripts/audit_source_metadata.py](scripts/audit_source_metadata.py): refreshes public repository/model metadata without downloading datasets or weights.
- [templates/run_record.json](templates/run_record.json): empty result/provenance record; null fields are not measured zeroes.
- [templates/claim_decision.json](templates/claim_decision.json): common claim/evidence/decision contract.

This plan originally contained no adapters, unified evaluator or downloader. The [work package D implementation](colab/README.md) now supplies its decision adapters, controls, evaluator, bounded primary-data loader and handoff export. The other stage adapters and training scripts remain separate deliverables. Upstream examples and CPU validation do not establish complete GPU pipeline benchmark results.

## Scope and compute boundaries

Keep all POC data use within the existing 20% default / 30% maximum unique-pair cap, including annotation and inspection. Use task-compatible data, not a union of nine unrelated label sets. Existing Colab access does not authorise new purchases, credit top-ups or paid model APIs. Jev and cloud-hosted comparisons remain optional and blocked until access and spending scope are explicitly settled. This research pass starts no GPU jobs or paid requests.

Ready means source/checkpoint identified; locally verified means a successful run with saved evidence. All candidates currently remain **not locally benchmarked**. The source audit distinguishes the two.
