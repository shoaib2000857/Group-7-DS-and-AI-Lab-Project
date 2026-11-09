# Milestone 3: evidence and defence notes

The report presents selected architectures and executable interfaces. A training configuration is a fixed experimental specification; it is not a claim that its training run completed. The original consolidation included the three-class BIT training configuration. Sarthak's 9 November supplied summary adds a two-epoch BIT experiment with a 100/25/100 protocol; its measured experiment is separate from the full training profile. The captioner/decision results are supplied earlier experiments. New coordinator tests use test doubles. No full learned A/B/C/D accuracy figure is claimed.

RSCaMa is selected because the domain-specific configurations outperform general VLMs in the automatic LEVIR caption screen. Released-checkpoint exposure and parser screening revisions make this model-selection evidence rather than an untouched final test. The subset retrain controls the released-caption-checkpoint issue. Strict and in-scope precision have different denominators; keep the report's column names attached to the numbers.

Qwen3-VL's 56.7% figure is raw decision agreement on 150 proxy-labelled claims from 30 validation pairs. It is not human factual accuracy and not the result after the 0.8 operating gate. The text judge is a shared-evidence comparator. Self-generated witness evidence can correlate with the judge; an independent human audit and matched witness comparison answer different questions.

Priyanshu's 50/50 result is on synthetic fixtures. The real-image demonstration has three authored claims on two pairs, not three independent pairs and not an automatically connected B-stage run. The default BIT producer string in that standalone code is metadata, not proof of a detector invocation. The coordinator supplies actual producer IDs and preserves original frame coordinates.

The structural witness checks visible change and quality, not instance-count truth. The coordinator defers exact-count claims without independent instance evidence. Crops are localisation aids; final image-backed decisions retain full-frame context. The training losses and early-stopping rules in section 7 are implemented configuration choices, not measured validation improvements.

The complete pre-consolidation branch history is bundled locally; every inventoried local file is hash-verified. Original experiments and trial/failure records remain under their existing paths or explicit archive copies. Use `../consolidation/PIPELINE_STATUS.md` for component-by-component readiness.


## 10 November local-result reconciliation

- Source: Sarthak's supplied Markdown, preserved byte-for-byte with SHA-256 receipt. It reports 83 passing tests, the caption reruns, supplementary caption metrics and the two-epoch BIT experiment. These are attributed supplied results, not fresh model runs by the report builder.
- All six historical caption rows were cross-checked against repository B0 JSON; displayed deltas were recalculated. Transfer values agree at reported precision. False-alarm counts use 40 unchanged caption pairs and 45 unchanged detector pairs.
- BIT class-mean road/building IoU (26.8%) and Otsu binary IoU (9.1%) have different targets. Only binary F1 and unchanged-pair alarms are compared directly.
- The supplied report references a sarthak/ artifact folder, evaluation_proof.json, runners and three figures. That folder was absent from fetched remote heads and the local repository. No raw captions, training logs, bootstrap draws or checkpoint were supplied with this Markdown. Figure 4 is generated from the supplied point estimates; it does not invent intervals or run records. Historical referee scores retain their L4 origin.
- The raw-release reproduction uses upstream official-split vocabulary construction, including evaluation-split caption vocabulary. Its aggregate reproduction does not establish clean held-out generalisation or pairwise prediction identity. The strict preparation package and training-only retrain vocabulary have separate scope.
- The real-image witness 2/3 outcome concerns coarse region presence, not human caption factuality. Human evaluation procedure and component orchestration retain the definitions in the report.
- Sign-offs: Shoaib explicitly confirmed that all four teammates discussed and approved the report and instructed initials for all members. The report records that confirmation on 10 November 2026. This does not represent new GitHub review submissions.
