# Milestone 3: evidence and defence notes

The report presents selected architectures and executable interfaces. Training configurations are proposed specifications unless a completed run is identified. Sarthak's local BIT-R18 pilot used only two epochs and 100 fitting pairs; it is a fit diagnostic, not a converged detector or evidence that BIT wins against the other eight candidates. The released RSCaMa checkpoint was rerun on the same 100-pair validation screen and reproduced its three reported quality metrics exactly; this is a parity check with disclosed vocabulary exposure, not fresh held-out evidence. The latest evaluation proof includes all six caption configurations and both transfer screens; referee measurements retain their earlier L4 scope. Coordinator tests still use labelled test doubles, and the complete learned A/B/C/D chain has not been executed on this machine.

RSCaMa was selected based on the historical automatic LEVIR caption screen, with a local six-configuration aggregate comparison recorded in evaluation_proof.json. Released-checkpoint exposure and parser screening revisions make this model-selection evidence rather than an untouched final test. Strict and in-scope precision have different denominators; keep the report's column names attached to the numbers.

Qwen3-VL's 56.7% figure is raw decision agreement on 150 proxy-labelled claims from 30 validation pairs. It is not human factual accuracy and not the result after the 0.8 operating gate. The text judge is a shared-evidence comparator. Self-generated witness evidence can correlate with the judge; an independent human audit and matched witness comparison answer different questions.

Priyanshu's 50/50 result is on synthetic fixtures. The local rerun again passed 50/50. A separate local real-image diagnostic used three authored claims and two source pairs; only two of three coarse witness/mask-region checks matched. The road crop was flagged changed despite containing no nonzero source-mask pixels. This is not an automatically connected B-stage run. The default BIT producer string in the standalone code is metadata, not proof of a detector invocation.

The structural witness checks visible change and quality, not instance-count truth. The coordinator defers exact-count claims without independent instance evidence. Crops are localisation aids; final image-backed decisions retain full-frame context. The training losses and early-stopping rules in section 7 are implemented configuration choices, not measured validation improvements.

The current consolidated suite passes 83 tests with no skips; these test software contracts, not learned generalisation. The human factuality audit has no completed independent rating file. Use [the Sarthak benchmark report](sarthak/Benchmark_Report.md) for local run details and `../consolidation/PIPELINE_STATUS.md` for component-by-component readiness. The complete pre-consolidation branch history and prior experiment records remain preserved in their existing paths or explicit archive copies.


## 10 November - fetched evidence reconciliation

Commit f6bc2ec supplies the benchmark report, evaluation_proof.json, plotting/evaluation/training scripts and evidence figures. All six caption rows, their historical/local deltas and the supplementary metrics agree with the nine-page submission. BIT's 45.8% F1, 26.8% change-class mean IoU and 31/45 alarms agree with the benchmark report. The proof hashes identify the friend's local raw run files and weights; those ignored data files remain on that host. The supplied scripts and proof are preserved under sarthak/.

That commit contained unresolved conflict markers in the POC Markdown and README and replaced the POC PDF with a different build. The POC source/PDF are reconciled to the unchanged, previously checked reports/ submission. The prior commit remains in Git history, and all newly supplied experiment code and evidence are retained. Sign-off and submission content are unchanged.
