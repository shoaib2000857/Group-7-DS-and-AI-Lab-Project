# Validation record - 27 October 2026

## Received GPU runs and runtime .2 repair

Three actual Colab L4 archives were received and preserved under `POC/benchmarks/results/2026-10-27/shoaib/archives/`; see the [checked findings](../results/2026-10-27/shoaib/README.md). All file-manifest checksums and ZIP CRC checks pass. The two smoke metric tables are identical; their IDs are contained in the screen set. The union is 40 real image pairs. None contains learned-model predictions: environment creation stopped every selected judge and the witness before inference.

Runtime `2026-10-27.2` replaces the failing stdlib-venv bootstrap with private `virtualenv==20.35.3`, keeps its seeding helpers outside the notebook kernel, captures bootstrap stdout/stderr and checks both required environments before data loading. A shared bootstrap failure saves a diagnostic ZIP and stops the run rather than continuing to a controls-only result. Reports also export explicit run-health status and visibly flag absent learned outputs.

The exact production bootstrap functions were exercised on Linux in WSL Ubuntu 24.04, using Python 3.12 and an ext4 `/tmp` destination: private pip 25.2 ran successfully, `sys.prefix != sys.base_prefix`, and system-site-packages was enabled. A first probe on the Windows-mounted project drive exceeded the 90-second creation limit; the native Linux filesystem probe passed. Keep WSL worker caches on its Linux filesystem; Colab's default working/cache paths remain local under `/content`. This verifies the replacement bootstrap, not CUDA model inference or the user's exact Python 3.13 Colab stack.

Fourteen tests now pass, adding checks that the bootstrap avoids the old stdlib-venv command, captures setup logs and stops early with an exported diagnostic on a shared bootstrap failure. The original checks below remain applicable. A fresh `.2` smoke run is still needed to validate GPU inference; `.1` failure archives should not be resumed into the revised runtime.

## Completed locally

- Notebook scaffold built with `nbformat`; all 30 cells validated and Python cells compiled. The runtime embedded in the single-file notebook has a checked SHA-256 and is regenerated from the readable source.
- Full top-to-bottom execution in **local CPU/data/export validation mode**, using one real official-training pair and one real official-validation pair from the author's LEVIR-MCI archive. The notebook's normal user default remains four development + six validation pairs. Local mode is explicitly marked and does not fabricate learned-model outputs.
- Checked actual remote ZIP layout, caption/flag schema and bounded HTTP-range responses. Only archive metadata and selected image members were downloaded; no full dataset archive or model weights were downloaded by these local checks. Selected IDs follow the same seeded prefixes as the default shared candidate manifest.
- Rule controls, independent reference storage, metrics, output tables, sample-pair figures, review-sheet generation, HTML results and combined ZIP export exercised on those inputs. A Windows transient ZIP replacement issue found during full execution was fixed with a bounded atomic-replace retry.
- Executed notebook exported to HTML and inspected in a rendered browser preview. Tables, charts and original image-pair outputs were inspected; screenshots and executed local notebooks stay in ignored `qa/`.
- Twelve targeted tests cover inference-payload reference separation, undefined precision/all-defer behaviour, multiclass Brier arithmetic, failure denominators, probability validation, strict decision-code parsing, the fixed acceptance gate, raw-versus-thresholded output separation, pair-disjoint constructed fixtures, unsafe ZIP paths, repeat-export checksums, visible missing outputs, resume-configuration mismatch and interrupted trailing prediction recovery. Some checks share one test method. Manufactured probability records are evaluator tests, never model findings.

## Not established by local checks

- No CUDA GPU or local PyTorch model stack was available in the authoring Python environment. **J01-J07 weight loading, inference quality, actual Colab memory fit, native server compatibility and GPU timing are unverified until run on the user's GPU session.** Candidate APIs were implemented from publisher sources and record failures rather than inventing outputs.
- No Colab account, CLI bridge, Drive authorisation, live Colab download interaction, optional Ollama/Kev installation or paid/hosted service was exercised. Colab supplies the final native GPU validation. The notebook includes local-only optional server routes; Jev remains access/cost deferred.
- No human factual audit, independent transfer benchmark, trained detector result, instance-count reference, full end-to-end improvement or custom-score validation has been completed. Synthetic fixtures and reference-annotation proxies are not substitutes.

## Reproduce

```powershell
python POC/benchmarks/colab/build_notebook.py
python -m unittest discover -s POC/benchmarks/colab -p test_benchmark_runtime.py -v
```

For local notebook plumbing validation, set `TW_NOTEBOOK_TEST=1` and execute the notebook with `nbclient` or `jupyter nbconvert --execute`. This explicitly reduces the sample to two real pairs and disables learned-model scheduling/download interaction; it is not the GPU benchmark. Do not set this variable for the actual Colab run.

For the intended GPU check, upload the notebook to Colab, choose a GPU and run all cells with the default `smoke` profile. Retain the complete ZIP including model failures; inspect the register and source/runtime records before interpreting the scores. Then pilot human annotations and select finalists on validation before sealed evaluation.
