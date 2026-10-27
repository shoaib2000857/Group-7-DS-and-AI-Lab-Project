# Validation record - 27 October 2026

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
