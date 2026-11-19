# Milestone 4 code walkthrough

## Start with the real-image replay

From the repository root, install `requirements-pipeline.txt` into the coordinator environment and run:

```powershell
python POC/milestone4/replay_demo.py
python POC/milestone4/review_server.py RUN_DIR/train_000010 RUN_DIR/val_000005
```

Replace `RUN_DIR` with the path printed by the replay script. Open `http://127.0.0.1:8010`. Flag the construction claim, save, then record a decision with a reviewer name and a reason. Reload and open Review history. Use Export audit JSON to inspect both the original and reviewed outcomes.

## Follow one input through the system

| Step | Open during the walkthrough | Output to show |
|---|---|---|
| Data and splits | `src/data/dataset.py`, M2 manifest and mask mapping | RGB T1/T2, matching dimensions, pair ID, split and hashes |
| Training and selection | `POC/milestone4/colab/bit_sweep.py`, recorded run config and histories | Focal/Dice objective, paired flips, validation class IoU and selected epoch |
| Learned detection and captioning | `src/models/workers.py`, `src/models/captioner.py`, `configs/pipeline.json` | 0/1/2 mask, caption with exact source spans after parsing |
| Region and witness | `src/inference/pipeline.py`, `src/evidence/` | Matching boxes in original T1 coordinates, padded crops and lighting/visibility flags |
| Prompt and decision | `src/models/referee.py`, `gated_decision` in the coordinator | T1/T2 order, fallible evidence, label-token probabilities, raw and gated outcomes. Evaluation references stay in the scoring driver. |
| Final output | `report.json` and the review UI | Input hashes, claim results, crops, corrections, flags and exported audit history |

Explain that the replay uses Otsu pixel differences, authored claims and a defer-only decision control. The accepted result in the audit smoke test is a scripted reviewer override. Show the mode field and use live reviewer input during the meeting. For learned mode, run `scripts/check_model_setup.py --environments --strict` once the worker assets are present, then use `scripts/run_pipeline.py` and pass the resulting run directory to `review_server.py`.

Discuss the classifier's foreground imbalance, the specialist captioner's performance on the shared screen, the witness's sensitivity to lighting and alignment, and the need for independent instance evidence for exact counts. Show a deferred count claim and a missing-assets preflight. Evidence retrieval follows run and claim IDs to the saved images and records. The pipeline does not require a vector index. See [model dependencies and setup](../../docs/models/README.md) for worker environments, CUDA compatibility and checkpoint acquisition.

## Colab training

Upload `colab/TerraWatch_M4_BIT_Training.ipynb` to Colab. Use T4 for the two-arm detector comparison or L4 if faster turnaround is necessary and the available compute budget allows it. Upload `colab/drive_members.py` when prompted and authenticate with access to the extracted LEVIR-MCI folders. The notebook checks member hashes, keeps a fixed validation population and saves configs, curves, checkpoints and the selection receipt in the downloaded results ZIP. The fitting budget is 36 minutes in total. Cloud allocation, Drive access and export time depend on the session.

## Replay the trained detector outputs

```powershell
python POC/milestone4/review_server.py POC/milestone4/evidence/trained_detector_demo/train_000010 POC/milestone4/evidence/trained_detector_demo/val_000005
```

These records contain the selected local BIT model's actual masks and crops. The captions are authored prompts and the referee is a defer-only control. The archive receipt identifies the selected checkpoint. Use `replay_demo.py --checkpoint PATH --bit-source PATH --detector-python PYTHON` to regenerate this replay from the weights.

The optional `colab/TerraWatch_M4_Learned_Demo.ipynb` uses an L4 and a training ZIP or BIT checkpoint to execute the complete learned path. It exports decisions as produced and can return deferrals for both pairs. Keep this separate from the measured training comparison. Weight downloads and package installation affect the runtime.

## Review the completed learned run

```powershell
python POC/milestone4/review_server.py POC/milestone4/evidence/colab_learned_demo/train_000010 POC/milestone4/evidence/colab_learned_demo/val_000005
```

Open `http://127.0.0.1:8010`. Inspect the construction deferral, then the no-change claim. Open the evidence details to show Qwen's raw accept score of 0.6357 on the no-change claim and the gate's final defer. During the meeting a reviewer can record an independent acceptance or correction and reload the UI to show the audit event. The scripted CPU review replay demonstrates that action, while the learned records preserve the original model outcomes.

The completed L4 training run used 192 strict training and 64 validation pairs, with four epochs per arm. The winning checkpoint is `lr_3e-4.pt` at epoch 3 in `colab/downloads/TerraWatch_M4_BIT_20261119_165506.zip`. Its SHA-256 is in [selection.json](evidence/colab_comparison/selection.json). Use this ZIP in Notebook 2 to repeat the learned demonstration with the selected checkpoint. The current learned run uses the earlier upload recorded by the notebook.
