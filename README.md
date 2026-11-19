# TerraWatch

TerraWatch compares aligned satellite images taken at two times, describes visible building and road changes, and links each claim to the evidence a reviewer can inspect. Group 7 develops the system for IIT Madras BSDA4001.

| Stage | Role | Main code |
|---|---|---|
| BIT-ResNet18 | Predict background, road and building change pixels | [Model workers](src/models/workers.py) |
| RSCaMa and claim parser | Generate a change caption and extract structured claims | [Captioner](src/models/captioner.py) · [Claims](src/models/claims.py) |
| Visual witness | Match regions to claims, inspect crops and record quality flags | [Evidence modules](src/evidence/) |
| Qwen3-VL and decision gate | Assess claims and produce accept, reject or defer outcomes | [Referee](src/models/referee.py) · [Coordinator](src/inference/pipeline.py) |
| Human review | Inspect evidence and record flags, decisions and corrections | [M4 review server](src/app/server.py) |

## Quick start

Use Python 3.11 or 3.12. From the repo root:

```sh
python -m pip install -r requirements-pipeline.txt
python scripts/replay_milestone4.py
python -m src.app.server RUN_DIR/train_000010 RUN_DIR/val_000005
```

Use the printed path for `RUN_DIR` and open `http://127.0.0.1:8010`. The CPU replay uses real LEVIR-MCI imagery, an Otsu mask, authored claims and a defer-only decision control. The viewer also imports learned-worker records produced by the coordinator.

For the learned path, follow the [model setup guide](docs/models/README.md) to install the three worker environments, acquire pinned model sources and checkpoints, configure `configs/pipeline.json` and run `scripts/run_pipeline.py`.

## Data, training and evaluation

- [EDA and preparation notebooks](notebooks/README.md) produce the audited split manifests and categorical masks. LEVIR-MCI is the primary paired image, mask and caption dataset. The strict cohort contains 9,544 pairs. SECOND-CC and historical WHU-CDC screens provide transfer comparisons.
- [Training and batch evaluation commands](docs/models/README.md) cover detector and captioner training, checkpoint loading and reference-isolated scoring.
- [Milestone 4 training notebook and walkthrough](docs/milestone4/Walkthrough.md) reproduce the bounded validation comparison and review workflow.
- Run `python -m pytest -q` to check CPU contracts and model orchestration.

## Project materials

| Location | Contents |
|---|---|
| `src/`, `scripts/`, `configs/` | Reusable implementation, commands and model/data profiles |
| `notebooks/` | Exploratory and executable data/model walkthroughs |
| `reports/`, `presentation/` | Milestone reports and meeting materials |
| `POC/` | Original experiments, benchmark evidence and M4 working artifacts |
| `docs/` | Dataset and model contracts, instructions and environment setup |

[Milestone 2 data report](reports/Milestone2_Report.md) · [Milestone 3 architecture report](reports/Milestone3_Report.md) · [Milestone 4 training and review](https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project/pull/20) · [Contribution log](WORKLOG.md)

Priyanshu Agarwal, Sarthak Saini, Sujal Raj Pradhan and Shoaib SSM.
