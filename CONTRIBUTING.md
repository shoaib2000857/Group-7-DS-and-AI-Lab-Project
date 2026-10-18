# Contributing Guidelines — Group 7 DS & AI Lab Project

Welcome to the **Group 7 DS & AI Lab Project** repository! To keep our codebase clean, reproducible, and easy to collaborate on across all team members, please follow the guidelines below.

---

## 1. Coding Conventions

- **Style Guide:** Follow [PEP 8](https://peps.python.org/pep-0008/) for all Python code.
- **Type Hints & Docstrings:** Use Python type annotations and clear docstrings for functions and classes in `src/`.
- **Notebooks vs. Source Code:**
  - Use `notebooks/` for Exploratory Data Analysis (EDA), initial experimentation, and visualizations.
  - Once code stabilizes, refactor reusable functions, classes, and pipelines into `src/`.
- **Configuration Management:** Avoid hardcoding file paths or hyperparameters inside scripts. Store parameters in `configs/*.yaml` and load them via `src/config.py`.
- **Data & Artifacts Hygiene:**
  - **Never commit large datasets** to Git. Place small representative samples in `data/sample/` and document download steps in `data/README.md`.
  - Keep generated outputs (predictions, reports, plots, logs) inside `outputs/`, not mixed with source code.

---

## 2. Branch Naming Conventions

Never push experimental or untested work directly to `main`. Create a descriptive branch for your task:

| Prefix | Purpose | Example |
| :--- | :--- | :--- |
| `feat/` | New feature, pipeline, or model | `feat/rag-pipeline`, `feat/yolo-detector` |
| `fix/` | Bug fix or error resolution | `fix/data-loader-nulls` |
| `exp/` | Notebook experiments or EDA | `exp/eda-initial-analysis` |
| `docs/` | Documentation, reports, or README updates | `docs/milestone-1-report` |
| `refactor/` | Code cleanup or restructuring | `refactor/preprocessing-module` |
| `test/` | Adding or updating unit tests | `test/pipeline-unit-tests` |

---

## 3. Pull Request (PR) Process

1. **Sync with `main`:** Before creating a branch or opening a PR, pull the latest changes:
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Create your branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. **Commit with clear messages:** Write concise, imperative commit messages (e.g., `Add data preprocessing script for text cleaning`).
4. **Run tests locally:** Ensure existing tests in `tests/` pass before pushing:
   ```bash
   pytest tests/
   ```
5. **Open a Pull Request:**
   - Provide a clear summary of what was added or changed.
   - Link any relevant issues or milestone tasks.
   - Request a review from at least one other group member before merging into `main`.

---

## 4. Team Responsibilities

| Area | Key Responsibilities | Directory Scope |
| :--- | :--- | :--- |
| **Project Management & Docs** | Maintaining milestone reports, presentations, architecture diagrams, and changelog | `docs/`, `README.md`, `CHANGELOG.md` |
| **Data Engineering & EDA** | Dataset collection, cleaning, preprocessing pipelines, and exploratory analysis | `data/`, `notebooks/01_EDA.ipynb`, `src/data/` |
| **Modeling & AI Pipelines** | Model training, retrieval/RAG, agentic workflows, and hyperparameter tuning | `src/models/`, `src/retrieval/`, `src/agents/`, `configs/` |
| **Evaluation & Testing** | Quantitative metrics, error analysis, unit/integration testing | `src/evaluation/`, `tests/`, `outputs/` |
| **Application & Deployment** | Building and maintaining the UI (Streamlit / Gradio / Flask) and inference hooks | `app/`, `src/inference/` |
