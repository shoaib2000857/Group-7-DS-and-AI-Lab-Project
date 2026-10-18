# Dataset Documentation (`data/`)

> **Important:** Do **not** commit large datasets to the Git repository. Only small sample files inside `data/sample/` should be tracked in version control.

---

## Directory Structure
- `raw/` — Original, immutable data dumps (ignored by Git).
- `processed/` — Cleaned, transformed, and feature-engineered datasets ready for modeling (ignored by Git).
- `sample/` — Small representative subset of the data for quick testing, CI verification, and demo runs (tracked by Git).

---

## Dataset Sources
*Update this section once the dataset is finalized:*
- **Primary Dataset:** *Name & URL / Source*
- **License / Terms of Use:** *Specify license*

---

## Download Instructions
*Provide commands or script instructions to download the full dataset into `data/raw/`:*

```bash
# Example:
# python src/data/download_data.py --output_dir data/raw/
```

---

## Preprocessing Pipeline
*Summarize the steps performed to convert `data/raw/` into `data/processed/` (e.g., deduplication, missing value imputation, tokenization, resizing, train/val/test split).*
