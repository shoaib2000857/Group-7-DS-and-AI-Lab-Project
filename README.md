# TerraWatch — Intelligent Satellite Image Change Detection and Analysis

> **Group 7 — IIT Madras BS in Data Science and Applications (DS & AI Lab Project)**

[![Tags](https://img.shields.io/badge/Domain-Remote%20Sensing%20%7C%20Computer%20Vision-blue)]()
[![AI](https://img.shields.io/badge/AI-Deep%20Learning%20%7C%20Multimodal%20%26%20GenAI-green)]()
[![Status](https://img.shields.io/badge/Status-Problem%20Statement%20Approved%20(Milestone%201)-brightgreen)]()

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Key Features](#-key-features)
4. [Architecture Diagram](#-architecture-diagram)
5. [Repository Structure](#-repository-structure)
6. [Dataset Details](#-dataset-details)
7. [Models Used](#-models-used)
8. [Installation Steps](#-installation-steps)
9. [Running Instructions](#-running-instructions)
10. [Results & Evaluation](#-results--evaluation)
11. [Demo Screenshots](#-demo-screenshots)
12. [Team Members](#-team-members)
13. [References](#-references)

---

## 🔍 Project Overview
**TerraWatch** is an intelligent system designed to detect, analyze, and interpret changes on the Earth's surface using multi-temporal satellite imagery. By combining deep learning-based computer vision for change detection and segmentation with multimodal/generative AI for contextual interpretation, TerraWatch aims to distinguish true structural and environmental changes from seasonal, atmospheric, or illumination variations—and provide actionable, human-readable insights.

**Tags:** `Deep Learning (DL)` · `Computer Vision` · `Remote Sensing` · `Change Detection` · `Image Segmentation` · `Multimodal AI` · `Generative AI`

---

## 🎯 Problem Statement
> **Status:** *Approved in principle — Developing comprehensive proposal for Milestone 1 (see [`docs/Problem_Statement.md`](docs/Problem_Statement.md)).*

This project aims to develop an intelligent system for detecting, analyzing, and interpreting changes in the Earth's surface using satellite imagery captured across different time periods. The project will explore deep learning and computer vision techniques to identify and characterize meaningful changes, such as variations in land use, vegetation cover, urban development, and environmental conditions, while distinguishing actual changes from differences caused by seasonal variations, lighting, and image quality.

Additionally, the project may investigate the use of multimodal AI and generative models to provide contextual interpretations and explanations of detected changes, including their possible causes and potential impacts. The specific applications, datasets, and methodologies will be determined through further research and feasibility analysis.

---

## ✨ Key Features
- **Multi-Temporal Satellite Image Alignment & Preprocessing:** Robust handling of lighting, seasonal shifts, and radiometric differences across time periods ($T_1 \rightarrow T_2$).
- **Deep Learning Change Detection & Segmentation:** Pixel-level and semantic identification of land-use change, vegetation loss/gain, urban expansion, and environmental shifts.
- **Multimodal & Generative Interpretation:** Automated natural-language explanations of detected changes, including potential causes and environmental/urban impacts.
- **Interactive Visualization & Reporting Dashboard:** Side-by-side temporal comparison, change masks, and analytical reports.

---

## 🏗️ Architecture Diagram
*Place the finalized architecture diagram at `docs/Architecture_Diagram.png`.*

```text
[Multi-Temporal Satellite Imagery (T1, T2)]
                  │
                  ▼
   [Radiometric & Geometric Preprocessing]
                  │
                  ▼
[DL Change Detection & Semantic Segmentation]
                  │
                  ▼
[Multimodal / Generative AI Contextual Analysis]
                  │
                  ▼
  [Interactive Dashboard & Change Reports]
```

<!-- Uncomment once Architecture_Diagram.png is added:
![Architecture Diagram](docs/Architecture_Diagram.png)
-->

---

## 📁 Repository Structure

```text
Group-7-DS-and-AI-Lab-Project/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── CHANGELOG.md
├── CONTRIBUTING.md
│
├── docs/
│   ├── Problem_Statement.md
│   ├── Project_Proposal.pdf
│   ├── Milestone1_Report.pdf
│   ├── Milestone2_Report.pdf
│   ├── Final_Report.pdf
│   ├── Architecture_Diagram.png
│   └── Presentation.pdf
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Experiments.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── retrieval/
│   ├── agents/
│   ├── evaluation/
│   ├── utils/
│   └── config.py
│
├── app/
│   ├── app.py
│   ├── pages/
│   ├── static/
│   └── templates/
│
├── configs/
│   ├── model.yaml
│   ├── train.yaml
│   └── inference.yaml
│
├── models/
│   ├── checkpoints/
│   └── README.md
│
├── outputs/
│   ├── predictions/
│   ├── reports/
│   ├── figures/
│   └── logs/
│
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_pipeline.py
```

---

## 📊 Dataset Details
*See [`data/README.md`](data/README.md) for detailed instructions on dataset sources, downloading scripts, and preprocessing steps. Candidate multi-temporal remote sensing datasets (e.g., LEVIR-CD, OSCD, SECOND, Sentinel-2 / Landsat archives) are under evaluation for Milestone 1.*

| Dataset Name | Source | Resolution / Modality | Description |
| :--- | :--- | :--- | :--- |
| *TBD (Milestone 1)* | *Link / Source* | *Multi-temporal RGB / Multispectral* | *Satellite change detection benchmark* |

---

## 🧠 Models Used
*See [`models/README.md`](models/README.md) for model architecture details, checkpoints, and weights.*

- **Change Detection & Segmentation Models:** *TBD (e.g., Siamese CNNs / Bi-temporal Transformers / ChangeFormer / BIT / UNet++)*
- **Multimodal / Generative Models:** *TBD (Vision-Language Models for change captioning & contextual explanation)*

---

## ⚙️ Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project.git
cd Group-7-DS-and-AI-Lab-Project
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running Instructions

### Run the Application
```bash
python app/app.py
```

### Run Unit Tests
```bash
pytest tests/
```

---

## 📈 Results & Evaluation
*Summary of key quantitative metrics (e.g., IoU, F1-Score, Precision, Recall, Overall Accuracy, Change Captioning Metrics) and qualitative findings. Full evaluation reports and figures are stored in [`outputs/`](outputs/).*

| Model / Experiment | F1-Score | mIoU | Precision | Recall | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| *Baseline* | - | - | - | - | *Initial benchmark* |

---

## 🖥️ Demo Screenshots
*Screenshots and demo visualizations of multi-temporal change detection masks and AI interpretations will be added here (stored in `outputs/figures/`).*

---

## 👥 Team Members (Group 7)

| Name | Roll Number | Role / Responsibilities | GitHub Profile |
| :--- | :--- | :--- | :--- |
| *Member 1* | *Roll No.* | *Role* | [@shoaib2000857](https://github.com/shoaib2000857) |
| *Member 2* | *Roll No.* | *Role* | *GitHub Link* |
| *Member 3* | *Roll No.* | *Role* | *GitHub Link* |
| *Member 4* | *Roll No.* | *Role* | *GitHub Link* |
| *Member 5* | *Roll No.* | *Role* | *GitHub Link* |

---

## 📚 References
1. *Academic papers, remote sensing benchmarks, and documentation to be populated during Milestone 1 literature review.*
