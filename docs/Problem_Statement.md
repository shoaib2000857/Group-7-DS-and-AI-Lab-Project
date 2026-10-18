# Problem Statement & Milestone 1 Proposal Roadmap

## 1. Approved Brief Problem Statement

**Title:** TerraWatch — Intelligent Satellite Image Change Detection and Analysis

**Description:**
This project aims to develop an intelligent system for detecting, analyzing, and interpreting changes in the Earth's surface using satellite imagery captured across different time periods. The project will explore deep learning and computer vision techniques to identify and characterize meaningful changes, such as variations in land use, vegetation cover, urban development, and environmental conditions, while distinguishing actual changes from differences caused by seasonal variations, lighting, and image quality.

Additionally, the project may investigate the use of multimodal AI and generative models to provide contextual interpretations and explanations of detected changes, including their possible causes and potential impacts. The specific applications, datasets, and methodologies will be determined through further research and feasibility analysis.

**Tags:** Deep Learning (DL), Computer Vision, Remote Sensing, Change Detection, Image Segmentation, Multimodal AI, Generative AI

---

## 2. Comprehensive Proposal & Milestone 1 Framework
*(Guided by Instructor Feedback upon Approval in Principle)*

As we expand this brief into our comprehensive Project Proposal (`docs/Project_Proposal.pdf`) and Milestone 1 Report (`docs/Milestone1_Report.pdf`), the following core pillars must be researched and documented:

### A. Specific Real-World Problem
- [ ] **Problem Definition:** What exact real-world surface change challenge are we targeting (e.g., rapid urban sprawl/encroachment, deforestation & vegetation loss, flood/disaster damage assessment, agricultural land-use shifts)?
- [ ] **Operational Pain Point:** Why is manual inspection or simple pixel-differencing of multi-temporal satellite imagery insufficient at scale (e.g., false positives from cloud cover, shadows, seasonal phenology, sensor angles)?

### B. Key Stakeholders & Beneficiaries
- [ ] **Primary Users / Decision Makers:** Who directly uses or benefits from the system (e.g., urban planners, environmental monitoring agencies, disaster response teams, forestry departments, agricultural analysts, municipal bodies)?
- [ ] **Actionable Value:** How do the system's outputs (change masks + multimodal explanations) help these stakeholders make faster or better-informed decisions?

### C. Current Landscape & Literature Review (Academic & Commercial SOTA)
- [ ] **Traditional Remote Sensing Approaches:** Algebraic methods (NDVI/NDWI differencing, CVA), classical ML (Random Forests, SVM on spectral bands), and their sensitivity to radiometric/seasonal noise.
- [ ] **Academic State-of-the-Art (Deep Learning):**
  - Bi-temporal Siamese CNNs (FC-EF, FC-Siam-conc, FC-Siam-diff, STANet, SNUNet)
  - Transformer-based Change Detection (BIT — Bitemporal Image Transformer, ChangeFormer)
  - Foundation Models & Vision-Language Change Captioning (RSICC, ChangeChat, RemoteCLIP, Prithvi / Clay geospatial foundation models)
- [ ] **Commercial Landscape:** Existing platforms (Google Earth Engine, Planet Analytic Feeds, Picterra, EOSDA LandViewer) and their limitations (e.g., lack of automated semantic reasoning/explanation, high cost, domain-specific rigidity).

### D. Proposed Contribution & Value Proposition
- [ ] **What Distinguishes TerraWatch:**
  - **Effectiveness:** Robustness against nuisance variations (seasonal/illumination shifts) vs. true structural/environmental change.
  - **Novelty & Interpretability:** Bridging low-level pixel/semantic segmentation masks with high-level Multimodal/Generative AI explanations (quantifying *what* changed, *how much* changed, and providing contextual interpretation of *possible causes and impacts*).
  - **Efficiency & Usability:** End-to-end workflow from bi-temporal image pairs to visual masks and natural-language analytical summaries.

### E. Data Availability, Quality & Feasibility Analysis
- [ ] **Candidate Datasets & Sources:**
  - **LEVIR-CD / LEVIR-CD+:** High-resolution (0.5m/pixel) bi-temporal Google Earth imagery for building/urban change detection (637+ image pairs of $1024 \times 1024$).
  - **SECOND (Semantic Change Detection Dataset):** 4,662 pairs of aerial images with multi-class land-cover change annotations (non-vegetated ground, trees, low vegetation, water, buildings, playgrounds).
  - **OSCD (Onera Satellite Change Detection):** Multispectral Sentinel-2 bi-temporal imagery across 24 global cities.
  - **LEVIR-CC / Dubai-CC:** Remote Sensing Image Change Captioning datasets pairing bi-temporal images with natural language descriptions of changes.
  - **Live / Open Satellite Feeds:** Copernicus Sentinel-2 / NASA Landsat for real-world inference demonstrations.
- [ ] **Feasibility Justification:** Volume of labeled pairs, spatial/spectral resolution quality, compute requirements for training/fine-tuning, and reliability of evaluation benchmarks.
