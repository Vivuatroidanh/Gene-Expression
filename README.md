# ALS Transcriptomic Diagnostic Pipeline

## Overview

This repository implements an end-to-end diagnostic pipeline for ALS (Amyotrophic Lateral Sclerosis), spanning raw transcriptomic input to cross-platform external validation. As illustrated in **Figure 1**, the pipeline is organised into three principal stages:

1. **Data Preprocessing & Partitioning** — raw transcriptomic data ingestion, quality control, normalisation, and batch correction.
2. **Multi-Stage, Stability-Enforced Gene Selection Cascade** — iterative feature selection using statistical and information-theoretic methods to identify robust biomarker candidates.
3. **Model Training, Evaluation, Interpretability & External Validation** — machine learning classifiers trained and evaluated with SHAP-based interpretability, followed by validation on an independent RNA-seq cohort.

---

## Data Acquisition

All datasets used in this pipeline are publicly available on the **NCBI Gene Expression Omnibus (GEO)**: https://www.ncbi.nlm.nih.gov/geo/

### Datasets to Download

#### GSE112676 & GSE112680 (Microarray — Training Cohort)

Navigate to each accession on GEO and download the following files:

| Accession  | Files Required |
|------------|----------------|
| GSE112676  | Series Matrix file (`GSE112676_series_matrix.txt.gz`) |
| GSE112680  | Series Matrix file (`GSE112680_series_matrix.txt.gz`) |

In addition, download the two **platform annotation files** (GPL) required for probe-to-gene mapping:

| Platform | Description |
|----------|-------------|
| GPL6947  | Illumina HumanHT-12 V3.0 expression beadchip |
| GPL10558 | Illumina HumanHT-12 V4.0 expression beadchip |

To download a GPL annotation file, go to:
`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL6947` (replace with `GPL10558` for the second one), then click **Download full table**.

---

#### GSE234297 (RNA-seq — External Validation Cohort)

Navigate to the GSE234297 accession on GEO and download the following files:

| File | Description |
|------|-------------|
| `GSE234297_series_matrix.txt.gz` | Series Matrix file |
| `GSE234297_gene_offset_matrix.txt.gz` | Raw gene count offset matrix |

---

### Quick Download (Google Drive Mirror)

If you prefer to skip manual GEO navigation, all required files are mirrored and ready to download from the following Google Drive folder:

> 📁 **[Download all data files here](https://drive.google.com/drive/folders/1DQm6ggMkk_LRjj1HHx3EkWKxGKFe3hPM?usp=drive_link)**

After downloading, place all files into the `data/` directory at the root of this repository before running any script.

---

## Environment Requirements

### R (version 4.5.1)

Install R 4.5.1 from the official CRAN repository: https://cran.r-project.org/

Required R packages and versions:

| Package | Version |
|---------|---------|
| limma   | 3.66.0  |
| SVA     | 3.58.0  |
| edgeR   | 4.8.1   |
| mRMRe   | 2.1.2.2 |
| MXM     | 1.5.5   |

Install all required packages by running the following in your R console:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("limma", "sva", "edgeR"))
install.packages(c("mRMRe", "MXM"))
```

---

### Python (version 3.11.13)

Install Python 3.11.13 from: https://www.python.org/downloads/

Required Python libraries and versions:

| Library      | Version |
|--------------|---------|
| scikit-learn | 1.7.2   |
| XGBoost      | 3.0.5   |
| SHAP         | 0.48.0  |

Install all required libraries via pip:

```bash
pip install scikit-learn==1.7.2 xgboost==3.0.5 shap==0.48.0
```

Or using a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Execute the scripts **in order** as described below. Each stage depends on the outputs of the previous one.

> **Note:**
> - **Python scripts** (`.py`) are run with the `python` command.
> - **R scripts** (`.R`) must be run using `Rscript` from the terminal — do **not** run them interactively in the R console unless specified.

---

### Stage 1 — Data Preprocessing & Partitioning

#### Step 1.1 — Batch Correction and Normalisation (R)

```bash
Rscript 01_preprocessing.R
```

#### Step 1.2 — Data Partitioning (R)

```bash
Rscript 02_partitioning.R
```

---

### Stage 2 — Gene Selection Cascade

#### Step 2.1 — Differential Expression Analysis with limma/edgeR (R)

```bash
Rscript 03_differential_expression.R
```

#### Step 2.2 — mRMR Feature Selection (R)

```bash
Rscript 04_mrmr_selection.R
```

#### Step 2.3 — MXM Stability-Based Selection (R)

```bash
Rscript 05_mxm_selection.R
```

---

### Stage 3 — Model Training, Evaluation & Validation

#### Step 3.1 — Model Training with XGBoost (Python)

```bash
python 06_train_model.py
```

#### Step 3.2 — Model Evaluation (Python)

```bash
python 07_evaluate_model.py
```

#### Step 3.3 — SHAP Interpretability Analysis (Python)

```bash
python 08_shap_analysis.py
```

#### Step 3.4 — External Validation on Independent RNA-seq Cohort (Python)

```bash
python 09_external_validation.py
```

---

## Recommended Execution Environment

| Component | Specification         |
|-----------|-----------------------|
| OS        | Ubuntu 22.04 / macOS  |
| R         | 4.5.1                 |
| Python    | 3.11.13               |
| RAM       | >= 16 GB recommended  |

---

## Notes

- Ensure all input data files are placed in the `data/` directory before running any script.
- Intermediate outputs from each stage will be saved to the `results/` directory automatically.
- All scripts must be run from the **root directory** of this repository.
- If you encounter package conflicts, consider using a Python virtual environment (`venv` or `conda`) and installing packages in isolation.

```bash
# Example: create and activate a virtual environment
python -m venv als_env
source als_env/bin/activate   # On Windows: als_env\Scripts\activate
pip install -r requirements.txt
```
