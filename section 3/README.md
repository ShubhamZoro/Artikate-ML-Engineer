# 🎫 Support Ticket Classifier

A fine-tuned **BERT-based text classification** pipeline that categorises customer support tickets into five predefined classes using the Hugging Face `transformers` library.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Categories](#categories)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Train](#1-train)
  - [2. Predict on Test Data](#2-predict-on-test-data)
  - [3. Evaluate on Eval Data](#3-evaluate-on-eval-data)
- [Model & Training Details](#model--training-details)
- [Output Artefacts](#output-artefacts)
- [Results](#results)
- [Design Decisions](#design-decisions)

---

## Overview

This project fine-tunes [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) on a stratified sample of **2,000 support tickets** drawn from `AI_tickets.csv`. The full pipeline covers:

1. **Stratified sampling** — every category is guaranteed representation.
2. **Label encoding** — integer labels persisted alongside the model for reproducible inference.
3. **80 / 10 / 10 split** — train / validation / test, stratified on class.
4. **Fine-tuning** with early stopping (patience = 2 epochs).
5. **Batch prediction** on `test_data.csv` — saves Excel + PNG report.
6. **Batch evaluation** on `eval_data.csv` — saves 2-sheet Excel + PNG report.

---

## Categories

| Label | Description |
|---|---|
| `billing` | Charges, invoices, refunds, subscription problems |
| `complaint` | General dissatisfaction or grievance |
| `feature_request` | Requests for new functionality |
| `other` | Tickets that don't fit the four main categories |
| `technical` | App bugs, crashes, sync failures |

---

## Project Structure

```
section 3/
├── AI_tickets.csv             # Training dataset
├── test_data.csv              # Test set for predict.py
├── eval_data.csv              # Evaluation set for eval_model.py
├── train.py                   # Fine-tune BERT; saves best model + test split
├── predict.py                 # Batch-predict test_data.csv → Excel + PNG report
├── eval_model.py              # Batch-evaluate eval_data.csv → Excel + PNG report
├── requirements.txt           # Pinned Python dependencies
├── Data Creation.ipynb        # Notebook used to generate/explore the dataset
├── .gitignore
└── results/                   # Auto-created by train.py
    ├── best_model/            # ⛔ git-ignored  (model weights + tokenizer + label_map.json)
    ├── test_set.csv           # ⛔ git-ignored  (auto-generated split)
    ├── predictions_test.xlsx  # ✅ tracked      (actual vs predicted for test_data.csv)
    ├── predictions_eval.xlsx  # ✅ tracked      (actual vs predicted for eval_data.csv)
    ├── report_test.png        # ✅ tracked      (metrics report for test_data.csv)
    ├── report_eval.png        # ✅ tracked      (metrics report for eval_data.csv)
    ├── confusion_matrix.png   # ✅ tracked      (from training run)
    └── confusion_matrix_eval.png  # ✅ tracked  (from eval_model.py)
```

> **Git tracking rules:** Model weights are excluded (too large), but all CSV inputs, PNG reports, and Excel spreadsheets are committed so results are fully reproducible without re-training.

---

## Setup

### Prerequisites

- Python **3.9 – 3.11**
- (Optional but recommended) a CUDA-capable GPU — the scripts auto-detect GPU and enable **mixed-precision (fp16)** training when available.

### Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venva
venva\Scripts\activate        # Windows
# source venva/bin/activate   # macOS / Linux

pip install -r requirements.txt
pip install openpyxl           # required for Excel output
```

---

## Usage

### 1. Train

```bash
python train.py
```

What it does:
- Loads `AI_tickets.csv` and draws a stratified 2,000-row sample.
- Encodes labels and saves `results/best_model/label_map.json`.
- Splits data 80 / 10 / 10 and saves `results/test_set.csv`.
- Fine-tunes `bert-base-uncased` for up to **5 epochs** with early stopping.
- Saves the best checkpoint to `results/best_model/`.
- Prints accuracy & weighted F1 on the test set and saves `results/confusion_matrix.png`.

---

### 2. Predict on Test Data

Batch-predict `test_data.csv` and generate a full report:

```bash
python predict.py
```

Custom file / paths:

```bash
python predict.py --input path/to/custom.csv --output path/to/out.xlsx --report path/to/report.png
```

**Outputs:**

| File | Contents |
|---|---|
| `results/predictions_test.xlsx` | One row per ticket: `issue_description`, `actual_category`, `predicted_category`, `correct`, per-class confidence scores |
| `results/report_test.png` | Dark-themed PNG: per-class metrics table + accuracy/F1 bar chart + confusion matrix |

**Example terminal output:**

```
============================================================
  Support Ticket Classifier — Batch Prediction
============================================================

[1/4] Loading input CSV: .../test_data.csv
      100 rows loaded.
[2/4] Loading model from '.../results/best_model' ...
      Device: cpu  |  Classes: ['billing', 'complaint', 'feature_request', 'other', 'technical']
[3/4] Running batch inference (batch_size=32) ...
      Done - 100 predictions.
[4/4] Saving outputs ...
  [OK] Excel saved  -> .../results/predictions_test.xlsx

============================================================
  Accuracy     : 0.9700
  F1 (weighted): 0.9706
  [OK] Report saved -> .../results/report_test.png
============================================================
```

---

### 3. Evaluate on Eval Data

Batch-evaluate `eval_data.csv` and generate a full report:

```bash
python eval_model.py
```

Custom file / paths:

```bash
python eval_model.py --input path/to/custom.csv --output path/to/out.xlsx --report path/to/report.png
```

**Outputs:**

| File | Contents |
|---|---|
| `results/predictions_eval.xlsx` | **Sheet 1 — Predictions:** one row per ticket with actual, predicted, correct flag, confidence scores |
| | **Sheet 2 — Summary:** per-class precision/recall/F1 + overall accuracy |
| `results/report_eval.png` | Same dark-themed PNG report (green colour scheme) |

---

## Model & Training Details

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max sequence length | 128 tokens |
| Training sample size | 2,000 (stratified) |
| Train / Val / Test split | 80% / 10% / 10% |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max epochs | 5 |
| Early stopping patience | 2 epochs |
| Best model metric | Validation accuracy |
| Mixed precision (fp16) | Auto (GPU only) |
| Optimizer | AdamW (HF default) |

---

## Output Artefacts

| File | Tracked in Git | Description |
|---|:---:|---|
| `results/best_model/` | ❌ | Full HuggingFace checkpoint (weights + tokenizer + label map) |
| `results/test_set.csv` | ❌ | Auto-generated train/test split |
| `results/predictions_test.xlsx` | ✅ | Actual vs predicted for `test_data.csv` |
| `results/predictions_eval.xlsx` | ✅ | Actual vs predicted for `eval_data.csv` (2 sheets) |
| `results/report_test.png` | ✅ | Visual report for `test_data.csv` |
| `results/report_eval.png` | ✅ | Visual report for `eval_data.csv` |
| `results/confusion_matrix.png` | ✅ | Confusion matrix from training run |
| `test_data.csv` | ✅ | Input test set (100 tickets) |
| `eval_data.csv` | ✅ | Input eval set (100 tickets) |
| `AI_tickets.csv` | ✅ | Full training dataset |

---

## Results

| Dataset | Accuracy | F1 (weighted) |
|---|:---:|:---:|
| `test_data.csv` (100 rows) | **97.00%** | **97.06%** |
| `eval_data.csv` (100 rows) | **98.00%** | **98.00%** |

### Per-class breakdown — `test_data.csv`

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| billing | 0.88 | 1.00 | 0.93 | 21 |
| complaint | 1.00 | 0.95 | 0.97 | 20 |
| feature_request | 1.00 | 1.00 | 1.00 | 20 |
| other | 1.00 | 0.95 | 0.97 | 19 |
| technical | 1.00 | 0.95 | 0.97 | 20 |

### Per-class breakdown — `eval_data.csv`

| Class | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| billing | 0.95 | 1.00 | 0.97 | 19 |
| complaint | 0.95 | 0.95 | 0.95 | 20 |
| feature_request | 1.00 | 1.00 | 1.00 | 20 |
| other | 1.00 | 1.00 | 1.00 | 21 |
| technical | 1.00 | 0.95 | 0.97 | 20 |

---

## Design Decisions

- **Stratified sampling** — a proportional floor of at least 50 samples per class is guaranteed before topping up to 2,000, preventing any category from being dropped.
- **Absolute paths in `train.py`** — uses `os.path.abspath(__file__)` as the base so the script runs correctly regardless of the working directory.
- **`eval_model.py` naming** — deliberately named `eval_model.py` (not `evaluate.py`) to avoid a Python import collision with the Hugging Face `evaluate` package.
- **`label_map.json`** — persisted with the model so `predict.py` and `eval_model.py` never need the original CSV or `LabelEncoder` at inference time.
- **Excel 2-sheet output in `eval_model.py`** — Sheet 1 contains row-level predictions; Sheet 2 contains an aggregated per-class summary for quick review.
- **PNG reports** — dark-themed matplotlib figures (indigo palette for test, emerald for eval) combining a metrics table, overall bar chart, and confusion matrix in a single image.
- **Git tracking** — model weights are git-ignored (large binaries), but all CSV data files and output PNG/XLSX artefacts are tracked so results are reproducible without re-running training.
- **Early stopping** — patience of 2 epochs prevents overfitting on the relatively small fine-tuning set.
- **`fp16` auto-detection** — mixed precision is enabled only when a CUDA GPU is present, keeping the script safe for CPU-only environments.
