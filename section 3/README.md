# 🎫 Support Ticket Classifier

A fine-tuned **BERT-based text classification** pipeline that categorises customer support tickets into four predefined classes using the Hugging Face `transformers` library.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Categories](#categories)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [1. Train](#1-train)
  - [2. Evaluate](#2-evaluate)
  - [3. Predict / Inference](#3-predict--inference)
- [Model & Training Details](#model--training-details)
- [Output Artefacts](#output-artefacts)
- [Design Decisions](#design-decisions)

---

## Overview

This project fine-tunes [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) on a stratified sample of **10,000 support tickets** drawn from a larger CSV dataset. The full pipeline covers:

1. **Stratified sampling** — every category is guaranteed representation.
2. **Label encoding** — integer labels persisted alongside the model for reproducible inference.
3. **80 / 10 / 10 split** — train / validation / test, stratified on class.
4. **Fine-tuning** with early stopping (patience = 2 epochs).
5. **Standalone evaluation** — reload the saved model and re-score any time.
6. **Single-ticket inference** — classify raw text from the command line.

---

## Categories

| Label | Description |
|---|---|
| `complaint` | General dissatisfaction or grievance |
| `technical_issue` | App bugs, crashes, sync failures |
| `billing` | Charges, refunds, subscription problems |
| `feature_request` | Requests for new functionality |

---

## Project Structure

```
section 3/
├── support_tickets.csv        # Raw dataset (not tracked in git)
├── train.py                   # Fine-tune BERT; saves best model + test set
├── eval_model.py              # Reload saved model; evaluate on held-out test set
├── predict.py                 # Single-ticket inference from CLI
├── requirements.txt           # Pinned Python dependencies
├── Data Creation.ipynb        # Notebook used to generate/explore the dataset
├── .gitignore
└── results/                   # Auto-created by train.py
    ├── best_model/            # Saved tokenizer + model weights + label_map.json
    ├── test_set.csv           # Held-out test split (for eval_model.py)
    ├── confusion_matrix.png   # Confusion matrix from training run
    └── confusion_matrix_eval.png  # Confusion matrix from eval_model.py
```

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

# Install pinned requirements
pip install -r requirements.txt
```

---

## Usage

### 1. Train

```bash
python train.py
```

What it does:
- Loads `support_tickets.csv` and draws a stratified 10,000-row sample.
- Encodes labels and saves `results/best_model/label_map.json`.
- Splits data 80 / 10 / 10 and saves `results/test_set.csv`.
- Fine-tunes `bert-base-uncased` for up to **5 epochs** with early stopping.
- Saves the best checkpoint to `results/best_model/`.
- Prints accuracy & weighted F1 on the test set and saves `results/confusion_matrix.png`.

### 2. Evaluate

Re-evaluate the saved model on the held-out test set at any time (no re-training):

```bash
python eval_model.py
```

Outputs:
- Accuracy & weighted F1 scores.
- Full per-class classification report.
- `results/confusion_matrix_eval.png`.

### 3. Predict / Inference

**Demo mode** — runs 5 built-in sample tickets:

```bash
python predict.py
```

**Custom ticket** — pass your own text:

```bash
python predict.py --text "I was charged twice for my subscription this month."
```

Example output:

```
============================================================
  Support Ticket Classifier — Inference
============================================================

Loading model from './results/best_model' …
Model loaded on device: cpu

[1] Ticket   : I was charged twice for my subscription this month.
    Prediction: BILLING
    Confidence: {'billing': 0.9412, 'complaint': 0.0312, 'feature_request': 0.0142, 'technical_issue': 0.0134}
```

---

## Model & Training Details

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max sequence length | 128 tokens |
| Training sample size | 10,000 (stratified) |
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

| File | Description |
|---|---|
| `results/best_model/` | Full HuggingFace model checkpoint (weights + tokenizer config) |
| `results/best_model/label_map.json` | `label2id` / `id2label` mappings for inference |
| `results/test_set.csv` | Held-out test split with raw text and integer labels |
| `results/confusion_matrix.png` | Confusion matrix generated at end of `train.py` |
| `results/confusion_matrix_eval.png` | Confusion matrix generated by `eval_model.py` |

---

## Design Decisions

- **Stratified sampling** — a proportional floor of at least 50 samples per class is guaranteed before topping up to 10,000, preventing any category from being dropped.
- **Absolute paths in `train.py`** — uses `os.path.abspath(__file__)` as the base so the script runs correctly regardless of the working directory.
- **`eval_model.py` naming** — the evaluation script is deliberately named `eval_model.py` (not `evaluate.py`) to avoid a Python import collision with the Hugging Face `evaluate` package.
- **`label_map.json`** — persisted with the model so `predict.py` and `eval_model.py` never need the original CSV or `LabelEncoder` object at inference time.
- **Early stopping** — patience of 2 epochs prevents overfitting on the relatively small fine-tuning set.
- **`fp16` auto-detection** — mixed precision is enabled only when a CUDA GPU is present, keeping the script safe for CPU-only environments.
