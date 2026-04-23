"""
predict.py — Batch-predict test_data.csv using the fine-tuned BERT model.

Outputs
-------
  results/predictions_test.xlsx   — Excel with actual vs predicted + confidence cols
  results/report_test.png         — Classification report + confusion matrix image

Usage:
    python predict.py
    python predict.py --input path/to/custom.csv --output path/to/out.xlsx
"""

import os
import json
import argparse
import textwrap

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from transformers import BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
BEST_MODEL_DIR = os.path.join(BASE_DIR, "results", "best_model")
DEFAULT_INPUT  = os.path.join(BASE_DIR, "test_data.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "results")
DEFAULT_EXCEL  = os.path.join(OUTPUT_DIR, "predictions_test.xlsx")
DEFAULT_REPORT = os.path.join(OUTPUT_DIR, "report_test.png")
MAX_LENGTH     = 128
BATCH_SIZE     = 32
TEXT_COL       = "issue_description"
LABEL_COL      = "category"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_model(model_dir: str):
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model     = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with open(os.path.join(model_dir, "label_map.json")) as f:
        label_map = json.load(f)
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    label2id = {v: k for k, v in id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, id2label, label2id, device


def batch_predict(texts, tokenizer, model, id2label, device):
    """Returns (predicted_labels, confidence_dicts) lists."""
    all_labels, all_confs = [], []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        for pred_id, prob_row in zip(preds, probs):
            all_labels.append(id2label[int(pred_id)])
            all_confs.append({id2label[i]: round(float(p), 4) for i, p in enumerate(prob_row)})
    return all_labels, all_confs


def save_excel(df_result, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_result.to_excel(writer, index=False, sheet_name="Predictions")
        ws = writer.sheets["Predictions"]
        # auto-fit column widths
        for col_cells in ws.columns:
            max_len = max(len(str(c.value)) if c.value else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 60)
    print(f"  [OK] Excel saved  -> {path}")


def save_report(y_true, y_pred, class_names, report_path):
    """Save a PNG report: classification table + confusion matrix."""
    cm     = confusion_matrix(y_true, y_pred, labels=class_names)
    report = classification_report(y_true, y_pred, labels=class_names, output_dict=True)

    acc    = accuracy_score(y_true, y_pred)
    f1_w   = f1_score(y_true, y_pred, average="weighted", labels=class_names)

    # ── build figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), facecolor="#0f172a")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                             left=0.05, right=0.97, top=0.88, bottom=0.06,
                             hspace=0.45, wspace=0.35)

    title_kw = dict(color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    txt_kw   = dict(color="#94a3b8", fontsize=10)

    # ── [0,0] Per-class metrics table ───────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, 0])
    ax_tbl.axis("off")
    ax_tbl.set_title("Per-Class Metrics", **title_kw)

    rows      = [[cls,
               f"{report[cls]['precision']:.3f}",
               f"{report[cls]['recall']:.3f}",
               f"{report[cls]['f1-score']:.3f}",
               str(int(report[cls]['support']))]
              for cls in class_names]
    cols      = ["Class", "Precision", "Recall", "F1", "Support"]

    tbl = ax_tbl.table(cellText=rows, colLabels=cols,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)
    n_rows = len(rows)
    n_cols = len(cols)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#334155")   # header
        else:
            cell.set_facecolor("#1e293b")   # data rows
        cell.set_edgecolor("#475569")
        cell.set_text_props(color="#e2e8f0")

    # ── [0,1] Summary metrics bar chart ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor("#1e293b")
    ax_bar.set_title("Overall Metrics", **title_kw)

    metrics  = ["Accuracy", "F1 (weighted)"]
    values   = [acc, f1_w]
    bar_cols = ["#6366f1", "#06b6d4"]
    bars     = ax_bar.barh(metrics, values, color=bar_cols, height=0.4)
    ax_bar.set_xlim(0, 1.15)
    ax_bar.tick_params(colors="#94a3b8")
    ax_bar.set_xlabel("Score", color="#94a3b8", fontsize=10)
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#334155")
    for bar, val in zip(bars, values):
        ax_bar.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color="#e2e8f0", fontsize=11, fontweight="bold")

    # ── [1, :] Confusion matrix (spans both columns) ────────────────────
    ax_cm = fig.add_subplot(gs[1, :])
    ax_cm.set_facecolor("#1e293b")
    ax_cm.set_title("Confusion Matrix", **title_kw)

    cmap = LinearSegmentedColormap.from_list("indigo", ["#1e293b", "#6366f1"])
    im   = ax_cm.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, ax=ax_cm, fraction=0.03, pad=0.02).ax.tick_params(colors="#94a3b8")

    n = len(class_names)
    ax_cm.set_xticks(np.arange(n));  ax_cm.set_xticklabels(class_names, color="#94a3b8", fontsize=9)
    ax_cm.set_yticks(np.arange(n));  ax_cm.set_yticklabels(class_names, color="#94a3b8", fontsize=9)
    plt.setp(ax_cm.get_xticklabels(), rotation=25, ha="right")
    ax_cm.set_xlabel("Predicted Label", color="#94a3b8", fontsize=10)
    ax_cm.set_ylabel("True Label",      color="#94a3b8", fontsize=10)
    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#334155")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax_cm.text(j, i, str(cm[i, j]),
                       ha="center", va="center", fontsize=11,
                       color="white" if cm[i, j] > thresh else "#94a3b8")

    # ── super-title ──────────────────────────────────────────────────────
    fig.suptitle("Support Ticket Classifier — Prediction Report  (test_data.csv)",
                 color="#f1f5f9", fontsize=15, fontweight="bold", y=0.96)

    plt.savefig(report_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] Report saved -> {report_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Batch-predict support tickets from a CSV file.")
    parser.add_argument("--input",  type=str, default=DEFAULT_INPUT,  help="Path to input CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_EXCEL,  help="Path to output Excel file")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT, help="Path to output report PNG")
    args = parser.parse_args()

    print("=" * 60)
    print("  Support Ticket Classifier — Batch Prediction")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading input CSV: {args.input}")
    df = pd.read_csv(args.input)
    assert TEXT_COL  in df.columns, f"Missing column '{TEXT_COL}'  in {args.input}"
    assert LABEL_COL in df.columns, f"Missing column '{LABEL_COL}' in {args.input}"
    print(f"      {len(df)} rows loaded.")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n[2/4] Loading model from '{BEST_MODEL_DIR}' ...")
    tokenizer, model, id2label, label2id, device = load_model(BEST_MODEL_DIR)
    print(f"      Device: {device}  |  Classes: {list(id2label.values())}")

    # ── Predict ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Running batch inference (batch_size={BATCH_SIZE}) ...")
    texts = df[TEXT_COL].fillna("").tolist()
    predicted_labels, confidence_dicts = batch_predict(texts, tokenizer, model, id2label, device)
    print(f"      Done - {len(predicted_labels)} predictions.")

    # ── Build result DataFrame ───────────────────────────────────────────
    df_result = df[[TEXT_COL, LABEL_COL]].copy()
    df_result.rename(columns={LABEL_COL: "actual_category"}, inplace=True)
    df_result["predicted_category"] = predicted_labels
    df_result["correct"]            = df_result["actual_category"] == df_result["predicted_category"]

    # Add per-class confidence columns
    class_names = [id2label[i] for i in range(len(id2label))]
    for cls in class_names:
        df_result[f"conf_{cls}"] = [c.get(cls, 0.0) for c in confidence_dicts]

    # ── Save Excel ───────────────────────────────────────────────────────
    print(f"\n[4/4] Saving outputs ...")
    save_excel(df_result, args.output)

    # ── Print summary ────────────────────────────────────────────────────
    y_true = df_result["actual_category"].tolist()
    y_pred = df_result["predicted_category"].tolist()
    acc    = accuracy_score(y_true, y_pred)
    f1_w   = f1_score(y_true, y_pred, average="weighted", labels=class_names)

    print(f"\n{'='*60}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true, y_pred, labels=class_names))

    # ── Save report ──────────────────────────────────────────────────────
    save_report(y_true, y_pred, class_names, args.report)
    print("=" * 60)
    print("\nDone!")


if __name__ == "__main__":
    main()
