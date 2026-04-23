"""
train.py — Fine-tune bert-base-uncased on 2,000 support ticket samples.

Usage:
    python train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

# NOTE: import `evaluate` AFTER all local imports.
# We renamed our local evaluate.py → eval_model.py to avoid this name collision.
import evaluate as hf_evaluate

# ─────────────────────────────────────────────
# Config — use absolute paths to avoid resolution issues
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CSV_PATH       = os.path.join(BASE_DIR, "AI_tickets.csv")
MODEL_NAME     = "bert-base-uncased"
OUTPUT_DIR     = os.path.join(BASE_DIR, "results")
BEST_MODEL_DIR = os.path.join(BASE_DIR, "results", "best_model")
SAMPLE_SIZE    = 2000
SEED           = 42
MAX_LENGTH     = 128
BATCH_SIZE     = 16
NUM_EPOCHS     = 5
LEARNING_RATE  = 2e-5

# Create output dirs upfront
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load & Sample
# ─────────────────────────────────────────────
print("=" * 60)
print("  Support Ticket Classifier — Training")
print("=" * 60)

print(f"\n[1/6] Loading {CSV_PATH} …")
df = pd.read_csv(CSV_PATH)
print(f"      Total rows: {len(df):,}  |  Columns: {list(df.columns)}")

all_categories = df["category"].unique().tolist()
print(f"      All categories in dataset ({len(all_categories)}): {all_categories}")

# ── Stratified sample: guarantees every category is represented ──
# Step 1: per-class floor of MIN_PER_CLASS samples (or class size if smaller)
MIN_PER_CLASS = 50
n_classes     = df["category"].nunique()

stratified_parts = []
for cat, group in df.groupby("category"):
    n_take = max(MIN_PER_CLASS, int(SAMPLE_SIZE * len(group) / len(df)))
    n_take = min(n_take, len(group))   # can't exceed class size
    stratified_parts.append(group.sample(n=n_take, random_state=SEED))

df_sample = pd.concat(stratified_parts)

# Step 2: if we're under SAMPLE_SIZE, top-up from the remaining rows
if len(df_sample) < SAMPLE_SIZE:
    already_selected = df_sample.index
    remaining = df.drop(index=already_selected)
    top_up = remaining.sample(n=SAMPLE_SIZE - len(df_sample), random_state=SEED)
    df_sample = pd.concat([df_sample, top_up])

# Step 3: if over SAMPLE_SIZE, trim back
df_sample = (
    df_sample
    .sample(n=SAMPLE_SIZE, random_state=SEED)
    .reset_index(drop=True)
)

print(f"      Sampled : {len(df_sample):,} rows (stratified — all categories guaranteed)")
print(f"\n      Class distribution:\n{df_sample['category'].value_counts().to_string()}")


# ─────────────────────────────────────────────
# 2. Encode Labels
# ─────────────────────────────────────────────
print("\n[2/6] Encoding labels …")
le = LabelEncoder()
df_sample["label"] = le.fit_transform(df_sample["category"])

label2id   = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
id2label   = {int(idx): cls for cls, idx in label2id.items()}
num_labels = len(le.classes_)

print(f"      Classes ({num_labels}): {label2id}")

# Persist label mapping for inference scripts
with open(os.path.join(BEST_MODEL_DIR, "label_map.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)


# ─────────────────────────────────────────────
# 3. Train / Val / Test Split  (80 / 10 / 10)
# ─────────────────────────────────────────────
print("\n[3/6] Splitting dataset (80/10/10) …")
X = df_sample["issue_description"].tolist()
y = df_sample["label"].tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print(f"      Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

# Save test set for eval_model.py
test_df = pd.DataFrame({"text": X_test, "label": y_test})
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_set.csv"), index=False)
print(f"      Test set saved → {os.path.join(OUTPUT_DIR, 'test_set.csv')}")


# ─────────────────────────────────────────────
# 4. Tokenise
# ─────────────────────────────────────────────
print(f"\n[4/6] Loading tokenizer: {MODEL_NAME} …")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(texts, labels):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    return Dataset.from_dict({**encodings, "labels": labels})

train_dataset = tokenize(X_train, y_train)
val_dataset   = tokenize(X_val,   y_val)
test_dataset  = tokenize(X_test,  y_test)
print("      Tokenisation complete.")


# ─────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────
print(f"\n[5/6] Loading model: {MODEL_NAME} …")
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)


# ─────────────────────────────────────────────
# 6. Train
# ─────────────────────────────────────────────
print(f"\n[6/6] Starting training  (epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}) …")
print("-" * 60)

accuracy_metric = hf_evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=20,
    report_to="none",           # disable wandb / other loggers
    seed=SEED,
    fp16=torch.cuda.is_available(),   # AMP on GPU, disabled on CPU
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# ─────────────────────────────────────────────
# Save best model
# ─────────────────────────────────────────────
print(f"\n✅ Training complete. Saving best model to '{BEST_MODEL_DIR}' …")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)


# ─────────────────────────────────────────────
# Evaluate on Test Set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Test Set Evaluation")
print("=" * 60)

test_preds_output = trainer.predict(test_dataset)
test_logits = test_preds_output.predictions
test_preds  = np.argmax(test_logits, axis=-1)

acc = accuracy_score(y_test, test_preds)
f1  = f1_score(y_test, test_preds, average="weighted")

print(f"\n  Accuracy  : {acc:.4f}")
print(f"  F1 (wtd)  : {f1:.4f}")
print("\n  Classification Report:\n")
print(classification_report(y_test, test_preds, target_names=le.classes_))


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────
cm          = confusion_matrix(y_test, test_preds)
class_names = le.classes_

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(num_labels),
    yticks=np.arange(num_labels),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel="True Label",
    xlabel="Predicted Label",
    title="Confusion Matrix — Test Set",
)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

thresh = cm.max() / 2.0
for i in range(num_labels):
    for j in range(num_labels):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11,
        )

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
print(f"\n  Confusion matrix saved → {cm_path}")
print("=" * 60)
print("\n🎉 Done! Run 'python eval_model.py' to re-evaluate, or 'python predict.py' to test inference.")
