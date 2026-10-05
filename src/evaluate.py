import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    set_seed,
    get_device,
    ensure_directories,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LABEL_MAP
)
from train import SocialMediaDataset

BATCH_SIZE = 16
MAX_LENGTH = 128


def evaluate_test_set():
    set_seed(42)
    ensure_directories()
    device = get_device()

    test_path = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test set not found at {test_path}. Please run 'python src/preprocess.py' first."
        )

    if not os.path.exists(os.path.join(MODELS_DIR, "config.json")):
        raise FileNotFoundError(
            f"Trained model not found in {MODELS_DIR}. Please run 'python src/train.py' first."
        )

    print(f"\n[INFO] Loading test dataset from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"[INFO] Unseen test samples count: {len(test_df)}")

    print(f"[INFO] Loading fine-tuned BERT model from: {MODELS_DIR}")
    tokenizer = BertTokenizer.from_pretrained(MODELS_DIR)
    model = BertForSequenceClassification.from_pretrained(MODELS_DIR)
    model.to(device)
    model.eval()

    test_dataset = SocialMediaDataset(test_df["text"], test_df["label"], tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_probs = []
    all_labels = []

    print("[INFO] Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    test_df["prediction"] = all_preds
    test_df["threat_probability"] = all_probs[:, 1]

    # Calculate Core Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision_threat = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall_threat = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1_threat = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    target_names = [LABEL_MAP[0], LABEL_MAP[1]]
    clf_report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    # Print Terminal Report
    print("\n" + "=" * 70)
    print("             FINAL BERT MODEL EVALUATION RESULTS (TEST SET)")
    print("=" * 70)
    print(f"Test Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Threat Precision:       {precision_threat:.4f} ({precision_threat*100:.2f}%)")
    print(f"Threat Recall:          {recall_threat:.4f} ({recall_threat*100:.2f}%)")
    print(f"Threat F1 Score:        {f1_threat:.4f} ({f1_threat*100:.2f}%)")
    print(f"Macro F1 Score:         {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Weighted F1 Score:      {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print("-" * 70)
    print("Classification Report:\n")
    print(clf_report)
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"                 Predicted {LABEL_MAP[0]:<7} Predicted {LABEL_MAP[1]}")
    print(f"Actual {LABEL_MAP[0]:<7} : {cm[0][0]:<17} {cm[0][1]}")
    print(f"Actual {LABEL_MAP[1]:<7} : {cm[1][0]:<17} {cm[1][1]}")
    print("=" * 70)

    # Outlier Breakdown Analysis
    print("\n" + "=" * 70)
    print("                   OUTLIER ROBUSTNESS ANALYSIS")
    print("=" * 70)
    regular_mask = (test_df["is_outlier"] == 0)
    outlier_mask = (test_df["is_outlier"] == 1)

    reg_labels = all_labels[regular_mask]
    reg_preds = all_preds[regular_mask]
    out_labels = all_labels[outlier_mask]
    out_preds = all_preds[outlier_mask]

    reg_acc = accuracy_score(reg_labels, reg_preds) if len(reg_labels) > 0 else 0.0
    reg_f1 = f1_score(reg_labels, reg_preds, average="macro", zero_division=0) if len(reg_labels) > 0 else 0.0

    out_acc = accuracy_score(out_labels, out_preds) if len(out_labels) > 0 else 0.0
    out_f1 = f1_score(out_labels, out_preds, average="macro", zero_division=0) if len(out_labels) > 0 else 0.0

    print(f"Regular Test Samples Count:  {len(reg_labels)}")
    print(f"  Regular Test Accuracy:     {reg_acc:.4f} ({reg_acc*100:.2f}%)")
    print(f"  Regular Test Macro F1:     {reg_f1:.4f} ({reg_f1*100:.2f}%)")
    print("-" * 70)
    print(f"Outlier Test Samples Count:  {len(out_labels)}")
    print(f"  Outlier Test Accuracy:     {out_acc:.4f} ({out_acc*100:.2f}%)")
    print(f"  Outlier Test Macro F1:     {out_f1:.4f} ({out_f1*100:.2f}%)")
    print("=" * 70 + "\n")

    # Save Classification Report to File
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("      SOCIAL MEDIA THREAT INTELLIGENCE - MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Threat Precision:  {precision_threat:.4f} ({precision_threat*100:.2f}%)\n")
        f.write(f"Threat Recall:     {recall_threat:.4f} ({recall_threat*100:.2f}%)\n")
        f.write(f"Threat F1 Score:   {f1_threat:.4f} ({f1_threat*100:.2f}%)\n")
        f.write(f"Macro F1 Score:    {macro_f1:.4f} ({macro_f1*100:.2f}%)\n")
        f.write(f"Weighted F1 Score: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(clf_report + "\n\n")
        f.write(f"Confusion Matrix:\nTN={cm[0][0]}, FP={cm[0][1]}\nFN={cm[1][0]}, TP={cm[1][1]}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"Regular Samples ({len(reg_labels)}): Accuracy={reg_acc*100:.2f}%, Macro F1={reg_f1*100:.2f}%\n")
        f.write(f"Outlier Samples ({len(out_labels)}): Accuracy={out_acc*100:.2f}%, Macro F1={out_f1*100:.2f}%\n")
        f.write("=" * 70 + "\n")

    print(f"[INFO] Saved classification report text to: {report_path}")

    # Plot and Save Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, fontsize=11)
    plt.yticks(tick_marks, target_names, fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13, fontweight="bold"
            )

    plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix plot to: {cm_path}")

    # Save structured evaluation metrics JSON for programmatic inspection
    metrics_summary = {
        "test_accuracy": accuracy,
        "threat_precision": precision_threat,
        "threat_recall": recall_threat,
        "threat_f1": f1_threat,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "regular_accuracy": reg_acc,
        "regular_macro_f1": reg_f1,
        "outlier_accuracy": out_acc,
        "outlier_macro_f1": out_f1
    }
    with open(os.path.join(RESULTS_DIR, "test_metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


if __name__ == "__main__":
    evaluate_test_set()
