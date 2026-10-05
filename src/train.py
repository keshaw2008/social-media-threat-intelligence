import os
import sys
import time
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    set_seed,
    get_device,
    ensure_directories,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LABEL_MAP,
    INVERSE_LABEL_MAP
)

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EPOCHS = 2
SEED = 42
LOGGING_STEPS = 250


class SocialMediaDataset(Dataset):
    """
    PyTorch Dataset for Social Media Threat Classification.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, dataloader, device):
    """
    Computes validation loss, accuracy, and macro F1 score.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


def train():
    set_seed(SEED)
    ensure_directories()
    device = get_device()

    train_path = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DATA_DIR, "validation.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            "Train/Validation splits not found. Please run 'python src/preprocess.py' first."
        )

    print(f"\n[INFO] Loading training data from: {train_path}", flush=True)
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"[INFO] Train samples: {len(train_df)} | Validation samples: {len(val_df)}", flush=True)

    # Initialize Tokenizer and Datasets
    print(f"[INFO] Initializing tokenizer: {MODEL_NAME}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = SocialMediaDataset(train_df["text"], train_df["label"], tokenizer, MAX_LENGTH)
    val_dataset = SocialMediaDataset(val_df["text"], val_df["label"], tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Initialize BERT Model for Binary Sequence Classification
    print(f"[INFO] Loading BERT Model: {MODEL_NAME} with num_labels=2", flush=True)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=LABEL_MAP,
        label2id=INVERSE_LABEL_MAP
    )
    model.to(device)

    # Optimizer and Learning Rate Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    total_update_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(total_update_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps
    )

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # Training Metrics History
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_f1 = -1.0
    print("\n" + "=" * 70, flush=True)
    print(f"               STARTING BERT MODEL FINE-TUNING", flush=True)
    print(f" Model:                       {MODEL_NAME}", flush=True)
    print(f" Epochs:                      {EPOCHS}", flush=True)
    print(f" Train Batch Size:            {TRAIN_BATCH_SIZE}", flush=True)
    print(f" Eval Batch Size:             {EVAL_BATCH_SIZE}", flush=True)
    print(f" Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}", flush=True)
    print(f" Effective Batch Size:        {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}", flush=True)
    print(f" Max Sequence Length:         {MAX_LENGTH}", flush=True)
    print(f" Learning Rate:               {LEARNING_RATE}", flush=True)
    print(f" Mixed Precision (FP16):      {use_amp}", flush=True)
    print(f" Device:                      {device.type.upper()}", flush=True)
    print("=" * 70 + "\n", flush=True)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        train_preds = []
        train_targets = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()

                if step % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if step % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.detach().cpu().numpy())

            if step % LOGGING_STEPS == 0 or step == len(train_loader):
                batch_loss = running_loss / step
                print(f" Epoch [{epoch}/{EPOCHS}] | Step [{step:5d}/{len(train_loader):5d}] | Current Avg Loss: {batch_loss:.4f}", flush=True)

        # Compute Epoch Metrics
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = accuracy_score(train_targets, train_preds)
        val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        epoch_time = time.time() - epoch_start
        print("-" * 70, flush=True)
        print(f" EPOCH {epoch} SUMMARY (Duration: {epoch_time:.1f}s):", flush=True)
        print(f"   Train Loss:     {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}%", flush=True)
        print(f"   Val Loss:       {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%", flush=True)
        print("-" * 70, flush=True)

        # Save Epoch Checkpoint
        epoch_ckpt_dir = os.path.join(MODELS_DIR, f"checkpoint-epoch-{epoch}")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        model.save_pretrained(epoch_ckpt_dir)
        tokenizer.save_pretrained(epoch_ckpt_dir)
        print(f" [CHECKPOINT] Saved epoch {epoch} checkpoint to {epoch_ckpt_dir}", flush=True)

        # Checkpoint Best Model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f" [BEST MODEL] Improved Val F1 to {val_f1*100:.2f}%. Saving best model to {MODELS_DIR}...", flush=True)
            model.save_pretrained(MODELS_DIR)
            tokenizer.save_pretrained(MODELS_DIR)

            # Save metadata mapping file
            metadata = {
                "model_name": MODEL_NAME,
                "epochs": EPOCHS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "max_length": MAX_LENGTH,
                "learning_rate": LEARNING_RATE,
                "best_val_f1": best_val_f1,
                "id2label": LABEL_MAP,
                "label2id": INVERSE_LABEL_MAP
            }
            with open(os.path.join(MODELS_DIR, "training_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

    total_training_time = time.time() - start_time
    print("\n" + "=" * 70, flush=True)
    print(f"[SUCCESS] Training completed in {total_training_time/60:.2f} minutes!", flush=True)
    print(f"Best Validation F1: {best_val_f1*100:.2f}%", flush=True)
    print(f"Best model and tokenizer saved to: {MODELS_DIR}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Generate and Save Actual Training History Plots
    epochs_range = list(range(1, EPOCHS + 1))

    # 1. Training Loss Plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history["train_loss"], marker="o", color="#d9534f", linewidth=2, label="Train Loss")
    plt.title("BERT Fine-Tuning: Training Loss per Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    train_loss_path = os.path.join(RESULTS_DIR, "training_loss.png")
    plt.savefig(train_loss_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved training loss plot to: {train_loss_path}", flush=True)

    # 2. Validation Loss Plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, history["val_loss"], marker="s", color="#337ab7", linewidth=2, label="Validation Loss")
    plt.title("BERT Fine-Tuning: Validation Loss per Epoch", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    val_loss_path = os.path.join(RESULTS_DIR, "validation_loss.png")
    plt.savefig(val_loss_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved validation loss plot to: {val_loss_path}", flush=True)

    # 3. Training & Validation Accuracy Plot
    plt.figure(figsize=(7, 5))
    plt.plot(epochs_range, [a * 100 for a in history["train_acc"]], marker="o", color="#5cb85c", linewidth=2, label="Train Accuracy (%)")
    plt.plot(epochs_range, [a * 100 for a in history["val_acc"]], marker="^", color="#f0ad4e", linewidth=2, label="Validation Accuracy (%)")
    plt.title("BERT Fine-Tuning: Accuracy Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.xticks(epochs_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    train_acc_path = os.path.join(RESULTS_DIR, "training_accuracy.png")
    plt.savefig(train_acc_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved training accuracy plot to: {train_acc_path}", flush=True)


if __name__ == "__main__":
    train()
