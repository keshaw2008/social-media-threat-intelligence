import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "validation.csv"
MODELS_DIR = BASE_DIR / "models" / "threat_bert"
CALIB_PATH = MODELS_DIR / "calibration.json"

print(f"[INFO] Loading validation data from: {DATA_PATH}")
val_df = pd.read_csv(DATA_PATH)
print(f"[INFO] Validation samples: {len(val_df)}")

val_texts = val_df['text'].tolist()
val_labels = np.array(val_df['label'].tolist())

print(f"[INFO] Loading fine-tuned BERT model from: {MODELS_DIR}")
tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))
model = BertForSequenceClassification.from_pretrained(str(MODELS_DIR))
model.eval()

all_logits = []
batch_size = 32

with torch.no_grad():
    for i in range(0, len(val_texts), batch_size):
        batch = val_texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())

val_logits = np.concatenate(all_logits, axis=0)

logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
labels_tensor = torch.tensor(val_labels, dtype=torch.long)
criterion = nn.CrossEntropyLoss()

uncalibrated_nll = criterion(logits_tensor, labels_tensor).item()

# Compute logit difference: Delta z = |z_0 - z_1|
delta_z = np.abs(val_logits[:, 0] - val_logits[:, 1])
max_delta_z = float(np.max(delta_z))

# Conservative temperature target: Max validation confidence <= 95.0% (target_ratio = ln(19) = 2.944439)
target_ratio = np.log(0.95 / 0.05)
conservative_T = round(max_delta_z / target_ratio, 4)

# Compute validation metrics with conservative T
scaled_logits = val_logits / conservative_T
exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
val_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
val_max_conf = float(np.max(val_probs))
val_mean_conf = float(np.mean(np.max(val_probs, axis=-1)))

conservative_nll = criterion(logits_tensor / conservative_T, labels_tensor).item()

print(f"[INFO] Uncalibrated Validation NLL (T=1.0):       {uncalibrated_nll:.4f}")
print(f"[INFO] Conservative Temperature T:               {conservative_T:.4f}")
print(f"[INFO] Calibrated Validation NLL:                 {conservative_nll:.4f}")
print(f"[INFO] Validation Max Confidence:                 {val_max_conf*100:.2f}%")
print(f"[INFO] Validation Mean Confidence:                {val_mean_conf*100:.2f}%")

calib_data = {
    "method": "conservative_temperature_scaling",
    "temperature": conservative_T,
    "validation_samples": len(val_df),
    "max_validation_logit_diff": round(max_delta_z, 4),
    "target_max_confidence": 0.95,
    "validation_max_confidence": round(val_max_conf, 4),
    "validation_mean_confidence": round(val_mean_conf, 4),
    "uncalibrated_nll": round(uncalibrated_nll, 4),
    "calibrated_nll": round(conservative_nll, 4)
}

with open(CALIB_PATH, "w", encoding="utf-8") as f:
    json.dump(calib_data, f, indent=2)

print(f"[SUCCESS] Saved conservative calibration metadata to: {CALIB_PATH}")
