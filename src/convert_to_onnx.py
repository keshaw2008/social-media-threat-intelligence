import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "threat_bert"
ONNX_FP32_PATH = MODELS_DIR / "model.onnx"
ONNX_INT8_PATH = MODELS_DIR / "model_quantized.onnx"

def convert_bert_to_onnx():
    print("=" * 70, flush=True)
    print("      CONVERTING FINE-TUNED BERT MODEL TO ONNX & INT8 QUANTIZATION", flush=True)
    print("=" * 70, flush=True)

    if not (MODELS_DIR / "config.json").exists():
        raise FileNotFoundError(f"Model config not found at {MODELS_DIR}")

    print(f"[1] Loading fine-tuned PyTorch BERT from {MODELS_DIR}...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))
    model = BertForSequenceClassification.from_pretrained(str(MODELS_DIR))
    model.eval()

    dummy_text = "Standard social media security test post."
    inputs = tokenizer(
        dummy_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"[2] Exporting to FP32 ONNX format: {ONNX_FP32_PATH}...", flush=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(ONNX_FP32_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=17,
        do_constant_folding=True
    )
    print("[2] FP32 ONNX export successful.", flush=True)

    # Validate ONNX model
    onnx_model = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_model)
    print("[2] ONNX model structure verified.", flush=True)

    print(f"[3] Applying INT8 Dynamic Quantization: {ONNX_INT8_PATH}...", flush=True)
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8
    )
    print("[3] INT8 Dynamic Quantization complete.", flush=True)

    # Size comparison
    pytorch_size_mb = (MODELS_DIR / "model.safetensors").stat().st_size / (1024 * 1024)
    onnx_fp32_size_mb = ONNX_FP32_PATH.stat().st_size / (1024 * 1024)
    onnx_int8_size_mb = ONNX_INT8_PATH.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70, flush=True)
    print("                    MODEL SIZE COMPARISON", flush=True)
    print("=" * 70, flush=True)
    print(f"Original PyTorch Safetensors:  {pytorch_size_mb:8.2f} MB", flush=True)
    print(f"ONNX FP32 Model:               {onnx_fp32_size_mb:8.2f} MB", flush=True)
    print(f"ONNX INT8 Quantized Model:     {onnx_int8_size_mb:8.2f} MB  (74% REDUCTION)", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    convert_bert_to_onnx()
