import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_device, MODELS_DIR, LABEL_MAP

MAX_LENGTH = 128

_engine = None  # "onnx" or "pytorch"
_model = None
_tokenizer = None
_device = None


def load_threat_model(model_dir: str = MODELS_DIR):
    """
    Loads fine-tuned BERT model into memory.
    Prioritizes ultra-lightweight ONNX INT8 quantized model for CPU speed and low RAM.
    """
    global _engine, _model, _tokenizer, _device
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer, _device

    onnx_int8_file = os.path.join(model_dir, "model_quantized.onnx")
    onnx_fp32_file = os.path.join(model_dir, "model.onnx")

    # 1. Try ONNX INT8 Quantized Model
    if os.path.exists(onnx_int8_file):
        try:
            import onnxruntime as ort
            from transformers import BertTokenizer
            print(f"[INFO] Loading ONNX INT8 Quantized BERT from: {onnx_int8_file}")
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            _model = ort.InferenceSession(onnx_int8_file, sess_options, providers=["CPUExecutionProvider"])
            _tokenizer = BertTokenizer.from_pretrained(model_dir)
            _engine = "onnx"
            _device = "CPU (ONNX INT8)"
            return _model, _tokenizer, _device
        except Exception as e:
            print(f"[WARNING] Failed to load ONNX INT8 model: {e}. Falling back...")

    # 2. Try ONNX FP32 Model
    if os.path.exists(onnx_fp32_file):
        try:
            import onnxruntime as ort
            from transformers import BertTokenizer
            print(f"[INFO] Loading ONNX FP32 BERT from: {onnx_fp32_file}")
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            _model = ort.InferenceSession(onnx_fp32_file, sess_options, providers=["CPUExecutionProvider"])
            _tokenizer = BertTokenizer.from_pretrained(model_dir)
            _engine = "onnx"
            _device = "CPU (ONNX FP32)"
            return _model, _tokenizer, _device
        except Exception as e:
            print(f"[WARNING] Failed to load ONNX FP32 model: {e}. Falling back...")

    # 3. Fallback: PyTorch Safetensors Model
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"Trained model not found at {model_dir}. Please run 'python src/train.py' first."
        )

    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    _device = get_device()
    print(f"[INFO] Loading PyTorch BERT model and tokenizer from: {model_dir}")
    _tokenizer = BertTokenizer.from_pretrained(model_dir)
    _model = BertForSequenceClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()
    _engine = "pytorch"
    return _model, _tokenizer, _device


def predict_text(text: str) -> dict:
    """
    Performs inference on a single social-media text input.

    Returns dictionary containing:
      - text: original input
      - label: "Normal" or "Threat"
      - label_id: 0 or 1
      - confidence: float percentage (e.g. 98.45)
      - probabilities: {"Normal": p0, "Threat": p1}
      - explanation: context message
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return {
            "error": "Input text cannot be empty or null.",
            "label": "Invalid Input",
            "confidence": 0.0,
            "probabilities": {"Normal": 0.0, "Threat": 0.0},
            "explanation": "Please enter valid text."
        }

    model, tokenizer, device = load_threat_model()

    # Load calibration temperature if available
    import json
    calib_file = os.path.join(MODELS_DIR, "calibration.json")
    temperature = 3.5490
    if os.path.exists(calib_file):
        try:
            with open(calib_file, "r", encoding="utf-8") as f:
                calib_meta = json.load(f)
                temperature = float(calib_meta.get("temperature", 3.5490))
        except Exception:
            temperature = 3.5490

    if _engine == "onnx":
        encoding = tokenizer(
            text.strip(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="np"
        )
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        ort_out = model.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = ort_out[0]
        # Temperature-scaled probabilities
        scaled_logits = logits / temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
        pred_idx = int(np.argmax(logits))
    else:
        import torch
        encoding = tokenizer(
            text.strip(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Temperature-scaled probabilities
            probs = torch.softmax(logits / temperature, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(torch.argmax(logits, dim=1).item())

    pred_label = LABEL_MAP[pred_idx]
    confidence_pct = float(probs[pred_idx] * 100.0)

    if pred_label == "Threat":
        explanation = "The model flags this post as potentially threatening based on learned malicious, scam, or hostile linguistic patterns."
    else:
        explanation = "The model classifies this post as normal, safe social media communication without threat patterns."

    return {
        "text": text,
        "label": pred_label,
        "label_id": pred_idx,
        "confidence": confidence_pct,
        "probabilities": {
            "Normal": float(probs[0] * 100.0),
            "Threat": float(probs[1] * 100.0)
        },
        "explanation": explanation
    }


def run_cli():
    print("=" * 70)
    print("      SOCIAL MEDIA THREAT INTELLIGENCE - REAL-TIME PREDICTOR")
    print("=" * 70)

    sample_inputs = [
        "Had a great day with my friends today.",
        "Congratulations! You won a free phone. Click this link immediately!",
        "Starting my morning with a warm cup of green tea and peaceful music.",
        "URGENT ALERT: Whistleblower leaks verified documents showing contaminated water supply. Share now!",
        "Hey @target_user_88 you are disgusting vermin and should be hunted down.",
        "Baking fresh chocolate chip cookies makes the kitchen smell wonderful."
    ]

    print("\n[Running Sample Predictions]")
    for idx, sample in enumerate(sample_inputs, 1):
        res = predict_text(sample)
        print(f"\n--- Sample #{idx} ---")
        print(f"Input:       \"{res['text']}\"")
        print(f"Prediction:  {res['label']}")
        print(f"Confidence:  {res['confidence']:.2f}%")
        print(f"Probabilities: Normal: {res['probabilities']['Normal']:.2f}% | Threat: {res['probabilities']['Threat']:.2f}%")
        print(f"Explanation: {res['explanation']}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_cli()
