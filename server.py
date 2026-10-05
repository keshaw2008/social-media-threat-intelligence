import os
import sys
import time
import json
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import BertTokenizer, BertForSequenceClassification

# Base directories using pathlib for cross-platform and cloud compatibility
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models" / "threat_bert"
RESULTS_DIR = BASE_DIR / "results"
WEB_DIR = BASE_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# Add src to sys.path
sys.path.append(str(SRC_DIR))
from utils import LABEL_MAP

# Global Model Cache
model_state = {
    "engine": None,         # "onnx" or "pytorch"
    "model": None,          # ort.InferenceSession or torch model
    "tokenizer": None,
    "device": None,
    "device_name": "CPU (ONNX INT8 Quantized)",
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "loaded": False
}


def load_model_on_startup():
    """
    Loads fine-tuned BERT model into memory.
    Prioritizes ultra-lightweight ONNX INT8 quantized model (~105MB, ~150MB RAM) for Render Free compatibility.
    Falls back to PyTorch if ONNX is not available.
    """
    if model_state["loaded"] and model_state["model"] is not None:
        return

    onnx_int8_file = MODELS_DIR / "model_quantized.onnx"
    onnx_fp32_file = MODELS_DIR / "model.onnx"

    # 1. Preferred: ONNX INT8 Quantized Model (~105 MB)
    if onnx_int8_file.exists():
        try:
            import onnxruntime as ort
            print(f"[STARTUP] Loading ONNX INT8 Quantized BERT from: {onnx_int8_file}", flush=True)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort_session = ort.InferenceSession(str(onnx_int8_file), sess_options, providers=["CPUExecutionProvider"])
            tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))

            model_state["engine"] = "onnx"
            model_state["model"] = ort_session
            model_state["tokenizer"] = tokenizer
            model_state["device_name"] = "CPU (ONNX INT8 Quantized)"
            model_state["loaded"] = True
            print("[STARTUP] ONNX INT8 Quantized BERT loaded successfully (RAM-optimized for 512MB limit).", flush=True)
            return
        except Exception as e:
            print(f"[STARTUP WARNING] ONNX INT8 load error: {e}. Falling back...", flush=True)

    # 2. Alternative: ONNX FP32 Model
    if onnx_fp32_file.exists():
        try:
            import onnxruntime as ort
            print(f"[STARTUP] Loading ONNX FP32 BERT from: {onnx_fp32_file}", flush=True)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            ort_session = ort.InferenceSession(str(onnx_fp32_file), sess_options, providers=["CPUExecutionProvider"])
            tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))

            model_state["engine"] = "onnx"
            model_state["model"] = ort_session
            model_state["tokenizer"] = tokenizer
            model_state["device_name"] = "CPU (ONNX FP32)"
            model_state["loaded"] = True
            print("[STARTUP] ONNX FP32 BERT loaded successfully.", flush=True)
            return
        except Exception as e:
            print(f"[STARTUP WARNING] ONNX FP32 load error: {e}. Falling back...", flush=True)

    # 3. Fallback: PyTorch Safetensors Model
    if not (MODELS_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"Trained BERT model not found at {MODELS_DIR}. Ensure model files exist."
        )

    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU (PyTorch)"
        torch.set_num_threads(2)

    print(f"[STARTUP] Loading PyTorch BERT model onto {device_name} from: {MODELS_DIR}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(str(MODELS_DIR))
    model = BertForSequenceClassification.from_pretrained(str(MODELS_DIR))
    model.to(device)
    model.eval()

    import gc
    gc.collect()

    model_state["engine"] = "pytorch"
    model_state["model"] = model
    model_state["tokenizer"] = tokenizer
    model_state["device"] = device
    model_state["device_name"] = device_name
    model_state["loaded"] = True
    print(f"[STARTUP] PyTorch BERT Model loaded successfully.", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model_on_startup()
    yield
    # Shutdown
    model_state["loaded"] = False
    print("[SHUTDOWN] Application resources released.", flush=True)


app = FastAPI(
    title="Social Media Threat Intelligence System using BERT",
    description="Cybersecurity AI system for detecting threat patterns in social media posts.",
    version="1.0.0",
    lifespan=lifespan
)

# Mount Static Files and Results
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


# Pydantic Schemas
class PredictionRequest(BaseModel):
    text: str = Field(..., description="English social-media post text to analyze", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You won a $5,000 gift card. Click http://claim-rewards-verify77.xyz/win immediately!"
            }
        }


class PredictionResponse(BaseModel):
    text: str
    label: str
    label_id: int
    confidence: float
    confidence_raw: float
    probabilities: dict
    inference_time_ms: float
    device: str
    explanation: str


# Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the modern dark cybersecurity web interface.
    """
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html template not found")
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """
    Returns server and model health status.
    """
    if not model_state["loaded"]:
        try:
            load_model_on_startup()
        except Exception as e:
            print(f"[HEALTH] Lazy load error: {e}", flush=True)

    return {
        "status": "healthy" if model_state["loaded"] else "degraded",
        "model_loaded": model_state["loaded"],
        "model_name": model_state["model_name"],
        "engine": model_state["engine"],
        "device": model_state["device_name"],
        "max_sequence_length": model_state["max_length"],
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_threat(payload: PredictionRequest):
    """
    Performs real-time BERT sequence classification on input text using ONNX INT8 or PyTorch.
    """
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if not model_state["loaded"] or model_state["model"] is None:
        load_model_on_startup()

    tokenizer = model_state["tokenizer"]
    engine = model_state["engine"]
    model = model_state["model"]

    start_time = time.perf_counter()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=model_state["max_length"],
        return_tensors="np" if engine == "onnx" else "pt"
    )

    if engine == "onnx":
        import numpy as np
        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)

        ort_outputs = model.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        logits = ort_outputs[0]
        # Softmax over logits
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True))[0]
        pred_idx = int(np.argmax(probs))
    else:
        import torch
        device = model_state["device"]
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(torch.argmax(logits, dim=1).item())

    inference_ms = (time.perf_counter() - start_time) * 1000.0

    pred_label = LABEL_MAP.get(pred_idx, "Unknown")
    confidence_raw = float(probs[pred_idx])
    confidence_pct = round(confidence_raw * 100.0, 2)

    if pred_label == "Threat":
        explanation = "The model classifies this post as potentially threatening based on learned malicious, scam, or hostile linguistic patterns."
    else:
        explanation = "The model classifies this post as normal, safe social media communication without threat patterns."

    return PredictionResponse(
        text=text,
        label=pred_label,
        label_id=pred_idx,
        confidence=confidence_pct,
        confidence_raw=confidence_raw,
        probabilities={
            "Normal": round(float(probs[0]) * 100.0, 2),
            "Threat": round(float(probs[1]) * 100.0, 2)
        },
        inference_time_ms=round(inference_ms, 2),
        device=model_state["device_name"],
        explanation=explanation
    )


@app.get("/api/metrics")
async def get_metrics():
    """
    Returns verified evaluation metrics from results files.
    """
    summary_path = RESULTS_DIR / "test_metrics_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {"message": "Metrics summary file not found."}


@app.get("/api/dataset-stats")
async def get_dataset_stats():
    """
    Returns dataset distribution statistics.
    """
    return {
        "total_samples": 10000,
        "normal_samples": 5000,
        "threat_samples": 5000,
        "outlier_samples": 350,
        "outlier_percentage": 3.5,
        "splits": {
            "train": 7000,
            "validation": 1500,
            "test": 1500
        },
        "target_model": "bert-base-uncased",
        "classes": {
            0: "Normal",
            1: "Threat"
        }
    }




if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
