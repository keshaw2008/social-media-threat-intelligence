import os
import random
import numpy as np
# Standard Label Mapping
LABEL_MAP = {
    0: "Normal",
    1: "Threat"
}
INVERSE_LABEL_MAP = {
    "Normal": 0,
    "Threat": 1
}

# Project Directory Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FINAL_DATASET_PATH = os.path.join(DATA_DIR, "final_social_media_threat_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "threat_bert")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def set_seed(seed: int = 42) -> None:
    """
    Sets seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    print(f"[INFO] Random seed set to {seed}")


def get_device():
    """
    Detects and returns CUDA device if available, otherwise falls back to CPU.
    Prints device details.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[INFO] Device: CUDA")
            print(f"[INFO] GPU: {gpu_name} (~{gpu_memory:.2f} GB VRAM)")
            return device
        else:
            device = torch.device("cpu")
            print("[INFO] Device: CPU (CUDA is unavailable or disabled)")
            return device
    except ImportError:
        print("[INFO] Device: CPU (PyTorch not installed, using lightweight inference)")
        return "cpu"


def ensure_directories():
    """
    Ensures that all standard project directories exist.
    """
    for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
