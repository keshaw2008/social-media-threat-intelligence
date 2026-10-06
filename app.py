import os
import sys
import time
from pathlib import Path
import gradio as gr

# Setup relative paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models" / "threat_bert"
RESULTS_DIR = BASE_DIR / "results"

sys.path.append(str(SRC_DIR))
from predict import predict_text, load_threat_model
from utils import get_device

# Preload model once on startup
try:
    load_threat_model(str(MODELS_DIR))
    device = get_device()
    device_info = f"Active ({device.type.upper()})"
except Exception as e:
    print(f"[WARNING] Model loading warning on startup: {e}")
    device_info = "Ready on first request"


def analyze_post(post_text: str):
    """
    Inference handler for Gradio UI using actual fine-tuned BERT model.
    """
    if not post_text or not post_text.strip():
        return (
            "[!] Error: Empty Input",
            "0.00%",
            {"Normal": 0.0, "Threat": 0.0},
            "Please provide a non-empty social-media post.",
            "0.0 ms"
        )

    start_time = time.perf_counter()
    result = predict_text(post_text.strip())
    inference_ms = (time.perf_counter() - start_time) * 1000.0

    if "error" in result and result.get("label") == "Invalid Input":
        return (
            "[!] Error: Invalid Input",
            "0.00%",
            {"Normal": 0.0, "Threat": 0.0},
            result.get("error", "Invalid input format."),
            f"{inference_ms:.1f} ms"
        )

    label = result["label"]
    confidence = f"{result['confidence']:.2f}%"
    explanation = result["explanation"]
    probabilities = {
        "Normal": result["probabilities"]["Normal"] / 100.0,
        "Threat": result["probabilities"]["Threat"] / 100.0
    }

    if label == "Threat":
        badge = "[THREAT DETECTED]"
    else:
        badge = "[NORMAL (SAFE)]"

    latency_str = f"{inference_ms:.1f} ms"
    return badge, confidence, probabilities, explanation, latency_str


# Custom CSS for modern dark cybersecurity theme
custom_css = """
#main-header { text-align: center; margin-bottom: 1.5rem; }
.cyber-badge { font-family: monospace; font-size: 0.85rem; font-weight: 600; color: #06b6d4; }
.metric-box { background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 1rem; text-align: center; }
.metric-val { font-size: 1.8rem; font-weight: 800; color: #38bdf8; }
.metric-label { font-size: 0.8rem; text-transform: uppercase; color: #9ca3af; font-family: monospace; }
"""

with gr.Blocks(title="Social Media Threat Intelligence System using BERT") as demo:
    gr.Markdown(
        f"""
        # 🛡️ Social Media Threat Intelligence System
        ### Fine-Tuned BERT (`bert-base-uncased`) Binary Threat Detector
        **Automated classification of social-media content** | Hardware Device: `{device_info}`
        """,
        elem_id="main-header"
    )

    with gr.Tabs():
        # Tab 1: Live Threat Analyzer
        with gr.TabItem("🔍 Live Threat Analyzer"):
            with gr.Row():
                with gr.Column(scale=6):
                    input_box = gr.Textbox(
                        label="Enter Social-Media Post",
                        placeholder="Type or paste a social-media post to analyze for threat patterns...",
                        lines=5,
                        max_lines=10
                    )
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze Threat", variant="primary", scale=2)
                        clear_btn = gr.Button("🔄 Clear", variant="secondary", scale=1)

                    gr.Examples(
                        examples=[
                            ["Had a great day with my friends today."],
                            ["Congratulations! You won a free phone. Click this link immediately!"],
                            ["We are going to totally destroy the enemy squad in Valorant tonight!"],
                            ["We know where your children go to school in Chicago. Keep your mouth shut."]
                        ],
                        inputs=input_box,
                        label="Quick Sample Inputs (Click to test)"
                    )

                with gr.Column(scale=5):
                    output_badge = gr.Textbox(label="Classification Verdict", interactive=False)
                    with gr.Row():
                        output_conf = gr.Textbox(label="Model Confidence", interactive=False, scale=1)
                        output_latency = gr.Textbox(label="Inference Latency", interactive=False, scale=1)
                    output_probs = gr.Label(label="Softmax Probability Distribution", num_top_classes=2)
                    output_expl = gr.Textbox(label="Linguistic Context & Rationale", interactive=False, lines=3)

            analyze_btn.click(
                fn=analyze_post,
                inputs=[input_box],
                outputs=[output_badge, output_conf, output_probs, output_expl, output_latency]
            )
            clear_btn.click(
                fn=lambda: ("", "", {}, "", ""),
                inputs=[],
                outputs=[input_box, output_badge, output_conf, output_probs, output_expl, output_latency]
            )

        # Tab 2: Performance & Test-Set Evaluation
        with gr.TabItem("📊 Performance & Evaluation"):
            gr.Markdown(
                """
                ### 🏆 Verified Performance on Held-Out Test Set (1,500 Unseen Samples)
                All metrics below are computed from actual model execution on the held-out test partition.
                """
            )
            with gr.Row():
                gr.Markdown(
                    """
                    | Metric | Score | Support |
                    | :--- | :---: | :---: |
                    | **Test Accuracy** | **88.00%** | 1,500 posts |
                    | **Threat Precision** | **82.02%** | 750 threats |
                    | **Threat Recall** | **97.33%** | 750 threats |
                    | **Threat F1-Score** | **89.02%** | 750 threats |
                    | **Macro F1** | **87.89%** | 1,500 posts |
                    | **Weighted F1** | **87.89%** | 1,500 posts |
                    """
                )
                gr.Markdown(
                    """
                    ### 🎯 Confusion Matrix (1,500 Test Samples)
                    * **True Negatives (TN)**: `590` (Actual Normal → Predicted Normal)
                    * **False Positives (FP)**: `160` (Actual Normal → Predicted Threat)
                    * **False Negatives (FN)**: `20` (Actual Threat → Predicted Normal)
                    * **True Positives (TP)**: `730` (Actual Threat → Predicted Threat)

                    ### ⚡ Outlier Robustness Breakdown
                    * **Regular Test Samples (1,421)**: `88.04%` Accuracy (87.92% Macro F1)
                    * **Noisy Outlier Test Samples (79)**: `87.34%` Accuracy (87.32% Macro F1)
                    """
                )

            with gr.Row():
                cm_img_path = RESULTS_DIR / "confusion_matrix.png"
                train_loss_path = RESULTS_DIR / "training_loss.png"
                train_acc_path = RESULTS_DIR / "training_accuracy.png"
                if cm_img_path.exists():
                    gr.Image(str(cm_img_path), label="Confusion Matrix Heatmap")
                if train_loss_path.exists():
                    gr.Image(str(train_loss_path), label="Training Loss Curve")
                if train_acc_path.exists():
                    gr.Image(str(train_acc_path), label="Accuracy Curves")

        # Tab 3: Dataset & Architecture
        with gr.TabItem("📈 Dataset & Pipeline"):
            gr.Markdown(
                """
                ### 📁 Dataset Collection & Preprocessing (10,000 Posts)
                * **Source**: Publicly available Kaggle datasets covering Fake News, Hate Speech, Spam, and Normal posts.
                * **Total Records**: 10,000 balanced English posts
                * **Normal Posts (`label=0`)**: 5,000 samples (50.0%)
                * **Threat Posts (`label=1`)**: 5,000 samples (50.0%)
                * **Outlier Analysis Samples**: 350 noisy samples (3.5%)
                * **Stratified Split**: 70% Train (7,000), 15% Validation (1,500), 15% Test (1,500) with fixed seed `42`.

                ### 🏗️ End-to-End System Pipeline
                `Public Dataset Collection` → `Integration & Cleaning` → `WordPiece Tokenizer` → `BERT Base (12 Layers, 768 Dim)` → `[CLS] Representation` → `Linear Classification Head` → `Softmax Normal / Threat Verdict`
                """
            )
            with gr.Row():
                class_dist_path = RESULTS_DIR / "class_distribution.png"
                outlier_dist_path = RESULTS_DIR / "outlier_distribution.png"
                text_len_path = RESULTS_DIR / "text_length_distribution.png"
                if class_dist_path.exists():
                    gr.Image(str(class_dist_path), label="Class Distribution")
                if outlier_dist_path.exists():
                    gr.Image(str(outlier_dist_path), label="Outlier Distribution")
                if text_len_path.exists():
                    gr.Image(str(text_len_path), label="Text Length Distribution")

        # Tab 4: About & Ethical Disclaimer
        with gr.TabItem("ℹ️ About & Scope"):
            gr.Markdown(
                """
                ### 🧠 Model Architecture Specifications
                * **Base Transformer**: `bert-base-uncased` (Bidirectional Encoder Representations from Transformers)
                * **Parameters**: ~110 Million trainable weights
                * **Sequence Length**: 128 Tokens with WordPiece subword vocabulary (30,522 tokens)
                * **Optimizer**: AdamW (`lr=2e-5`, `weight_decay=0.01`) with linear warmup schedule
                * **Classes**: Binary (`0: Normal`, `1: Threat`)

                ### ⚠️ Notice & Limitations
                * *This system is a **threat-intelligence classification prototype** trained on learned linguistic patterns.*
                * *It identifies patterns characteristic of malicious posts (scams, phishing, targeted hostility, fake emergency claims).*
                * *It does not independently verify whether a post is factually true or whether a person is actually dangerous.*
                """
            )

    gr.Markdown(
        """
        ---
        **Social Media Threat Intelligence System using BERT** • *Academic Project*
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate", neutral_hue="slate"), css=custom_css)
