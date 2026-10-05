---
title: Social Media Threat Intelligence System using BERT
emoji: 🛡️
colorFrom: cyan
colorTo: blue
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI-powered detection of harmful social-media content using BERT
---

# Social Media Threat Intelligence System using BERT

An end-to-end NLP and Threat Intelligence system for automated classification of social-media posts into **Normal** or **Threat** categories using a fine-tuned Bidirectional Encoder Representations from Transformers (**BERT** - `bert-base-uncased`) model.

---

## 1. Project Overview & Abstract

Social-media platforms generate massive volumes of real-time user posts every second. While the majority of content represents benign everyday communication, a subset carries malicious intent, including **misinformation/fake alerts**, **cyber harassment & targeted hostility**, and **scams/phishing fraud**.

This project provides a complete end-to-end threat intelligence system:
1. **Synthetic Dataset Engineering**: Generates 10,000 diverse English social-media posts with controlled class balance (5,000 Normal vs. 5,000 Threat) and 3.5% intentional noise/outlier samples.
2. **Preprocessing Pipeline**: Preserves social-media semantics while cleaning whitespace and managing formatting.
3. **Deep Learning Fine-Tuning**: Fine-tunes `bert-base-uncased` with PyTorch, AdamW optimization, dynamic learning rate scheduling, and mixed precision.
4. **Evaluation & Outlier Analysis**: Evaluates on an unseen stratified test set (1,500 samples) with full classification metrics and robustness checks.
5. **Interactive Gradio Portal**: Multi-tab cybersecurity dashboard (`app.py`) ready for Hugging Face Spaces.
6. **Modern FastAPI Web Application**: High-performance REST backend (`server.py`) with cyber-mesh dashboard (`web/templates/index.html`).

> **Notice & Prototype Scope**: *This project is a threat-intelligence classification prototype that detects learned malicious linguistic patterns. It is designed for cybersecurity monitoring and does not independently verify the factual truth of arbitrary real-world statements.*

---

## 2. Problem Statement & Objectives

### Problem Statement
Traditional rule-based keyword filters fail to capture subtle linguistic cues, varied sentence structures, obfuscated phishing links, and noisy social-media formatting (such as repeated characters, excessive punctuation, and mixed casing).

### Objectives
* Build a reproducible binary classifier targeting two distinct classes:
  * **Class 0 (`Normal`)**: Safe everyday conversations, education, travel, food, tech, hobbies, and general opinions.
  * **Class 1 (`Threat`)**: Malicious posts covering fake news/emergency claims, targeted harassment/hostility, and financial scams/phishing links.
* Fine-tune `bert-base-uncased` with sequence classification layers.
* Incorporate realistic noisy outliers to assess real-world robustness.
* Deploy an intuitive cybersecurity web portal and Gradio interface.

---

## 3. System Architecture

```text
               +------------------------------------+
               |      Social Media Post (Input)     |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |       Lightweight Preprocessor     |
               | (Whitespace, URL/Tag Normalization)|
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |           BERT Tokenizer           |
               | (WordPiece, [CLS], [SEP], Masking) |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |    BERT Base Encoder (12 Layers)   |
               |     768-dim Hidden Vectors         |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |        [CLS] Representation        |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |    Linear Binary Classifier Head   |
               |           (num_labels = 2)         |
               +------------------------------------+
                                 |
                                 v
               +------------------------------------+
               |        Softmax Probabilities       |
               +------------------------------------+
                                 |
                     +-----------+-----------+
                     |                       |
                     v                       v
            [Normal (Class 0)]      [Threat (Class 1)]
             Confidence: XX.XX%      Confidence: XX.XX%
```

---

## 4. Dataset Description & Synthesis

The dataset was generated synthetically using structured combinatorial slot-filling and contextual variations:

* **Total Samples**: 10,000 unique records
* **Label 0 (`Normal`)**: 5,000 samples (50.0%)
* **Label 1 (`Threat`)**: 5,000 samples (50.0%)
* **Outliers (`is_outlier=1`)**: 350 samples (3.5%)

### Threat Categories Covered:
1. **Fake News / Fabricated Emergencies**: False public health claims, fake government bans, fabricated disaster announcements, leaked conspiracies.
2. **Targeted Hostility / Cyber Harassment**: Group-targeted exclusionary rhetoric, aggressive cyberbullying, intimidation (using safe synthetic patterns without explicit slurs).
3. **Spam / Phishing / Scams**: Fake cryptocurrency giveaways, lottery claims, urgent account suspensions, parcel delivery fraud.

### Dataset Schema:
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique identifier (1 to 10,000) |
| `text` | String | English social media post content |
| `label` | Integer | `0` = Normal, `1` = Threat |
| `is_outlier` | Integer | `0` = Regular sample, `1` = Intentionally noisy outlier |

---

## 5. Dataset Splitting

The dataset is partitioned using a reproducible stratified split (`seed=42`) ensuring balanced representation of classes and outliers:

* **Training Set (70%)**: 7,000 samples (3,500 Normal, 3,500 Threat) -> `data/processed/train.csv`
* **Validation Set (15%)**: 1,500 samples (750 Normal, 750 Threat) -> `data/processed/validation.csv`
* **Test Set (15%)**: 1,500 samples (750 Normal, 750 Threat) -> `data/processed/test.csv`

---

## 6. Training Configuration

* **Model**: `bert-base-uncased` (110M parameters)
* **Max Sequence Length**: `128`
* **Train Batch Size**: `2`
* **Eval Batch Size**: `2`
* **Gradient Accumulation Steps**: `4` (Effective batch size = `8`)
* **Optimizer**: AdamW (`lr=2e-5`, `weight_decay=0.01`)
* **Warmup Ratio**: `0.1` (Linear schedule with warmup)
* **Epochs**: `2`
* **Mixed Precision (FP16)**: Auto-detected (enabled on CUDA)
* **Random Seed**: `42`

### Training History (Actual Execution Logs):
* **Epoch 1**:
  * Train Loss: `0.0635` | Train Accuracy: `97.43%`
  * Validation Loss: `0.0013` | Validation Accuracy: `99.93%` | Validation F1: `99.93%`
* **Epoch 2**:
  * Train Loss: `0.0016` | Train Accuracy: `99.97%`
  * Validation Loss: `0.0000` | Validation Accuracy: `100.00%` | Validation F1: `100.00%`

---

## 7. Model Evaluation Results (Unseen Test Set)

Evaluated exclusively on the unseen test set of **1,500 samples**:

```text
======================================================================
             FINAL BERT MODEL EVALUATION RESULTS (TEST SET)
======================================================================
Test Accuracy:          1.0000 (100.00%)
Threat Precision:       1.0000 (100.00%)
Threat Recall:          1.0000 (100.00%)
Threat F1 Score:        1.0000 (100.00%)
Macro F1 Score:         1.0000 (100.00%)
Weighted F1 Score:      1.0000 (100.00%)
----------------------------------------------------------------------
Classification Report:

              precision    recall  f1-score   support

      Normal     1.0000    1.0000    1.0000       750
      Threat     1.0000    1.0000    1.0000       750

    accuracy                         1.0000      1500
   macro avg     1.0000    1.0000    1.0000      1500
weighted avg     1.0000    1.0000    1.0000      1500

----------------------------------------------------------------------
Confusion Matrix:
                 Predicted Normal  Predicted Threat
Actual Normal  : 750               0
Actual Threat  : 0                 750
======================================================================
```

### Outlier Robustness Analysis:
* **Regular Test Samples (1,448 samples)**:
  * Accuracy: `100.00%`
  * Macro F1: `100.00%`
* **Outlier Test Samples (52 noisy samples)**:
  * Accuracy: `100.00%`
  * Macro F1: `100.00%`

---

## 8. Web Application Endpoints

The FastAPI backend exposes the following REST API endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive dark-themed cybersecurity dashboard |
| `GET` | `/health` | Returns server health, hardware device, and BERT status |
| `POST` | `/predict` | Performs sequence classification on input text (`{"text": "..."}`) |
| `GET` | `/api/metrics` | Serves actual evaluation metrics and confusion matrix data |
| `GET` | `/api/dataset-stats` | Serves dataset distribution and split parameters |
| `GET` | `/gradio` | Accesses the mounted Gradio interface |

---

## 9. Project Structure

```text
Social-Media-Threat-Intelligence/
│
├── app.py                      # Primary Gradio application (Hugging Face Spaces Entrypoint)
├── server.py                   # FastAPI application & REST backend
├── main.py                     # Uvicorn entry point
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation & Space configuration
├── .gitignore                  # Git exclusions
│
├── web/                        # Web application frontend
│   ├── templates/
│   │   └── index.html          # Dark cybersecurity dashboard template
│   └── static/
│       ├── css/
│       │   └── style.css       # Cybersecurity styling & animations
│       └── js/
│           └── main.js         # Async API handler & UI interactivity
│
├── data/
│   ├── raw/
│   │   └── synthetic_social_media_posts.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── final_social_media_threat_dataset.csv
│
├── models/
│   └── threat_bert/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_metadata.json
│
├── results/
│   ├── class_distribution.png
│   ├── outlier_distribution.png
│   ├── text_length_distribution.png
│   ├── training_loss.png
│   ├── validation_loss.png
│   ├── training_accuracy.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── test_metrics_summary.json
│
└── src/
    ├── utils.py
    ├── generate_dataset.py
    ├── preprocess.py
    ├── analyze_dataset.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

---

## 10. How to Run Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Gradio Application (Hugging Face Spaces Default)
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:7860`**

### Step 3: Run the FastAPI Cybersecurity Portal (Alternative)
```bash
python server.py
# OR
uvicorn server:app --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://127.0.0.1:8000`**

---

## 11. Manual Deployment Guide for Hugging Face Spaces

To deploy this project to Hugging Face Spaces manually:

1. **Create a New Space**:
   * Navigate to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   * Choose a Space name (e.g., `social-media-threat-intelligence-bert`).
   * Select **Gradio** as the Space SDK.
   * Choose **Public** visibility.
   * Select the free **CPU basic (2 vCPU, 16 GB RAM)** hardware tier.

2. **Upload / Push Project Files**:
   * Clone your newly created Space repository locally:
     ```bash
     git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
     ```
   * Copy the following files/folders from this project directory into your cloned repository:
     * `app.py`
     * `requirements.txt`
     * `README.md`
     * `.gitignore`
     * `models/threat_bert/` (Make sure `model.safetensors`, `config.json`, `tokenizer.json`, and `tokenizer_config.json` are included)
     * `results/`
     * `src/`
     * `data/` (Optional, for reference)
     * `web/` (Optional, for FastAPI dashboard)
   * Track model weights with Git LFS:
     ```bash
     git lfs install
     git lfs track "models/threat_bert/model.safetensors"
     git add .gitattributes
     ```
   * Commit and push:
     ```bash
     git add .
     git commit -m "Deploy Social Media Threat Intelligence BERT system"
     git push
     ```

3. **Automatic Build & Launch**:
   * Hugging Face Spaces will automatically install dependencies from `requirements.txt` and launch `app.py`.
   * Your public live application will be available at: `https://huggingface.co/spaces/<your-username>/<your-space-name>` (and embeddable via direct `.hf.space` URL).
