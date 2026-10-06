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
short_description: Automated detection of harmful social-media content using BERT
---

# Social Media Threat Intelligence System using BERT

An end-to-end NLP and Threat Intelligence system for automated classification of social-media posts into **Normal** or **Threat** categories using a fine-tuned Bidirectional Encoder Representations from Transformers (**BERT** — `bert-base-uncased`) model.

---

## 1. Project Overview & Abstract

Social-media platforms generate massive volumes of user posts every second. While the majority of content represents benign everyday communication, a subset carries malicious intent, including **misinformation/fake alerts**, **cyber harassment & targeted hostility**, and **scams/phishing fraud**.

This project provides a complete end-to-end threat intelligence system:
1. **Dataset Collection & Integration**: Utilizes externally collected, publicly available Kaggle datasets covering Fake News, Hate Speech, Spam, and Normal social-media posts (10,000 balanced samples: 5,000 Normal vs. 5,000 Threat).
2. **Preprocessing Pipeline**: Cleans, deduplicates, and formats social-media text while preserving linguistic cues for transformer tokenization.
3. **Deep Learning Fine-Tuning**: Fine-tunes `bert-base-uncased` with PyTorch, AdamW optimization, dynamic learning rate scheduling, and mixed precision.
4. **Probability Calibration**: Applies validation-fitted temperature scaling ($T = 3.5490$) to provide reliable, realistic prediction confidences.
5. **Evaluation & Outlier Analysis**: Evaluates on an unseen stratified test set (1,500 samples) with comprehensive classification metrics and outlier robustness checks.
6. **Production Deployment**: High-performance REST backend (`server.py`) using ONNX Runtime INT8 quantization for sub-50ms CPU inference.

> **Notice & Prototype Scope**: *This project is a threat-intelligence classification prototype that detects learned malicious linguistic patterns. It is designed for cybersecurity monitoring and does not independently verify whether a post is factually true or whether a person is actually dangerous.*

---

## 2. Problem Statement & Objectives

### Problem Statement
Traditional rule-based keyword filters fail to capture subtle linguistic cues, varied sentence structures, obfuscated phishing links, and noisy social-media formatting (such as repeated characters, excessive punctuation, and mixed casing).

### Objectives
* Build a reproducible binary classifier targeting two distinct classes:
  * **Class 0 (`Normal`)**: Safe everyday conversations, education, travel, food, sports, hobbies, and general opinions.
  * **Class 1 (`Threat`)**: Malicious posts covering fake news, targeted harassment/hostility, and financial scams/phishing links.
* Fine-tune `bert-base-uncased` with sequence classification layers.
* Incorporate noisy outlier observations found in social-media text to assess real-world robustness.
* Deploy an intuitive, responsive academic web portal and REST API.

---

## 3. System Architecture & Processing Pipeline

```text
Public Kaggle Datasets (Fake News, Hate Speech, Spam, Normal Posts)
                               ↓
                      Dataset Integration
                               ↓
                        Data Cleaning
                               ↓
                      Duplicate Checking
                               ↓
                    Missing-Value Handling
                               ↓
                      Text Preprocessing
                               ↓
               Label Mapping (Normal: 0, Threat: 1)
                               ↓
                       Outlier Analysis
                               ↓
              Train / Validation / Test Stratified Split
                               ↓
                      BERT Tokenization
                               ↓
                    BERT Model Fine-Tuning
                               ↓
                    Evaluation & Calibration
                               ↓
                     ONNX INT8 Optimization
                               ↓
                       FastAPI Backend
                               ↓
                    Web-Based Threat Analyzer
```

---

## 4. Dataset Description & Sources

The project uses externally collected publicly available datasets obtained from Kaggle. The source data cover four distinct categories:

1. **Fake News**: Misleading, fabricated, or potentially unreliable emergency claims and misinformation.
2. **Hate Speech**: Hostile, abusive, discriminatory, or harmful textual content.
3. **Spam**: Suspicious promotional, scam-like, phishing-like, or unsolicited financial messages.
4. **Normal Posts**: Ordinary non-threatening social-media content (daily activities, casual conversations, sports, food, travel, hobbies, general commentary).

### Binary Class Mapping:
* **Source Categories (Fake News, Hate Speech, Spam)** $\rightarrow$ **Class 1 (`Threat`)** (5,000 samples)
* **Source Category (Normal Posts)** $\rightarrow$ **Class 0 (`Normal`)** (5,000 samples)
* **Total Records**: 10,000 balanced samples

### Dataset Schema:
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique identifier (1 to 10,000) |
| `text` | String | English social media post content |
| `label` | Integer | `0` = Normal, `1` = Threat |
| `is_outlier` | Integer | `0` = Regular sample, `1` = Noisy outlier pattern |

---

## 5. Dataset Splitting

The dataset is partitioned using a reproducible stratified split (`seed=42`) ensuring balanced class distributions across all splits:

* **Training Set (70%)**: 7,000 samples (3,500 Normal, 3,500 Threat) $\rightarrow$ `data/processed/train.csv`
* **Validation Set (15%)**: 1,500 samples (750 Normal, 750 Threat) $\rightarrow$ `data/processed/validation.csv`
* **Test Set (15%)**: 1,500 samples (750 Normal, 750 Threat) $\rightarrow$ `data/processed/test.csv`

---

## 6. Training Configuration

* **Model**: `bert-base-uncased` (110M parameters)
* **Max Sequence Length**: `128`
* **Train Batch Size**: `16`
* **Eval Batch Size**: `32`
* **Gradient Accumulation Steps**: `1`
* **Optimizer**: AdamW (`lr=2e-5`, `weight_decay=0.01`)
* **Warmup Ratio**: `0.1` (Linear schedule with warmup)
* **Epochs**: `3`
* **Random Seed**: `42`

---

## 7. Model Evaluation Results (Unseen Test Set)

Evaluated on the unseen test set of **1,500 samples**:

```text
======================================================================
             FINAL BERT MODEL EVALUATION RESULTS (TEST SET)
======================================================================
Test Accuracy:          0.8800 (88.00%)
Threat Precision:       0.8202 (82.02%)
Threat Recall:          0.9733 (97.33%)
Threat F1 Score:        0.8902 (89.02%)
Macro F1 Score:         0.8789 (87.89%)
Weighted F1 Score:      0.8789 (87.89%)
----------------------------------------------------------------------
Classification Report:

              precision    recall  f1-score   support

      Normal     0.9672    0.7867    0.8676       750
      Threat     0.8202    0.9733    0.8902       750

    accuracy                         0.8800      1500
   macro avg     0.8937    0.8800    0.8789      1500
weighted avg     0.8937    0.8800    0.8789      1500

----------------------------------------------------------------------
Confusion Matrix:
                 Predicted Normal  Predicted Threat
Actual Normal  : 590               160
Actual Threat  : 20                730
======================================================================
```

### Outlier Robustness Analysis:
* **Regular Test Samples (1,421 samples)**:
  * Accuracy: `88.04%`
  * Macro F1: `87.92%`
* **Outlier Test Samples (79 noisy samples)**:
  * Accuracy: `87.34%`
  * Macro F1: `87.32%`

---

## 8. Probability Calibration

Temperature scaling was applied using validation outputs ($T = 3.5490$) to produce conservative, reliable confidence estimates without modifying the predicted class:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

* **Calibration Method**: Validation-fitted Temperature Scaling
* **Temperature Parameter ($T$)**: `3.5490`
* **Validation NLL**: `0.2871` (calibrated) vs `0.6961` (uncalibrated)
* **Maximum Validation Confidence**: $\le 95.00\%$

---

## 9. Web Application Endpoints

The FastAPI backend exposes the following REST API endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the academic project web dashboard |
| `GET` | `/health` | Returns server health, hardware device, and BERT status |
| `POST` | `/predict` | Performs sequence classification on input text (`{"text": "..."}`) |
| `GET` | `/api/metrics` | Serves actual evaluation metrics and confusion matrix data |

---

## 10. Project Structure

```text
social-media-threat-intelligence/
│
├── server.py                   # FastAPI REST backend & inference server
├── main.py                     # Uvicorn launcher
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── render.yaml                 # Render deployment configuration
│
├── web/                        # Web application frontend
│   ├── templates/
│   │   └── index.html          # Clean academic dashboard template
│   └── static/
│       ├── css/
│       │   └── style.css       # Academic styling & responsive layout
│       └── js/
│           └── main.js         # Client API integration & UI handlers
│
├── data/                       # Dataset directories
│   ├── raw/
│   │   └── collected_social_media_posts.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── final_social_media_threat_dataset.csv
│
├── models/
│   └── threat_bert/
│       ├── config.json
│       ├── model_quantized.onnx.gz
│       ├── calibration.json
│       ├── tokenizer.json
│       └── training_metadata.json
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── test_metrics_summary.json
│
└── src/
    ├── utils.py
    ├── preprocess.py
    ├── train.py
    ├── evaluate.py
    ├── fit_calibration.py
    └── predict.py
```

---

## 11. How to Run Locally

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Start the Web Application
```bash
python server.py
# OR
uvicorn server:app --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://127.0.0.1:8000`**
