# ACADEMIC PROJECT REPORT

## PROJECT TITLE:
# Social Media Threat Intelligence System using BERT

**Academic Level:** Bachelor of Technology (B.Tech) Capstone Project  
**Domain:** Artificial Intelligence, Natural Language Processing, Cyber Threat Intelligence  
**Model Architecture:** `bert-base-uncased` (Bidirectional Encoder Representations from Transformers)  
**Target Language:** English  
**Classification Task:** Binary Classification — Normal (0), Threat (1)  
**Deployment Engine:** ONNX Runtime (INT8 Quantized) & FastAPI  

---

## 1. Abstract

The explosive expansion of social-media networks has democratized information exchange while simultaneously amplifying severe digital threats, such as viral misinformation, targeted cyber harassment, toxic hostility, predatory financial spam, and phishing schemes. Traditional cybersecurity solutions relying on regex pattern matching, keyword blacklists, and bag-of-words machine learning models fall short in capturing contextual semantics, obfuscated language, and noisy digital formatting.

This project presents the **Social Media Threat Intelligence System using BERT**, an end-to-end deep learning framework designed to analyze and classify social-media posts into two core categories: **Normal** (Class 0) and **Threat** (Class 1). The core classifier fine-tunes `bert-base-uncased` using PyTorch and Hugging Face Transformers. The dataset was collected from publicly available Kaggle datasets covering fake news, hate speech, spam, and normal social-media communications. The collected datasets were integrated, preprocessed, and balanced into a 10,000-sample benchmark. The system includes an outlier robustness analysis, validation-fitted temperature scaling calibration, and high-performance ONNX INT8 deployment via FastAPI.

---

## 2. Problem Statement & Objectives

### Problem Statement
Automated threat detection on social media faces key challenges:
1. **Contextual Ambiguity**: Words change meaning drastically depending on surrounding context, causing rule-based keyword filters to generate false positives.
2. **Obfuscation and Noise**: Malicious actors use leetspeak, repeated characters, excessive punctuation, and mixed casing to bypass naive heuristic filters.
3. **Overconfidence in Neural Networks**: Raw softmax probabilities in modern transformer models often exhibit severe overconfidence, requiring principled probability calibration.

### Objectives
1. Integrate and clean publicly available Kaggle datasets spanning four categories: Fake News, Hate Speech, Spam, and Normal Posts.
2. Map source categories into a structured binary classification format (Normal: 0, Threat: 1).
3. Fine-tune `bert-base-uncased` with AdamW optimization and learning rate scheduling.
4. Apply conservative Temperature Scaling ($T = 3.5490$) using the validation set to produce realistic confidence scores.
5. Evaluate model performance and outlier robustness on an unseen 1,500-sample test set.
6. Deploy an optimized ONNX INT8 production web service using FastAPI and a clean academic UI.

---

## 3. Dataset & Data Processing

### Dataset Sources
The dataset was collected from publicly available Kaggle datasets covering four distinct source categories:
1. **Fake News**: Misleading or potentially unreliable emergency claims and misinformation.
2. **Hate Speech**: Hostile, abusive, discriminatory, or harmful textual content.
3. **Spam**: Suspicious promotional, scam-like, phishing-like, or unsolicited messages.
4. **Normal Posts**: Ordinary non-threatening social-media content (casual conversations, daily activities, entertainment, sports, education, food, travel, hobbies, general opinions).

### Binary Class Mapping
* **Threat (Class 1)**: Fake News + Hate Speech + Spam (5,000 samples)
* **Normal (Class 0)**: Normal social-media posts (5,000 samples)
* **Total Records**: 10,000 balanced samples

### Preprocessing & Splitting Pipeline
1. **Dataset Integration**: Consolidating multi-source CSV records.
2. **Deduplication**: Removing identical text/label duplicate records.
3. **Text Preprocessing**: Normalizing irregular whitespace, preserving emojis and punctuation.
4. **Outlier Analysis**: Identifying noisy textual patterns (such as extreme text lengths and repeated characters).
5. **Stratified Splitting**:
   - **Training Set (70%)**: 7,000 samples (3,500 Normal, 3,500 Threat)
   - **Validation Set (15%)**: 1,500 samples (750 Normal, 750 Threat)
   - **Test Set (15%)**: 1,500 samples (750 Normal, 750 Threat)

---

## 4. Model Architecture & Training

* **Base Architecture**: `bert-base-uncased` (12 layers, 768 hidden dimensions, 12 attention heads, 110M parameters)
* **Tokenization**: WordPiece tokenizer (30,522 vocabulary size, max sequence length 128)
* **Optimizer**: AdamW (`lr = 2e-5`, `weight_decay = 0.01`)
* **Warmup**: Linear warmup over 10% of total training steps
* **Loss Function**: Cross-Entropy Loss
* **Batch Size**: 16 (Train), 32 (Evaluation)

---

## 5. Model Evaluation Results (Unseen Test Set)

The model achieved the following verified results on the held-out test partition (1,500 samples):

| Metric | Score | Support |
| :--- | :---: | :---: |
| **Test Accuracy** | **88.00%** | 1,500 samples |
| **Threat Precision** | **82.02%** | 750 threat samples |
| **Threat Recall** | **97.33%** | 750 threat samples |
| **Threat F1-Score** | **89.02%** | 750 threat samples |
| **Macro F1-Score** | **87.89%** | 1,500 samples |
| **Weighted F1-Score** | **87.89%** | 1,500 samples |

### Confusion Matrix:
* **True Negatives (Normal $\rightarrow$ Normal)**: 590
* **False Positives (Normal $\rightarrow$ Threat)**: 160
* **False Negatives (Threat $\rightarrow$ Normal)**: 20
* **True Positives (Threat $\rightarrow$ Threat)**: 730

### Outlier Robustness:
* **Regular Test Samples (1,421 samples)**: Accuracy: `88.04%` | Macro F1: `87.92%`
* **Outlier Test Samples (79 samples)**: Accuracy: `87.34%` | Macro F1: `87.32%`

---

## 6. Confidence Calibration

To prevent excessive overconfidence on live predictions, validation-fitted **Temperature Scaling** was applied:
* **Temperature Parameter**: $T = 3.5490$
* **Validation NLL**: Reduced from $0.6961$ to $0.2871$
* **Maximum Validation Confidence**: Bounded at $\le 95.00\%$
* **Mathematical Invariance**: Classification decisions ($\arg\max$) remain identical.

---

## 7. System Deployment & Web Interface

The system is deployed on Render using FastAPI and ONNX Runtime INT8 quantization:
* **Inference Latency**: Sub-100ms CPU inference time.
* **Frontend**: Responsive, clean academic user interface built with semantic HTML, CSS, and vanilla JavaScript.
* **API Endpoints**: `/predict`, `/health`.
