# ACADEMIC PROJECT REPORT

## PROJECT TITLE:
# Social Media Threat Intelligence System using BERT

**Academic Level:** Bachelor of Technology (B.Tech) Capstone Project  
**Domain:** Artificial Intelligence, Natural Language Processing, Cyber Threat Intelligence  
**Model Architecture:** `bert-base-multilingual-cased` (Multilingual BERT)  
**Target Languages:** English and Arabic  
**Target Classes:** Fake News (0), Hate Speech (1), Spam (2), Normal (3)  

---

## 1. Abstract

The explosive expansion of social media networks has democratized information exchange while simultaneously amplifying severe digital risks, such as viral fake news, toxic hate speech, predatory financial spam, and phishing schemes. Traditional cybersecurity solutions relying on regex pattern matching, blacklists, and bag-of-words machine learning algorithms fall short in capturing contextual nuances, multilingual code-switching, and adversarial variations. 

This project presents the **Social Media Threat Intelligence System using BERT**, an end-to-end deep learning framework designed to process and classify multilingual social media posts into four key categories: Fake News, Hate Speech, Spam, and Normal content. The core classifier utilizes `bert-base-multilingual-cased`, fine-tuned using PyTorch and Hugging Face Transformers. The system processes a curated, reproducible 10,000-sample dataset encompassing English and Arabic entries with intentional adversarial outliers. An evaluation framework rigorously tests overall classification metrics and outlier robustness. Finally, an interactive Gradio web application is deployed to provide real-time threat categorization, confidence scores, and probability distributions.

---

## 2. Introduction

Online social media platforms such as X (formerly Twitter), Facebook, Reddit, and Telegram host billions of active users posting in dozens of languages daily. While these platforms facilitate real-time communication, they are frequently weaponized by malicious actors to:
- Disseminate medical and political disinformation (Fake News).
- Harass, demean, and incite hatred against vulnerable demographic groups (Hate Speech).
- Deploy automated commercial spam, lottery fraud, and malicious phishing hyperlinks (Spam).

Manual content moderation is mentally exhausting, expensive, and unable to scale with real-time data ingestion rates. Automated natural language processing systems powered by deep contextual language models offer a scalable and consistent defense mechanism.

---

## 3. Problem Statement

Automated threat detection in social media faces several critical hurdles:
1. **Multilingual and Cross-Lingual Nuances**: Attackers exploit multiple languages, especially Arabic and English, where non-Latin characters require Unicode-safe handling.
2. **Contextual Polysemy**: Words change meaning drastically depending on surrounding words, making shallow n-gram or TF-IDF models prone to false positives.
3. **Adversarial Noise and Outliers**: Social media text is inherently noisy, containing intentional character substitutions (leetspeak), excessive exclamation marks, repeated words, and ALL-CAPS text designed to bypass naive heuristic filters.
4. **Data Leakage in Research**: Traditional data splitting frequently duplicates identical templates across training and testing splits, artificially inflating reported model accuracy.

---

## 4. Existing System

Traditional threat detection implementations typically utilize:
- **Rule-Based & Keyword Blacklists**: Simple regex checks for banned terms or known spam URLs.
- **Traditional Machine Learning**: Naive Bayes, Support Vector Machines (SVM), and Random Forests using TF-IDF or CountVectorizer representations.
- **Monolingual Word Embeddings**: Word2Vec or GloVe combined with LSTM or BiLSTM networks.

### Disadvantages of Existing Systems:
- Inability to understand bidirectional sentence context.
- Keyword filters are easily bypassed by replacing letters with digits (e.g., `f@k3`, `c1ick`).
- Word embeddings assign static vector representations regardless of context.
- Monolingual designs fail entirely when posts contain foreign scripts like Arabic.

---

## 5. Proposed System

The proposed system adopts a modern Transformer-based deep learning architecture using **Multilingual BERT (`bert-base-multilingual-cased`)**. 

### Key Innovations:
1. **Bidirectional Deep Representations**: Evaluates text context simultaneously from left-to-right and right-to-left across 12 transformer layers.
2. **Native Multilingual Vocabulary**: Employs a 119,547-token WordPiece vocabulary covering Latin, Arabic, and 100+ global languages natively.
3. **Unicode-Safe Preprocessing**: Implements NFC normalization, preserving Arabic characters, punctuation, and structural cues without information loss.
4. **Outlier Robustness Auditing**: Evaluates performance on clean versus adversarial inputs to assess practical reliability.
5. **Interactive Real-Time Portal**: Deploys a user-friendly Gradio web portal with instant visual threat categorization.

---

## 6. Objectives

1. Synthesize a balanced, realistic, and reproducible 10,000-sample social media threat dataset (2,500 samples per class) with intentional outliers in English and Arabic.
2. Build an audit-driven preprocessing pipeline with zero text leakage across 70% Train, 15% Validation, and 15% Test splits.
3. Fine-tune `bert-base-multilingual-cased` using PyTorch, AdamW optimizer, and CrossEntropyLoss.
4. Measure precision, recall, F1-score, and confusion matrices across all classes.
5. Perform an explicit Outlier Robustness Analysis comparing standard test posts against noisy outlier samples.
6. Package the trained model into a production-ready Gradio UI.

---

## 7. Dataset Description

The project utilizes a synthetic social media dataset synthesized with high combinatorial diversity to simulate authentic digital threats:

| Threat Category | Label ID | Core Themes | Sample Count |
| :--- | :---: | :--- | :---: |
| **Fake News** | 0 | Secret cures, 5G conspiracies, unverified leaks, fake authorities | 2,500 |
| **Hate Speech** | 1 | Xenophobic tropes, migrant hostility, group discrimination | 2,500 |
| **Spam** | 2 | Fake gift cards, crypto pump scams, prize draws, phishing links | 2,500 |
| **Normal** | 3 | University life, hobbies, cooking, travel, technology discussions | 2,500 |
| **Total** | — | **Balanced 4-Class Distribution** | **10,000** |

### Language Breakdown:
- **English Samples**: ~82.5% (8,255 rows)
- **Arabic Samples**: ~17.5% (1,745 rows), with heavy representation in Spam and balanced presence across all categories.

---

## 8. Data Preprocessing

A conservative, semantic-preserving preprocessing pipeline is employed:
- **Unicode NFC Normalization**: Ensures consistent byte representation for Arabic characters and ligatures.
- **Control Character Removal**: Strips non-printing zero-width characters (`\u200B-\u200D\uFEFF`).
- **Whitespace Regularization**: Consolidates excessive tab and space repetitions.
- **Preservation of Contextual Tokens**: Punctuation, currency signs (`$`, `USD`), URLs, and emojis are retained, as they provide critical signals for spam and threat detection.
- **Deduplication Audit**: Stringent verification guarantees that no identical post appears in multiple splits.

---

## 9. Outlier Analysis

Social media threats frequently manifest as adversarial outliers designed to game automated systems. The dataset incorporates 15 intentional outliers per class (60 total) exhibiting:
1. **Excessive Punctuation**: Multiple question/exclamation marks (`????!!!!`).
2. **Word Repetition**: Stacking keywords (`FREE FREE FREE CASH NOW`).
3. **Unusual Spacing**: Inserting spaces between letters (`W A K E   U P`).
4. **Symbolic Substitution (Leetspeak)**: Replacing letters with numbers/symbols (`c0nsp1r@cy`, `$p@m`).
5. **Length Extremes**: Very short posts (`win`, `ok`) or long run-on rants (>400 characters).
6. **Code-Switching**: Combining Arabic and English in a single sentence.

---

## 10. BERT Architecture

BERT (Bidirectional Encoder Representations from Transformers) relies on the Transformer encoder mechanism:
- **Layers**: 12 Transformer Encoder blocks.
- **Hidden Dimension**: 768 dimensions per token representation.
- **Attention Heads**: 12 self-attention heads per layer.
- **Vocabulary Size**: 119,547 tokens (WordPiece multilingual cased).
- **Self-Attention Formulation**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The `[CLS]` token prepended to every input sequence aggregates sentence-level bidirectional representations, which are passed through a dropout layer and linear classification projection to yield logits for the 4 threat classes.

---

## 11. System Architecture

```text
               User / Platform Input
                         │
                         ▼
             Unicode-Safe Preprocessor
                         │
                         ▼
           Multilingual WordPiece Tokenizer
            (Tokens -> Input IDs + Attention Mask)
                         │
                         ▼
             BERT-base-multilingual-cased
           (12 Layers, 12 Heads, 768 Hidden Dim)
                         │
                         ▼
               Classification Head (Dense)
                         │
                         ▼
                  Softmax Function
                         │
                         ▼
        Predicted Class, Confidence, Probabilities
                         │
                         ▼
           Gradio GUI / REST Prediction Engine
```

---

## 12. Methodology

1. **Generation & Verification**: Generate 10,000 synthetic posts with a fixed seed (`seed=42`).
2. **Quality Auditing**: Audit for null values, empty strings, and duplicates.
3. **Stratified Splitting**: Split data into Train (70%), Validation (15%), and Test (15%) preserving class balance.
4. **Tokenization**: Tokenize using PyTorch DataLoader with `max_length=128`.
5. **Fine-Tuning**: Update weights using AdamW optimizer with warmup and CrossEntropyLoss.
6. **Validation & Checkpointing**: Save the best checkpoint based on validation accuracy.
7. **Evaluation & Reporting**: Compute complete metrics, confusion matrix, and outlier analysis.
8. **Deployment**: Expose model via Gradio web interface.

---

## 13. Training Configuration

| Parameter | Configuration Value | Rationale |
| :--- | :--- | :--- |
| **Model** | `bert-base-multilingual-cased` | Native cross-lingual support for EN & AR |
| **Sequence Length** | 128 tokens | Balances social post context and memory efficiency |
| **Optimizer** | AdamW | Decoupled weight decay prevents overfitting |
| **Learning Rate** | $3 \times 10^{-5}$ | Optimal fine-tuning rate for BERT encoders |
| **Weight Decay** | 0.01 | Regularization on non-bias parameters |
| **Batch Size** | 32 | Memory-efficient mini-batching |
| **Epochs** | 3 | Sufficient for convergence without catastrophic forgetting |
| **Warmup Ratio** | 10% of total steps | Prevents disruptive gradient updates in early steps |

---

## 14. Evaluation Metrics

Model performance is evaluated across:
- **Accuracy**: Overall proportion of correctly predicted posts.
- **Precision**: $\frac{TP}{TP + FP}$ (Exactness of positive threat predictions).
- **Recall**: $\frac{TP}{TP + FN}$ (Completeness of threat identification).
- **Macro F1-Score**: Harmonic mean of Precision and Recall averaged equally across all classes.
- **Weighted F1-Score**: F1-score weighted by support per class.
- **Confusion Matrix**: Multi-class contingency table showing true vs. predicted classifications.

---

## 15. Results

The fine-tuned multilingual BERT model achieved high classification performance across all four classes on the independent 1,500-sample test set. 

*(Exact numerical metrics, per-class F1-scores, and confusion matrices are dynamically generated and recorded in `results/metrics.json` and `results/classification_report.txt`).*

---

## 16. Robustness Analysis

The outlier evaluation evaluates model reliability against noise:
- **Group A (All Test Posts)**: Baseline benchmark across the complete test population.
- **Group B (Clean Posts)**: Benchmark on standard posts with conventional syntax.
- **Group C (Outlier Posts)**: Stress test on posts with extreme punctuation, leetspeak, and repetition.

### Key Observation:
While BERT's subword tokenization allows it to parse fragmented words effectively, extreme token repetitions and non-standard spacing can shift attention weights. Recording this delta provides realistic insights into production deployment constraints.

---

## 17. Applications

1. **Social Media Content Moderation**: Automated queue filtering for Trust & Safety teams.
2. **Threat Intelligence Monitoring**: Early warning systems for disinformation and coordinated harassment campaigns.
3. **Enterprise Brand Defense**: Detecting fraudulent brand impersonation and scam accounts targeting customers.
4. **Public Health Misinformation Tracking**: Identifying viral fabricated health claims during crises.

---

## 18. Advantages

- **True Multilingual Capability**: Seamlessly classifies English and Arabic without needing separate language-specific models.
- **Context-Aware**: Distinguishes between benign mentions of sensitive topics and malicious intent.
- **Zero Data Leakage**: Guaranteed integrity through string deduplication before stratified splitting.
- **Modular & Deployable**: Clean separation between data generation, preprocessing, training, evaluation, and GUI serving.

---

## 19. Limitations

1. **Synthetic Nature of Training Data**: Synthetic templates, while realistic, cannot cover every emergent slang or evolving internet meme.
2. **Text-Only Scope**: Does not process images, video clips, or audio attachments.
3. **Fixed Sequence Length**: Posts exceeding 128 subword tokens are truncated.
4. **Computational Requirements**: BERT inference requires more computational resources than simple linear models.

---

## 20. Future Scope

1. **Multimodal Expansion**: Incorporating Vision Transformers (ViT) or CLIP for meme and image-text threat detection.
2. **Explainability Frameworks**: Implementing LIME, SHAP, or Attention Visualizers to highlight specific threatening tokens.
3. **Adversarial Hardening**: Applying automated adversarial perturbations during training.
4. **Edge Deployment**: Quantizing the model into ONNX or TensorRT format for mobile and low-latency edge deployment.

---

## 21. Conclusion

The **Social Media Threat Intelligence System using BERT** successfully provides a robust, multilingual deep learning solution for modern digital threat classification. By employing `bert-base-multilingual-cased`, Unicode-safe data engineering, and rigorous outlier testing, the system demonstrates strong potential in categorizing Fake News, Hate Speech, Spam, and Normal content across English and Arabic. The resulting open-source pipeline and Gradio user interface provide an educational, practical, and extensible foundation for next-generation content moderation systems.
