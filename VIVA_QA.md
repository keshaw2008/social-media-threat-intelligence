# VIVA VOCE EXAMINATION QUESTIONS & ANSWERS

## Project Title:
# Social Media Threat Intelligence System using BERT

---

### 1. What is the project?
**Answer:**  
The project is an automated Threat Intelligence System that analyzes social-media posts and classifies them into two categories:
1. **Normal (Class 0)**: Everyday conversational posts, hobbies, sports, education, travel, food, and general benign opinions.
2. **Threat (Class 1)**: Malicious posts covering fake news, hate speech, cyber harassment, and financial scams/phishing fraud.

It uses a fine-tuned BERT (`bert-base-uncased`) model deployed via ONNX Runtime INT8 and FastAPI.

---

### 2. What is the dataset source and how was it prepared?
**Answer:**  
The dataset was collected from publicly available Kaggle datasets covering four categories: Fake News, Hate Speech, Spam, and Normal posts. The source data were integrated, cleaned, deduplicated, and mapped into a structured binary classification benchmark of 10,000 balanced samples (5,000 Normal vs. 5,000 Threat).

---

### 3. Why is the system a binary classifier if there are 4 dataset categories?
**Answer:**  
In practical cybersecurity threat intelligence triage, the operational requirement is distinguishing non-threatening benign content (Class 0: Normal) from harmful content requiring moderation or threat escalation (Class 1: Threat). Fake News, Hate Speech, and Spam represent the threat-oriented source categories, which are mapped to Class 1, while Normal Posts represent Class 0.

---

### 4. Why BERT?
**Answer:**  
BERT (**Bidirectional Encoder Representations from Transformers**) evaluates text context simultaneously from left-to-right and right-to-left. Traditional models (such as Naive Bayes or LSTMs) struggle with subtle context, polysemy (words with multiple meanings based on context), and obfuscated phrasing. BERT's self-attention mechanism captures deep semantic interactions across all words in a post.

---

### 5. Why not traditional machine learning (e.g., Naive Bayes or SVM with TF-IDF)?
**Answer:**  
Traditional ML models rely on static bag-of-words or TF-IDF representations, which:
1. Ignore word order, grammar, and sentence structure.
2. Cannot understand context (e.g., distinguishing between quoting a threat warning vs. posting an actual threat).
3. Fail when attackers obfuscate text using leetspeak or altered spelling.

BERT solves these limitations using contextual embeddings.

---

### 6. What is tokenization?
**Answer:**  
Tokenization breaks raw text strings into numerical tokens. BERT uses **WordPiece tokenization**. Common words are preserved as whole tokens, while uncommon or slang words are split into subwords (e.g., `"unbelievable"` $\rightarrow$ `["un", "##believ", "##able"]`). This completely prevents Out-Of-Vocabulary (OOV) errors.

---

### 7. What are input IDs and attention masks?
**Answer:**  
- **Input IDs**: Unique numerical integer indices assigned to each subword token from BERT's 30,522-token vocabulary.
- **Attention Mask**: A binary vector ($1$ for real tokens, $0$ for `[PAD]` padding tokens) that instructs the self-attention mechanism to ignore padding.

---

### 8. What is fine-tuning?
**Answer:**  
Fine-tuning is transfer learning where a pretrained `bert-base-uncased` base encoder is equipped with a linear classification head (2 output units) and trained on our target dataset with a small learning rate ($2 \times 10^{-5}$) using AdamW optimization.

---

### 9. What is Cross-Entropy Loss?
**Answer:**  
Cross-Entropy Loss is the standard objective for classification:
$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{p}_i)$$
It penalizes confident incorrect predictions heavily, driving the network to output high probabilities for the correct class.

---

### 10. What evaluation metrics were achieved?
**Answer:**  
Evaluated on an unseen test set of 1,500 samples:
- **Test Accuracy**: `88.00%`
- **Threat Precision**: `82.02%`
- **Threat Recall**: `97.33%`
- **Threat F1-Score**: `89.02%`
- **Macro F1-Score**: `87.89%`
- **Weighted F1-Score**: `87.89%`

---

### 11. What is outlier analysis in this project?
**Answer:**  
Outlier analysis evaluates how the model performs on noisy, atypical social-media posts (e.g., extreme text lengths, repeated characters, excessive punctuation, and irregular formatting). The model maintained `87.34%` accuracy on noisy outlier samples compared to `88.04%` on regular samples, demonstrating robust generalization.

---

### 12. What is probability calibration / temperature scaling?
**Answer:**  
Modern deep neural networks often output overly confident probabilities (e.g., 99.99%). **Temperature Scaling** divides the unnormalized logit outputs by a learned temperature parameter $T > 1$ before applying softmax:
$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
With $T = 3.5490$ fitted on our validation set, prediction confidence is realistically calibrated without altering the predicted class ($\arg\max$).

---

### 13. How is the system deployed?
**Answer:**  
The model is quantized to **INT8 precision** and converted to **ONNX format** (`model_quantized.onnx`), shrinking model size to ~105MB and RAM usage to ~150MB. It is served via a **FastAPI** REST backend and hosted on **Render** with a clean academic web UI.
