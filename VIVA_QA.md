# VIVA VOCE EXAMINATION QUESTIONS & ANSWERS

## Project Title:
# Social Media Threat Intelligence System using BERT

---

### 1. What is the project?
**Answer:**  
The project is an AI-powered Threat Intelligence System that automatically analyzes social media posts in real-time and classifies them into four categories:
1. **Fake News**: Misinformation, conspiracy theories, and unverified breaking claims.
2. **Hate Speech**: Abusive, discriminatory, and hostile content targeting groups.
3. **Spam**: Fake lotteries, promotional fraud, and phishing links.
4. **Normal**: Harmless daily conversations, hobbies, travel, and general updates.

It uses a fine-tuned Multilingual BERT model to process posts written in English and Arabic.

---

### 2. Why BERT?
**Answer:**  
BERT (**Bidirectional Encoder Representations from Transformers**) reads text in both directions (left-to-right and right-to-left) simultaneously. Traditional models (like RNNs or LSTMs) read text sequentially in one direction. BERT’s self-attention mechanism captures deep semantic context, polysemy (words with multiple meanings based on context), and complex sentence structures much better than older models.

---

### 3. Why multilingual BERT (`bert-base-multilingual-cased`)?
**Answer:**  
Social media platforms are inherently multilingual. A single post or platform feed often contains multiple languages such as English and Arabic. `bert-base-multilingual-cased` was pre-trained on Wikipedia corpora across 104 languages using a shared subword WordPiece vocabulary. This allows one single neural network to understand and classify English and Arabic text without needing separate translation models.

---

### 4. Why not traditional machine learning (e.g., Naive Bayes or SVM)?
**Answer:**  
Traditional ML models rely on **bag-of-words** or **TF-IDF**, which:
1. Ignore word order and grammar.
2. Cannot understand context (e.g., distinguishing between a user quoting a scam warning vs. posting an actual scam).
3. Fail when attackers use typos or leetspeak (e.g., `f@k3`).
4. Require separate manual feature engineering for different languages.

BERT solves all of these limitations using contextual embeddings.

---

### 5. What is tokenization?
**Answer:**  
Tokenization is the process of breaking down raw text strings into smaller units called tokens. BERT uses **WordPiece tokenization**. If a word is common, it stays as a single token; if a word is uncommon, slang, or misspelled, BERT breaks it down into subwords (e.g., `"unbelievable"` $\rightarrow$ `["un", "##believ", "##able"]`). This prevents Out-Of-Vocabulary (OOV) errors.

---

### 6. What are input IDs?
**Answer:**  
Input IDs are unique numerical integers assigned to each token from BERT’s fixed vocabulary table (119,547 tokens for multilingual BERT). Neural networks cannot process text strings directly; they require mathematical tensors, so input IDs represent each token as an index number.

---

### 7. What is an attention mask?
**Answer:**  
Because BERT processes inputs in fixed-length batches (e.g., length 128), shorter sentences are padded with zero tokens (`[PAD]`). The **attention mask** is a binary vector (containing 1s and 0s) that tells the model:
- `1`: This is a real token, compute attention on it.
- `0`: This is artificial padding, ignore it during attention calculation.

---

### 8. What is fine-tuning?
**Answer:**  
BERT is initially pre-trained on massive unlabeled text corpora using Masked Language Modeling. **Fine-tuning** is the transfer learning process where we take this pre-trained base model, attach a new classification head (a linear layer with 4 outputs), and train the entire network with labeled social media data using a small learning rate ($3 \times 10^{-5}$) so it specializes in our specific threat categories.

---

### 9. Why AdamW optimizer?
**Answer:**  
Standard Adam optimizer applies $L_2$ regularization directly to the moving gradients, which distorts weight decay when combined with momentum. **AdamW** decouples weight decay from the gradient updates, providing proper regularization that prevents overfitting and stabilizes transformer training.

---

### 10. What is Cross Entropy Loss?
**Answer:**  
Cross Entropy Loss is the standard loss function for multi-class classification. It compares the model's predicted probability distribution (after Softmax) against the true one-hot encoded ground truth label:
$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
It penalizes confident incorrect predictions heavily, driving the network to output high probabilities for the correct class.

---

### 11. Why F1-score instead of just Accuracy?
**Answer:**  
Accuracy simply measures the overall percentage of correct predictions, which can be misleading if classes are imbalanced or if errors in certain classes have different operational costs. **F1-score** is the harmonic mean of **Precision** (how many predicted threats were real threats) and **Recall** (how many real threats the model caught). Macro F1 ensures all 4 classes are treated with equal importance.

---

### 12. What is an outlier?
**Answer:**  
In NLP and social media, an outlier is an unusual, noisy, or anomalous text sample that deviates from standard grammar and syntax. Examples include:
- Excessive punctuation (`????!!!!`)
- Repeated words (`FREE FREE FREE CASH`)
- Unusual spacing (`W A K E   U P`)
- Mixed symbols/leetspeak (`c0nsp1r@cy`, `$p@m`)
- Extreme post lengths (very short or very long)
- Mixed-language code-switching (English + Arabic in one post)

---

### 13. How are outliers handled in this project?
**Answer:**  
Rather than deleting outliers, we intentionally synthesize and preserve them with an `is_outlier = True` flag. During post-training evaluation, we run a dedicated **Outlier Robustness Analysis** comparing performance on normal test samples versus noisy outlier samples. This measures whether adversarial tricks degrade the model's accuracy.

---

### 14. Why stratified splitting?
**Answer:**  
Stratified splitting ensures that the proportion of each class (and outlier representation) remains identical across the Training (70%), Validation (15%), and Test (15%) splits. It prevents sampling bias where one split might randomly receive too few samples of a particular category.

---

### 15. How does the model classify a post from end-to-end?
**Answer:**  
1. Raw post text enters Unicode-safe cleaning.
2. Tokenizer converts text into `input_ids` and `attention_mask` (max length 128).
3. Tensors pass through BERT's 12 transformer encoder layers.
4. The representation of the `[CLS]` token is passed through a Linear classification head.
5. Softmax converts the 4 raw output logits into percentages/probabilities summing to 1.0.
6. The class with the highest probability is returned as the predicted threat.

---

### 16. How does the system detect Spam?
**Answer:**  
BERT identifies characteristic spam markers: urgent promotional greetings (`URGENT NOTIFICATION`, `CONGRATULATIONS`), unearned rewards (`$1,000 gift card`, `free smartphone`, `1000x crypto profits`), urgent deadlines, and suspicious call-to-action links (`bit.ly`, `claim-reward.xyz`).

---

### 17. How does the system detect Fake News?
**Answer:**  
The model recognizes sensationalist misinformation patterns: claims of secret suppressed cures, hidden 5G or government conspiracies, fabricated authority citations (`leaked documents reveal`, `scientists silenced by media`), and urgent calls to share before deletion (`Share before it's deleted!`).

---

### 18. How does the system detect Hate Speech?
**Answer:**  
The model captures dehumanizing language targeting demographic groups (immigrants, refugees, minorities), violent expulsion rhetoric (`deport them all`, `cleanse our nation`), and derogatory stereotypes.

---

### 19. How does the system identify Normal posts?
**Answer:**  
Normal posts exhibit casual, conversational language discussing everyday life: university studies, exams, cooking, workouts, weather, music, hobbies, and spending time with friends, lacking aggressive calls to action, hate rhetoric, or scam hooks.

---

### 20. What are the limitations of synthetic data?
**Answer:**  
While synthetic data allows us to generate balanced, safe, and controlled academic datasets with explicit ground truths, it may not encompass every emergent real-world slang term, subtle irony, cultural sarcasm, or newly evolving disinformation narratives found on live social platforms.

---

### 21. What happens if the input is Arabic?
**Answer:**  
The system handles Arabic natively. Our preprocessing is Unicode NFC safe and does not strip Arabic script or diacritics. Because `bert-base-multilingual-cased` was pre-trained on Arabic Wikipedia and our dataset contains Arabic samples across all categories, the model extracts semantic representations from Arabic text just as effectively as English.

---

### 22. What happens if the input is very short (e.g., 1 or 2 words)?
**Answer:**  
Very short inputs (like `"win"` or `"ok"`) contain minimal contextual clues. BERT will tokenize the single word and pad the remaining sequence with `[PAD]` tokens. The model will make a prediction based on prior associations of that word, but confidence may be lower compared to complete sentences.

---

### 23. What happens if the model encounters an unseen type of threat?
**Answer:**  
BERT generalizes well to semantic similarities through its deep embeddings. However, for a completely novel threat category (such as a brand-new cyberattack vector never seen in training), the model will map the post to the closest existing category based on lexical overlap or assign a dispersed probability distribution with low confidence.

---

### 24. How can this system be improved in the future?
**Answer:**  
1. **Multimodal Analysis**: Integrating Computer Vision (e.g., Vision Transformers or CLIP) to analyze images, memes, and videos alongside text.
2. **Explainability (XAI)**: Adding Integrated Gradients, SHAP, or Attention Visualizers to highlight the exact keywords that triggered the classification.
3. **Adversarial Hardening**: Training with back-translation and character-level perturbations to improve resilience against evasive spelling.
4. **Model Quantization**: Exporting to ONNX / TensorRT with FP16 or INT8 precision for ultra-low latency real-time deployment.
