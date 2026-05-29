import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import sys

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# 1. Text Preprocessing & Tokenization Utilities
# -----------------------------------------------------------------------------

def clean_text(text):
    """
    Cleans raw text by converting it to lowercase, removing HTML tags,
    removing punctuation and numbers, and splitting into individual words.
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags (e.g. <br /><br />)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove non-alphabetic characters (numbers, punctuation, symbols)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize by splitting on whitespace
    words = text.split()
    return words

# -----------------------------------------------------------------------------
# 2. Text Vectorizer (Bag-of-Words Representation)
# -----------------------------------------------------------------------------

class TextVectorizer:
    """
    Fits a vocabulary on training text and transforms documents into
    Bag-of-Words (BoW) count tensors.
    """
    def __init__(self, max_features=5000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.vocab_size = 0

    def fit(self, texts):
        """
        Builds the vocabulary from a list of documents.
        Filters out rare words based on min_df and limits total words to max_features.
        """
        word_counts = {}
        doc_counts = {}
        
        # Count word occurrences and document frequencies
        for text in texts:
            words = clean_text(text)
            unique_words = set(words)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
            for w in unique_words:
                doc_counts[w] = doc_counts.get(w, 0) + 1

        # Keep words that appear in at least min_df documents
        filtered_words = [w for w, doc_freq in doc_counts.items() if doc_freq >= self.min_df]
        
        # Sort words by total frequency to extract top max_features
        sorted_words = sorted(filtered_words, key=lambda w: word_counts[w], reverse=True)
        
        if self.max_features is not None:
            sorted_words = sorted_words[:self.max_features]
            
        # Create mapping from word to unique index
        self.vocab = {word: idx for idx, word in enumerate(sorted_words)}
        self.vocab_size = len(self.vocab)
        
    def transform(self, texts):
        """
        Transforms a list of documents into a PyTorch FloatTensor of shape (num_docs, vocab_size),
        where each row represents the word counts (Bag-of-Words) of a document.
        """
        num_docs = len(texts)
        bow_tensor = torch.zeros((num_docs, self.vocab_size), dtype=torch.float32)
        
        for doc_idx, text in enumerate(texts):
            words = clean_text(text)
            for w in words:
                if w in self.vocab:
                    word_idx = self.vocab[w]
                    bow_tensor[doc_idx, word_idx] += 1.0
                    
        return bow_tensor

# -----------------------------------------------------------------------------
# 3. Multinomial Naive Bayes Classifier
# -----------------------------------------------------------------------------

class PyTorchMultinomialNB(nn.Module):
    """
    A Multinomial Naive Bayes classifier implemented with PyTorch.
    Suitable for discrete counts (e.g. term frequency bag-of-words).
    
    Model parameters are calculated analytically and stored as PyTorch buffers.
    Inference is fully vectorized using PyTorch tensor operations.
    """
    def __init__(self, num_classes=2, vocab_size=5000, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.alpha = alpha  # Laplace smoothing parameter
        
        # Register buffers so they are part of the module state but not tracked by autograd
        self.register_buffer("class_log_prior", torch.zeros(num_classes))
        self.register_buffer("feature_log_prob", torch.zeros((num_classes, vocab_size)))
        
    def fit(self, X, y):
        """
        X: torch.Tensor of shape (num_samples, vocab_size) - word frequency counts
        y: torch.Tensor of shape (num_samples,) - label indices [0, num_classes - 1]
        """
        num_samples = X.shape[0]
        class_counts = torch.zeros(self.num_classes, device=X.device)
        word_counts_per_class = torch.zeros((self.num_classes, self.vocab_size), device=X.device)
        
        for c in range(self.num_classes):
            class_mask = (y == c)
            class_counts[c] = class_mask.sum().float()
            word_counts_per_class[c] = X[class_mask].sum(dim=0)
            
        # P(c) = count(c) / total_samples
        self.class_log_prior.copy_(torch.log(class_counts / num_samples))
        
        # P(w_i | c) = (count(w_i, c) + alpha) / (total_words_in_class_c + alpha * vocab_size)
        total_words_in_class = word_counts_per_class.sum(dim=1, keepdim=True)
        smoothed_prob = (word_counts_per_class + self.alpha) / (total_words_in_class + self.alpha * self.vocab_size)
        self.feature_log_prob.copy_(torch.log(smoothed_prob))
        
    def forward(self, X):
        """
        Computes the log posterior probability: log P(c) + sum_i (x_i * log P(w_i | c))
        """
        return self.class_log_prior + X @ self.feature_log_prob.T
        
    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)

# -----------------------------------------------------------------------------
# 4. Bernoulli Naive Bayes Classifier
# -----------------------------------------------------------------------------

class PyTorchBernoulliNB(nn.Module):
    """
    A Bernoulli Naive Bayes classifier implemented with PyTorch.
    Suitable for binary features (word presence vs. absence).
    
    Model parameters are calculated analytically and stored as PyTorch buffers.
    Inference is fully vectorized using PyTorch tensor operations.
    """
    def __init__(self, num_classes=2, vocab_size=5000, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.alpha = alpha  # Laplace smoothing parameter
        
        self.register_buffer("class_log_prior", torch.zeros(num_classes))
        self.register_buffer("log_p_present", torch.zeros((num_classes, vocab_size)))
        self.register_buffer("log_p_absent", torch.zeros((num_classes, vocab_size)))
        
    def fit(self, X, y):
        """
        X: torch.Tensor of shape (num_samples, vocab_size) - word frequency or binary presence
        y: torch.Tensor of shape (num_samples,) - label indices [0, num_classes - 1]
        """
        num_samples = X.shape[0]
        X_bin = (X > 0).float()
        
        class_counts = torch.zeros(self.num_classes, device=X.device)
        doc_counts_per_class = torch.zeros((self.num_classes, self.vocab_size), device=X.device)
        
        for c in range(self.num_classes):
            class_mask = (y == c)
            class_counts[c] = class_mask.sum().float()
            doc_counts_per_class[c] = X_bin[class_mask].sum(dim=0)
            
        # P(c) = count(c) / total_samples
        self.class_log_prior.copy_(torch.log(class_counts / num_samples))
        
        # P(w_i present | c) = (docs_in_c_with_w_i + alpha) / (docs_in_c + 2 * alpha)
        p_present = (doc_counts_per_class + self.alpha) / (class_counts.unsqueeze(1) + 2 * self.alpha)
        
        self.log_p_present.copy_(torch.log(p_present))
        self.log_p_absent.copy_(torch.log(1.0 - p_present))
        
    def forward(self, X):
        """
        Computes the log posterior probability for Bernoulli Naive Bayes.
        """
        X_bin = (X > 0).float()
        sum_log_p_absent = self.log_p_absent.sum(dim=1)
        log_likelihood = X_bin @ (self.log_p_present - self.log_p_absent).T
        return self.class_log_prior + log_likelihood + sum_log_p_absent
        
    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)

# -----------------------------------------------------------------------------
# 5. K-Fold Cross Validation Comparison & Main Loop
# -----------------------------------------------------------------------------

def run_cross_validation_comparison(data_path, K=5, vocab_size=5000):
    print(f"Loading dataset from: {data_path}...")
    df = pd.read_csv(data_path, sep='\t')
    df = df.dropna(subset=['sentiment', 'text'])
    
    texts = df['text'].values
    labels = (df['sentiment'].values == 1).astype(int)
    
    num_samples = len(texts)
    print(f"Loaded {num_samples} reviews. Sentiments: Positive={sum(labels == 1)}, Negative={sum(labels == 0)}")
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    fold_size = num_samples // K
    
    print(f"\nStarting {K}-Fold Cross Validation Comparison...")
    print(f"Vocabulary Size: {vocab_size} | Samples per fold: ~{fold_size}\n")
    
    multinomial_accuracies = []
    bernoulli_accuracies = []
    
    for fold in range(K):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < K - 1 else num_samples
        
        val_indices = indices[start_idx:end_idx]
        train_indices = torch.cat([indices[:start_idx], indices[end_idx:]])
        
        # Split texts
        train_texts = [texts[idx] for idx in train_indices.tolist()]
        val_texts = [texts[idx] for idx in val_indices.tolist()]
        
        train_labels = torch.tensor([labels[idx] for idx in train_indices.tolist()], dtype=torch.long)
        val_labels = torch.tensor([labels[idx] for idx in val_indices.tolist()], dtype=torch.long)
        
        # Fit vectorizer on train fold only
        vectorizer = TextVectorizer(max_features=vocab_size, min_df=2)
        vectorizer.fit(train_texts)
        
        # Transform texts into Bag-of-Words count tensors
        X_train = vectorizer.transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        
        # Evaluate Multinomial Naive Bayes
        mnb_model = PyTorchMultinomialNB(num_classes=2, vocab_size=vectorizer.vocab_size, alpha=1.0)
        mnb_model.fit(X_train, train_labels)
        mnb_model.eval()
        with torch.no_grad():
            mnb_preds = mnb_model.predict(X_val)
            mnb_acc = (mnb_preds == val_labels).sum().item() / len(val_labels)
            multinomial_accuracies.append(mnb_acc)
            
        # Evaluate Bernoulli Naive Bayes
        bnb_model = PyTorchBernoulliNB(num_classes=2, vocab_size=vectorizer.vocab_size, alpha=1.0)
        bnb_model.fit(X_train, train_labels)
        bnb_model.eval()
        with torch.no_grad():
            bnb_preds = bnb_model.predict(X_val)
            bnb_acc = (bnb_preds == val_labels).sum().item() / len(val_labels)
            bernoulli_accuracies.append(bnb_acc)
            
        print(f"Fold {fold+1}/{K} | Vocabulary: {vectorizer.vocab_size:4d} | Multinomial Acc: {mnb_acc * 100:6.2f}% | Bernoulli Acc: {bnb_acc * 100:6.2f}%")
        
    avg_mnb = np.mean(multinomial_accuracies)
    std_mnb = np.std(multinomial_accuracies)
    
    avg_bnb = np.mean(bernoulli_accuracies)
    std_bnb = np.std(bernoulli_accuracies)
    
    print("\n" + "=" * 65)
    print("CROSS VALIDATION COMPARISON SUMMARY")
    print("=" * 65)
    print(f"Model                  | Average Accuracy | Standard Deviation")
    print("-" * 65)
    print(f"Multinomial Naive Bayes| {avg_mnb * 100:15.2f}% | {std_mnb * 100:17.2f}%")
    print(f"Bernoulli Naive Bayes  | {avg_bnb * 100:15.2f}% | {std_bnb * 100:17.2f}%")
    print("=" * 65)
    
    # Train final models on the entire dataset for interactive prediction
    print("\nTraining final models on the entire dataset for predictions...")
    final_vectorizer = TextVectorizer(max_features=vocab_size, min_df=2)
    final_vectorizer.fit(texts)
    X_all = final_vectorizer.transform(texts)
    final_labels = torch.tensor(labels, dtype=torch.long)
    
    final_mnb = PyTorchMultinomialNB(num_classes=2, vocab_size=final_vectorizer.vocab_size, alpha=1.0)
    final_mnb.fit(X_all, final_labels)
    
    final_bnb = PyTorchBernoulliNB(num_classes=2, vocab_size=final_vectorizer.vocab_size, alpha=1.0)
    final_bnb.fit(X_all, final_labels)
    
    sentiment_map = {0: "NEGATIVE 🔴", 1: "POSITIVE 🟢"}
    
    # Detect if we are running in an interactive TTY
    is_interactive = sys.stdin.isatty() or "--interactive" in sys.argv
    
    if is_interactive:
        print("\n" + "=" * 65)
        print("INTERACTIVE SENTIMENT PREDICTION")
        print("=" * 65)
        print("Type a review sentence to predict its sentiment.")
        print("Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_input = input("\nEnter a sentence: ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    print("Exiting interactive prediction. Goodbye!")
                    break
                    
                if not user_input.strip():
                    continue
                    
                X_input = final_vectorizer.transform([user_input])
                
                # Predict using Multinomial NB
                final_mnb.eval()
                with torch.no_grad():
                    mnb_log_post = final_mnb(X_input)
                    mnb_pred = torch.argmax(mnb_log_post, dim=1).item()
                    mnb_conf = torch.softmax(mnb_log_post, dim=1)[0]
                    
                # Predict using Bernoulli NB
                final_bnb.eval()
                with torch.no_grad():
                    bnb_log_post = final_bnb(X_input)
                    bnb_pred = torch.argmax(bnb_log_post, dim=1).item()
                    bnb_conf = torch.softmax(bnb_log_post, dim=1)[0]
                    
                print("-" * 65)
                print(f"Multinomial NB : {sentiment_map[mnb_pred]} (Confidence: {mnb_conf[mnb_pred].item()*100:.2f}%)")
                print(f"Bernoulli NB   : {sentiment_map[bnb_pred]} (Confidence: {bnb_conf[bnb_pred].item()*100:.2f}%)")
                print("-" * 65)
                
            except KeyboardInterrupt:
                print("\nExiting interactive prediction. Goodbye!")
                break
    else:
        # Non-interactive mode (e.g. running in testing pipelines)
        print("\nRunning in non-interactive mode. Displaying sample predictions:")
        test_samples = [
            "Amazing food!!!!."
        ]
        
        for sample in test_samples:
            X_input = final_vectorizer.transform([sample])
            
            with torch.no_grad():
                # Multinomial NB
                mnb_log_post = final_mnb(X_input)
                mnb_pred = torch.argmax(mnb_log_post, dim=1).item()
                mnb_conf = torch.softmax(mnb_log_post, dim=1)[0]
                
                # Bernoulli NB
                bnb_log_post = final_bnb(X_input)
                bnb_pred = torch.argmax(bnb_log_post, dim=1).item()
                bnb_conf = torch.softmax(bnb_log_post, dim=1)[0]
                
            print("-" * 65)
            print(f"Input sentence : \"{sample}\"")
            print(f"Multinomial NB : {sentiment_map[mnb_pred]} (Confidence: {mnb_conf[mnb_pred].item()*100:.2f}%)")
            print(f"Bernoulli NB   : {sentiment_map[bnb_pred]} (Confidence: {bnb_conf[bnb_pred].item()*100:.2f}%)")
        print("-" * 65)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "reviews.tsv")
    
    run_cross_validation_comparison(data_file, K=5, vocab_size=5000)
