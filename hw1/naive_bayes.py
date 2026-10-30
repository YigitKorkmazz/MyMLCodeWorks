import numpy as np

#alpha =0 : no smoothing, alpha = 1: there is smoothing
def train_naive_bayes(X_train, y_train, alpha=0):
    if alpha < 0:
        raise ValueError("alpha negatif olamaz.")

    number_of_reviews, vocabulary_size = X_train.shape
    priors = np.zeros(2)
    theta = np.zeros((2, vocabulary_size))

    for label in (0, 1):
        class_reviews = X_train[y_train == label]
        word_counts = class_reviews.sum(axis=0)
        total_words = word_counts.sum()
        priors[label] = len(class_reviews) / number_of_reviews

        if alpha == 0:
            theta[label] = word_counts / total_words
        else:
            theta[label] = (word_counts + alpha) / (total_words + alpha * vocabulary_size)

    return theta, priors

#for a class, calculate the log scores of all comments
def calculate_scores(X, word_probabilities, prior):
    log_probabilities = np.zeros_like(word_probabilities)
    positive_words = word_probabilities > 0
    log_probabilities[positive_words] = np.log(word_probabilities[positive_words])

    scores = X @ log_probabilities + np.log(prior)
    zero_words = word_probabilities == 0
    if np.any(zero_words):
        contains_zero_word = np.any(X[:, zero_words] > 0, axis=1)
        scores[contains_zero_word] = -np.inf

    return scores

#if the positive score is larger, predict 1. otherwise, predict 0.
def predict(X, theta, priors):
    negative_scores = calculate_scores(X, theta[0], priors[0])
    positive_scores = calculate_scores(X, theta[1], priors[1])

    predictions = (positive_scores > negative_scores).astype(int)
    return predictions
