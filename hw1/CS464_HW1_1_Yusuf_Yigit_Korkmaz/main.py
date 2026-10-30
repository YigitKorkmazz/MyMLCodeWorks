from pathlib import Path

import numpy as np
import pandas as pd

from naive_bayes import train_naive_bayes, predict


DATA_DIR = Path(__file__).resolve().parent / "data"

#read the data from csv files
def load_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "x_train.csv")
    val = pd.read_csv(data_dir / "x_val.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv", header=None).to_numpy().ravel()
    y_val = pd.read_csv(data_dir / "y_val.csv", header=None).to_numpy().ravel()

    if not train.columns.equals(val.columns):
        raise ValueError("Training and validation vocabularies must match in order.")

    X_train = train.to_numpy()
    X_val = val.to_numpy()
    return X_train, y_train, X_val, y_val

#negative and positive comment number
def print_class_distribution(name, labels):
    negative_count = np.sum(labels == 0)
    positive_count = np.sum(labels == 1)
    total = len(labels)

    print(f"\n{name} class distribution (N={total}):")
    print(f"  negative (0): {negative_count} ({100 * negative_count / total:.4f}%)")
    print(f"  positive (1): {positive_count} ({100 * positive_count / total:.4f}%)")

#compare the estimated labels with the actual ones
def evaluate(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    incorrect = np.sum(y_true != y_pred)

    #confusion matrix
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    confusion_matrix = np.array([
        [true_negative, false_positive],
        [false_negative, true_positive],
    ])
    return {"accuracy": accuracy, "incorrect": incorrect, "confusion_matrix": confusion_matrix}


def run_model(X_train, y_train, X_val, y_val, alpha):
    theta, priors = train_naive_bayes(X_train, y_train, alpha=alpha)
    predictions = predict(X_val, theta, priors)
    results = evaluate(y_val, predictions)

    if alpha == 0:
        print("\nUnsmoothed model (alpha=0):")
    else:
        print(f"\nSmoothed model (alpha={alpha}):")

    print(f"  Training priors: pi_0={priors[0]:.4f}, pi_1={priors[1]:.4f}")
    print(f"  Validation accuracy: {results['accuracy']:.4f}")
    print(f"  Incorrect predictions: {results['incorrect']}")
    print("  Confusion matrix (rows=true, columns=predicted; order=0,1):")
    print(results["confusion_matrix"])
    print("  Zero-probability words (negative):", np.sum(theta[0] == 0))
    print("  Zero-probability words (positive):", np.sum(theta[1] == 0))
    return results


def main():
    #read the data and print the class distribution
    X_train, y_train, X_val, y_val = load_data()
    print(f"Training shape: {X_train.shape}; validation shape: {X_val.shape}")
    print_class_distribution("Training", y_train)
    print_class_distribution("Validation", y_val)
    print("\nTraining token totals:")
    print("  negative:", X_train[y_train == 0].sum())
    print("  positive:", X_train[y_train == 1].sum())

    # alpha = 0 means no smoothing
    unsmoothed = run_model(X_train, y_train, X_val, y_val, alpha=0)

    # alpha = 1 means there is laplace smoothing
    smoothed = run_model(X_train, y_train, X_val, y_val, alpha=1)

    # compare the accuracy of both models
    difference = 100 * (unsmoothed["accuracy"] - smoothed["accuracy"])
    print(f"\nAccuracy difference (unsmoothed - smoothed): {difference:.4f} percentage points")


if __name__ == "__main__":
    main()
