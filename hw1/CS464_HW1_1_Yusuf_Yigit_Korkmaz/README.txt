CS464 Homework 1 - IMDb Movie Review Classification
Yusuf Yigit Korkmaz | Student ID: 22201749 | Section: 1

ENVIRONMENT
Python 3.14.3, NumPy 2.4.6, pandas 3.0.3

To install the tested package versions in your Python environment:
    python3 -m pip install numpy==2.4.6 pandas==3.0.3

RUN
From the directory containing main.py:
    python3 main.py


The data path is resolved relative to main.py, so the working directory
does not affect it. The CSV files belong in the data/ folder next to main.py.

Optional commands for running one model:
    python3 unsmoothed.py
    python3 smoothed.py

CODE STRUCTURE AND SMOOTHING
main.py reads the data, runs both models, and prints their results.
naive_bayes.py contains training and prediction; keep it next to main.py.

The two model calls are written explicitly inside main():
    unsmoothed = run_model(X_train, y_train, X_val, y_val, alpha=0)
    smoothed = run_model(X_train, y_train, X_val, y_val, alpha=1)

The alpha argument is passed to train_naive_bayes() in naive_bayes.py:
    alpha=0: theta = word_counts / total_words (no smoothing).
    alpha=1: theta = (word_counts + 1) / (total_words + vocabulary_size).
The second case is Laplace smoothing. Both models use the same predict()
function; the choice affects training probabilities, not the scoring rule.

Read the functions in this order:
    load_data() -> run_model() -> train_naive_bayes() -> predict() -> evaluate()
theta[0] stores negative-class word probabilities; theta[1] stores positive.
priors[0] and priors[1] store the corresponding class prior probabilities.

DATA LAYOUT
    main.py
    naive_bayes.py
    unsmoothed.py
    smoothed.py
    README.txt
    data/
        x_train.csv
        y_train.csv
        x_val.csv
        y_val.csv
        README.txt

Feature CSVs contain vocabulary headers; label CSVs have no header.
The supplied 1600/400 split and vocabulary order are preserved.
Only training data is used to estimate class priors and word probabilities.

DEFAULT PARAMETERS AND CONVENTIONS
Both models run when main.py is executed; no command-line options are needed.
Unsmoothed model: alpha=0. Laplace model: alpha=1. Vocabulary: 2000 words.
The same empirical training priors are used by both models.
Natural logarithms are used. Absent words contribute zero, including when
theta=0. Observing a word with theta=0 gives a class score of -infinity.
Every tie, including two -infinity scores, predicts label 0 (negative).
Confusion matrices use true labels as rows and predicted labels as columns,
with negative (0) first and positive (1) second.

OUTPUT
The program prints data shapes, class counts and percentages in both
partitions, training priors, and training token totals. For each model it
prints validation accuracy to four decimal places, the number of incorrect
predictions, the confusion matrix, and zero-probability vocabulary counts.
Finally, it prints the accuracy difference in percentage points.
All results are computed from the CSV files at runtime.

Expected results for the supplied data:
    Training:   800 negative (50.00%), 800 positive (50.00%).
    Validation: 200 negative (50.00%), 200 positive (50.00%).
    Priors: pi_0=0.5000, pi_1=0.5000.
    Unsmoothed: accuracy=0.8225, incorrect=71, matrix=[[171,29],[42,158]].
                Zero-probability words: negative=5, positive=7.
    Laplace:    accuracy=0.8200, incorrect=72, matrix=[[171,29],[43,157]].
                Zero-probability words: negative=0, positive=0.
    Unsmoothed accuracy is higher by 0.25 percentage points.
