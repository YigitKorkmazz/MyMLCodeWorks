IMDb Movie Review Classification - CS464 Homework 1, Question 2

FILES
x_train.csv: 1600 reviews x 2000 word-count features, with a header.
y_train.csv: 1600 labels, without a header.
x_val.csv: 400 reviews x 2000 word-count features, with a header.
y_val.csv: 400 labels, without a header.

All CSV files use commas as delimiters. Labels are 0 (negative) and 1 (positive).
Feature row i and label row i describe the same review. The feature header is
the vocabulary, in the same order in both partitions. Values are nonnegative
integer token counts, not binary indicators or TF-IDF weights.

Example loading code:
    import numpy as np
    x_train = np.loadtxt('data/x_train.csv', delimiter=',', skiprows=1)
    y_train = np.loadtxt('data/y_train.csv', dtype=int)

PREPROCESSING
The fixed 2000-review teaching subset comes only from labeled reviews in the
original Stanford training partition. It is split into 1600 training and 400
validation examples using a fixed random seed (464). Repeated token sequences
are excluded. No review in the official test partition is used.

HTML tags are removed, HTML entities are decoded, and text is lowercased.
Tokens match [a-z]{2,}: contiguous runs of at least two English letters.
There is no stemming and stop words are retained. The vocabulary comprises
the 2000 words with the largest total training counts. Frequency ties are
resolved alphabetically. Validation reviews use this fixed training vocabulary;
out-of-vocabulary words are ignored.

The supplied data is already processed. Train only on x_train/y_train, then
evaluate on x_val/y_val. This is a classroom subset, not the full IMDb benchmark.

SOURCE AND ATTRIBUTION
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
and Christopher Potts. Learning Word Vectors for Sentiment Analysis.
Proceedings of ACL 2011, pages 142-150.
Dataset: https://ai.stanford.edu/~amaas/data/sentiment/
Paper: https://aclanthology.org/P11-1015/
The dataset authors request citation of this paper when using the dataset.
