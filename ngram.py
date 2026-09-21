"""Laplace count-model adapter; original n_gram.py is preserved unchanged.

Reuse NGram's initialization, train and unigram counting. Tuple keys are required:
the original string concatenation loses token boundaries and integer IDs would
be added rather than concatenated. Sequence-wise counts also avoid crossing lines.
"""
from collections import Counter
import math
import numpy as np
from n_gram import NGram


class NGramLM(NGram):
    def __init__(self, n, vocab_size):
        super().__init__(n)
        self.vocab_size = vocab_size
        self.vocabulary_size = vocab_size
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def fit(self, sequences):
        super().train([t for sequence in sequences for t in sequence[self.n - 1:]])
        self.vocabulary_size = self.vocab_size
        self.unigram_counts = self.count_unigrams()
        self.ngram_counts.clear()
        self.context_counts.clear()
        for sequence in sequences:
            for i in range(self.n - 1, len(sequence)):
                context = tuple(sequence[i - self.n + 1:i])
                self.ngram_counts[context + (sequence[i],)] += 1
                self.context_counts[context] += 1
        return self

    def log_prob(self, token, context):
        context = tuple(context)
        return math.log((self.ngram_counts[context + (token,)] + 1) /
                        (self.context_counts[context] + self.vocab_size))

    def next_token_log_probs(self, context):
        return np.array([self.log_prob(token, context) for token in range(self.vocab_size)])

    @property
    def stored_counts(self):
        return len(self.ngram_counts) + len(self.context_counts) + len(self.unigram_counts)
