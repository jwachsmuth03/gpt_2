"""Shared model-independent evaluation for Assignments 2 and 3.

The scalar perplexity and generate functions are the same implementations used
in the preceding Assignment 3 deliverable; Assignment 2 now documents and tests
that shared interface. They are new additions to the original group files.
"""
from pathlib import Path
import math
import numpy as np
from bpe import normalize


class EncodedSequences(list):
    """List of BOS/EOS-padded sequences retaining the original character count."""
    def __init__(self, sequences, character_count):
        super().__init__(sequences)
        self.character_count = character_count


def encode_lines(tokenizer, lines, n=3):
    normalized = [normalize(line) for line in lines]
    return EncodedSequences(
        [[tokenizer.bos_id] * (n - 1) + tokenizer.encode(line) + [tokenizer.eos_id]
         for line in normalized], sum(map(len, normalized)))


def perplexity(model, sequences):
    negative_log_likelihood, predicted_tokens = 0.0, 0
    for sequence in sequences:
        for i in range(model.n - 1, len(sequence)):
            context = sequence[i - model.n + 1:i]
            negative_log_likelihood -= model.log_prob(sequence[i], context)
            predicted_tokens += 1
    if predicted_tokens == 0 or sequences.character_count <= 0:
        raise ValueError("Perplexity needs predicted tokens and a positive character count.")
    return (math.exp(negative_log_likelihood / predicted_tokens),
            math.exp(negative_log_likelihood / sequences.character_count))


def generate(model, tokenizer, prompt, mode="sample", max_tokens=50, seed=0):
    if mode not in {"sample", "argmax"}:
        raise ValueError("mode must be sample or argmax")
    rng = np.random.default_rng(seed)
    ids = [tokenizer.bos_id] * (model.n - 1) + tokenizer.encode(normalize(prompt))
    for _ in range(max_tokens):
        context = ids[-(model.n - 1):] if model.n > 1 else []
        log_probs = model.next_token_log_probs(context)
        if mode == "argmax":
            token = int(np.argmax(log_probs))
        else:
            probabilities = np.exp(log_probs - np.max(log_probs))
            probabilities /= probabilities.sum()
            token = int(rng.choice(len(probabilities), p=probabilities))
        ids.append(token)
        if token == tokenizer.eos_id:
            return tokenizer.decode(ids) + " [stopped: <eos>]"
    return tokenizer.decode(ids) + f" [stopped: {max_tokens}-token limit]"


def encode_file(tokenizer, path, n=3):
    """Read one line per sequence, using the same padding/counting convention."""
    return encode_lines(tokenizer, Path(path).read_text(encoding="utf-8").splitlines(), n)
