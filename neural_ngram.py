"""Assignment 3 interface around the group's UNCHANGED NeuralNGram network."""
import numpy as np
import torch
import torch.nn.functional as F

from neural_n_gram import NeuralNGram


class NeuralNGramLM(NeuralNGram):
    def __init__(self, n, vocab_size, n_embd=64, n_hidden=256):
        if n < 2 or vocab_size < 1:
            raise ValueError("Use n >= 2 and a nonempty vocabulary.")
        # The original constructor creates precisely the required layers.
        # Token IDs already come from BPE, so no string-to-index conversion is needed.
        super().__init__(n, n_embd, n_hidden, corpus=range(vocab_size))
        self.vocab_size = vocab_size

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or idx.shape[1] != self.n - 1:
            raise ValueError(f"Expected (batch_size, {self.n - 1}) context IDs.")
        # Reuse the original embedding -> flatten -> Linear -> ReLU -> Linear verbatim.
        logits = super().forward(idx)
        loss = None if targets is None else F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def next_token_log_probs(self, context):
        self.eval()
        if len(context) != self.n - 1:
            raise ValueError(f"Expected {self.n - 1} context tokens.")
        idx = torch.tensor([list(context)], dtype=torch.long,
                           device=next(self.parameters()).device)
        logits, _ = self(idx)
        # float64 keeps scalar and batched evaluation numerically consistent.
        return F.log_softmax(logits.double(), dim=-1)[0].cpu().numpy()

    @torch.no_grad()
    def log_prob(self, token, context):
        return float(self.next_token_log_probs(context)[token])
