"""Batched neural evaluation; corpus helpers are shared with Assignment 2."""
import math
import torch
import torch.nn.functional as F
from language_data import (
    SHAKESPEARE_URL, SHAKESPEARE_SHA256, PTB_URL, PTB_MD5,
    download_if_missing, load_shakespeare,
)


class WindowData:
    """A flat ID stream and valid target positions; BOS is never a target."""
    def __init__(self, sequences, n):
        stream, positions = [], []
        for sequence in sequences:
            offset = len(stream)
            stream.extend(sequence)
            positions.extend(range(offset + n - 1, offset + len(sequence)))
        self.flat_ids = torch.tensor(stream, dtype=torch.long)
        self.target_positions = torch.tensor(positions, dtype=torch.long)
        self.offsets = torch.arange(-(n - 1), 0, dtype=torch.long)
        self.character_count = sequences.character_count
        if not len(positions):
            raise ValueError("No training/evaluation examples.")

    def at(self, example_indices, device):
        positions = self.target_positions[example_indices]
        x = self.flat_ids[positions[:, None] + self.offsets[None, :]]
        y = self.flat_ids[positions]
        return x.to(device), y.to(device)


@torch.no_grad()
def batched_perplexity(model, sequences, batch_size=512):
    model.eval()
    windows = WindowData(sequences, model.n)
    device = next(model.parameters()).device
    total = 0.0
    for start in range(0, len(windows.target_positions), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(windows.target_positions)))
        x, y = windows.at(indices, device)
        logits, _ = model(x)
        total += F.cross_entropy(logits.double(), y, reduction="sum").item()
    return (math.exp(total / len(windows.target_positions)),
            math.exp(total / windows.character_count))
