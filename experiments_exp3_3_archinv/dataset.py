"""Periodic Annotation Dataset.

Each sequence has random i.i.d. input tokens x_t ~ Uniform({0..vocab_size-1}).
The target is y_t = f(t mod period) where f is a fixed random mapping.
The ONLY positional signal is the PE -- x_t carries no position information.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class PeriodicAnnotationDataset(Dataset):
    """
    Args:
        n_seqs:     number of sequences
        seq_len:    length of each sequence
        period:     period P; targets repeat with period P
        vocab_size: input token vocabulary size (default 256)
        f_seed:     RNG seed for the fixed f: {0..P-1} -> {0..vocab_size-1}
        data_seed:  RNG seed for random input tokens x_t
    """

    def __init__(
        self,
        n_seqs: int,
        seq_len: int,
        period: int,
        vocab_size: int = 256,
        f_seed: int = 0,
        data_seed: int = 42,
    ):
        super().__init__()
        self.n_seqs = n_seqs
        self.seq_len = seq_len
        self.period = period
        self.vocab_size = vocab_size

        # Fixed target function f: {0..P-1} -> {0..vocab_size-1}
        rng_f = np.random.default_rng(f_seed)
        self.f = rng_f.integers(0, vocab_size, size=period)  # shape (P,)

        # Random input tokens: (n_seqs, seq_len)
        rng_data = np.random.default_rng(data_seed)
        x_np = rng_data.integers(0, vocab_size, size=(n_seqs, seq_len), dtype=np.int64)
        self.x = torch.from_numpy(x_np)  # (n_seqs, seq_len)

        # Targets: y_t = f[t mod P], same for every sequence
        positions = np.arange(seq_len, dtype=np.int64) % period
        y_np = self.f[positions].astype(np.int64)  # (seq_len,)
        self.y = torch.from_numpy(y_np)  # (seq_len,)

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int):
        return self.x[idx], self.y  # both LongTensors of shape (seq_len,)
