"""Position-encoding modules for the ReadSeeker PE-injection ablation.

Three arms, all callable as `pe(position_ids) -> (B, T, d_model)` so they
drop into BertEmbeddings.position_embeddings as a replacement:

  - LearnedAbsolutePE: thin wrapper around nn.Embedding(max_pos, d_model);
    reproduces DNABERT's default. Extends gracefully to longer sequences
    by initialising new positions to zero (the honest failure mode for
    length-OOD: the table simply doesn't cover positions > max_pos_pretrained).

  - SinusoidalPE: Vaswani 2017 sinusoidal table; defined for any position
    by construction so length-OOD is well-defined.

  - CodonFourierAdditive: learnable projection of [sin(2πt/3), cos(2πt/3)]
    to d_model, ADDED on top of an inner PE (typically a LearnedAbsolutePE).
    Mechanistically motivated: P=3 codon periodicity is the dominant
    CDS-vs-nonCDS signal after GC content. Defined for any position.

The forward signature matches `nn.Embedding(max_pos, d_model)` so these
modules can replace `bert.embeddings.position_embeddings` directly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnedAbsolutePE(nn.Module):
    """Learned absolute PE matching DNABERT/BERT's default.

    Initialised from a pretrained `nn.Embedding(max_pos_pretrained, d_model)`;
    if `max_len > max_pos_pretrained` the extra positions are initialised
    to zero. This is the honest length-OOD baseline: at unseen positions
    the model receives a zero positional signal.
    """

    def __init__(self, d_model: int, max_len: int,
                 pretrained_emb: nn.Embedding | None = None):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        if pretrained_emb is not None:
            n_pre = pretrained_emb.num_embeddings
            n_copy = min(n_pre, max_len)
            with torch.no_grad():
                self.emb.weight[:n_copy].copy_(pretrained_emb.weight[:n_copy])
                if max_len > n_pre:
                    self.emb.weight[n_pre:].zero_()

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(position_ids)


class SinusoidalPE(nn.Module):
    """Vaswani-style sinusoidal PE; closed-form, defined for any position."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe)
        self.max_len = max_len

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe[position_ids]


class CodonFourierAdditive(nn.Module):
    """Learned APE + additive learnable projection of P=3 Fourier features.

    Output at position t:
        emb_inner(t) + W @ [sin(2πt/3), cos(2πt/3)]
    where `emb_inner` is the wrapped baseline PE (typically a
    `LearnedAbsolutePE` initialised from DNABERT's pretrained table) and
    W is a learnable Linear(2 -> d_model). Zero-initialised W means the
    additive contribution starts at zero, so the wrapped model behaves
    identically to the baseline at init and only diverges as W learns
    to use the periodic signal.
    """

    def __init__(self, inner: nn.Module, d_model: int,
                 period: float = 3.0, zero_init: bool = True):
        super().__init__()
        self.inner = inner
        self.period = period
        self.proj = nn.Linear(2, d_model, bias=False)
        if zero_init:
            nn.init.zeros_(self.proj.weight)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        base = self.inner(position_ids)
        t = position_ids.float()
        angle = 2 * math.pi * t / self.period
        phi = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return base + self.proj(phi)


class MultiPeriodAdditive(nn.Module):
    """Learned APE + zero-init projection of sin/cos features at chosen periods.

    Same recipe as `CodonFourierAdditive` (zero-init projection so step-0 is
    identical to inner LearnedAbsolutePE) but with multiple periods stacked
    into the input feature. Biologically-motivated default periods:
        P=3    (codon — CDS-specific)
        P=10.4 (DNA helical period — nucleosome positioning preference)
        P=75   (nucleosomal harmonic — strong in non-CDS yeast regions)
        P=147  (nucleosomal core repeat — strong in non-CDS regions)
    The FFT analysis (`fft_periodicity.py`) showed P=3 dominates inside CDS
    and P~150/100/75 dominate outside CDS, so this 4-period set spans both
    sides of the discriminative axis.
    """

    def __init__(self, inner: nn.Module, d_model: int,
                 periods: tuple[float, ...] = (3.0, 10.4, 75.0, 147.0),
                 zero_init: bool = True):
        super().__init__()
        self.inner = inner
        self.register_buffer("periods",
                             torch.tensor(periods, dtype=torch.float32))
        self.proj = nn.Linear(2 * len(periods), d_model, bias=False)
        if zero_init:
            nn.init.zeros_(self.proj.weight)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        base = self.inner(position_ids)
        t = position_ids.float().unsqueeze(-1)
        angles = 2 * math.pi * t / self.periods.to(position_ids.device)
        phi = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)
        phi = phi.reshape(*position_ids.shape, 2 * self.periods.numel())
        return base + self.proj(phi)


class SinusoidalAdditive(nn.Module):
    """Learned APE + zero-initialised projection of frozen Vaswani sinusoidal.

    Fair length-OOD control for `codon_fourier`: same "preserve pretrained
    learned APE, add an extrapolable positional signal via a zero-init
    projection" recipe, but with a multi-frequency sinusoidal feature
    instead of a P=3 codon-aligned one.

    The sinusoidal table itself is a fixed buffer (Vaswani-style), but a
    learnable Linear(d_model -> d_model) sits on top with zero-init weights
    so the additive contribution starts at exactly zero — at step 0 the
    wrapped model is identical to the inner LearnedAbsolutePE, matching
    CodonFourierAdditive's zero-init invariant. Without this the fixed
    sinusoidal addend perturbs the pretrained activations from step 0
    and the model fails to fine-tune.
    """

    def __init__(self, inner: nn.Module, d_model: int, max_len: int = 8192):
        super().__init__()
        self.inner = inner
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("sin_table", pe)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(position_ids) + self.proj(self.sin_table[position_ids])


def make_pe(name: str, d_model: int, max_len: int,
            pretrained_emb: nn.Embedding | None = None,
            period: float = 3.0) -> nn.Module:
    """Construct the PE module for a given arm.

    learned_ape          : LearnedAbsolutePE seeded from `pretrained_emb`
                           (DNABERT's pretrained PE table; the ReadSeeker
                           baseline).
    sinusoidal_replace   : SinusoidalPE replacing the learned APE entirely.
                           Destructive of pretrained knowledge; included
                           only as an ablation showing that PE swap is not
                           free.
    sinusoidal_additive  : LearnedAbsolutePE (seeded) + frozen Vaswani
                           sinusoidal addend. Fair extrapolable-PE control
                           for `codon_fourier`.
    codon_fourier        : LearnedAbsolutePE (seeded) + learnable P=3
                           Fourier addend. The main treatment.
    multi_period         : LearnedAbsolutePE (seeded) + learnable additive
                           projection of sin/cos at biologically-motivated
                           periods (P=3, 10.4, 75, 147). FFT-informed
                           multi-period control vs sinusoidal_additive.
    """
    if name == "learned_ape":
        return LearnedAbsolutePE(d_model, max_len, pretrained_emb)
    if name in ("sinusoidal", "sinusoidal_replace"):
        return SinusoidalPE(d_model, max_len)
    if name == "sinusoidal_additive":
        inner = LearnedAbsolutePE(d_model, max_len, pretrained_emb)
        return SinusoidalAdditive(inner, d_model, max_len=max_len)
    if name == "codon_fourier":
        inner = LearnedAbsolutePE(d_model, max_len, pretrained_emb)
        return CodonFourierAdditive(inner, d_model, period=period)
    if name == "helical_fourier":
        # Same recipe as codon_fourier but injects sin/cos at the DNA
        # helical period (P=10.4) instead of the codon period (P=3).
        # The biologically-relevant period for cyclizability / nucleosome
        # rotational positioning.
        inner = LearnedAbsolutePE(d_model, max_len, pretrained_emb)
        return CodonFourierAdditive(inner, d_model, period=10.4)
    if name == "multi_period":
        inner = LearnedAbsolutePE(d_model, max_len, pretrained_emb)
        return MultiPeriodAdditive(inner, d_model)
    if name == "dual_period":
        # Default dual = P=3 + P=150 (empirical FFT peak).
        inner = LearnedAbsolutePE(d_model, max_len, pretrained_emb)
        return MultiPeriodAdditive(inner, d_model, periods=(3.0, 150.0))
    # Parametric "dual_period_3_<P2>" arms — period sweep on the second
    # period to test sensitivity to period precision (cf earlier finding
    # that synthetic results are exquisitely period-sensitive).
    if name.startswith("dual_period_3_"):
        try:
            p2 = float(name[len("dual_period_3_"):])
        except ValueError as e:
            raise ValueError(f"could not parse period from {name!r}") from e
        inner = LearnedAbsolutePE(d_model, max_len, pretrained_emb)
        return MultiPeriodAdditive(inner, d_model, periods=(3.0, p2))
    raise ValueError(f"unknown PE arm: {name!r}")
