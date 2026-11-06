"""Models for the hero-SSM experiment.

Three architectures:
  transformer  -- TaggerGPT with swappable PE (copied from experiments_hero)
  mamba        -- TaggerMamba using mamba-ssm; no PE needed/used
  s4           -- TaggerS4D using a minimal pure-PyTorch diagonal S4

For SSM models, positional information must come entirely from the recurrent
dynamics -- there is no explicit PE injection.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ModelType = Literal["transformer", "mamba", "s4"]
PEType    = Literal["exact_fourier", "standard_sinusoidal", "rope", "learned_ape", "no_pe",
                    "learned_fourier_1freq", "learned_fourier_kfreq", "learnable_sinusoidal"]

# ---------------------------------------------------------------------------
# Positional encodings (transformer only)
# ---------------------------------------------------------------------------

class ExactFourierPE(nn.Module):
    def __init__(self, d_model: int, period: int):
        super().__init__()
        self.period = period
        self.proj = nn.Linear(2, d_model, bias=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        angle = 2 * math.pi * t / self.period
        phi = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.proj(phi)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:seq_len]


class LearnedAPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.emb(torch.arange(seq_len, device=device))


class LearnedFourierPE(nn.Module):
    """Same parametric form as ExactFourierPE but the angular frequency is
    a learnable parameter (or k learnable parameters).

    phi(t) = [sin(2π ω_1 t), cos(2π ω_1 t), ..., sin(2π ω_k t), cos(2π ω_k t)]
    -> Linear(2k, d_model)

    The frequencies live in log-space (so they stay positive) and are
    log-uniform initialized in [1/init_max_period, 0.5]. With k=1 the
    question is "can SGD find ω = 1/P from random init?". With k>1 the
    question is "does extra capacity in frequency space help or hurt?".
    """

    def __init__(self, d_model: int, n_freqs: int = 1,
                 init_max_period: int = 4096,
                 init_period: float | None = None):
        super().__init__()
        self.n_freqs = n_freqs
        if init_period is not None:
            # Deterministic init at a chosen period (for the Rashomon
            # init-sweep experiment). All k frequencies start at the same
            # period; this is the cleanest test of "does SGD move log_freqs?"
            log_freqs = torch.full((n_freqs,), math.log(1.0 / init_period))
        else:
            # Log-uniform random init in [1/init_max_period, 0.5]
            low  = 1.0 / init_max_period
            high = 0.5
            log_freqs = torch.empty(n_freqs).uniform_(math.log(low), math.log(high))
        self.log_freqs = nn.Parameter(log_freqs)
        self.proj = nn.Linear(2 * n_freqs, d_model, bias=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        omegas = torch.exp(self.log_freqs).to(device)                    # (k,)
        angles = 2 * math.pi * t.unsqueeze(-1) * omegas                  # (T, k)
        phi    = torch.stack([torch.sin(angles), torch.cos(angles)], -1) # (T, k, 2)
        phi    = phi.reshape(seq_len, 2 * self.n_freqs)                  # (T, 2k)
        return self.proj(phi)                                             # (T, d_model)


class LearnableSinusoidalPE(nn.Module):
    """Vaswani-style sinusoidal PE with all d_model/2 frequencies learnable.

    Same number of parameters and same initialization as the standard
    SinusoidalPE, but the frequencies are nn.Parameters instead of a fixed
    buffer. Tests whether the standard PE can be "rescued" on the periodic
    task by letting SGD adjust its frequencies.
    """

    def __init__(self, d_model: int):
        super().__init__()
        n = d_model // 2
        # Init: ω_k = 10000^(-k/d) for k = 0..n-1, which is half the
        # exponent of the fixed sinusoidal PE above (ω_k = 10000^(-2k/d)),
        # so this arm starts from a different frequency set.
        log_freqs = torch.arange(0, n, dtype=torch.float32) * \
                    (-math.log(10000.0) / d_model)
        self.log_freqs = nn.Parameter(log_freqs)  # learnable
        self._d_model = d_model

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        omegas = torch.exp(self.log_freqs).to(device)                  # (d/2,)
        angles = t.unsqueeze(-1) * omegas                               # (T, d/2)
        out = torch.empty(seq_len, self._d_model, device=device)
        out[:, 0::2] = torch.sin(angles)
        out[:, 1::2] = torch.cos(angles)
        return out


def _build_rope_cache(seq_len, head_dim, device, base=10000.0):
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, theta)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---------------------------------------------------------------------------
# Transformer (TaggerGPT)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, pe_type):
        super().__init__()
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.head_dim = n_embd // n_head
        self.pe_type = pe_type
        self.c_attn  = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj  = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.pe_type == "rope":
            cos, sin = _build_rope_cache(T, self.head_dim, x.device)
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, n_head, n_embd, pe_type):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, pe_type)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TaggerGPT(nn.Module):
    def __init__(self, vocab_size=256, n_layer=4, n_head=4, n_embd=256,
                 pe_type: PEType = "exact_fourier", period: int = 64, max_len: int = 4096):
        super().__init__()
        self.pe_type = pe_type
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        if pe_type == "exact_fourier":
            self.pe = ExactFourierPE(n_embd, period)
        elif pe_type == "standard_sinusoidal":
            self.pe = SinusoidalPE(n_embd, max_len)
        elif pe_type == "learned_ape":
            self.pe = LearnedAPE(n_embd, max_len)
        elif pe_type == "learned_fourier_1freq":
            self.pe = LearnedFourierPE(n_embd, n_freqs=1, init_max_period=max_len)
        elif pe_type == "learned_fourier_kfreq":
            self.pe = LearnedFourierPE(n_embd, n_freqs=4, init_max_period=max_len)
        elif pe_type == "learnable_sinusoidal":
            self.pe = LearnableSinusoidalPE(n_embd)
        elif pe_type in ("rope", "no_pe"):
            self.pe = None
        else:
            raise ValueError(f"Unknown pe_type: {pe_type}")
        self.blocks = nn.Sequential(*[TransformerBlock(n_head, n_embd, pe_type) for _ in range(n_layer)])
        self.ln_f   = nn.LayerNorm(n_embd)
        self.head   = nn.Linear(n_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.tok_emb(x)
        if self.pe is not None:
            h = h + self.pe(x.shape[1], x.device)
        return self.head(self.ln_f(self.blocks(h)))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# S4D -- faithful PyTorch implementation of S4D-Lin
#
# Reference: Gu et al. (2022), "On the Parameterization and Initialization
# of Diagonal State Space Models", NeurIPS 2022.
# Official code: https://github.com/state-spaces/s4 (models/s4d.py)
#
# Continuous SSM: x'(t) = A x(t) + B u(t),  y(t) = C x(t) + D u(t)
# A diagonal with eigenvalues lambda_k = -(|a_k|) + i*pi*(k+1)  [S4D-Lin init]
# Discretized via ZOH; output computed via FFT convolution.
# ---------------------------------------------------------------------------

class S4DLayer(nn.Module):
    """S4D-Lin layer (Gu et al., NeurIPS 2022). Input/output: (B, L, d_model).

    Implements the diagonal S4D-Lin variant exactly as described in Section 3
    of the paper and the official reference implementation at
    https://github.com/state-spaces/s4 (models/s4d.py).
    """

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        N = d_state // 2  # work with complex conjugate pairs; real kernel = 2*Re[...]

        # Harmonic init: imaginary parts at pi, 2pi, ..., N*pi (per channel)
        A_real = 0.5 * torch.ones(d_model, N)
        A_imag = math.pi * torch.arange(1, N + 1).float()
        A_imag = A_imag.unsqueeze(0).expand(d_model, -1).clone()
        self.A_real = nn.Parameter(A_real)   # negated in forward (kept positive for stability)
        self.A_imag = nn.Parameter(A_imag)

        # Log discretization step per channel; init at exp(-2) ~ 0.135
        self.log_dt = nn.Parameter(torch.full((d_model,), -2.0))

        # B, C projection vectors (real; complex conjugate pairs handled by factor 2)
        self.B = nn.Parameter(torch.randn(d_model, N) / math.sqrt(N))
        self.C = nn.Parameter(torch.randn(d_model, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.ones(d_model))  # skip / direct term

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _kernel(self, L: int) -> torch.Tensor:
        """Return convolution kernel K of shape (d_model, L)."""
        dt    = torch.exp(self.log_dt).unsqueeze(-1)          # (H, 1)
        A     = -torch.abs(self.A_real) + 1j * self.A_imag    # (H, N) complex, stable
        A_bar = torch.exp(dt * A)                              # ZOH discrete eigenvalues
        # ZOH B_bar ≈ dt * B  (first-order approximation, exact when A is diagonal)
        B_bar = dt * self.B                                    # (H, N) real ≈ complex
        # Kernel: K[l] = 2 Re[ sum_n C_n * A_bar_n^l * B_bar_n ]
        l     = torch.arange(L, device=dt.device, dtype=torch.float32)
        log_A = torch.log(A_bar)                               # (H, N) complex
        A_pow = torch.exp(log_A.unsqueeze(-1) * l)            # (H, N, L) complex
        K = 2.0 * (self.C.unsqueeze(-1) * A_pow * B_bar.unsqueeze(-1)).real.sum(1)
        return K  # (H, L) real

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, d_model)."""
        L   = u.shape[1]
        K   = self._kernel(L)                                   # (H, L)
        u_f = torch.fft.rfft(u.transpose(1, 2), n=2 * L)       # (B, H, L+1)
        K_f = torch.fft.rfft(K,                 n=2 * L)       # (H, L+1)
        y   = torch.fft.irfft(u_f * K_f.unsqueeze(0), n=2 * L)[..., :L]
        y   = y.transpose(1, 2) + u * self.D                   # (B, L, H) + skip
        return self.out_proj(y)


class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm   = S4DLayer(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _build_pe(pe_type: PEType, d_model: int, period: int, max_len: int = 4096):
    """Return a PE module, or None for no_pe / rope (rope is attention-only)."""
    if pe_type == "exact_fourier":
        return ExactFourierPE(d_model, period)
    if pe_type == "standard_sinusoidal":
        return SinusoidalPE(d_model, max_len)
    if pe_type == "learned_ape":
        return LearnedAPE(d_model, max_len)
    if pe_type == "learned_fourier_1freq":
        return LearnedFourierPE(d_model, n_freqs=1, init_max_period=max_len)
    if pe_type == "learned_fourier_kfreq":
        return LearnedFourierPE(d_model, n_freqs=4, init_max_period=max_len)
    if pe_type == "learnable_sinusoidal":
        return LearnableSinusoidalPE(d_model)
    return None  # no_pe or rope (rope not meaningful for SSMs)


class TaggerS4(nn.Module):
    def __init__(self, vocab_size=256, n_layer=4, d_model=256, d_state=64,
                 pe_type: PEType = "no_pe", period: int = 64, max_len: int = 4096):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pe = _build_pe(pe_type, d_model, period, max_len)
        self.pe_type = pe_type
        self.layers  = nn.ModuleList([S4Block(d_model, d_state) for _ in range(n_layer)])
        self.norm_f  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.tok_emb(x)
        if self.pe is not None:
            h = h + self.pe(x.shape[1], x.device)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_f(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Mamba (mamba-ssm package, https://github.com/state-spaces/mamba)
# Install: pip install mamba-ssm causal-conv1d
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm  = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class TaggerMamba(nn.Module):
    def __init__(self, vocab_size=256, n_layer=4, d_model=256, d_state=64, d_conv=4, expand=2,
                 pe_type: PEType = "no_pe", period: int = 64, max_len: int = 4096):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pe = _build_pe(pe_type, d_model, period, max_len)
        self.pe_type = pe_type
        self.layers  = nn.ModuleList(
            [MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layer)]
        )
        self.norm_f  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        h = self.tok_emb(x)
        if self.pe is not None:
            h = h + self.pe(x.shape[1], x.device)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_f(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: ModelType,
    pe_type: PEType = "exact_fourier",
    vocab_size: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 256,
    period: int = 64,
    d_state_ssm: int = 64,
    max_len: int = 4096,
) -> nn.Module:
    if model_type == "transformer":
        return TaggerGPT(
            vocab_size=vocab_size, n_layer=n_layer, n_head=n_head,
            n_embd=n_embd, pe_type=pe_type, period=period, max_len=max_len,
        )
    elif model_type == "s4":
        return TaggerS4(
            vocab_size=vocab_size, n_layer=n_layer,
            d_model=n_embd, d_state=d_state_ssm,
            pe_type=pe_type, period=period, max_len=max_len,
        )
    elif model_type == "mamba":
        return TaggerMamba(
            vocab_size=vocab_size, n_layer=n_layer,
            d_model=n_embd, d_state=d_state_ssm,
            pe_type=pe_type, period=period, max_len=max_len,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
