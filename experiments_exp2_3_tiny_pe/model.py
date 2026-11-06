"""TinyTransformer + injected PE + cls/amplitude/sorted heads.

DNA-naive model: random-init transformer, no internal PE, no pretraining.
The only positional signal is the additive PE we inject. With the cls
head this means: at training, the model learns position-conditional
features locked to whatever phase the training distribution provided.
At eval with phase shift, those features rotate out of distribution.

Phase-invariant heads (amplitude / sorted) aggregate hidden states with
operations that don't depend on global cyclic shifts.

This is the small, capacity-bottlenecked, DNA-naive testbed where we
expect a meaningful cls-vs-amplitude gap to manifest, in contrast to
the DNABERT-6 fine-tune where pretraining gives the model enough
content prior to learn approximate phase invariance implicitly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from experiments_exp2_3_tiny_pe.pe import make_pe


# Standard genetic code. Inlined here so this module doesn't take an import
# dependency on the (much larger) classical_features module.
_CODON_TO_AA = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}
_AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY*"   # 20 standard + stop


def _build_codon_to_aa_table() -> torch.Tensor:
    """64-int LUT mapping codon index (b1*16 + b2*4 + b3, NUC_VOCAB encoding)
    to AA class 0..20 in `_AA_LETTERS`.
    """
    table = torch.full((64,), _AA_LETTERS.index("*"), dtype=torch.long)
    base_chars = "ACGT"   # matches NUC_VOCAB: A=0, C=1, G=2, T=3
    for codon, aa in _CODON_TO_AA.items():
        b1 = base_chars.index(codon[0])
        b2 = base_chars.index(codon[1])
        b3 = base_chars.index(codon[2])
        table[b1 * 16 + b2 * 4 + b3] = _AA_LETTERS.index(aa)
    return table


_NUM_AA = len(_AA_LETTERS)            # 21 (20 standard + stop)
_STOP_AA_IDX = _AA_LETTERS.index("*")  # 20


# Vocabulary: A=0 C=1 G=2 T=3 N=4 PAD=5 CLS=6 SEP=7
NUC_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4,
             "PAD": 5, "CLS": 6, "SEP": 7}
VOCAB_SIZE = 8
PAD_ID = 5
CLS_ID = 6
SEP_ID = 7


def encode_sequence(seq: str, max_len: int,
                    add_special: bool = True) -> tuple[list[int], list[int]]:
    """Encode an uppercase DNA sequence as token ids + attention mask.

    With `add_special=True` we prepend CLS and append SEP, so the actual
    body lives at positions [1, T-2]. The cls head reads position 0 of
    the hidden states; the amplitude / sorted heads read positions 1..
    The mask is 1 where the model should attend, 0 for PAD.
    """
    s = seq.upper()
    body = [NUC_VOCAB.get(c, NUC_VOCAB["N"]) for c in s]
    if add_special:
        ids = [CLS_ID] + body + [SEP_ID]
    else:
        ids = body
    attn = [1] * len(ids)
    if len(ids) < max_len:
        ids  = ids  + [PAD_ID] * (max_len - len(ids))
        attn = attn + [0]      * (max_len - len(attn))
    else:
        ids  = ids[:max_len]
        attn = attn[:max_len]
    return ids, attn


class _Embeddings(nn.Module):
    """Mimic DNABERT's `backbone.embeddings.position_embeddings` API so
    the existing `make_pe` helpers can drop in.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        # Stub used only as a pretrained-emb donor. With max_len=N we
        # initialise to zeros; codon_fourier wraps it as inner LearnedAPE.
        self.position_embeddings = None


class TinyPEModel(nn.Module):
    """Random-init transformer + injected PE + head.

    head_type:
      - 'cls': read [CLS] position only.
      - 'amplitude': [CLS] || |FFT(body)| at one bin per period.
      - 'sorted': [CLS] || sorted 3-frame mean of body.

    task:
      - 'binary': classification head -> 2-class logits, returns (B, 2).
      - 'regression': scalar head -> (B,) predictions.
    """

    def __init__(self, pe_arm: str, max_len: int,
                 d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 ff: int = 256, dropout: float = 0.1,
                 head_type: str = "cls",
                 codon_periods: tuple[float, ...] | list[float] = (3.0,),
                 task: str = "binary",
                 feature_dropout: float = 0.0):
        super().__init__()
        if head_type not in ("cls", "amplitude", "sorted", "biology"):
            raise ValueError(f"unknown head_type: {head_type!r}")
        if task not in ("binary", "regression"):
            raise ValueError(f"unknown task: {task!r}")
        if not (0.0 <= feature_dropout < 1.0):
            raise ValueError(f"feature_dropout must be in [0,1): {feature_dropout!r}")
        self.task = task
        self.head_type = head_type
        self.codon_periods = tuple(float(p) for p in codon_periods)
        self.feature_dropout = float(feature_dropout)
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pe = make_pe(pe_arm, d_model=d_model, max_len=max_len,
                          pretrained_emb=None)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, activation="relu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        if head_type == "cls":
            in_dim = d_model
        elif head_type == "amplitude":
            in_dim = (1 + len(self.codon_periods)) * d_model
        elif head_type == "sorted":
            in_dim = 4 * d_model
        elif head_type == "biology":
            # Strict superset of amplitude: [CLS] + K * |FFT|@P_k of hidden
            # + 4 per-base FFT@P=3 from input_ids
            # + 3 stop-codon counts per frame, sorted ascending
            # + 3 * 21 AA-composition histograms per frame, ranked by stop count.
            in_dim = ((1 + len(self.codon_periods)) * d_model
                      + 4 + 3 + 3 * _NUM_AA)

        out_dim = 2 if task == "binary" else 1
        self.head = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

        self.register_buffer(
            "position_ids", torch.arange(max_len).unsqueeze(0),
            persistent=False)
        self.register_buffer(
            "codon_aa_table", _build_codon_to_aa_table(),
            persistent=False)

    def _amplitude_features(self, hidden, attention_mask):
        body = hidden[:, 1:, :]
        T_body = body.shape[1]
        K = len(self.codon_periods)
        if T_body == 0:
            return torch.zeros(hidden.shape[0], K * hidden.shape[2],
                               device=hidden.device)
        # Mean-subtract over VALID positions, then re-zero PADs. This makes
        # FFT bin 0 (DC) exactly zero, so when a requested period rounds
        # to bin 0 (period >> T_body) the corresponding amplitude feature
        # is zero rather than the un-subtracted hidden-state mean (which
        # dominates the head input and drowns out the other bins).
        if attention_mask is not None:
            mask = attention_mask[:, 1:].to(body.dtype).unsqueeze(-1)
            n_valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            body_mean = (body * mask).sum(dim=1, keepdim=True) / n_valid
            body = (body - body_mean) * mask
        else:
            body = body - body.mean(dim=1, keepdim=True)
        spec = torch.fft.rfft(body, dim=1)
        amps = []
        for P in self.codon_periods:
            bin_p = max(0, min(spec.shape[1] - 1, round(T_body / P)))
            amps.append(spec[:, bin_p, :].abs())
        return torch.cat(amps, dim=-1)

    def _sorted_frame_features(self, hidden, attention_mask):
        body = hidden[:, 1:, :]
        if attention_mask is None:
            mask = torch.ones(body.shape[:2], device=body.device, dtype=body.dtype)
        else:
            mask = attention_mask[:, 1:].to(body.dtype)
        T_body = body.shape[1]
        if T_body == 0:
            return torch.zeros(hidden.shape[0], 3 * hidden.shape[2],
                               device=hidden.device)
        positions = torch.arange(T_body, device=body.device)
        means = []
        for k in range(3):
            frame_mask = mask * (positions % 3 == k).to(mask.dtype)
            denom = frame_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_k = (body * frame_mask.unsqueeze(-1)).sum(dim=1) / denom
            means.append(mean_k)
        stacked = torch.stack(means, dim=1)
        norms = stacked.norm(dim=-1)
        order = norms.argsort(dim=-1)
        sorted_means = torch.gather(
            stacked, 1, order.unsqueeze(-1).expand(-1, -1, stacked.shape[-1]))
        return sorted_means.flatten(1)

    def _per_base_fft_p3(self, input_ids: torch.Tensor,
                          attention_mask: torch.Tensor | None) -> torch.Tensor:
        """|FFT(base_indicator)|@P=3 for each base in {A=0, C=1, G=2, T=3}.

        Captures codon-frame base-composition information directly from the
        raw sequence (not learned hidden states), so it transfers across
        species without depending on the encoder.
        """
        body_ids = input_ids[:, 1:]
        T = body_ids.shape[1]
        if T == 0:
            return torch.zeros(input_ids.shape[0], 4, device=input_ids.device)
        if attention_mask is not None:
            mask = attention_mask[:, 1:].to(torch.float32)
        else:
            mask = torch.ones_like(body_ids, dtype=torch.float32)
        n_valid = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        feats = []
        for base_id in (0, 1, 2, 3):
            ind = (body_ids == base_id).to(torch.float32)
            mean = (ind * mask).sum(dim=1, keepdim=True) / n_valid
            ind = (ind - mean) * mask
            spec = torch.fft.rfft(ind, dim=1)
            bin_p3 = max(0, min(spec.shape[1] - 1, round(T / 3.0)))
            feats.append(spec[:, bin_p3].abs().unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def _stop_codons_per_frame_sorted(self, input_ids: torch.Tensor,
                                       attention_mask: torch.Tensor | None
                                       ) -> torch.Tensor:
        """Count of in-frame TAA / TAG / TGA codons in each of 3 frames,
        sorted ascending. Universal genetic code -> universal across species.
        """
        body_ids = input_ids[:, 1:]
        B, T = body_ids.shape
        if attention_mask is not None:
            body_mask = attention_mask[:, 1:].to(torch.bool)
        else:
            body_mask = torch.ones_like(body_ids, dtype=torch.bool)
        counts = torch.zeros(B, 3, device=body_ids.device, dtype=torch.float32)
        for frame in range(3):
            n_codons = (T - frame) // 3
            if n_codons <= 0:
                continue
            idx = torch.arange(n_codons, device=body_ids.device) * 3 + frame
            b1 = body_ids[:, idx]
            b2 = body_ids[:, idx + 1]
            b3 = body_ids[:, idx + 2]
            valid = (body_mask[:, idx] & (b1 < 4)
                     & body_mask[:, idx + 1] & (b2 < 4)
                     & body_mask[:, idx + 2] & (b3 < 4))
            is_stop = (((b1 == 3) & (b2 == 0) & (b3 == 0))   # TAA
                       | ((b1 == 3) & (b2 == 0) & (b3 == 2))  # TAG
                       | ((b1 == 3) & (b2 == 2) & (b3 == 0))  # TGA
                       )
            counts[:, frame] = (is_stop & valid).to(torch.float32).sum(dim=1)
        return torch.sort(counts, dim=-1).values

    def _aa_composition_per_frame_ranked(self, input_ids: torch.Tensor,
                                          attention_mask: torch.Tensor | None
                                          ) -> torch.Tensor:
        """21-D AA histogram for each of 3 frames, frames sorted ascending by
        their stop-class count. Returns a (B, 63) feature.

        Translation uses the standard genetic code (`_CODON_TO_AA`). Codons
        containing PAD or N are dropped. Histograms are L1-normalized by the
        valid-codon count per frame so different sequence lengths produce
        comparable values.
        """
        body_ids = input_ids[:, 1:]
        B, T = body_ids.shape
        if attention_mask is not None:
            body_mask = attention_mask[:, 1:].to(torch.bool)
        else:
            body_mask = torch.ones_like(body_ids, dtype=torch.bool)
        histograms = torch.zeros(B, 3, _NUM_AA,
                                  device=body_ids.device, dtype=torch.float32)
        for frame in range(3):
            n_codons = (T - frame) // 3
            if n_codons <= 0:
                continue
            idx = torch.arange(n_codons, device=body_ids.device) * 3 + frame
            b1 = body_ids[:, idx]
            b2 = body_ids[:, idx + 1]
            b3 = body_ids[:, idx + 2]
            valid = (body_mask[:, idx] & (b1 < 4)
                     & body_mask[:, idx + 1] & (b2 < 4)
                     & body_mask[:, idx + 2] & (b3 < 4))
            codon_idx = (b1.clamp(0, 3) * 16
                         + b2.clamp(0, 3) * 4
                         + b3.clamp(0, 3))               # (B, n_codons)
            aa_idx = self.codon_aa_table[codon_idx]      # (B, n_codons)
            valid_f = valid.to(torch.float32)
            hist = torch.zeros(B, _NUM_AA, device=body_ids.device,
                               dtype=torch.float32)
            hist.scatter_add_(1, aa_idx, valid_f)
            n_valid = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            histograms[:, frame, :] = hist / n_valid
        # Sort the 3 frames by their stop-class count (ascending: frame
        # with the fewest stops first -> the in-frame ORF candidate).
        stop_counts = histograms[:, :, _STOP_AA_IDX]
        order = torch.argsort(stop_counts, dim=-1)
        sorted_hist = torch.gather(
            histograms, 1, order.unsqueeze(-1).expand(-1, -1, _NUM_AA))
        return sorted_hist.flatten(1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.token_emb(input_ids)
        pos = self.position_ids[:, :T]
        h = h + self.pe(pos)
        h = self.input_norm(h)
        h = self.input_dropout(h)
        # Convert HuggingFace-style mask (1=keep, 0=PAD) to PyTorch's
        # src_key_padding_mask (True=PAD).
        key_pad = None
        if attention_mask is not None:
            key_pad = (attention_mask == 0)
        h = self.encoder(h, src_key_padding_mask=key_pad)
        h = self.encoder_norm(h)
        cls = h[:, 0]
        if self.head_type == "cls":
            feat = cls
        elif self.head_type == "amplitude":
            feat = torch.cat([cls, self._amplitude_features(h, attention_mask)],
                             dim=-1)
        elif self.head_type == "sorted":
            feat = torch.cat([cls, self._sorted_frame_features(h, attention_mask)],
                             dim=-1)
        elif self.head_type == "biology":
            groups = [
                cls,
                self._amplitude_features(h, attention_mask),
                self._per_base_fft_p3(input_ids, attention_mask),
                self._stop_codons_per_frame_sorted(input_ids, attention_mask),
                self._aa_composition_per_frame_ranked(input_ids, attention_mask),
            ]
            if self.training and self.feature_dropout > 0:
                # Independent Bernoulli per group per batch (mask shared across
                # samples). Inverted-scaling so expected magnitude matches eval.
                # Each group is one DGP hypothesis (period-3 in learned space,
                # per-base codon-frame asymmetry, no-stops-in-CDS, biological
                # AA composition). Forcing the head to handle random subsets of
                # hypotheses encourages distributed weights and OOD robustness.
                keep = (torch.rand(len(groups), device=cls.device)
                        >= self.feature_dropout)
                if not bool(keep.any().item()):
                    # Safety: never zero ALL groups in a batch.
                    keep[int(torch.randint(0, len(groups), (1,)).item())] = True
                scale = 1.0 / (1.0 - self.feature_dropout)
                groups = [g * scale if bool(keep[i].item())
                          else torch.zeros_like(g)
                          for i, g in enumerate(groups)]
            feat = torch.cat(groups, dim=-1)
        out = self.head(feat)
        if self.task == "regression":
            out = out.squeeze(-1)
        return out

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
