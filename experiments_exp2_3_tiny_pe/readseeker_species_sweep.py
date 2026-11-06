"""TinyTransformer ReadSeeker species-transfer sweep.

Train ONCE on yeast (chr I-IX), evaluate per-organism on the full genome
of each organism in --all-organisms (default: yeast, ecoli, bsubtilis,
spombe). Yeast eval uses the held-out test chromosome (chr XI) so the
yeast number is a true matched-organism ID baseline rather than a
training-data leak.

The test the gap between cls and amplitude/sorted heads should grow on
species that share less compositional structure with yeast. Backbones
of biology (codon period 3) are universal, so phase-INV heads should
hold; cls heads that learned yeast-specific position/content shortcuts
are expected to drop more.

Run:
  conda run -n feateng python -m experiments_exp2_3_tiny_pe.readseeker_species_sweep \\
      --variants \\
          sinusoidal:cls:3 \\
          codon_fourier:amplitude:3 \\
          codon_fourier:sorted:3 \\
      --seed 42 --train-bp 300 \\
      --all-organisms yeast ecoli bsubtilis spombe
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from experiments_exp2_3_tiny_pe.model import TinyPEModel, encode_sequence  # noqa: E402
from experiments_exp2_3_tiny_pe.data import (  # noqa: E402
    load_organism, sample_reads, sample_reads_clean, YEAST_SPLIT)


def _sample(genome, chroms, *, read_bp, n_per_class, seed,
            clean_labels: bool, rc_minus_positives: bool, label: str = ""):
    """Dispatch to v0 or clean-label sampler. Returns list[(seq, label)] only.

    The clean sampler also returns per-call statistics; we log them to
    stdout under the given `label` and discard the dict from the return.
    """
    if not clean_labels:
        return sample_reads(genome, chroms, read_bp=read_bp,
                            n_per_class=n_per_class, seed=seed)
    reads, stats = sample_reads_clean(
        genome, chroms, read_bp=read_bp,
        n_per_class=n_per_class, seed=seed,
        rc_minus_positives=rc_minus_positives)
    n_pos = sum(1 for _, y in reads if y == 1)
    n_neg = sum(1 for _, y in reads if y == 0)
    print(f"  [sample{('-' + label) if label else ''}] "
          f"pos={n_pos} (plus={stats['cds_plus']}, "
          f"minus={stats['cds_minus']}{' RCd' if rc_minus_positives else ''}) "
          f"neg={n_neg}  rejects: overlap={stats['rejected_overlap']} "
          f"N={stats['rejected_n']}")
    return reads


def _accuracy(y, pred):
    return float((y == pred).mean())


def _roc_auc(y, score):
    """Mann-Whitney U binary AUROC (no sklearn)."""
    y = np.asarray(y); s = np.asarray(score)
    pos_mask = (y == 1); neg_mask = (y == 0)
    n_pos = pos_mask.sum(); n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    R_pos = ranks[pos_mask].sum()
    U = R_pos - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


class TinyReadSeekerDataset(Dataset):
    def __init__(self, reads, max_len: int):
        self.reads = reads
        self.max_len = max_len

    def __len__(self): return len(self.reads)

    def __getitem__(self, idx):
        seq, label = self.reads[idx]
        s = seq.decode("ascii", errors="replace")
        ids, attn = encode_sequence(s, max_len=self.max_len)
        return dict(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(attn, dtype=torch.long),
            label=torch.tensor(label, dtype=torch.long),
        )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_proba = [], []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y    = batch["label"].to(device)
        logits = model(ids, attention_mask=mask)
        proba = torch.softmax(logits, dim=-1)[:, 1]
        all_y.append(y.detach().cpu()); all_proba.append(proba.detach().cpu())
    y = torch.cat(all_y).numpy()
    proba = torch.cat(all_proba).numpy()
    pred = (proba > 0.5).astype(int)
    return dict(acc=_accuracy(y, pred),
                auroc=_roc_auc(y, proba),
                n=len(y))


def sample_organism_eval_reads(genome, read_bp: int, n_per_class: int,
                                seed: int, organism_name: str,
                                yeast_test_chroms: list[str] | None = None,
                                args=None):
    """Sample CDS/nonCDS reads for cross-organism evaluation.

    For yeast we restrict to the held-out test chromosome (chr XI by
    default) so the yeast eval number doesn't leak training data. For
    every other organism we sample across the full genome (no chrom
    split exists for them in this codebase).
    """
    if organism_name == "yeast" and yeast_test_chroms is not None:
        chroms = yeast_test_chroms
    else:
        chroms = list(genome.chrom_seq.keys())
    return _sample(genome, chroms,
                   read_bp=read_bp, n_per_class=n_per_class, seed=seed,
                   clean_labels=getattr(args, "clean_labels", False),
                   rc_minus_positives=getattr(args, "rc_minus_positives", True),
                   label=organism_name)


def train_one(pe_arm, head_type, periods, args, train_genome, device):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_reads = _sample(train_genome, YEAST_SPLIT["train"],
                          read_bp=args.train_bp,
                          n_per_class=args.n_train, seed=args.seed,
                          clean_labels=args.clean_labels,
                          rc_minus_positives=args.rc_minus_positives,
                          label="train_yeast")
    val_reads   = _sample(train_genome, YEAST_SPLIT["val"],
                          read_bp=args.train_bp,
                          n_per_class=args.n_eval, seed=args.seed + 1,
                          clean_labels=args.clean_labels,
                          rc_minus_positives=args.rc_minus_positives,
                          label="val_yeast")
    train_ds = TinyReadSeekerDataset(train_reads, max_len=args.train_bp + 4)
    val_ds   = TinyReadSeekerDataset(val_reads,   max_len=args.train_bp + 4)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              num_workers=2, pin_memory=True)

    model = TinyPEModel(pe_arm=pe_arm, max_len=args.max_len,
                        d_model=args.d_model, n_layers=args.n_layers,
                        n_heads=args.n_heads, ff=args.ff,
                        head_type=head_type, codon_periods=periods,
                        task="binary",
                        feature_dropout=args.feature_dropout).to(device)
    n_params = model.trainable_param_count()

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n_steps = max(1, len(train_loader) * args.epochs)
    n_warm = int(0.1 * n_steps)
    def lr_lambda(step):
        if step < n_warm: return step / max(1, n_warm)
        prog = (step - n_warm) / max(1, n_steps - n_warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y    = batch["label"].to(device)
            logits = model(ids, attention_mask=mask)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            m_val = evaluate(model, val_loader, device)
            print(f"  ep{ep+1:>3d}/{args.epochs}  val_acc={m_val['acc']:.3f}  "
                  f"val_auroc={m_val['auroc']:.3f}")
    return model, n_params, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+",
                   default=["sinusoidal:cls:3",
                            "learned_ape:cls:3",
                            "codon_fourier:cls:3",
                            "codon_fourier:amplitude:3",
                            "codon_fourier:sorted:3",
                            "multi_period:amplitude:3,10.4,75,147"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-bp", type=int, default=300)
    p.add_argument("--all-organisms", nargs="+",
                   default=["yeast", "ecoli", "bsubtilis", "spombe"],
                   help="Organisms to evaluate on. Yeast is always the "
                        "training organism.")
    p.add_argument("--yeast-test-chroms", nargs="+",
                   default=YEAST_SPLIT["test_id"],
                   help="Yeast chromosomes used for the matched-organism ID "
                        "eval baseline (default chr XI).")
    p.add_argument("--max-len", type=int, default=400)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--clean-labels", action="store_true", default=False,
                   help="Use the field-standard any-strand CDS labels "
                        "(positives=read-center-in-CDS-on-either-strand, "
                        "negatives=read-span-no-CDS-on-either-strand). "
                        "Default False uses the v0 plus-strand-only labels "
                        "for backward compatibility with older results. "
                        "New experiments should set this flag.")
    p.add_argument("--rc-minus-positives", action="store_true", default=True,
                   help="When --clean-labels: reverse-complement minus-strand "
                        "CDS positives so the model always sees CDS in 5'->3' "
                        "orientation. Default True. Set --no-rc-minus-positives "
                        "to disable (positives stay in plus-strand orientation, "
                        "treating minus-strand CDS reads as same-class but "
                        "RC-flipped).")
    p.add_argument("--no-rc-minus-positives", dest="rc_minus_positives",
                   action="store_false")
    p.add_argument("--feature-dropout", type=float, default=0.0,
                   help="Group dropout probability for the biology head. "
                        "Each batch independently zeros each of 5 feature "
                        "groups (CLS / |FFT|hidden / per-base FFT / stop "
                        "counts / AA comp) with this prob, with 1/(1-p) "
                        "scaling on kept groups. Encourages the linear "
                        "head to use each biology hypothesis individually "
                        "rather than relying on a yeast-specific "
                        "combination.")
    p.add_argument("--out", type=str,
                   default=str(THIS_DIR.parent / "results"
                                / "exp2_3_species"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading {len(args.all_organisms)} organism(s) ...")
    t0 = time.time()
    genomes = {n: load_organism(n) for n in args.all_organisms}
    if "yeast" not in genomes:
        raise SystemExit("yeast must be in --all-organisms (it is the "
                         "training organism).")
    print(f"[data] {time.time()-t0:.1f}s. CDS counts: " +
          "  ".join(f"{n}={int(sum(m.sum() for m in g.chrom_cds_mask.values())):,}"
                    for n, g in genomes.items()))

    # Pre-build per-organism eval datasets (so each variant's eval loop is
    # cheap; no need to redraw reads per variant).
    eval_dsets = {}
    for org in args.all_organisms:
        reads = sample_organism_eval_reads(
            genomes[org], read_bp=args.train_bp,
            n_per_class=args.n_eval,
            seed=args.seed + 100 + abs(hash(org)) % 1000,
            organism_name=org,
            yeast_test_chroms=args.yeast_test_chroms,
            args=args)
        eval_dsets[org] = TinyReadSeekerDataset(reads, max_len=args.max_len)
        print(f"  [eval] {org:<14}  n_reads={len(reads):>5}")

    rows = []
    for variant in args.variants:
        pe_arm, head_type, periods_str = variant.split(":")
        periods = tuple(float(x) for x in periods_str.split(","))
        tag = f"{pe_arm}_{head_type}_P{periods_str.replace(',', '-')}"
        print(f"\n{'='*60}\n=== {tag} ===\n{'='*60}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, n_params, train_time = train_one(
            pe_arm, head_type, periods, args, genomes["yeast"], device)
        row = dict(pe=pe_arm, head_type=head_type, periods=list(periods),
                   train_bp=args.train_bp, train_time_s=train_time,
                   n_params=n_params, seed=args.seed,
                   train_organism="yeast",
                   all_organisms=list(args.all_organisms),
                   feature_dropout=args.feature_dropout,
                   clean_labels=args.clean_labels,
                   rc_minus_positives=args.rc_minus_positives)
        for org in args.all_organisms:
            loader = DataLoader(eval_dsets[org], batch_size=args.batch,
                                num_workers=2, pin_memory=True)
            m = evaluate(model, loader, device)
            for k, v in m.items():
                row[f"eval_{org}_{k}"] = v
            print(f"  eval {org:<14} acc={m['acc']:.3f}  auroc={m['auroc']:.3f}")
        if torch.cuda.is_available():
            row["gpu_mem_peak_gb"] = float(torch.cuda.max_memory_allocated() / 1e9)
            row["gpu_name"] = torch.cuda.get_device_name(0)
            row["device"] = "cuda"
        else:
            row["gpu_mem_peak_gb"] = 0.0
            row["gpu_name"] = None
            row["device"] = "cpu"
        rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = dict(args=vars(args), rows=rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] wrote {out_dir / 'summary.json'}")
    print("\n# Headline (AUROC per organism, train=yeast)")
    head = f"{'variant':<48}"
    for org in args.all_organisms:
        head += f"{org:>14}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = f"{r['pe']}+{r['head_type']}+P{r['periods']}".ljust(48)
        for org in args.all_organisms:
            line += f"{r.get(f'eval_{org}_auroc', float('nan')):>14.3f}"
        print(line)


if __name__ == "__main__":
    main()
