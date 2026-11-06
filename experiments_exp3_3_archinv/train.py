"""Train transformer / Mamba / S4D on periodic annotation task.

Usage examples:
    python train.py --model_type transformer --pe_type exact_fourier --period 64
    python train.py --model_type mamba  --period 128 --seed 42
    python train.py --model_type s4     --period 256 --seed 43
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments_exp3_3_archinv.dataset import PeriodicAnnotationDataset
from experiments_exp3_3_archinv.model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_schedule(optimizer, n_steps: int, warmup: int = 200, lr_min_ratio: float = 0.1):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, n_steps - warmup)
        return lr_min_ratio + (1.0 - lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_id(model, x: torch.Tensor, y: torch.Tensor, batch_size: int = 64) -> float:
    model.eval()
    correct = total = 0
    for i in range(0, len(x), batch_size):
        xb = x[i:i+batch_size].to(DEVICE)
        yb = y.unsqueeze(0).expand(len(xb), -1).to(DEVICE)
        preds = model(xb).argmax(dim=-1)
        correct += (preds == yb).sum().item()
        total   += yb.numel()
    model.train()
    return correct / total


@torch.no_grad()
def evaluate_ood(model, x_full: torch.Tensor, y_ood: torch.Tensor,
                 L_train: int, batch_size: int = 64) -> float:
    """Feed full (L_train + L_ood) sequences; evaluate only tail positions."""
    model.eval()
    correct = total = 0
    for i in range(0, len(x_full), batch_size):
        xb      = x_full[i:i+batch_size].to(DEVICE)
        logits  = model(xb)[:, L_train:]        # (B, L_ood, vocab_size)
        yb      = y_ood.unsqueeze(0).expand(len(xb), -1).to(DEVICE)
        correct += (logits.argmax(-1) == yb).sum().item()
        total   += yb.numel()
    model.train()
    return correct / total


def build_eval_datasets(args):
    ds_id = PeriodicAnnotationDataset(
        n_seqs=args.n_eval, seq_len=args.L_train, period=args.period,
        vocab_size=args.vocab_size, f_seed=0, data_seed=999,
    )
    ds_full = PeriodicAnnotationDataset(
        n_seqs=args.n_eval, seq_len=args.L_train + args.L_ood, period=args.period,
        vocab_size=args.vocab_size, f_seed=0, data_seed=999,
    )
    return ds_id.x, ds_id.y, ds_full.x, ds_full.y[args.L_train:]


def condition_str(args) -> str:
    if args.model_type == "transformer":
        return f"transformer__{args.pe_type}"
    if args.pe_type and args.pe_type != "no_pe":
        return f"{args.model_type}__{args.pe_type}"
    return args.model_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="transformer",
                        choices=["transformer", "mamba", "s4"])
    parser.add_argument("--pe_type",    default="exact_fourier",
                        choices=["exact_fourier", "standard_sinusoidal", "rope",
                                 "learned_ape", "no_pe",
                                 "learned_fourier_1freq",
                                 "learned_fourier_kfreq",
                                 "learnable_sinusoidal"])
    parser.add_argument("--period",     type=int, default=64)
    parser.add_argument("--L_train",    type=int, default=None,
                        help="Training length (default: 4 * period)")
    parser.add_argument("--L_ood",      type=int, default=None,
                        help="OOD length (default: 2 * period)")
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--n_seqs",     type=int, default=10000)
    parser.add_argument("--n_eval",     type=int, default=2000,
                        help="Sequences for ID and OOD eval (each)")
    parser.add_argument("--n_layer",    type=int, default=4)
    parser.add_argument("--n_head",     type=int, default=4)
    parser.add_argument("--n_embd",     type=int, default=256)
    parser.add_argument("--d_state_ssm", type=int, default=64,
                        help="SSM state dimension (S4D only)")
    parser.add_argument("--n_steps",    type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--eval_every",  type=int, default=500)
    parser.add_argument("--patience",    type=int, default=10,
                        help="Early stop after N evals with no loss improvement. "
                             "10 evals = 5000 steps at default eval_every=500, ~half "
                             "the default budget. Tune down for shorter runs.")
    parser.add_argument("--min_delta",   type=float, default=1e-5,
                        help="Min loss improvement to reset patience counter. "
                             "1e-5 = essentially noise-floor; bump for noisier targets.")
    parser.add_argument("--output",      type=str, default=None)
    parser.add_argument("--save_model",  action="store_true",
                        help="Save final model weights alongside the JSON result")
    parser.add_argument("--sweep_tag",   type=str, default=None,
                        help="Tag identifying which sweep (A/B/C/D/baseline) this run belongs to")
    args = parser.parse_args()

    # Derived defaults
    if args.L_train is None:
        args.L_train = 4 * args.period
    if args.L_ood is None:
        args.L_ood = 2 * args.period

    set_seed(args.seed)

    ds_train = PeriodicAnnotationDataset(
        n_seqs=args.n_seqs, seq_len=args.L_train, period=args.period,
        vocab_size=args.vocab_size, f_seed=0, data_seed=args.seed,
    )
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = build_model(
        model_type   = args.model_type,
        pe_type      = args.pe_type,
        vocab_size   = args.vocab_size,
        n_layer      = args.n_layer,
        n_head       = args.n_head,
        n_embd       = args.n_embd,
        period       = args.period,
        d_state_ssm  = args.d_state_ssm,
        max_len      = args.L_train + args.L_ood,
    ).to(DEVICE)

    cond = condition_str(args)
    print(f"Model: {model.count_params():,} params | condition: {cond} | P={args.period} | device: {DEVICE}")
    print(f"  L_train={args.L_train}  L_ood={args.L_ood}  n_steps={args.n_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = make_lr_schedule(optimizer, args.n_steps)
    criterion = nn.CrossEntropyLoss()

    x_id, y_id, x_full, y_ood = build_eval_datasets(args)

    history          = []
    step             = 0
    best_loss        = float("inf")
    patience_counter = 0
    t0               = time.time()
    iter_loader      = iter(loader)

    while step < args.n_steps:
        try:
            x_batch, y_batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x_batch, y_batch = next(iter_loader)

        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.expand(len(x_batch), -1).to(DEVICE)

        logits = model(x_batch)
        loss   = criterion(logits.reshape(-1, args.vocab_size), y_batch.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.eval_every == 0 or step == args.n_steps:
            acc_id  = evaluate_id(model, x_id, y_id)
            acc_ood = evaluate_ood(model, x_full, y_ood, args.L_train)
            elapsed = time.time() - t0
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"ID {acc_id:.4f} | OOD {acc_ood:.4f} | {elapsed:.1f}s")
            history.append({"step": step, "loss": loss.item(),
                            "acc_id": acc_id, "acc_ood": acc_ood})

            # Early stopping. Two independent triggers; either fires:
            #  (a) Solved: ID and OOD both >= 99% for last 3 evals
            #              (skips the patience wait when the task is trivially solved)
            #  (b) Patience: no loss improvement (> min_delta) for `patience` evals
            #              (classic plateau detection; works for cases where
            #               OOD never reaches 99% but training has plateaued)
            if len(history) >= 3 and all(
                h["acc_id"] >= 0.99 and h["acc_ood"] >= 0.99
                for h in history[-3:]
            ):
                print(f"[early-stop] step {step}: ID & OOD >= 0.99 (3 evals)")
                break
            current_loss = loss.item()
            if current_loss < best_loss - args.min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[early-stop] step {step}: no loss improvement for "
                      f"{args.patience} evals (best={best_loss:.6f})")
                break

    acc_id_final  = evaluate_id(model, x_id, y_id)
    acc_ood_final = evaluate_ood(model, x_full, y_ood, args.L_train)
    wall_time_s   = time.time() - t0
    chance        = 1.0 / args.vocab_size

    if torch.cuda.is_available():
        gpu_name         = torch.cuda.get_device_name(DEVICE)
        gpu_mem_peak_gb  = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    else:
        gpu_name         = None
        gpu_mem_peak_gb  = None

    # For learnable PEs: capture final ω so we can verify SGD found 1/P.
    learned_freqs = None
    pe_module = getattr(model, "pe", None)
    if pe_module is not None and hasattr(pe_module, "log_freqs"):
        learned_freqs = torch.exp(pe_module.log_freqs.detach()).cpu().tolist()

    result = {
        "experiment_id": "3.3",
        "condition":   cond,
        "sweep_tag":   args.sweep_tag,
        "model_type":  args.model_type,
        "pe_type":     args.pe_type if args.model_type == "transformer" else None,
        "period":      args.period,
        "L_train":     args.L_train,
        "L_ood":       args.L_ood,
        "vocab_size":  args.vocab_size,
        "n_layer":     args.n_layer,
        "n_head":      args.n_head,
        "n_embd":      args.n_embd,
        "d_state_ssm": args.d_state_ssm,
        "n_steps":     args.n_steps,
        "seed":        args.seed,
        "n_params":    model.count_params(),
        "wall_time_s": wall_time_s,
        "device":      str(DEVICE),
        "gpu_name":    gpu_name,
        "gpu_mem_peak_gb": gpu_mem_peak_gb,
        "acc_id":      acc_id_final,
        "acc_ood":     acc_ood_final,
        "err_id":      1.0 - acc_id_final,
        "err_ood":     1.0 - acc_ood_final,
        "chance":      chance,
        "learned_freqs": learned_freqs,   # null for non-learnable PE types
        "history":     history,
    }

    print(f"\n=== FINAL | {cond} P={args.period} seed={args.seed} ===")
    print(f"  ID  acc: {acc_id_final:.4f}  (err {1-acc_id_final:.4f})")
    print(f"  OOD acc: {acc_ood_final:.4f}  (err {1-acc_ood_final:.4f})")
    print(f"  Chance:  {chance:.4f}")

    out_dir = args.output or os.path.join(
        os.path.dirname(__file__), "..", "results", "exp3_3"
    )
    os.makedirs(out_dir, exist_ok=True)
    ts    = int(time.time())
    # Encode L_train and d_state in the stem so sweeps with the same
    # (model, pe, P, seed) but different L_train / d_state do not collide.
    extra = f"__L{args.L_train}__O{args.L_ood}__ds{args.d_state_ssm}"
    stem  = f"exp3_3__{cond}__P{args.period}__seed{args.seed}{extra}__{ts}"

    fpath = os.path.join(out_dir, stem + ".json")
    with open(fpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {fpath}")

    if args.save_model:
        mpath = os.path.join(out_dir, stem + ".pt")
        torch.save(model.state_dict(), mpath)
        print(f"Model: {mpath}")


if __name__ == "__main__":
    main()
