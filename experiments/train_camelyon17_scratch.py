"""train_camelyon17_scratch.py -- train an image classifier FROM SCRATCH
(random init, no pretrained weights) on Camelyon17 for one arm, and measure
the cross-hospital OOD generalization gap.

This is the counterpart to the frozen-feature probe: a from-scratch CNN has
no color-robustness prior, so it latches onto hospital-specific H&E stain
color as a shortcut and generalizes poorly to a held-out hospital. The test
is whether a FIXED, first-principles input transform phi (Macenko stain
normalization, grounded in the Beer-Lambert optical-density model) closes
that OOD gap, while a stain-agnostic per-image normalization control does
not.

Portable: needs only torch + torchvision + numpy + scikit-learn. Reads the
pre-materialized uint8 arrays produced by prepare_camelyon_arrays.py, so it
needs NEITHER `wilds` NOR `torchstain` (neither is available in the minimal
cluster env this was trained in).

Model selection never uses OOD labels: OOD metrics are reported BOTH at the last epoch
and at the epoch chosen by best ID accuracy (OOD labels are never used for
selection).

Usage:
  python train_camelyon17_scratch.py --arm stainnorm --seed 0 \
      --cache-dir ../data/camelyon_cache --out ../results/expfm_v_histopath_scratch
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arm(cache_dir, arm):
    def L(p):
        return np.load(os.path.join(cache_dir, p))
    x_tr = L(f"x_train_{arm}.npy"); y_tr = L("y_train.npy")
    x_id = L(f"x_idtest_{arm}.npy"); y_id = L("y_idtest.npy")
    x_ood = L(f"x_oodtest_{arm}.npy"); y_ood = L("y_oodtest.npy")
    return x_tr, y_tr, x_id, y_id, x_ood, y_ood


class PatchDS(torch.utils.data.Dataset):
    """uint8 (N,96,96,3) -> standardized CHW float tensor, with optional
    label-preserving geometric augmentation (flips + 90-deg rotations) and
    optional stochastic colour augmentation.

    colour_aug is the standard non-domain-specific baseline for cross-scanner
    robustness in histopathology (per-channel random gain and bias, the "HSV/RGB
    light" family in Tellez et al. 2019). It is a TRAINING-TIME stochastic
    perturbation, not a fixed input map, so it is the control for
    "does closing the cross-hospital gap actually require the domain-derived
    stain model, or would generic colour robustness do it?"
    """

    def __init__(self, x_uint8, y, mean, std, augment, colour_aug=False,
                 gain=0.30, bias=0.10):
        self.x = x_uint8
        self.y = y.astype(np.int64)
        self.mean = mean.reshape(3, 1, 1)
        self.std = std.reshape(3, 1, 1)
        self.augment = augment
        self.colour_aug = colour_aug
        self.gain = gain
        self.bias = bias

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i].astype(np.float32) / 255.0          # HWC
        img = np.transpose(img, (2, 0, 1))                  # CHW
        if self.augment:
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k=k, axes=(1, 2))
            img = np.ascontiguousarray(img)
            if self.colour_aug:
                a = np.random.uniform(1.0 - self.gain, 1.0 + self.gain, size=(3, 1, 1))
                b = np.random.uniform(-self.bias, self.bias, size=(3, 1, 1))
                img = np.clip(img * a + b, 0.0, 1.0).astype(np.float32)
        img = (img - self.mean) / self.std
        return torch.from_numpy(img.astype(np.float32)), int(self.y[i])


def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        m = torchvision.models.resnet18(weights=None, num_classes=2)
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights=None, num_classes=2)
    elif arch == "densenet121":
        m = torchvision.models.densenet121(weights=None, num_classes=2)
    else:
        raise ValueError(arch)
    return m


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        prob1 = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        ps.append(prob1); ys.append(yb.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = float(((p >= 0.5).astype(int) == y).mean())
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "auc": auc, "n": int(len(y))}


def run(args):
    set_seed(args.seed)
    x_tr, y_tr, x_id, y_id, x_ood, y_ood = load_arm(args.cache_dir, args.arm)

    # per-arm channel mean/std from THIS arm's training patches
    flat = x_tr.reshape(-1, 3).astype(np.float32) / 255.0
    mean = flat.mean(0); std = flat.std(0) + 1e-6
    print(f"[{args.arm}] train mean={mean.round(3).tolist()} std={std.round(3).tolist()}")

    train_ds = PatchDS(x_tr, y_tr, mean, std, augment=True, colour_aug=args.colour_aug)
    id_ds = PatchDS(x_id, y_id, mean, std, augment=False)
    ood_ds = PatchDS(x_ood, y_ood, mean, std, augment=False)

    dl = lambda ds, sh: torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=sh, num_workers=args.workers,
        pin_memory=True, drop_last=sh, persistent_workers=args.workers > 0)
    train_loader = dl(train_ds, True)
    id_loader = dl(id_ds, False)
    ood_loader = dl(ood_ds, False)

    model = build_model(args.arch).to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    history = []
    t_start = time.time()
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True); yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * len(xb)
        sched.step()
        idm = evaluate(model, id_loader)
        oodm = evaluate(model, ood_loader)
        history.append({"epoch": ep, "train_loss": run_loss / len(train_ds), "id": idm, "ood": oodm})
        print(f"[{args.arm} s{args.seed}] ep{ep:02d} loss={run_loss/len(train_ds):.4f} "
              f"ID acc={idm['acc']:.3f} auc={idm['auc']:.3f}  "
              f"OOD acc={oodm['acc']:.3f} auc={oodm['auc']:.3f}  ({time.time()-t0:.1f}s)")

    # model selection: pick epoch by best ID accuracy (never OOD)
    best_ep = int(np.argmax([h["id"]["acc"] for h in history]))
    result = {
        "arm": args.arm, "seed": args.seed, "arch": args.arch, "colour_aug": bool(args.colour_aug),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "wd": args.wd,
        "n_train": int(len(x_tr)), "n_idtest": int(len(x_id)), "n_oodtest": int(len(x_ood)),
        "wall_time_s": round(time.time() - t_start, 1),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
        "last": {"id": history[-1]["id"], "ood": history[-1]["ood"]},
        "id_selected": {"epoch": best_ep, "id": history[best_ep]["id"], "ood": history[best_ep]["ood"]},
        "history": history,
    }
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.arm}+caug" if args.colour_aug else args.arm
    out_path = os.path.join(args.out, f"scratch_{args.arch}_{tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {out_path}")
    sel = result["id_selected"]
    print(f"[{args.arm} s{args.seed}] ID-selected (ep{sel['epoch']}): "
          f"OOD acc={sel['ood']['acc']:.3f} auc={sel['ood']['auc']:.3f}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["raw", "stainnorm", "channorm", "colorctrl"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "densenet121"])
    p.add_argument("--cache-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "camelyon_cache"))
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "expfm_v_histopath_scratch"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--colour-aug", action="store_true",
                   help="stochastic per-channel colour jitter during training (generic control)")
    args = p.parse_args()
    run(args)
