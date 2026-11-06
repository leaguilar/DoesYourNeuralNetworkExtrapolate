"""prepare_camelyon_arrays.py -- materialize Camelyon17 patch arrays (raw +
first-principles stain-normalized + a stain-agnostic control) for the
from-scratch cross-hospital OOD training experiment.

Runs LOCALLY (in an env that has torchstain). It reads the raw
Camelyon17-WILDS PNG patches directly from metadata.csv (no `wilds`
package needed), builds ONE fixed data sample shared across all training
seeds, applies each arm's fixed input transform once, and writes uint8
arrays to a cache dir. The cluster training job then only needs
torch/torchvision/numpy and loads these arrays; it does NOT need
wilds/torchstain.

Splits (cross-hospital OOD, matching the WILDS axis):
  train hospitals = centers {0, 3, 4}
  OOD test        = center 2, an entirely held-out hospital (== WILDS test)
  ID test         = patient-level holdout WITHIN centers {0, 3, 4}
                    (same hospitals, unseen patients -> no patient leakage)

Arms (fixed input transform applied to each 96x96 RGB patch):
  raw        identity
  stainnorm  Macenko H&E stain normalization (phi, first-principles)
  channorm   per-image channel standardization (adaptive but stain-agnostic
             control -- isolates the optical-density model in phi)

Usage:
  conda run -n data_analysis python prepare_camelyon_arrays.py \
      --n-train 60000 --n-idtest 10000 --n-oodtest 20000 \
      --out ../data/camelyon_cache
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from PIL import Image

from histopath_stain_transforms import (
    ARM_CHANNORM, ARM_COLORCTRL, ARM_RAW, ARM_STAINNORM,
    ChanNormControl, ColorControlPsi, MacenkoPhi,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
CAMELYON_ROOT = os.path.join(_HERE, "..", "data", "camelyon17", "camelyon17_v1.0")

TRAIN_CENTERS = (0, 3, 4)
OOD_CENTER = 2


def patch_path(patient, node, x, y) -> str:
    p = f"{int(patient):03d}"
    n = int(node)
    d = f"patient_{p}_node_{n}"
    fn = f"patch_{d}_x_{int(x)}_y_{int(y)}.png"
    return os.path.join(CAMELYON_ROOT, "patches", d, fn)


def balanced_sample(df: pd.DataFrame, n_total: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Class-balanced random subsample of a metadata frame."""
    per_cls = n_total // 2
    parts = []
    for cls in (0, 1):
        sub = df[df.tumor == cls]
        take = min(per_cls, len(sub))
        parts.append(sub.sample(n=take, random_state=rng.randint(2**31 - 1)))
    out = pd.concat(parts)
    return out.sample(frac=1.0, random_state=rng.randint(2**31 - 1)).reset_index(drop=True)


def load_patches(df: pd.DataFrame) -> np.ndarray:
    imgs = np.empty((len(df), 96, 96, 3), dtype=np.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        with Image.open(patch_path(row.patient, row.node, row.x_coord, row.y_coord)) as im:
            imgs[i] = np.asarray(im.convert("RGB"))
    return imgs


def apply_transform(imgs: np.ndarray, fn) -> np.ndarray:
    if fn is None:
        return imgs
    out = np.empty_like(imgs)
    for i in range(len(imgs)):
        out[i] = fn(imgs[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--n-idtest", type=int, default=10000)
    ap.add_argument("--n-oodtest", type=int, default=20000)
    ap.add_argument("--holdout-frac", type=float, default=0.15,
                    help="fraction of train-hospital PATCHES held out (disjoint) for the ID test")
    ap.add_argument("--seed", type=int, default=20260724, help="data-sampling seed (fixed across training seeds)")
    ap.add_argument("--out", default=os.path.join(_HERE, "..", "data", "camelyon_cache"))
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    meta = pd.read_csv(os.path.join(CAMELYON_ROOT, "metadata.csv"), index_col=0)

    train_pool = meta[meta.center.isin(TRAIN_CENTERS)]
    ood_pool = meta[meta.center == OOD_CENTER]

    # In-distribution test = a DISJOINT random patch-level holdout from the
    # SAME training hospitals {0,3,4}. This is the standard in-distribution
    # generalization reference: same hospitals, unseen patches, so all arms
    # score near ceiling ID and it is a stable reference. The OOD axis (the
    # actual claim) is the entirely held-out hospital (center 2 == WILDS test),
    # which is untouched by this choice.
    idx = train_pool.index.to_numpy().copy()
    rng.shuffle(idx)
    n_hold = int(round(len(idx) * args.holdout_frac))
    idtest_df = train_pool.loc[idx[:n_hold]]
    train_df = train_pool.loc[idx[n_hold:]]

    print(f"train hospitals {TRAIN_CENTERS}: {len(train_pool)} patches "
          f"-> {len(train_df)} train / {len(idtest_df)} id-test (disjoint patch holdout)")
    print(f"pools: train={len(train_df)}  id_test={len(idtest_df)}  ood(center {OOD_CENTER})={len(ood_pool)}")

    train_s = balanced_sample(train_df, args.n_train, rng)
    idtest_s = balanced_sample(idtest_df, args.n_idtest, rng)
    oodtest_s = balanced_sample(ood_pool, args.n_oodtest, rng)
    print(f"sampled: train={len(train_s)}  id_test={len(idtest_s)}  ood_test={len(oodtest_s)}")

    t0 = time.time()
    train_raw = load_patches(train_s)
    idtest_raw = load_patches(idtest_s)
    oodtest_raw = load_patches(oodtest_s)
    print(f"loaded {len(train_raw)+len(idtest_raw)+len(oodtest_raw)} patches in {time.time()-t0:.1f}s")

    # phi reference fit ONCE on a single training-domain patch (no OOD stats used)
    phi = MacenkoPhi(reference_patch=train_raw[0])
    channorm = ChanNormControl()
    colorctrl = ColorControlPsi(seed=20260724)
    arm_fns = {ARM_RAW: None, ARM_STAINNORM: phi, ARM_CHANNORM: channorm, ARM_COLORCTRL: colorctrl}

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "y_train.npy"), train_s.tumor.to_numpy().astype(np.int64))
    np.save(os.path.join(args.out, "y_idtest.npy"), idtest_s.tumor.to_numpy().astype(np.int64))
    np.save(os.path.join(args.out, "y_oodtest.npy"), oodtest_s.tumor.to_numpy().astype(np.int64))

    for arm, fn in arm_fns.items():
        t0 = time.time()
        tr = apply_transform(train_raw, fn)
        idt = apply_transform(idtest_raw, fn)
        ood = apply_transform(oodtest_raw, fn)
        np.save(os.path.join(args.out, f"x_train_{arm}.npy"), tr)
        np.save(os.path.join(args.out, f"x_idtest_{arm}.npy"), idt)
        np.save(os.path.join(args.out, f"x_oodtest_{arm}.npy"), ood)
        extra = ""
        if arm == ARM_STAINNORM:
            extra = f" (Macenko: {phi.n_ok} ok / {phi.n_fail} fallback)"
        print(f"arm {arm:10s} written in {time.time()-t0:.1f}s{extra}")

    manifest = {
        "dataset": "Camelyon17-WILDS",
        "train_centers": list(TRAIN_CENTERS),
        "ood_center": OOD_CENTER,
        "id_test_split": "disjoint patch-level holdout within train centers",
        "holdout_frac": args.holdout_frac,
        "n_train": int(len(train_s)), "n_idtest": int(len(idtest_s)), "n_oodtest": int(len(oodtest_s)),
        "data_seed": args.seed,
        "arms": list(arm_fns.keys()),
        "macenko_n_ok": int(phi.n_ok), "macenko_n_fail": int(phi.n_fail),
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved cache + manifest to {args.out}")


if __name__ == "__main__":
    main()
