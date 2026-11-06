"""
expfm_v_histopath_stain.py — Frozen vision foundation model + Macenko stain
normalization input transform on a real cross-hospital OOD benchmark.

Claim tested
------------
Applying a FIXED, domain-derived input transform phi (Macenko H&E stain
normalization, grounded in the Beer-Lambert optical-density model of light
absorption by hematoxylin/eosin dye) to the inputs of a FROZEN pretrained
backbone closes an out-of-distribution generalization gap caused by a REAL
domain shift (a held-out HOSPITAL with a different scanner/staining
protocol), converting extrapolation (new hospital) into something closer to
interpolation. A fixed, non-adaptive color perturbation of matched form
(same kind of pixel-wise RGB remap) is the negative control and should NOT
close the gap, because it does not adapt to each image's own stain
statistics the way Macenko normalization does.

Setup
-----
- Data: Camelyon17-WILDS (lymph-node histopathology patches, 96x96 RGB,
  tumor vs normal; Bandi et al. 2018 / Koh et al. 2021 WILDS benchmark).
  Official WILDS split_scheme='official':
    train   -> centers {0, 3, 4}, source-designated train slides  (ID train)
    id_val  -> centers {0, 3, 4}, held-out slides, SAME hospitals (ID test)
    test    -> center 2, an ENTIRELY held-out hospital             (OOD test)
  (center 1 / WILDS "val" split is the official OOD *validation* center;
   unused here since we only need one ID and one OOD test split.)
- Backbone: FROZEN (no fine-tuning), timm `resnet50.a1_in1k` (ImageNet,
  default) as a fixed feature extractor (2048-d GAP-pooled features); an
  optional DINOv2 ViT-S/14 (`vit_small_patch14_dinov2.lvd142m`, 384-d) can
  be added via --backbone for a robustness check. Weights are NEVER updated.
- phi = Macenko stain normalization (torchstain, numpy backend). The
  reference (target) stain matrix is fit ONCE, on a SINGLE representative
  patch drawn from the TRAINING domain only; every image in every arm/split
  is remapped onto that same fixed reference appearance. No OOD statistics
  are used anywhere in fitting phi.
- Probe: logistic regression (scikit-learn, L2, standardized features)
  trained ONLY on ID-train features, separately per arm.
- Arms (phi/control applied to the raw 96x96 patch before the frozen
  backbone; see histopath_stain_transforms.py for the exact transforms):
    raw         identity (no preprocessing)
    stainnorm   Macenko H&E stain normalization (phi, principled)
    colorctrl   NEGATIVE CONTROL: one fixed random RGB affine remap, same
                map for every image regardless of hospital of origin

Metrics reported per arm, for BOTH id_val (ID test) and test (OOD test):
accuracy and ROC-AUC.

Usage
-----
  conda run -n data_analysis python expfm_v_histopath_stain.py --smoke
  conda run -n data_analysis python expfm_v_histopath_stain.py
  conda run -n data_analysis python expfm_v_histopath_stain.py --backbone resnet50 dinov2
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as tF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset

from histopath_stain_transforms import ARM_COLORCTRL, ARM_RAW, ARM_STAINNORM, ColorControlPsi, MacenkoPhi

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(_HERE, "..", "data", "camelyon17")
RESULTS_DIR = os.path.join(_HERE, "..", "results", "expfm_v_histopath")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SPLIT = "train"     # centers {0,3,4}, source train slides   -> ID train
ID_TEST_SPLIT = "id_val"  # centers {0,3,4}, held-out slides       -> ID test
OOD_TEST_SPLIT = "test"   # center 2, entirely held-out hospital   -> OOD test


# ---------------------------------------------------------------------------
# Data: stratified index sampling + PIL -> uint8 array loading
# ---------------------------------------------------------------------------

def sample_indices(dataset, split_name, n_per_class, seed):
    """Class-balanced random sample of indices from a named WILDS split."""
    split_val = dataset.split_dict[split_name]
    y = dataset.y_array.numpy()
    idx_all = np.where(dataset.split_array == split_val)[0]
    rng = np.random.RandomState(seed)
    sel = []
    for cls in (0, 1):
        cls_idx = idx_all[y[idx_all] == cls].copy()
        rng.shuffle(cls_idx)
        sel.append(cls_idx[: min(n_per_class, len(cls_idx))])
    sel = np.concatenate(sel)
    rng.shuffle(sel)
    return sel


def load_raw_arrays(dataset, indices):
    """PIL -> (N, 96, 96, 3) uint8 RGB array, plus labels."""
    imgs = np.empty((len(indices), 96, 96, 3), dtype=np.uint8)
    for i, idx in enumerate(indices):
        imgs[i] = np.array(dataset.get_input(int(idx)))
    labels = dataset.y_array.numpy()[indices]
    return imgs, labels


def apply_transform(imgs: np.ndarray, transform_fn) -> np.ndarray:
    if transform_fn is None:
        return imgs
    out = np.empty_like(imgs)
    for i in range(len(imgs)):
        out[i] = transform_fn(imgs[i])
    return out


# ---------------------------------------------------------------------------
# Frozen backbones (feature extractors, never updated)
# ---------------------------------------------------------------------------

BACKBONE_SPECS = {
    "resnet50": dict(name="resnet50.a1_in1k", img_size=224),
    "dinov2": dict(name="vit_small_patch14_dinov2.lvd142m", img_size=224),
}


def get_backbone(key: str):
    spec = BACKBONE_SPECS[key]
    kwargs = dict(pretrained=True, num_classes=0)
    if key == "dinov2":
        kwargs["img_size"] = spec["img_size"]
    model = timm.create_model(spec["name"], **kwargs)
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEVICE)
    return model, mean, std, spec["img_size"]


@torch.no_grad()
def extract_features(model, mean, std, img_size, imgs_uint8: np.ndarray, batch=128):
    """(N, 96, 96, 3) uint8 -> resize -> ImageNet-style normalize -> frozen
    backbone -> (N, D) pooled feature matrix."""
    out = []
    for s in range(0, len(imgs_uint8), batch):
        chunk = imgs_uint8[s : s + batch]
        t = torch.from_numpy(chunk).float().div_(255.0).permute(0, 3, 1, 2).to(DEVICE)
        t = tF.interpolate(t, size=img_size, mode="bilinear", align_corners=False)
        t = (t - mean) / std
        f = model(t)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    n_train, n_idtest, n_oodtest = (
        (150, 150, 150) if args.smoke else (args.n_train, args.n_idtest, args.n_oodtest)
    )

    print(f"Loading Camelyon17-WILDS metadata from {DATA_ROOT} ...")
    dataset = Camelyon17Dataset(root_dir=DATA_ROOT, download=False, split_scheme="official")
    centers = dataset.metadata_array[:, 0].numpy()
    print("Split sizes (full):",
          {name: int((dataset.split_array == val).sum()) for name, val in dataset.split_dict.items()})
    print("Centers present in train/id_val:", sorted(set(centers[dataset.split_array == dataset.split_dict[TRAIN_SPLIT]].tolist())))
    print("Centers present in test (OOD):", sorted(set(centers[dataset.split_array == dataset.split_dict[OOD_TEST_SPLIT]].tolist())))

    train_idx = sample_indices(dataset, TRAIN_SPLIT, n_train // 2, seed=args.seed)
    idtest_idx = sample_indices(dataset, ID_TEST_SPLIT, n_idtest // 2, seed=args.seed + 1)
    oodtest_idx = sample_indices(dataset, OOD_TEST_SPLIT, n_oodtest // 2, seed=args.seed + 2)
    print(f"Sampled: train={len(train_idx)}  id_test={len(idtest_idx)}  ood_test={len(oodtest_idx)}")

    t0 = time.time()
    train_raw, train_y = load_raw_arrays(dataset, train_idx)
    idtest_raw, idtest_y = load_raw_arrays(dataset, idtest_idx)
    oodtest_raw, oodtest_y = load_raw_arrays(dataset, oodtest_idx)
    print(f"Loaded raw patches from disk in {time.time() - t0:.1f}s")

    # Fit phi's reference stain matrix on a SINGLE training-domain patch only.
    ref_patch = train_raw[0]
    phi = MacenkoPhi(reference_patch=ref_patch)
    psi_control = ColorControlPsi(seed=20260724)

    transforms = {ARM_RAW: None, ARM_STAINNORM: phi, ARM_COLORCTRL: psi_control}

    t0 = time.time()
    transformed = {}
    for arm, fn in transforms.items():
        transformed[arm] = dict(
            train=apply_transform(train_raw, fn),
            idtest=apply_transform(idtest_raw, fn),
            oodtest=apply_transform(oodtest_raw, fn),
        )
    print(f"Applied all arm transforms in {time.time() - t0:.1f}s "
          f"(Macenko: {phi.n_ok} ok / {phi.n_fail} fallback-to-raw)")

    results = {
        "config": {
            "smoke": args.smoke,
            "n_train": int(n_train), "n_idtest": int(n_idtest), "n_oodtest": int(n_oodtest),
            "n_train_sampled": int(len(train_idx)), "n_idtest_sampled": int(len(idtest_idx)),
            "n_oodtest_sampled": int(len(oodtest_idx)),
            "seed": args.seed,
            "dataset": "Camelyon17-WILDS",
            "split_scheme": "official",
            "id_train_split": TRAIN_SPLIT, "id_test_split": ID_TEST_SPLIT, "ood_test_split": OOD_TEST_SPLIT,
            "id_train_centers": sorted(set(centers[dataset.split_array == dataset.split_dict[TRAIN_SPLIT]].tolist())),
            "ood_test_centers": sorted(set(centers[dataset.split_array == dataset.split_dict[OOD_TEST_SPLIT]].tolist())),
            "macenko_n_ok": int(phi.n_ok), "macenko_n_fail": int(phi.n_fail),
        },
        "backbones": {},
    }

    for bb_key in args.backbone:
        print(f"\n=== backbone: {bb_key} (frozen) ===")
        model, mean, std, img_size = get_backbone(bb_key)
        bb_results = {}
        for arm in transforms:
            t0 = time.time()
            f_train = extract_features(model, mean, std, img_size, transformed[arm]["train"])
            f_idtest = extract_features(model, mean, std, img_size, transformed[arm]["idtest"])
            f_oodtest = extract_features(model, mean, std, img_size, transformed[arm]["oodtest"])

            scaler = StandardScaler().fit(f_train)
            probe = LogisticRegression(max_iter=5000, C=1.0)
            probe.fit(scaler.transform(f_train), train_y)

            def eval_split(f, y):
                fs = scaler.transform(f)
                pred = probe.predict(fs)
                prob = probe.predict_proba(fs)[:, 1]
                acc = float(accuracy_score(y, pred))
                try:
                    auc = float(roc_auc_score(y, prob))
                except ValueError:
                    auc = float("nan")
                return {"acc": acc, "auc": auc, "n": int(len(y)), "n_pos": int(y.sum())}

            id_metrics = eval_split(f_idtest, idtest_y)
            ood_metrics = eval_split(f_oodtest, oodtest_y)
            bb_results[arm] = {"id": id_metrics, "ood": ood_metrics}
            print(f"  {arm:10s} ID  acc={id_metrics['acc']:.3f} auc={id_metrics['auc']:.3f}   "
                  f"OOD acc={ood_metrics['acc']:.3f} auc={ood_metrics['auc']:.3f}   ({time.time()-t0:.1f}s)")
        results["backbones"][bb_key] = bb_results
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_path = os.path.join(RESULTS_DIR, f"results_{tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-idtest", type=int, default=1500)
    p.add_argument("--n-oodtest", type=int, default=1500)
    p.add_argument("--backbone", nargs="+", default=["resnet50"], choices=["resnet50", "dinov2"])
    args = p.parse_args()
    run(args)
