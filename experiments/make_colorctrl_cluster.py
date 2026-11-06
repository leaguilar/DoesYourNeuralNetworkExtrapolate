"""make_colorctrl_cluster.py -- generate the colorctrl (fixed-recolor negative
control) arrays on the cluster directly from the already-materialized raw
arrays.

Pure numpy (no torchstain / no wilds), so it runs in the same minimal cluster
env as the training job. The
colorctrl transform is a SINGLE fixed affine remap of RGB space, applied
identically to every image (not a per-image normalization), matched in form
to phi but NOT adaptive -- the negative control that should NOT close the
cross-hospital gap. Definition is copied verbatim from
histopath_stain_transforms.ColorControlPsi so the two stay identical.

Cache location: $CAMELYON_CACHE, else ../data/camelyon_cache relative to this
file (the same default as prepare_camelyon_arrays.py and
train_camelyon17_scratch.py).
"""

import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("CAMELYON_CACHE",
                       os.path.join(_HERE, "..", "data", "camelyon_cache"))
SEED = 20260724


def build_map(seed=SEED, mix_eps=0.12, gain_range=(0.85, 1.15), shift_range=(-20.0, 20.0)):
    rng = np.random.RandomState(seed)
    mix = np.eye(3) + mix_eps * rng.uniform(-1.0, 1.0, size=(3, 3))
    gain = rng.uniform(gain_range[0], gain_range[1], size=3)
    M = mix * gain[None, :]
    shift = rng.uniform(shift_range[0], shift_range[1], size=3)
    return M, shift


def apply_map(imgs, M, shift, chunk=4000):
    """Chunked over the batch axis to keep memory bounded (a full float64
    copy of the train array would be ~13 GB and gets OOM-killed)."""
    shp = imgs.shape
    Mf = M.T.astype(np.float32)
    sf = shift.astype(np.float32)
    out = np.empty(shp, dtype=np.uint8)
    for s in range(0, len(imgs), chunk):
        c = imgs[s:s + chunk].astype(np.float32).reshape(-1, 3)
        c = c @ Mf + sf
        np.clip(c, 0, 255, out=c)
        out[s:s + chunk] = c.reshape((-1,) + shp[1:]).astype(np.uint8)
    return out


def main():
    M, shift = build_map()
    for split in ("train", "idtest", "oodtest"):
        src = os.path.join(CACHE, f"x_{split}_raw.npy")
        dst = os.path.join(CACHE, f"x_{split}_colorctrl.npy")
        arr = np.load(src)
        out = apply_map(arr, M, shift)
        np.save(dst, out)
        print(f"{split}: {arr.shape} -> {dst}")
    print("colorctrl arrays written.")


if __name__ == "__main__":
    main()
