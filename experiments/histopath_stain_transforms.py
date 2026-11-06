"""
histopath_stain_transforms.py — phi (Macenko stain normalization) and the
matched negative control for expfm_v_histopath_stain.py.

phi = Macenko H&E stain normalization (torchstain, numpy backend). Macenko's
method estimates, PER IMAGE, the two dominant stain vectors (hematoxylin,
eosin) in optical-density space and re-expresses the image in terms of a
single fixed REFERENCE stain matrix. It is "principled" in the domain sense:
it encodes the physical Beer-Lambert optical-density model of light
absorption by H&E dye and is the standard pathology-community preprocessing
for cross-scanner / cross-lab color harmonization (Macenko et al., ISBI 2009).

Critically, phi is DATA-ADAPTIVE: it looks at each input image's own stain
statistics and maps them onto a common target, so two images with different
raw stain appearance (e.g. from two different hospitals) are pulled toward
the SAME reference appearance.

colorctrl = the negative control. It is also a fixed, pixel-wise, affine
remap of RGB space (same functional form as phi: a 3x3 linear remix + additive
shift, clipped to [0, 255]), so it is not a trivial no-op and does perturb
appearance -- but it is a single FIXED map, generated once from a random seed
and applied IDENTICALLY to every image regardless of that image's own color
statistics or hospital of origin. It is not derived from any stain/optical
model and does not adapt to the input, so it cannot equalize a systematic
cross-hospital shift in average stain appearance the way phi does.
"""

from __future__ import annotations

import numpy as np
import torchstain


class MacenkoPhi:
    """phi: canonicalize H&E stain appearance via the Macenko optical-density
    stain-separation model. The reference (target) stain matrix is fit ONCE,
    on a single representative patch from the TRAINING domain only; every
    image in every split/arm is then remapped onto that same fixed target.
    """

    def __init__(self, reference_patch: np.ndarray):
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
        self.normalizer.fit(np.asarray(reference_patch))
        self.n_ok = 0
        self.n_fail = 0

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """img: (H, W, 3) uint8 RGB -> stain-normalized uint8 RGB.

        On the rare Macenko degeneracy (near-blank / low-tissue patch, where
        optical-density thresholding leaves too few pixels to fit stain
        vectors), fall back to the raw image and count the failure; this
        records how often phi is even well-defined.
        """
        try:
            out, _, _ = self.normalizer.normalize(np.asarray(img), stains=False)
            self.n_ok += 1
            return out
        except Exception:
            self.n_fail += 1
            return np.asarray(img)


class ColorControlPsi:
    """NEGATIVE CONTROL: one fixed random affine remap of RGB space (linear
    channel mix + additive shift), same map for every image, every split,
    every hospital. Matched in FORM to phi (both are pixel-wise linear-ish
    remaps of the 3-channel color space) but not adaptive and not derived
    from any domain (optical-density / stain) model.
    """

    def __init__(self, seed: int = 20260724, mix_eps: float = 0.12,
                 gain_range=(0.85, 1.15), shift_range=(-20.0, 20.0)):
        rng = np.random.RandomState(seed)
        mix = np.eye(3) + mix_eps * rng.uniform(-1.0, 1.0, size=(3, 3))
        gain = rng.uniform(gain_range[0], gain_range[1], size=3)
        self.M = mix * gain[None, :]
        self.shift = rng.uniform(shift_range[0], shift_range[1], size=3)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.asarray(img).astype(np.float64).reshape(-1, 3)
        x = x @ self.M.T + self.shift
        x = np.clip(x, 0, 255)
        return x.reshape(np.asarray(img).shape).astype(np.uint8)


class ChanNormControl:
    """ADAPTIVE-BUT-STAIN-AGNOSTIC CONTROL. Per-image, per-channel mean/std
    standardization, then remapped to a fixed target (mean, std) and clipped
    to [0, 255]. Like Macenko phi it adapts to EACH image's own statistics
    (so it is not a trivial fixed map the model can invert once), but unlike
    phi it uses NO optical-density / H&E stain model: it only rescales each
    RGB channel's brightness/contrast, it does not align the stain BASIS
    (the cross-scanner hue rotation in optical-density space). It therefore
    isolates the first-principles content of phi: if phi closes the
    cross-hospital gap and this generic per-image normalization does not, the
    gap closure is due to the stain-separation model, not to "any per-image
    color normalization."
    """

    def __init__(self, target_mean: float = 128.0, target_std: float = 48.0, eps: float = 1e-6):
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.eps = float(eps)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.asarray(img).astype(np.float64)
        mu = x.mean(axis=(0, 1), keepdims=True)
        sd = x.std(axis=(0, 1), keepdims=True)
        z = (x - mu) / (sd + self.eps)
        out = self.target_mean + self.target_std * z
        return np.clip(out, 0, 255).astype(np.uint8)


ARM_RAW = "raw"
ARM_STAINNORM = "stainnorm"
ARM_CHANNORM = "channorm"
ARM_COLORCTRL = "colorctrl"
ARMS = (ARM_RAW, ARM_STAINNORM, ARM_CHANNORM, ARM_COLORCTRL)
