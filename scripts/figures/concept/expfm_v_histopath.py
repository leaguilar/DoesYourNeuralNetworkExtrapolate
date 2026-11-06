"""Exp 2.4 (Camelyon17) concept figure: a fixed stain map turns cross-hospital
extrapolation into interpolation.

Three panels, one row:

  (a) Real 96x96 H&E patches, one per training hospital plus the held-out
      hospital, raw on top and after phi below. The hospital colour cast is
      visible in the raw row and gone in the phi row.
  (b) The mechanism. Patch-mean colour, whitened by the training cloud's own
      covariance, so distances read directly in training-sd units. Raw: the
      held-out hospital sits displaced outside the training support
      (extrapolation). After phi: it nests inside it (interpolation).
  (c) The consequence. OOD AUC per arm, all five seeds overlaid on the mean.

phi = Macenko stain normalisation, from the Beer-Lambert optical-density model
OD = -log((I+1)/I0). The reference stain basis W_ref is fit ONCE on a single
training patch (x_train_raw[0]); it sees no OOD image and no OOD statistic.

Honest reading: `channorm` (per-image channel standardisation) matches phi, so
phi is not uniquely capable. The arm that fails is `colorctrl`, one fixed
random RGB affine applied identically to every image, which is matched to phi
in form but non-adaptive.

Run as a script:
    conda run -n data_analysis python scripts/plot_concept.py --exp histopath
Run in VS Code:
    open this file, click 'Run Cell' through the # %% blocks.
"""

# %% imports
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Circle, Rectangle

# Make sibling modules importable in both shell-import and cell-mode contexts.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _style import B_NEAR, B_OOD, B_TRAIN, C_EXP, C_OOD, C_TRAIN, RC

_SRC = Path(__file__).resolve().parents[3]
OUT = _SRC / "figures" / "exp2_4_histopath_concept.pdf"
# Compact main-text variant: the patch strip plus the support-coverage
# scatters, i.e. this figure minus its AUC strip, drawn at its final printed
# size (~0.40\linewidth) so it is never downscaled.
OUT_PANEL = _SRC / "figures" / "exp2_4_histopath_panel.pdf"
CACHE_DIR = _SRC / "data" / "camelyon_cache"
RESULTS_DIR = _SRC / "results" / "expfm_v_histopath_scratch"
# Small derived artefact so the figure rebuilds without re-scanning the 11 GB
# patch cache. Regenerated automatically from the cache when absent.
STATS_NPZ = _SRC / "results" / "expfm_v_histopath_scratch" / "concept_colour_stats.npz"

# Patch strip: one representative patch per hospital (each is the patch closest
# to its hospital's mean patch colour), plus one held-out-hospital patch.
# Indices into x_train_*.npy / x_oodtest_*.npy; hospital labels reproduced from
# metadata.csv with the frozen sampling seed 20260724.
STRIP_TRAIN = [(54150, 0), (42437, 3), (6396, 4)]
STRIP_OOD = 9345

# Arms, ordered by mean OOD AUC so the story reads left to right.
ARM_ORDER = ["colorctrl", "raw", "raw+caug", "stainnorm", "channorm"]
ARM_LABELS = {
    "colorctrl": "colorctrl",
    "raw":       "raw",
    "raw+caug":  "raw+caug",
    "stainnorm": "stainnorm ($\\varphi$)",
    "channorm":  "channorm",
}
ARM_COLORS = {
    "colorctrl": B_OOD,
    "raw":       "#8a8a8a",
    "raw+caug":  B_NEAR,
    "stainnorm": B_TRAIN,
    "channorm":  C_EXP,
}
# Mean-value label side, staggered so adjacent near-ceiling arms do not collide.
LABEL_ABOVE = {"colorctrl": True, "raw": True, "raw+caug": False,
               "stainnorm": False, "channorm": True}

# chi^2 quantiles for the nominal 95% region (2 and 3 dof).
_CHI2_95_2D = 5.991464547107979
_CHI2_95_3D = 7.814727903251179


# %% derived colour statistics --------------------------------------------------
def _patch_means(path, chunk=2000):
    """Per-patch mean RGB, streamed so the 1.7 GB arrays are never resident."""
    a = np.load(path, mmap_mode="r")
    out = np.empty((a.shape[0], 3), dtype=np.float64)
    for i in range(0, a.shape[0], chunk):
        out[i:i + chunk] = np.asarray(a[i:i + chunk], dtype=np.float64).mean(axis=(1, 2))
    return out


def _support_geometry(train_pm, ood_pm):
    """Whiten patch-mean colour by the TRAINING cloud's covariance.

    In these coordinates the training cloud is isotropic with unit sd, so every
    distance is quoted in training-sd units and the two representations are
    directly comparable. Axis 1 is the hospital-shift direction; axis 2 is the
    orthogonal direction carrying the most remaining spread.
    """
    mu = train_pm.mean(axis=0)
    cov = np.cov(train_pm.T)
    evals, evecs = np.linalg.eigh(cov)
    whiten = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    delta = ood_pm.mean(axis=0) - mu
    e1 = whiten @ delta
    e1 /= np.linalg.norm(e1)

    tw = (train_pm - mu) @ whiten
    ow = (ood_pm - mu) @ whiten
    resid = np.vstack([tw, ow]) @ (np.eye(3) - np.outer(e1, e1))
    _, _, vt = np.linalg.svd(resid, full_matrices=False)
    e2 = vt[0] - (vt[0] @ e1) * e1
    e2 /= np.linalg.norm(e2)

    basis = np.stack([e1, e2])
    train_2d = tw @ basis.T
    ood_2d = ow @ basis.T

    # Pooled per-channel sd of patch-mean colour, for the plain |dmu| readout.
    n_t, n_o = len(train_pm), len(ood_pm)
    sd_pool = np.sqrt((train_pm.var(axis=0) * n_t + ood_pm.var(axis=0) * n_o) / (n_t + n_o))

    return dict(
        train_2d=train_2d,
        ood_2d=ood_2d,
        shift=float(np.linalg.norm(ow.mean(axis=0))),          # Mahalanobis, 3-D
        dmu=float(np.linalg.norm(delta)),                       # raw RGB units
        dmu_sd=float(np.linalg.norm(delta) / sd_pool.mean()),   # per-channel sd units
        cov_ood_2d=float(((ood_2d ** 2).sum(axis=1) < _CHI2_95_2D).mean()),
        cov_train_2d=float(((train_2d ** 2).sum(axis=1) < _CHI2_95_2D).mean()),
        cov_ood_3d=float(((ow ** 2).sum(axis=1) < _CHI2_95_3D).mean()),
        cov_train_3d=float(((tw ** 2).sum(axis=1) < _CHI2_95_3D).mean()),
    )


def _build_stats(n_scatter=1500, seed=0):
    """Recompute everything panel (a)/(b) needs from the patch cache."""
    rng = np.random.RandomState(seed)
    blob = {}

    for tag, train_name, ood_name in [("raw", "x_train_raw", "x_oodtest_raw"),
                                      ("phi", "x_train_stainnorm", "x_oodtest_stainnorm")]:
        tr = _patch_means(CACHE_DIR / f"{train_name}.npy")
        oo = _patch_means(CACHE_DIR / f"{ood_name}.npy")
        g = _support_geometry(tr, oo)
        it = rng.choice(len(g["train_2d"]), size=min(n_scatter, len(g["train_2d"])), replace=False)
        io = rng.choice(len(g["ood_2d"]), size=min(n_scatter, len(g["ood_2d"])), replace=False)
        blob[f"{tag}_train_2d"] = g["train_2d"][it].astype(np.float32)
        blob[f"{tag}_ood_2d"] = g["ood_2d"][io].astype(np.float32)
        for k in ("shift", "dmu", "dmu_sd", "cov_ood_2d", "cov_train_2d",
                  "cov_ood_3d", "cov_train_3d"):
            blob[f"{tag}_{k}"] = np.float64(g[k])
        blob[f"{tag}_mu_train"] = tr.mean(axis=0)
        blob[f"{tag}_mu_ood"] = oo.mean(axis=0)

    # Patch strip: the actual images the network saw, before and after phi.
    xtr_raw = np.load(CACHE_DIR / "x_train_raw.npy", mmap_mode="r")
    xtr_phi = np.load(CACHE_DIR / "x_train_stainnorm.npy", mmap_mode="r")
    xoo_raw = np.load(CACHE_DIR / "x_oodtest_raw.npy", mmap_mode="r")
    xoo_phi = np.load(CACHE_DIR / "x_oodtest_stainnorm.npy", mmap_mode="r")
    idx = [i for i, _ in STRIP_TRAIN]
    blob["strip_raw"] = np.stack([np.asarray(xtr_raw[i]) for i in idx]
                                 + [np.asarray(xoo_raw[STRIP_OOD])]).astype(np.uint8)
    blob["strip_phi"] = np.stack([np.asarray(xtr_phi[i]) for i in idx]
                                 + [np.asarray(xoo_phi[STRIP_OOD])]).astype(np.uint8)
    blob["strip_centre"] = np.array([c for _, c in STRIP_TRAIN] + [2], dtype=np.int64)

    STATS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(STATS_NPZ, **blob)
    return blob


def _load_stats():
    if STATS_NPZ.is_file():
        with np.load(STATS_NPZ) as z:
            return {k: z[k] for k in z.files}
    if not CACHE_DIR.is_dir():
        return None
    print(f"  [histopath] building {STATS_NPZ.name} from the patch cache (one-off, ~4 min)")
    return _build_stats()


def _load_auc():
    """{arm: (mean_ood, sd_ood, [per-seed ood]), 'id': mean ID AUC} at the ID-selected epoch."""
    if not RESULTS_DIR.is_dir():
        return None
    per_arm, id_auc = {}, []
    for p in sorted(RESULTS_DIR.glob("scratch_*_seed*.json")):
        with open(p) as f:
            r = json.load(f)
        arm = "raw+caug" if r.get("colour_aug") else r.get("arm")
        sel = r["id_selected"]
        per_arm.setdefault(arm, []).append(sel["ood"]["auc"])
        id_auc.append(sel["id"]["auc"])
    if not per_arm:
        return None
    out = {a: (float(np.mean(v)), float(np.std(v, ddof=1)), np.asarray(v))
           for a, v in per_arm.items()}
    out["_id"] = float(np.mean(id_auc))
    return out


# %% panel painters -------------------------------------------------------------
def _mosaic(strip_raw, strip_phi, gutter=7, split=22):
    """Compose the 2 x 4 patch strip into one image, with a wider OOD gap."""
    n = strip_raw.shape[0]
    h = w = strip_raw.shape[1]
    xs = []
    x = 0
    for j in range(n):
        xs.append(x)
        x += w + (split if j == n - 2 else gutter)
    total_w = xs[-1] + w
    canvas = np.full((2 * h + gutter, total_w, 3), 255, dtype=np.uint8)
    for j in range(n):
        canvas[0:h, xs[j]:xs[j] + w] = strip_raw[j]
        canvas[h + gutter:2 * h + gutter, xs[j]:xs[j] + w] = strip_phi[j]
    return canvas, xs, h, w, gutter


def _panel_a(ax, st):
    strip_raw, strip_phi = st["strip_raw"], st["strip_phi"]
    centres = st["strip_centre"]
    canvas, xs, h, w, g = _mosaic(strip_raw, strip_phi)

    ax.imshow(canvas, interpolation="nearest")
    ax.set_anchor("N")
    ax.set_xlim(-34, canvas.shape[1] + 2)
    ax.set_ylim(canvas.shape[0] + 4, -50)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)

    for j, c in enumerate(centres):
        is_ood = bool(c == 2)
        col = B_OOD if is_ood else B_TRAIN
        for y0 in (0, h + g):
            ax.add_patch(Rectangle((xs[j] - 0.5, y0 - 0.5), w, h, fill=False,
                                   ec=col, lw=1.2 if is_ood else 0.8,
                                   zorder=5, clip_on=False))
        ax.text(xs[j] + w / 2, -5, f"c{c}", ha="center", va="bottom",
                fontsize=7, color=col, fontweight="bold" if is_ood else "normal")

    ax.text((xs[0] + xs[2] + w) / 2, -26, "training hospitals", ha="center",
            va="bottom", fontsize=7, color=B_TRAIN)
    ax.text(xs[-1] + w / 2, -26, "held out", ha="center", va="bottom",
            fontsize=7, color=B_OOD, fontweight="bold")

    ax.text(-8, h / 2, "raw", ha="right", va="center", fontsize=7.5, rotation=90)
    ax.text(-8, h + g + h / 2, "after $\\varphi$", ha="right", va="center",
            fontsize=7.5, rotation=90)

    sep = (xs[-1] + xs[-2] + w) / 2
    ax.plot([sep, sep], [-2, canvas.shape[0]], ls=(0, (2, 2)), lw=0.7,
            color="#999999", zorder=6, clip_on=False)


def _panel_a_note(ax):
    ax.axis("off")
    ax.text(0.0, 0.98,
            "$\\varphi$ = Macenko stain normalisation, from the\n"
            "Beer-Lambert optical-density model\n"
            "$\\mathrm{OD}=-\\log((I{+}1)/I_0)$. The reference stain\n"
            "basis $W_\\mathrm{ref}$ is fit ONCE on a single training\n"
            "patch: it sees no OOD image, no OOD statistic.",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3,
            color="#444444", linespacing=1.45)


def _scatter_support(ax, tr, oo, shift, cov_ood, title, verdict, verdict_col):
    ax.add_patch(Circle((0, 0), np.sqrt(_CHI2_95_2D), fill=True, fc=C_TRAIN,
                        ec=B_TRAIN, lw=0.9, ls=(0, (3, 2)), alpha=0.55, zorder=1))
    ax.scatter(tr[:, 0], tr[:, 1], s=1.6, c=B_TRAIN, alpha=0.30, lw=0,
               rasterized=True, zorder=2)
    ax.scatter(oo[:, 0], oo[:, 1], s=1.4, c=B_OOD, alpha=0.26, lw=0,
               rasterized=True, zorder=3)
    ax.scatter([0], [0], s=26, marker="+", c=B_TRAIN, lw=1.3, zorder=6)
    ax.scatter([oo[:, 0].mean()], [oo[:, 1].mean()], s=26, marker="x",
               c=B_OOD, lw=1.3, zorder=6)
    ax.annotate("", xy=(shift, 0), xytext=(0, 0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=0.9,
                                shrinkA=0, shrinkB=0))
    ax.text(max(shift / 2, 0.35), 0.6, f"{shift:.1f} sd",
            ha="center" if shift > 1 else "left", va="bottom",
            fontsize=6.6, color="#1a1a1a", fontweight="bold")
    ax.set_xlim(-4.2, 7.4)
    ax.set_ylim(-4.4, 4.4)
    ax.set_yticks([-3, 0, 3])
    ax.set_xticks([-3, 0, 3, 6])
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.text(0.985, 0.04, f"{verdict}\n{cov_ood:.0%} of c2 inside", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.6, color=verdict_col,
            fontweight="bold", linespacing=1.25)


def _panel_c(ax, auc):
    rng = np.random.RandomState(7)
    arms = [a for a in ARM_ORDER if a in auc]
    for i, arm in enumerate(arms):
        m, sd, seeds = auc[arm]
        col = ARM_COLORS[arm]
        ax.add_patch(Rectangle((i - 0.30, m - sd), 0.60, 2 * sd, fc=col,
                               alpha=0.16, ec="none", zorder=1))
        ax.plot([i - 0.34, i + 0.34], [m, m], color=col, lw=2.0, solid_capstyle="butt",
                zorder=3)
        jit = rng.uniform(-0.15, 0.15, size=len(seeds))
        ax.scatter(i + jit, seeds, s=9, facecolor="white", edgecolor=col,
                   lw=0.8, zorder=4)
        if LABEL_ABOVE.get(arm, m < 0.9):
            gap = 0.030 if m > 0.9 else 0.018   # clear the ID-AUC ceiling line
            ax.text(i, max(seeds) + gap, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=6.3, color=col)
        else:
            ax.text(i, min(seeds) - 0.020, f"{m:.3f}", ha="center", va="top",
                    fontsize=6.3, color=col)

    ax.axhline(auc["_id"], color="#777777", lw=0.8, ls=(0, (4, 2)), zorder=0)
    ax.text(-0.45, auc["_id"] + 0.007,
            f"ID AUC $\\approx$ {auc['_id']:.3f}",
            ha="left", va="bottom", fontsize=6.3, color="#777777")
    ax.axhline(0.5, color="#bbbbbb", lw=0.7, ls=":", zorder=0)
    ax.text(len(arms) - 0.5, 0.506, "chance", ha="right", va="bottom",
            fontsize=6.2, color="#999999")

    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_ylim(0.47, 1.065)
    ax.set_xticks(range(len(arms)))
    # At 32 degrees the long arm names ("raw+caug", "stainnorm (phi)") stayed
    # wide enough to collide with their neighbours at printed size.
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=6.4,
                       rotation=48, ha="right", rotation_mode="anchor")
    for tick, arm in zip(ax.get_xticklabels(), arms):
        tick.set_color(ARM_COLORS[arm])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("OOD AUC, held-out hospital c2", fontsize=7.5)
    ax.set_title("ResNet-18 from scratch\n5 seeds, ID-selected epoch",
                 fontsize=7.2, pad=3)
    ax.tick_params(axis="x", length=0, pad=1)


# %% render — pure: builds and returns fig (or None when no data), no save, no close
def render():
    st = _load_stats()
    auc = _load_auc()
    if st is None or auc is None:
        print("  [histopath] missing patch cache or scratch results, skipping")
        return None

    with plt.rc_context(RC):
        # 5.42in + the tight-bbox overhang lands the saved PDF on the NeurIPS
        # \textwidth of 5.5in, so \includegraphics[width=\linewidth] is 1:1.
        fig = plt.figure(figsize=(5.42, 2.85))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.46, 0.84, 0.86], wspace=0.46,
                              left=0.045, right=0.995, top=0.860, bottom=0.185)

        gs_a = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0],
                                       height_ratios=[0.98, 0.55], hspace=0.05)
        ax_a = fig.add_subplot(gs_a[0, 0])
        ax_a_note = fig.add_subplot(gs_a[1, 0])
        _panel_a(ax_a, st)
        _panel_a_note(ax_a_note)

        gs_b = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.55)
        ax_b0 = fig.add_subplot(gs_b[0, 0])
        ax_b1 = fig.add_subplot(gs_b[1, 0])

        _scatter_support(
            ax_b0, st["raw_train_2d"], st["raw_ood_2d"],
            float(st["raw_shift"]), float(st["raw_cov_ood_2d"]),
            "raw", "EXTRAPOLATION", B_OOD)
        _scatter_support(
            ax_b1, st["phi_train_2d"], st["phi_ood_2d"],
            float(st["phi_shift"]), float(st["phi_cov_ood_2d"]),
            "after $\\varphi$", "INTERPOLATION", B_TRAIN)
        ax_b0.set_xticklabels([])
        ax_b1.set_xlabel("hospital-shift direction (training sd)", fontsize=6.8,
                         labelpad=1)
        ax_b0.set_ylabel("orthog.", fontsize=6.8, labelpad=1)
        ax_b1.set_ylabel("orthog.", fontsize=6.8, labelpad=1)

        ax_c = fig.add_subplot(gs[0, 2])
        _panel_c(ax_c, auc)

        for ax, dx in ((ax_a, -0.01), (ax_b0, -0.26), (ax_c, -0.30)):
            lab = {id(ax_a): "(a)", id(ax_b0): "(b)", id(ax_c): "(c)"}[id(ax)]
            ax.annotate(lab, xy=(dx, 1.0), xycoords="axes fraction",
                        xytext=(0, 12), textcoords="offset points",
                        fontsize=9, fontweight="bold", va="center", ha="left")

        # Shared legend for panel (b), spanning the figure foot.
        fig.legend(
            handles=[
                plt.Line2D([], [], marker="o", ls="", ms=3, mfc=B_TRAIN, mec="none",
                           label="training hospitals c0, c3, c4"),
                plt.Line2D([], [], marker="o", ls="", ms=3, mfc=B_OOD, mec="none",
                           label="held-out hospital c2"),
                plt.Line2D([], [], color=B_TRAIN, lw=0.9, ls=(0, (3, 2)),
                           label="training 95% region"),
            ],
            loc="lower center", bbox_to_anchor=(0.47, -0.045), ncol=3,
            frameon=False, fontsize=6.4, handletextpad=0.5,
            borderpad=0.0, columnspacing=1.6)
    return fig


# %% build — CLI / shell entry point
def build(output_dir=None):
    out = Path(output_dir) / OUT.name if output_dir else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = render()
        if fig is None:
            return None
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved → {out}")
    return out


# %% compact main-text panel ----------------------------------------------------
# The compact cut mirrors the appendix figure minus its per-arm AUC strip: the
# patch strip on the left, the two colour scatters stacked on the right. Only
# two of the three training hospitals are shown, so the patches stay large
# enough to read at 0.40\linewidth; the scatters still use every hospital.
PANEL_COLS = [0, 1, 3]          # indices into strip_raw/strip_phi: c0, c3, c2
PANEL_CROP = 78                 # centre crop (px) so nuclei survive the shrink

# Figure geometry, in inches, laid out by hand: at this size a gridspec cannot
# be trusted to leave room for the annotations.
_W, _H = 2.16, 1.82
_STRIP_X0, _STRIP_W = 0.150, 0.975      # patch mosaic
_STRIP_TOP = 0.150                       # headroom above the mosaic, for labels
_SC_X0, _SC_W = 1.245, 0.900             # scatters
_SC_TOP = 0.135                          # title above the upper box
_SC_GAP = 0.275                          # coverage label + title between the boxes
# The scatter box is drawn at the data aspect ratio, so the training 95% region
# reads as a circle; the y range is trimmed to what the clouds occupy.
_XLIM = (-3.45, 6.85)
_YLIM = (-2.95, 2.95)


def _compact_mosaic_shape(cols=PANEL_COLS, crop=PANEL_CROP, gutter=5, split=13):
    """(height, width) in pixels of the mosaic `_compact_mosaic` will build."""
    n = len(cols)
    return 2 * crop + gutter, n * crop + (n - 2) * gutter + split


def _compact_mosaic(strip_raw, strip_phi, cols, crop, gutter=5, split=13):
    """2 x len(cols) patch mosaic; returns the canvas and column offsets."""
    n = len(cols)
    full = strip_raw.shape[1]
    o = (full - crop) // 2
    sl = slice(o, o + crop)
    h = w = crop
    xs, x = [], 0
    for j in range(n):
        xs.append(x)
        x += w + (split if j == n - 2 else gutter)
    canvas = np.full((2 * h + gutter, xs[-1] + w, 3), 255, dtype=np.uint8)
    for j, k in enumerate(cols):
        canvas[0:h, xs[j]:xs[j] + w] = strip_raw[k][sl, sl]
        canvas[h + gutter:2 * h + gutter, xs[j]:xs[j] + w] = strip_phi[k][sl, sl]
    return canvas, xs, h, w, gutter


def _compact_strip(ax, st, cols=PANEL_COLS, crop=PANEL_CROP):
    """Panel (a) of the appendix figure, shrunk: real patches, raw over phi."""
    centres = [int(st["strip_centre"][k]) for k in cols]
    canvas, xs, h, w, g = _compact_mosaic(st["strip_raw"], st["strip_phi"],
                                          cols, crop)
    cw, ch = canvas.shape[1], canvas.shape[0]
    # Headroom above the image, in data (pixel) units, for the two label rows.
    head = ch * (_STRIP_TOP / (_STRIP_W * ch / cw))

    # interpolation="none" makes the PDF backend embed the patches at their
    # native pixel resolution (~260 dpi at this printed size) instead of
    # resampling them down to the 72 dpi vector grid, which smears the nuclei.
    ax.imshow(canvas, interpolation="none", zorder=2)
    ax.set_xlim(-0.5, cw - 0.5)
    ax.set_ylim(ch - 0.5, -head)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    for j, c in enumerate(centres):
        is_ood = c == 2
        col = B_OOD if is_ood else B_TRAIN
        for y0 in (0, h + g):
            ax.add_patch(Rectangle((xs[j] - 0.5, y0 - 0.5), w, h, fill=False,
                                   ec=col, lw=1.1 if is_ood else 0.8,
                                   zorder=5, clip_on=False))
        ax.text(xs[j] + w / 2, -0.10 * head, f"c{c}", ha="center", va="bottom",
                fontsize=6.4, color=col, fontweight="bold" if is_ood else "normal")

    ax.text((xs[0] + xs[-2] + w) / 2, -0.62 * head, "training", ha="center",
            va="bottom", fontsize=6.4, color=B_TRAIN)
    ax.text(xs[-1] + w / 2, -0.62 * head, "held out", ha="center", va="bottom",
            fontsize=6.4, color=B_OOD, fontweight="bold")

    ax.text(-0.055 * cw, h / 2, "raw", ha="right", va="center", fontsize=6.6,
            rotation=90)
    ax.text(-0.055 * cw, h + g + h / 2, "$\\varphi$", ha="right", va="center",
            fontsize=7.6)

    sep = (xs[-1] + xs[-2] + w) / 2
    ax.plot([sep, sep], [-0.16 * head, ch - 0.5], ls=(0, (2, 1.8)), lw=0.6,
            color="#aaaaaa", zorder=6, clip_on=False)


def _compact_scatter(ax, tr, oo, shift, cov_ood, title, cov_col):
    """One small whitened-support scatter, drawn without axis numbers."""
    ax.add_patch(Circle((0, 0), np.sqrt(_CHI2_95_2D), fill=True, fc=C_TRAIN,
                        ec=B_TRAIN, lw=0.8, ls=(0, (2.5, 1.8)), alpha=0.75, zorder=1))
    ax.scatter(tr[:, 0], tr[:, 1], s=1.0, c=B_TRAIN, alpha=0.40, lw=0,
               rasterized=True, zorder=2)
    ax.scatter(oo[:, 0], oo[:, 1], s=1.0, c=B_OOD, alpha=0.40, lw=0,
               rasterized=True, zorder=3)
    ax.annotate("", xy=(shift, 0), xytext=(0, 0), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color="#1a1a1a", lw=0.8,
                                shrinkA=0, shrinkB=0, mutation_scale=6))
    ax.text(shift / 2 if shift > 1 else shift + 0.30, 0.35, f"{shift:.1f} sd",
            ha="center" if shift > 1 else "left", va="bottom",
            fontsize=6.3, color="#1a1a1a", fontweight="bold", zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.7))

    ax.set_title(title, fontsize=7.2, pad=1.8, color="#1a1a1a")
    # The coverage verdict sits outside the box: at this size any text inside
    # would mask the training 95% region it is talking about.
    ax.set_xlabel(f"{cov_ood:.0%} of c2 inside", fontsize=5.9, labelpad=1.6,
                  color=cov_col, fontweight="bold")

    ax.set_xlim(*_XLIM)
    ax.set_ylim(*_YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
        s.set_color("#bbbbbb")


def render_panel():
    """Compact main-text cut: the patch strip plus the support-coverage scatters.

    Sized for ~0.40\\linewidth (2.16in) beside the CDS panel in the cross-domain
    composite, and drawn at final size so nothing is downscaled in LaTeX. The
    per-arm AUC strip of the appendix figure is dropped: those numbers are
    tabulated in the main text.
    """
    st = _load_stats()
    if st is None:
        print("  [histopath_panel] missing colour statistics, skipping")
        return None

    rng = np.random.RandomState(3)
    def thin(a, n=460):
        idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
        return a[idx]

    def rect(x0, y0, w, h):
        return [x0 / _W, y0 / _H, w / _W, h / _H]

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(_W, _H))

        # Right: the two scatters, stacked, drawn at the data aspect ratio so
        # the training 95% region reads as a circle.
        sc_h = _SC_W * (_YLIM[1] - _YLIM[0]) / (_XLIM[1] - _XLIM[0])
        y_top = _H - _SC_TOP - sc_h
        y_bot = y_top - _SC_GAP - sc_h
        # ax0 = fig.add_axes(rect(_SC_X0, y_top, _SC_W, sc_h))
        # ax1 = fig.add_axes(rect(_SC_X0, y_bot, _SC_W, sc_h))

        # _compact_scatter(ax0, thin(st["raw_train_2d"]), thin(st["raw_ood_2d"]),
        #                  float(st["raw_shift"]), float(st["raw_cov_ood_2d"]),
        #                  "raw", B_OOD)
        # _compact_scatter(ax1, thin(st["phi_train_2d"]), thin(st["phi_ood_2d"]),
        #                  float(st["phi_shift"]), float(st["phi_cov_ood_2d"]),
        #                  "after $\\varphi$", B_TRAIN)
        # fig.text((_SC_X0 + _SC_W / 2) / _W, 0.025 / _H,
        #          "whitened patch colour\n(training sd units)", ha="center",
        #          va="bottom", fontsize=5.6, color="#666666", linespacing=1.25)

        # Left: the patch strip, top-aligned with the upper scatter.
        m_h, m_w = _compact_mosaic_shape()
        strip_h = _STRIP_TOP + _STRIP_W * m_h / m_w
        strip_y0 = _H - _SC_TOP - strip_h
        ax_s = fig.add_axes(rect(_STRIP_X0, strip_y0, _STRIP_W, strip_h))
        _compact_strip(ax_s, st)

        # Left foot: what phi is, then the legend for the scatters.
        fig.text(_STRIP_X0 / _W, (strip_y0 - 0.060) / _H,
                 "$\\varphi$ = Macenko stain norm.,\n"
                 "fit on one training patch",
                 ha="left", va="top", fontsize=5.6, color="#555555",
                 linespacing=1.35)
        # fig.legend(
        #     handles=[
        #         plt.Line2D([], [], marker="o", ls="", ms=2.6, mfc=B_TRAIN,
        #                    mec="none", label="training c0, c3, c4"),
        #         plt.Line2D([], [], marker="o", ls="", ms=2.6, mfc=B_OOD,
        #                    mec="none", label="held-out c2"),
        #         plt.Line2D([], [], color=B_TRAIN, lw=0.8, ls=(0, (2.5, 1.8)),
        #                    label="training 95% region"),
        #     ],
        #     loc="lower left", bbox_to_anchor=(_STRIP_X0 / _W, 0.025 / _H),
        #     ncol=1, frameon=False, fontsize=5.8, handlelength=1.3,
        #     handletextpad=0.45, labelspacing=0.30, borderpad=0.0,
        #     borderaxespad=0.0)
    return fig


def build_panel(output_dir=None):
    out = Path(output_dir) / OUT_PANEL.name if output_dir else OUT_PANEL
    out.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        fig = render_panel()
        if fig is None:
            return None
        # dpi only sets the resolution of the rasterized scatter clouds; the
        # page stays vector and 155 x 128 pt (2.16 x 1.78 in).
        fig.savefig(out, bbox_inches="tight", pad_inches=0.01, dpi=600)
        plt.close(fig)
    print(f"  Saved → {out}")
    return out


# %% live preview — runs only under "Run Cell" or direct script execution
if __name__ == "__main__":
    fig = render()
    fig
