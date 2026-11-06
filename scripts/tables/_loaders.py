"""Data loaders for paper tables. All return per-experiment runs (lists of dicts)
or summary objects already aggregated upstream."""

import json
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: str) -> dict[str, list[dict]]:
    """Load JSON files from results/exp*/ subdirs; keep latest per (exp_id, seed)."""
    root = Path(results_dir)
    files = sorted(root.glob("exp*.json")) + sorted(root.glob("*/exp*.json"))
    by_key = {}
    for p in files:
        parts = p.stem.split("__")
        if len(parts) < 3:
            continue
        by_key[(parts[0], parts[1])] = p
    grouped = defaultdict(list)
    for p in by_key.values():
        with open(p) as f:
            data = json.load(f)
        grouped[data["experiment_id"]].append(data)
    return dict(grouped)


def load_hero_results(results_dir: str) -> list[dict]:
    """Load hero experiment JSONs from results_dir/hero/."""
    hero_dir = Path(results_dir) / "hero"
    if not hero_dir.is_dir():
        return []
    records = []
    for p in sorted(hero_dir.glob("hero__*.json")):
        with open(p) as f:
            records.append(json.load(f))
    return records


def load_exp3_3_results(results_dir: str) -> list[dict]:
    """Load Exp 3.3 architecture-invariance JSONs (transformer / mamba / s4 across PE x period x seed).

    Looks in results_dir/exp3_3/, then ../../results/exp3_3/.
    """
    candidates = [
        Path(results_dir) / "exp3_3",
        Path(results_dir).parent.parent / "results" / "exp3_3",
    ]
    d = next((c for c in candidates if c.is_dir()), None)
    if d is None:
        return []
    out = []
    for p in sorted(d.glob("exp3_3__*.json")):
        with open(p) as f:
            r = json.load(f)
        # Recover pe_type for SSMs (train.py writes None for non-transformers).
        # Filename: exp3_3__<model>__[pe]__P<P>__seed<S>__<ts>.json
        # When pe_type == "no_pe", non-transformer files drop the pe segment.
        parts = p.stem.split("__")
        if parts[2].startswith("P"):
            r["model_type"] = parts[1]
            r["pe_type"]    = "no_pe"
        else:
            r["model_type"] = parts[1]
            r["pe_type"]    = parts[2]
        out.append(r)
    return out


def load_exp2_3_full_results(results_dir: str) -> list[dict]:
    """Per-cell summaries from `results_dir/exp2_3_full/<cell>/`."""
    d = Path(results_dir) / "exp2_3_full"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*/summary.json")):
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            out.extend(data)
        else:
            out.append(data)
    return out


def load_exp2_3_learned_sweep(results_dir: str) -> list[dict]:
    p = Path(results_dir) / "exp2_3_learned_sweep" / "summary.json"
    if not p.is_file():
        return []
    with open(p) as f:
        return json.load(f)


def load_exp2_3_window_sweep(results_dir: str) -> list[dict]:
    p = Path(results_dir) / "exp2_3_window_sweep" / "summary.json"
    if not p.is_file():
        return []
    with open(p) as f:
        return json.load(f)


def load_exp2_3_fft(results_dir: str) -> dict | None:
    """Top CDS-specific spectral peaks for the Exp 2.3 paper section."""
    p = Path(results_dir) / "exp2_3_fft" / "summary.json"
    if not p.is_file():
        return None
    with open(p) as f:
        return json.load(f)


def load_exp2_3_species_clean(results_dir: str) -> list[dict]:
    """Per-seed result rows for the clean-label species sweep (Phase C)."""
    rows = []
    for subdir in ("exp2_3_species_clean_full",
                   "exp2_3_species_clean_ape"):
        d = Path(results_dir) / subdir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("seed*/summary.json")):
            with open(p) as f:
                data = json.load(f)
            rows.extend(data.get("rows", []))
    return rows


def load_histopath_scratch(results_dir: str) -> list[dict]:
    """Camelyon17 from-scratch ResNet-18 runs (`expfm_v_histopath_scratch/`).

    Dedicated loader: the filenames are `scratch_<arch>_<arm>_seed<S>.json`,
    which do not match the `exp*__<seed>__<ts>.json` convention that
    `load_results` globs, so these runs are invisible to it.

    The colour-augmentation arm is written by the experiment as
    `arm == "raw"` plus `colour_aug: true`; it is normalised here to a
    distinct `arm_key == "raw+caug"` so the five arms never collide.
    """
    d = Path(results_dir) / "expfm_v_histopath_scratch"
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("scratch_*_seed*.json")):
        with open(p) as f:
            r = json.load(f)
        r["arm_key"] = "raw+caug" if r.get("colour_aug") else r.get("arm")
        out.append(r)
    return out


def load_histopath(results_dir: str) -> dict | None:
    """Camelyon17 results as `{"scratch": [...]}`, or None when absent.

    Only the from-scratch ResNet-18 runs are reported. The frozen pretrained
    backbone + linear-probe arm (`expfm_v_histopath/results_full_seed0.json`)
    is deliberately NOT loaded: those backbones have unknown pretraining
    exposure and may already have seen histopathology, so the comparison is
    confounded, and it was a single seed. It is not part of the paper.
    """
    scratch = load_histopath_scratch(results_dir)
    if not scratch:
        return None
    return {"scratch": scratch}
