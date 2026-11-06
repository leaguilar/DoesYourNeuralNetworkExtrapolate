#!/usr/bin/env python3
"""
Generate the appendix compute-resources table by extracting wall-clock,
device, and run-count info directly from the result JSONs / log files of
every paper-referenced experiment.

Sources, in order of precedence per experiment:
  1. `wall_time_s` / `gpu_name` / `gpu_mem_peak_gb` written by the train scripts
     that instrument themselves (exp3_3 archinv, exp2_3 cds rows).
  2. `run_resources.json` written by `experiments/run_all.py --log-resources`
     (per-(exp_id, seed) wall_s/cpu_s/ram_mb/gpu_mb under a ResourceTracker).
  3. File-mtime gaps within a per-experiment directory as a fallback estimate
     when neither of the above is available.

Usage (from project root):
    python scripts/resources_tables.py
    python scripts/resources_tables.py --output_dir tables/

Then `python scripts/sync_paper.py --tables-only` copies it into
`paper/tables/table_resources.tex`.
"""

import argparse
import json
import os
import platform
import statistics
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPTS_DIR  = Path(__file__).resolve().parent       # …/scripts/
_SRC_DIR      = _SCRIPTS_DIR.parent                   # …/src/
_PROJECT_ROOT = _SRC_DIR.parent                       # …/0_FeatureEngineering/

DEFAULT_RESULTS_DIR_LOCAL  = _SRC_DIR / "results"     # bulk Tier 1-3 + tor + RW1
DEFAULT_RESULTS_DIR_EULER  = _PROJECT_ROOT / "results"  # exp3_3 primary mirror


# ---------------------------------------------------------------------------
# Per-experiment registry
# ---------------------------------------------------------------------------
# Each entry describes one row in the resources table. `paths` is a list of
# (kind, glob) pairs, where `kind` selects the per-file extractor.
#   - "json_single": one JSON per run, top-level wall_time_s preferred
#   - "json_rows":   one JSON whose `rows` list has per-row train_time_s
#   - "json_old":    one JSON per (exp, seed) without timing — use mtime gaps

@dataclass
class ExpEntry:
    key:         str       # short id (also used as a stable sort key)
    label:       str       # LaTeX-friendly display name
    section_ref: str       # \ref{} target in the paper
    paths:       list      # list[(kind, glob)] — relative to results/
    note:        str = ""  # optional caption note (e.g. "primary only")
    default_venue: str = "local"  # fallback when gpu_name doesn't disambiguate
                                  # ("local" = laptop RTX 3000 / "euler" = cluster)
    paper_seeds: tuple[int, ...] | None = None


REGISTRY: list[ExpEntry] = [
    ExpEntry("1.1",  r"Exp.~1.1: $\sin(x)$ vs.\ Taylor",
             "sec:exp11",
             [("json_single", "exp1_1/exp1_1__seed*.json")],
             default_venue="euler", paper_seeds=(42, 43, 44)),
    ExpEntry("1.2",  r"Exp.~1.2: power law vs.\ exponential",
             "sec:exp12",
             [("json_single", "exp1_2/exp1_2__seed*.json")],
             default_venue="euler"),
    ExpEntry("1.3",  r"Exp.~1.3: Correctness and SINDy battery",
             "app:exp13",
             [("json_single", "exp1_3/exp1_3__seed*.json")],
             default_venue="local", paper_seeds=(7, 42, 123)),
    ExpEntry("rw_mak",  r"Exp.~2.1: mass-action kinetics ODE / vector field",
             "sec:exp21",
             [("json_single", "exp2_1/exp2_1__seed*.json")],
             default_venue="euler"),
    ExpEntry("rw1",  r"Exp.~2.2: Kepler multivariate",
             "sec:exp22",
             [("json_single", "exp2_2/exp2_2__seed*.json")],
             default_venue="euler"),
    ExpEntry("rw3",  r"Exp.~2.3 (CDS): cross-species CDS detection",
             "sec:exp23",
             [("json_rows", "exp2_3_species_clean_full/seed*/summary.json")],
             default_venue="local"),
    ExpEntry("3.1",  r"Exp.~3.1: near-boundary val.\ vs.\ CV",
             "sec:exp31",
             [("json_single", "exp3_1/exp3_1__seed*.json")],
             default_venue="local"),
    ExpEntry("rw2",  r"Exp.~3.3 Architectural invariance: PE $\times$ architecture, primary block",
             "sec:exp33",
             [("json_single_tag:none", "exp3_3/exp3_3__*__seed*.json")],
             note=r"13 (model, PE) $\times$ 4 periods $\times$ 3 seeds (no RoPE on SSMs); fills the non-learned rows of \Cref{tab:exp33_main}",
             default_venue="euler"),
    ExpEntry("rw2_sweepL",
             r"\quad\,$\hookrightarrow$ Sweep~L: period-length sweep, learned PEs",
             "sec:exp33",
             [("json_single_tag:L", "exp3_3/exp3_3__*__seed*.json")],
             note=r"completes the headline grid (\Cref{tab:exp33_main}); together with the primary block this is the $264$-run study cited in the abstract",
             default_venue="euler"),
    ExpEntry("rw2_sweepsABCD",
             r"\quad\,$\hookrightarrow$ sweeps A--D: $L_\mathrm{train}$ / $L_\mathrm{OOD}$ / $d_\mathrm{state}$ / PE-init",
             "sec:exp33",
             [("json_single_tag:A,B,C,D", "exp3_3/exp3_3__*__seed*.json")],
             note=r"robustness checks (\Cref{app:exp33_sweeps}); not part of the headline grid",
             default_venue="euler"),
    ExpEntry("tor1", r"Exp.~4.1 (Tor-Repr): representation alignment",
             "sec:exp41",
             [("json_single", "exp4_1/exp4_1__seed*.json")],
             default_venue="euler"),
    ExpEntry("tor2", r"Exp.~4.2 (Tor-Cov): critical coverage scaling",
             "sec:exp42",
             [("json_single", "exp4_2/exp4_2__seed*.json")],
             default_venue="euler"),
    ExpEntry("tor3", r"Exp.~4.3 (Tor-Cap): capacity at varied coverage",
             "sec:exp43",
             [("json_single", "exp4_3/exp4_3__seed*.json")],
             default_venue="euler"),
    ExpEntry("tor4", r"Exp.~4.4 (Tor-Noise): noise robustness across $\sigma$",
             "sec:exp44",
             [("json_single", "exp4_4/exp4_4__seed*.json")],
             default_venue="euler"),
]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class Aggregated:
    n_runs:        int = 0
    walls:         list = field(default_factory=list)  # seconds, per run
    devices:       set = field(default_factory=set)
    gpus:          set = field(default_factory=set)
    peak_gpu_gb:   list = field(default_factory=list)
    n_params_max:  int = 0
    timing_source: str = ""   # "json", "json_rows", "mtime", "run_log", "missing"


def _load_run_log(results_dir: Path) -> dict:
    """Merge every `run_resources*.json` written by experiments/run_all.py.

    Each invocation of `run_all.py --log-resources` overwrites
    `run_resources.json` with only the experiments it ran, so we keep
    timestamped copies (e.g. `run_resources_tier123.json`) and merge
    here.  Within a single (exp_id, seed) we keep the most recent record.
    """
    files = sorted(results_dir.glob("run_resources*.json"))
    if not files:
        return {}
    device = ""
    by_key: dict[tuple, dict] = {}
    for p in files:
        mtime = p.stat().st_mtime
        try:
            with open(p) as fp:
                payload = json.load(fp)
        except Exception:
            continue
        device = payload.get("device", device) or device
        for r in payload.get("runs", []):
            r = dict(r)
            r["__mtime__"] = mtime
            key = (r.get("exp_id"), r.get("seed"))
            cur = by_key.get(key)
            if cur is None or r["__mtime__"] >= cur["__mtime__"]:
                by_key[key] = r
    by_exp: dict[str, list] = {}
    for r in by_key.values():
        by_exp.setdefault(r["exp_id"], []).append(r)
    return {"device": device, "by_exp": by_exp}


def _dedupe_latest_per_seed(files: list[Path]) -> list[Path]:
    """Keep only the most recent file per (exp_prefix, seed) when several
    timestamped re-runs exist (e.g. exp1_1 has 22 files for 3 seeds)."""
    by_key: dict[tuple, Path] = {}
    for f in files:
        parts = f.stem.split("__")
        if len(parts) < 3:
            by_key[(f.stem,)] = f
            continue
        # Strip the trailing timestamp; key on (exp_id, seed, sweep_tags)
        key = tuple(parts[:-1])
        cur = by_key.get(key)
        if cur is None or f.stat().st_mtime > cur.stat().st_mtime:
            by_key[key] = f
    return sorted(by_key.values())


def _aggregate_json_single(files: list[Path], agg: Aggregated):
    walls_present = False
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        agg.n_runs += 1
        if "wall_time_s" in d and isinstance(d["wall_time_s"], (int, float)):
            agg.walls.append(float(d["wall_time_s"]))
            walls_present = True
        if "device" in d:
            agg.devices.add(d["device"])
        if "gpu_name" in d and d["gpu_name"]:
            agg.gpus.add(d["gpu_name"])
        if "gpu_mem_peak_gb" in d and isinstance(d["gpu_mem_peak_gb"], (int, float)):
            agg.peak_gpu_gb.append(float(d["gpu_mem_peak_gb"]))
        if "n_params" in d and isinstance(d["n_params"], (int, float)):
            agg.n_params_max = max(agg.n_params_max, int(d["n_params"]))
    return walls_present


def _aggregate_json_rows(files: list[Path], agg: Aggregated):
    """For files like exp2_3_tiny_pe summary.json with `rows[]` of per-condition runs."""
    walls_present = False
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        rows = d.get("rows") or []
        for r in rows:
            agg.n_runs += 1
            if "train_time_s" in r:
                agg.walls.append(float(r["train_time_s"]))
                walls_present = True
            if "n_params" in r and isinstance(r["n_params"], (int, float)):
                agg.n_params_max = max(agg.n_params_max, int(r["n_params"]))
            if "gpu_name" in r and r["gpu_name"]:
                agg.gpus.add(r["gpu_name"])
            if "device" in r and r["device"]:
                agg.devices.add(r["device"])
            if ("gpu_mem_peak_gb" in r
                    and isinstance(r["gpu_mem_peak_gb"], (int, float))):
                agg.peak_gpu_gb.append(float(r["gpu_mem_peak_gb"]))
    return walls_present


def _mtime_fallback(files: list[Path], agg: Aggregated):
    """Estimate per-run wall time as median gap between successive mtimes."""
    if not files:
        return
    mts = sorted(os.path.getmtime(f) for f in files)
    gaps = [mts[i] - mts[i - 1] for i in range(1, len(mts))]
    gaps = [g for g in gaps if 1.0 <= g <= 4 * 3600]  # drop noise / batch boundaries
    if not gaps:
        return
    med = statistics.median(gaps)
    # extrapolate to all runs: assume each run took ~median gap
    for _ in files:
        agg.walls.append(med)


def _apply_filter(files: list[Path], kind: str) -> list[Path]:
    """Apply the optional `kind:filter_tag` suffix to drop / keep files
    based on filename only. JSON-content filters (`json_single_tag:<set>`)
    are dispatched separately in `aggregate_experiment`."""
    if ":" not in kind:
        return files
    _, tag = kind.split(":", 1)
    if tag == "not_L":
        return [f for f in files if "__L" not in f.name]
    if tag == "has_L":
        return [f for f in files if "__L" in f.name]
    return files


def _filter_by_sweep_tag(files: list[Path], wanted: set[str]) -> list[Path]:
    """Keep only JSON files whose `sweep_tag` field is in `wanted`. The
    literal ``"none"`` matches files where `sweep_tag` is missing or null
    (i.e. the primary block of the Exp. 3.3 grid)."""
    out = []
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        tag = d.get("sweep_tag")
        key = "none" if tag is None else str(tag)
        if key in wanted:
            out.append(f)
    return out


def _paper_seed_files(files: list[Path], entry: ExpEntry) -> list[Path]:
    """Preserve the manuscript's run selection when extra seeds exist on disk."""
    if entry.paper_seeds is None:
        return files
    seed_tags = {f"seed{seed}" for seed in entry.paper_seeds}
    return [path for path in files if seed_tags.intersection(path.stem.split("__"))]


def aggregate_experiment(entry: ExpEntry, results_dir: Path,
                         run_log: dict) -> Aggregated:
    agg = Aggregated()
    walls_seen = False
    rows_seen  = False
    for kind, glob in entry.paths:
        files = _paper_seed_files(sorted(results_dir.glob(glob)), entry)
        base_kind = kind.split(":", 1)[0]
        if base_kind == "json_single_tag":
            _, tagspec = kind.split(":", 1)
            wanted = set(tagspec.split(","))
            files = _filter_by_sweep_tag(files, wanted)
            files = _dedupe_latest_per_seed(files)
            if not files:
                continue
            if _aggregate_json_single(files, agg):
                walls_seen = True
            continue
        files = _apply_filter(files, kind)
        if not files:
            continue
        if base_kind == "json_single":
            files = _dedupe_latest_per_seed(files)
            if _aggregate_json_single(files, agg):
                walls_seen = True
        elif base_kind == "json_rows":
            if _aggregate_json_rows(files, agg):
                rows_seen = True

    if walls_seen:
        agg.timing_source = "json"
    elif rows_seen:
        agg.timing_source = "json_rows"
    else:
        # Try run_resources.json next.  experiments/run_all.py uses keys like
        # "tor1_representation" while the registry uses the short "tor1".
        ALIASES = {
            "tor1": "4.1",
            "tor2": "4.2",
            "tor3": "4.3",
        }
        log_key = ALIASES.get(entry.key, entry.key)
        runs = run_log.get("by_exp", {}).get(log_key, [])
        if entry.paper_seeds is not None:
            runs = [r for r in runs if r.get("seed") in entry.paper_seeds]
        if runs:
            for r in runs:
                if "wall_s" in r:
                    agg.walls.append(float(r["wall_s"]))
            if not agg.n_runs:
                agg.n_runs = len(runs)
            if run_log.get("device"):
                agg.devices.add(run_log["device"])
            agg.timing_source = "run_log"
        else:
            # Last resort: mtime gaps over all files for this entry
            all_files = []
            for _, glob in entry.paths:
                all_files += _paper_seed_files(list(results_dir.glob(glob)), entry)
            _mtime_fallback(all_files, agg)
            if agg.walls:
                agg.timing_source = "mtime"
            else:
                agg.timing_source = "missing"
    return agg


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

def _fmt_time_s(s: float) -> str:
    """Format seconds → '<1 s', '12 s', '4.3 min', '2.1 h', '1.4 d'."""
    if s <= 0:
        return "--"
    if s < 1:
        return r"$<$1 s"
    if s < 60:
        return f"{s:.0f} s"
    if s < 60 * 60:
        return f"{s/60:.1f} min"
    if s < 24 * 60 * 60:
        return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} d"


def _fmt_count(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


def _fmt_gpu_mem(gb: float) -> str:
    """Format peak GPU memory: '--' if absent / zero, MB below 1 GB, GB above."""
    if gb is None or gb < 1e-3:
        return "--"
    if gb < 1.0:
        return f"{gb*1024:.0f}\\,MB"
    return f"{gb:.1f}\\,GB"


def _shorten_gpu(name: str) -> str:
    return (name.replace("NVIDIA GeForce ", "")
                .replace("NVIDIA ", "")
                .replace("Generation Laptop GPU", "Laptop")
                .replace("Generation", ""))


def _device_str(agg: Aggregated, default: str) -> str:
    """Always emit a single short string of the form 'RTX 4090' /
    'RTX 3000 Ada Laptop' / 'CPU'. Drop the 'GPU:' prefix for compactness.

    A `gpu_name` is reported by `utils.collect_meta` whenever CUDA is
    available, even when the experiment never moves any tensor to the GPU
    (e.g. pure NumPy / pysindy / OLS). Treat such runs as CPU: if every
    captured `gpu_mem_peak_gb` is (effectively) zero, the GPU was idle and
    the row should not claim GPU compute."""
    if agg.gpus:
        if agg.peak_gpu_gb and max(agg.peak_gpu_gb) < 1e-3:
            return "CPU"
        names = sorted(set(_shorten_gpu(n) for n in agg.gpus))
        return ", ".join(names)
    if agg.devices:
        if any("cuda" in d.lower() or "gpu" in d.lower() for d in agg.devices):
            # ResourceTracker writes a verbose 'GPU (NVIDIA ...)' string;
            # extract the name and shorten.
            for d in agg.devices:
                if "GPU" in d and "(" in d:
                    inner = d.split("(", 1)[1].rstrip(") ")
                    return _shorten_gpu(inner)
            return _shorten_gpu(default)
        if any("cpu" in d.lower() for d in agg.devices):
            return "CPU"
        return ", ".join(sorted(agg.devices))
    return _shorten_gpu(default)


def _classify_venue(agg: Aggregated, entry: ExpEntry) -> str:
    """Return 'local' (laptop RTX 3000) or 'euler' (cluster RTX 4090 / CPU node)
    based on observed gpu_name, with a fallback to entry.default_venue."""
    for n in agg.gpus:
        if "4090" in n:
            return "euler"
        if "3000" in n or "Laptop" in n:
            return "local"
    return entry.default_venue


def _render_row(entry: ExpEntry, agg: Aggregated, local_default: str) -> tuple[str, float]:
    """Render one tabular row plus the row's contribution to the wall-total."""
    runs_s = _fmt_count(agg.n_runs) if agg.n_runs else "--"
    if agg.walls:
        med_s   = statistics.median(agg.walls)
        total_s = sum(agg.walls)
        med_fmt   = _fmt_time_s(med_s)
        total_fmt = _fmt_time_s(total_s)
        if agg.timing_source == "mtime":
            med_fmt   = med_fmt + r"$^\dagger$"
            total_fmt = total_fmt + r"$^\dagger$"
    else:
        total_s = 0.0
        med_fmt   = "--"
        total_fmt = "--"
    device = _device_str(agg, local_default)
    peak_gb = max(agg.peak_gpu_gb) if agg.peak_gpu_gb else 0.0
    if device == "CPU":
        peak_fmt = "--"
    else:
        peak_fmt = _fmt_gpu_mem(peak_gb)
    line = (f"    {entry.label} & {runs_s} & "
            f"{med_fmt} & {device} & {peak_fmt} & {total_fmt} \\\\")
    return line, total_s


def render_table(rows: list[tuple[ExpEntry, Aggregated]], local_default: str) -> str:
    """Render a tabular for the appendix, split into a Local-laptop section
    (RTX 3000 Ada) and an Euler section (RTX 4090 GPU node + CPU node) with
    per-section subtotals. Designed to fit in \\textwidth at \\footnotesize."""
    classified = [(entry, agg, _classify_venue(agg, entry)) for entry, agg in rows]
    local_rows = [(e, a) for e, a, v in classified if v == "local"]
    euler_rows = [(e, a) for e, a, v in classified if v == "euler"]

    out = []
    out.append(r"\begin{table}[ht]")
    out.append(r"  \centering")
    out.append(r"  \footnotesize")
    cap = (
        r"Compute resources per experiment, split by venue: laptop "
        r"(RTX 3000 Ada, 8\,GB) vs.\ HPC cluster (RTX 4090 24\,GB GPU or "
        r"CPU node). Wall time, device, and peak GPU memory (max across "
        r"runs of \texttt{torch.cuda.max\_memory\_allocated}) are read from "
        r"each run's saved JSON; rows that never moved a tensor onto the "
        r"GPU are reported as CPU even when CUDA was available. "
        r"The Exp.~3.3 headline grid (\Cref{tab:exp33_main}) is the "
        r"primary block plus Sweep~L (rows~1 and~2 of the Exp.~3.3 block "
        r"below); their run counts sum to the $264$-run figure cited in "
        r"the main text. Sweeps A--D are independent robustness checks "
        r"(\Cref{app:exp33_sweeps}) and are not part of the headline."
    )
    out.append(r"  \caption{" + cap + r"}")
    out.append(r"  \label{tab:resources}")
    out.append(r"  \resizebox{\textwidth}{!}{%")
    out.append(r"  \begin{tabular}{l r r l r r}")
    out.append(r"    \toprule")
    out.append(r"    \textbf{Experiment} & "
               r"\textbf{\#runs} & "
               r"\textbf{wall/run (med.)} & "
               r"\textbf{Device} & "
               r"\textbf{Peak GPU mem} & "
               r"\textbf{Wall (total)} \\")
    out.append(r"    \midrule")

    grand_total_s = 0.0

    def _emit_section(label: str, section_rows):
        nonlocal grand_total_s
        if not section_rows:
            return
        out.append(rf"    \multicolumn{{6}}{{l}}{{\textit{{{label}}}}} \\")
        out.append(r"    \cmidrule(l){1-6}")
        section_total_s = 0.0
        section_runs = 0
        for entry, agg in section_rows:
            line, t = _render_row(entry, agg, local_default)
            out.append(line)
            section_total_s += t
            section_runs   += agg.n_runs
        out.append(r"    \cmidrule(l){1-6}")
        out.append(
            r"    \textit{Subtotal} & "
            + _fmt_count(section_runs)
            + r" &  &  &  & "
            + _fmt_time_s(section_total_s)
            + r" \\"
        )
        out.append(r"    \midrule")
        grand_total_s += section_total_s

    _emit_section("Local laptop (RTX 3000 Ada Generation Laptop GPU)", local_rows)
    _emit_section("Shared HPC cluster (RTX 4090 / CPU)",                 euler_rows)

    out.append(
        r"    \textbf{Total} & "
        + _fmt_count(sum(a.n_runs for _, a in rows))
        + r" &  &  &  & "
        + _fmt_time_s(grand_total_s)
        + r" \\"
    )
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}%")
    out.append(r"  }")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _local_device_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return platform.processor() or "CPU"


def main_inline(results_dir, output_dir, print_only: bool = False):
    """Generate table_resources.tex.  Callable from make_tables.py."""
    results_dir = Path(results_dir)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_log = _load_run_log(results_dir)

    rows = []
    for entry in REGISTRY:
        agg = aggregate_experiment(entry, results_dir, run_log)
        rows.append((entry, agg))

    tex = render_table(rows, local_default=_local_device_str())
    if print_only:
        print(tex)
        return
    out_path = output_dir / "table_resources.tex"
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",
                        default=str(DEFAULT_RESULTS_DIR_LOCAL),
                        help="Top-level results directory (default: results/).")
    parser.add_argument("--output_dir",
                        default=str(_SRC_DIR / "tables"),
                        help="Where to write table_resources.tex.")
    parser.add_argument("--print_only", action="store_true",
                        help="Print summary to stdout; do not write the .tex file.")
    args = parser.parse_args()
    main_inline(args.results_dir, args.output_dir, args.print_only)


if __name__ == "__main__":
    main()
