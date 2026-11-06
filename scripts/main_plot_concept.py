#!/usr/bin/env python3
"""
Abbreviated concept figures for the main paper §4 (cross-domain validation).

Active artefacts live in `figures/main/*.py` and are wired through DISPATCH
below. Two outputs:
  - figures/main/exp2_2_concept_main.pdf            (Kepler single-panel)
  - figures/main/exp2_3_species_concept_main.pdf    (CDS DGP schematic)

Usage (from project root):
    conda run -n data_analysis python scripts/main_plot_concept.py
    conda run -n data_analysis python scripts/main_plot_concept.py --output_dir figures/main

Run sync_paper.py afterwards to copy them into paper/figures/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Importing as `python scripts/main_plot_concept.py` puts scripts on
# sys.path so that the figures/ subpackage resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures.main.exp2_2_kepler import build as build_exp2_2_kepler
from figures.main.exp2_3_cds import build as build_exp2_3_cds
from figures.main.exp2_1_mak import build as build_exp2_1_mak


DISPATCH = {
    "exp2_2_kepler": build_exp2_2_kepler,
    "exp2_3_cds":    build_exp2_3_cds,
    "exp2_1_mak":    build_exp2_1_mak,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="figures/main")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generating main-paper concept figures → {out}/")
    for key, fn in DISPATCH.items():
        try:
            fn(str(out))
        except Exception as e:
            print(f"  [error] {key}: {e}")
    print("Done. Run sync_paper.py to copy into paper/figures/.")


if __name__ == "__main__":
    main()
