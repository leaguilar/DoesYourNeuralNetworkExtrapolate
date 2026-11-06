# Does Your Neural Network Extrapolate? Feature Engineering as Identifiability Bias for OOD Generalization

Code, per-seed results and proofs for the NeurIPS 2026 paper by Leonel Aguilar,
Jan Nagler, Christoph Hoelscher and Nino Antulov-Fantulin.

Every table and figure of the paper is built by one module in here from the
per-seed JSON next to it. Every proposition, lemma and corollary of the paper is
stated and proved in Lean 4 under `lean/`.

```bibtex
@inproceedings{aguilar2026extrapolate,
  title     = {Does Your Neural Network Extrapolate? Feature Engineering as
               Identifiability Bias for OOD Generalization},
  author    = {Aguilar, Leonel and Nagler, Jan and Hoelscher, Christoph and
               Antulov-Fantulin, Nino},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```

Released under the MIT License (`LICENSE`).

## Layout

```
environment.yml                # conda env spec (`feateng`)
experiments/                   # Tier 1 controlled, Tier 2 (Kepler/MAK/Camelyon17), Tier 3.1, Tier 4 scope
experiments_exp2_3_tiny_pe/    # Tier 2: read-level species sweep (Exp 2.3)
experiments_exp3_3_archinv/    # Tier 3.3: Transformer / Mamba / S4D architecture-invariance grid
scripts/tables/                # results JSON -> the LaTeX tables printed in the paper
scripts/tables/main/           # the main-text variants of those tables
scripts/figures/               # results JSON -> the paper's figures (concept/, main/, results/)
scripts/resources_tables.py    # per-experiment wall time, device and peak GPU memory
results/                       # per-seed JSON outputs aggregated by the paper's tables
lean/                          # Lean 4 proofs of every statement in the paper
```

`scripts/` modules resolve `results/` and write `tables/` and `figures/` relative
to this directory, so they run in place with no configuration.

## Environment

Most experiments run under a single conda env:

```
conda env create -f environment.yml
conda activate feateng
```

`envs/` holds the exact solved environments behind the reported numbers, one
per venue: `workstation.lock.yml` for the Tier 1 and Tier 2 experiments and
every table and figure, `cluster-gpu-ssm.lock.yml` for the Exp 3.3 Mamba and
S4D grid, `cluster-cpu.lock.yml` for the remaining cluster jobs, and
`timesfm.lock.yml` for the TimesFM condition of Exp 2.2. Create one with
`conda env create -f envs/workstation.lock.yml`. The versions behind the
reported numbers are Python 3.12, numpy 2.1.3, PyTorch 2.5.1 with CUDA 12.1,
scikit-learn 1.6.1 and pysindy 2.1.0.

Three foundation-model baselines have non-trivial setup and are commented out in `environment.yml`. Install them only for the experiments that need them:

- **TabPFN**: `pip install tabpfn-client`, which requires a free account at priorlabs.ai. With the package absent the probe is skipped. With the package present and the account or the service unavailable, the cloud call raises and the run stops, so pass `--skip-foundation` to run the local conditions only.
- **TimesFM** : best installed in its own env per the TimesFM Hugging Face model card.
- **Mamba** (Exp 3.3, only `--model_type mamba`): `pip install mamba-ssm`, which requires a CUDA toolchain.

Some docstrings name the environment as `data_analysis`, which is the name it
carries on the authors' machines. Here it is `feateng`, and the commands in the
tables below are the ones to use.

## Running an experiment

All scripts run from the repo root via `python -m`. Replace `feateng` with your env name if different. Add `--save` to write the per-seed JSON that the tables read, and `--skip-foundation` to the Tier 1 and Tier 2 scripts to leave out the two cloud or external baselines (TabPFN, TimesFM). Seeds 42, 43 and 44 are the reported ones, set with `--seed`.

| Exp | Section | Command |
|---|---|---|
| 1.1 | Controlled: sin vs Taylor-9 | `python -m experiments.exp1_1_sin --save` |
| 1.2 | Controlled: power-law vs exp | `python -m experiments.exp1_2_powerlaw_vs_exp` |
| 1.3 | Controlled: SINDy coordinate diagnostic | `python -m experiments.exp1_3_sindy_coord` |
| 2.1 | Cross-domain: mass-action kinetics | `python -m experiments.exp2_1_mak_vf` |
| 2.2 | Cross-domain: Kepler's third law | `python -m experiments.exp2_2_kepler` |
| 2.3 | Cross-domain: readseeker species sweep | `python -m experiments_exp2_3_tiny_pe.readseeker_species_sweep` |
| 2.4 | Cross-domain: Camelyon17 histopathology | see *Exp 2.4 pipeline* below |
| 3.1 | Tools: regime sweep | `python -m experiments.exp3_1_regime_sweep` |
| 3.1 | Tools: selection protocol | `python -m experiments.exp3_1_selection_protocol` |
| 3.1 | Tools: near-boundary segment-width sweep (appendix) | `python -m experiments.exp3_1_width_sweep --seed 42 --save` |
| 3.3 | Tools: architecture-invariance grid | `python -m experiments_exp3_3_archinv.train --model_type transformer --pe_type exact_fourier --period 64 --seed 42` |
| 4.1 | Scope: torus representation | `python -m experiments.exp4_1_tor_repr` |
| 4.2 | Scope: torus coverage | `python -m experiments.exp4_2_tor_cov` |
| 4.3 | Scope: torus capacity | `python -m experiments.exp4_3_tor_cap` |
| 4.4 | Scope: torus noise robustness | `python -m experiments.exp4_4_tor_noise` |
| 4.5 | Scope: phi-misspecification sweep | `python -m experiments.rebuttal_e3_misspec_robustness --save` |

For Exp 3.3 the full grid is `model_type ∈ {transformer, mamba, s4} × pe_type ∈ {no_pe, learned_ape, exact_fourier} × period ∈ {64, 128, 256, 512} × seed ∈ {42, 43, 44}`. Pass the corresponding flags to `train.py`.

**Exp 3.1 width sweep.** Run seeds 42, 43, 44 with `--save`, then pool them and
print the paper numbers:

```
for s in 42 43 44; do python -m experiments.exp3_1_width_sweep --seed $s --save; done
python -m experiments.exp3_1_width_sweep --report
```

**Exp 4.5 misspecification sweep.** `--save` runs the three misspecification
axes (frequency, coordinate coverage, feature noise) for seeds 0, 1, 2 plus the
near-boundary selection sub-experiment. `--quick` is a one-seed smoke test and
`--cpu` forces CPU.

## Exp 2.4 pipeline (Camelyon17 histopathology)

Four stages. Only the last one is expensive, and only stages 2–4 are needed to
reproduce the reported numbers.

1. **Get the data.** Camelyon17-WILDS is not bundled (tens of GB of PNG
   patches). Download it once with the `wilds` package
   (`pip install wilds`, then
   `python -c "import wilds; wilds.get_dataset('camelyon17', download=True, root_dir='data/camelyon17')"`)
   or from the public WILDS archive, so that
   `data/camelyon17/camelyon17_v1.0/` contains `metadata.csv` and `patches/`.

2. **Materialize the arrays.** One fixed data sample is drawn and each arm's
   fixed input transform is applied once, so every training seed sees exactly
   the same pixels:

   ```
   python experiments/prepare_camelyon_arrays.py \
       --n-train 60000 --n-idtest 10000 --n-oodtest 20000 \
       --out data/camelyon_cache
   python experiments/make_colorctrl_cluster.py   # colorctrl arm; CAMELYON_CACHE=... to relocate
   ```

   This writes the `x_{train,idtest,oodtest}_{raw,stainnorm,channorm,colorctrl}.npy`
   uint8 arrays (≈11 GB) plus the label/metadata JSON. They are derived data and
   are deliberately not shipped. `prepare_camelyon_arrays.py` needs `torchstain`
   (in `environment.yml`), and nothing downstream does. Splits are the cross-hospital
   WILDS axis: train on centres {0, 3, 4}, held-out hospital = centre 2, and an
   ID test set of unseen patients within {0, 3, 4}.

3. **Train.** One `(arm, seed)` per invocation, and the reported grid is
   4 arms × 5 seeds, plus the `raw` arm re-run with `--colour-aug` for the
   `raw+caug` arm:

   ```
   python experiments/train_camelyon17_scratch.py --arm stainnorm --seed 0 \
       --cache-dir data/camelyon_cache --out results/expfm_v_histopath_scratch \
       --epochs 20 --batch 256 --lr 0.01
   ```

   Weights are randomly initialized (no pretraining), and the reported epoch is
   chosen by ID accuracy, so no OOD label ever enters model selection. Training
   needs only torch, torchvision, numpy and scikit-learn.

   `experiments/slurm_{submit,job}_camelyon_scratch.sh` are the SLURM array
   templates we actually used (20 tasks = 4 arms × 5 seeds). Every site-specific
   knob is an environment variable with a documented default, so they carry no
   hard-coded partition, GPU type or software stack: the repo root, billing
   accounts and conda env come from `SCRATCH_ROOT`, `ACCOUNT_A` / `ACCOUNT_B`,
   `CONDA_ENV` / `CONDA_BIN`, and the scheduler resources from
   `CLUSTER_PARTITION`, `CLUSTER_GPUS` and `CLUSTER_MODULES`.

4. **Aggregate.** `scripts/tables/histopath.py` and `scripts/tables/main/histopath.py`
   build the tables. `scripts/figures/concept/expfm_v_histopath.py` builds the figure.
   See *Paper artefacts* below.

`experiments/expfm_v_histopath_stain.py` is the frozen-pretrained-backbone
linear-probe variant of the same comparison, kept here because it documents the
transform definitions and the official WILDS split usage (it needs `timm` and
`wilds`). Its results are **not** part of the paper and are not shipped: those
backbones have unknown pretraining exposure, possibly including
histopathology, so the comparison is confounded, and it was a single seed.

## Paper artefacts

Five commands rebuild everything, from the directory that holds this README:

```
python scripts/make_tables.py        # every appendix table
python scripts/main_make_tables.py   # every main-text table
python scripts/plot_concept.py       # every concept figure
python scripts/main_plot_concept.py  # the main-text concept figures
python scripts/plot_results.py       # the Exp 4 result panels
```

They read `results/` and write `tables/*.tex`, `tables/main/*.tex`,
`figures/*.pdf` and `figures/main/*.pdf` next to this README. All 28 tables of
the paper then carry the same numbers as the published copies. Captions differ,
because they were edited after the tables were built.

Most modules also run on their own, which is the quicker loop when iterating on
one artefact. The two main-text tables built by the legacy row builders
(Exp 2.3 species, Exp 3.3 architecture invariance) only run through
`main_make_tables.py`, which puts the builders on the import path.

Tables, in the order the paper prints them:

| Table | Module |
|---|---|
| Exp 1.1 sin vs Taylor-9 | `python scripts/tables/exp1_1.py` |
| Exp 1.2 power law vs exponential (main, appendix) | `python scripts/tables/main/exp1_2.py`, `python scripts/tables/exp1_2.py` |
| Exp 2.1 mass-action kinetics (main, appendix) | `python scripts/tables/main/exp2_1_mak.py`, `python scripts/tables/exp2_1_mak.py` |
| Exp 2.2 Kepler (main, appendix) | `python scripts/tables/main/exp2_2_kepler.py`, `python scripts/tables/exp2_2_kepler.py` |
| Exp 2.3 CDS species (main) | `python scripts/main_make_tables.py` |
| Exp 2.4 Camelyon17 (main, full arm grid) | `python scripts/tables/main/histopath.py`, `python scripts/tables/histopath.py` |
| Exp 3.1 selection protocol | `python scripts/tables/exp3_1.py` |
| Exp 3.1 near-boundary width sweep | `python scripts/tables/exp3_1_width_sweep.py` |
| Exp 3.2 SINDy diagnostic battery | `python scripts/tables/sindy_delta_consolidated.py` |
| Exp 3.3 architecture invariance (main, in-text) | `python scripts/main_make_tables.py`, `python scripts/tables/main/exp3_3_archinv_mini.py` |
| Exp 3.3 robustness sweeps A, C, D, L | `python scripts/tables/exp3_3_sweep_{A,C,D,L}.py` |
| Exp 4.1 to 4.4 torus scope study | `python scripts/tables/exp4_{1,2,3,4}.py` |
| Exp 4.4 Bayes-ratio companion | `python scripts/tables/exp4_4_ratio.py` |
| Exp 4.5 phi-misspecification sweep | `python scripts/tables/e3_misspec.py` |
| Compute resources per experiment | `python scripts/resources_tables.py --output_dir tables/` |

The compute-resources table reads the wall time, device and peak GPU memory
recorded in each run's JSON. The paper groups the same records into a different
set of rows, so the two layouts do not line up row for row even though the
recorded values agree.

Figures:

| Figure | Module |
|---|---|
| Fig 1, Exp 1.1 concept | `python scripts/figures/concept/exp1_1.py` |
| Exp 1.3 concept | `python scripts/figures/concept/exp1_3.py` |
| Exp 2.1 mass-action vector field | `python scripts/figures/main/exp2_1_mak.py` |
| Exp 2.2 Kepler concept | `python scripts/figures/main/exp2_2_kepler.py` |
| Exp 2.3 CDS concept | `python scripts/figures/main/exp2_3_cds.py` |
| Exp 2.4 Camelyon17 concept and main-text panel | `python scripts/figures/concept/expfm_v_histopath.py` |
| Exp 3.3 concept, panels (a, b) and (c, d) | `python scripts/figures/concept/exp3_3_archinv.py` |
| Exp 4.1 to 4.4 torus panels | `python scripts/figures/results/exp4_{1,2,3,4}.py` |
| Appendix roadmap | `pdflatex scripts/figures/main/roadmap_tikz.tex` |

Import `build(output_dir=...)` from any of the table modules to redirect the
output. Three small derived files let three figures rebuild without the bulk
data behind them.

- Panels (a) and (b) of the Exp 2.4 concept figure read
  `results/expfm_v_histopath_scratch/concept_colour_stats.npz`, 240 KB of
  patch-colour statistics plus the four example patches printed in the figure,
  so the figure rebuilds without the 11 GB patch cache. Delete it and the
  module recomputes it from that cache.
- The Exp 2.1 vector-field figure reads
  `results/exp2_1/figure_artefacts.npz`, 240 KB holding the fitted fields it
  draws. Delete it and the experiment has to be re-run.
- The raw-space overlay of Fig 1 reads `figures/_cache/exp1_1_predictions.npz`,
  23 KB of held-out predictions from the four models of Exp 1.1. Delete it and
  the figure draws without the overlay until
  `scripts/figures/concept/_exp1_1_predictions.py` regenerates it, which needs
  the TabPFN and TimesFM baselines installed.

## Lean 4 proofs

`lean/` states and proves every definition, proposition, lemma, corollary and
remark of the paper in Lean 4 with Mathlib. Nothing is assumed beyond Mathlib
and no proof is left open. The 73 headline results depend only on Lean's three
standard axioms.

```
cd lean
lake exe cache get          # first time only, downloads Mathlib
lake build
bash check.sh               # build, no open proofs, axiom audit
```

`lean/README.md` maps each statement of the paper to its Lean name and file, and
says where each hypothesis is needed.

## Datasets

No data is bundled. Scripts download or expect public datasets at first run:

- **NASA Exoplanet Archive** (Exp 2.2): fetched via Hugging Face datasets (`juliensimon/nasa-exoplanets`), cached under the standard `~/.cache/huggingface/` location.
- **NCBI RefSeq genomes** (Exp 2.3): six organisms, and the assembly accessions are listed in the paper appendix. Place each genome FASTA under `data/genomes/<organism>/`.
- **Camelyon17-WILDS** (Exp 2.4): see stage 1 of the Exp 2.4 pipeline above. Neither the raw patches nor the derived `data/camelyon_cache/` arrays are shipped, and both are regenerated by `prepare_camelyon_arrays.py`.

Synthetic datasets (Exp 1.x, 3.1, 3.3, 4.x) are generated on the fly and are fully specified by the equations and ranges in the relevant paper sections.

## JSON results

`results/<exp_dir>/` contains the per-seed `*.json` outputs that the paper's headline tables aggregate (mean ± 1σ over seeds 42, 43, 44, or over seeds 0–4 for Exp 2.4). Layout:

```
results/exp1_1/                : one JSON per (DGP, model, seed)
results/exp1_2/                : power-law / exp seed sweep
results/exp1_3/                : SINDy battery (synthetic DGP × feature map)
results/exp2_1/                : MAK vector-field, per-network seed
results/exp2_2/                : Kepler seed sweep + foundation-model probe
results/exp2_3_species_clean_*/: tiny-PE readseeker species sweep
results/exp3_1*/               : regime sweep + selection protocol + width sweep
results/exp3_3/                : architecture-invariance grid (96 cells × 3 seeds)
results/exp4_{1,2,3,4}/        : torus scope study
results/expE3_misspec/         : Exp 4.5 phi-misspecification sweep (3 seeds)
results/expfm_v_histopath_scratch/ : Exp 2.4 Camelyon17, one JSON per (arm, seed)
```

Model checkpoints and bulk numeric arrays are excluded throughout. The only
non-JSON payload is the Exp 2.4 concept-figure statistics blob noted above.
