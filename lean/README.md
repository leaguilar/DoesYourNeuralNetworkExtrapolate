# The paper's statements, in Lean 4

Every definition, proposition, lemma and corollary of the paper, stated and
proved in Lean 4 with Mathlib, together with the remarks that make a
mathematical claim. Nothing is assumed beyond Mathlib. No proof is left open:
the library builds, no file contains an open proof or `native_decide`, and all
73 headline results depend only on Lean's three standard axioms, `propext`,
`Classical.choice` and `Quot.sound`.

The one result with real content beyond textbook algebra is Proposition 1. Its
proof rests on the Gaussian two-point problem: no test on `n` Gaussian samples
can separate two mean vectors with average success above `Φ(S / (2σ))`, and
the likelihood-ratio test attains that value. Mathlib has neither the
Neyman–Pearson lemma nor a density for the multivariate Gaussian, so the
library proves both pieces it needs: the product density of independent
Gaussian coordinates and the optimality of the likelihood-ratio half-space.

## Where the hypotheses bite

**Corollary 3 needs its divergence hypothesis on the whole strip.** The bound
`‖Δ‖_{L²(D_V)} ≥ r δ_V / √3` integrates `|Δ(b + u)| ≥ r u` over `(0, δ_V]`, so
that is where the hypothesis must hold, and where the paper states it. A local
version is not enough. A gap that grows at rate 1 on `(0, 1)` and vanishes
beyond diverges at rate 1 near the boundary, yet on a strip of width 2 its
squared norm is `1/6`, below the bound's `4/3` (`small_u_reading_fails`).

**In Proposition 2(d), `α < 1/4` orders the two bands.** The two threshold
conditions constrain different numbers, `t_W` and `t_V`, so they can always
hold together (`prop2d_conditions_always_jointly_satisfiable`). What
`α < 1/4` decides is that the window's ceiling `1/2 + α` lies strictly below
the strip's floor `1 - α` (`prop2d_band_iff`).

**Proposition 2(b) is exact on average, and exact per hypothesis once the
means differ.** Under equal priors the likelihood-ratio test succeeds with
probability exactly `Φ(t_V / 2)` for every design (`lr_success_add`). The
same value holds for each hypothesis separately when the two mean vectors
differ on the design (`lr_success_one`, `lr_success_two`). When they coincide
the test has nothing to go on, and a deterministic tie rule gives success 1
under one hypothesis and 0 under the other, which still averages to
`Φ(0) = 1/2`.

## The map from the paper to the library

| Paper statement | Lean name | File |
|---|---|---|
| Definition 1, `ε_W`-observational equivalence | `ObsEquiv` | `Basic.lean` |
| Proposition 1, as stated (worst hypothesis) | `erm_blind_min` | `ERM.lean` |
| Proposition 1, summed success of any randomised test | `erm_blind` | `ERM.lean` |
| Proposition 1, any deterministic selection rule | `erm_blind_deterministic` | `ERM.lean` |
| Proposition 1, Step 1 (worst is at most average) | `min_le_half_add` | `ERM.lean` |
| Proposition 1, Step 2 (two-point optimality) | `twoPoint_le`, `np_le_lr` | `TwoPoint.lean` |
| Proposition 1, attainment by the likelihood-ratio test | `twoPoint_lr_attains` | `TwoPoint.lean` |
| Proposition 1, Step 3 (Jensen over the design) | `jensen_design`, `concaveOn_Phi_sqrt` | `ERM.lean`, `Phi.lean` |
| `E[Sₙ²] = n ‖Δ‖²`, also used in Remark 2 | `integral_designGap` | `ERM.lean` |
| Linear form of Proposition 1 | `linear_form`, `Phi_le_tangent` | `ERM.lean`, `Phi.lean` |
| Linear bound reaches 1 exactly at `σ = √n ε_W / √(2π)` | `linear_form_eq_one_iff` | `ERM.lean` |
| Le Cam–Pinsker comparison, constants | `linear_constant_bounds` | `ERM.lean` |
| Le Cam–Pinsker comparison, the resulting bound `1/2 + t_W/4` | `pinsker_shape` | `ERM.lean` |
| Numerical illustration, Pinsker side is vacuous | `pinsker_illustration_vacuous` | `ERM.lean` |
| Proposition 2(a) | `prop2a` | `ERM.lean` |
| Proposition 2(b) | `lr_success_add`, `lr_success_one`, `lr_success_two` | `TwoPoint.lean` |
| Proposition 2(c) | `prop2c` | `Floor.lean` |
| Proposition 2(c), equality for the ramp | `prop2c_eq`, `rampGap_obsEquiv` | `Floor.lean` |
| Proposition 2(d) | `prop2d`, `prop2d_band_iff` | `ERM.lean` |
| Remark 2, equally spaced design | `equispaced_empirical_ge` | `Floor.lean` |
| Remark 3, divergence at rate `u^p` | `strip_sqNorm_ge_rpow` | `Floor.lean` |
| Corollary 3, norm form | `geometric_floor_norm`, `strip_norm_ge` | `Floor.lean` |
| Corollary 3, `t_V ≥ t_W √(n_V / n)` | `geometric_floor` | `Floor.lean` |
| Corollary 3, tightness | `geometric_floor_tight` | `Floor.lean` |
| Corollary 3, a local divergence hypothesis is not enough | `small_u_reading_fails` | `Floor.lean` |
| Lemma 4, realisable-case canonicalisation | `canonical_epsOOD_eq_zero` | `Canonical.lean` |
| Lemma 4 with `ψ = id` | `canonical_epsOOD_eq_zero_id` | `Canonical.lean` |
| Lemma 5, sinusoidal instance | `sin_realisable`, `sin_epsOOD_eq_zero` | `Canonical.lean` |
| Remark 4, phase and amplitude | `phase_amplitude_realisable`, `span_sin_cos_eq_phase_amplitude` | `Canonical.lean` |
| Lemma 6, power-law instance | `powerLaw_realisable`, `powerLaw_epsOOD_eq_zero` | `Canonical.lean` |
| Lemma 7, exponential instance | `exponential_realisable`, `exponential_epsOOD_eq_zero` | `Canonical.lean` |
| Remark 5, power law as the exponential in `log x` | `powerLaw_eq_exponential_comp_log` | `Canonical.lean` |
| Remark 6, least squares is the log-normal maximum likelihood | `gaussLogLik_eq`, `gaussLogLik_le_iff` | `Canonical.lean` |
| Corollary 8, identifiability of the least-squares fit | `isOLS_iff` | `OLS.lean` |
| Corollary 8, the three instances | `sin_ols_recovery`, `powerLaw_ols_recovery`, `exponential_ols_recovery` | `OLS.lean` |
| Corollary 8, zero OOD risk | `sin_ols_epsOOD`, `powerLaw_ols_epsOOD`, `exponential_ols_epsOOD` | `OLS.lean` |

Remark 1 restates which hypothesis enters which part of Proposition 2, and the
statements above carry exactly those hypotheses.

Supporting results that carry the probability:

| What it says | Lean name | File |
|---|---|---|
| Tonelli for a product of functions of the coordinates | `lintegral_fin_prod_eq_prod` | `GaussVec.lean` |
| A product of densities is the density of the product measure | `pi_withDensity` | `GaussVec.lean` |
| The Gaussian vector has the product Gaussian density | `gaussVec_eq_withDensity` | `GaussVec.lean` |
| `∑ dᵢ ηᵢ ~ N(0, σ² ‖d‖²)` for i.i.d. `N(0, σ²)` noise | `map_linear_gaussVec` | `GaussVec.lean` |
| The likelihood-ratio half-space is where the first density dominates | `mem_lrRegion_iff` | `TwoPoint.lean` |
| `Φ` is strictly increasing, symmetric, and concave on `[0, ∞)` | `strictMono_Phi`, `Phi_neg`, `concaveOn_Phi` | `Phi.lean` |
| Every level in `(0, 1)` is a value of `Φ`, so `Φ⁻¹` is defined there | `Phi_surj`, `Phi_PhiInv` | `ERM.lean` |
| An event on two coordinates of an i.i.d. design is null when its slices are | `design_null_of_slices` | `OLS.lean` |

## How the objects are modelled

`Φ` is Mathlib's distribution function of the standard normal law. Its
density, symmetry and tangent bound are derived from that definition, not
assumed.

Observational equivalence is written through the squared distance: `P₁ - P₂`
is square integrable under the training distribution `D`, and its second
moment is at most `ε_W²`. The window `W` enters only through `D`, which is how
Definition 1 measures the distance.

The sample is modelled exactly as the paper describes it. The design
`(x₁, …, xₙ)` is i.i.d. from `D`, the noise `(η₁, …, ηₙ)` is i.i.d.
`N(0, σ²)` and independent of the design, and `yᵢ = P(xᵢ) + ηᵢ`. A test is any
measurable function of the whole sample with values in `[0, 1]`, read as the
probability of answering "1". This covers deterministic tests, randomised
tests, and every selection criterion computed from the window, cross-validation
included, in one statement. Proposition 1 never mentions a hypothesis class,
so its bound holds with the same constants for every class that contains both
candidates.

The uniform distribution on the strip `V = [b, b + δ_V]` is written as the
normalised interval integral. `uniformSqNorm_eq_integral_cond` checks that this
equals the second moment under Mathlib's own uniform measure, Lebesgue measure
conditioned on the strip. The signal-to-noise ratio `t_V` is a real number only
when `‖Δ‖_{L²(D_V)}` is finite, so the strip results assume `Δ²` integrable on
the strip.

The label map in Lemma 4 needs only a left inverse on the values of the target.
Strict monotonicity supplies one (`exists_leftInverse_of_strictMono`).

A weight is an ordinary-least-squares fit when no weight has a smaller residual
sum of squares. The library never forms `(ΦᵀΦ)⁻¹Φᵀz`. Under realisability the
canonical weight has zero residual, so it is a fit, and it is the only fit once
the first two feature rows are linearly independent. Absolute continuity of the
design distribution enters only to give countable sets probability zero, which
is what makes those rows independent almost surely. The power-law instance also
assumes the design lies in `ℝ_{>0}`, where its feature map is defined.

## What is not in the library

The Le Cam–Pinsker route itself, the Kullback–Leibler identity for the Gaussian
shift and Pinsker's inequality, is not formalised. The bound that route gives,
`1/2 + t_W/4`, is proved here as a consequence of the sharper linear form.

Numerical values quoted in the appendices are evaluations, not theorems. The
exact two-point value `Φ(t_W / 2) ≈ 0.842` of the numerical illustration and
the observational-equivalence gaps quoted in the No Free Lunch appendix belong
to that group. The remark that the gap between a periodic target and its
truncated Taylor expansion grows with an exponent slightly above 1 is an
observation about particular functions and is not formalised.

## Building and checking

    export PATH="$HOME/.elan/bin:$PATH"
    lake exe cache get          # first time only, downloads Mathlib
    lake build
    bash check.sh               # the gate: build, no open proofs, axioms

`check.sh` fails if the build fails, if any file contains an open proof,
`native_decide` or an added axiom, or if any of the 73 headline results rests
on an axiom other than the three standard ones. The build tree under `.lake`
is not tracked. `lake exe cache get` rebuilds it in minutes.
