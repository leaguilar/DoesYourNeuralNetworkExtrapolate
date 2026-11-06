import Mathlib

/-!
# Basic objects

The standard normal distribution function, observational equivalence on a
training window, and the out-of-distribution risk.
-/

open MeasureTheory ProbabilityTheory Real

namespace DGPCommitment

/-- The standard normal distribution function `Φ`. -/
noncomputable def Phi : ℝ → ℝ := cdf (gaussianReal 0 1)

/-- **Definition 1.** `P₁` and `P₂` are `ε`-observationally equivalent under the training
distribution `D` when the `L²(D)` distance between them is at most `ε`. The distance is
written through its square so that it can be used directly under an integral. -/
def ObsEquiv {X : Type*} [MeasurableSpace X] (D : Measure X) (P₁ P₂ : X → ℝ) (ε : ℝ) : Prop :=
  Integrable (fun x ↦ (P₁ x - P₂ x) ^ 2) D ∧ ∫ x, (P₁ x - P₂ x) ^ 2 ∂D ≤ ε ^ 2

/-- The out-of-distribution risk of a predictor `fhat` for the target `f` under a test
distribution `Dtest`. -/
noncomputable def epsOOD {X : Type*} [MeasurableSpace X] (Dtest : Measure X)
    (f fhat : X → ℝ) : ℝ :=
  ∫ x, (fhat x - f x) ^ 2 ∂Dtest

end DGPCommitment
