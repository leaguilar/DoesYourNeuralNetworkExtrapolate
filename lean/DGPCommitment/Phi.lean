import DGPCommitment.Basic

/-!
# The standard normal distribution function

The facts about `Φ` the propositions use: its density, symmetry, strict
monotonicity, the tangent bound behind the linear form of Proposition 1, and
the concavity of `v ↦ Φ(c √v)` behind the Jensen step.
-/

open MeasureTheory ProbabilityTheory Real Set Filter Topology

namespace DGPCommitment

/-- The standard normal density. -/
noncomputable def phiPDF (x : ℝ) : ℝ := gaussianPDFReal 0 1 x

theorem phiPDF_eq (x : ℝ) : phiPDF x = (√(2 * π))⁻¹ * exp (-x ^ 2 / 2) := by
  simp [phiPDF, gaussianPDFReal_def]

theorem phiPDF_pos (x : ℝ) : 0 < phiPDF x :=
  gaussianPDFReal_pos 0 1 x one_ne_zero

theorem phiPDF_zero : phiPDF 0 = 1 / √(2 * π) := by
  simp [phiPDF_eq]

theorem phiPDF_le_zero (x : ℝ) : phiPDF x ≤ phiPDF 0 := by
  rw [phiPDF_eq, phiPDF_eq]
  refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_) (by positivity)
  nlinarith [sq_nonneg x]

theorem continuous_phiPDF : Continuous phiPDF := by
  have : phiPDF = fun x ↦ (√(2 * π))⁻¹ * exp (-x ^ 2 / 2) := funext phiPDF_eq
  rw [this]
  fun_prop

theorem antitoneOn_phiPDF : AntitoneOn phiPDF (Ici 0) := by
  intro x hx y hy hxy
  rw [phiPDF_eq, phiPDF_eq]
  gcongr

theorem intervalIntegrable_phiPDF (a b : ℝ) : IntervalIntegrable phiPDF volume a b :=
  continuous_phiPDF.intervalIntegrable a b

/-- Probabilities of intervals are integrals of the density. -/
theorem Phi_sub_Phi {x y : ℝ} (hxy : x ≤ y) : Phi y - Phi x = ∫ t in x..y, phiPDF t := by
  have hm := (cdf (gaussianReal 0 1)).measure_Ioc x y
  rw [measure_cdf, gaussianReal_apply_eq_integral 0 one_ne_zero,
    ← intervalIntegral.integral_of_le hxy] at hm
  have hnn : 0 ≤ ∫ t in x..y, phiPDF t :=
    intervalIntegral.integral_nonneg hxy fun t _ ↦ (phiPDF_pos t).le
  have hd : 0 ≤ Phi y - Phi x := sub_nonneg.2 (monotone_cdf _ hxy)
  exact ((ENNReal.ofReal_eq_ofReal_iff hd hnn).1 hm.symm).symm.symm ▸
    ((ENNReal.ofReal_eq_ofReal_iff hnn hd).1 hm).symm ▸ rfl

/-- `Φ x = Φ 0 + ∫₀ˣ φ` for every `x`, with the signed interval integral. -/
theorem Phi_eq_Phi_zero_add (x : ℝ) : Phi x = Phi 0 + ∫ t in (0 : ℝ)..x, phiPDF t := by
  rcases le_total 0 x with h | h
  · linarith [Phi_sub_Phi h]
  · have := Phi_sub_Phi h
    rw [intervalIntegral.integral_symm]
    linarith

theorem hasDerivAt_Phi (x : ℝ) : HasDerivAt Phi (phiPDF x) x := by
  have hΦ : Phi = fun y ↦ Phi 0 + ∫ t in (0 : ℝ)..y, phiPDF t := funext Phi_eq_Phi_zero_add
  rw [hΦ]
  exact (intervalIntegral.integral_hasDerivAt_right (intervalIntegrable_phiPDF 0 x)
    (continuous_phiPDF.stronglyMeasurableAtFilter _ _) continuous_phiPDF.continuousAt).const_add _

theorem differentiable_Phi : Differentiable ℝ Phi := fun x ↦ (hasDerivAt_Phi x).differentiableAt

theorem continuous_Phi : Continuous Phi := differentiable_Phi.continuous

theorem deriv_Phi : deriv Phi = phiPDF := funext fun x ↦ (hasDerivAt_Phi x).deriv

theorem Phi_nonneg (x : ℝ) : 0 ≤ Phi x := cdf_nonneg _ x

theorem Phi_le_one (x : ℝ) : Phi x ≤ 1 := cdf_le_one _ x

theorem monotone_Phi : Monotone Phi := monotone_cdf _

theorem strictMono_Phi : StrictMono Phi := by
  intro x y hxy
  have h := Phi_sub_Phi hxy.le
  have hpos : 0 < ∫ t in x..y, phiPDF t :=
    intervalIntegral.intervalIntegral_pos_of_pos_on (intervalIntegrable_phiPDF x y)
      (fun t _ ↦ phiPDF_pos t) hxy
  linarith

/-- `Φ x = P(Z ≤ x)` for `Z` standard normal, as a real number. -/
theorem Phi_eq_real (x : ℝ) : Phi x = (gaussianReal 0 1).real (Iic x) := cdf_eq_real _ x

/-- The upper tail: `P(Z > x) = 1 - Φ x`. -/
theorem real_Ioi_eq (x : ℝ) : (gaussianReal 0 1).real (Ioi x) = 1 - Phi x := by
  rw [Phi_eq_real, ← compl_Iic, measureReal_compl measurableSet_Iic, probReal_univ]

/-- Symmetry: `P(Z ≥ -x) = Φ x`. -/
theorem real_Ici_neg_eq (x : ℝ) : (gaussianReal 0 1).real (Ici (-x)) = Phi x := by
  have hsym : (gaussianReal 0 1).map (fun y ↦ -y) = gaussianReal 0 1 := by
    simpa using gaussianReal_map_neg (μ := (0 : ℝ)) (v := 1)
  have : (gaussianReal 0 1).real (Ici (-x)) =
      ((gaussianReal 0 1).map (fun y ↦ -y)).real (Ici (-x)) := by rw [hsym]
  rw [this, map_measureReal_apply measurable_neg measurableSet_Ici, Phi_eq_real]
  congr 1
  ext y
  simp

/-- `P(Z > -x) = Φ x`, the strict version. -/
theorem real_Ioi_neg_eq (x : ℝ) : (gaussianReal 0 1).real (Ioi (-x)) = Phi x := by
  have := nullSingletonClass_gaussianReal (μ := (0 : ℝ)) (v := 1) one_ne_zero
  rw [← real_Ici_neg_eq]
  exact measureReal_congr Ioi_ae_eq_Ici

theorem Phi_neg (x : ℝ) : Phi (-x) = 1 - Phi x := by
  have h := real_Ioi_eq (-x)
  rw [real_Ioi_neg_eq] at h
  linarith

theorem Phi_zero : Phi 0 = 1 / 2 := by
  have h := Phi_neg 0
  rw [neg_zero] at h
  linarith

theorem half_le_Phi {u : ℝ} (hu : 0 ≤ u) : 1 / 2 ≤ Phi u := Phi_zero ▸ monotone_Phi hu

/-- **Tangent bound.** `Φ u ≤ 1/2 + u / √(2π)` for `u ≥ 0`, because the density is largest
at `0`. This is the relaxation behind the linear form of Proposition 1. -/
theorem Phi_le_tangent {u : ℝ} (hu : 0 ≤ u) : Phi u ≤ 1 / 2 + u / √(2 * π) := by
  rw [Phi_eq_Phi_zero_add, Phi_zero]
  have hle : ∫ t in (0 : ℝ)..u, phiPDF t ≤ ∫ _ in (0 : ℝ)..u, phiPDF 0 :=
    intervalIntegral.integral_mono_on hu (intervalIntegrable_phiPDF 0 u) intervalIntegrable_const
      fun t _ ↦ phiPDF_le_zero t
  rw [intervalIntegral.integral_const, smul_eq_mul, sub_zero, phiPDF_zero] at hle
  have : u * (1 / √(2 * π)) = u / √(2 * π) := by ring
  linarith

/-- `Φ` is concave on `[0, ∞)`: its derivative, the density, decreases there. -/
theorem concaveOn_Phi : ConcaveOn ℝ (Ici 0) Phi := by
  refine AntitoneOn.concaveOn_of_deriv (convex_Ici 0) continuous_Phi.continuousOn
    differentiable_Phi.differentiableOn ?_
  rw [deriv_Phi, interior_Ici]
  exact antitoneOn_phiPDF.mono Ioi_subset_Ici_self

/-- For `c ≥ 0`, `v ↦ Φ(c √v)` is concave on `[0, ∞)`. With `c = 1/(2σ)` this is the
function `g` of the proof of Proposition 1. -/
theorem concaveOn_Phi_sqrt {c : ℝ} (hc : 0 ≤ c) :
    ConcaveOn ℝ (Ici 0) (fun v ↦ Phi (c * √v)) := by
  refine ⟨convex_Ici 0, fun x hx y hy a b ha hb hab ↦ ?_⟩
  have hx' : c * √x ∈ Ici (0 : ℝ) := mul_nonneg hc (Real.sqrt_nonneg x)
  have hy' : c * √y ∈ Ici (0 : ℝ) := mul_nonneg hc (Real.sqrt_nonneg y)
  have h1 := concaveOn_Phi.2 hx' hy' ha hb hab
  have hs := strictConcaveOn_sqrt.concaveOn.2 hx hy ha hb hab
  simp only [smul_eq_mul] at h1 hs ⊢
  have h2 : a * (c * √x) + b * (c * √y) ≤ c * √(a * x + b * y) := by
    have := mul_le_mul_of_nonneg_left hs hc
    linarith
  exact h1.trans (monotone_Phi h2)

theorem monotoneOn_Phi_sqrt {c : ℝ} (hc : 0 ≤ c) :
    MonotoneOn (fun v ↦ Phi (c * √v)) (Ici 0) := fun _ _ _ _ hxy ↦
  monotone_Phi (mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hxy) hc)

theorem continuous_Phi_sqrt (c : ℝ) : Continuous (fun v ↦ Phi (c * √v)) :=
  continuous_Phi.comp (continuous_const.mul Real.continuous_sqrt)

end DGPCommitment
