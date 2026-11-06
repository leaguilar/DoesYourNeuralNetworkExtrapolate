import DGPCommitment.GaussVec

/-!
# The Gaussian two-point problem

Two simple hypotheses, `Y ~ N(μ₁, σ² I)` and `Y ~ N(μ₂, σ² I)`, with equal
priors. A (possibly randomised) test is a measurable `φ` with values in
`[0, 1]`, read as the probability of answering "1". Its two success
probabilities are `∫ φ dP₁` and `∫ (1 - φ) dP₂`.

* `twoPoint_le`: no test has average success above `Φ(S / (2σ))`, with
  `S = ‖μ₁ - μ₂‖`. This is Step 2 of the proof of Proposition 1
  (Neyman–Pearson optimality).
* `lr_success_add`: the likelihood-ratio test attains it. This is Proposition
  2(b) and the attainment clause of Proposition 1.
* `lr_success_one`, `lr_success_two`: when `μ₁ ≠ μ₂`, each hypothesis is
  recognised with probability exactly `Φ(S / (2σ))`.
-/

open MeasureTheory ProbabilityTheory Real Set
open scoped ENNReal NNReal

namespace DGPCommitment

variable {n : ℕ}

/-- The Euclidean distance `‖μ₁ - μ₂‖₂` between the two mean vectors. -/
noncomputable def sep (μ₁ μ₂ : Fin n → ℝ) : ℝ := √(∑ i, (μ₁ i - μ₂ i) ^ 2)

theorem sep_nonneg (μ₁ μ₂ : Fin n → ℝ) : 0 ≤ sep μ₁ μ₂ := Real.sqrt_nonneg _

theorem sep_sq (μ₁ μ₂ : Fin n → ℝ) : sep μ₁ μ₂ ^ 2 = ∑ i, (μ₁ i - μ₂ i) ^ 2 :=
  Real.sq_sqrt (by positivity)

theorem sep_pos_iff (μ₁ μ₂ : Fin n → ℝ) : 0 < sep μ₁ μ₂ ↔ μ₁ ≠ μ₂ := by
  unfold sep
  rw [Real.sqrt_pos]
  constructor
  · intro h heq
    subst heq
    simp at h
  · intro h
    obtain ⟨i, hi⟩ : ∃ i, μ₁ i ≠ μ₂ i := Function.ne_iff.1 h
    have : 0 < (μ₁ i - μ₂ i) ^ 2 := by positivity
    exact lt_of_lt_of_le this
      (Finset.single_le_sum (fun j _ ↦ sq_nonneg (μ₁ j - μ₂ j)) (Finset.mem_univ i))

/-- The likelihood-ratio statistic `⟨y - (μ₁ + μ₂)/2, μ₁ - μ₂⟩`. -/
noncomputable def lrStat (μ₁ μ₂ : Fin n → ℝ) (y : Fin n → ℝ) : ℝ :=
  ∑ i, (μ₁ i - μ₂ i) * (y i - (μ₁ i + μ₂ i) / 2)

/-- The likelihood-ratio test answers "1" on this half-space. -/
def lrRegion (μ₁ μ₂ : Fin n → ℝ) : Set (Fin n → ℝ) := {y | 0 ≤ lrStat μ₁ μ₂ y}

theorem measurable_lrStat (μ₁ μ₂ : Fin n → ℝ) : Measurable (lrStat μ₁ μ₂) := by
  unfold lrStat
  fun_prop

theorem measurableSet_lrRegion (μ₁ μ₂ : Fin n → ℝ) : MeasurableSet (lrRegion μ₁ μ₂) :=
  measurableSet_le measurable_const (measurable_lrStat μ₁ μ₂)

/-! ### The density comparison -/

theorem gaussVecPDF_eq (μ : Fin n → ℝ) (σ : ℝ) (y : Fin n → ℝ) :
    gaussVecPDF μ σ y = ENNReal.ofReal
      ((√(2 * π * σ ^ 2))⁻¹ ^ n * exp (-(∑ i, (y i - μ i) ^ 2) / (2 * σ ^ 2))) := by
  unfold gaussVecPDF
  simp_rw [gaussianPDF_def]
  rw [← ENNReal.ofReal_prod_of_nonneg (fun i _ ↦ gaussianPDFReal_nonneg _ _ _)]
  congr 1
  simp_rw [gaussianPDFReal_def, coe_varOf, Finset.prod_mul_distrib, Finset.prod_const,
    Finset.card_univ, Fintype.card_fin, ← Real.exp_sum]
  congr 2
  simp only [neg_div, Finset.sum_div, Finset.sum_neg_distrib]

/-- The likelihood-ratio region is exactly where the first density dominates. -/
theorem mem_lrRegion_iff (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : σ ≠ 0) (y : Fin n → ℝ) :
    y ∈ lrRegion μ₁ μ₂ ↔ gaussVecPDF μ₂ σ y ≤ gaussVecPDF μ₁ σ y := by
  have hσ2 : 0 < 2 * σ ^ 2 := by positivity
  have hC : 0 < (√(2 * π * σ ^ 2))⁻¹ ^ n := by
    have : 0 < √(2 * π * σ ^ 2) := Real.sqrt_pos.2 (by positivity)
    positivity
  rw [gaussVecPDF_eq, gaussVecPDF_eq, ENNReal.ofReal_le_ofReal_iff (by positivity),
    mul_le_mul_iff_of_pos_left hC, Real.exp_le_exp, div_le_div_iff_of_pos_right hσ2,
    neg_le_neg_iff]
  have key : ∑ i, (y i - μ₂ i) ^ 2 - ∑ i, (y i - μ₁ i) ^ 2 = 2 * lrStat μ₁ μ₂ y := by
    rw [← Finset.sum_sub_distrib, lrStat, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    ring
  simp only [lrRegion, mem_ofPred_eq]
  constructor <;> intro h <;> linarith

/-! ### Neyman–Pearson optimality -/

/-- **Step 2 of Proposition 1, optimality half.** No randomised test does better than the
likelihood-ratio test. -/
theorem np_le_lr (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : σ ≠ 0)
    (φ : (Fin n → ℝ) → ℝ≥0∞) (hφ : Measurable φ) (hφ1 : ∀ y, φ y ≤ 1) :
    ∫⁻ y, φ y ∂(gaussVec μ₁ σ) + ∫⁻ y, (1 - φ y) ∂(gaussVec μ₂ σ) ≤
      gaussVec μ₁ σ (lrRegion μ₁ μ₂) + gaussVec μ₂ σ (lrRegion μ₁ μ₂)ᶜ := by
  set H := lrRegion μ₁ μ₂
  have hH : MeasurableSet H := measurableSet_lrRegion μ₁ μ₂
  have hp₁ := measurable_gaussVecPDF μ₁ σ
  have hp₂ := measurable_gaussVecPDF μ₂ σ
  rw [gaussVec_eq_withDensity μ₁ hσ, gaussVec_eq_withDensity μ₂ hσ,
    lintegral_withDensity_eq_lintegral_mul _ hp₁ hφ,
    lintegral_withDensity_eq_lintegral_mul _ hp₂ (g := fun y ↦ 1 - φ y) (measurable_const.sub hφ),
    withDensity_apply _ hH, withDensity_apply _ hH.compl,
    ← lintegral_indicator hH, ← lintegral_indicator hH.compl,
    ← lintegral_add_left (hp₁.mul hφ), ← lintegral_add_left (hp₁.indicator hH)]
  refine lintegral_mono fun y ↦ ?_
  simp only [Pi.mul_apply]
  by_cases hy : y ∈ H
  · have hle := (mem_lrRegion_iff μ₁ μ₂ hσ y).1 hy
    rw [indicator_of_mem hy, indicator_of_notMem (show y ∉ Hᶜ from fun h ↦ h hy), add_zero]
    calc gaussVecPDF μ₁ σ y * φ y + gaussVecPDF μ₂ σ y * (1 - φ y)
        ≤ gaussVecPDF μ₁ σ y * φ y + gaussVecPDF μ₁ σ y * (1 - φ y) := by gcongr
      _ = gaussVecPDF μ₁ σ y := by rw [← mul_add, add_tsub_cancel_of_le (hφ1 y), mul_one]
  · have hle : gaussVecPDF μ₁ σ y ≤ gaussVecPDF μ₂ σ y :=
      le_of_lt (not_le.1 fun h ↦ hy ((mem_lrRegion_iff μ₁ μ₂ hσ y).2 h))
    rw [indicator_of_notMem hy, indicator_of_mem (show y ∈ Hᶜ from hy), zero_add]
    calc gaussVecPDF μ₁ σ y * φ y + gaussVecPDF μ₂ σ y * (1 - φ y)
        ≤ gaussVecPDF μ₂ σ y * φ y + gaussVecPDF μ₂ σ y * (1 - φ y) := by gcongr
      _ = gaussVecPDF μ₂ σ y := by rw [← mul_add, add_tsub_cancel_of_le (hφ1 y), mul_one]

/-! ### The exact success probability of the likelihood-ratio test -/

/-- `N(0, s²)` gives the half-line `[-c, ∞)` probability `Φ(c / s)`. -/
theorem gaussianReal_Ici_neg {s : ℝ} (hs : 0 < s) (c : ℝ) :
    gaussianReal 0 (varOf s) (Ici (-c)) = ENNReal.ofReal (Phi (c / s)) := by
  have hmap : gaussianReal 0 (varOf s) = (gaussianReal 0 1).map (fun x ↦ s * x) := by
    rw [gaussianReal_map_const_mul, mul_zero, mul_one]
    rfl
  rw [hmap, Measure.map_apply (measurable_const_mul s) measurableSet_Ici,
    ← real_Ici_neg_eq, ofReal_measureReal]
  congr 1
  ext x
  simp only [mem_preimage, mem_Ici]
  rw [show -(c / s) = (-c) / s by ring, div_le_iff₀ hs, mul_comm]

theorem gaussianReal_Iio {s : ℝ} (hs : 0 < s) (c : ℝ) :
    gaussianReal 0 (varOf s) (Iio c) = ENNReal.ofReal (Phi (c / s)) := by
  have hmap : gaussianReal 0 (varOf s) = (gaussianReal 0 1).map (fun x ↦ s * x) := by
    rw [gaussianReal_map_const_mul, mul_zero, mul_one]
    rfl
  have := nullSingletonClass_gaussianReal (μ := (0 : ℝ)) (v := 1) one_ne_zero
  rw [hmap, Measure.map_apply (measurable_const_mul s) measurableSet_Iio,
    show (fun x ↦ s * x) ⁻¹' Iio c = Iio (c / s) by
      ext x
      simp only [mem_preimage, mem_Iio]
      rw [lt_div_iff₀ hs, mul_comm],
    measure_congr Iio_ae_eq_Iic, ← ofReal_measureReal, ← Phi_eq_real]

theorem lrStat_add (μ₁ μ₂ z : Fin n → ℝ) :
    lrStat μ₁ μ₂ (μ₁ + z) = sep μ₁ μ₂ ^ 2 / 2 + ∑ i, (μ₁ i - μ₂ i) * z i ∧
      lrStat μ₁ μ₂ (μ₂ + z) = -(sep μ₁ μ₂ ^ 2 / 2) + ∑ i, (μ₁ i - μ₂ i) * z i := by
  rw [sep_sq]
  constructor <;>
  · simp only [lrStat, Pi.add_apply, Finset.sum_div, ← Finset.sum_neg_distrib,
      ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    ring

/-- **Proposition 2(b), per hypothesis.** When the means differ, the likelihood-ratio test
recognises `P₁` with probability exactly `Φ(S / (2σ))`. -/
theorem lr_success_one (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : 0 < σ) (hne : μ₁ ≠ μ₂) :
    gaussVec μ₁ σ (lrRegion μ₁ μ₂) = ENNReal.ofReal (Phi (sep μ₁ μ₂ / (2 * σ))) := by
  have hS := (sep_pos_iff μ₁ μ₂).2 hne
  rw [gaussVec_eq_map_add, Measure.map_apply (measurable_const_add μ₁)
    (measurableSet_lrRegion μ₁ μ₂)]
  have hpre : (fun z ↦ μ₁ + z) ⁻¹' lrRegion μ₁ μ₂ =
      (fun z ↦ ∑ i, (μ₁ i - μ₂ i) * z i) ⁻¹' Ici (-(sep μ₁ μ₂ ^ 2 / 2)) := by
    ext z
    simp only [mem_preimage, lrRegion, mem_ofPred_eq, mem_Ici, (lrStat_add μ₁ μ₂ z).1]
    constructor <;> intro h <;> linarith
  rw [hpre, ← Measure.map_apply (by fun_prop) measurableSet_Ici]
  have hlaw := map_linear_gaussVec (fun i ↦ μ₁ i - μ₂ i) σ
  rw [hlaw, show σ * √(∑ i, (μ₁ i - μ₂ i) ^ 2) = σ * sep μ₁ μ₂ from rfl,
    gaussianReal_Ici_neg (mul_pos hσ hS)]
  have hσ' := hσ.ne'
  have hS' := hS.ne'
  congr 2
  field_simp

/-- **Proposition 2(b), per hypothesis.** When the means differ, the likelihood-ratio test
recognises `P₂` with probability exactly `Φ(S / (2σ))`. -/
theorem lr_success_two (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : 0 < σ) (hne : μ₁ ≠ μ₂) :
    gaussVec μ₂ σ (lrRegion μ₁ μ₂)ᶜ = ENNReal.ofReal (Phi (sep μ₁ μ₂ / (2 * σ))) := by
  have hS := (sep_pos_iff μ₁ μ₂).2 hne
  rw [gaussVec_eq_map_add, Measure.map_apply (measurable_const_add μ₂)
    (measurableSet_lrRegion μ₁ μ₂).compl]
  have hpre : (fun z ↦ μ₂ + z) ⁻¹' (lrRegion μ₁ μ₂)ᶜ =
      (fun z ↦ ∑ i, (μ₁ i - μ₂ i) * z i) ⁻¹' Iio (sep μ₁ μ₂ ^ 2 / 2) := by
    ext z
    simp only [mem_preimage, lrRegion, mem_compl_iff, mem_ofPred_eq, mem_Iio, not_le,
      (lrStat_add μ₁ μ₂ z).2]
    constructor <;> intro h <;> linarith
  rw [hpre, ← Measure.map_apply (by fun_prop) measurableSet_Iio]
  have hlaw := map_linear_gaussVec (fun i ↦ μ₁ i - μ₂ i) σ
  rw [hlaw, show σ * √(∑ i, (μ₁ i - μ₂ i) ^ 2) = σ * sep μ₁ μ₂ from rfl,
    gaussianReal_Iio (mul_pos hσ hS)]
  have hσ' := hσ.ne'
  have hS' := hS.ne'
  congr 2
  field_simp

/-- **Proposition 2(b).** Under equal priors the likelihood-ratio test succeeds with
probability exactly `Φ(S / (2σ))`, stated as the sum of its two success probabilities. This
holds with no condition on the means. -/
theorem lr_success_add (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : 0 < σ) :
    gaussVec μ₁ σ (lrRegion μ₁ μ₂) + gaussVec μ₂ σ (lrRegion μ₁ μ₂)ᶜ =
      ENNReal.ofReal (2 * Phi (sep μ₁ μ₂ / (2 * σ))) := by
  by_cases hne : μ₁ = μ₂
  · subst hne
    have huniv : lrRegion μ₁ μ₁ = univ := by
      ext y
      simp [lrRegion, lrStat]
    have h0 : sep μ₁ μ₁ = 0 := by simp [sep]
    rw [huniv, compl_univ, measure_univ, measure_empty, h0, zero_div, Phi_zero]
    norm_num
  · rw [lr_success_one μ₁ μ₂ hσ hne, lr_success_two μ₁ μ₂ hσ hne,
      ← ENNReal.ofReal_add (Phi_nonneg _) (Phi_nonneg _)]
    congr 1
    ring

/-- **Step 2 of Proposition 1.** For two Gaussian vectors with common covariance `σ² I`,
every randomised test has summed success at most `2 Φ(S / (2σ))`, that is, average success at
most `Φ(S / (2σ))`. -/
theorem twoPoint_le (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : 0 < σ)
    (φ : (Fin n → ℝ) → ℝ≥0∞) (hφ : Measurable φ) (hφ1 : ∀ y, φ y ≤ 1) :
    ∫⁻ y, φ y ∂(gaussVec μ₁ σ) + ∫⁻ y, (1 - φ y) ∂(gaussVec μ₂ σ) ≤
      ENNReal.ofReal (2 * Phi (sep μ₁ μ₂ / (2 * σ))) :=
  (np_le_lr μ₁ μ₂ hσ.ne' φ hφ hφ1).trans (lr_success_add μ₁ μ₂ hσ).le

/-- The likelihood-ratio test, as a randomised test, attains `twoPoint_le`. -/
theorem twoPoint_lr_attains (μ₁ μ₂ : Fin n → ℝ) {σ : ℝ} (hσ : 0 < σ) :
    ∫⁻ y, (lrRegion μ₁ μ₂).indicator 1 y ∂(gaussVec μ₁ σ) +
      ∫⁻ y, (1 - (lrRegion μ₁ μ₂).indicator 1 y) ∂(gaussVec μ₂ σ) =
      ENNReal.ofReal (2 * Phi (sep μ₁ μ₂ / (2 * σ))) := by
  have hH := measurableSet_lrRegion μ₁ μ₂
  have hc : (fun y ↦ 1 - (lrRegion μ₁ μ₂).indicator (1 : (Fin n → ℝ) → ℝ≥0∞) y) =
      (lrRegion μ₁ μ₂)ᶜ.indicator 1 := by
    ext y
    by_cases hy : y ∈ lrRegion μ₁ μ₂ <;> simp [hy]
  rw [hc, lintegral_indicator_one hH, lintegral_indicator_one hH.compl]
  exact lr_success_add μ₁ μ₂ hσ

end DGPCommitment
