import DGPCommitment.TwoPoint
import DGPCommitment.Floor

/-!
# Proposition 1: ERM is blind to observationally-equivalent processes

The sample `Dₙ = (x, y)` has `xᵢ` i.i.d. from the training distribution `D`,
noise `ηᵢ` i.i.d. `N(0, σ²)` independent of the design, and `yᵢ = P(xᵢ) + ηᵢ`.
`sampleLaw D n σ P` is the law of `Dₙ`. A test is any measurable function of
`Dₙ` with values in `[0, 1]`, so deterministic tests, randomised tests and
every selection criterion computed on the window (cross-validation, training
loss, held-out loss) are covered by one statement.

The proof follows the appendix: condition on the design and use the exact
Gaussian two-point bound (`twoPoint_le`), then average over the design with
Jensen's inequality for the concave map `v ↦ Φ(√v / (2σ))`.
-/

open MeasureTheory ProbabilityTheory Real Set
open scoped ENNReal NNReal

namespace DGPCommitment

variable {X : Type*} [MeasurableSpace X]

/-- The law of the design `(x₁, …, xₙ)`, i.i.d. from `D`. -/
noncomputable def design (D : Measure X) (n : ℕ) : Measure (Fin n → X) :=
  Measure.pi fun _ ↦ D

instance (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) :
    IsProbabilityMeasure (design D n) := by
  unfold design
  infer_instance

/-- The law of the sample `Dₙ = (x, y)` when the data come from the process `P`. -/
noncomputable def sampleLaw (D : Measure X) (n : ℕ) (σ : ℝ) (P : X → ℝ) :
    Measure ((Fin n → X) × (Fin n → ℝ)) :=
  ((design D n).prod (gaussVec 0 σ)).map fun p ↦ (p.1, (fun i ↦ P (p.1 i)) + p.2)

theorem measurable_sampleMap {n : ℕ} {P : X → ℝ} (hP : Measurable P) :
    Measurable fun p : (Fin n → X) × (Fin n → ℝ) ↦ (p.1, (fun i ↦ P (p.1 i)) + p.2) := by
  refine measurable_fst.prodMk (Measurable.add ?_ measurable_snd)
  exact measurable_pi_iff.2 fun i ↦ hP.comp ((measurable_pi_apply i).comp measurable_fst)

instance (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) (σ : ℝ) (P : X → ℝ) :
    IsFiniteMeasure (sampleLaw D n σ P) := by
  unfold sampleLaw
  infer_instance

/-- Tonelli over design and noise, with the noise law held fixed. -/
theorem lintegral_sampleLaw (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) (σ : ℝ)
    {P : X → ℝ} (hP : Measurable P)
    (φ : (Fin n → X) × (Fin n → ℝ) → ℝ≥0∞) (hφ : Measurable φ) :
    ∫⁻ d, φ d ∂(sampleLaw D n σ P) =
      ∫⁻ x, ∫⁻ z, φ (x, (fun i ↦ P (x i)) + z) ∂(gaussVec 0 σ) ∂(design D n) := by
  unfold sampleLaw
  rw [lintegral_map hφ (measurable_sampleMap hP),
    lintegral_prod (fun p ↦ φ (p.1, (fun i ↦ P (p.1 i)) + p.2))
      (hφ.comp (measurable_sampleMap hP)).aemeasurable]

omit [MeasurableSpace X] in
/-- Conditioning on the design: given `x`, the responses are the Gaussian vector with mean
`(P(xᵢ))ᵢ`. -/
theorem lintegral_noise_eq {n : ℕ} (x : Fin n → X) (σ : ℝ) (P : X → ℝ)
    (ψ : (Fin n → ℝ) → ℝ≥0∞) (hψ : Measurable ψ) :
    ∫⁻ z, ψ ((fun i ↦ P (x i)) + z) ∂(gaussVec 0 σ) =
      ∫⁻ y, ψ y ∂(gaussVec (fun i ↦ P (x i)) σ) := by
  rw [gaussVec_eq_map_add (fun i ↦ P (x i)) σ, lintegral_map hψ (measurable_const_add _)]

/-! ### The design average -/

/-- The squared distance between the two mean vectors that a design `x` produces. -/
noncomputable def designGap (P₁ P₂ : X → ℝ) {n : ℕ} (x : Fin n → X) : ℝ :=
  ∑ i, (P₁ (x i) - P₂ (x i)) ^ 2

omit [MeasurableSpace X] in
theorem sep_eq_sqrt_designGap (P₁ P₂ : X → ℝ) {n : ℕ} (x : Fin n → X) :
    sep (fun i ↦ P₁ (x i)) (fun i ↦ P₂ (x i)) = √(designGap P₁ P₂ x) := rfl

/-- `E[Sₙ²] = n ‖P₁ - P₂‖²_{L²(D)}` for an i.i.d. design. The same identity gives
`E[S_V²] = n_V ‖Δ‖²_{L²(D_V)}` in the remark on which measure the subscript names. -/
theorem integral_designGap (D : Measure X) [IsProbabilityMeasure D] (n : ℕ)
    {P₁ P₂ : X → ℝ} (hP₁ : Measurable P₁) (hP₂ : Measurable P₂)
    (hint : Integrable (fun x ↦ (P₁ x - P₂ x) ^ 2) D) :
    Integrable (designGap P₁ P₂) (design D n) ∧
      ∫ x, designGap P₁ P₂ x ∂(design D n) = n * ∫ x, (P₁ x - P₂ x) ^ 2 ∂D := by
  have hm : AEStronglyMeasurable (fun x ↦ (P₁ x - P₂ x) ^ 2) D :=
    ((hP₁.sub hP₂).pow_const 2).aestronglyMeasurable
  have hterm : ∀ i : Fin n, Integrable (fun x : Fin n → X ↦ (P₁ (x i) - P₂ (x i)) ^ 2)
      (design D n) := fun i ↦
    integrable_comp_eval (μ := fun _ : Fin n ↦ D) (f := fun x ↦ (P₁ x - P₂ x) ^ 2) hint
  refine ⟨?_, ?_⟩
  · unfold designGap
    exact integrable_finsetSum _ fun i _ ↦ hterm i
  · unfold designGap
    rw [integral_finsetSum _ fun i _ ↦ hterm i]
    simp_rw [design, integral_comp_eval (μ := fun _ : Fin n ↦ D)
      (f := fun x ↦ (P₁ x - P₂ x) ^ 2) hm]
    simp

/-- **Step 3 of Proposition 1.** Averaging the conditional bound over the design:
`E_X[Φ(Sₙ / (2σ))] ≤ Φ(t_W / 2)`, by Jensen for the concave map `v ↦ Φ(√v / (2σ))`. -/
theorem jensen_design (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) {σ : ℝ} (hσ : 0 < σ)
    {P₁ P₂ : X → ℝ} (hP₁ : Measurable P₁) (hP₂ : Measurable P₂) {ε : ℝ} (hε : 0 ≤ ε)
    (hobs : ObsEquiv D P₁ P₂ ε) :
    ∫ x, Phi (sep (fun i ↦ P₁ (x i)) (fun i ↦ P₂ (x i)) / (2 * σ)) ∂(design D n) ≤
      Phi (tW n ε σ / 2) := by
  set c : ℝ := 1 / (2 * σ) with hc
  have hc0 : 0 ≤ c := by positivity
  set g : ℝ → ℝ := fun v ↦ Phi (c * √v)
  obtain ⟨hQi, hQ⟩ := integral_designGap D n hP₁ hP₂ hobs.1
  have hrew : (fun x : Fin n → X ↦ Phi (sep (fun i ↦ P₁ (x i)) (fun i ↦ P₂ (x i)) / (2 * σ))) =
      g ∘ designGap P₁ P₂ := by
    ext x
    simp only [Function.comp, g, sep_eq_sqrt_designGap, hc]
    ring_nf
  rw [hrew]
  have hgi : Integrable (g ∘ designGap P₁ P₂) (design D n) := by
    refine Integrable.of_bound (C := 1) ?_ (Filter.Eventually.of_forall fun x ↦ ?_)
    · exact ((continuous_Phi_sqrt c).measurable.comp
        (Finset.measurable_sum _ fun i _ ↦
          ((hP₁.comp (measurable_pi_apply i)).sub (hP₂.comp (measurable_pi_apply i))).pow_const
            2)).aestronglyMeasurable
    · simp only [Function.comp, g, Real.norm_eq_abs]
      rw [abs_of_nonneg (Phi_nonneg _)]
      exact Phi_le_one _
  have hjensen := (concaveOn_Phi_sqrt hc0).le_map_integral (continuous_Phi_sqrt c).continuousOn
    isClosed_Ici (Filter.Eventually.of_forall fun x ↦
      (show (0 : ℝ) ≤ designGap P₁ P₂ x from Finset.sum_nonneg fun i _ ↦ sq_nonneg _))
    hQi hgi
  refine hjensen.trans ?_
  have hmono : ∫ x, designGap P₁ P₂ x ∂(design D n) ≤ n * ε ^ 2 := by
    rw [hQ]
    exact mul_le_mul_of_nonneg_left hobs.2 (Nat.cast_nonneg n)
  have hQnn : 0 ≤ ∫ x, designGap P₁ P₂ x ∂(design D n) :=
    integral_nonneg fun x ↦ Finset.sum_nonneg fun i _ ↦ sq_nonneg _
  refine (monotoneOn_Phi_sqrt hc0 hQnn (by positivity : (0 : ℝ) ≤ n * ε ^ 2) hmono).trans_eq ?_
  simp only [tW, hc]
  congr 1
  rw [Real.sqrt_mul (Nat.cast_nonneg n), Real.sqrt_sq hε]
  field_simp

theorem measurable_sep_design {n : ℕ} {P₁ P₂ : X → ℝ} (hP₁ : Measurable P₁)
    (hP₂ : Measurable P₂) :
    Measurable fun x : Fin n → X ↦ sep (fun i ↦ P₁ (x i)) (fun i ↦ P₂ (x i)) := by
  simp_rw [sep_eq_sqrt_designGap, designGap]
  exact Real.continuous_sqrt.measurable.comp (Finset.measurable_sum _ fun i _ ↦
    ((hP₁.comp (measurable_pi_apply i)).sub (hP₂.comp (measurable_pi_apply i))).pow_const 2)

/-! ### Proposition 1 -/

/-- **Proposition 1 (ERM is blind to observationally-equivalent processes).** If `P₁, P₂` are
`ε_W`-observationally equivalent under `D`, every test computed on `n` noisy samples has summed
success at most `2 Φ(t_W / 2)`, that is, average success at most `Φ(t_W / 2)` with
`t_W = √n ε_W / σ`. -/
theorem erm_blind (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) {σ : ℝ} (hσ : 0 < σ)
    {P₁ P₂ : X → ℝ} (hP₁ : Measurable P₁) (hP₂ : Measurable P₂) {ε : ℝ} (hε : 0 ≤ ε)
    (hobs : ObsEquiv D P₁ P₂ ε)
    (φ : (Fin n → X) × (Fin n → ℝ) → ℝ≥0∞) (hφ : Measurable φ) (hφ1 : ∀ d, φ d ≤ 1) :
    ∫⁻ d, φ d ∂(sampleLaw D n σ P₁) + ∫⁻ d, (1 - φ d) ∂(sampleLaw D n σ P₂) ≤
      ENNReal.ofReal (2 * Phi (tW n ε σ / 2)) := by
  have hφ' : Measurable fun d ↦ 1 - φ d := measurable_const.sub hφ
  have hm₁ : Measurable fun x : Fin n → X ↦
      ∫⁻ z, φ (x, (fun i ↦ P₁ (x i)) + z) ∂(gaussVec 0 σ) :=
    (hφ.comp (measurable_sampleMap hP₁)).lintegral_prod_right'
  rw [lintegral_sampleLaw D n σ hP₁ φ hφ, lintegral_sampleLaw D n σ hP₂ _ hφ',
    ← lintegral_add_left hm₁]
  set S : (Fin n → X) → ℝ := fun x ↦ sep (fun i ↦ P₁ (x i)) (fun i ↦ P₂ (x i))
  have hS : Measurable S := measurable_sep_design hP₁ hP₂
  have hB : Integrable (fun x ↦ 2 * Phi (S x / (2 * σ))) (design D n) := by
    refine Integrable.of_bound (C := 2) ((((continuous_Phi.measurable.comp
      (hS.div_const _))).const_mul 2).aestronglyMeasurable) (Filter.Eventually.of_forall
      fun x ↦ ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (by linarith [Phi_nonneg (S x / (2 * σ))])]
    linarith [Phi_le_one (S x / (2 * σ))]
  calc
    _ ≤ ∫⁻ x, ENNReal.ofReal (2 * Phi (S x / (2 * σ))) ∂(design D n) := by
      refine lintegral_mono fun x ↦ ?_
      have hx : Measurable fun y ↦ φ (x, y) := hφ.comp measurable_prodMk_left
      rw [lintegral_noise_eq x σ P₁ (fun y ↦ φ (x, y)) hx,
        lintegral_noise_eq x σ P₂ (fun y ↦ 1 - φ (x, y)) (measurable_const.sub hx)]
      exact twoPoint_le _ _ hσ (fun y ↦ φ (x, y)) hx fun y ↦ hφ1 _
    _ = ENNReal.ofReal (∫ x, 2 * Phi (S x / (2 * σ)) ∂(design D n)) :=
      (ofReal_integral_eq_lintegral_ofReal hB
        (Filter.Eventually.of_forall fun x ↦ by
          have := Phi_nonneg (S x / (2 * σ))
          simp only [Pi.zero_apply]
          linarith)).symm
    _ ≤ ENNReal.ofReal (2 * Phi (tW n ε σ / 2)) := by
      refine ENNReal.ofReal_le_ofReal ?_
      rw [integral_const_mul]
      exact mul_le_mul_of_nonneg_left (jensen_design D n hσ hP₁ hP₂ hε hobs) (by norm_num)

/-- **Step 1 of Proposition 1.** The worse of the two success probabilities is at most their
average, so Proposition 1 bounds `min_k Pr_k[T = k]`. -/
theorem min_le_half_add (a b : ℝ≥0∞) : min a b ≤ (a + b) / 2 := by
  rcases le_total a b with h | h
  · rw [min_eq_left h, ENNReal.le_div_iff_mul_le (by norm_num) (by norm_num), mul_two]
    exact add_le_add le_rfl h
  · rw [min_eq_right h, ENNReal.le_div_iff_mul_le (by norm_num) (by norm_num), mul_two]
    exact add_le_add h le_rfl

/-- **Proposition 1, as stated.** `min_k Pr_k[T = k] ≤ Φ(t_W / 2)` for every randomised test. -/
theorem erm_blind_min (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) {σ : ℝ} (hσ : 0 < σ)
    {P₁ P₂ : X → ℝ} (hP₁ : Measurable P₁) (hP₂ : Measurable P₂) {ε : ℝ} (hε : 0 ≤ ε)
    (hobs : ObsEquiv D P₁ P₂ ε)
    (φ : (Fin n → X) × (Fin n → ℝ) → ℝ≥0∞) (hφ : Measurable φ) (hφ1 : ∀ d, φ d ≤ 1) :
    min (∫⁻ d, φ d ∂(sampleLaw D n σ P₁)) (∫⁻ d, (1 - φ d) ∂(sampleLaw D n σ P₂)) ≤
      ENNReal.ofReal (Phi (tW n ε σ / 2)) := by
  refine (min_le_half_add _ _).trans ?_
  rw [ENNReal.div_le_iff (by norm_num) (by norm_num), ← ENNReal.ofReal_ofNat 2,
    ← ENNReal.ofReal_mul (Phi_nonneg _), mul_comm]
  exact erm_blind D n hσ hP₁ hP₂ hε hobs φ hφ hφ1

/-- **Proposition 1 for a deterministic test.** Any measurable selection rule `T` on the sample,
cross-validation and held-out loss included, picks `P₁` under `P₁` and `P₂` under `P₂` with
summed probability at most `2 Φ(t_W / 2)`. -/
theorem erm_blind_deterministic (D : Measure X) [IsProbabilityMeasure D] (n : ℕ) {σ : ℝ}
    (hσ : 0 < σ) {P₁ P₂ : X → ℝ} (hP₁ : Measurable P₁) (hP₂ : Measurable P₂) {ε : ℝ}
    (hε : 0 ≤ ε) (hobs : ObsEquiv D P₁ P₂ ε)
    (T : (Fin n → X) × (Fin n → ℝ) → Bool) (hT : Measurable T) :
    sampleLaw D n σ P₁ (T ⁻¹' {true}) + sampleLaw D n σ P₂ (T ⁻¹' {false}) ≤
      ENNReal.ofReal (2 * Phi (tW n ε σ / 2)) := by
  set A := T ⁻¹' {true}
  have hA : MeasurableSet A := hT (measurableSet_singleton true)
  have hAc : T ⁻¹' {false} = Aᶜ := by
    ext d
    simp [A]
  have h := erm_blind D n hσ hP₁ hP₂ hε hobs (A.indicator (1 : (Fin n → X) × (Fin n → ℝ) → ℝ≥0∞))
    (measurable_one.indicator hA) (fun d ↦ by by_cases hd : d ∈ A <;> simp [hd])
  have hc : (fun d ↦ 1 - A.indicator (1 : (Fin n → X) × (Fin n → ℝ) → ℝ≥0∞) d) = Aᶜ.indicator 1 := by
    ext d
    by_cases hd : d ∈ A <;> simp [hd]
  rw [hc, lintegral_indicator_one hA, lintegral_indicator_one hA.compl] at h
  rw [hAc]
  exact h

/-! ### The linear form and the Le Cam–Pinsker comparison -/

/-- **Linear form of Proposition 1.** `Φ(t_W / 2) ≤ 1/2 + t_W / (2√(2π))`. -/
theorem linear_form {t : ℝ} (ht : 0 ≤ t) : Phi (t / 2) ≤ 1 / 2 + t / (2 * √(2 * π)) := by
  have := Phi_le_tangent (u := t / 2) (by positivity)
  calc Phi (t / 2) ≤ 1 / 2 + t / 2 / √(2 * π) := this
    _ = 1 / 2 + t / (2 * √(2 * π)) := by rw [div_div]

/-- The linear bound reaches `1`, and so stops constraining any test, exactly at
`σ = √n ε_W / √(2π)`. -/
theorem linear_form_eq_one_iff {n : ℕ} {ε σ : ℝ} (hσ : 0 < σ) :
    1 / 2 + tW n ε σ / (2 * √(2 * π)) = 1 ↔ σ = √n * ε / √(2 * π) := by
  have hπ : 0 < √(2 * π) := Real.sqrt_pos.2 (by positivity)
  have h1 : ∀ a : ℝ, 1 / 2 + a / (2 * √(2 * π)) = 1 ↔ a = √(2 * π) := fun a ↦ by
    constructor
    · intro h
      have h' : a / (2 * √(2 * π)) = 1 / 2 := by linarith
      rw [div_eq_iff (by positivity)] at h'
      linarith
    · intro h
      have h' : √(2 * π) / (2 * √(2 * π)) = 1 / 2 := by
        rw [div_eq_iff (by positivity)]
        ring
      rw [h, h']
      norm_num
  unfold tW
  rw [h1, div_eq_iff hσ.ne', eq_div_iff hπ.ne']
  constructor <;> intro h <;> linarith

/-- `√(2π) > 2`, so the exact route's constant beats Pinsker's. -/
theorem two_lt_sqrt_two_pi : 2 < √(2 * π) := by
  rw [Real.lt_sqrt (by norm_num)]
  nlinarith [Real.pi_gt_three]

/-- **Relation to the Le Cam–Pinsker route.** The linear constant `1/(2√(2π))` lies in
`(0.199, 0.2)`, strictly below Pinsker's `1/4`. -/
theorem linear_constant_bounds :
    (0.199 : ℝ) < 1 / (2 * √(2 * π)) ∧ 1 / (2 * √(2 * π)) < 0.2 ∧ 1 / (2 * √(2 * π)) < 1 / 4 := by
  have hlo : √(2 * π) < 2.51 := by
    rw [Real.sqrt_lt' (by norm_num)]
    nlinarith [Real.pi_lt_d2]
  have hhi : 2.5 < √(2 * π) := by
    rw [Real.lt_sqrt (by norm_num)]
    nlinarith [Real.pi_gt_d2]
  refine ⟨?_, ?_, ?_⟩
  · rw [lt_div_iff₀ (by positivity)]
    nlinarith
  · rw [div_lt_iff₀ (by positivity)]
    nlinarith
  · rw [div_lt_div_iff₀ (by positivity) (by norm_num)]
    nlinarith [two_lt_sqrt_two_pi]

/-- **The Pinsker-shaped bound.** Every test also satisfies `min_k Pr_k[T = k] ≤ 1/2 + t_W/4`,
the bound the Le Cam–Pinsker route gives. It follows here from the sharper linear form. -/
theorem pinsker_shape {t : ℝ} (ht : 0 ≤ t) : Phi (t / 2) ≤ 1 / 2 + t / 4 := by
  refine (linear_form ht).trans ?_
  have h := (linear_constant_bounds).2.2
  have : t / (2 * √(2 * π)) = t * (1 / (2 * √(2 * π))) := by ring
  rw [this]
  nlinarith

/-- **The numerical illustration, Pinsker side.** At `n = 200`, `σ = 0.5`, `ε_W = 0.071` the
Pinsker-shaped bound is at least `1`, so it constrains no test. (The exact value
`Φ(t_W / 2) ≈ 0.842` is a numerical evaluation and is not part of this library.) -/
theorem pinsker_illustration_vacuous : 1 ≤ 1 / 2 + tW 200 0.071 0.5 / 4 := by
  unfold tW
  have h : (14.1 : ℝ) ≤ √(200 : ℕ) := by
    rw [Real.le_sqrt (by norm_num) (by norm_num)]
    norm_num
  push_cast at h ⊢
  norm_num
  nlinarith [h]

/-! ### Proposition 2(a) and 2(d) -/

/-- **Proposition 2(a)** is Proposition 1. -/
alias prop2a := erm_blind

/-- Every level strictly between `0` and `1` is a value of `Φ`. -/
theorem Phi_surj {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1) : ∃ q, Phi q = p := by
  have h0 := tendsto_cdf_atBot (gaussianReal 0 1)
  have h1 := tendsto_cdf_atTop (gaussianReal 0 1)
  obtain ⟨a, ha⟩ := (h0.eventually (gt_mem_nhds hp0)).exists
  obtain ⟨b, hb⟩ := (h1.eventually (lt_mem_nhds hp1)).exists
  exact mem_range_of_exists_le_of_exists_ge continuous_Phi ⟨a, ha.le⟩ ⟨b, hb.le⟩

/-- The standard normal quantile `Φ⁻¹`, meaningful on `(0, 1)`. -/
noncomputable def PhiInv (p : ℝ) : ℝ := Function.invFun Phi p

theorem Phi_PhiInv {p : ℝ} (hp0 : 0 < p) (hp1 : p < 1) : Phi (PhiInv p) = p :=
  Function.invFun_eq (Phi_surj hp0 hp1)

theorem le_two_PhiInv_iff {t p : ℝ} (hp0 : 0 < p) (hp1 : p < 1) :
    t ≤ 2 * PhiInv p ↔ Phi (t / 2) ≤ p := by
  rw [← Phi_PhiInv hp0 hp1, strictMono_Phi.le_iff_le, Phi_PhiInv hp0 hp1]
  constructor <;> intro h <;> linarith

/-- **Proposition 2(d).** For `α ∈ (0, 1/4)`, the two threshold conditions put the window below
`1/2 + α` and the strip above `1 - α`, and these two levels are strictly ordered, so the strip
resolves what the window cannot. -/
theorem prop2d {α tWv tVv : ℝ} (hα0 : 0 < α) (hα : α < 1 / 4)
    (hW : tWv ≤ 2 * PhiInv (1 / 2 + α)) (hV : 2 * PhiInv (1 - α) ≤ tVv) :
    Phi (tWv / 2) ≤ 1 / 2 + α ∧ 1 - α ≤ Phi (tVv / 2) ∧ 1 / 2 + α < 1 - α := by
  refine ⟨(le_two_PhiInv_iff (by linarith) (by linarith)).1 hW, ?_, by linarith⟩
  rw [← Phi_PhiInv (p := 1 - α) (by linarith) (by linarith)]
  exact monotone_Phi (by linarith)

/-- **Precision note on Proposition 2(d).** The proof says the two conditions are
"simultaneously satisfiable exactly when `α < 1/4`". The conditions constrain two different
numbers, `t_W` and `t_V`, so they can always hold together. What `α < 1/4` decides is whether
the window band lies strictly below the strip band. -/
theorem prop2d_band_iff (α : ℝ) : 1 / 2 + α < 1 - α ↔ α < 1 / 4 := by
  constructor <;> intro h <;> linarith

theorem prop2d_conditions_always_jointly_satisfiable {α : ℝ} (_hα0 : 0 < α) (_hα1 : α < 1 / 2) :
    ∃ tWv tVv : ℝ, tWv ≤ 2 * PhiInv (1 / 2 + α) ∧ 2 * PhiInv (1 - α) ≤ tVv :=
  ⟨2 * PhiInv (1 / 2 + α), 2 * PhiInv (1 - α), le_rfl, le_rfl⟩

end DGPCommitment
