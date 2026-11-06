import DGPCommitment.Basic

/-!
# Realisable-case canonicalisation and the three linearising instances

Lemma 4 of the appendix and its three instances (Lemmas 5, 6, 7), with the
remarks on phase and amplitude, on the reparameterisation between the power-law
and exponential instances, and on the noise model under `ψ = log`.
-/

open MeasureTheory ProbabilityTheory Real Matrix

namespace DGPCommitment

/-! ### Lemma 4 -/

/-- A strictly monotone label map has a left inverse, which is all Lemma 4 uses of
"strictly monotonic (hence invertible)". -/
theorem exists_leftInverse_of_strictMono {ψ : ℝ → ℝ} (hψ : StrictMono ψ) :
    ∃ ψinv : ℝ → ℝ, ∀ y, ψinv (ψ y) = y :=
  ⟨Function.invFun ψ, Function.leftInverse_invFun hψ.injective⟩

/-- Under realisability on all of `X`, the committed predictor is the target itself. -/
theorem realisable_predictor_eq {X Z : Type*} (f : X → ℝ) (φ : X → Z) (ψ ψinv : ℝ → ℝ)
    (hinv : ∀ x, ψinv (ψ (f x)) = f x) (gstar : Z → ℝ)
    (hreal : ∀ x, gstar (φ x) = ψ (f x)) :
    ψinv ∘ gstar ∘ φ = f := by
  funext x
  simp [Function.comp, hreal, hinv]

/-- **Lemma 4 (Realisable-case canonicalisation).** If some `g⋆ ∈ M` realises `ψ ∘ f`
through `φ` on the whole input space, and the learner returns `g⋆` almost surely, then the
predictor `ψ⁻¹ ∘ ĝ ∘ φ` has zero out-of-distribution risk, almost surely, for every test
distribution. -/
theorem canonical_epsOOD_eq_zero {X Z Ω : Type*} [MeasurableSpace X] [MeasurableSpace Ω]
    (f : X → ℝ) (φ : X → Z) (ψ ψinv : ℝ → ℝ) (M : Set (Z → ℝ))
    (hinv : ∀ x, ψinv (ψ (f x)) = f x)
    (gstar : Z → ℝ) (_hgM : gstar ∈ M) (hreal : ∀ x, gstar (φ x) = ψ (f x))
    (P : Measure Ω) (ghat : Ω → Z → ℝ) (hid : ∀ᵐ ω ∂P, ghat ω = gstar)
    (Dtest : Measure X) :
    ∀ᵐ ω ∂P, epsOOD Dtest f (ψinv ∘ ghat ω ∘ φ) = 0 := by
  filter_upwards [hid] with ω hω
  rw [hω, realisable_predictor_eq f φ ψ ψinv hinv gstar hreal]
  simp [epsOOD]

/-- Lemma 4 with `ψ = id`, the standard formulation. -/
theorem canonical_epsOOD_eq_zero_id {X Z Ω : Type*} [MeasurableSpace X] [MeasurableSpace Ω]
    (f : X → ℝ) (φ : X → Z) (M : Set (Z → ℝ))
    (gstar : Z → ℝ) (hgM : gstar ∈ M) (hreal : ∀ x, gstar (φ x) = f x)
    (P : Measure Ω) (ghat : Ω → Z → ℝ) (hid : ∀ᵐ ω ∂P, ghat ω = gstar)
    (Dtest : Measure X) :
    ∀ᵐ ω ∂P, epsOOD Dtest f (ghat ω ∘ φ) = 0 :=
  canonical_epsOOD_eq_zero f φ id id M (fun _ ↦ rfl) gstar hgM hreal P ghat hid Dtest

/-- The risk vanishes when the predictor agrees with the target almost everywhere under the
test distribution. The three instances use this with the test distribution's support. -/
theorem epsOOD_eq_zero_of_ae_eq {X : Type*} [MeasurableSpace X] (Dtest : Measure X)
    (f fhat : X → ℝ) (h : ∀ᵐ x ∂Dtest, fhat x = f x) : epsOOD Dtest f fhat = 0 := by
  unfold epsOOD
  rw [integral_congr_ae (g := fun _ ↦ (0 : ℝ))]
  · simp
  · filter_upwards [h] with x hx
    simp [hx]

/-! ### The three instances -/

/-- The sinusoidal feature map `φ x = (sin x, cos x)`. -/
noncomputable def phiSin (x : ℝ) : Fin 2 → ℝ := ![sin x, cos x]

/-- **Lemma 5 (Sinusoidal instance).** The weight `(1, 0)` realises `sin` through
`(sin, cos)` on all of `ℝ`. -/
theorem sin_realisable (x : ℝ) : ![1, 0] ⬝ᵥ phiSin x = sin x := by
  simp [phiSin, dotProduct, Fin.sum_univ_two]

/-- Lemma 5, risk form: the linear predictor with weight `(1, 0)` has zero risk on every
test distribution. -/
theorem sin_epsOOD_eq_zero (Dtest : Measure ℝ) :
    epsOOD Dtest sin (fun x ↦ ![1, 0] ⬝ᵥ phiSin x) = 0 :=
  epsOOD_eq_zero_of_ae_eq Dtest _ _ (Filter.Eventually.of_forall sin_realisable)

/-- **Remark (Phase-amplitude generalisation), first half.** The weight
`(A cos φ, A sin φ)` realises `A sin (x + φ)`. -/
theorem phase_amplitude_realisable (A φ x : ℝ) :
    ![A * cos φ, A * sin φ] ⬝ᵥ phiSin x = A * sin (x + φ) := by
  simp only [phiSin, dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, sin_add]
  ring

/-- **Remark (Phase-amplitude generalisation), second half.** Every element of the span of
`sin` and `cos` is a phase-shifted sinusoid with phase in `[0, 2π)`. -/
theorem span_sin_cos_eq_phase_amplitude (a b : ℝ) :
    ∃ A φ : ℝ, φ ∈ Set.Ico 0 (2 * π) ∧ ∀ x, a * sin x + b * cos x = A * sin (x + φ) := by
  set A : ℝ := √(a ^ 2 + b ^ 2) with hA
  by_cases h0 : A = 0
  · have hab : a ^ 2 + b ^ 2 = 0 := by
      have := Real.sqrt_eq_zero'.1 h0
      nlinarith [sq_nonneg a, sq_nonneg b]
    have ha : a = 0 := by nlinarith [sq_nonneg a, sq_nonneg b]
    have hb : b = 0 := by nlinarith [sq_nonneg a, sq_nonneg b]
    exact ⟨0, 0, ⟨le_rfl, by positivity⟩, fun x ↦ by simp [ha, hb]⟩
  · set z : ℂ := ⟨a, b⟩ with hz
    have hnorm : ‖z‖ = A := by
      rw [hA, Complex.norm_def, Complex.normSq_apply]
      congr 1
      simp [hz]
      ring
    set θ : ℝ := Complex.arg z
    have hz0 : z ≠ 0 := by
      intro hz0
      apply h0
      rw [← hnorm, hz0, norm_zero]
    have hcos : cos θ = a / A := by
      rw [Complex.cos_arg hz0, hnorm]
    have hsin : sin θ = b / A := by
      rw [Complex.sin_arg, hnorm]
    have hApos : 0 < A := lt_of_le_of_ne (Real.sqrt_nonneg _) (Ne.symm h0)
    set φ : ℝ := toIcoMod Real.two_pi_pos 0 θ
    have hφ : φ ∈ Set.Ico 0 (2 * π) := by
      have := toIcoMod_mem_Ico Real.two_pi_pos 0 θ
      simpa using this
    obtain ⟨k, hk⟩ : ∃ k : ℤ, φ = θ - k • (2 * π) :=
      ⟨toIcoDiv Real.two_pi_pos 0 θ, rfl⟩
    have hcosφ : cos φ = a / A := by
      rw [hk, zsmul_eq_mul, show θ - (k : ℝ) * (2 * π) = θ - (k : ℤ) * (2 * π) by rfl,
        Real.cos_sub_int_mul_two_pi, hcos]
    have hsinφ : sin φ = b / A := by
      rw [hk, zsmul_eq_mul, Real.sin_sub_int_mul_two_pi, hsin]
    refine ⟨A, φ, hφ, fun x ↦ ?_⟩
    rw [sin_add, hcosφ, hsinφ]
    field_simp

/-- The log-log feature map `φ x = (1, log x)`. -/
noncomputable def phiLogLog (x : ℝ) : Fin 2 → ℝ := ![1, log x]

/-- The log-y feature map `φ x = (1, x)`. -/
def phiAffine (x : ℝ) : Fin 2 → ℝ := ![1, x]

/-- **Lemma 6 (Power-law instance).** With `ψ = log`, the weight `(log a, β)` realises the
power law `a x ^ β` through `(1, log x)` on all of `ℝ_{>0}`. -/
theorem powerLaw_realisable {a : ℝ} (ha : 0 < a) (β : ℝ) {x : ℝ} (hx : 0 < x) :
    ![log a, β] ⬝ᵥ phiLogLog x = log (a * x ^ β) := by
  simp only [phiLogLog, dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero,
    Matrix.cons_val_one, mul_one]
  rw [Real.log_mul ha.ne' (Real.rpow_pos_of_pos hx β).ne', Real.log_rpow hx]

/-- Lemma 6, risk form: the committed predictor `exp ∘ g⋆ ∘ φ` has zero risk on every test
distribution supported on `ℝ_{>0}`. -/
theorem powerLaw_epsOOD_eq_zero {a : ℝ} (ha : 0 < a) (β : ℝ) (Dtest : Measure ℝ)
    (hsupp : ∀ᵐ x ∂Dtest, 0 < x) :
    epsOOD Dtest (fun x ↦ a * x ^ β) (fun x ↦ exp (![log a, β] ⬝ᵥ phiLogLog x)) = 0 := by
  apply epsOOD_eq_zero_of_ae_eq
  filter_upwards [hsupp] with x hx
  rw [powerLaw_realisable ha β hx, Real.exp_log (mul_pos ha (Real.rpow_pos_of_pos hx β))]

/-- **Lemma 7 (Exponential instance).** With `ψ = log`, the weight `(log a, β)` realises the
exponential `a e ^ (β x)` through `(1, x)` on all of `ℝ`. -/
theorem exponential_realisable {a : ℝ} (ha : 0 < a) (β x : ℝ) :
    ![log a, β] ⬝ᵥ phiAffine x = log (a * exp (β * x)) := by
  simp only [phiAffine, dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero,
    Matrix.cons_val_one, mul_one]
  rw [Real.log_mul ha.ne' (Real.exp_pos _).ne', Real.log_exp]

/-- Lemma 7, risk form, on every test distribution on `ℝ`. -/
theorem exponential_epsOOD_eq_zero {a : ℝ} (ha : 0 < a) (β : ℝ) (Dtest : Measure ℝ) :
    epsOOD Dtest (fun x ↦ a * exp (β * x)) (fun x ↦ exp (![log a, β] ⬝ᵥ phiAffine x)) = 0 := by
  apply epsOOD_eq_zero_of_ae_eq
  refine Filter.Eventually.of_forall fun x ↦ ?_
  rw [exponential_realisable ha β x, Real.exp_log (mul_pos ha (Real.exp_pos _))]

/-- **Remark (Reparameterisation between power-law and exponential).** On `ℝ_{>0}` the power
law is the exponential instance read in the coordinate `log x`, so the two lemmas differ only
in the feature map. -/
theorem powerLaw_eq_exponential_comp_log {a : ℝ} (β : ℝ) {x : ℝ} (hx : 0 < x) :
    a * x ^ β = a * exp (β * log x) ∧ phiLogLog x = phiAffine (log x) := by
  refine ⟨?_, rfl⟩
  rw [Real.rpow_def_of_pos hx, mul_comm (log x) β]

/-! ### Remark (Noise model under `ψ = log`) -/

/-- Gaussian log-likelihood of residuals `r i` at variance `v`. -/
noncomputable def gaussLogLik {n : ℕ} (v : NNReal) (r : Fin n → ℝ) : ℝ :=
  ∑ i, log (gaussianPDFReal 0 v (r i))

/-- **Remark (Noise model under `ψ = log`).** Under Gaussian noise on the transformed target,
the log-likelihood is an affine, strictly decreasing function of the residual sum of squares,
so maximising the likelihood and minimising the squared residual pick the same weights. This is
why ordinary least squares in canonical coordinates is the maximum-likelihood estimator exactly
under multiplicative log-normal noise. -/
theorem gaussLogLik_eq {n : ℕ} {v : NNReal} (hv : v ≠ 0) (r : Fin n → ℝ) :
    gaussLogLik v r = -(n : ℝ) * log √(2 * π * v) - (∑ i, r i ^ 2) / (2 * v) := by
  have hvpos : (0 : ℝ) < v := by positivity
  have hs : 0 < √(2 * π * v) := Real.sqrt_pos.2 (by positivity)
  unfold gaussLogLik
  have : ∀ i, log (gaussianPDFReal 0 v (r i)) = -log √(2 * π * v) - r i ^ 2 / (2 * v) := by
    intro i
    rw [gaussianPDFReal_def, Real.log_mul (inv_pos.2 hs).ne' (Real.exp_pos _).ne',
      Real.log_inv, Real.log_exp]
    ring
  simp_rw [this, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, ← Finset.sum_div]
  ring

theorem gaussLogLik_le_iff {n : ℕ} {v : NNReal} (hv : v ≠ 0) (r s : Fin n → ℝ) :
    gaussLogLik v r ≤ gaussLogLik v s ↔ ∑ i, s i ^ 2 ≤ ∑ i, r i ^ 2 := by
  have hvpos : (0 : ℝ) < 2 * v := by positivity
  rw [gaussLogLik_eq hv, gaussLogLik_eq hv, sub_le_sub_iff_left, div_le_div_iff_of_pos_right hvpos]

end DGPCommitment
