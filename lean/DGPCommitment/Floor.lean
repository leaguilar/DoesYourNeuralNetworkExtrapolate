import DGPCommitment.Basic

/-!
# The geometric floor

Proposition 2(c), Corollary 3 with its tightness construction, the remark on
divergence faster than linear, and the remark on the equally spaced design.

The strip is `V = [b, b + δ]` and `D_V` is the uniform distribution on it. Its
squared `L²` norm is `uniformSqNorm Δ b δ = (1/δ) ∫₀^δ Δ(b+u)² du`;
`uniformSqNorm_eq_integral_cond` checks that this is the norm under Mathlib's
own uniform (conditioned Lebesgue) measure.
-/

open MeasureTheory ProbabilityTheory Real Set

namespace DGPCommitment

/-- Squared `L²` norm of `Δ` under the uniform distribution on `[b, b + δ]`. -/
noncomputable def uniformSqNorm (Δ : ℝ → ℝ) (b δ : ℝ) : ℝ :=
  δ⁻¹ * ∫ u in (0 : ℝ)..δ, Δ (b + u) ^ 2

/-- Fidelity: `uniformSqNorm` is the second moment of `Δ` under the uniform probability
measure on `[b, b + δ]`, written as Lebesgue measure conditioned on the strip. -/
theorem uniformSqNorm_eq_integral_cond (Δ : ℝ → ℝ) (b : ℝ) {δ : ℝ} (hδ : 0 < δ) :
    ∫ x, Δ x ^ 2 ∂(volume[|Icc b (b + δ)]) = uniformSqNorm Δ b δ := by
  have hshift := intervalIntegral.integral_comp_add_left (fun x ↦ Δ x ^ 2) b (a := 0) (b := δ)
  simp only [add_zero] at hshift
  rw [ProbabilityTheory.cond, integral_smul_measure, Real.volume_Icc,
    show b + δ - b = δ by ring, integral_Icc_eq_integral_Ioc,
    ← intervalIntegral.integral_of_le (by linarith : b ≤ b + δ), ← hshift]
  simp only [uniformSqNorm]
  rw [ENNReal.toReal_inv, ENNReal.toReal_ofReal hδ.le, smul_eq_mul]

/-! ### Divergence at a polynomial rate -/

/-- **Remark (Beyond linear divergence).** If `Δ` leaves the window at least as fast as
`r u ^ p` on the whole strip, its squared norm under the uniform strip distribution is at
least `r² δ^(2p) / (2p + 1)`. -/
theorem strip_sqNorm_ge_rpow {Δ : ℝ → ℝ} {b δ r p : ℝ} (hδ : 0 < δ) (hr : 0 ≤ r) (hp : 0 < p)
    (hdiv : ∀ u ∈ Ioc 0 δ, r * u ^ p ≤ |Δ (b + u)|)
    (hint : IntervalIntegrable (fun u ↦ Δ (b + u) ^ 2) volume 0 δ) :
    r ^ 2 * δ ^ (2 * p) / (2 * p + 1) ≤ uniformSqNorm Δ b δ := by
  have hlow : ∀ u ∈ Icc (0 : ℝ) δ, r ^ 2 * u ^ (2 * p) ≤ Δ (b + u) ^ 2 := by
    intro u hu
    rcases eq_or_lt_of_le hu.1 with h0 | hpos
    · subst h0
      rw [Real.zero_rpow (by positivity)]
      simp only [mul_zero]
      positivity
    · have h1 := hdiv u ⟨hpos, hu.2⟩
      have hnn : 0 ≤ r * u ^ p := mul_nonneg hr (Real.rpow_nonneg hu.1 p)
      have h2 : (r * u ^ p) ^ 2 ≤ Δ (b + u) ^ 2 := by
        rw [← sq_abs (Δ (b + u))]
        exact pow_le_pow_left₀ hnn h1 2
      calc r ^ 2 * u ^ (2 * p) = (r * u ^ p) ^ 2 := by
            rw [mul_pow, ← Real.rpow_natCast (u ^ p) 2, ← Real.rpow_mul hu.1]
            norm_num [mul_comm]
        _ ≤ _ := h2
  have hrint : IntervalIntegrable (fun u : ℝ ↦ r ^ 2 * u ^ (2 * p)) volume 0 δ :=
    (intervalIntegral.intervalIntegrable_rpow' (by linarith)).const_mul _
  have hmono := intervalIntegral.integral_mono_on hδ.le hrint hint hlow
  have hcalc : ∫ u in (0 : ℝ)..δ, r ^ 2 * u ^ (2 * p) = r ^ 2 * (δ ^ (2 * p + 1) / (2 * p + 1)) := by
    rw [intervalIntegral.integral_const_mul, integral_rpow (Or.inl (by linarith))]
    rw [Real.zero_rpow (by linarith), sub_zero]
  rw [hcalc] at hmono
  unfold uniformSqNorm
  have hδinv : 0 < δ⁻¹ := inv_pos.2 hδ
  calc r ^ 2 * δ ^ (2 * p) / (2 * p + 1)
      = δ⁻¹ * (r ^ 2 * (δ ^ (2 * p + 1) / (2 * p + 1))) := by
        rw [Real.rpow_add hδ, Real.rpow_one]
        field_simp
    _ ≤ δ⁻¹ * ∫ u in (0 : ℝ)..δ, Δ (b + u) ^ 2 := by gcongr

/-- **Proposition 2(c), norm form.** Linear divergence at rate `r` on the whole strip forces
`‖Δ‖²_{L²(D_V)} ≥ r² δ² / 3`. -/
theorem strip_sqNorm_ge {Δ : ℝ → ℝ} {b δ r : ℝ} (hδ : 0 < δ) (hr : 0 ≤ r)
    (hdiv : ∀ u ∈ Ioc 0 δ, r * u ≤ |Δ (b + u)|)
    (hint : IntervalIntegrable (fun u ↦ Δ (b + u) ^ 2) volume 0 δ) :
    r ^ 2 * δ ^ 2 / 3 ≤ uniformSqNorm Δ b δ := by
  have h := strip_sqNorm_ge_rpow (p := 1) hδ hr one_pos
    (fun u hu ↦ by simpa [Real.rpow_one] using hdiv u hu) hint
  norm_num at h
  exact h

/-- `‖Δ‖_{L²(D_V)} ≥ r δ / √3`, the norm of Proposition 2(c) and Corollary 3. -/
theorem strip_norm_ge {Δ : ℝ → ℝ} {b δ r : ℝ} (hδ : 0 < δ) (hr : 0 ≤ r)
    (hdiv : ∀ u ∈ Ioc 0 δ, r * u ≤ |Δ (b + u)|)
    (hint : IntervalIntegrable (fun u ↦ Δ (b + u) ^ 2) volume 0 δ) :
    r * δ / √3 ≤ √(uniformSqNorm Δ b δ) := by
  have h := strip_sqNorm_ge hδ hr hdiv hint
  rw [show r * δ / √3 = √((r * δ) ^ 2 / 3) by
    rw [Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 3), Real.sqrt_sq (by positivity)]]
  exact Real.sqrt_le_sqrt (by nlinarith)

/-- The strip signal-to-noise ratio `t_V = √n_V ‖Δ‖_{L²(D_V)} / σ`. -/
noncomputable def tV (nV : ℕ) (sqNorm σ : ℝ) : ℝ := √nV * √sqNorm / σ

/-- The window signal-to-noise ratio `t_W = √n ε_W / σ`. -/
noncomputable def tW (n : ℕ) (ε σ : ℝ) : ℝ := √n * ε / σ

/-- **Proposition 2(c).** `t_V ≥ r δ √n_V / (√3 σ)`. -/
theorem prop2c {Δ : ℝ → ℝ} {b δ r σ : ℝ} (nV : ℕ) (hδ : 0 < δ) (hr : 0 ≤ r) (hσ : 0 < σ)
    (hdiv : ∀ u ∈ Ioc 0 δ, r * u ≤ |Δ (b + u)|)
    (hint : IntervalIntegrable (fun u ↦ Δ (b + u) ^ 2) volume 0 δ) :
    r * δ * √nV / (√3 * σ) ≤ tV nV (uniformSqNorm Δ b δ) σ := by
  have h := strip_norm_ge hδ hr hdiv hint
  unfold tV
  rw [show r * δ * √nV / (√3 * σ) = √nV * (r * δ / √3) / σ by
    field_simp]
  gcongr

/-- The extremal gap `max 0 (r (x - b))`. -/
noncomputable def rampGap (r b : ℝ) (x : ℝ) : ℝ := max 0 (r * (x - b))

theorem rampGap_shift {r b : ℝ} (hr : 0 ≤ r) {u : ℝ} (hu : 0 ≤ u) : rampGap r b (b + u) = r * u := by
  simp [rampGap, max_eq_right (mul_nonneg hr hu)]

theorem rampGap_sqNorm {r b δ : ℝ} (hδ : 0 < δ) (hr : 0 ≤ r) :
    uniformSqNorm (rampGap r b) b δ = r ^ 2 * δ ^ 2 / 3 := by
  unfold uniformSqNorm
  have : ∫ u in (0 : ℝ)..δ, rampGap r b (b + u) ^ 2 = ∫ u in (0 : ℝ)..δ, r ^ 2 * u ^ 2 := by
    apply intervalIntegral.integral_congr
    intro u hu
    rw [uIcc_of_le hδ.le] at hu
    simp only [rampGap_shift hr hu.1, mul_pow]
  rw [this, intervalIntegral.integral_const_mul, integral_pow]
  field_simp
  ring

/-- **Proposition 2(c), equality case.** The ramp `max 0 (r (x - b))` attains the bound. -/
theorem prop2c_eq {b δ r σ : ℝ} (nV : ℕ) (hδ : 0 < δ) (hr : 0 ≤ r) (hσ : 0 < σ) :
    tV nV (uniformSqNorm (rampGap r b) b δ) σ = r * δ * √nV / (√3 * σ) := by
  unfold tV
  rw [rampGap_sqNorm hδ hr, Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 3),
    show r ^ 2 * δ ^ 2 = (r * δ) ^ 2 by ring, Real.sqrt_sq (by positivity)]
  field_simp

/-- The ramp vanishes on the window `W ⊆ (-∞, b]`, so it is observationally equivalent to the
zero process for every tolerance. This is what lets the equality case also satisfy the
hypothesis of Proposition 2(a). -/
theorem rampGap_obsEquiv {X : Type*} [MeasurableSpace X] {r b : ℝ} (hr : 0 ≤ r) (e : X → ℝ)
    (D : Measure X) (hW : ∀ᵐ x ∂D, e x ≤ b) (ε : ℝ) :
    ObsEquiv D (fun x ↦ rampGap r b (e x)) (fun _ ↦ 0) ε := by
  have hz : (fun x ↦ (rampGap r b (e x) - 0) ^ 2) =ᵐ[D] fun _ ↦ (0 : ℝ) := by
    filter_upwards [hW] with x hx
    have h : r * (e x - b) ≤ 0 := mul_nonpos_of_nonneg_of_nonpos hr (by linarith)
    simp [rampGap, max_eq_left h]
  refine ⟨(integrable_congr hz).2 (integrable_zero _ _ _), ?_⟩
  rw [integral_congr_ae hz]
  simp only [integral_zero]
  positivity

/-! ### Corollary 3 -/

/-- **Corollary 3 (Geometric floor).** Past width `√3 ε_W / r` the strip norm reaches `ε_W`. -/
theorem geometric_floor_norm {Δ : ℝ → ℝ} {b δ r ε : ℝ} (hδ : 0 < δ) (hr : 0 < r)
    (hdiv : ∀ u ∈ Ioc 0 δ, r * u ≤ |Δ (b + u)|)
    (hint : IntervalIntegrable (fun u ↦ Δ (b + u) ^ 2) volume 0 δ)
    (hwidth : √3 * ε / r ≤ δ) :
    ε ≤ √(uniformSqNorm Δ b δ) := by
  have h := strip_norm_ge hδ hr.le hdiv hint
  have : ε ≤ r * δ / √3 := by
    rw [le_div_iff₀ (by positivity)]
    rw [div_le_iff₀ hr] at hwidth
    nlinarith
  linarith

/-- **Corollary 3, signal-to-noise form.** `t_V ≥ t_W √(n_V / n)`. -/
theorem geometric_floor {Δ : ℝ → ℝ} {b δ r ε σ : ℝ} {n nV : ℕ} (hn : 0 < n)
    (hδ : 0 < δ) (hr : 0 < r) (hσ : 0 < σ)
    (hdiv : ∀ u ∈ Ioc 0 δ, r * u ≤ |Δ (b + u)|)
    (hint : IntervalIntegrable (fun u ↦ Δ (b + u) ^ 2) volume 0 δ)
    (hwidth : √3 * ε / r ≤ δ) :
    tW n ε σ * √(nV / n) ≤ tV nV (uniformSqNorm Δ b δ) σ := by
  have h := geometric_floor_norm hδ hr hdiv hint hwidth
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  unfold tW tV
  rw [Real.sqrt_div' _ hnpos.le]
  have hsn : 0 < √(n : ℝ) := Real.sqrt_pos.2 hnpos
  rw [show √(n : ℝ) * ε / σ * (√(nV : ℝ) / √(n : ℝ)) = √(nV : ℝ) * ε / σ by
    field_simp]
  gcongr

/-- **Corollary 3, tightness.** For the ramp, the strip norm is exactly `r δ / √3`, so any
width strictly below `√3 ε / r` leaves the strip norm strictly below `ε`. -/
theorem geometric_floor_tight {b δ r ε : ℝ} (hδ : 0 < δ) (hr : 0 < r)
    (hnarrow : δ < √3 * ε / r) :
    √(uniformSqNorm (rampGap r b) b δ) < ε := by
  rw [rampGap_sqNorm hδ hr.le, Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 3),
    show r ^ 2 * δ ^ 2 = (r * δ) ^ 2 by ring, Real.sqrt_sq (by positivity),
    div_lt_iff₀ (by positivity)]
  rw [lt_div_iff₀ hr] at hnarrow
  nlinarith

/-- **A local divergence hypothesis is not enough.** Corollary 3 needs `|Δ(b + u)| ≥ r u` on
the whole strip. Here the gap grows at rate `1` on `(0, 1)`, so it diverges at rate `1` near the
boundary, yet on the strip of width `2` its squared norm is `1/6`, below the bound's
`r² δ² / 3 = 4/3`. -/
theorem small_u_reading_fails :
    ∃ (Δ : ℝ → ℝ) (u₀ : ℝ), 0 < u₀ ∧ (∀ u ∈ Ioo 0 u₀, 1 * u ≤ |Δ (0 + u)|) ∧
      IntervalIntegrable (fun u ↦ Δ (0 + u) ^ 2) volume 0 2 ∧
      uniformSqNorm Δ 0 2 < 1 ^ 2 * 2 ^ 2 / 3 := by
  set Δ : ℝ → ℝ := fun u ↦ if u ≤ 1 then u else 0 with hΔ
  have h1 : IntervalIntegrable (fun u ↦ Δ (0 + u) ^ 2) volume 0 1 := by
    refine ((continuous_pow 2).intervalIntegrable 0 1).congr_uIoo fun u hu ↦ ?_
    rw [uIoo_of_le zero_le_one] at hu
    simp [hΔ, hu.2.le]
  have h2 : IntervalIntegrable (fun u ↦ Δ (0 + u) ^ 2) volume 1 2 := by
    refine (intervalIntegrable_const (c := (0 : ℝ))).congr_uIoo fun u hu ↦ ?_
    rw [uIoo_of_le (by norm_num : (1 : ℝ) ≤ 2)] at hu
    simp [hΔ, not_le.2 hu.1]
  have hsplit : ∫ u in (0 : ℝ)..2, Δ (0 + u) ^ 2 = 1 / 3 := by
    rw [← intervalIntegral.integral_add_adjacent_intervals h1 h2]
    have e1 : ∫ u in (0 : ℝ)..1, Δ (0 + u) ^ 2 = ∫ u in (0 : ℝ)..1, u ^ 2 := by
      apply intervalIntegral.integral_congr_ae
      filter_upwards with u hu
      rw [uIoc_of_le zero_le_one] at hu
      simp [hΔ, hu.2]
    have e2 : ∫ u in (1 : ℝ)..2, Δ (0 + u) ^ 2 = ∫ _ in (1 : ℝ)..2, (0 : ℝ) := by
      apply intervalIntegral.integral_congr_ae
      filter_upwards with u hu
      rw [uIoc_of_le (by norm_num : (1 : ℝ) ≤ 2)] at hu
      simp [hΔ, not_le.2 hu.1]
    rw [e1, e2, integral_pow]
    norm_num
  refine ⟨Δ, 1, one_pos, fun u hu ↦ ?_, h1.trans h2, ?_⟩
  · simp [hΔ, hu.2.le, abs_of_pos hu.1]
  · unfold uniformSqNorm
    rw [hsplit]
    norm_num

/-! ### Remark (Which measure the subscript names) -/

theorem sum_range_succ_sq (m : ℕ) :
    6 * ∑ i ∈ Finset.range m, ((i : ℝ) + 1) ^ 2 = (m : ℝ) * ((m : ℝ) + 1) * (2 * (m : ℝ) + 1) := by
  induction m with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ, mul_add, ih]
    push_cast
    ring

/-- **Remark (Which measure the subscript names).** For the equally spaced design
`v_i = b + i δ / m`, `i = 1, …, m`, the empirical squared norm already dominates the uniform
bound `r² δ² / 3`, so Proposition 2(c) holds for that design verbatim. -/
theorem equispaced_empirical_ge {Δ : ℝ → ℝ} {b δ r : ℝ} {m : ℕ} (hm : 0 < m) (hδ : 0 < δ)
    (hr : 0 ≤ r) (hdiv : ∀ u ∈ Ioc 0 δ, r * u ≤ |Δ (b + u)|) :
    r ^ 2 * δ ^ 2 / 3 ≤ (m : ℝ)⁻¹ * ∑ i ∈ Finset.range m, Δ (b + (i + 1) * δ / m) ^ 2 := by
  have hmr : (0 : ℝ) < m := by exact_mod_cast hm
  have hpt : ∀ i ∈ Finset.range m,
      r ^ 2 * δ ^ 2 / (m : ℝ) ^ 2 * ((i : ℝ) + 1) ^ 2 ≤ Δ (b + (i + 1) * δ / m) ^ 2 := by
    intro i hi
    have hi' : (i : ℝ) + 1 ≤ m := by
      have := Finset.mem_range.1 hi
      exact_mod_cast this
    have hu : (i + 1) * δ / m ∈ Ioc (0 : ℝ) δ := by
      refine ⟨by positivity, ?_⟩
      rw [div_le_iff₀ hmr]
      nlinarith
    have h1 := hdiv _ hu
    have hnn : 0 ≤ r * ((i + 1) * δ / m) := by positivity
    have h2 := pow_le_pow_left₀ hnn h1 2
    rw [sq_abs] at h2
    calc r ^ 2 * δ ^ 2 / (m : ℝ) ^ 2 * ((i : ℝ) + 1) ^ 2 = (r * ((i + 1) * δ / m)) ^ 2 := by
          field_simp
      _ ≤ _ := h2
  have hsum := Finset.sum_le_sum hpt
  rw [← Finset.mul_sum] at hsum
  set S : ℝ := ∑ i ∈ Finset.range m, ((i : ℝ) + 1) ^ 2 with hSdef
  have hclosed : 6 * S = (m : ℝ) * ((m : ℝ) + 1) * (2 * (m : ℝ) + 1) := sum_range_succ_sq m
  have hlow : 2 * (m : ℝ) ^ 3 ≤ 6 * S := by rw [hclosed]; nlinarith
  have hrd : 0 ≤ r ^ 2 * δ ^ 2 := by positivity
  have key : r ^ 2 * δ ^ 2 / 3 ≤ (m : ℝ)⁻¹ * (r ^ 2 * δ ^ 2 / (m : ℝ) ^ 2 * S) := by
    rw [show (m : ℝ)⁻¹ * (r ^ 2 * δ ^ 2 / (m : ℝ) ^ 2 * S) = r ^ 2 * δ ^ 2 * (S / (m : ℝ) ^ 3) by
      field_simp]
    have : (1 : ℝ) / 3 ≤ S / (m : ℝ) ^ 3 := by
      rw [div_le_div_iff₀ (by norm_num) (by positivity)]
      linarith
    calc r ^ 2 * δ ^ 2 / 3 = r ^ 2 * δ ^ 2 * (1 / 3) := by ring
      _ ≤ _ := mul_le_mul_of_nonneg_left this hrd
  calc r ^ 2 * δ ^ 2 / 3 ≤ _ := key
    _ ≤ _ := by gcongr

end DGPCommitment
