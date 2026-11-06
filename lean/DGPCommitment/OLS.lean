import DGPCommitment.Canonical
import DGPCommitment.ERM

/-!
# Corollary 8: ordinary least squares recovers the canonical weight

Ordinary least squares is modelled as its defining property: a weight is an
OLS fit when no other weight has a smaller residual sum of squares. No matrix
inverse is involved. Under realisability the canonical weight `g⋆` has zero
residual, so it is a fit, and it is the only fit as soon as the first two
feature rows are linearly independent. For an absolutely continuous design
distribution those two rows are independent almost surely, for each of the
three instances.
-/

open MeasureTheory ProbabilityTheory Real Set Matrix
open scoped ENNReal

namespace DGPCommitment

/-- Residual sum of squares of the weight `w` on features `φ(xᵢ)` and targets `zᵢ`. -/
noncomputable def rss {n : ℕ} (φ : ℝ → Fin 2 → ℝ) (x z : Fin n → ℝ) (w : Fin 2 → ℝ) : ℝ :=
  ∑ i, (z i - w ⬝ᵥ φ (x i)) ^ 2

/-- `w` is an ordinary-least-squares fit. -/
def IsOLS {n : ℕ} (φ : ℝ → Fin 2 → ℝ) (x z : Fin n → ℝ) (w : Fin 2 → ℝ) : Prop :=
  ∀ w', rss φ x z w ≤ rss φ x z w'

/-- The determinant of the two feature rows `φ a` and `φ b`. -/
def rowDet (φ : ℝ → Fin 2 → ℝ) (a b : ℝ) : ℝ := φ a 0 * φ b 1 - φ a 1 * φ b 0

theorem rss_nonneg {n : ℕ} (φ : ℝ → Fin 2 → ℝ) (x z : Fin n → ℝ) (w : Fin 2 → ℝ) :
    0 ≤ rss φ x z w :=
  Finset.sum_nonneg fun _ _ ↦ sq_nonneg _

/-- **Identifiability for OLS.** If the targets are realised by `g⋆` and the first two feature
rows are linearly independent, the OLS fits are exactly `g⋆`. -/
theorem isOLS_iff {m : ℕ} (φ : ℝ → Fin 2 → ℝ) (x z : Fin (m + 2) → ℝ) (gstar : Fin 2 → ℝ)
    (hz : ∀ i, z i = gstar ⬝ᵥ φ (x i)) (hdet : rowDet φ (x 0) (x 1) ≠ 0) (w : Fin 2 → ℝ) :
    IsOLS φ x z w ↔ w = gstar := by
  have hzero : rss φ x z gstar = 0 := by
    unfold rss
    exact Finset.sum_eq_zero fun i _ ↦ by rw [hz i, sub_self]; ring
  constructor
  · intro h
    have hw : rss φ x z w = 0 := le_antisymm (hzero ▸ h gstar) (rss_nonneg φ x z w)
    have hres : ∀ i, z i - w ⬝ᵥ φ (x i) = 0 := by
      intro i
      have := (Finset.sum_eq_zero_iff_of_nonneg fun j _ ↦ sq_nonneg _).1 hw i
        (Finset.mem_univ i)
      exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 this
    have e0 := hres 0
    have e1 := hres 1
    rw [hz 0] at e0
    rw [hz 1] at e1
    simp only [dotProduct, Fin.sum_univ_two] at e0 e1
    unfold rowDet at hdet
    have u0 : (w 0 - gstar 0) * (φ (x 0) 0 * φ (x 1) 1 - φ (x 0) 1 * φ (x 1) 0) = 0 := by
      linear_combination (-φ (x 1) 1) * e0 + φ (x 0) 1 * e1
    have u1 : (w 1 - gstar 1) * (φ (x 0) 0 * φ (x 1) 1 - φ (x 0) 1 * φ (x 1) 0) = 0 := by
      linear_combination φ (x 1) 0 * e0 - φ (x 0) 0 * e1
    have h0 := sub_eq_zero.1 ((mul_eq_zero.1 u0).resolve_right hdet)
    have h1 := sub_eq_zero.1 ((mul_eq_zero.1 u1).resolve_right hdet)
    ext i
    fin_cases i
    · exact h0
    · exact h1
  · rintro rfl w'
    rw [hzero]
    exact rss_nonneg φ x z w'

/-! ### The collinearity event is null -/

/-- An event on the first two coordinates of an i.i.d. design is null as soon as each of its
slices is null. -/
theorem design_null_of_slices (D : Measure ℝ) [IsProbabilityMeasure D] (m : ℕ)
    (R : ℝ → ℝ → Prop) (hR : MeasurableSet {p : ℝ × ℝ | R p.1 p.2})
    (hslice : ∀ a, D {b | R a b} = 0) :
    design D (m + 2) {x | R (x 0) (x 1)} = 0 := by
  have hmp := measurePreserving_piFinSuccAbove (fun _ : Fin (m + 2) ↦ D) 1
  set e := MeasurableEquiv.piFinSuccAbove (fun _ : Fin (m + 2) ↦ ℝ) 1
  set S : Set (ℝ × (Fin (m + 1) → ℝ)) := {p | R (p.2 0) p.1}
  have hS : MeasurableSet S :=
    hR.preimage (f := fun p : ℝ × (Fin (m + 1) → ℝ) ↦ (p.2 0, p.1)) (by fun_prop)
  have h10 : (1 : Fin (m + 2)).succAbove 0 = 0 := Fin.succAbove_of_castSucc_lt _ _ (by simp)
  have hE : {x : Fin (m + 2) → ℝ | R (x 0) (x 1)} = e ⁻¹' S := by
    ext x
    simp [e, S, MeasurableEquiv.piFinSuccAbove_apply, Fin.insertNthEquiv, Fin.removeNth_apply,
      h10]
  rw [design, hE, hmp.measure_preimage hS.nullMeasurableSet, Measure.prod_apply_symm hS]
  simp [S, hslice]

/-- A distribution absolutely continuous with respect to Lebesgue measure gives countable sets
probability zero. -/
theorem countable_null {D : Measure ℝ} (hac : D ≪ volume) {s : Set ℝ} (hs : s.Countable) :
    D s = 0 :=
  hac (hs.measure_zero volume)

/-- Sinusoidal instance: the two feature rows are collinear only on a null event. -/
theorem sin_rowDet_ae (D : Measure ℝ) [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ) :
    ∀ᵐ x ∂(design D (m + 2)), rowDet phiSin (x 0) (x 1) ≠ 0 := by
  have hnull := design_null_of_slices D m (fun a b ↦ rowDet phiSin a b = 0)
    (measurableSet_eq_fun (by unfold rowDet phiSin; fun_prop) measurable_const)
    fun a ↦ by
      refine countable_null hac ((Set.countable_range fun k : ℤ ↦ a - k * π).mono ?_)
      intro b hb
      simp only [rowDet, phiSin, Matrix.cons_val_zero, Matrix.cons_val_one,
        mem_ofPred_eq] at hb
      have : sin (a - b) = 0 := by rw [sin_sub]; linarith
      obtain ⟨k, hk⟩ := Real.sin_eq_zero_iff.1 this
      exact ⟨k, by linarith⟩
  rw [ae_iff]
  simpa using hnull

/-- Exponential instance: the rows `(1, a)` and `(1, b)` are collinear only when `a = b`. -/
theorem affine_rowDet_ae (D : Measure ℝ) [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ) :
    ∀ᵐ x ∂(design D (m + 2)), rowDet phiAffine (x 0) (x 1) ≠ 0 := by
  have hnull := design_null_of_slices D m (fun a b ↦ rowDet phiAffine a b = 0)
    (measurableSet_eq_fun (by unfold rowDet phiAffine; fun_prop) measurable_const)
    fun a ↦ by
      refine countable_null hac ((Set.countable_singleton a).mono ?_)
      intro b hb
      simp only [rowDet, phiAffine, Matrix.cons_val_zero, Matrix.cons_val_one,
        mem_ofPred_eq] at hb
      simp only [mem_singleton_iff]
      linarith
  rw [ae_iff]
  simpa using hnull

/-- Power-law instance: the rows `(1, log a)` and `(1, log b)` are collinear only on a finite
set of `b` for each `a`. -/
theorem logLog_rowDet_ae (D : Measure ℝ) [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ) :
    ∀ᵐ x ∂(design D (m + 2)), rowDet phiLogLog (x 0) (x 1) ≠ 0 := by
  have hnull := design_null_of_slices D m (fun a b ↦ rowDet phiLogLog a b = 0)
    (measurableSet_eq_fun (by unfold rowDet phiLogLog; fun_prop) measurable_const)
    fun a ↦ by
      refine countable_null hac ((({a, -a, 0, 1, -1} : Set ℝ).toFinite.countable).mono ?_)
      intro b hb
      simp only [rowDet, phiLogLog, Matrix.cons_val_zero, Matrix.cons_val_one,
        mem_ofPred_eq, one_mul, mul_one] at hb
      have hlog : log b = log a := by linarith
      simp only [mem_insert_iff, mem_singleton_iff]
      by_cases hb0 : b = 0
      · right; right; left; exact hb0
      by_cases ha0 : a = 0
      · subst ha0
        rw [Real.log_zero, Real.log_eq_zero] at hlog
        rcases hlog with h | h | h
        · exact absurd h hb0
        · right; right; right; left; exact h
        · right; right; right; right; exact h
      · have := Real.log_injOn_pos (Set.mem_Ioi.2 (abs_pos.2 hb0)) (Set.mem_Ioi.2 (abs_pos.2 ha0))
          (by rw [Real.log_abs, Real.log_abs, hlog])
        rcases abs_eq_abs.1 this with h | h
        · left; exact h
        · right; left; exact h
  rw [ae_iff]
  simpa using hnull

/-! ### Corollary 8 -/

/-- **Corollary 8, sinusoidal instance.** For an absolutely continuous design and `n ≥ 2`
samples, almost surely the only OLS fit of `sin` on `(sin x, cos x)` is `(1, 0)`. -/
theorem sin_ols_recovery (D : Measure ℝ) [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ) :
    ∀ᵐ x ∂(design D (m + 2)), ∀ w,
      IsOLS phiSin x (fun i ↦ sin (x i)) w ↔ w = ![1, 0] := by
  filter_upwards [sin_rowDet_ae D hac m] with x hx w
  exact isOLS_iff phiSin x _ ![1, 0] (fun i ↦ (sin_realisable (x i)).symm) hx w

/-- **Corollary 8, exponential instance.** Almost surely the only OLS fit of `log (a e^(βx))`
on `(1, x)` is `(log a, β)`. -/
theorem exponential_ols_recovery {a : ℝ} (ha : 0 < a) (β : ℝ) (D : Measure ℝ)
    [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ) :
    ∀ᵐ x ∂(design D (m + 2)), ∀ w,
      IsOLS phiAffine x (fun i ↦ log (a * exp (β * x i))) w ↔ w = ![log a, β] := by
  filter_upwards [affine_rowDet_ae D hac m] with x hx w
  exact isOLS_iff phiAffine x _ _ (fun i ↦ (exponential_realisable ha β (x i)).symm) hx w

/-- **Corollary 8, power-law instance.** For a design supported on `ℝ_{>0}`, almost surely the
only OLS fit of `log (a x^β)` on `(1, log x)` is `(log a, β)`. -/
theorem powerLaw_ols_recovery {a : ℝ} (ha : 0 < a) (β : ℝ) (D : Measure ℝ)
    [IsProbabilityMeasure D] (hac : D ≪ volume) (hpos : ∀ᵐ t ∂D, 0 < t) (m : ℕ) :
    ∀ᵐ x ∂(design D (m + 2)), ∀ w,
      IsOLS phiLogLog x (fun i ↦ log (a * x i ^ β)) w ↔ w = ![log a, β] := by
  have hall : ∀ᵐ x ∂(design D (m + 2)), ∀ i, 0 < x i := by
    rw [ae_all_iff]
    intro i
    exact (measurePreserving_eval (fun _ : Fin (m + 2) ↦ D) i).quasiMeasurePreserving.ae hpos
  filter_upwards [logLog_rowDet_ae D hac m, hall] with x hx hxpos w
  exact isOLS_iff phiLogLog x _ _ (fun i ↦ (powerLaw_realisable ha β (hxpos i)).symm) hx w

/-- **Corollary 8, risk form.** Any estimator that returns an OLS fit recovers the canonical
weight almost surely, so the committed predictor has zero OOD risk almost surely, on every test
distribution. Stated for the sinusoidal instance. The other two follow the same pattern with
`exponential_ols_recovery`, `powerLaw_ols_recovery` and Lemmas 6 and 7. -/
theorem sin_ols_epsOOD (D : Measure ℝ) [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ)
    (ghat : (Fin (m + 2) → ℝ) → Fin 2 → ℝ)
    (hfit : ∀ x, IsOLS phiSin x (fun i ↦ sin (x i)) (ghat x)) (Dtest : Measure ℝ) :
    ∀ᵐ x ∂(design D (m + 2)), epsOOD Dtest sin (fun t ↦ ghat x ⬝ᵥ phiSin t) = 0 := by
  filter_upwards [sin_ols_recovery D hac m] with x hx
  rw [(hx (ghat x)).1 (hfit x)]
  exact sin_epsOOD_eq_zero Dtest

theorem exponential_ols_epsOOD {a : ℝ} (ha : 0 < a) (β : ℝ) (D : Measure ℝ)
    [IsProbabilityMeasure D] (hac : D ≪ volume) (m : ℕ)
    (ghat : (Fin (m + 2) → ℝ) → Fin 2 → ℝ)
    (hfit : ∀ x, IsOLS phiAffine x (fun i ↦ log (a * exp (β * x i))) (ghat x))
    (Dtest : Measure ℝ) :
    ∀ᵐ x ∂(design D (m + 2)),
      epsOOD Dtest (fun t ↦ a * exp (β * t)) (fun t ↦ exp (ghat x ⬝ᵥ phiAffine t)) = 0 := by
  filter_upwards [exponential_ols_recovery ha β D hac m] with x hx
  rw [(hx (ghat x)).1 (hfit x)]
  exact exponential_epsOOD_eq_zero ha β Dtest

theorem powerLaw_ols_epsOOD {a : ℝ} (ha : 0 < a) (β : ℝ) (D : Measure ℝ)
    [IsProbabilityMeasure D] (hac : D ≪ volume) (hpos : ∀ᵐ t ∂D, 0 < t) (m : ℕ)
    (ghat : (Fin (m + 2) → ℝ) → Fin 2 → ℝ)
    (hfit : ∀ x, IsOLS phiLogLog x (fun i ↦ log (a * x i ^ β)) (ghat x))
    (Dtest : Measure ℝ) (hsupp : ∀ᵐ t ∂Dtest, 0 < t) :
    ∀ᵐ x ∂(design D (m + 2)),
      epsOOD Dtest (fun t ↦ a * t ^ β) (fun t ↦ exp (ghat x ⬝ᵥ phiLogLog t)) = 0 := by
  filter_upwards [powerLaw_ols_recovery ha β D hac hpos m] with x hx
  rw [(hx (ghat x)).1 (hfit x)]
  exact powerLaw_epsOOD_eq_zero ha β Dtest hsupp

end DGPCommitment
