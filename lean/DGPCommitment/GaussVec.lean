import DGPCommitment.Phi

/-!
# Gaussian vectors with independent coordinates

The law of `Y = μ + η` with `η` i.i.d. `N(0, σ²)` on `Fin n → ℝ`, its density,
and the law of a linear statistic of it. These are the ingredients of the
two-point problem behind Propositions 1 and 2.
-/

open MeasureTheory ProbabilityTheory Real Set
open scoped ENNReal NNReal

namespace DGPCommitment

/-! ### Fubini for products, and the density of a product measure -/

/-- Tonelli in `n` variables for a product of nonnegative functions of the coordinates. -/
theorem lintegral_fin_prod_eq_prod {n : ℕ} {μ : Fin n → Measure ℝ} [∀ i, SigmaFinite (μ i)]
    (f : Fin n → ℝ → ℝ≥0∞) (hf : ∀ i, Measurable (f i)) :
    ∫⁻ x : Fin n → ℝ, ∏ i, f i (x i) ∂(Measure.pi μ) = ∏ i, ∫⁻ x, f i x ∂(μ i) := by
  induction n with
  | zero => simp
  | succ n ih =>
    calc
      _ = ∫⁻ x : ℝ × (Fin n → ℝ), f 0 x.1 * ∏ i : Fin n, f i.succ (x.2 i)
            ∂((μ 0).prod (Measure.pi fun i ↦ μ i.succ)) := by
        have hg := (measurePreserving_piFinSuccAbove (fun i ↦ μ i) 0).symm
        refine (hg.lintegral_comp_emb (MeasurableEquiv.measurableEmbedding _)
          (fun x : Fin (n + 1) → ℝ ↦ ∏ i, f i (x i))).symm.trans ?_
        congr 1
        ext x
        simp [MeasurableEquiv.piFinSuccAbove_symm_apply, Fin.insertNthEquiv,
          Fin.prod_univ_succ]
      _ = (∫⁻ x, f 0 x ∂μ 0) * ∏ i : Fin n, ∫⁻ x, f i.succ x ∂(μ i.succ) := by
        rw [← ih (fun i ↦ f i.succ) (fun i ↦ hf _)]
        exact lintegral_prod_mul (hf 0).aemeasurable
          (Finset.measurable_prod _ fun i _ ↦ (hf _).comp (measurable_pi_apply i)).aemeasurable
      _ = _ := by rw [Fin.prod_univ_succ]

/-- A product of measures with densities has the product density. -/
theorem pi_withDensity {n : ℕ} (f : Fin n → ℝ → ℝ≥0∞) (hf : ∀ i, Measurable (f i))
    (hfin : ∀ i x, f i x ≠ ∞) :
    Measure.pi (fun i ↦ (volume : Measure ℝ).withDensity (f i)) =
      (volume : Measure (Fin n → ℝ)).withDensity (fun y ↦ ∏ i, f i (y i)) := by
  have : ∀ i, SigmaFinite ((volume : Measure ℝ).withDensity (f i)) := fun i ↦
    SigmaFinite.withDensity_of_ne_top (Filter.Eventually.of_forall (hfin i))
  refine Measure.pi_eq fun s hs ↦ ?_
  have hbox : MeasurableSet (univ.pi s) := MeasurableSet.univ_pi hs
  rw [withDensity_apply _ hbox, volume_pi, ← lintegral_indicator hbox]
  have hind : (univ.pi s).indicator (fun y ↦ ∏ i, f i (y i)) =
      fun y ↦ ∏ i, (s i).indicator (f i) (y i) := by
    ext y
    by_cases hy : y ∈ univ.pi s
    · rw [indicator_of_mem hy]
      exact Finset.prod_congr rfl fun i _ ↦ (indicator_of_mem (hy i (mem_univ i)) _).symm
    · rw [indicator_of_notMem hy]
      obtain ⟨i, hi⟩ : ∃ i, y i ∉ s i := by simpa [Set.mem_pi] using hy
      exact (Finset.prod_eq_zero (Finset.mem_univ i) (indicator_of_notMem hi _)).symm
  rw [hind, lintegral_fin_prod_eq_prod _ fun i ↦ (hf i).indicator (hs i)]
  refine Finset.prod_congr rfl fun i _ ↦ ?_
  rw [withDensity_apply _ (hs i), lintegral_indicator (hs i)]

/-! ### The Gaussian vector -/

/-- The variance `σ²` as a nonnegative real. -/
def varOf (σ : ℝ) : ℝ≥0 := ⟨σ ^ 2, sq_nonneg σ⟩

theorem varOf_ne_zero {σ : ℝ} (hσ : σ ≠ 0) : varOf σ ≠ 0 := by
  intro h
  apply hσ
  have : σ ^ 2 = 0 := congrArg NNReal.toReal h
  exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 this

@[simp] theorem coe_varOf (σ : ℝ) : (varOf σ : ℝ) = σ ^ 2 := rfl

@[simp] theorem varOf_one : varOf 1 = 1 :=
  NNReal.eq (show (1 : ℝ) ^ 2 = 1 by norm_num)

theorem varOf_mul (a b : ℝ) : varOf (a * b) = ⟨a ^ 2, sq_nonneg a⟩ * varOf b :=
  NNReal.eq (show (a * b) ^ 2 = a ^ 2 * b ^ 2 by ring)

/-- The law of `Y = μ + η` with `η` i.i.d. `N(0, σ²)`. -/
noncomputable def gaussVec {n : ℕ} (μ : Fin n → ℝ) (σ : ℝ) : Measure (Fin n → ℝ) :=
  Measure.pi fun i ↦ gaussianReal (μ i) (varOf σ)

instance {n : ℕ} (μ : Fin n → ℝ) (σ : ℝ) : IsProbabilityMeasure (gaussVec μ σ) := by
  unfold gaussVec
  infer_instance

/-- The density of the Gaussian vector (with respect to Lebesgue measure on `Fin n → ℝ`). -/
noncomputable def gaussVecPDF {n : ℕ} (μ : Fin n → ℝ) (σ : ℝ) (y : Fin n → ℝ) : ℝ≥0∞ :=
  ∏ i, gaussianPDF (μ i) (varOf σ) (y i)

theorem gaussVec_eq_withDensity {n : ℕ} (μ : Fin n → ℝ) {σ : ℝ} (hσ : σ ≠ 0) :
    gaussVec μ σ = volume.withDensity (gaussVecPDF μ σ) := by
  unfold gaussVec gaussVecPDF
  simp_rw [gaussianReal_of_var_ne_zero _ (varOf_ne_zero hσ)]
  exact pi_withDensity _ (fun i ↦ measurable_gaussianPDF _ _) fun i x ↦ gaussianPDF_ne_top

theorem measurable_gaussVecPDF {n : ℕ} (μ : Fin n → ℝ) (σ : ℝ) :
    Measurable (gaussVecPDF μ σ) := by
  unfold gaussVecPDF
  exact Finset.measurable_prod _ fun i _ ↦ (measurable_gaussianPDF _ _).comp (measurable_pi_apply i)

/-- `Y = μ + η`: the Gaussian vector is the noise vector shifted by its mean. -/
theorem gaussVec_eq_map_add {n : ℕ} (μ : Fin n → ℝ) (σ : ℝ) :
    gaussVec μ σ = (gaussVec 0 σ).map (fun z ↦ μ + z) := by
  have h := Measure.pi_map_pi (μ := fun _ : Fin n ↦ gaussianReal 0 (varOf σ))
    (f := fun i (z : ℝ) ↦ μ i + z) (fun i ↦ (measurable_const_add (μ i)).aemeasurable)
  simp only [gaussianReal_map_const_add, zero_add] at h
  unfold gaussVec
  simp only [Pi.zero_apply]
  rw [← h]
  rfl

/-- The noise vector at standard deviation `σ` is `σ` times a standard one. -/
theorem gaussVec_zero_eq_map_smul {n : ℕ} (σ : ℝ) :
    gaussVec (0 : Fin n → ℝ) σ = (gaussVec 0 1).map (fun z ↦ σ • z) := by
  have h := Measure.pi_map_pi (μ := fun _ : Fin n ↦ gaussianReal 0 (varOf 1))
    (f := fun _ (z : ℝ) ↦ σ * z) (fun i ↦ (measurable_const_mul σ).aemeasurable)
  simp only [gaussianReal_map_const_mul, mul_zero] at h
  unfold gaussVec
  simp only [Pi.zero_apply]
  convert h.symm using 3
  rw [varOf_one, mul_one]
  rfl

/-! ### The law of a linear statistic -/

/-- For a standard Gaussian vector `Z`, `∑ dᵢ Zᵢ ~ N(0, ‖d‖²)`. -/
theorem map_linear_gaussVec_std {n : ℕ} (d : Fin n → ℝ) :
    (gaussVec (0 : Fin n → ℝ) 1).map (fun z ↦ ∑ i, d i * z i) =
      gaussianReal 0 (varOf √(∑ i, d i ^ 2)) := by
  have hstd : gaussVec (0 : Fin n → ℝ) 1 = Measure.pi (fun _ : Fin n ↦ gaussianReal 0 1) := by
    unfold gaussVec
    simp
  set L : StrongDual ℝ (EuclideanSpace ℝ (Fin n)) := innerSL ℝ (WithLp.toLp 2 d)
  have hcomp : (fun z : Fin n → ℝ ↦ ∑ i, d i * z i) = L ∘ WithLp.toLp 2 := by
    ext z
    simp [L, PiLp.inner_apply, mul_comm]
  rw [hstd, hcomp, ← Measure.map_map L.continuous.measurable (by fun_prop),
    map_pi_eq_stdGaussian, IsGaussian.map_eq_gaussianReal L, integral_strongDual_stdGaussian,
    variance_dual_stdGaussian]
  congr 1
  ext
  simp only [coe_varOf, L, innerSL_apply_norm]
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity), Real.sq_sqrt (by positivity),
    Real.toNNReal_of_nonneg (by positivity)]
  simp

/-- For the noise vector at standard deviation `σ`, `∑ dᵢ ηᵢ ~ N(0, σ² ‖d‖²)`. -/
theorem map_linear_gaussVec {n : ℕ} (d : Fin n → ℝ) (σ : ℝ) :
    (gaussVec (0 : Fin n → ℝ) σ).map (fun z ↦ ∑ i, d i * z i) =
      gaussianReal 0 (varOf (σ * √(∑ i, d i ^ 2))) := by
  rw [gaussVec_zero_eq_map_smul, Measure.map_map (by fun_prop) (by fun_prop)]
  have : (fun z : Fin n → ℝ ↦ ∑ i, d i * z i) ∘ (fun z ↦ σ • z) =
      (fun t ↦ σ * t) ∘ (fun z ↦ ∑ i, d i * z i) := by
    ext z
    simp [Finset.mul_sum, mul_left_comm]
  rw [this, ← Measure.map_map (by fun_prop) (by fun_prop), map_linear_gaussVec_std,
    gaussianReal_map_const_mul, mul_zero]
  congr 1
  exact (varOf_mul σ _).symm

end DGPCommitment
