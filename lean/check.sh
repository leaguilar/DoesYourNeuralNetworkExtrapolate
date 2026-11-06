#!/usr/bin/env bash
# The gate for this library. A green `lake build` alone proves nothing,
# because a file with an open proof still builds, so this also refuses any
# open proof and reads the axioms of every headline result.
set -uo pipefail
export PATH="$HOME/.elan/bin:$PATH"
cd "$(dirname "$0")" || exit 1

fail=0

echo "== build =="
lake build || fail=1

echo "== no open proofs =="
if grep -rnw "sorry\|admit\|native_decide" DGPCommitment/ DGPCommitment.lean \
   || grep -rn "^axiom\|sorryAx" DGPCommitment/ DGPCommitment.lean; then
  echo "FAIL: an open proof or an added axiom is present"
  fail=1
else
  echo "ok: no open proof, no native_decide, no added axiom"
fi

echo "== axioms of every headline result =="
names=(
  # Definition 1 and Lemma 4
  realisable_predictor_eq canonical_epsOOD_eq_zero canonical_epsOOD_eq_zero_id
  exists_leftInverse_of_strictMono epsOOD_eq_zero_of_ae_eq
  # Lemmas 5, 6, 7 and their remarks
  sin_realisable sin_epsOOD_eq_zero phase_amplitude_realisable
  span_sin_cos_eq_phase_amplitude powerLaw_realisable powerLaw_epsOOD_eq_zero
  exponential_realisable exponential_epsOOD_eq_zero powerLaw_eq_exponential_comp_log
  gaussLogLik_eq gaussLogLik_le_iff
  # Corollary 8
  isOLS_iff design_null_of_slices sin_rowDet_ae affine_rowDet_ae logLog_rowDet_ae
  sin_ols_recovery exponential_ols_recovery powerLaw_ols_recovery
  sin_ols_epsOOD exponential_ols_epsOOD powerLaw_ols_epsOOD
  # Proposition 1
  erm_blind erm_blind_min erm_blind_deterministic min_le_half_add
  jensen_design integral_designGap
  linear_form linear_form_eq_one_iff linear_constant_bounds pinsker_shape
  pinsker_illustration_vacuous
  # the Gaussian two-point problem (Step 2 of Proposition 1, Proposition 2(b))
  lintegral_fin_prod_eq_prod pi_withDensity gaussVec_eq_withDensity
  map_linear_gaussVec mem_lrRegion_iff np_le_lr
  lr_success_one lr_success_two lr_success_add twoPoint_le twoPoint_lr_attains
  # Proposition 2(a), 2(c), 2(d) and Corollary 3
  prop2a prop2c prop2c_eq rampGap_obsEquiv prop2d prop2d_band_iff
  prop2d_conditions_always_jointly_satisfiable Phi_surj Phi_PhiInv
  geometric_floor_norm geometric_floor geometric_floor_tight small_u_reading_fails
  # the strip remarks
  strip_sqNorm_ge_rpow strip_sqNorm_ge strip_norm_ge equispaced_empirical_ge
  uniformSqNorm_eq_integral_cond
  # the normal distribution function
  Phi_le_tangent concaveOn_Phi concaveOn_Phi_sqrt strictMono_Phi Phi_neg Phi_zero
)
tmp="$(mktemp -d)"
{
  echo "import DGPCommitment"
  for n in "${names[@]}"; do echo "#print axioms DGPCommitment.$n"; done
} > "$tmp/Ax.lean"
out="$(lake env lean "$tmp/Ax.lean" 2>&1)"
echo "$out"
if echo "$out" | grep -q "error"; then
  echo "FAIL: the axiom check did not run cleanly"
  fail=1
fi
bad="$(echo "$out" | grep -o "\[[^]]*\]" | tr -d '[] ' | tr ',' '\n' \
  | grep -v -x -e propext -e Classical.choice -e Quot.sound -e '' || true)"
if [ -n "$bad" ]; then
  echo "FAIL: a result rests on something beyond propext, Classical.choice, Quot.sound:"
  echo "$bad" | sort -u
  fail=1
fi
count="$(echo "$out" | grep -c "depends on axioms\|does not depend on any axioms")"
echo "checked $count of ${#names[@]} results"
[ "$count" -eq "${#names[@]}" ] || fail=1
rm -rf "$tmp"

echo
if [ "$fail" -eq 0 ]; then echo "ALL CHECKS PASSED"; else echo "CHECKS FAILED"; fi
exit "$fail"
