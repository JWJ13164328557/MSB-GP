"""Run release smoke tests without requiring pytest."""
from pathlib import Path
import sys
import importlib.util

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from test_smoke import test_basic_models_fit_and_predict, test_rsi_and_calibration
from test_reml_and_kernels import (test_reml_gradient_matches_finite_difference, test_kernel_state_training_and_cross_shapes, test_diagnostics_and_reliability, test_training_cross_kernel_consistency, test_shared_kernel_state_supports_model_and_calibration_variants, test_sparse_genotype_input_matches_dense, test_memmap_genotype_input_matches_dense, test_memmap_validation_rejects_bad_codes_and_nan, test_reml_multistart_diagnostics_and_objective, test_additive_only_msbgp_near_gblup, test_candidate_stack_prediction_uses_shared_cross_cache_path, test_native_msbcal_default_has_no_external_model_calls, test_native_epi_candidate_evaluated_on_auto_kernels)
from test_fixed_effects import test_gls_projection_and_optimizer_diagnostics, test_reliability_uses_gls_projection
from test_kernel_prior_gate import test_gate_gradient_matches_finite_difference, test_gate_shrinks_nonlinear_mass_on_additive_data
from test_bayesc_prior import test_bayesc_default_is_sparse_and_pi_semantics, test_bayesc_pi_controls_sparsity_and_fixes_ridge_default
from test_objective_redlines import (test_pure_additive_gate_rejects_or_matches_core, test_dominance_tracked_by_core, test_higher_order_score_positive_on_epistasis, test_gate_disabled_is_backward_compatible)


def main():
    test_basic_models_fit_and_predict()
    test_rsi_and_calibration()
    test_reml_gradient_matches_finite_difference()
    test_kernel_state_training_and_cross_shapes()
    test_training_cross_kernel_consistency()
    test_shared_kernel_state_supports_model_and_calibration_variants()
    test_sparse_genotype_input_matches_dense()
    test_memmap_genotype_input_matches_dense()
    test_memmap_validation_rejects_bad_codes_and_nan()
    test_reml_multistart_diagnostics_and_objective()
    test_additive_only_msbgp_near_gblup()
    test_candidate_stack_prediction_uses_shared_cross_cache_path()
    test_native_msbcal_default_has_no_external_model_calls()
    test_native_epi_candidate_evaluated_on_auto_kernels()
    test_diagnostics_and_reliability()
    test_gls_projection_and_optimizer_diagnostics()
    test_reliability_uses_gls_projection()
    test_gate_gradient_matches_finite_difference()
    test_gate_shrinks_nonlinear_mass_on_additive_data()
    test_pure_additive_gate_rejects_or_matches_core()
    test_dominance_tracked_by_core()
    test_higher_order_score_positive_on_epistasis()
    test_gate_disabled_is_backward_compatible()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()



