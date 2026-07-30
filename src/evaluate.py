"""
PR-AUC-primary metrics, corrected significance testing across the
20 pre-specified pairs.

Functions to implement:
- compute_metrics(y_true, y_pred) -> dict
- aggregate_fold_results(results) -> mean, std
- paired_significance_test(scores_a, scores_b, method="nadeau_bengio") -> p_value
- apply_correction(p_values, method="holm_bonferroni") -> corrected_p_values
"""
