"""
PR-AUC/AUROC-primary metrics (co-equal, each its own correction family),
Wilcoxon (primary) + Nadeau-Bengio (secondary) significance testing across
the pre-specified pairs, Holm-Bonferroni correction. All three verified
against known cases (identical scores -> p=1, clearly-separated scores ->
low p, manual Holm-Bonferroni arithmetic) before being written here.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, confusion_matrix, log_loss,
)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # confusion_matrix needs labels= pinned in case a fold's y_true/y_pred
    # happens to contain only one class (small/imbalanced folds) -- without
    # it sklearn would return a 1x1 matrix instead of 2x2.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # log_loss needs both classes present in y_true, and y_prob clipped away
    # from exact 0/1 to avoid -inf.
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    bce_loss = (log_loss(y_true, y_prob_clipped, labels=[0, 1])
                if len(np.unique(y_true)) > 1 else float("nan"))

    return {
        "pr_auc": average_precision_score(y_true, y_prob),
        "auroc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "bce_loss": bce_loss,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def aggregate_fold_results(fold_metric_dicts):
    """List of per-fold metric dicts -> {metric: (mean, std)}."""
    keys = fold_metric_dicts[0].keys()
    return {k: (float(np.mean([d[k] for d in fold_metric_dicts])),
                float(np.std([d[k] for d in fold_metric_dicts])))
            for k in keys}


def nadeau_bengio_test(scores_a, scores_b, n_train, n_test):
    """Corrected resampled t-test — accounts for correlated fold estimates
    under repeated k-fold CV, unlike a plain paired t-test."""
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)
    mean_d, var_d = diffs.mean(), diffs.var(ddof=1)
    correction = (1.0 / n) + (n_test / n_train)
    denom = np.sqrt(correction * var_d)
    if denom == 0:
        return 0.0, 1.0
    t_stat = mean_d / denom
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return float(t_stat), float(p_value)


def wilcoxon_test(scores_a, scores_b):
    """Primary test — nonparametric, no distributional assumption on the
    fold-level score differences."""
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b)
        return float(stat), float(p)
    except ValueError:
        # all differences are zero, or too few samples for scipy's exact method
        return 0.0, 1.0


def holm_bonferroni(p_values):
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.empty(n)
    prev_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(max((n - rank) * p_values[idx], prev_max), 1.0)
        corrected[idx] = adj
        prev_max = adj
    return corrected


def run_comparison(fold_scores, pairs, n_train, n_test):
    """fold_scores: dict scenario_key -> list of fold-level scores (one metric).
    pairs: list of (key_a, key_b) tuples — the pre-specified comparisons.
    Returns a DataFrame-ready list of dicts with both tests + corrected p."""
    import pandas as pd

    rows = []
    for a, b in pairs:
        if a not in fold_scores or b not in fold_scores:
            rows.append({"pair": f"{a} vs {b}", "skipped": True,
                         "reason": "one or both scenarios not yet trained (e.g. F)"})
            continue
        w_stat, w_p = wilcoxon_test(fold_scores[a], fold_scores[b])
        nb_t, nb_p = nadeau_bengio_test(fold_scores[a], fold_scores[b], n_train, n_test)
        rows.append({
            "pair": f"{a} vs {b}", "skipped": False,
            "mean_a": np.mean(fold_scores[a]), "mean_b": np.mean(fold_scores[b]),
            "wilcoxon_p": w_p, "nadeau_bengio_p": nb_p,
        })

    df = pd.DataFrame(rows)
    valid = df[~df["skipped"]] if "skipped" in df.columns else df
    if len(valid):
        df.loc[valid.index, "wilcoxon_p_corrected"] = holm_bonferroni(valid["wilcoxon_p"].values)
        df.loc[valid.index, "significant"] = df.loc[valid.index, "wilcoxon_p_corrected"] < 0.05
    return df
