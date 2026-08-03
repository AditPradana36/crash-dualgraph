"""
Plot per-epoch training history saved by train.run_scenario's history_dir.

Usage in a Colab cell:
    from plot_history import plot_fold, plot_all_folds
    plot_fold(CHECKPOINT_DIR, "A_linear", "fold_rep0", 0)
    plot_all_folds(CHECKPOINT_DIR, "A_linear")
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load(checkpoint_dir, tag, fold_col, fold_id):
    path = Path(checkpoint_dir) / f"{tag}_history" / f"{fold_col}_fold{fold_id}.json"
    return json.loads(path.read_text())


def plot_fold(checkpoint_dir, tag, fold_col, fold_id):
    """Loss + val_pr_auc + val_auroc curves for one fold."""
    history = _load(checkpoint_dir, tag, fold_col, fold_id)
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], color="tab:red")
    ax1.set_title(f"{tag} | {fold_col} fold {fold_id} — train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss")

    ax2.plot(epochs, [h["val_pr_auc"] for h in history], label="val PR-AUC", color="tab:blue")
    ax2.plot(epochs, [h["val_auroc"] for h in history], label="val AUROC", color="tab:green")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (AUROC=0.5)")
    best_epoch = max(range(len(history)), key=lambda i: history[i]["val_pr_auc"])
    ax2.axvline(best_epoch, color="black", linestyle=":", linewidth=1, label=f"best epoch ({best_epoch})")
    ax2.set_title(f"{tag} | {fold_col} fold {fold_id} — val metrics")
    ax2.set_xlabel("epoch")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_all_folds(checkpoint_dir, tag, metric="val_pr_auc"):
    """Overlay one metric's curve across every fold that has a saved history
    for this tag -- fast way to see if failure is fold-specific or universal."""
    history_dir = Path(checkpoint_dir) / f"{tag}_history"
    files = sorted(history_dir.glob("*.json"))
    if not files:
        print(f"No history files found in {history_dir}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for f in files:
        history = json.loads(f.read_text())
        label = f.stem
        ax.plot([h["epoch"] for h in history], [h[metric] for h in history], label=label, alpha=0.8)

    if metric in ("val_pr_auc", "val_auroc"):
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"{tag} — {metric} across all completed folds")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.show()
