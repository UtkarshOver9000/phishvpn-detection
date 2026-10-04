"""
Figures for the README: loss curves, ROC/PR curves and the confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402


def loss_curves(boost_hist: dict, mlp_hist: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rounds = np.arange(1, len(boost_hist["val_log_loss_per_round"]) + 1)
    axes[0].plot(rounds, boost_hist["train_log_loss_per_round"], label="train")
    axes[0].plot(rounds, boost_hist["val_log_loss_per_round"], label="validation")
    axes[0].axvline(boost_hist["best_round"], color="grey", ls="--", label=f"kept: round {boost_hist['best_round']}")
    axes[0].set(title="Gradient boosting: log loss per round", xlabel="boosting round", ylabel="log loss")
    axes[0].legend()

    epochs = np.arange(1, len(mlp_hist["val_log_loss_per_epoch"]) + 1)
    axes[1].plot(epochs, mlp_hist["train_log_loss_per_epoch"], marker="o", label="train")
    axes[1].plot(epochs, mlp_hist["val_log_loss_per_epoch"], marker="o", label="validation")
    axes[1].axvline(mlp_hist["best_epoch"], color="grey", ls="--", label=f"kept: epoch {mlp_hist['best_epoch']}")
    axes[1].set(title="MLP: log loss per epoch", xlabel="epoch", ylabel="log loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def roc_pr(y_true, probas: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        prec, rec, _ = precision_recall_curve(y_true, proba)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(rec, prec, label=name)
    axes[0].plot([0, 1], [0, 1], color="grey", ls=":")
    axes[0].set(title="ROC (held-out test domains)", xlabel="false positive rate", ylabel="true positive rate")
    axes[1].set(title="Precision-recall (held-out test domains)", xlabel="recall", ylabel="precision")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def confusion(cm: dict, title: str, out: Path) -> None:
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap="Blues")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > grid.max() / 2 else "black")
    ax.set_xticks([0, 1], ["legit", "phishing"])
    ax.set_yticks([0, 1], ["legit", "phishing"])
    ax.set(xlabel="predicted", ylabel="actual", title=title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
