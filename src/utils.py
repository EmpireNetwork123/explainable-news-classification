"""Evaluation and plotting helpers shared by both models."""
import json

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config


def compute_metrics(y_true, y_pred):
    labels = list(range(len(config.LABEL_NAMES)))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, target_names=config.LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax,
        xticklabels=config.LABEL_NAMES, yticklabels=config.LABEL_NAMES,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(log_history, path):
    """Plot per-epoch training loss, validation loss and validation accuracy."""
    train = {round(e["epoch"]): e["loss"] for e in log_history if "loss" in e}
    evals = [e for e in log_history if "eval_loss" in e]
    epochs = [round(e["epoch"]) for e in evals]

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(epochs, [train.get(ep) for ep in epochs], "o-", label="Training loss")
    ax.plot(epochs, [e["eval_loss"] for e in evals], "x--", label="Validation loss")
    ax.plot(epochs, [e["eval_accuracy"] for e in evals], "s-.", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(loc="center right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
