"""Metrics and plots (paper §4.2 end, §5.2, Figure 2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data import CATEGORY_NAMES, to_category


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    logits = model(torch.from_numpy(X).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def multiclass_accuracy(pred, y):
    return float((pred == y).mean())


def binary_accuracy(pred, y, normal_index=0):
    return float(((pred == normal_index) == (y == normal_index)).mean())


def evaluate(model, X, y, device):
    pred = predict(model, X, device)
    return multiclass_accuracy(pred, y), binary_accuracy(pred, y), pred


def plot_fig2(train_acc, test_acc, path):
    """Styled like the paper's Figure 2: epochs from 0, red train / blue test curves."""
    epochs = np.arange(len(train_acc))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, acc, title, color in ((axes[0], train_acc, "(a) TRAIN", "red"),
                                  (axes[1], test_acc, "(b) TEST", "blue")):
        ax.plot(epochs, acc, color=color)
        ax.set_xlabel("epochs")
        ax.set_ylabel("accuracy")
        ax.text(0.02, 0.97, title, transform=ax.transAxes, va="top", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_category_confusion(pred, y_test_raw, class_names, path):
    """5-category (Normal/DoS/Probe/R2L/U2R) confusion matrix on the test set.
    Predictions of the 'other' class have no category and are left out."""
    true_cat = [to_category(l) for l in y_test_raw]
    pred_cat = [to_category(class_names[p]) for p in pred]
    cm = confusion_matrix(true_cat, pred_cat, labels=CATEGORY_NAMES)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.1f}%)", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(CATEGORY_NAMES)), CATEGORY_NAMES)
    ax.set_yticks(range(len(CATEGORY_NAMES)), CATEGORY_NAMES)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.set_title("Test-set confusion matrix (seed 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm
