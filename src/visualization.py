"""Plotting helpers used by the Streamlit interface."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def metrics_comparison(metrics_by_model):
    names = list(metrics_by_model)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.18
    positions = np.arange(len(names))
    for index, metric in enumerate(metrics):
        values = [metrics_by_model[name].get(metric, np.nan) for name in names]
        ax.bar(positions + (index - 1.5) * width, values, width, label=metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out classification metrics")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def confusion_matrix_plot(y_true, predictions, title):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="mako", cbar=False,
                xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"], ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def decision_boundary(predict_fn, X_train, y_train, X_test, y_test, title, resolution=70):
    """Draw a 2-D slice; predict_fn accepts a (n, 2) grid and returns labels."""
    X = np.vstack((X_train[:, :2], X_test[:, :2]))
    x0 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, int(resolution))
    x1 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, int(resolution))
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict_fn(grid)).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.contourf(xx, yy, zz, alpha=0.22, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o", edgecolors="white", linewidths=0.3, label="Train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x", linewidths=1.4, label="Test")
    ax.set_title(title)
    ax.set_xlabel("Scaled feature 1")
    ax.set_ylabel("Scaled feature 2")
    ax.legend()
    fig.tight_layout()
    return fig


def kernel_heatmap(K, title="Quantum Kernel Similarity Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(np.asarray(K), cmap="mako", ax=ax, cbar_kws={"label": "Fidelity"})
    ax.set_title(title)
    ax.set_xlabel("Training sample")
    ax.set_ylabel("Training sample")
    fig.tight_layout()
    return fig


def training_loss(history):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(np.arange(1, len(history) + 1), history, marker=".", linewidth=1.6)
    ax.set_title("VQC optimization objective")
    ax.set_xlabel("Optimizer callback")
    ax.set_ylabel("Objective value")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def training_time_chart(timings):
    names = list(timings)
    values = [timings[name] for name in names]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(names, values)
    ax.set_ylabel("Seconds")
    ax.set_title("Training time (kernel construction included)")
    ax.tick_params(axis="x", labelrotation=12)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig
