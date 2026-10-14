import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def decision_boundary(model, X_train, y_train, X_test, y_test, title):
    X = np.vstack([X_train, X_test])
    x0 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    x1 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    xx, yy = np.meshgrid(x0, x1)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(xx, yy, zz, alpha=0.25)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o", label="Train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x", label="Test")
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    fig.tight_layout()
    return fig


def kernel_heatmap(K):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(K, ax=ax)
    ax.set_title("Quantum Kernel Similarity Matrix")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    return fig
