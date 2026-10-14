import matplotlib
matplotlib.use("Agg")

import numpy as np

from src.visualization import (
    confusion_matrix_plot,
    decision_boundary,
    kernel_heatmap,
    metrics_comparison,
    training_loss,
    training_time_chart,
)


def test_all_required_visualization_helpers_return_figures():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    metrics = {
        "Classical RBF SVM": {"Accuracy": 0.75, "Precision": 2/3, "Recall": 1.0, "F1 Score": 0.8},
        "Hybrid VQC": {"Accuracy": 0.5, "Precision": 0.5, "Recall": 0.5, "F1 Score": 0.5},
    }
    simple_model_predict = lambda points: (points[:, 0] + points[:, 1] > 0).astype(int)
    train_x = np.array([[-1.0, -1.0], [-0.8, -0.5], [0.8, 0.5], [1.0, 1.0]])
    test_x = np.array([[-0.6, -0.4], [0.6, 0.4]])

    figures = [
        metrics_comparison(metrics),
        confusion_matrix_plot(y_true, predictions, "Confusion"),
        decision_boundary(simple_model_predict, train_x, y_true, test_x, np.array([0, 1]), "Boundary", resolution=15),
        kernel_heatmap(np.eye(4)),
        training_loss([0.8, 0.6, 0.4]),
        training_time_chart({"Classical": 0.01, "Quantum": 0.2}),
    ]
    assert all(hasattr(fig, "savefig") for fig in figures)
    import matplotlib.pyplot as plt
    for fig in figures:
        plt.close(fig)
