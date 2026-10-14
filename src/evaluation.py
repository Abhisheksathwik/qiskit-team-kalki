"""Common metrics for every model in the benchmark."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, predictions):
    """Return comparable binary-classification metrics and a confusion matrix."""
    y_true = np.asarray(y_true).ravel()
    predictions = np.asarray(predictions).ravel()
    if y_true.shape != predictions.shape:
        raise ValueError("y_true and predictions must have the same shape")
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, average="binary", zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, average="binary", zero_division=0)),
        "F1 Score": float(f1_score(y_true, predictions, average="binary", zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
    }


def aggregate_seed_results(run_rows):
    """Aggregate numeric metrics by model, reporting mean and sample standard deviation."""
    import pandas as pd

    frame = pd.DataFrame(run_rows)
    metrics = [column for column in ("Accuracy", "Precision", "Recall", "F1 Score",
                                      "Training Time (s)", "Inference Time (s)")
               if column in frame.columns]
    if not metrics:
        return pd.DataFrame()
    return frame.groupby("Model", as_index=False)[metrics].agg(["mean", "std"])
