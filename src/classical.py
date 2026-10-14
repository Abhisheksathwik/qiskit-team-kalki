"""Classical baseline model."""
from __future__ import annotations
import time
from sklearn.svm import SVC


def train_classical(X_train, y_train, X_test, *, seed: int = 42, C: float = 1.0, gamma: str | float = "scale"):
    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    inference_seconds = time.perf_counter() - start

    return {
        "model": model,
        "predictions": predictions,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "preprocessing_seconds": 0.0,
    }
