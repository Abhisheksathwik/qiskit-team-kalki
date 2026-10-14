import numpy as np

from src.classical import train_classical
from src.data import load_data
from src.evaluation import evaluate


def test_classical_model_returns_common_result_contract():
    data = load_data("Two Moons", samples=60, test_size=0.2, seed=5, n_qubits=2)
    result = train_classical(data.X_train, data.y_train, data.X_test, seed=5)
    assert set(("model", "predictions", "train_seconds", "inference_seconds", "preprocessing_seconds")) <= set(result)
    assert result["predictions"].shape == data.y_test.shape
    assert result["train_seconds"] >= 0
    assert result["inference_seconds"] >= 0
    metrics = evaluate(data.y_test, result["predictions"])
    assert 0 <= metrics["Accuracy"] <= 1
