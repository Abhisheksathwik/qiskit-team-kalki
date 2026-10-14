from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import load_data
from src.evaluation import evaluate
from src.reporting import build_insights, build_markdown_report


@pytest.mark.parametrize("n_qubits", [2, 4])
def test_two_moons_data_shapes_scaling_and_reproducibility(n_qubits):
    first = load_data("Two Moons", samples=80, test_size=0.2, seed=11, n_qubits=n_qubits)
    second = load_data("Two Moons", samples=80, test_size=0.2, seed=11, n_qubits=n_qubits)
    assert first.X_train.shape == (64, n_qubits)
    assert first.X_test.shape == (16, n_qubits)
    assert first.X_train_q.shape == first.X_train.shape
    assert first.X_test_q.shape == first.X_test.shape
    assert np.isfinite(first.X_train).all()
    assert np.isfinite(first.X_test_q).all()
    assert np.min(first.X_train_q) >= 0
    assert np.max(first.X_train_q) <= np.pi
    np.testing.assert_allclose(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.y_test, second.y_test)


def test_breast_cancer_feature_selection_is_compact_and_sample_count_is_honoured():
    data = load_data("Breast Cancer", samples=80, test_size=0.2, seed=7, n_qubits=2)
    assert data.X_train.shape[1] == 2
    assert data.X_test.shape[1] == 2
    assert len(data.feature_names) == 2
    assert data.selector is not None
    assert len(data.y_train) + len(data.y_test) == 80


def test_invalid_qubit_count_is_rejected():
    with pytest.raises(ValueError, match="n_qubits"):
        load_data("Two Moons", n_qubits=3)


def test_evaluation_returns_binary_metrics_and_confusion_matrix():
    metrics = evaluate(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert metrics["Accuracy"] == pytest.approx(0.75)
    assert metrics["Precision"] == pytest.approx(2 / 3)
    assert metrics["Recall"] == pytest.approx(1.0)
    assert metrics["F1 Score"] == pytest.approx(0.8)
    assert metrics["Confusion Matrix"].tolist() == [[1, 1], [0, 2]]


def test_insight_engine_only_describes_supplied_measurements():
    models = {
        "Classical RBF SVM": {"metrics": {"F1 Score": 0.75}, "train_seconds": 0.02, "inference_seconds": 0.003},
        "Hybrid VQC": {"metrics": {"F1 Score": 0.65}, "train_seconds": 2.8, "inference_seconds": 0.05},
    }
    data = SimpleNamespace(y_train=np.array([0, 1]))
    insights = build_insights(models, data)
    assert any("Classical RBF SVM" in sentence and "F1-score" in sentence for sentence in insights)
    assert any("training time" in sentence for sentence in insights)
    assert any("do not establish practical quantum advantage" in sentence for sentence in insights)


def test_markdown_report_includes_results_and_limitations():
    bundle = {
        "created_at": "2026-10-14T00:00:00+00:00",
        "config": {
            "dataset": "Two Moons", "samples": 80, "test_size": 0.2, "qubits": 2,
            "vqc_layers": 1, "optimizer": "COBYLA", "optimizer_iterations": 20, "shots": 512,
        },
        "runs": [{"seed": 42, "models": {"Classical RBF SVM": {}}, "failures": {}}],
    }
    rows = pd.DataFrame([{
        "Model": "Classical RBF SVM", "Seed": 42, "Accuracy": 0.8, "Precision": 0.8,
        "Recall": 0.8, "F1 Score": 0.8, "Training Time (s)": 0.02,
        "Inference Time (s)": 0.001, "Kernel Build Time (s)": 0.0,
    }])
    report = build_markdown_report(bundle, rows, ["Example measured observation."])
    assert "QuantumClassify Lab" in report
    assert "Classical RBF SVM" in report
    assert "Example measured observation." in report
    assert "not evidence" in report
