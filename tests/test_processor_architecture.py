import numpy as np
import pytest

from src.problem import make_ps_q1_data
from src.processors import build_noise_model, load_processor
from src.metrics import output_distribution_fidelity


@pytest.mark.parametrize(("name", "expected_qubits"), [("A", 5), ("B", 9)])
def test_processor_configs_are_valid_connected_graphs(name, expected_qubits):
    config = load_processor(name)
    assert config["num_qubits"] == expected_qubits
    assert config["basis_gates"] == ["rz", "sx", "x", "cx"]
    assert config["transpilation"]["optimization_level"] == 1
    assert config["transpilation"]["seed_transpiler"] == 42
    assert len(config["coupling_edges"]) == (5 if name == "A" else 12)
    assert config["noise"]["one_qubit_depolarizing_probability"] == pytest.approx(0.001)
    assert config["noise"]["cx_depolarizing_probability"] == pytest.approx(0.005)
    assert config["noise"]["readout_error_probability"] == pytest.approx(0.01)


def test_ps_q1_dataset_matches_official_shape_and_is_reproducible():
    first = make_ps_q1_data(seed=42)
    second = make_ps_q1_data(seed=42)
    assert first.X_train.shape == (160, 4)
    assert first.X_test.shape == (40, 4)
    assert first.X_train_q.shape == first.X_train.shape
    assert first.X_test_q.shape == first.X_test.shape
    assert np.isfinite(first.X_train_q).all()
    assert np.min(first.X_train_q) >= 0.0
    assert np.max(first.X_train_q) <= np.pi
    np.testing.assert_array_equal(first.y_test, second.y_test)
    np.testing.assert_allclose(first.X_train_q, second.X_train_q)


def test_official_noise_model_constructs_for_both_processors():
    pytest.importorskip("qiskit_aer")
    for name in ("A", "B"):
        model = build_noise_model(name)
        assert model is not None


def test_distribution_fidelity_is_bounded_and_identical_counts_equal_one():
    assert output_distribution_fidelity({"00": 5, "11": 5}, {"00": 5, "11": 5}) == pytest.approx(1.0)
    value = output_distribution_fidelity({"00": 10}, {"11": 10})
    assert 0.0 <= value <= 1.0
    assert value == pytest.approx(0.0)
