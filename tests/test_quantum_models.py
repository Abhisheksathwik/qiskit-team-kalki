import numpy as np
import pytest

from src.data import load_data


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_vqc_smoke_train_predict():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_machine_learning")
    from src.vqc import train_vqc

    data = load_data("Two Moons", samples=40, test_size=0.25, seed=3, n_qubits=2)
    result = train_vqc(
        data.X_train_q, data.y_train, data.X_test_q,
        maxiter=8, reps=1, optimizer_name="COBYLA", shots=128, seed=3, noise_strength=0.005,
        feature_map_name="Z-only", ansatz_name="Efficient SU(2)"
    )
    assert result["predictions"].shape == data.y_test.shape
    assert np.isfinite(result["predictions"]).all()
    assert result["train_seconds"] >= 0
    assert result["n_qubits"] == 2


def test_quantum_kernel_smoke_train_predict_and_kernel_shapes():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_machine_learning")
    from src.quantum_kernel import train_quantum_kernel

    data = load_data("Two Moons", samples=40, test_size=0.25, seed=4, n_qubits=2)
    result = train_quantum_kernel(
        data.X_train_q, data.y_train, data.X_test_q,
        seed=4, max_train_samples=40, shots=128, noise_strength=0.005
    )
    assert result["predictions"].shape == data.y_test.shape
    assert result["kernel_train"].shape == (data.train_size, data.train_size)
    assert result["kernel_test"].shape == (data.test_count, data.train_size)
    assert np.allclose(result["kernel_train"], result["kernel_train"].T, atol=1e-6)
    assert result["train_seconds"] >= result["kernel_build_seconds"]

@pytest.mark.parametrize("processor_name", ["A", "B"])
def test_vqc_can_train_with_official_processor_noise_and_topology(processor_name):
    """A low-shot smoke test covers processor-specific VQC sampling without changing its model family."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_machine_learning")
    from src.vqc import train_vqc
    from src.processors import load_processor

    data = load_data("Two Moons", samples=40, test_size=0.25, seed=19, n_qubits=2)
    result = train_vqc(
        data.X_train_q, data.y_train, data.X_test_q,
        maxiter=10, reps=1, optimizer_name="COBYLA", shots=128, seed=19,
        feature_map_name="Z-only", ansatz_name="Efficient SU(2)",
        processor_config=load_processor(processor_name),
    )
    assert result["predictions"].shape == data.y_test.shape
    assert np.isfinite(result["predictions"]).all()
    assert result["processor_name"] == f"Processor {processor_name}"
