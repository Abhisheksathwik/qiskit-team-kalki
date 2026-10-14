import pytest

from src.circuits import ANSATZ_OPTIONS, FEATURE_MAP_OPTIONS, build_ansatz, build_feature_map


@pytest.mark.parametrize("n_qubits", [2, 4])
@pytest.mark.parametrize("feature_map_name", FEATURE_MAP_OPTIONS)
def test_feature_map_builders_preserve_feature_dimension(n_qubits, feature_map_name):
    circuit = build_feature_map(n_qubits, feature_map_name)
    assert circuit.num_qubits == n_qubits
    assert len(circuit.parameters) == n_qubits


@pytest.mark.parametrize("n_qubits", [2, 4])
@pytest.mark.parametrize("ansatz_name", ANSATZ_OPTIONS)
def test_ansatz_builders_produce_trainable_parameters(n_qubits, ansatz_name):
    circuit = build_ansatz(n_qubits, reps=1, name=ansatz_name)
    assert circuit.num_qubits == n_qubits
    assert len(circuit.parameters) > 0


def test_invalid_architecture_names_are_rejected():
    with pytest.raises(ValueError, match="Unsupported feature map"):
        build_feature_map(2, "not a feature map")
    with pytest.raises(ValueError, match="Unsupported ansatz"):
        build_ansatz(2, 1, "not an ansatz")
