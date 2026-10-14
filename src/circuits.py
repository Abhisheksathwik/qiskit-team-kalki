"""Current function-based Qiskit circuit builders used by both quantum pipelines."""
from __future__ import annotations


FEATURE_MAP_OPTIONS = ("ZZ entangling", "Z-only")
ANSATZ_OPTIONS = ("Real Amplitudes", "Efficient SU(2)")


def build_feature_map(n_qubits: int, name: str = "ZZ entangling"):
    from qiskit.circuit.library import z_feature_map, zz_feature_map

    if n_qubits not in (2, 4):
        raise ValueError("n_qubits must be either 2 or 4")
    if name == "ZZ entangling":
        return zz_feature_map(feature_dimension=n_qubits, reps=1, entanglement="linear")
    if name == "Z-only":
        return z_feature_map(feature_dimension=n_qubits, reps=1)
    raise ValueError(f"Unsupported feature map: {name}")


def build_ansatz(n_qubits: int, reps: int = 1, name: str = "Real Amplitudes"):
    from qiskit.circuit.library import efficient_su2, real_amplitudes

    if n_qubits not in (2, 4):
        raise ValueError("n_qubits must be either 2 or 4")
    if reps < 1 or reps > 5:
        raise ValueError("ansatz reps must be between 1 and 5")
    if name == "Real Amplitudes":
        return real_amplitudes(num_qubits=n_qubits, reps=int(reps), entanglement="linear")
    if name == "Efficient SU(2)":
        return efficient_su2(num_qubits=n_qubits, reps=int(reps), entanglement="linear")
    raise ValueError(f"Unsupported ansatz: {name}")
