"""Shared Qiskit Aer V2 sampler construction for ideal/noisy simulations."""
from __future__ import annotations

from contextlib import contextmanager
import warnings


@contextmanager
def suppress_aer_compatibility_warnings():
    """Hide known Aer/Qiskit deprecation chatter during simulator execution only."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"qiskit_aer(?:\.|$)",
        )
        yield


def build_sampler(*, shots: int = 1024, seed: int = 42, noise_strength: float = 0.0):
    """Create a seeded V2 sampler, optionally adding depolarizing gate noise.

    noise_strength is the depolarizing probability for each configured one-qubit gate.
    The two-qubit depolarizing probability is twice that value. Zero means an ideal
    shot-based simulator (sampling noise still remains when shots are finite).
    """
    from qiskit_aer.primitives import SamplerV2

    if shots < 1:
        raise ValueError("shots must be a positive integer")
    if not 0.0 <= noise_strength <= 0.05:
        raise ValueError("noise_strength must be between 0 and 0.05")

    options = None
    if noise_strength > 0:
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise_model = NoiseModel()
        one_qubit_error = depolarizing_error(float(noise_strength), 1)
        two_qubit_error = depolarizing_error(min(float(noise_strength) * 2.0, 0.1), 2)
        one_qubit_gates = ["h", "p", "rx", "ry", "rz", "u", "u1", "u2", "u3", "x", "y", "z"]
        two_qubit_gates = ["cx", "cz"]
        noise_model.add_all_qubit_quantum_error(one_qubit_error, one_qubit_gates)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, two_qubit_gates)
        options = {"backend_options": {"noise_model": noise_model}}

    return SamplerV2(
        default_shots=int(shots),
        seed=int(seed),
        options=options,
    )
