"""Fidelity quantum kernel classifier with a classical precomputed-kernel SVM."""
from __future__ import annotations

import time
import numpy as np

from src.quantum_backend import build_sampler, suppress_aer_compatibility_warnings


def train_quantum_kernel(
    X_train_q, y_train, X_test_q, *, reps=1, seed=42, max_train_samples=105,
    shots=1024, noise_strength=0.0,
):
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from sklearn.svm import SVC

    if len(X_train_q) > max_train_samples:
        raise ValueError(
            f"Quantum kernel received {len(X_train_q)} training samples; "
            f"the configured safety cap is {max_train_samples}. Reduce dataset size."
        )

    n_qubits = int(X_train_q.shape[1])
    feature_map = zz_feature_map(feature_dimension=n_qubits, reps=1, entanglement="linear")
    with suppress_aer_compatibility_warnings():
        sampler = build_sampler(shots=int(shots), seed=int(seed), noise_strength=float(noise_strength))
        fidelity = ComputeUncompute(sampler=sampler)
        kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

        start = time.perf_counter()
        K_train = np.asarray(kernel.evaluate(X_train_q), dtype=float)
        kernel_build_seconds = time.perf_counter() - start

        classifier = SVC(kernel="precomputed", random_state=seed)
        start = time.perf_counter()
        classifier.fit(K_train, y_train)
        classifier_train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        K_test = np.asarray(kernel.evaluate(X_test_q, X_train_q), dtype=float)
        predictions = classifier.predict(K_test)
        inference_seconds = time.perf_counter() - start

    return {
        "model": classifier,
        "predictions": predictions,
        "train_seconds": kernel_build_seconds + classifier_train_seconds,
        "kernel_build_seconds": kernel_build_seconds,
        "classifier_train_seconds": classifier_train_seconds,
        "inference_seconds": inference_seconds,
        "kernel_train": K_train,
        "kernel_test": K_test,
        "feature_map": feature_map,
        "n_qubits": n_qubits,
        "reps": int(reps),
        "shots": int(shots),
        "noise_strength": float(noise_strength),
    }
