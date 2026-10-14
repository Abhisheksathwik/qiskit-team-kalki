"""Hybrid Variational Quantum Classifier using V2 sampler primitives."""
from __future__ import annotations

import time
import numpy as np

from src.quantum_backend import build_sampler


def train_vqc(
    X_train_q, y_train, X_test_q, *, maxiter=40, reps=1,
    optimizer_name="COBYLA", shots=1024, seed=42, noise_strength=0.0,
):
    from qiskit.circuit.library import real_amplitudes, zz_feature_map
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA, SPSA

    n_qubits = int(X_train_q.shape[1])
    if n_qubits not in (2, 4):
        raise ValueError("VQC currently supports 2 or 4 qubits")
    feature_map = zz_feature_map(feature_dimension=n_qubits, reps=1, entanglement="linear")
    ansatz = real_amplitudes(num_qubits=n_qubits, reps=int(reps), entanglement="linear")
    history = []

    def callback(weights, objective_value):
        history.append(float(objective_value))

    optimizer_key = optimizer_name.upper()
    if optimizer_key == "SPSA":
        optimizer = SPSA(maxiter=int(maxiter), learning_rate=0.05, perturbation=0.1)
    elif optimizer_key == "COBYLA":
        optimizer = COBYLA(maxiter=int(maxiter))
    else:
        raise ValueError("optimizer_name must be COBYLA or SPSA")

    sampler = build_sampler(shots=int(shots), seed=int(seed), noise_strength=float(noise_strength))
    model = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    start = time.perf_counter()
    model.fit(X_train_q, np.asarray(y_train, dtype=int))
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = np.asarray(model.predict(X_test_q)).astype(int).ravel()
    inference_seconds = time.perf_counter() - start

    return {
        "model": model,
        "predictions": predictions,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "history": history,
        "feature_map": feature_map,
        "ansatz": ansatz,
        "shots": int(shots),
        "n_qubits": n_qubits,
        "reps": int(reps),
        "optimizer": optimizer_key,
        "noise_strength": float(noise_strength),
    }
