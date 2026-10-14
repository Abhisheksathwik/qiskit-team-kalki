"""Hybrid Variational Quantum Classifier (VQC)."""
from __future__ import annotations
import time
import numpy as np


def train_vqc(
    X_train_q, y_train, X_test_q, *, maxiter=40, reps=1,
    optimizer_name="COBYLA", shots=1024, seed=42,
):
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit_aer.primitives import Sampler
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA, SPSA

    n_qubits = int(X_train_q.shape[1])
    if n_qubits not in (2, 4):
        raise ValueError("VQC currently supports 2 or 4 qubits")
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1, entanglement="linear")
    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=int(reps), entanglement="linear")
    history = []

    def callback(weights, objective_value):
        history.append(float(objective_value))

    if optimizer_name.upper() == "SPSA":
        optimizer = SPSA(maxiter=int(maxiter), learning_rate=0.05, perturbation=0.1)
    else:
        optimizer = COBYLA(maxiter=int(maxiter))

    sampler = Sampler(
        run_options={"shots": int(shots)},
        backend_options={"seed_simulator": int(seed)},
    )
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
        "optimizer": optimizer_name.upper(),
    }
