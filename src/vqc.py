import time
import numpy as np


def train_vqc(X_train_q, y_train, X_test_q, maxiter=40):
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_aer.primitives import Sampler

    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    history = []

    def callback(weights, objective_value):
        history.append(float(objective_value))

    vqc = VQC(
        sampler=Sampler(),
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback,
    )
    start = time.perf_counter()
    vqc.fit(X_train_q, y_train)
    elapsed = time.perf_counter() - start
    pred = np.asarray(vqc.predict(X_test_q)).astype(int).ravel()
    return vqc, pred, elapsed, history, feature_map, ansatz
