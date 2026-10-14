import time


def train_quantum_kernel(X_train_q, y_train, X_test_q):
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from sklearn.svm import SVC

    feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    start = time.perf_counter()
    K_train = kernel.evaluate(X_train_q)
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    K_test = kernel.evaluate(X_test_q, X_train_q)
    pred = clf.predict(K_test)
    elapsed = time.perf_counter() - start
    return clf, pred, elapsed, K_train, feature_map
