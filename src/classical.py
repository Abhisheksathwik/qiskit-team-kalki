import time
from sklearn.svm import SVC


def train_classical(X_train, y_train, X_test):
    start = time.perf_counter()
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, model.predict(X_test), elapsed
