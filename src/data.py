import numpy as np
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(name="Two Moons", samples=100, test_size=0.2, seed=42):
    if name == "Two Moons":
        X, y = make_moons(n_samples=samples, noise=0.18, random_state=seed)
        feature_names = ["Feature 1", "Feature 2"]
    else:
        data = load_breast_cancer()
        X, y = data.data[:, :2], data.target
        feature_names = list(data.feature_names[:2])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    qscale = MinMaxScaler(feature_range=(0, np.pi))
    X_train_q = qscale.fit_transform(X_train)
    X_test_q = qscale.transform(X_test)
    return X_train, X_test, X_train_q, X_test_q, y_train, y_test, feature_names
