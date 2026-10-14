"""Official QFF 2026 PS-Q1 synthetic classification dataset.

The guide's requested four-feature dataset is generated locally. scikit-learn's
make_classification defaults to two redundant features, which would make
n_informative=3 + n_redundant=2 exceed n_features=4. We explicitly set
n_redundant=0 so the stated four-feature / three-informative setup is valid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PSQ1Data:
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_q: np.ndarray
    X_test_q: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    standard_scaler: StandardScaler
    angle_scaler: MinMaxScaler
    seed: int = 42


def make_ps_q1_data(seed: int = 42) -> PSQ1Data:
    """Generate the guide's 200-sample, four-feature, binary PS-Q1 dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=int(seed),
    )
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=int(seed), stratify=y
    )
    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train_raw)
    X_test = standard_scaler.transform(X_test_raw)
    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
    X_train_q = angle_scaler.fit_transform(X_train)
    X_test_q = angle_scaler.transform(X_test)
    return PSQ1Data(
        X_train=X_train,
        X_test=X_test,
        X_train_q=X_train_q,
        X_test_q=X_test_q,
        y_train=np.asarray(y_train, dtype=int),
        y_test=np.asarray(y_test, dtype=int),
        standard_scaler=standard_scaler,
        angle_scaler=angle_scaler,
        seed=int(seed),
    )
