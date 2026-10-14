"""Leakage-safe data loading and preprocessing for QuantumClassify Lab."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class PreparedData:
    dataset_name: str
    n_qubits: int
    test_size: float
    seed: int
    X_train: np.ndarray
    X_test: np.ndarray
    X_train_q: np.ndarray
    X_test_q: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    angle_scaler: MinMaxScaler
    standard_scaler: StandardScaler
    selector: Any = None

    @property
    def train_size(self) -> int:
        return int(len(self.y_train))

    @property
    def test_count(self) -> int:
        return int(len(self.y_test))


def _two_moons_features(n_samples: int, seed: int, n_features: int):
    X, y = make_moons(n_samples=n_samples, noise=0.18, random_state=seed)
    names = ["Moon x", "Moon y"]
    if n_features == 4:
        X = np.column_stack((X, X[:, 0] * X[:, 1], X[:, 0] ** 2 - X[:, 1] ** 2))
        names.extend(["x × y", "x² − y²"])
    return X, y, names


def load_data(name: str = "Two Moons", samples: int = 80, test_size: float = 0.2, seed: int = 42, n_qubits: int = 2) -> PreparedData:
    """Prepare one shared stratified split; fit every transform on training data only."""
    if n_qubits not in (2, 4):
        raise ValueError("n_qubits must be either 2 or 4")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.10 and 0.40")
    if samples < 40:
        raise ValueError("Use at least 40 samples for a meaningful split")

    selector = None
    if name == "Two Moons":
        X, y, names = _two_moons_features(samples, seed, n_qubits)
    elif name in ("Breast Cancer", "Breast Cancer Wisconsin"):
        breast = load_breast_cancer()
        X, y = breast.data, breast.target
        if samples < len(y):
            X, _, y, _ = train_test_split(X, y, train_size=samples, stratify=y, random_state=seed)
        names = list(breast.feature_names)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    if name != "Two Moons":
        selector = SelectKBest(score_func=f_classif, k=n_qubits)
        X_train_raw = selector.fit_transform(X_train_raw, y_train)
        X_test_raw = selector.transform(X_test_raw)
        names = [feature for feature, keep in zip(names, selector.get_support()) if keep]

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train_raw)
    X_test = standard_scaler.transform(X_test_raw)
    angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi), clip=True)
    X_train_q = angle_scaler.fit_transform(X_train)
    X_test_q = angle_scaler.transform(X_test)

    return PreparedData(
        dataset_name=name, n_qubits=n_qubits, test_size=test_size, seed=seed,
        X_train=X_train, X_test=X_test, X_train_q=X_train_q, X_test_q=X_test_q,
        y_train=np.asarray(y_train, dtype=int), y_test=np.asarray(y_test, dtype=int),
        feature_names=names, angle_scaler=angle_scaler, standard_scaler=standard_scaler,
        selector=selector,
    )
