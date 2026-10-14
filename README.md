QuantumClassify Lab

Hybrid Variational Quantum Classifiers vs. Quantum Kernel Learning
An interactive Quantum Machine Learning experimentation platform 

Overview:

QuantumClassify Lab is an interactive application for exploring and comparing classical machine learning with two quantum machine learning approaches:
Hybrid Variational Quantum Classifier (VQC) — a trainable quantum circuit optimized using a classical optimization algorithm.

Quantum Kernel Classifier — a quantum feature map combined with a classical Support Vector Machine.
Classical baseline — a conventional RBF-kernel Support Vector Machine.

The application allows users to configure experiments, visualize quantum circuits, inspect quantum kernel matrices, compare classification metrics, and investigate the computational cost of each approach.

The objective is not to assume that quantum machine learning is superior, but to evaluate its behavior through controlled and reproducible experiments.
Problem Statement

Classical machine learning algorithms can perform well on many classification tasks. Quantum machine learning explores whether quantum state representations and parameterized quantum circuits can provide useful alternatives for particular learning problems.

However, quantum approaches also introduce challenges, including circuit execution cost, optimization difficulty, measurement noise, and hardware limitations.

This project investigates:

How does a hybrid variational quantum classifier perform against a classical SVM?
How does quantum kernel learning compare with a trainable quantum classifier?
How do circuit depth, dataset size, and optimizer settings affect performance?
What trade-offs exist between predictive performance and computational cost?

Objectives

Implement a working VQC using Qiskit Machine Learning.
Implement a fidelity-based quantum kernel classifier.
Establish a fair classical ML baseline.
Evaluate models using held-out test data.
Visualize decision boundaries, training behavior, and kernel matrices.
Build an interactive Streamlit dashboard.

Produce reproducible experiments and an automatically generated comparison report.
