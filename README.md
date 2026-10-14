# ⚛️ QuantumClassify Lab

**Qiskit Fall Fest · Quantum Machine Learning benchmark**

An interactive research dashboard comparing a classical RBF Support Vector Machine (SVM), a Hybrid Variational Quantum Classifier (VQC), and a fidelity quantum-kernel SVM on a shared, stratified held-out test set.

> This is an experimental comparison platform. It does not assume quantum advantage, and a small simulator benchmark cannot establish hardware speedup.

## Core features

- Two Moons and Breast Cancer Wisconsin binary classification datasets.
- Compact 2- and 4-feature / qubit experiments.
- Leakage-safe scaling and training-fitted feature selection.
- Three separate pipelines: classical RBF SVM, trainable VQC, and quantum kernel + classical SVM.
- Adjustable VQC ansatz layers, optimizer, iteration count, shot count, seed, and repeated runs.
- Accuracy, precision, recall, F1, confusion matrices, training time, inference time, and quantum-kernel build time.
- Visuals: score comparison, confusion matrices, decision boundaries, VQC optimization loss, quantum-kernel fidelity heatmap, circuit diagrams, and training-time chart.
- Evidence-based findings and a downloadable Markdown experiment report.
- Detailed metrics CSV, saved history JSON, and session-cached experiment results.
- GitHub Actions CI and automated tests.

## Technology

Python 3.11, Qiskit, Qiskit Aer, Qiskit Machine Learning, scikit-learn, NumPy, pandas, Matplotlib, Seaborn, Streamlit, and pytest.

## Quick start — Windows PowerShell

    git clone https://github.com/Abhisheksathwik/qiskit-team-kalki.git
    cd qiskit-team-kalki
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

If PowerShell blocks environment activation, use Ubuntu/WSL or activate the virtual environment through a different terminal.

## Quick start — Ubuntu / WSL

    git clone https://github.com/Abhisheksathwik/qiskit-team-kalki.git
    cd qiskit-team-kalki
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    streamlit run app.py

Streamlit prints a local URL, normally http://localhost:8501.

## One-command launchers

- Windows PowerShell: run `./run.ps1` from the repository root.
- Ubuntu/WSL: run `chmod +x run.sh && ./run.sh`.

## Docker

Build and start the application from the repository root:

    docker build -t quantumclassify-lab .
    docker run --rm -p 8501:8501 quantumclassify-lab

Then open http://localhost:8501. The container writes experiment history locally inside the container; download reports you need to keep.

## Run tests

    pytest -q
    python -m compileall -q app.py src tests

GitHub Actions runs the compile and test checks on pushes and pull requests to main.

## Project layout

    qiskit-team-kalki/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .streamlit/config.toml
    ├── src/
    │   ├── data.py
    │   ├── classical.py
    │   ├── vqc.py
    │   ├── quantum_kernel.py
    │   ├── evaluation.py
    │   ├── visualization.py
    │   ├── reporting.py
    │   └── history.py
    ├── tests/
    └── results/ (local experiment history; not committed)

## Model overview

### 1. Classical RBF SVM

A conventional scikit-learn SVC with an RBF kernel provides a reference for both performance and runtime.

### 2. Hybrid Variational Quantum Classifier (VQC)

A quantum feature map encodes the classical features. A trainable ansatz is optimized by a classical optimizer, and measurements drive predictions. The VQC tab displays the feature map, ansatz, optimization trace when available, metrics, and confusion matrix.

### 3. Quantum Kernel + SVM

The fidelity kernel computes similarity between encoded quantum states:

    K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2

The train kernel matrix is consumed by a classical SVM using a precomputed kernel. Test-to-train similarities are computed separately. Kernel build time is reported separately from classical SVM fitting.

## Experimental fairness

- Identical stratified split and seed for every model within a run.
- Feature selection and scaling are fitted only on training data.
- Same retained features and qubit count for both quantum methods.
- Held-out test metrics are separated from training, inference, and kernel-build time.
- Repeated seeds can be summarized with mean and sample standard deviation.
- Never claim quantum advantage based on one small dataset or simulator result.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select repository Abhisheksathwik/qiskit-team-kalki, branch main, and entry file app.py.
4. Deploy using requirements.txt from the repository root.
5. Begin with Two Moons, 2 qubits, one ansatz layer, and modest optimizer iterations.

Quantum simulations can be computationally expensive, particularly for quantum kernels. Local results/history.json is not a durable database on ephemeral hosting, so download reports to retain results.

## Limitations and future work

- Simulator behaviour and costs differ from real quantum hardware.
- Kernel matrix evaluation scales quadratically in sample count.
- VQC optimization is sensitive to initialization, depth, optimizer, and sampling noise.
- The Breast Cancer demo selects a compact feature subset using the training split.
- Four-feature decision-boundary visualizations are two-dimensional slices.
- Future work: noise models, optional IBM Quantum backend integration, more datasets, broader seed sweeps, and automated feature-map/ansatz comparisons.

## References

- Qiskit Machine Learning documentation: https://qiskit-community.github.io/qiskit-machine-learning/
- Qiskit Machine Learning release notes: https://qiskit-community.github.io/qiskit-machine-learning/release_notes.html
- Streamlit documentation: https://docs.streamlit.io/
- Schuld & Killoran, Quantum machine learning in feature Hilbert spaces, Physical Review Letters 122 (2019).
- McClean et al., Barren plateaus in quantum neural network training landscapes, Nature Communications 9 (2018).
