# ⚛️ QuantumClassify Lab

## Qiskit Fall Fest — Quantum Machine Learning Project

QuantumClassify Lab is an interactive platform for comparing classical machine learning with two Quantum Machine Learning approaches on the same classification task.

### Models

1. **Classical RBF SVM** — baseline model.
2. **Hybrid Variational Quantum Classifier (VQC)** — parameterized quantum circuit optimized by a classical optimizer.
3. **Quantum Kernel + SVM** — quantum feature-space similarity used by a classical SVM.

### Architecture

```text
Dataset → Preprocessing → ┌─ Classical SVM ─────┐
                          ├─ Hybrid VQC ────────┤ → Evaluation → Dashboard
                          └─ Quantum Kernel+SVM ┘
```

### Dashboard features

- Two Moons and Breast Cancer datasets
- Configurable samples, test split and VQC optimizer iterations
- Accuracy, precision, recall and F1
- Training-time comparison
- Decision-boundary plots
- VQC feature-map and ansatz circuit display
- VQC training-loss curve
- Quantum-kernel similarity heatmap
- CSV experiment export

### Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Ubuntu/WSL:

```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Demo flow

Start with **Two Moons**, run all three models, compare the metrics, inspect the VQC circuit and loss curve, then show the quantum-kernel heatmap.

### Research note

This project is an experimental comparison. A higher score on one small dataset does not prove quantum advantage. The presentation should report actual measured results and discuss simulator cost, circuit depth, optimization difficulty, finite-shot effects and hardware noise.
