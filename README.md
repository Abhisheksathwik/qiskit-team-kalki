# QuantumClassify Lab — Qiskit Fall Fest 2026, Phase 1

**Selected official problem statement: PS-Q1 — Variational Quantum Classifier (VQC).**

QuantumClassify Lab retains its three separate models:
1. Classical RBF-SVM baseline.
2. Hybrid variational quantum classifier (VQC).
3. Fidelity quantum-kernel SVM.

The existing Streamlit app continues to offer all three models. The formal Phase 1 benchmark adds the required processor co-design workflow: a four-qubit PS-Q1 VQC is trained and evaluated using the same data split and seed on the five-qubit Processor A and nine-qubit Processor B, then compared with the CPU-only RBF-SVM baseline. The quantum-kernel SVM remains available as a supplementary dashboard experiment; it is not claimed as the selected official problem statement.

This is a simulator study, not a real IBM Quantum hardware submission. It does not assume or claim quantum advantage.

## Official PS-Q1 experiment

The Phase 1 guide specifies a four-feature, binary synthetic classification task with 200 samples, three informative features, and an 80/20 split. `src/problem.py` implements this generated locally with `random_state=42` by default. It explicitly sets `n_redundant=0`: scikit-learn otherwise defaults to two redundant features, which would conflict with `n_features=4` and `n_informative=3`.

The test split remains held out. Standardization and angle scaling are fitted on the training split only. The quantum features are angle encoded on four logical qubits.

The benchmark executes:
- CPU classical RBF-SVM baseline on the same held-out split.
- VQC under ideal shot-based simulation for reference.
- The same VQC configuration on Processor A and Processor B using the guide's synthetic noise model.

The same seed, feature map, ansatz, optimizer, training split and shot count are used for Processor A and B. Processor-specific noise settings are identical; only the physical graph and routing target differ.

## Participant-defined processor architectures

Processor A uses 5 physical qubits on a connected ring-with-branch graph with 5 undirected coupling edges. Processor B uses 9 physical qubits on a connected sparse graph with a perimeter plus 3 cross-links (12 undirected edges). The topology is intentionally defined in the repository rather than supplied as a pre-compiled JSON.

Both processor JSON files specify:
- Native basis: `rz`, `sx`, `x`, `cx`.
- Transpilation optimization level: 1.
- Transpiler seed: 42.
- Default shots: 4096.
- One-qubit depolarizing probability: 0.001.
- CX depolarizing probability: 0.005.
- Readout bit-flip probability: 0.01.

The noise model is built in `src/processors.py` and applied to both processors at identical rates. `src/routing.py` compiles a representative circuit with and without the topology constraint. `src/metrics.py` records logical/compiled depth, CX count, routing SWAP count, CX overhead, and a measured ideal/noisy output-distribution fidelity. That last quantity is the squared Bhattacharyya coefficient between measurement distributions; it is **not** quantum state fidelity or measured hardware fidelity.

## Run the project

### 1. Create an environment (Windows PowerShell)

    git clone https://github.com/Abhisheksathwik/qiskit-team-kalki.git
    cd qiskit-team-kalki
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

### 2. Launch the existing three-model dashboard

    streamlit run app.py

This interactive application still contains the classical RBF-SVM, VQC, and quantum-kernel SVM. It includes experiment visualizations, measured metrics, reports and exports.

### 3. Run the official architecture benchmark

Open `main.ipynb` in Jupyter and run cells from top to bottom, or run from the project root:

    python -m src.architecture_benchmark

The full official configuration uses 4096 shots and 20 COBYLA iterations. A low-shot/small-iteration run may be used only as a smoke test; restore the official shot count and record the actual configuration for the final submission. Quantum circuit simulation and VQC optimization may take significant time.

A successful run writes measured outputs to:
- `results/tables/model_comparison.csv`
- `results/tables/processor_architecture_comparison.csv`
- `results/tables/benchmark_summary.json`
- `results/figures/processor_coupling_graphs.png`
- `results/figures/processor_metric_comparison.png`
- `report.md`

No numeric results are committed as examples. These files are generated when the experiment actually runs. Failed model runs are marked as failed; they are not filled with illustrative values.

### 4. Validate

    python -m compileall -q app.py src tests
    pytest -q

GitHub Actions runs compilation and unit tests on pushes to `main`.

## Project layout

    app.py
    main.ipynb
    requirements.txt
    processors/
      processor_A.json
      processor_B.json
    src/
      problem.py
      processors.py
      routing.py
      metrics.py
      architecture_benchmark.py
      visualization.py
      classical.py
      vqc.py
      quantum_kernel.py
      evaluation.py
      reporting.py
      history.py
    data/
      raw/
      processed/
    results/
      tables/
      figures/
    report.md
    tests/

## Interpretation rules

- CPU baseline performance is not duplicated or labelled as Processor A/B accuracy.
- Processor A/B accuracy comes from separate VQC runs using each configured topology and the same synthetic gate/readout noise rates.
- Depth and CX counts are taken from actual transpiled circuits, not inferred from the logical circuit.
- Routing overhead is reported as the explicit SWAP gates retained by the SWAP-visible routing pass and the additional CX count after native-basis compilation. These are related but not interchangeable metrics.
- Output-distribution fidelity is measured on the same fixed representative circuit under ideal and noisy Aer simulation. It is not a hardware-calibrated fidelity.
- A small, noisy simulation does not demonstrate quantum advantage or prove that a larger processor is faster.
- Processor C and custom Processor D are not included, as the Phase 1 guide reserves them for Phase 2 finalists.

## Reference

Qiskit Fall Fest 2026 — Phase 1 Master Guide, SRM University-AP, pages 1–7. Submission portal listed by the guide: https://qffsrmap2026.com/learning/hackathon.
