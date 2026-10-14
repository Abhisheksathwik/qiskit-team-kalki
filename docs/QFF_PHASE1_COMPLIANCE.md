# QFF 2026 Phase 1 Compliance Checklist

This checklist is based on the uploaded Qiskit Fall Fest 2026 Phase 1 Master Guide. It distinguishes implemented code from a full benchmark run or external submission, neither of which is assumed to have happened just because the files exist.

## Official task scope

- [x] One official problem statement is documented: **PS-Q1 — Variational Quantum Classifier (VQC)**.
- [x] Existing project model types are preserved: classical RBF-SVM, hybrid VQC, and quantum-kernel SVM remain separate. The formal PS-Q1 architecture benchmark uses VQC on A/B and SVM as a CPU baseline; the quantum-kernel model remains supplementary in the existing app.
- [x] No Processor C or custom Processor D implementation is added.

## Processor A / Processor B design

- [x] Participant-defined Processor A JSON with 5 physical qubits and a connected coupling graph.
- [x] Participant-defined Processor B JSON with 9 physical qubits and a connected coupling graph.
- [x] Both processor files specify `rz`, `sx`, `x`, `cx` as basis gates.
- [x] Both processor files specify optimization level 1, transpiler seed 42, and 4096 default shots.
- [x] Both processors use the official synthetic noise rates: one-qubit depolarizing probability 0.001, CX depolarizing probability 0.005, and readout error probability 0.01.
- [x] Qiskit Aer noise model construction applies the same rates to both processor targets.
- [x] Routing/transpilation module compiles the representative VQC circuit against both coupling graphs.
- [x] Circuit metrics include logical/compiled depth, logical/compiled CX counts, explicit routing SWAP count, CX overhead, and ideal/noisy measured output-distribution fidelity.

## Official PS-Q1 task

- [x] Programmatically generated binary dataset with 200 samples and four features, including three informative features.
- [x] Reproducible seed 42 and stratified 80/20 train/test split.
- [x] Standardization and angle scaling are fitted on the training split only.
- [x] VQC is run on ideal simulation, Processor A and Processor B using the same circuit family, configuration, seed and held-out split.
- [x] Classical RBF-SVM provides a CPU baseline on the same split.
- [x] The explicit `n_redundant=0` parameter resolves a conflict between `n_features=4`, `n_informative=3`, and scikit-learn's default `n_redundant=2`.
- [x] Notebook and script export measured results, figures, and a report; failed runs stay marked as failed instead of receiving fabricated scores.

## Repository structure and reproducibility

- [x] `main.ipynb`, `requirements.txt` with exact direct-dependency pins, `processors/`, `src/`, `data/raw/`, `data/processed/`, `results/tables/`, `results/figures/`, `report.md`, and `assets/` directory.
- [x] Unit/smoke tests cover the dataset, processor configs, noise model, routing and processor-specific VQC sampler.
- [x] Latest verified GitHub Actions workflow passed.
- [ ] Execute the full official benchmark at 4096 shots and retain its generated CSV/JSON/PNG outputs. A lower-shot CI smoke test is not a substitute for this final run.
- [ ] Inspect the generated `report.md` and result tables after the full run; record any failed rows transparently.
- [ ] Submit the public GitHub URL through the QFF portal. The repository update itself does not submit the project.

## Measurement terminology

Output-distribution fidelity is defined here as the squared Bhattacharyya coefficient between ideal and noisy measurement distributions of a fixed representative circuit. It is not quantum-state fidelity or physical hardware fidelity. Simulator run time is not real-device wall-clock time.
