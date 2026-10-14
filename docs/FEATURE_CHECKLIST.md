# Deck-to-Implementation Checklist

This checklist maps the QuantumClassify Lab presentation brief to repository features. A checked box means the feature has been implemented in code; it does not by itself mean that a hardware run or an external deployment has been performed.

## Core benchmark

- [x] Classical RBF SVM baseline.
- [x] Hybrid VQC using a feature map, trainable ansatz, and classical optimizer.
- [x] Fidelity quantum kernel followed by classical SVM training.
- [x] Two Moons and Breast Cancer Wisconsin datasets.
- [x] Two- and four-feature/qubit configurations.
- [x] Stratified common train/test split per seed.
- [x] Fit scaling and feature selection on training data only.
- [x] Fixed/repeatable seeds and option for multiple independent runs.
- [x] Report accuracy, precision, recall, F1, training time, and inference time.
- [x] Report quantum kernel construction time separately.
- [x] Confusion matrices, metrics comparison, decision boundaries, VQC loss, quantum-kernel heatmap, circuit diagrams, training-time chart.
- [x] Plain-language findings based on the measured run output.
- [x] Report download (Markdown), metric CSV export, local history JSON export.
- [x] Configurable dashboard with Overview, VQC, Quantum Kernel, Report & History, and Methodology tabs.
- [x] Bounded sample sizes to keep simulator experiments appropriate for a demo.
- [x] Local automated tests and GitHub Actions compile/test checks.
- [x] Installation, testing, architecture, and deployment instructions.

## Research fairness and limitations

- [x] The three approaches stay separate; quantum kernel learning is not mislabeled as VQC.
- [x] No hard-coded benchmark scores or fabricated outcomes.
- [x] Reports warn that small simulator runs do not demonstrate quantum advantage.
- [x] Four-feature decision boundaries are labelled/treated as 2-D slices.
- [x] Kernel computation cost is highlighted.

## Deliberately not claimed as completed

These are extension items described in the presentation as future work or optional enhancements. They need additional credentials, hardware access, or a separate implementation and must not be presented as already delivered.

- [ ] Real IBM Quantum hardware backend execution (requires user-provided IBM Quantum credentials and a compatible backend).
- [ ] Hardware-aware transpilation and backend-specific validation.
- [ ] An explicit configurable noisy-channel model applied consistently to both quantum classifiers.
- [ ] Shot-noise / noise-strength sweeps and statistical comparison against ideal simulation.
- [ ] Automated hyperparameter, feature-map, and ansatz sweeps.
- [ ] Persistent database or cloud object store for history on ephemeral hosting.
- [ ] Publicly deployed Streamlit app (needs a deployment account and deployment action).
- [ ] Large multi-dataset benchmark and statistical significance analysis.

## Final release checks

- [ ] Latest GitHub Actions run is green after the newest commit.
- [ ] Run one local demo with classical SVM only.
- [ ] Run one local demo with all three models at 2 qubits and modest iterations.
- [ ] Open all dashboard tabs and inspect downloaded reports.
- [ ] Perform a fresh clone and installation on the presentation laptop.
- [ ] If hardware integration is added, run an authenticated end-to-end backend test.
