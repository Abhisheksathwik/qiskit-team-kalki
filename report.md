# Qiskit Fall Fest 2026 — Phase 1 Architecture Report

## Status before benchmark execution

The official benchmark code and reproducible notebook are present. **No benchmark measurements are embedded in this template.** Run `python -m src.architecture_benchmark` or execute `main.ipynb` to generate the tables, plots, and measured report.

## Selected problem statement

**PS-Q1 — Variational Quantum Classifier (VQC).** The formal target is 200 locally generated samples, four features, three informative features, binary labels, and an 80/20 held-out split. Default seed: 42.

## Processor definitions

- **Processor A:** participant-designed connected five-qubit ring-with-branch graph.
- **Processor B:** participant-designed connected sparse nine-qubit graph with cross-links.
- Both use `rz`, `sx`, `x`, `cx`; optimization level 1; transpiler seed 42; 4096 shots; one-qubit depolarizing probability 0.001; CX depolarizing probability 0.005; readout error probability 0.01.

See `processors/processor_A.json` and `processors/processor_B.json` for exact edges and design rationale.

## Architectural trade-off (design rationale, not a measured result)

Processor A has five physical qubits and five coupling edges, giving a compact design with a four-node cycle and one attached branch. Processor B has nine physical qubits and twelve coupling edges, including three cross-links that introduce alternative routes. The design hypothesis is that Processor B may provide more routing choices for some interactions, while its larger physical register does not by itself guarantee lower depth, fewer CX gates, fewer SWAPs, better noisy accuracy, or higher output-distribution fidelity. The experiment must establish those outcomes from the actual compiled circuits and simulator runs.

## Measured outputs

The benchmark writes `results/tables/model_comparison.csv`, `results/tables/processor_architecture_comparison.csv`, `results/tables/benchmark_summary.json`, and figures under `results/figures/`. The report is rewritten with the observations and results from the run. Do not present an unexecuted template as evidence or fill missing measurements with estimates.

## Metrics

The processor comparison includes logical and compiled circuit depth, logical and compiled CX count, explicit routing SWAP count, routing CX overhead, held-out VQC accuracy, and ideal/noisy output-distribution fidelity for a fixed representative circuit. The classical output-distribution fidelity is defined as the squared Bhattacharyya coefficient between measurement distributions; it is not state fidelity.

## Limitations

The classical RBF-SVM is a CPU baseline and does not map to either quantum graph. The current quantum-kernel SVM remains available in the original interactive app as a supplementary model. Simulator timings are not hardware timings, and no result from this small benchmark alone establishes quantum advantage.
