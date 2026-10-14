"""End-to-end QFF PS-Q1 benchmark on Processor A and Processor B.

Existing project model types are preserved: classical RBF-SVM, hybrid VQC,
and quantum-kernel SVM remain separate. The official PS-Q1 architecture
benchmark uses VQC on both processors and the RBF-SVM as its CPU baseline;
the quantum-kernel pipeline remains available in the original dashboard as a
supplementary experiment, rather than being mislabeled as PS-Q1.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.classical import train_classical
from src.circuits import build_ansatz, build_feature_map
from src.evaluation import evaluate
from src.metrics import circuit_metrics, output_distribution_fidelity
from src.problem import make_ps_q1_data
from src.processors import build_aer_backend, load_processor
from src.routing import compile_for_processor
from src.vqc import train_vqc
from src.visualization import plot_architecture_comparison, plot_coupling_graphs


ROOT = Path(__file__).resolve().parents[1]
RESULT_TABLES = ROOT / "results" / "tables"
RESULT_FIGURES = ROOT / "results" / "figures"


def _representative_probe(feature_map_name: str, ansatz_name: str):
    """Build a deterministic, parameter-bound VQC circuit for routing/noise metrics."""
    feature_map = build_feature_map(4, feature_map_name)
    ansatz = build_ansatz(4, reps=1, name=ansatz_name)
    circuit = feature_map.compose(ansatz)
    bindings = {parameter: float(0.13 * (index + 1)) for index, parameter in enumerate(circuit.parameters)}
    circuit = circuit.assign_parameters(bindings)
    circuit.measure_all()
    return circuit


def _measure_distribution_fidelity(compiled_circuit, processor, *, shots: int, seed: int) -> float:
    """Compare ideal and noisy counts for the same routed probe circuit."""
    ideal_backend = build_aer_backend(processor, include_noise=False, seed=seed)
    noisy_backend = build_aer_backend(processor, include_noise=True, seed=seed)
    ideal_counts = ideal_backend.run(compiled_circuit, shots=int(shots), seed_simulator=int(seed)).result().get_counts()
    noisy_counts = noisy_backend.run(compiled_circuit, shots=int(shots), seed_simulator=int(seed)).result().get_counts()
    return output_distribution_fidelity(ideal_counts, noisy_counts)


def _write_report(config: dict, model_rows: list[dict], architecture_rows: list[dict]) -> str:
    lines = [
        "# Qiskit Fall Fest 2026 — Phase 1 Architecture Report",
        "",
        "## Selected problem statement",
        "",
        "**PS-Q1: Variational Quantum Classifier (VQC).** The formal benchmark uses 200 generated samples, four features, three informative features, a binary target, seed 42 by default, and a stratified 80/20 split. Features/scalers are fitted on training data only.",
        "",
        "## Processor designs",
        "",
        "Processor A is a participant-designed five-qubit ring with one branch. Processor B is a participant-designed connected nine-qubit sparse graph with cross-links. Both use the same rz/sx/x/cx basis, optimization level 1, transpiler seed 42, 4096 shots, and synthetic noise rates (1-qubit p=0.001, CX p=0.005, readout p=0.01).",
        "",
        "## Model results (measured by the notebook/script)",
        "",
        "| Model | Execution | Accuracy | Precision | Recall | F1 | Train seconds | Inference seconds | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in model_rows:
        def numeric(key):
            value = row.get(key)
            return "" if value is None or pd.isna(value) else f"{float(value):.4f}"
        lines.append(
            f"| {row.get('Model', '')} | {row.get('Execution', '')} | {numeric('Accuracy')} | {numeric('Precision')} | {numeric('Recall')} | {numeric('F1 Score')} | {numeric('Training Time (s)')} | {numeric('Inference Time (s)')} | {row.get('Status', '')} |"
        )
    lines.extend([
        "",
        "## Processor A/B circuit metrics",
        "",
        "| Processor | Qubits | Logical depth | Compiled depth | Logical CX | Compiled CX | Explicit SWAPs | CX overhead | Output-distribution fidelity | VQC accuracy | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in architecture_rows:
        def show(key, digits=3):
            value = row.get(key)
            return "" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"
        lines.append(
            f"| {row.get('Processor', '')} | {row.get('Physical Qubits', '')} | {row.get('Logical Circuit Depth', '')} | {row.get('Compiled Circuit Depth', '')} | {row.get('Logical CX Count', '')} | {row.get('Compiled CX Count', '')} | {row.get('SWAP Routing Count', '')} | {row.get('Routing CX Overhead', '')} | {show('Output-distribution Fidelity')} | {show('Accuracy', 4)} | {row.get('Status', '')} |"
        )
    lines.extend(["", "## Observed trade-offs", ""])
    by_processor = {row.get("Processor"): row for row in architecture_rows}
    comparisons = [
        ("Compiled Circuit Depth", "compiled circuit depth"),
        ("Compiled CX Count", "compiled CX count"),
        ("SWAP Routing Count", "explicit routing SWAP count"),
        ("Routing CX Overhead", "routing CX overhead"),
        ("Output-distribution Fidelity", "ideal/noisy output-distribution fidelity"),
        ("Accuracy", "held-out VQC accuracy"),
        ("F1 Score", "held-out VQC F1 score"),
    ]
    if all(name in by_processor for name in ("Processor A", "Processor B")):
        a_row, b_row = by_processor["Processor A"], by_processor["Processor B"]
        for key, label in comparisons:
            a_value, b_value = a_row.get(key), b_row.get(key)
            if a_value is None or b_value is None or pd.isna(a_value) or pd.isna(b_value):
                continue
            a_value, b_value = float(a_value), float(b_value)
            delta = b_value - a_value
            if abs(delta) < 1e-12:
                lines.append(f"- Processor A and Processor B had the same measured {label} ({a_value:.4f}) for this configuration.")
            else:
                direction = "lower" if delta < 0 else "higher"
                lines.append(f"- Processor B had {direction} {label} than Processor A ({a_value:.4f} on A; {b_value:.4f} on B). This is specific to this circuit, seed, and configuration.")
    else:
        lines.append("- A complete Processor A/B pair was not available in this run, so no comparative conclusion is drawn.")

    ideal_vqc = next((row for row in model_rows
                      if row.get("Model") == "Hybrid VQC" and row.get("Execution") == "Ideal simulator"
                      and row.get("Status") == "ok"), None)
    if ideal_vqc and ideal_vqc.get("Accuracy") is not None:
        ideal_accuracy = float(ideal_vqc["Accuracy"])
        for processor_name in ("Processor A", "Processor B"):
            noisy_row = next((row for row in model_rows
                              if row.get("Model") == "Hybrid VQC" and row.get("Execution") == processor_name
                              and row.get("Status") == "ok"), None)
            if noisy_row and noisy_row.get("Accuracy") is not None:
                change = float(noisy_row["Accuracy"]) - ideal_accuracy
                lines.append(
                    f"- The {processor_name} VQC accuracy differed from the ideal-simulator VQC by {change:+.4f} "
                    f"(ideal {ideal_accuracy:.4f}; {processor_name} {float(noisy_row['Accuracy']):.4f}). "
                    "This reflects the complete training-and-inference pipeline under the chosen noise model, not an isolated post-training noise test."
                )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- The SVM baseline runs on a CPU; it is not mapped to a quantum processor.",
        "- Processor A/B classification scores come from the same VQC type, same dataset split, same feature map/ansatz, seed, and iteration budget with identical synthetic per-gate/readout noise rates. The connectivity and routing target differ.",
        "- Output-distribution fidelity is the squared Bhattacharyya coefficient between measured ideal/noisy output distributions of one fixed representative circuit. It is not quantum-state fidelity and not measured hardware fidelity.",
        "- A small simulator study does not establish quantum advantage. Simulator timings are not real-device wall-clock benchmarks.",
        "- All metric values come from the run that produced this report. A failed row is not replaced by an assumed score.",
        "",
        "## Reproducibility configuration",
        "",
        f"- Seed: {config['seed']}",
        f"- Optimizer: {config['optimizer']}",
        f"- Optimizer iterations: {config['maxiter']}",
        f"- Shots: {config['shots']}",
        f"- Feature map: {config['feature_map']}",
        f"- Ansatz: {config['ansatz']}",
        "",
        "## Supplementary models retained",
        "",
        "The original Streamlit app still contains all three models: classical RBF-SVM, hybrid VQC, and fidelity quantum-kernel SVM. The quantum-kernel pipeline is preserved as a supplementary dashboard experiment, not used as a substitute for the official PS-Q1 deliverable.",
        "",
    ])
    report = "\n".join(lines)
    (ROOT / "report.md").write_text(report, encoding="utf-8")
    return report


def run_ps_q1_benchmark(*, shots: int = 4096, maxiter: int = 20, seed: int = 42,
                        feature_map_name: str = "ZZ entangling", ansatz_name: str = "Real Amplitudes"):
    """Run PS-Q1 VQC on ideal simulator, Processor A, and Processor B.

    Outputs are written only from measured runs. If a device-specific training
    run fails, its row is marked failed and no accuracy is fabricated.
    """
    RESULT_TABLES.mkdir(parents=True, exist_ok=True)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    data = make_ps_q1_data(seed=seed)
    config_record = {
        "problem_statement": "PS-Q1",
        "samples": 200,
        "n_features": 4,
        "n_informative": 3,
        "test_fraction": 0.20,
        "seed": int(seed),
        "shots": int(shots),
        "maxiter": int(maxiter),
        "optimizer": "COBYLA",
        "feature_map": feature_map_name,
        "ansatz": ansatz_name,
    }
    model_rows: list[dict] = []
    architecture_rows: list[dict] = []

    classical = train_classical(data.X_train, data.y_train, data.X_test, seed=seed)
    baseline_metrics = evaluate(data.y_test, classical["predictions"])
    model_rows.append({
        "Model": "Classical RBF SVM",
        "Execution": "CPU baseline (not processor-mapped)",
        **{key: baseline_metrics[key] for key in ("Accuracy", "Precision", "Recall", "F1 Score")},
        "Training Time (s)": float(classical["train_seconds"]),
        "Inference Time (s)": float(classical["inference_seconds"]),
        "Status": "ok",
    })

    execution_targets = [
        ("Ideal simulator", None),
        ("Processor A", load_processor("A")),
        ("Processor B", load_processor("B")),
    ]
    for execution_name, processor_config in execution_targets:
        base_row = {"Model": "Hybrid VQC", "Execution": execution_name, "Processor": execution_name, "Status": "failed"}
        try:
            result = train_vqc(
                data.X_train_q, data.y_train, data.X_test_q,
                maxiter=int(maxiter), reps=1, optimizer_name="COBYLA",
                shots=int(shots), seed=int(seed), noise_strength=0.0,
                feature_map_name=feature_map_name, ansatz_name=ansatz_name,
                processor_config=processor_config,
            )
            measured = evaluate(data.y_test, result["predictions"])
            base_row.update({key: measured[key] for key in ("Accuracy", "Precision", "Recall", "F1 Score")})
            base_row.update({
                "Training Time (s)": float(result["train_seconds"]),
                "Inference Time (s)": float(result["inference_seconds"]),
                "Status": "ok",
            })
        except Exception as exc:
            base_row["Error"] = f"{type(exc).__name__}: {exc}"
        model_rows.append(base_row)

        if processor_config is None:
            continue

        try:
            logical_probe = _representative_probe(feature_map_name, ansatz_name)
            compilation = compile_for_processor(logical_probe, processor_config)
            arch = circuit_metrics(compilation)
            arch["Execution"] = execution_name
            arch["Accuracy"] = base_row.get("Accuracy")
            arch["VQC Training Time (s)"] = base_row.get("Training Time (s)")
            arch["VQC Inference Time (s)"] = base_row.get("Inference Time (s)")
            arch["Error"] = base_row.get("Error", "")
            arch["Output-distribution Fidelity"] = _measure_distribution_fidelity(
                compilation["compiled"], processor_config, shots=int(shots), seed=int(seed)
            )
            arch["Status"] = base_row["Status"]
        except Exception as exc:
            arch = {
                "Processor": execution_name,
                "Physical Qubits": int(processor_config["num_qubits"]),
                "Execution": execution_name,
                "Accuracy": base_row.get("Accuracy"),
                "Status": "failed",
                "Error": f"{type(exc).__name__}: {exc}",
            }
        architecture_rows.append(arch)

    model_frame = pd.DataFrame(model_rows)
    architecture_frame = pd.DataFrame(architecture_rows)
    model_frame.to_csv(RESULT_TABLES / "model_comparison.csv", index=False)
    architecture_frame.to_csv(RESULT_TABLES / "processor_architecture_comparison.csv", index=False)
    (RESULT_TABLES / "benchmark_summary.json").write_text(
        json.dumps({"configuration": config_record, "models": model_rows, "architecture": architecture_rows}, indent=2, default=str),
        encoding="utf-8",
    )

    configs = [load_processor("A"), load_processor("B")]
    graph_figure = plot_coupling_graphs(configs)
    graph_figure.savefig(RESULT_FIGURES / "processor_coupling_graphs.png", dpi=160)
    plt.close(graph_figure)
    comparison_figure = plot_architecture_comparison(architecture_rows)
    comparison_figure.savefig(RESULT_FIGURES / "processor_metric_comparison.png", dpi=160)
    plt.close(comparison_figure)
    report = _write_report(config_record, model_rows, architecture_rows)
    return {"config": config_record, "models": model_frame, "architecture": architecture_frame, "report": report}


if __name__ == "__main__":
    output = run_ps_q1_benchmark()
    print("Model comparison:")
    print(output["models"].to_string(index=False))
    print("\nProcessor A/B architecture comparison:")
    print(output["architecture"].to_string(index=False))
    print("\nWrote measured tables under results/tables/, figures under results/figures/, and report.md.")
