"""Rule-based insights and downloadable experiment report generation."""
from __future__ import annotations


def build_insights(models, data):
    if not models:
        return ["No model completed successfully; no comparison can be made."]
    names = list(models)
    best = max(names, key=lambda name: models[name]["metrics"]["F1 Score"])
    fastest_train = min(names, key=lambda name: models[name]["train_seconds"])
    fastest_inference = min(names, key=lambda name: models[name]["inference_seconds"])
    insights = [
        f'{best} achieved the highest F1-score on the primary held-out split ({models[best]["metrics"]["F1 Score"]:.3f}).',
        f'{fastest_train} had the shortest measured training time ({models[fastest_train]["train_seconds"]:.3f} seconds).',
        f'{fastest_inference} had the shortest measured inference time ({models[fastest_inference]["inference_seconds"]:.3f} seconds).',
    ]
    qk = models.get("Quantum Kernel + SVM")
    if qk:
        matrix = qk.get("kernel_train")
        labels = data.y_train
        if matrix is not None and len(labels) == matrix.shape[0]:
            same, different = [], []
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    (same if labels[i] == labels[j] else different).append(float(matrix[i, j]))
            if same and different:
                same_mean = sum(same) / len(same)
                diff_mean = sum(different) / len(different)
                if same_mean > diff_mean:
                    insights.append(f"The training kernel had higher mean within-class fidelity ({same_mean:.3f}) than between-class fidelity ({diff_mean:.3f}) for this split.")
                else:
                    insights.append(f"The training kernel did not show higher average within-class fidelity (within={same_mean:.3f}, between={diff_mean:.3f}) for this split.")
    if "Hybrid VQC" in models:
        insights.append("VQC runtime includes parameter optimization and simulator circuit evaluations; it is not directly comparable to a hardware runtime.")
    if qk:
        insights.append(f'Quantum-kernel construction took {qk.get("kernel_build_seconds", 0.0):.3f} seconds and is reported separately from SVM fitting.')
    insights.append("These measurements describe this dataset and configuration; they do not establish practical quantum advantage.")
    return insights


def build_markdown_report(bundle, rows_frame, insights):
    config = bundle["config"]
    lines = [
        "# QuantumClassify Lab — Experiment Report",
        "",
        f'Generated (UTC): {bundle["created_at"]}',
        "",
        "## Configuration",
        "",
        f'- Dataset: {config["dataset"]}',
        f'- Requested samples: {config["samples"]}',
        f'- Test fraction: {config["test_size"]:.0%}',
        f'- Qubits / retained features: {config["qubits"]}',
        f'- VQC layers: {config["vqc_layers"]}',
        f'- Optimizer: {config["optimizer"]}',
        f'- Optimizer iterations: {config["optimizer_iterations"]}',
        f'- VQC shots: {config["shots"]}',
        f'- Seeds: {", ".join(str(run["seed"]) for run in bundle["runs"])}',
        "",
        "## Results",
        "",
    ]
    columns = [c for c in ("Model", "Seed", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)", "Inference Time (s)", "Kernel Build Time (s)") if c in rows_frame.columns]
    if rows_frame.empty:
        lines.append("No model completed successfully.")
    elif columns:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in rows_frame[columns].iterrows():
            cells = []
            for column in columns:
                value = row[column]
                cells.append(f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
            lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Evidence-based observations", ""])
    lines.extend([f"- {sentence}" for sentence in insights])
    lines.extend([
        "",
        "## Fairness and limitations",
        "",
        "- All models in a seed use the same stratified train/test split.",
        "- Feature selection and scaling are fitted only on training data.",
        "- Test metrics are not used to tune model parameters.",
        "- Training, inference, and kernel construction are reported separately.",
        "- A small simulator experiment is not evidence of practical quantum advantage or hardware speedup.",
        "- Four-feature decision-boundary plots are two-dimensional slices with remaining features fixed.",
        "",
        "## Completed model runs",
        "",
    ])
    for run in bundle["runs"]:
        lines.append(f'- Seed {run["seed"]}: {", ".join(run["models"].keys()) or "no successful models"}')
        for model_name, error in run["failures"].items():
            lines.append(f'  - {model_name} failed: {error}')
    lines.append("")
    return "\n".join(lines)
