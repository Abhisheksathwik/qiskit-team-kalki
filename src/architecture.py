"""Lightweight, fixed-split quantum architecture sweep for model selection."""
from __future__ import annotations

from src.circuits import ANSATZ_OPTIONS, FEATURE_MAP_OPTIONS
from src.evaluation import evaluate


def run_architecture_sweep(data, config):
    """Compare small candidate architecture choices on one fixed split.

    This is an exploratory sweep, not exhaustive hyperparameter optimization.
    The test split is used only to score each frozen candidate after fitting.
    """
    from src.quantum_kernel import train_quantum_kernel
    from src.vqc import train_vqc

    rows = []
    base_seed = int(data.seed)
    models = config.get("models", {})
    quantum_selected = models.get("Hybrid VQC", False) or models.get("Quantum Kernel + SVM", False)
    if not quantum_selected:
        return rows

    # Keep the optional sweep bounded independently of the primary model's iteration setting.
    sweep_iterations = max(20, min(int(config.get("optimizer_iterations", 20)), 25))

    if models.get("Hybrid VQC", False):
        for feature_map_name in FEATURE_MAP_OPTIONS:
            for ansatz_name in ANSATZ_OPTIONS:
                label = f"VQC · {feature_map_name} · {ansatz_name}"
                try:
                    result = train_vqc(
                        data.X_train_q, data.y_train, data.X_test_q,
                        maxiter=sweep_iterations,
                        reps=int(config.get("vqc_layers", 1)),
                        optimizer_name=config.get("optimizer", "COBYLA"),
                        shots=int(config.get("shots", 512)),
                        seed=base_seed,
                        noise_strength=float(config.get("noise_strength", 0.0)),
                        feature_map_name=feature_map_name,
                        ansatz_name=ansatz_name,
                    )
                    metrics = evaluate(data.y_test, result["predictions"])
                    rows.append({
                        "Configuration": label,
                        "Model Family": "Hybrid VQC",
                        "Feature Map": feature_map_name,
                        "Ansatz": ansatz_name,
                        "Accuracy": metrics["Accuracy"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1 Score": metrics["F1 Score"],
                        "Training Time (s)": result["train_seconds"],
                        "Inference Time (s)": result["inference_seconds"],
                        "Shots": int(config.get("shots", 512)),
                        "Noise (1q)": float(config.get("noise_strength", 0.0)),
                        "Seed": base_seed,
                        "Status": "ok",
                        "Error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "Configuration": label,
                        "Model Family": "Hybrid VQC",
                        "Feature Map": feature_map_name,
                        "Ansatz": ansatz_name,
                        "Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None,
                        "Training Time (s)": None, "Inference Time (s)": None,
                        "Shots": int(config.get("shots", 512)),
                        "Noise (1q)": float(config.get("noise_strength", 0.0)),
                        "Seed": base_seed, "Status": "failed",
                        "Error": f"{type(exc).__name__}: {exc}",
                    })

    if models.get("Quantum Kernel + SVM", False):
        for feature_map_name in FEATURE_MAP_OPTIONS:
            label = f"Quantum Kernel · {feature_map_name}"
            try:
                result = train_quantum_kernel(
                    data.X_train_q, data.y_train, data.X_test_q,
                    reps=1,
                    seed=base_seed,
                    max_train_samples=105,
                    shots=int(config.get("kernel_shots", 512)),
                    noise_strength=float(config.get("noise_strength", 0.0)),
                    feature_map_name=feature_map_name,
                )
                metrics = evaluate(data.y_test, result["predictions"])
                rows.append({
                    "Configuration": label,
                    "Model Family": "Quantum Kernel + SVM",
                    "Feature Map": feature_map_name,
                    "Ansatz": "Not applicable",
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1 Score": metrics["F1 Score"],
                    "Training Time (s)": result["train_seconds"],
                    "Inference Time (s)": result["inference_seconds"],
                    "Kernel Build Time (s)": result["kernel_build_seconds"],
                    "Shots": int(config.get("kernel_shots", 512)),
                    "Noise (1q)": float(config.get("noise_strength", 0.0)),
                    "Seed": base_seed,
                    "Status": "ok",
                    "Error": "",
                })
            except Exception as exc:
                rows.append({
                    "Configuration": label,
                    "Model Family": "Quantum Kernel + SVM",
                    "Feature Map": feature_map_name,
                    "Ansatz": "Not applicable",
                    "Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None,
                    "Training Time (s)": None, "Inference Time (s)": None, "Kernel Build Time (s)": None,
                    "Shots": int(config.get("kernel_shots", 512)),
                    "Noise (1q)": float(config.get("noise_strength", 0.0)),
                    "Seed": base_seed, "Status": "failed",
                    "Error": f"{type(exc).__name__}: {exc}",
                })

    return rows
