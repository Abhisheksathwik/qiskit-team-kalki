"""QuantumClassify Lab: interactive, reproducible QML benchmark."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.classical import train_classical
from src.data import PreparedData, load_data
from src.evaluation import evaluate
from src.history import load_history, save_history
from src.reporting import build_insights, build_markdown_report
from src.visualization import (
    confusion_matrix_plot,
    decision_boundary,
    kernel_heatmap,
    metrics_comparison,
    training_loss,
    training_time_chart,
)

st.set_page_config(
    page_title="QuantumClassify Lab",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1450px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; border: 1px solid #273449;
           background: linear-gradient(120deg, rgba(124,58,237,.22), rgba(14,165,233,.10));}
    .eyebrow {letter-spacing: .16em; text-transform: uppercase; font-size: .76rem; opacity: .75;}
    .hero h1 {margin: .35rem 0 .5rem 0; font-size: 2.45rem;}
    .small-note {font-size: .88rem; opacity: .78;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">QML benchmark · experimental research platform</div>
      <h1>⚛️ QuantumClassify Lab</h1>
      Compare a classical RBF-SVM with a hybrid variational quantum classifier and a
      fidelity quantum-kernel SVM — on the same held-out samples, with costs reported honestly.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

with st.sidebar:
    st.markdown("## Experiment configuration")
    dataset_name = st.selectbox("Dataset", ["Two Moons", "Breast Cancer"])
    sample_count = st.slider("Total samples", min_value=60, max_value=120, value=80, step=20)
    test_fraction = st.slider("Held-out test fraction", min_value=0.15, max_value=0.30, value=0.20, step=0.05)
    qubits = st.radio("Qubits / retained features", [2, 4], horizontal=True)
    depth = st.slider("VQC ansatz layers", min_value=1, max_value=3, value=1)
    optimizer_name = st.selectbox("Classical optimizer", ["COBYLA", "SPSA"])
    optimizer_iterations = st.slider("Optimizer iterations", min_value=10, max_value=80, value=30, step=10)
    shots = st.select_slider("Measurement shots (VQC)", options=[256, 512, 1024, 2048], value=1024)
    base_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    repeats = st.selectbox("Independent seed runs", [1, 3], index=0, help="Three runs report mean and standard deviation but can take considerably longer.")
    st.markdown("---")
    st.markdown("### Models to run")
    run_classical = st.checkbox("Classical RBF SVM", value=True)
    run_vqc = st.checkbox("Hybrid VQC", value=True)
    run_kernel = st.checkbox("Quantum Kernel + SVM", value=True)
    run_experiment = st.button("🚀 Run experiment", type="primary", use_container_width=True)
    st.caption("Quantum-kernel construction scales quadratically with training-set size. This demo caps total samples at 120.")

CONFIG = {
    "dataset": dataset_name,
    "samples": int(sample_count),
    "test_size": float(test_fraction),
    "qubits": int(qubits),
    "vqc_layers": int(depth),
    "optimizer": optimizer_name,
    "optimizer_iterations": int(optimizer_iterations),
    "shots": int(shots),
    "seed": int(base_seed),
    "repeats": int(repeats),
    "models": {
        "Classical RBF SVM": bool(run_classical),
        "Hybrid VQC": bool(run_vqc),
        "Quantum Kernel + SVM": bool(run_kernel),
    },
}


def _jsonable_row(model_name, metrics, model_result, run_seed, cfg):
    row = {"Model": model_name, "Seed": int(run_seed), "Dataset": cfg["dataset"]}
    for key in ("Accuracy", "Precision", "Recall", "F1 Score"):
        row[key] = float(metrics[key])
    row["Training Time (s)"] = float(model_result["train_seconds"])
    row["Inference Time (s)"] = float(model_result["inference_seconds"])
    if "kernel_build_seconds" in model_result:
        row["Kernel Build Time (s)"] = float(model_result["kernel_build_seconds"])
    else:
        row["Kernel Build Time (s)"] = 0.0
    return row


def _fit_seed(cfg, seed):
    data = load_data(
        name=cfg["dataset"],
        samples=cfg["samples"],
        test_size=cfg["test_size"],
        seed=int(seed),
        n_qubits=cfg["qubits"],
    )
    outputs = {}
    failures = {}

    if cfg["models"]["Classical RBF SVM"]:
        try:
            result = train_classical(data.X_train, data.y_train, data.X_test, seed=seed)
            outputs["Classical RBF SVM"] = {
                **result,
                "metrics": evaluate(data.y_test, result["predictions"]),
            }
        except Exception as exc:
            failures["Classical RBF SVM"] = f"{type(exc).__name__}: {exc}"

    if cfg["models"]["Hybrid VQC"]:
        try:
            from src.vqc import train_vqc
            result = train_vqc(
                data.X_train_q, data.y_train, data.X_test_q,
                maxiter=cfg["optimizer_iterations"], reps=cfg["vqc_layers"],
                optimizer_name=cfg["optimizer"], shots=cfg["shots"], seed=seed,
            )
            outputs["Hybrid VQC"] = {
                **result,
                "metrics": evaluate(data.y_test, result["predictions"]),
            }
        except Exception as exc:
            failures["Hybrid VQC"] = f"{type(exc).__name__}: {exc}"

    if cfg["models"]["Quantum Kernel + SVM"]:
        try:
            from src.quantum_kernel import train_quantum_kernel
            result = train_quantum_kernel(
                data.X_train_q, data.y_train, data.X_test_q, reps=1, seed=seed,
                max_train_samples=100,
            )
            outputs["Quantum Kernel + SVM"] = {
                **result,
                "metrics": evaluate(data.y_test, result["predictions"]),
            }
        except Exception as exc:
            failures["Quantum Kernel + SVM"] = f"{type(exc).__name__}: {exc}"

    return {"data": data, "models": outputs, "failures": failures, "seed": int(seed)}


if run_experiment:
    if not any(CONFIG["models"].values()):
        st.error("Select at least one model before running the experiment.")
    else:
        all_runs = []
        rows = []
        with st.status("Running benchmark", expanded=True) as status:
            for repeat_index in range(CONFIG["repeats"]):
                seed = CONFIG["seed"] + repeat_index * 17
                st.write(f"Preparing identical train/test split for seed {seed}…")
                run_result = _fit_seed(CONFIG, seed)
                all_runs.append(run_result)
                for model_name, model_result in run_result["models"].items():
                    rows.append(_jsonable_row(model_name, model_result["metrics"], model_result, seed, CONFIG))
                for model_name, error in run_result["failures"].items():
                    st.warning(f"{model_name} did not complete for seed {seed}: {error}")
            status.update(label="Experiment finished", state="complete", expanded=False)

        st.session_state["qcl_run_bundle"] = {
            "config": CONFIG,
            "runs": all_runs,
            "rows": rows,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        summary = pd.DataFrame(rows)
        insights = build_insights(all_runs[0]["models"], all_runs[0]["data"])
        record = {
            "created_at": st.session_state["qcl_run_bundle"]["created_at"],
            "config": CONFIG,
            "summary": summary.to_dict(orient="records"),
            "insights": insights,
        }
        save_history(record)

if "qcl_run_bundle" not in st.session_state:
    st.session_state["qcl_run_bundle"] = None

bundle = st.session_state["qcl_run_bundle"]
tabs = st.tabs(["Overview", "VQC", "Quantum Kernel", "Report & History", "Methodology"])

if bundle is None:
    with tabs[0]:
        st.subheader("Your experiment workspace")
        st.write("Configure a run in the sidebar, then select Run experiment. Results stay in this session while you explore the tabs.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Models", "3", "1 classical · 2 quantum")
        c2.metric("Quantum feature options", "2 or 4", "qubits")
        c3.metric("Evaluation", "Held-out", "stratified split")
        st.markdown("### The three approaches")
        cards = st.columns(3)
        cards[0].markdown("#### Classical RBF SVM\nA reproducible CPU baseline with an implicit nonlinear kernel.")
        cards[1].markdown("#### Hybrid VQC\nA trainable quantum circuit updated by a classical optimizer.")
        cards[2].markdown("#### Quantum Kernel + SVM\nA fidelity matrix supplies similarities to a classical precomputed-kernel SVM.")
        st.info("No fabricated benchmark values are displayed. Results appear only after models are run.")
    with tabs[1]:
        st.write("Run an experiment to inspect the VQC feature map, trainable ansatz, optimization trace, metrics, and confusion matrix.")
    with tabs[2]:
        st.write("Run an experiment to inspect the fidelity kernel matrix, feature map, confusion matrix, and kernel computation cost.")
    with tabs[3]:
        st.subheader("Experiment history")
        history = load_history()
        if history:
            st.dataframe(pd.DataFrame([{
                "Date (UTC)": item.get("created_at"),
                "Dataset": item.get("config", {}).get("dataset"),
                "Qubits": item.get("config", {}).get("qubits"),
                "Seeds": item.get("config", {}).get("repeats"),
                "Models completed": len(item.get("summary", [])),
            } for item in history]), use_container_width=True)
        else:
            st.caption("No saved runs yet. Local history is created after the first experiment.")
    with tabs[4]:
        st.markdown("""
        **Research question:** How do hybrid VQC and quantum-kernel learning compare with classical RBF-SVM classification in prediction quality and computational cost?

        **Fairness rules:** stratified split; transformations fitted on training data only; shared features, seed, and held-out rows; timing reported separately; no presumed quantum advantage.

        **Limitations:** small datasets and simulators do not establish practical quantum advantage. Kernel construction scales quadratically in the number of training samples.
        """)
    st.stop()

primary = bundle["runs"][0]
data: PreparedData = primary["data"]
primary_models = primary["models"]
primary_failures = primary["failures"]
rows = bundle["rows"]
rows_frame = pd.DataFrame(rows)
if rows_frame.empty:
    st.error("All selected models failed. See the model errors below and verify installation requirements.")
    for name, err in primary_failures.items():
        st.code(f"{name}: {err}")
    st.stop()

aggregate_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)", "Inference Time (s)", "Kernel Build Time (s)"]
agg = rows_frame.groupby("Model")[aggregate_metrics].agg(["mean", "std"]).reset_index()
agg.columns = [
    "Model" if column[0] == "Model" else (f"{column[0]} mean" if column[1] == "mean" else f"{column[0]} std")
    for column in agg.columns.to_flat_index()
]
mean_metric_dict = {
    model_name: {
        metric: float(rows_frame.loc[rows_frame["Model"] == model_name, metric].mean())
        for metric in ("Accuracy", "Precision", "Recall", "F1 Score")
    }
    for model_name in rows_frame["Model"].unique()
}
training_means = {
    name: float(rows_frame.loc[rows_frame["Model"] == name, "Training Time (s)"].mean())
    for name in rows_frame["Model"].unique()
}
insights = build_insights(primary_models, data)

with tabs[0]:
    st.subheader("Experiment scoreboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dataset", bundle["config"]["dataset"])
    m2.metric("Train samples", data.train_size)
    m3.metric("Test samples", data.test_count)
    m4.metric("Qubits", data.n_qubits)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Training Time (s)", "Inference Time (s)", "Kernel Build Time (s)"]
    # Display means and standard deviations when the repeat option is enabled.
    if bundle["config"]["repeats"] > 1:
        st.caption("Values below are mean ± sample standard deviation across independent seeds.")
        st.dataframe(agg, use_container_width=True)
    else:
        st.dataframe(rows_frame[display_cols], use_container_width=True)

    st.pyplot(metrics_comparison(mean_metric_dict), use_container_width=True)
    st.pyplot(training_time_chart(training_means), use_container_width=True)

    st.subheader("Confusion matrices")
    confusion_cols = st.columns(min(3, max(1, len(primary_models))))
    for column, (name, result) in zip(confusion_cols, primary_models.items()):
        with column:
            st.pyplot(confusion_matrix_plot(data.y_test, result["predictions"], name), use_container_width=True)
    st.subheader("Decision boundaries")
    st.caption("For four-feature experiments, this is a two-dimensional slice with the remaining scaled features fixed at zero.")
    boundary_cols = st.columns(min(3, max(1, len(primary_models))))
    for column, (name, result) in zip(boundary_cols, primary_models.items()):
        model = result["model"]
        def make_predictor(model_name, model_result):
            def _predict(grid2):
                full = np.zeros((len(grid2), data.X_train.shape[1]), dtype=float)
                full[:, :2] = grid2
                if model_name == "Classical RBF SVM":
                    return model_result["model"].predict(full)
                angle_grid = data.angle_scaler.transform(full)
                if model_name == "Hybrid VQC":
                    return np.asarray(model_result["model"].predict(angle_grid)).astype(int).ravel()
                # The precomputed-kernel SVC needs only its support-vector similarities
                # for a decision function. This avoids needlessly predicting against all
                # training rows for every boundary-grid point.
                from sklearn.svm import SVC
                qmodel = model_result["model"]
                from qiskit_machine_learning.kernels import FidelityQuantumKernel
                kernel = FidelityQuantumKernel(feature_map=model_result["feature_map"])
                support_indices = qmodel.support_
                K_grid_support = kernel.evaluate(angle_grid, data.X_train_q[support_indices])
                decision = K_grid_support @ qmodel.dual_coef_[0] + qmodel.intercept_[0]
                return np.where(decision >= 0, qmodel.classes_[1], qmodel.classes_[0])
            return _predict
        with column:
            try:
                # Quantum-kernel boundary evaluation is expensive; use a coarser grid for this view.
                resolution = 14 if name == "Quantum Kernel + SVM" else 70
                st.pyplot(decision_boundary(
                    make_predictor(name, result), data.X_train, data.y_train,
                    data.X_test, data.y_test, name, resolution=resolution
                ), use_container_width=True)
            except Exception as exc:
                st.warning(f"Could not draw {name} boundary: {type(exc).__name__}: {exc}")

with tabs[1]:
    if "Hybrid VQC" not in primary_models:
        st.info("VQC was not selected or did not complete in the primary seed.")
        if "Hybrid VQC" in primary_failures:
            st.code(primary_failures["Hybrid VQC"])
    else:
        model = primary_models["Hybrid VQC"]
        st.subheader("Hybrid variational quantum classifier")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f'{model["metrics"]["Accuracy"]:.3f}')
        b.metric("F1 score", f'{model["metrics"]["F1 Score"]:.3f}')
        c.metric("Qubits", model["n_qubits"])
        d.metric("Shots", model["shots"])
        st.markdown("#### Quantum feature map circuit")
        try:
            st.pyplot(model["feature_map"].draw(output="mpl"), use_container_width=True)
        except Exception:
            st.code(str(model["feature_map"].draw(output="text")))
        st.markdown("#### Trainable ansatz circuit")
        try:
            st.pyplot(model["ansatz"].draw(output="mpl"), use_container_width=True)
        except Exception:
            st.code(str(model["ansatz"].draw(output="text")))
        if model["history"]:
            st.pyplot(training_loss(model["history"]), use_container_width=True)
        else:
            st.info("This optimizer did not emit a loss history.")
        st.pyplot(confusion_matrix_plot(data.y_test, model["predictions"], "VQC confusion matrix"), use_container_width=True)
        st.markdown("**How it works:** the feature map encodes classical features, the ansatz supplies trainable parameters, and a classical optimizer updates those parameters according to the measured classification objective.")

with tabs[2]:
    if "Quantum Kernel + SVM" not in primary_models:
        st.info("Quantum Kernel + SVM was not selected or did not complete in the primary seed.")
        if "Quantum Kernel + SVM" in primary_failures:
            st.code(primary_failures["Quantum Kernel + SVM"])
    else:
        model = primary_models["Quantum Kernel + SVM"]
        st.subheader("Fidelity quantum kernel")
        a, b, c = st.columns(3)
        a.metric("Accuracy", f'{model["metrics"]["Accuracy"]:.3f}')
        b.metric("F1 score", f'{model["metrics"]["F1 Score"]:.3f}')
        c.metric("Kernel matrix", f'{model["kernel_train"].shape[0]} × {model["kernel_train"].shape[1]}')
        st.pyplot(kernel_heatmap(model["kernel_train"]), use_container_width=True)
        st.markdown("#### Quantum feature map circuit")
        try:
            st.pyplot(model["feature_map"].draw(output="mpl"), use_container_width=True)
        except Exception:
            st.code(str(model["feature_map"].draw(output="text")))
        st.pyplot(confusion_matrix_plot(data.y_test, model["predictions"], "Quantum-kernel SVM confusion matrix"), use_container_width=True)
        st.markdown("**How it works:** each input is encoded as a quantum state. The fidelity similarity matrix is passed to a classical SVM through a precomputed kernel. The test-to-train kernel matrix is evaluated separately.")

with tabs[3]:
    st.subheader("Experiment Insight Engine")
    for sentence in insights:
        st.markdown(f"- {sentence}")
    st.markdown("#### Reproducibility record")
    st.json({
        "configuration": bundle["config"],
        "seeds": [item["seed"] for item in bundle["runs"]],
        "completed_models_by_seed": {str(item["seed"]): list(item["models"].keys()) for item in bundle["runs"]},
        "created_at_utc": bundle["created_at"],
    })
    markdown_report = build_markdown_report(bundle, rows_frame, insights)
    st.download_button(
        "Download experiment report (Markdown)",
        data=markdown_report,
        file_name="quantumclassify_experiment_report.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download detailed metrics (CSV)",
        data=rows_frame.to_csv(index=False),
        file_name="quantumclassify_metrics.csv",
        mime="text/csv",
    )
    history = load_history()
    if history:
        st.markdown("#### Saved run history")
        history_rows = []
        for item in history[-30:][::-1]:
            config = item.get("config", {})
            history_rows.append({
                "Date (UTC)": item.get("created_at"),
                "Dataset": config.get("dataset"),
                "Qubits": config.get("qubits"),
                "Seeds": config.get("repeats"),
                "Models completed": len(item.get("summary", [])),
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
        history_json = json.dumps(history, indent=2)
        st.download_button("Download saved history (JSON)", history_json, "quantumclassify_history.json", "application/json")
    else:
        st.caption("No saved history is available yet.")

with tabs[4]:
    st.subheader("Methodology and fairness")
    st.markdown("""
    1. Generate or sample a dataset with a fixed random seed.
    2. Create one stratified training/test split shared by all selected models.
    3. Fit feature selection and scaling using training data only.
    4. Train the classical SVM, VQC, and quantum-kernel SVM as separate pipelines.
    5. Report predictive metrics separately from training, inference, and kernel-build time.
    6. For repeated seeds, summarize mean and sample standard deviation.
    """)
    st.subheader("Interpretation guardrails")
    st.warning("Higher performance on one small dataset is not proof of quantum advantage. Simulator results are not the same as hardware results.")
    st.markdown("""
    - Quantum kernel construction grows approximately quadratically with training sample count.
    - VQC optimization can depend strongly on initialization, circuit depth, optimizer, and shot count.
    - The Breast Cancer dataset uses a small, training-fitted feature-selection subset.
    - Boundary charts for four features are projections, not a full view of the four-dimensional decision surface.
    """)
