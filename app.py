import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_data
from src.classical import train_classical
from src.evaluation import evaluate

st.set_page_config(page_title="QuantumClassify Lab", page_icon="⚛️", layout="wide")

st.title("⚛️ QuantumClassify Lab")
st.caption("Hybrid Variational Quantum Classifier • Quantum Kernel • Classical SVM")

with st.sidebar:
    st.header("Experiment Settings")
    dataset = st.selectbox("Dataset", ["Two Moons", "Breast Cancer"])
    samples = st.slider("Samples", 60, 160, 100, 20) if dataset == "Two Moons" else 120
    test_size = st.slider("Test size", 0.15, 0.30, 0.20, 0.05)
    iterations = st.slider("VQC optimizer iterations", 10, 80, 40, 10)
    run = st.button("🚀 Run Experiment", type="primary")

if "results" not in st.session_state:
    st.session_state.results = None

if run:
    with st.spinner("Running classical and quantum experiments..."):
        data = load_data(dataset, samples, test_size)
        X_train, X_test, X_train_q, X_test_q, y_train, y_test, names = data

        cm = train_classical(X_train, y_train, X_test)
        results = {"data": data, "classical": cm, "classical_metrics": evaluate(y_test, cm[1])}

        try:
            from src.vqc import train_vqc
            out = train_vqc(X_train_q, y_train, X_test_q, iterations)
            results["vqc"] = out
            results["vqc_metrics"] = evaluate(y_test, out[1])
        except Exception as e:
            results["vqc_error"] = str(e)

        try:
            from src.quantum_kernel import train_quantum_kernel
            out = train_quantum_kernel(X_train_q, y_train, X_test_q)
            results["qk"] = out
            results["qk_metrics"] = evaluate(y_test, out[1])
        except Exception as e:
            results["qk_error"] = str(e)

        st.session_state.results = results

if st.session_state.results is None:
    st.info("Choose settings and click **Run Experiment**.")
    st.markdown("""
### What this application demonstrates

**Classical SVM** — conventional nonlinear classification.

**Hybrid VQC** — a trainable parameterized quantum circuit optimized by a classical optimizer.

**Quantum Kernel + SVM** — a quantum feature map produces a similarity matrix used by a classical SVM.

The same held-out test set is used for the comparison.
""")
    st.stop()

r = st.session_state.results
X_train, X_test, X_train_q, X_test_q, y_train, y_test, names = r["data"]

rows = [{
    "Model": "Classical RBF SVM",
    "Accuracy": r["classical_metrics"]["Accuracy"],
    "Precision": r["classical_metrics"]["Precision"],
    "Recall": r["classical_metrics"]["Recall"],
    "F1 Score": r["classical_metrics"]["F1 Score"],
    "Training Time (s)": r["classical"][2],
}]
if "vqc" in r:
    rows.append({"Model": "Hybrid VQC", "Accuracy": r["vqc_metrics"]["Accuracy"], "Precision": r["vqc_metrics"]["Precision"], "Recall": r["vqc_metrics"]["Recall"], "F1 Score": r["vqc_metrics"]["F1 Score"], "Training Time (s)": r["vqc"][2]})
if "qk" in r:
    rows.append({"Model": "Quantum Kernel + SVM", "Accuracy": r["qk_metrics"]["Accuracy"], "Precision": r["qk_metrics"]["Precision"], "Recall": r["qk_metrics"]["Recall"], "F1 Score": r["qk_metrics"]["F1 Score"], "Training Time (s)": r["qk"][2]})

df = pd.DataFrame(rows)
st.subheader("Model Comparison")
st.dataframe(df, use_container_width=True)

c1, c2, c3 = st.columns(3)
for col, row in zip([c1, c2, c3], rows):
    col.metric(row["Model"], f'{row["Accuracy"]:.1%}', f'F1 {row["F1 Score"]:.3f}')

tab1, tab2, tab3, tab4 = st.tabs(["📈 Decision Boundaries", "⚛️ VQC", "🔬 Quantum Kernel", "📝 Findings"])

with tab1:
    from src.visualization import decision_boundary
    st.pyplot(decision_boundary(r["classical"][0], X_train, y_train, X_test, y_test, "Classical RBF SVM"))
    if "vqc" in r:
        st.pyplot(decision_boundary(r["vqc"][0], X_train_q, y_train, X_test_q, y_test, "Hybrid VQC"))

with tab2:
    if "vqc" in r:
        model, pred, elapsed, history, fmap, ansatz = r["vqc"]
        st.subheader("Quantum Feature Map")
        st.code(str(fmap.draw(output="text")))
        st.subheader("Trainable Ansatz")
        st.code(str(ansatz.draw(output="text")))
        if history:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(history)
            ax.set_title("VQC Training Loss")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Objective")
            fig.tight_layout()
            st.pyplot(fig)
    else:
        st.error("VQC could not run in this environment.")
        st.code(r.get("vqc_error", "Unknown error"))

with tab3:
    if "qk" in r:
        from src.visualization import kernel_heatmap
        st.pyplot(kernel_heatmap(r["qk"][3]))
        st.subheader("Quantum Feature Map")
        st.code(str(r["qk"][4].draw(output="text")))
    else:
        st.error("Quantum kernel could not run in this environment.")
        st.code(r.get("qk_error", "Unknown error"))

with tab4:
    best = df.loc[df["F1 Score"].idxmax(), "Model"]
    st.success(f"Best F1-score in this experiment: **{best}**")
    st.write("Use measured results in the presentation. Simulator results do not establish real-hardware quantum advantage.")

st.download_button("⬇️ Download results CSV", df.to_csv(index=False), "quantum_classify_results.csv", "text/csv")
