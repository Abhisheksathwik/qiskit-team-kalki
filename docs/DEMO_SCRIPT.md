# QuantumClassify Lab — Demo Script

Recommended live demo configuration: Two Moons, 60–80 total samples, 2 qubits, one VQC ansatz layer, COBYLA, 10–20 optimizer iterations, and one seed. Run quantum simulation before the judges arrive if time is limited.

## 1. Opening (20 seconds)

“QuantumClassify Lab is an experiment platform that compares a classical RBF support-vector machine with two different quantum machine-learning approaches: a hybrid variational quantum classifier and a fidelity quantum-kernel classifier. We test the methods instead of assuming a quantum advantage.”

## 2. Configure the experiment

Show the dataset, test fraction, number of qubits/features, VQC layer count, optimizer, shots, and seed. Explain that all models in a run share the stratified held-out split.

## 3. Run and compare

Click Run experiment. Show the scoreboard and metrics chart. Mention that training, inference, and quantum-kernel construction are timed separately because they have different cost profiles.

## 4. Explain the quantum approaches

In the VQC tab, show the quantum feature map, trainable ansatz, and training objective history if available. Explain that a classical optimizer changes circuit parameters.

In the Quantum Kernel tab, show the fidelity kernel matrix and explain that its similarities are used by a classical SVM. The matrix grows quadratically with the number of training points.

## 5. Discuss the evidence

Read the automatically generated findings and inspect confusion matrices and decision boundaries. Do not claim quantum advantage if the measured evidence does not establish it.

## 6. Export

Download the CSV and Markdown report. Explain that the report includes the configuration, seed, metrics, observations, and limitations so the experiment can be reviewed.

## 7. Close

“Our contribution is a reproducible benchmark and an explanation interface. A quantum method is useful only if the measured evidence and resource costs justify it.”
