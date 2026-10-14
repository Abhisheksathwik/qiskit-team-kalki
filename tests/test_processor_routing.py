import pytest

from qiskit import QuantumCircuit

from src.metrics import circuit_metrics
from src.routing import compile_for_processor


@pytest.mark.parametrize(("processor", "expected_qubits"), [("A", 5), ("B", 9)])
def test_circuit_can_be_transpiled_to_each_processor(processor, expected_qubits):
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    circuit.rz(0.3, 1)
    circuit.measure_all()

    compilation = compile_for_processor(circuit, processor)
    report = circuit_metrics(compilation)

    assert compilation["compiled"].num_qubits <= expected_qubits
    assert report["Processor"] == f"Processor {processor}"
    assert report["Physical Qubits"] == expected_qubits
    assert report["Compiled Circuit Depth"] >= 0
    assert report["Compiled CX Count"] >= 0
    assert report["Routing CX Overhead"] >= 0
    assert report["CX-based SWAP-equivalent Overhead"] >= 0.0
