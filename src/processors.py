"""Participant-designed Processor A/B topology and synthetic noise models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_DIR = ROOT / "processors"


def load_processor(processor: str | dict[str, Any]) -> dict[str, Any]:
    """Load a JSON processor config or validate a supplied config object."""
    if isinstance(processor, dict):
        config = dict(processor)
    else:
        key = str(processor).strip().upper().replace("PROCESSOR ", "")
        if key not in {"A", "B"}:
            path = Path(str(processor))
            if not path.is_file():
                raise ValueError("processor must be 'A', 'B', a config dictionary, or a JSON path")
            config = json.loads(path.read_text(encoding="utf-8"))
        else:
            path = PROCESSOR_DIR / f"processor_{key}.json"
            config = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "num_qubits", "coupling_edges", "basis_gates", "transpilation", "noise"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Processor config is missing fields: {sorted(missing)}")
    n_qubits = int(config["num_qubits"])
    if n_qubits not in (5, 9):
        raise ValueError("Phase 1 processor sizes must be 5 qubits (A) or 9 qubits (B)")
    seen = set()
    for edge in config["coupling_edges"]:
        if len(edge) != 2:
            raise ValueError(f"Invalid coupling edge: {edge!r}")
        a, b = map(int, edge)
        if a == b or min(a, b) < 0 or max(a, b) >= n_qubits:
            raise ValueError(f"Coupling edge {edge!r} is outside the processor graph")
        canonical = tuple(sorted((a, b)))
        if canonical in seen:
            raise ValueError(f"Duplicate undirected coupling edge: {canonical}")
        seen.add(canonical)
    adjacency = {node: set() for node in range(n_qubits)}
    for a, b in seen:
        adjacency[a].add(b)
        adjacency[b].add(a)
    reached, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(adjacency[node] - reached)
    if len(reached) != n_qubits:
        raise ValueError(f"{config['name']} coupling graph must be connected")
    if set(config["basis_gates"]) != {"rz", "sx", "x", "cx"}:
        raise ValueError("Phase 1 basis gates must be exactly rz, sx, x, cx")
    return config


def directed_coupling_edges(config: dict[str, Any]) -> list[list[int]]:
    """Expand undirected config edges into both directed CX orientations."""
    return [direction for a, b in config["coupling_edges"] for direction in ([int(a), int(b)], [int(b), int(a)])]


def build_noise_model(config: str | dict[str, Any]):
    """Create the official Phase 1 depolarizing + readout noise model."""
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    processor = load_processor(config)
    rates = processor["noise"]
    p1 = float(rates["one_qubit_depolarizing_probability"])
    p2 = float(rates["cx_depolarizing_probability"])
    pr = float(rates["readout_error_probability"])
    for value, name in ((p1, "one-qubit"), (p2, "CX"), (pr, "readout")):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} error probability must be in [0, 1]")
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    model.add_all_qubit_readout_error(ReadoutError([[1.0 - pr, pr], [pr, 1.0 - pr]]))
    return model


def sampler_backend_options(config: str | dict[str, Any]) -> dict[str, Any]:
    """Backend options for Aer SamplerV2, including topology and official noise."""
    processor = load_processor(config)
    return {
        "basis_gates": list(processor["basis_gates"]),
        "coupling_map": directed_coupling_edges(processor),
        "noise_model": build_noise_model(processor),
    }


def build_aer_backend(config: str | dict[str, Any], *, include_noise: bool = True, seed: int = 42):
    """Create an AerSimulator whose target reflects one configured processor."""
    from qiskit_aer import AerSimulator

    processor = load_processor(config)
    options = {
        "basis_gates": list(processor["basis_gates"]),
        "coupling_map": directed_coupling_edges(processor),
        "seed_simulator": int(seed),
    }
    if include_noise:
        options["noise_model"] = build_noise_model(processor)
    return AerSimulator(**options)
