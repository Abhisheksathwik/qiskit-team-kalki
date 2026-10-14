"""Physical circuit metrics for the QFF Phase 1 architecture benchmark."""
from __future__ import annotations


def circuit_metrics(compilation: dict) -> dict:
    """Summarize logical and routed circuit costs without fabricating scores."""
    config = compilation["processor"]
    logical = compilation["unconstrained"]
    swap_probe = compilation["routed_with_swaps"]
    compiled = compilation["compiled"]
    logical_ops = logical.count_ops()
    routed_ops = compiled.count_ops()
    swap_ops = swap_probe.count_ops()
    logical_cx = int(logical_ops.get("cx", 0))
    compiled_cx = int(routed_ops.get("cx", 0))
    swap_count = int(swap_ops.get("swap", 0))
    cx_overhead = max(0, compiled_cx - logical_cx)
    return {
        "Processor": config["name"],
        "Physical Qubits": int(config["num_qubits"]),
        "Logical Circuit Depth": int(logical.depth()),
        "Compiled Circuit Depth": int(compiled.depth()),
        "Logical CX Count": logical_cx,
        "Compiled CX Count": compiled_cx,
        "SWAP Routing Count": swap_count,
        "Routing CX Overhead": int(cx_overhead),
        "CX-based SWAP-equivalent Overhead": float(cx_overhead / 3.0),
        "Native Basis": ", ".join(config["basis_gates"]),
        "Optimization Level": int(config["transpilation"]["optimization_level"]),
        "Transpiler Seed": int(config["transpilation"]["seed_transpiler"]),
        "Shots": int(config.get("shots", 4096)),
        "Status": "compiled",
    }


def output_distribution_fidelity(ideal_counts: dict, noisy_counts: dict) -> float:
    """Squared Bhattacharyya coefficient for ideal/noisy measured distributions.

    This is classical output-distribution fidelity (not state fidelity and not
    a hardware-calibrated fidelity estimate).
    """
    def normalise(counts):
        cleaned = {str(key).replace(" ", ""): float(value) for key, value in counts.items()}
        total = sum(cleaned.values())
        if total <= 0:
            raise ValueError("Measurement counts must have positive total")
        return {key: value / total for key, value in cleaned.items()}

    ideal = normalise(ideal_counts)
    noisy = normalise(noisy_counts)
    support = set(ideal) | set(noisy)
    coefficient = sum((ideal.get(key, 0.0) * noisy.get(key, 0.0)) ** 0.5 for key in support)
    return float(min(1.0, max(0.0, coefficient * coefficient)))
