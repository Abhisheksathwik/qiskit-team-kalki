"""Reproducible routing/transpilation against participant-defined processors."""
from __future__ import annotations

from qiskit import transpile

from src.processors import directed_coupling_edges, load_processor


def compile_for_processor(circuit, processor: str | dict):
    """Return unconstrained, SWAP-visible routed, and native-basis compiled circuits.

    Qiskit can decompose inserted SWAPs into CX gates during native-basis
    translation. A separate SWAP-visible pass is retained to count explicit
    routing SWAPs, while the final compiled circuit is in the required basis.
    """
    config = load_processor(processor)
    settings = config["transpilation"]
    level = int(settings.get("optimization_level", 1))
    seed = int(settings.get("seed_transpiler", 42))
    basis = list(config["basis_gates"])
    coupling = directed_coupling_edges(config)

    unconstrained = transpile(
        circuit, basis_gates=basis, optimization_level=level, seed_transpiler=seed
    )
    routed_with_swaps = transpile(
        circuit,
        basis_gates=[*basis, "swap"],
        coupling_map=coupling,
        optimization_level=level,
        seed_transpiler=seed,
        routing_method="basic",
    )
    compiled = transpile(
        circuit,
        basis_gates=basis,
        coupling_map=coupling,
        optimization_level=level,
        seed_transpiler=seed,
        routing_method="basic",
    )
    return {
        "processor": config,
        "unconstrained": unconstrained,
        "routed_with_swaps": routed_with_swaps,
        "compiled": compiled,
    }
