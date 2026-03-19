# =============================================================================
# ansatz.py — circuit ansatz (clean + Pauli-noise variant)
# =============================================================================

import numpy as np
import pennylane as qml

from .config import N_LAYERS, WIRES, RANGES


def apply_sel(weights):
    """Noiseless StronglyEntanglingLayers."""
    qml.StronglyEntanglingLayers(
        weights,
        wires=WIRES,
        ranges=RANGES,
        imprimitive=qml.CNOT,
    )


def apply_sel_with_layerwise_pauli(weights, noise_p: float):
    """
    Pauli-noise variant: before each SEL layer insert X/Z/Y PauliError
    on every qubit, then apply that layer.
    """
    p = np.clip(float(noise_p), 1e-12, 1.0 - 1e-12)

    for layer_idx in range(N_LAYERS):
        for wire in WIRES:
            qml.PauliError("X", p, wires=wire)
            qml.PauliError("Z", p, wires=wire)
            qml.PauliError("Y", p, wires=wire)

        qml.StronglyEntanglingLayers(
            weights[layer_idx : layer_idx + 1],
            wires=WIRES,
            ranges=[RANGES[layer_idx]],
            imprimitive=qml.CNOT,
        )
