# =============================================================================
# ansatz.py — circuit ansatz (clean + Pauli-noise variants)
#
# Functions:
#   apply_sel                        : noiseless StronglyEntanglingLayers
#   apply_sel_with_layerwise_pauli   : X/Z/Y PauliError before each SEL layer
#   apply_sel_with_matched_pauli     : matched PauliError before each RZ/RY/RZ
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


def apply_sel_with_matched_pauli(weights, noise_p: float):
    """
    Matched Pauli-noise variant: SEL은 RZ·RY·RZ로 직접 전개하고,
    각 rotation gate 앞에 같은 축의 PauliError를 삽입.
      RZ(φ)   ← PauliError("Z", p)
      RY(θ)   ← PauliError("Y", p)
      RZ(ω)   ← PauliError("Z", p)
    CNOT entangling 패턴은 StronglyEntanglingLayers와 동일하게 유지.
    """
    p = np.clip(float(noise_p), 1e-12, 1.0 - 1e-12)
    n = len(WIRES)

    for layer_idx in range(N_LAYERS):
        r = RANGES[layer_idx]
        for wire in WIRES:
            phi   = weights[layer_idx, wire, 0]
            theta = weights[layer_idx, wire, 1]
            omega = weights[layer_idx, wire, 2]

            qml.PauliError("Z", p, wires=wire)
            qml.RZ(phi, wires=wire)

            qml.PauliError("Y", p, wires=wire)
            qml.RY(theta, wires=wire)

            qml.PauliError("Z", p, wires=wire)
            qml.RZ(omega, wires=wire)

        for i, wire in enumerate(WIRES):
            qml.CNOT(wires=[wire, WIRES[(i + r) % n]])
