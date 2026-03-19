# =============================================================================
# noise.py — cost-function factories (clean / Pauli-noise / shot-noise)
# =============================================================================

import pennylane as qml

from . import config as C
from .hamiltonian import H
from .ansatz import apply_sel, apply_sel_with_layerwise_pauli, apply_sel_with_matched_pauli

_NOISE_ANSATZ = {
    "matched":   apply_sel_with_matched_pauli,
    "layerwise": apply_sel_with_layerwise_pauli,
}


def make_clean_cost():
    """Exact statevector expectation value."""
    dev = qml.device("default.qubit", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        apply_sel(weights)
        return qml.expval(H)

    return cost


def make_pauli_cost(noise_p: float):
    """Density-matrix simulation with Pauli noise.

    Noise insertion method is selected by C.NOISE_MODE:
      'matched'   — PauliError(Z/Y/Z) before each RZ/RY/RZ  (default)
      'layerwise' — PauliError(X/Z/Y) before each SEL layer
    """
    ansatz_fn = _NOISE_ANSATZ.get(C.NOISE_MODE)
    if ansatz_fn is None:
        raise ValueError(f"Unknown NOISE_MODE={C.NOISE_MODE!r}. Choose 'matched' or 'layerwise'.")

    dev = qml.device("default.mixed", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        ansatz_fn(weights, noise_p)
        return qml.expval(H)

    return cost


def make_shot_cost(shots: int, device_seed: int):
    """Finite-shot sampling (noiseless circuit, noisy gradient)."""
    dev = qml.device(
        "default.qubit",
        wires=C.N_QUBITS,
        shots=shots,
        seed=device_seed,
    )

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        apply_sel(weights)
        return qml.expval(H)

    return cost


# Module-level singleton — avoids rebuilding on every import
CLEAN_COST = make_clean_cost()
