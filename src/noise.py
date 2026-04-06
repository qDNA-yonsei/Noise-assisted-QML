# =============================================================================
# noise.py — cost-function factories (clean / regularized / shot-noise)
# =============================================================================

import pennylane as qml

from . import config as C
from .hamiltonian import H
from .ansatz import apply_sel, apply_sel_with_layerwise_pauli, apply_sel_with_matched_pauli

_NOISE_ANSATZ = {
    "matched": apply_sel_with_matched_pauli,
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


def make_regularized_cost():
    """
    Density-matrix simulation with explicit injected noise.

    IMPORTANT USER-FACING CONVENTION
    --------------------------------
    This project always exposes the regularization control as a single variable
    named `p`, regardless of the selected noise mode.

    - noise_mode='matched'
        The user-facing `p` is interpreted as the paper regularization
        strength. The exact paper channel is then implemented internally via
        qml.PauliError(..., p/2).

    - noise_mode='layerwise'
        The user-facing `p` is used directly as the qml.PauliError probability.

    So the external interface is unified, while the matched path still remains
    exactly faithful to the paper.
    """
    ansatz_fn = _NOISE_ANSATZ.get(C.NOISE_MODE)
    if ansatz_fn is None:
        raise ValueError(f"Unknown NOISE_MODE={C.NOISE_MODE!r}. Choose 'matched' or 'layerwise'.")

    dev = qml.device("default.mixed", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights, noise_strength):
        ansatz_fn(weights, noise_strength)
        return qml.expval(H)

    return cost


def make_pauli_cost(noise_strength: float):
    """Freeze the user-facing regularization strength and return a unary cost function."""
    def cost(weights):
        return REGULARIZED_COST(weights, noise_strength)
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


CLEAN_COST = make_clean_cost()
REGULARIZED_COST = make_regularized_cost()
