# =============================================================================
# noise.py — cost-function factories (clean / Pauli-noise / shot-noise)
# =============================================================================

import pennylane as qml

from config import N_QUBITS
from hamiltonian import H
from ansatz import apply_sel, apply_sel_with_layerwise_pauli


def make_clean_cost():
    """Exact statevector expectation value."""
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        apply_sel(weights)
        return qml.expval(H)

    return cost


def make_pauli_cost(noise_p: float):
    """Density-matrix simulation with layerwise Pauli noise."""
    dev = qml.device("default.mixed", wires=N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        apply_sel_with_layerwise_pauli(weights, noise_p)
        return qml.expval(H)

    return cost


def make_shot_cost(shots: int, device_seed: int):
    """Finite-shot sampling (noiseless circuit, noisy gradient)."""
    dev = qml.device(
        "default.qubit",
        wires=N_QUBITS,
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
