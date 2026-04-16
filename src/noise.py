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



def make_clean_cost_for_hamiltonian(hamiltonian):
    """Exact statevector expectation value for an explicit Hamiltonian."""
    dev = qml.device("default.qubit", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        apply_sel(weights)
        return qml.expval(hamiltonian)

    return cost



def make_regularized_cost_for_hamiltonian(hamiltonian, noise_mode=None):
    """
    Density-matrix simulation with explicit injected noise for an explicit Hamiltonian.

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
    """
    mode = C.NOISE_MODE if noise_mode is None else noise_mode
    ansatz_fn = _NOISE_ANSATZ.get(mode)
    if ansatz_fn is None:
        raise ValueError(f"Unknown NOISE_MODE={mode!r}. Choose 'matched' or 'layerwise'.")

    dev = qml.device("default.mixed", wires=C.N_QUBITS)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights, noise_strength):
        ansatz_fn(weights, noise_strength)
        return qml.expval(hamiltonian)

    return cost



def make_clean_cost():
    """Exact statevector expectation value for the module-level selected Hamiltonian."""
    return make_clean_cost_for_hamiltonian(H)



def make_regularized_cost():
    """Regularized cost for the module-level selected Hamiltonian."""
    return make_regularized_cost_for_hamiltonian(H, noise_mode=C.NOISE_MODE)



def make_pauli_cost(noise_strength: float):
    """Freeze the user-facing regularization strength and return a unary cost function."""
    def cost(weights):
        return REGULARIZED_COST(weights, noise_strength)
    return cost



def make_shot_cost_for_hamiltonian(hamiltonian, shots: int, device_seed: int):
    """Finite-shot sampling for an explicit Hamiltonian."""
    dev = qml.device(
        "default.qubit",
        wires=C.N_QUBITS,
        shots=shots,
        seed=device_seed,
    )

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def cost(weights):
        apply_sel(weights)
        return qml.expval(hamiltonian)

    return cost



def make_shot_cost(shots: int, device_seed: int):
    """Finite-shot sampling (noiseless circuit, noisy gradient)."""
    return make_shot_cost_for_hamiltonian(H, shots=shots, device_seed=device_seed)


CLEAN_COST = make_clean_cost()
REGULARIZED_COST = make_regularized_cost()
