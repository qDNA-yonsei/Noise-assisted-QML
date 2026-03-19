# =============================================================================
# hamiltonian.py — Hamiltonian definition
# =============================================================================

import pennylane as qml
from .config import N_QUBITS


def build_local_z_hamiltonian(n_qubits: int) -> qml.Hamiltonian:
    """H = Σ_i Z_i  (global min energy = -n_qubits)"""
    coeffs = [1.0] * n_qubits
    obs    = [qml.PauliZ(i) for i in range(n_qubits)]
    return qml.Hamiltonian(coeffs, obs)


H = build_local_z_hamiltonian(N_QUBITS)
