# =============================================================================
# hamiltonian.py — selectable Hamiltonian families + exact ground energy
# =============================================================================

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pennylane as qml

from . import config as C


@dataclass(frozen=True)
class HamiltonianInfo:
    name: str
    display_name: str
    periodic_boundary: bool
    params: dict
    exact_ground_energy: float


def _sorted_unique_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return unique undirected pairs with sorted endpoints and stable ordering."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i, j in pairs:
        if i == j:
            continue
        pair = (i, j) if i < j else (j, i)
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def nearest_neighbor_pairs(n_qubits: int, periodic: bool) -> list[tuple[int, int]]:
    pairs = [(i, i + 1) for i in range(max(0, n_qubits - 1))]
    if periodic and n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    return _sorted_unique_pairs(pairs)


def next_nearest_neighbor_pairs(n_qubits: int, periodic: bool) -> list[tuple[int, int]]:
    pairs = [(i, i + 2) for i in range(max(0, n_qubits - 2))]
    if periodic and n_qubits > 3:
        pairs.extend([(n_qubits - 2, 0), (n_qubits - 1, 1)])
    return _sorted_unique_pairs(pairs)


def build_single_z_hamiltonian(n_qubits: int, coeff: float = 1.0) -> qml.Hamiltonian:
    """H = coeff * Σ_i Z_i."""
    coeffs = [float(coeff)] * n_qubits
    obs = [qml.PauliZ(i) for i in range(n_qubits)]
    return qml.Hamiltonian(coeffs, obs)


def build_tfim_longitudinal_hamiltonian(
    n_qubits: int,
    jzz: float,
    hx: float,
    hz: float,
    periodic: bool,
) -> qml.Hamiltonian:
    """
    1D transverse-field Ising model with an additional longitudinal field.

    Hamiltonian used in this project:
        H = -jzz * Σ Z_i Z_{i+1} - hx * Σ X_i - hz * Σ Z_i
    """
    coeffs: list[float] = []
    obs: list = []

    for i, j in nearest_neighbor_pairs(n_qubits, periodic):
        coeffs.append(-float(jzz))
        obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    for i in range(n_qubits):
        coeffs.append(-float(hx))
        obs.append(qml.PauliX(i))

    for i in range(n_qubits):
        coeffs.append(-float(hz))
        obs.append(qml.PauliZ(i))

    return qml.Hamiltonian(coeffs, obs)


def build_j1j2_heisenberg_hamiltonian(
    n_qubits: int,
    j1: float,
    j2: float,
    periodic: bool,
) -> qml.Hamiltonian:
    """
    1D J1-J2 Heisenberg chain.

    Hamiltonian used in this project:
        H = j1 * Σ_nn  (XX + YY + ZZ) + j2 * Σ_nnn (XX + YY + ZZ)
    """
    coeffs: list[float] = []
    obs: list = []

    def add_heisenberg_pair(i: int, j: int, coupling: float) -> None:
        c = float(coupling)
        coeffs.extend([c, c, c])
        obs.extend([
            qml.PauliX(i) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliY(j),
            qml.PauliZ(i) @ qml.PauliZ(j),
        ])

    for i, j in nearest_neighbor_pairs(n_qubits, periodic):
        add_heisenberg_pair(i, j, j1)

    for i, j in next_nearest_neighbor_pairs(n_qubits, periodic):
        add_heisenberg_pair(i, j, j2)

    return qml.Hamiltonian(coeffs, obs)


def exact_ground_energy(hamiltonian: qml.Hamiltonian, n_qubits: int) -> float:
    """Compute the exact ground-state energy by dense diagonalization."""
    mat = qml.matrix(hamiltonian, wire_order=list(range(n_qubits)))
    eigvals = np.linalg.eigvalsh(np.array(mat, dtype=complex))
    return float(np.min(np.real(eigvals)))


def build_selected_hamiltonian() -> tuple[qml.Hamiltonian, HamiltonianInfo]:
    name = C.HAMILTONIAN_NAME
    n = C.N_QUBITS

    if name == "single_Z":
        h = build_single_z_hamiltonian(n_qubits=n, coeff=C.SINGLE_Z_COEFF)
        display_name = "single_Z"
        params = {
            "single_z_coeff": float(C.SINGLE_Z_COEFF),
        }
    elif name == "tfim_longitudinal":
        h = build_tfim_longitudinal_hamiltonian(
            n_qubits=n,
            jzz=C.TFIM_JZZ,
            hx=C.TFIM_HX,
            hz=C.TFIM_HZ,
            periodic=C.PERIODIC_BOUNDARY,
        )
        display_name = "1D TFIM + longitudinal field"
        params = {
            "jzz": float(C.TFIM_JZZ),
            "hx": float(C.TFIM_HX),
            "hz": float(C.TFIM_HZ),
        }
    elif name == "j1j2_heisenberg":
        h = build_j1j2_heisenberg_hamiltonian(
            n_qubits=n,
            j1=C.J1,
            j2=C.J2,
            periodic=C.PERIODIC_BOUNDARY,
        )
        display_name = "1D J1-J2 Heisenberg"
        params = {
            "j1": float(C.J1),
            "j2": float(C.J2),
        }
    else:
        raise ValueError(
            f"Unknown HAMILTONIAN_NAME={name!r}. "
            "Choose from 'single_Z', 'tfim_longitudinal', or 'j1j2_heisenberg'."
        )

    ground = exact_ground_energy(h, n)
    info = HamiltonianInfo(
        name=name,
        display_name=display_name,
        periodic_boundary=bool(C.PERIODIC_BOUNDARY),
        params=params,
        exact_ground_energy=ground,
    )
    return h, info


def hamiltonian_info_dict() -> dict:
    return asdict(H_INFO)


def hamiltonian_summary_string() -> str:
    param_str = ", ".join(f"{k}={v}" for k, v in H_INFO.params.items())
    bc = "periodic" if H_INFO.periodic_boundary else "open"
    base = f"{H_INFO.display_name} ({H_INFO.name})"
    if H_INFO.name == "single_Z":
        return f"{base} | {param_str} | exact_ground_energy={H_INFO.exact_ground_energy:.8f}"
    return f"{base} | boundary={bc} | {param_str} | exact_ground_energy={H_INFO.exact_ground_energy:.8f}"


H, H_INFO = build_selected_hamiltonian()
# Keep the legacy variable name so that the rest of the project continues to
# read the exact ground-state energy from the config module.
C.GLOBAL_MIN_ENERGY = H_INFO.exact_ground_energy
