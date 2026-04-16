# =============================================================================
# hamiltonian.py — selectable Hamiltonian families + exact ground energy
# =============================================================================

from __future__ import annotations

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
    exact_top_energy: float
    spectrum_span: float


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


def exact_spectrum_bounds(hamiltonian: qml.Hamiltonian, n_qubits: int) -> tuple[float, float]:
    """Compute the exact lowest and highest energies by dense diagonalization."""
    mat = qml.matrix(hamiltonian, wire_order=list(range(n_qubits)))
    eigvals = np.linalg.eigvalsh(np.array(mat, dtype=complex))
    eigvals = np.real(eigvals)
    return float(np.min(eigvals)), float(np.max(eigvals))


def exact_ground_energy(hamiltonian: qml.Hamiltonian, n_qubits: int) -> float:
    """Backward-compatible helper returning only the exact ground-state energy."""
    ground, _ = exact_spectrum_bounds(hamiltonian, n_qubits)
    return ground


def build_hamiltonian_by_name(
    name: str,
    *,
    n_qubits: int,
    periodic: bool,
    params: dict,
) -> tuple[qml.Hamiltonian, HamiltonianInfo]:
    """
    Build a Hamiltonian family from an explicit parameter dict.

    This is the non-global helper used by the separate seed-sweep comparison
    mode so that different random coefficient draws can be handled without
    mutating the module-level default Hamiltonian.
    """
    if name == "single_Z":
        coeff = float(params["single_z_coeff"])
        h = build_single_z_hamiltonian(n_qubits=n_qubits, coeff=coeff)
        display_name = "single_Z"
        info_params = {"single_z_coeff": coeff}
    elif name == "tfim_longitudinal":
        jzz = float(params["jzz"])
        hx = float(params["hx"])
        hz = float(params["hz"])
        h = build_tfim_longitudinal_hamiltonian(
            n_qubits=n_qubits,
            jzz=jzz,
            hx=hx,
            hz=hz,
            periodic=periodic,
        )
        display_name = "1D TFIM + longitudinal field"
        info_params = {"jzz": jzz, "hx": hx, "hz": hz}
    elif name == "j1j2_heisenberg":
        j1 = float(params["j1"])
        j2 = float(params["j2"])
        h = build_j1j2_heisenberg_hamiltonian(
            n_qubits=n_qubits,
            j1=j1,
            j2=j2,
            periodic=periodic,
        )
        display_name = "1D J1-J2 Heisenberg"
        info_params = {"j1": j1, "j2": j2}
    else:
        raise ValueError(
            f"Unknown HAMILTONIAN_NAME={name!r}. "
            "Choose from 'single_Z', 'tfim_longitudinal', or 'j1j2_heisenberg'."
        )

    ground, top = exact_spectrum_bounds(h, n_qubits)
    info = HamiltonianInfo(
        name=name,
        display_name=display_name,
        periodic_boundary=bool(periodic),
        params=info_params,
        exact_ground_energy=ground,
        exact_top_energy=top,
        spectrum_span=float(top - ground),
    )
    return h, info


def build_selected_hamiltonian() -> tuple[qml.Hamiltonian, HamiltonianInfo]:
    name = C.HAMILTONIAN_NAME
    n = C.N_QUBITS

    if name == "single_Z":
        params = {"single_z_coeff": float(C.SINGLE_Z_COEFF)}
    elif name == "tfim_longitudinal":
        params = {
            "jzz": float(C.TFIM_JZZ),
            "hx": float(C.TFIM_HX),
            "hz": float(C.TFIM_HZ),
        }
    elif name == "j1j2_heisenberg":
        params = {
            "j1": float(C.J1),
            "j2": float(C.J2),
        }
    else:
        raise ValueError(
            f"Unknown HAMILTONIAN_NAME={name!r}. "
            "Choose from 'single_Z', 'tfim_longitudinal', or 'j1j2_heisenberg'."
        )

    return build_hamiltonian_by_name(
        name,
        n_qubits=n,
        periodic=bool(C.PERIODIC_BOUNDARY),
        params=params,
    )


def hamiltonian_info_dict() -> dict:
    return asdict(H_INFO)


def hamiltonian_summary_string() -> str:
    param_str = ", ".join(f"{k}={v}" for k, v in H_INFO.params.items())
    bc = "periodic" if H_INFO.periodic_boundary else "open"
    base = f"{H_INFO.display_name} ({H_INFO.name})"
    if H_INFO.name == "single_Z":
        return (
            f"{base} | {param_str} | "
            f"exact_ground_energy={H_INFO.exact_ground_energy:.8f} | "
            f"exact_top_energy={H_INFO.exact_top_energy:.8f}"
        )
    return (
        f"{base} | boundary={bc} | {param_str} | "
        f"exact_ground_energy={H_INFO.exact_ground_energy:.8f} | "
        f"exact_top_energy={H_INFO.exact_top_energy:.8f}"
    )


H, H_INFO = build_selected_hamiltonian()
# Keep the legacy variable name so that the rest of the project continues to
# read the exact ground-state energy from the config module.
C.GLOBAL_MIN_ENERGY = H_INFO.exact_ground_energy
