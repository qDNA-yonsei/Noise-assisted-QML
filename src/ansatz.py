# =============================================================================
# ansatz.py — circuit ansatz (clean + Pauli-noise variants)
# =============================================================================

import pennylane as qml

from .config import N_LAYERS, WIRES, RANGES


def _validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return value


def _matched_user_p_to_paulierror_prob(user_p: float) -> float:
    """
    Convert the user-facing matched-mode parameter `p` into the actual
    probability passed to qml.PauliError.

    IMPORTANT TERMINOLOGY NOTE
    --------------------------
    This project intentionally uses the single name `p` in both noise modes,
    because that is simpler to work with from the command line and in config
    files.

    However, the meaning of that user-facing `p` is mode-dependent:

    1) matched mode
       - `p` means the PAPER regularization strength.
       - The paper channel is
             E_P(p)(rho) = (1 - p/2) rho + (p/2) P rho P.
       - PennyLane's qml.PauliError("P", q) expects the *actual* Pauli-flip
         probability q in
             rho -> (1 - q) rho + q P rho P.
       - Therefore, to match the paper exactly we must use q = p / 2.

    2) layerwise mode
       - `p` is used directly as the PauliError probability.

    So the public interface is unified, but the matched implementation still
    reproduces the paper exactly by converting user-facing `p` to `p/2`
    internally.
    """
    user_p = _validate_unit_interval(user_p, "matched-mode user-facing p")
    return 0.5 * user_p


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
    Legacy layerwise Pauli-noise variant.

    In this mode the user-facing parameter `p` is also the direct probability
    given to qml.PauliError.

    This is NOT the exact protocol from the paper. It simply inserts
    X/Z/Y PauliError(p) channels before each SEL layer.
    """
    p = _validate_unit_interval(noise_p, "layerwise noise probability p")

    for layer_idx in range(N_LAYERS):
        for wire in WIRES:
            if p > 0.0:
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
    Exact matched protocol from Bagaev et al. (2025), but exposed through the
    unified user-facing parameter name `p`.

    USER-FACING INTERFACE
    ---------------------
    The caller passes a value `p in [0, 1]`.

    INTERNAL IMPLEMENTATION
    -----------------------
    To reproduce the paper exactly, this function interprets that user-facing
    `p` as the paper regularization strength and converts it internally to the
    actual qml.PauliError probability `p/2`.

    In other words:
        user-facing p  ->  paper strength p  ->  qml.PauliError probability p/2

    This keeps the CLI/config simple while preserving exact agreement with the
    paper's matched noise channel.
    """
    paulierror_prob = _matched_user_p_to_paulierror_prob(noise_p)
    n = len(WIRES)

    for layer_idx in range(N_LAYERS):
        r = RANGES[layer_idx]
        for wire in WIRES:
            phi = weights[layer_idx, wire, 0]
            theta = weights[layer_idx, wire, 1]
            omega = weights[layer_idx, wire, 2]

            if paulierror_prob > 0.0:
                qml.PauliError("Z", paulierror_prob, wires=wire)
            qml.RZ(phi, wires=wire)

            if paulierror_prob > 0.0:
                qml.PauliError("Y", paulierror_prob, wires=wire)
            qml.RY(theta, wires=wire)

            if paulierror_prob > 0.0:
                qml.PauliError("Z", paulierror_prob, wires=wire)
            qml.RZ(omega, wires=wire)

        for i, wire in enumerate(WIRES):
            qml.CNOT(wires=[wire, WIRES[(i + r) % n]])
