# =============================================================================
# utils.py — param init, optimizer helpers, Hessian diagnostics, convergence
# =============================================================================

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from . import config as C
from .noise import CLEAN_COST


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------


def init_params(seed: int) -> pnp.ndarray:
    """Uniform random init in [0, 2π)."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.0, 2.0 * np.pi, size=(C.N_LAYERS, C.N_QUBITS, 3))
    return pnp.array(arr, requires_grad=True)



def to_trainable(x) -> pnp.ndarray:
    return pnp.array(np.array(x), requires_grad=True)


# ---------------------------------------------------------------------------
# Hessian / gradient diagnostics
# ---------------------------------------------------------------------------


def hessian_info(cost_fn, params) -> dict:
    """Returns grad_norm, lambda_min, lambda_max for an arbitrary unary cost_fn."""
    shape = (C.N_LAYERS, C.N_QUBITS, 3)
    x0 = pnp.array(np.array(params).reshape(-1), requires_grad=True)

    def flat_cost(x):
        return cost_fn(x.reshape(shape))

    grad_fn = qml.grad(flat_cost)
    hess_fn = qml.jacobian(grad_fn)

    g = np.array(grad_fn(x0), dtype=float)
    h_mat = np.array(hess_fn(x0), dtype=float)
    h_mat = 0.5 * (h_mat + h_mat.T)
    eigvals = np.linalg.eigvalsh(h_mat)

    return {
        "grad_norm": float(np.linalg.norm(g)),
        "lambda_min": float(eigvals[0]),
        "lambda_max": float(eigvals[-1]),
    }


def clean_hessian_info(params) -> dict:
    """Returns grad_norm, lambda_min, lambda_max on the clean cost landscape."""
    return hessian_info(CLEAN_COST, params)


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------


def validate_optimizer(name: str) -> str:
    name = name.lower()
    if name not in {"gd", "adam", "spsa"}:
        raise ValueError(f"optimizer must be 'gd', 'adam', or 'spsa', got {name!r}")
    return name



def make_stepper(
    cost_fn,
    optimizer_type: str,
    max_steps_for_spsa: int,
    lr_gd: float,
    lr_adam: float,
    spsa_alpha: float,
    spsa_gamma: float,
    spsa_c: float,
    spsa_A,
    spsa_a,
):
    """Returns (step_fn, description_str)."""
    optimizer_type = validate_optimizer(optimizer_type)

    def _unwrap(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    if optimizer_type == "adam":
        opt = qml.AdamOptimizer(stepsize=lr_adam)
        step = lambda p: to_trainable(_unwrap(opt.step(cost_fn, p)))
        desc = f"adam(lr={lr_adam})"

    elif optimizer_type == "gd":
        grad_fn = qml.grad(cost_fn)
        step = lambda p: to_trainable(p - lr_gd * grad_fn(p))
        desc = f"gd(lr={lr_gd})"

    else:  # spsa
        opt = qml.SPSAOptimizer(
            maxiter=max(1, int(max_steps_for_spsa)),
            alpha=spsa_alpha,
            gamma=spsa_gamma,
            c=spsa_c,
            A=spsa_A,
            a=spsa_a,
        )
        step = lambda p: to_trainable(_unwrap(opt.step(cost_fn, p)))
        desc = (
            f"spsa(alpha={spsa_alpha}, gamma={spsa_gamma}, c={spsa_c}, "
            f"A={spsa_A}, a={spsa_a}, maxiter={max(1, int(max_steps_for_spsa))})"
        )

    return step, desc



def optimize_fixed_steps(
    cost_fn,
    params0,
    steps: int,
    optimizer_type: str,
    lr_gd: float,
    lr_adam: float,
    spsa_alpha: float,
    spsa_gamma: float,
    spsa_c: float,
    spsa_A,
    spsa_a,
    eval_fn=None,
) -> dict:
    """
    Run optimizer for a fixed number of steps.

    Returns dict with train_hist, eval_hist, final_params, etc.
    eval_fn defaults to cost_fn (set to CLEAN_COST for noisy optimizers).
    """
    if eval_fn is None:
        eval_fn = cost_fn

    params = to_trainable(params0)
    step, desc = make_stepper(
        cost_fn, optimizer_type, steps,
        lr_gd, lr_adam,
        spsa_alpha, spsa_gamma, spsa_c, spsa_A, spsa_a,
    )

    train_hist = [float(cost_fn(params))]
    eval_hist = [float(eval_fn(params))]

    for _ in range(steps):
        params = step(params)
        train_hist.append(float(cost_fn(params)))
        eval_hist.append(float(eval_fn(params)))

    return {
        "final_params": to_trainable(params),
        "train_hist": np.array(train_hist),
        "eval_hist": np.array(eval_hist),
        "final_train": float(train_hist[-1]),
        "final_eval": float(eval_hist[-1]),
        "total_steps": int(steps),
        "optimizer_type": optimizer_type,
        "optimizer_desc": desc,
    }


# ---------------------------------------------------------------------------
# Convergence checker (checkpoint-based)
# ---------------------------------------------------------------------------


def converged_ckpt(
    history_ckpt: list,
    win_ckpt: int,
    std_tol: float,
    rate_tol: float,
) -> tuple[bool, float, float]:
    """Returns (converged, std, |Δmean|) based on checkpoint history."""
    if len(history_ckpt) < 2 * win_ckpt:
        return False, float("inf"), float("inf")

    recent = np.array(history_ckpt[-win_ckpt:], dtype=float)
    prev = np.array(history_ckpt[-2 * win_ckpt:-win_ckpt], dtype=float)

    std = float(np.std(recent))
    delta = float(abs(np.mean(recent) - np.mean(prev)))
    return (std < std_tol and delta < rate_tol), std, delta
