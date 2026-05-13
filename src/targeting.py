# =============================================================================
# targeting.py — gate/parameter targeting utilities for masked Pauli noise
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from . import config as C
from .utils import to_trainable, make_stepper


@dataclass
class TargetScoreResult:
    scores: np.ndarray
    score_hist: np.ndarray
    clean_eval_hist: np.ndarray
    warmup_final_params: np.ndarray
    score_steps: int
    score_type: str


def _num_params_from_scores(scores: np.ndarray) -> int:
    return int(np.array(scores).size)


def selected_count(num_params: int, fraction: float) -> int:
    """Ceil-based target-count rule with at least one selected parameter."""
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"Target fraction must lie in (0, 1], got {fraction}.")
    return max(1, min(int(num_params), int(math.ceil(fraction * num_params))))


def compute_clean_warmup_scores(init_params, clean_cost, score_steps: int | None = None) -> TargetScoreResult:
    """
    Estimate per-parameter gate importance from a short clean warm-up trajectory.

    The warm-up is used only for scoring/mask selection. The caller should reset
    to the original init_params before running targeted Pauli-noise optimization.

    Current score type:
        mean_abs_grad: mean over warm-up of |dE_clean/dtheta_j|.

    If score_steps <= 0, one gradient is evaluated at the initial point without
    taking any optimizer step.
    """
    steps = int(C.TARGET_SCORE_STEPS if score_steps is None else score_steps)
    score_type = str(C.TARGET_SCORE_TYPE)
    if score_type != "mean_abs_grad":
        raise ValueError(f"Unsupported TARGET_SCORE_TYPE={score_type!r}; currently only 'mean_abs_grad' is implemented.")

    params = to_trainable(init_params)
    grad_fn = qml.grad(clean_cost)

    grad_abs_hist = []
    clean_eval_hist = [float(clean_cost(params))]

    if steps > 0:
        step_fn, _ = make_stepper(
            cost_fn=clean_cost,
            optimizer_type=C.CLEAN_OPTIMIZER,
            max_steps_for_spsa=steps,
            lr_gd=C.LR_CLEAN_GD,
            lr_adam=C.LR_CLEAN_ADAM,
            spsa_alpha=C.CLEAN_SPSA_ALPHA,
            spsa_gamma=C.CLEAN_SPSA_GAMMA,
            spsa_c=C.CLEAN_SPSA_C,
            spsa_A=C.CLEAN_SPSA_A,
            spsa_a=C.CLEAN_SPSA_a,
        )
        for _ in range(steps):
            g = np.array(grad_fn(params), dtype=float)
            grad_abs_hist.append(np.abs(g))
            params = step_fn(params)
            clean_eval_hist.append(float(clean_cost(params)))
    else:
        g = np.array(grad_fn(params), dtype=float)
        grad_abs_hist.append(np.abs(g))

    score_hist = np.array(grad_abs_hist, dtype=float)
    scores = np.mean(score_hist, axis=0)

    return TargetScoreResult(
        scores=np.array(scores, dtype=float),
        score_hist=score_hist,
        clean_eval_hist=np.array(clean_eval_hist, dtype=float),
        warmup_final_params=to_trainable(params),
        score_steps=steps,
        score_type=score_type,
    )


def make_target_mask(
    scores: np.ndarray,
    *,
    selection_mode: str,
    fraction: float,
    random_seed: int | None = None,
) -> np.ndarray:
    """Create a binary mask with the same shape as scores."""
    mode = str(selection_mode).lower()
    if mode not in {"top", "bottom", "random", "all"}:
        raise ValueError(f"selection_mode must be top, bottom, random, or all; got {selection_mode!r}.")

    scores_arr = np.array(scores, dtype=float)
    flat = scores_arr.reshape(-1)
    n = flat.size

    if mode == "all" or float(fraction) >= 1.0:
        mask_flat = np.ones(n, dtype=float)
        return mask_flat.reshape(scores_arr.shape)

    k = selected_count(n, fraction)
    if mode == "top":
        # stable deterministic ordering: largest scores first, then lower index
        order = np.lexsort((np.arange(n), -flat))
        chosen = order[:k]
    elif mode == "bottom":
        order = np.lexsort((np.arange(n), flat))
        chosen = order[:k]
    else:
        if random_seed is None:
            raise ValueError("random_seed must be provided for selection_mode='random'.")
        rng = np.random.default_rng(int(random_seed))
        chosen = rng.choice(n, size=k, replace=False)

    mask_flat = np.zeros(n, dtype=float)
    mask_flat[np.array(chosen, dtype=int)] = 1.0
    return mask_flat.reshape(scores_arr.shape)


def target_method_specs(*, include_shots: bool = True) -> list[dict]:
    """
    Build the targeted-Pauli method list.

    Defaults implement:
      top 10%, top 50%, top 100%; bottom/random 10%, 50%.
    Top 100% is the all-gate Pauli-noise baseline.
    """
    specs: list[dict] = [
        {
            "method_order": 1,
            "method_key": "clean",
            "method_label": "clean",
            "method_group": "clean",
            "schedule_mode": "",
            "shots": "",
            "target_selection_mode": "",
            "target_fraction": "",
        }
    ]
    next_order = 2

    schedule_modes = list(C.TARGET_SUITE_SCHEDULES)
    if not schedule_modes:
        schedule_modes = [C.SCHEDULE_MODE]

    fractions = [float(x) for x in C.TARGET_FRACTIONS]
    selection_modes = [str(x).lower() for x in C.TARGET_SELECTION_MODES]

    # Ensure the intended default family even if the user overrides the order.
    for schedule_mode in schedule_modes:
        schedule_tag = "fixed" if schedule_mode == "fixed_step" else str(schedule_mode)
        for frac in fractions:
            for mode in selection_modes:
                # Only top-100 is meaningful; random/bottom 100% would duplicate all-gate noise.
                if frac >= 1.0 and mode != "top":
                    continue
                pct = int(round(100 * frac))
                method_key = f"pauli_{mode}{pct}_{schedule_tag}"
                method_label = method_key
                specs.append({
                    "method_order": next_order,
                    "method_key": method_key,
                    "method_label": method_label,
                    "method_group": "targeted_pauli",
                    "schedule_mode": schedule_mode,
                    "shots": "",
                    "target_selection_mode": mode,
                    "target_fraction": float(frac),
                })
                next_order += 1

    if include_shots:
        for shots in C.METHOD_SUITE_SHOT_LIST:
            shots = int(shots)
            specs.append({
                "method_order": next_order,
                "method_key": f"shot_{shots}",
                "method_label": f"shot_{shots}",
                "method_group": "shot",
                "schedule_mode": "",
                "shots": shots,
                "target_selection_mode": "",
                "target_fraction": "",
            })
            next_order += 1

    return specs


def random_mask_seed(*, training_seed: int, fraction: float, sample_index: int = 0, method_key: str = "") -> int:
    """Deterministic per-training-seed random-mask seed."""
    frac_code = int(round(float(fraction) * 10_000))
    key_code = sum((i + 1) * ord(ch) for i, ch in enumerate(str(method_key))) % 100_000
    return int(C.TARGET_RANDOM_BASE_SEED + 1_000_000 * int(sample_index) + 1_000 * int(training_seed) + frac_code + key_code)


def mask_metadata(mask: np.ndarray, *, selection_mode: str, fraction: float, random_seed=None) -> dict:
    arr = np.array(mask, dtype=float)
    return {
        "target_selection_mode": selection_mode,
        "target_fraction": float(fraction),
        "target_selected_count": int(np.sum(arr > 0.5)),
        "target_total_params": int(arr.size),
        "target_random_seed": int(random_seed) if random_seed is not None else "",
        "target_score_steps": int(C.TARGET_SCORE_STEPS),
        "target_score_type": str(C.TARGET_SCORE_TYPE),
    }
