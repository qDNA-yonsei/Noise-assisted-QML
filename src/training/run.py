# =============================================================================
# training/run.py — experiment training modes
# =============================================================================

import time
from math import ceil

import numpy as np

from .. import config as C
from ..noise import CLEAN_COST, REGULARIZED_COST, make_shot_cost
from ..result import CleanRunResult, PauliAnnealingResult, ShotRepResult, ShotRunResult
from ..utils import to_trainable, clean_hessian_info, optimize_fixed_steps, make_stepper, converged_ckpt


def _regularization_label() -> str:
    """
    Always expose the user-facing schedule variable as `p`.

    In matched mode this `p` is the paper regularization strength and is later
    converted internally to qml.PauliError(p/2).

    In layerwise mode this `p` is used directly as qml.PauliError(p).
    """
    return "p"


def _validate_schedule_values(values, *, label: str) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Regularization schedule must contain at least one value.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        bad = arr[(arr < 0.0) | (arr > 1.0)]
        raise ValueError(f"All scheduled {label} values must lie in [0, 1]. Bad values: {bad.tolist()}")
    return arr


def _fixed_step_schedule(total_steps: int) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError(f"PAULI_TOTAL_STEPS must be positive, got {total_steps}.")

    label = _regularization_label()
    max_value = C.P_MAX
    if not (0.0 <= max_value <= 1.0):
        raise ValueError(f"Maximum {label} must lie in [0, 1], got {max_value}.")

    denom = max(total_steps - 1, 1)
    idx = np.arange(total_steps, dtype=float)
    values = max_value * np.exp(-C.NOISE_DECAY_A * idx / denom)
    return _validate_schedule_values(values, label=label)


def run_clean(init_params, steps: int) -> CleanRunResult:
    print(f"\n[Clean | opt={C.CLEAN_OPTIMIZER} | steps={steps}]")
    t0 = time.time()

    res = optimize_fixed_steps(
        cost_fn=CLEAN_COST,
        params0=init_params,
        steps=steps,
        optimizer_type=C.CLEAN_OPTIMIZER,
        lr_gd=C.LR_CLEAN_GD,
        lr_adam=C.LR_CLEAN_ADAM,
        spsa_alpha=C.CLEAN_SPSA_ALPHA,
        spsa_gamma=C.CLEAN_SPSA_GAMMA,
        spsa_c=C.CLEAN_SPSA_C,
        spsa_A=C.CLEAN_SPSA_A,
        spsa_a=C.CLEAN_SPSA_a,
        eval_fn=CLEAN_COST,
    )
    hess = clean_hessian_info(res["final_params"])
    lr = C.LR_CLEAN_ADAM if C.CLEAN_OPTIMIZER == "adam" else C.LR_CLEAN_GD

    print(f"  final energy = {res['final_eval']:.8f}  |  time = {time.time()-t0:.1f}s")

    return CleanRunResult(
        optimizer_type=C.CLEAN_OPTIMIZER,
        lr=lr,
        total_steps=steps,
        final_eval=float(res["final_eval"]),
        eval_hist=res["eval_hist"],
        final_params=res["final_params"],
        grad_norm=float(hess["grad_norm"]),
        lambda_min=float(hess["lambda_min"]),
        lambda_max=float(hess["lambda_max"]),
    )


def _run_regularized_manual(init_params) -> PauliAnnealingResult:
    """
    Manual/adaptive schedule.

    Uses the directly specified schedule in C.PAULI_SCHEDULE and advances to the
    next stage when convergence is detected, matching the original project flow.

    IMPORTANT USER-FACING CONVENTION
    --------------------------------
    The schedule values are always called `p` in both modes.

    - matched  : user-facing p -> internal qml.PauliError(p/2)
    - layerwise: user-facing p -> internal qml.PauliError(p)
    """
    params = to_trainable(init_params)
    noise_label = _regularization_label()
    schedule = _validate_schedule_values(C.PAULI_SCHEDULE, label=noise_label)
    win_ckpt = max(2, int(ceil(C.PAULI_WINDOW / C.PAULI_CHECK_EVERY)))

    all_hist_ckpt = []
    all_hist_steps = []
    noise_values_per_step: list[float] = []
    total_steps = 0

    print(
        f"\n[Regularized optimization | mode={C.NOISE_MODE} | schedule_mode=manual | "
        f"{noise_label}_schedule={schedule.tolist()} | opt={C.PAULI_OPTIMIZER}]"
    )
    if C.NOISE_MODE == "matched":
        print("  Note: in matched mode, each user-facing p is converted internally to qml.PauliError(p/2).")
    else:
        print("  Note: in layerwise mode, each user-facing p is passed directly to qml.PauliError(p).")

    for stage_idx, noise_value in enumerate(schedule, start=1):
        use_clean = float(noise_value) <= float(C.CLEAN_THRESHOLD)
        if use_clean:
            cost_fn = CLEAN_COST
            label_text = f"clean (threshold={C.CLEAN_THRESHOLD})"
        else:
            current_noise = {"value": float(noise_value)}

            def cost_fn(weights):
                return REGULARIZED_COST(weights, current_noise["value"])

            label_text = "regularized"

        step_fn, opt_desc = make_stepper(
            cost_fn=cost_fn,
            optimizer_type=C.PAULI_OPTIMIZER,
            max_steps_for_spsa=C.PAULI_MAX_STEPS,
            lr_gd=C.LR_PAULI_GD,
            lr_adam=C.LR_PAULI_ADAM,
            spsa_alpha=C.PAULI_SPSA_ALPHA,
            spsa_gamma=C.PAULI_SPSA_GAMMA,
            spsa_c=C.PAULI_SPSA_C,
            spsa_A=C.PAULI_SPSA_A,
            spsa_a=C.PAULI_SPSA_a,
        )

        print(
            f"\n  Stage {stage_idx}/{len(schedule)} | {noise_label}={float(noise_value):.6f} | "
            f"{label_text} | {opt_desc}"
        )
        print(f"  {'step':>5} | {'E_clean':>12} | {'std':>9} | {'|Δmean|':>9}")

        stage_ckpt = []
        step_i = 0
        t0 = time.time()

        while step_i < C.PAULI_MAX_STEPS:
            params = step_fn(params)
            step_i += 1
            noise_values_per_step.append(float(noise_value))

            if step_i % C.PAULI_CHECK_EVERY == 0:
                e_clean = float(CLEAN_COST(params))
                stage_ckpt.append(e_clean)
                all_hist_ckpt.append(e_clean)
                all_hist_steps.append(total_steps + step_i)

                conv, std, delta = converged_ckpt(stage_ckpt, win_ckpt, C.PAULI_STD_TOL, C.PAULI_RATE_TOL)
                note = "<-- converged" if (step_i >= C.PAULI_MIN_STEPS and conv) else ""
                print(f"  {step_i:>5} | {e_clean:>12.6f} | {std:>9.5f} | {delta:>9.5f}  {note}")

                if step_i >= C.PAULI_MIN_STEPS and conv:
                    break

        conv, std, delta = converged_ckpt(stage_ckpt, win_ckpt, C.PAULI_STD_TOL, C.PAULI_RATE_TOL)
        if step_i >= C.PAULI_MAX_STEPS and not conv:
            print(f"  *** max_steps={C.PAULI_MAX_STEPS} reached without convergence ***")
        print(f"  Stage done: steps={step_i}, std={std:.5f}, |Δmean|={delta:.5f}, time={time.time()-t0:.1f}s")

        total_steps += step_i

    final_eval = float(CLEAN_COST(params))
    hess = clean_hessian_info(params)

    print(f"\nManual/adaptive regularized run done: final_clean_eval={final_eval:.8f} | total_steps={total_steps}")

    return PauliAnnealingResult(
        optimizer_type=C.PAULI_OPTIMIZER,
        noise_mode=C.NOISE_MODE,
        schedule_mode="manual",
        noise_label=noise_label,
        noise_values=np.array(noise_values_per_step, dtype=float),
        total_steps=int(total_steps),
        final_clean_eval=final_eval,
        final_params=to_trainable(params),
        final_grad_norm=float(hess["grad_norm"]),
        final_lambda_min=float(hess["lambda_min"]),
        history=np.array(all_hist_ckpt, dtype=float),
        history_steps=np.array(all_hist_steps, dtype=int),
    )


def _run_regularized_fixed_step(init_params) -> PauliAnnealingResult:
    """
    Paper-style fixed-step regularized optimisation.

    Uses a continuously decaying schedule
        p(i) = p_max * exp(-a * i / i_max)
    over a fixed number of optimisation steps.

    IMPORTANT USER-FACING CONVENTION
    --------------------------------
    The schedule variable is always exposed as `p`.

    - matched  : user-facing p -> internal qml.PauliError(p/2)
    - layerwise: user-facing p -> internal qml.PauliError(p)
    """
    params = to_trainable(init_params)
    total_steps = int(C.PAULI_TOTAL_STEPS)
    noise_values = _fixed_step_schedule(total_steps)
    noise_label = _regularization_label()

    current_noise = {"value": float(noise_values[0])}

    def dynamic_cost(weights):
        return REGULARIZED_COST(weights, current_noise["value"])

    step_fn, opt_desc = make_stepper(
        cost_fn=dynamic_cost,
        optimizer_type=C.PAULI_OPTIMIZER,
        max_steps_for_spsa=total_steps,
        lr_gd=C.LR_PAULI_GD,
        lr_adam=C.LR_PAULI_ADAM,
        spsa_alpha=C.PAULI_SPSA_ALPHA,
        spsa_gamma=C.PAULI_SPSA_GAMMA,
        spsa_c=C.PAULI_SPSA_C,
        spsa_A=C.PAULI_SPSA_A,
        spsa_a=C.PAULI_SPSA_a,
    )

    print(
        f"\n[Regularized optimization | mode={C.NOISE_MODE} | schedule_mode=fixed_step | "
        f"{noise_label}_max={noise_values[0]:.6f} | decay_a={C.NOISE_DECAY_A} | steps={total_steps} | {opt_desc}]"
    )
    if C.NOISE_MODE == "matched":
        print("  Note: in matched mode, each user-facing p is converted internally to qml.PauliError(p/2).")
    else:
        print("  Note: in layerwise mode, each user-facing p is passed directly to qml.PauliError(p).")
    print(f"  {'step':>5} | {noise_label:>10} | {'E_clean':>12}")

    history = [float(CLEAN_COST(params))]
    history_steps = [0]

    t0 = time.time()
    for step_idx in range(total_steps):
        current_noise["value"] = float(noise_values[step_idx])
        params = step_fn(params)

        clean_eval = float(CLEAN_COST(params))
        history.append(clean_eval)
        history_steps.append(step_idx + 1)

        if ((step_idx + 1) % C.PAULI_LOG_EVERY == 0) or (step_idx == 0) or (step_idx + 1 == total_steps):
            print(f"  {step_idx + 1:>5} | {current_noise['value']:>10.6f} | {clean_eval:>12.6f}")

    final_eval = float(history[-1])
    hess = clean_hessian_info(params)

    print(f"  done in {time.time()-t0:.1f}s | final_clean_eval={final_eval:.8f}")

    return PauliAnnealingResult(
        optimizer_type=C.PAULI_OPTIMIZER,
        noise_mode=C.NOISE_MODE,
        schedule_mode="fixed_step",
        noise_label=noise_label,
        noise_values=np.array(noise_values, dtype=float),
        total_steps=total_steps,
        final_clean_eval=final_eval,
        final_params=to_trainable(params),
        final_grad_norm=float(hess["grad_norm"]),
        final_lambda_min=float(hess["lambda_min"]),
        history=np.array(history, dtype=float),
        history_steps=np.array(history_steps, dtype=int),
    )


def run_pauli_annealing(init_params) -> PauliAnnealingResult:
    if C.SCHEDULE_MODE == "manual":
        return _run_regularized_manual(init_params)
    if C.SCHEDULE_MODE == "fixed_step":
        return _run_regularized_fixed_step(init_params)
    raise ValueError(f"Unknown SCHEDULE_MODE={C.SCHEDULE_MODE!r}. Choose 'manual' or 'fixed_step'.")


def _escape_threshold(clean_final: float) -> float:
    if C.SHOT_ESCAPE_THRESHOLD is not None:
        return float(C.SHOT_ESCAPE_THRESHOLD)
    return clean_final - C.SHOT_ESCAPE_DELTA


def run_shot_experiments(init_params, steps: int, clean_final: float) -> list[ShotRunResult]:
    threshold = _escape_threshold(clean_final)
    results: list[ShotRunResult] = []

    print(f"\n[Shot-noise | opt={C.SHOT_OPTIMIZER} | steps={steps} | escape_thr={threshold:.4f}]")

    for shots in C.SHOT_LIST:
        print(f"\n  shots = {shots}")
        reps: list[ShotRepResult] = []

        for rep in range(C.SHOT_REPEATS):
            device_seed = C.SHOT_DEVICE_BASE_SEED + 10_000 * shots + rep
            shot_cost = make_shot_cost(shots, device_seed)
            t0 = time.time()

            res = optimize_fixed_steps(
                cost_fn=shot_cost,
                params0=init_params,
                steps=steps,
                optimizer_type=C.SHOT_OPTIMIZER,
                lr_gd=C.LR_SHOT_GD,
                lr_adam=C.LR_SHOT_ADAM,
                spsa_alpha=C.SHOT_SPSA_ALPHA,
                spsa_gamma=C.SHOT_SPSA_GAMMA,
                spsa_c=C.SHOT_SPSA_C,
                spsa_A=C.SHOT_SPSA_A,
                spsa_a=C.SHOT_SPSA_a,
                eval_fn=CLEAN_COST,
            )
            eval_hist = res["eval_hist"]
            min_e = float(np.min(eval_hist))
            escaped = min_e <= threshold

            print(
                f"    rep={rep:02d} | final_clean={res['final_eval']:.6f} | "
                f"min_clean={min_e:.6f} | escaped={escaped} | time={time.time()-t0:.1f}s"
            )
            reps.append(
                ShotRepResult(
                    rep=int(rep),
                    device_seed=int(device_seed),
                    eval_hist=eval_hist,
                    final_clean_eval=float(res["final_eval"]),
                    min_clean_eval=min_e,
                    escaped=escaped,
                )
            )

        finals = np.array([r.final_clean_eval for r in reps])
        escape_count = int(sum(r.escaped for r in reps))
        best_rep = min(reps, key=lambda r: r.min_clean_eval)
        non_escaped = [r for r in reps if not r.escaped]
        stuck_rep = (
            min(non_escaped, key=lambda r: abs(r.final_clean_eval - clean_final))
            if non_escaped and escape_count > 0
            else None
        )

        results.append(
            ShotRunResult(
                shots=int(shots),
                optimizer_type=C.SHOT_OPTIMIZER,
                escape_threshold=float(threshold),
                repeats=reps,
                escape_count=escape_count,
                num_repeats=int(C.SHOT_REPEATS),
                mean_final_clean_eval=float(finals.mean()),
                std_final_clean_eval=float(finals.std()),
                best_rep=best_rep,
                stuck_rep=stuck_rep,
            )
        )

    return results
