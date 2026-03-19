# =============================================================================
# training/run.py — experiment training modes
# =============================================================================
"""
Three experiment training modes, each returning its dedicated result type:

    run_clean(init_params, steps)          -> CleanRunResult
    run_pauli_annealing(init_params)       -> PauliAnnealingResult
    run_shot_experiments(init_params, steps) -> list[ShotRunResult]
"""

import time
from math import ceil

import numpy as np

import config as C
from noise import CLEAN_COST, make_pauli_cost, make_shot_cost
from result import (
    CleanRunResult,
    PauliStageResult,
    PauliAnnealingResult,
    ShotRepResult,
    ShotRunResult,
)
from utils import (
    to_trainable,
    clean_hessian_info,
    optimize_fixed_steps,
    make_stepper,
    converged_ckpt,
)


# ---------------------------------------------------------------------------
# Clean baseline
# ---------------------------------------------------------------------------

def run_clean(init_params, steps: int) -> CleanRunResult:
    """
    Training mode: noiseless fixed-step optimisation.
    Uses CLEAN_OPTIMIZER / LR_CLEAN_* from config.
    """
    print(f"\n[Clean | opt={C.CLEAN_OPTIMIZER} | steps={steps}]")
    t0 = time.time()

    res  = optimize_fixed_steps(
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
    lr   = C.LR_CLEAN_ADAM if C.CLEAN_OPTIMIZER == "adam" else C.LR_CLEAN_GD

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


# ---------------------------------------------------------------------------
# Adaptive Pauli annealing
# ---------------------------------------------------------------------------

def run_pauli_annealing(init_params) -> PauliAnnealingResult:
    """
    Training mode: adaptive Pauli annealing.
    Iterates over PAULI_SCHEDULE; advances to next noise level once converged
    (or max_steps reached).
    """
    params   = to_trainable(init_params)
    schedule = C.PAULI_SCHEDULE
    win_ckpt = max(2, int(ceil(C.PAULI_WINDOW / C.PAULI_CHECK_EVERY)))

    all_hist_ckpt  = []
    all_hist_steps = []
    stages: list[PauliStageResult] = []
    total_steps = 0

    print(f"\n[Pauli annealing | opt={C.PAULI_OPTIMIZER} | schedule={schedule}]")

    for stage_idx, noise_p in enumerate(schedule, start=1):
        use_clean = noise_p <= C.CLEAN_THRESHOLD
        cost_fn   = CLEAN_COST if use_clean else make_pauli_cost(noise_p)

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

        label = "clean" if use_clean else "noisy"
        print(f"\n  Stage {stage_idx}/{len(schedule)} | p={noise_p:.3f} ({label}) | {opt_desc}")
        print(f"  {'step':>5} | {'E_clean':>12} | {'std':>9} | {'|Δmean|':>9}")

        stage_ckpt  = []
        stage_steps = []
        step_i = 0
        t0 = time.time()

        while step_i < C.PAULI_MAX_STEPS:
            params = step_fn(params)
            step_i += 1

            if step_i % C.PAULI_CHECK_EVERY == 0:
                e_clean = float(CLEAN_COST(params))
                stage_ckpt.append(e_clean)
                stage_steps.append(total_steps + step_i)

                conv, std, delta = converged_ckpt(stage_ckpt, win_ckpt, C.PAULI_STD_TOL, C.PAULI_RATE_TOL)
                note = "<-- converged" if (step_i >= C.PAULI_MIN_STEPS and conv) else ""
                print(f"  {step_i:>5} | {e_clean:>12.6f} | {std:>9.5f} | {delta:>9.5f}  {note}")

                if step_i >= C.PAULI_MIN_STEPS and conv:
                    break

        conv, std, delta = converged_ckpt(stage_ckpt, win_ckpt, C.PAULI_STD_TOL, C.PAULI_RATE_TOL)
        if step_i >= C.PAULI_MAX_STEPS and not conv:
            print(f"  *** max_steps={C.PAULI_MAX_STEPS} reached without convergence ***")
        print(f"  Stage done: steps={step_i}, std={std:.5f}, |Δmean|={delta:.5f}, time={time.time()-t0:.1f}s")

        stages.append(PauliStageResult(
            stage_idx=stage_idx,
            noise_p=float(noise_p),
            steps_used=int(step_i),
            converged=bool(conv),
            final_std=float(std),
            final_delta=float(delta),
            eval_hist=np.array(stage_ckpt, dtype=float),
            hist_steps=np.array(stage_steps, dtype=int),
        ))
        all_hist_ckpt.extend(stage_ckpt)
        all_hist_steps.extend(stage_steps)
        total_steps += step_i

    final_eval = float(CLEAN_COST(params))
    hess = clean_hessian_info(params)

    print(f"\nAdaptive Pauli done: final_clean_eval={final_eval:.8f} | total_steps={total_steps}")

    return PauliAnnealingResult(
        optimizer_type=C.PAULI_OPTIMIZER,
        noise_schedule=list(schedule),
        stages=stages,
        total_steps=int(total_steps),
        final_clean_eval=float(final_eval),
        final_params=to_trainable(params),
        final_grad_norm=float(hess["grad_norm"]),
        final_lambda_min=float(hess["lambda_min"]),
        history=np.array(all_hist_ckpt, dtype=float),
        history_steps=np.array(all_hist_steps, dtype=int),
    )


# ---------------------------------------------------------------------------
# Shot-noise experiments
# ---------------------------------------------------------------------------

def _escape_threshold(clean_final: float) -> float:
    if C.SHOT_ESCAPE_THRESHOLD is not None:
        return float(C.SHOT_ESCAPE_THRESHOLD)
    return clean_final - C.SHOT_ESCAPE_DELTA


def run_shot_experiments(
    init_params,
    steps: int,
    clean_final: float,
) -> list[ShotRunResult]:
    """
    Training mode: shot-noise experiments.
    Runs SHOT_REPEATS repetitions for each shots value in SHOT_LIST.
    eval_fn = CLEAN_COST so trajectories are comparable to clean baseline.
    """
    threshold = _escape_threshold(clean_final)
    results: list[ShotRunResult] = []

    print(f"\n[Shot-noise | opt={C.SHOT_OPTIMIZER} | steps={steps} | escape_thr={threshold:.4f}]")

    for shots in C.SHOT_LIST:
        print(f"\n  shots = {shots}")
        reps: list[ShotRepResult] = []

        for rep in range(C.SHOT_REPEATS):
            device_seed = C.SHOT_DEVICE_BASE_SEED + 10_000 * shots + rep
            shot_cost   = make_shot_cost(shots, device_seed)
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
            min_e     = float(np.min(eval_hist))
            escaped   = min_e <= threshold

            print(
                f"    rep={rep:02d} | final_clean={res['final_eval']:.6f} | "
                f"min_clean={min_e:.6f} | escaped={escaped} | time={time.time()-t0:.1f}s"
            )
            reps.append(ShotRepResult(
                rep=int(rep),
                device_seed=int(device_seed),
                eval_hist=eval_hist,
                final_clean_eval=float(res["final_eval"]),
                min_clean_eval=min_e,
                escaped=escaped,
            ))

        finals        = np.array([r.final_clean_eval for r in reps])
        escape_count  = int(sum(r.escaped for r in reps))
        best_rep      = min(reps, key=lambda r: r.min_clean_eval)
        non_escaped   = [r for r in reps if not r.escaped]
        stuck_rep     = (
            min(non_escaped, key=lambda r: abs(r.final_clean_eval - clean_final))
            if non_escaped and escape_count > 0
            else None
        )

        results.append(ShotRunResult(
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
        ))

    return results
