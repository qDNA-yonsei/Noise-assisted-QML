# =============================================================================
# training/targeted_pauli.py — targeted Pauli-noise standard experiment
# =============================================================================

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from .. import config as C
from ..hamiltonian import hamiltonian_info_dict, hamiltonian_summary_string
from ..noise import CLEAN_COST
from ..targeting import (
    compute_clean_warmup_scores,
    make_target_mask,
    mask_metadata,
    random_mask_seed,
    target_method_specs,
)
from .seed_search import run_seed_search, diagnose_seed
from .run import run_clean, run_pauli_annealing, run_shot_experiments


class _Tee:
    def __init__(self, filepath):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def _config_dict(ts: str, seed: int) -> dict:
    return {
        "timestamp": ts,
        "seed": int(seed),
        "experiment_mode": C.EXPERIMENT_MODE,
        "hamiltonian": hamiltonian_info_dict(),
        "noise_mode": C.NOISE_MODE,
        "n_qubits": C.N_QUBITS,
        "n_layers": C.N_LAYERS,
        "ranges": C.RANGES,
        "schedule_mode": C.SCHEDULE_MODE,
        "pauli_schedule": C.PAULI_SCHEDULE,
        "p_max": C.P_MAX,
        "noise_decay_a": C.NOISE_DECAY_A,
        "pauli_total_steps": C.PAULI_TOTAL_STEPS,
        "target_score_steps": C.TARGET_SCORE_STEPS,
        "target_score_type": C.TARGET_SCORE_TYPE,
        "target_fractions": C.TARGET_FRACTIONS,
        "target_selection_modes": C.TARGET_SELECTION_MODES,
        "target_random_base_seed": C.TARGET_RANDOM_BASE_SEED,
        "target_reset_after_scoring": True,
    }


def _target_specs_for_standard() -> list[dict]:
    # Use target_method_specs for consistency, but only with current C.SCHEDULE_MODE.
    old_schedules = list(C.TARGET_SUITE_SCHEDULES)
    C.TARGET_SUITE_SCHEDULES = [C.SCHEDULE_MODE]
    try:
        specs = target_method_specs(include_shots=False)
    finally:
        C.TARGET_SUITE_SCHEDULES = old_schedules
    return [s for s in specs if s["method_group"] == "targeted_pauli"]


def _plot_targeted_standard(run_dir: str, seed: int, clean_run, pauli_results: list[dict], shot_runs: list) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))

    ax = axes[0]
    ax.plot(np.arange(len(clean_run.eval_hist)), clean_run.eval_hist, lw=2.0, label="clean")
    for rec in pauli_results:
        r = rec["result"]
        ax.plot(r.history_steps, r.history, lw=1.5, label=rec["method_key"])
    for sr in shot_runs:
        ax.plot(np.arange(len(sr.best_rep.eval_hist)), sr.best_rep.eval_hist, lw=1.2, ls="--", label=f"shot_{sr.shots}_best")
    ax.axhline(C.GLOBAL_MIN_ENERGY, color="red", ls="--", lw=1.1, label="exact ground")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("clean exact energy")
    ax.set_title(f"Targeted Pauli-noise trajectories | seed={seed}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = ["clean"] + [rec["method_key"] for rec in pauli_results]
    values = [clean_run.final_eval] + [rec["result"].final_clean_eval for rec in pauli_results]
    xpos = np.arange(len(labels))
    ax.bar(xpos, values, edgecolor="black", lw=0.5)
    ax.axhline(C.GLOBAL_MIN_ENERGY, color="red", ls="--", lw=1.1)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("final clean energy")
    ax.set_title("Final-energy comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(run_dir, "targeted_standard_comparison.png")
    plt.savefig(out, dpi=170)
    plt.close(fig)
    return out


def run_targeted_standard() -> dict:
    if C.NOISE_MODE != "matched":
        raise ValueError("targeted_standard currently supports only --noise-mode matched.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if C.RUN_SEED_SEARCH:
        search_result = run_seed_search()
        diag = search_result.selected
    else:
        if C.INIT_SEED is None:
            raise ValueError("RUN_SEED_SEARCH=False but INIT_SEED is None")
        search_result = None
        diag = diagnose_seed(int(C.INIT_SEED))

    run_name = f"{ts}_targeted_seed{diag.seed}_{C.PAULI_OPTIMIZER}_{C.SCHEDULE_MODE}"
    run_dir = os.path.join("outputs", "targeted_pauli", run_name)
    os.makedirs(run_dir, exist_ok=True)

    tee = _Tee(os.path.join(run_dir, f"{run_name}.log"))
    sys.stdout = tee

    print("=" * 100)
    print("Targeted Pauli-noise standard mode")
    print(f"Timestamp      : {ts}")
    print(f"Seed           : {diag.seed}")
    print(f"Hamiltonian    : {hamiltonian_summary_string()}")
    print(f"Noise mode     : {C.NOISE_MODE} (targeted masks supported)")
    print(f"Schedule mode  : {C.SCHEDULE_MODE}")
    print(f"Score warm-up  : {C.TARGET_SCORE_STEPS} clean steps, score={C.TARGET_SCORE_TYPE}")
    print("Warm-up is used only for mask selection; all targeted runs reset to the original init params.")
    print("=" * 100)

    np.save(os.path.join(run_dir, "initial_params.npy"), np.array(diag.init_params))

    score_result = compute_clean_warmup_scores(diag.init_params, CLEAN_COST, C.TARGET_SCORE_STEPS)
    scores = score_result.scores
    np.save(os.path.join(run_dir, "target_scores.npy"), scores)
    np.save(os.path.join(run_dir, "target_score_hist.npy"), score_result.score_hist)
    np.save(os.path.join(run_dir, "target_warmup_clean_eval_hist.npy"), score_result.clean_eval_hist)

    flat_scores = scores.reshape(-1)
    print("\nTarget score summary")
    print(f"  total parameters = {flat_scores.size}")
    print(f"  min/mean/max score = {flat_scores.min():.6e} / {flat_scores.mean():.6e} / {flat_scores.max():.6e}")

    specs = _target_specs_for_standard()
    pauli_results = []
    summary_methods = []

    for spec in specs:
        mode = spec["target_selection_mode"]
        frac = float(spec["target_fraction"])
        rseed = None
        if mode == "random":
            rseed = random_mask_seed(training_seed=diag.seed, fraction=frac, sample_index=0, method_key=spec["method_key"])
        mask = make_target_mask(scores, selection_mode=mode, fraction=frac, random_seed=rseed)
        meta = mask_metadata(mask, selection_mode=mode, fraction=frac, random_seed=rseed)

        if C.TARGET_SAVE_MASKS:
            np.save(os.path.join(run_dir, f"{spec['method_key']}_mask.npy"), mask)

        print(
            f"\nRunning {spec['method_key']} | selected={meta['target_selected_count']}/{meta['target_total_params']} "
            f"| mode={mode} | fraction={frac} | random_seed={rseed if rseed is not None else '-'}"
        )
        result = run_pauli_annealing(diag.init_params, noise_mask=mask, run_label=spec["method_key"])
        pauli_results.append({"spec": spec, "method_key": spec["method_key"], "result": result, "mask": mask, "metadata": meta})
        summary_methods.append({
            **spec,
            **meta,
            "total_steps": int(result.total_steps),
            "final_clean_eval": float(result.final_clean_eval),
            "final_grad_norm": float(result.final_grad_norm),
            "final_lambda_min": float(result.final_lambda_min),
        })
        np.save(os.path.join(run_dir, f"{spec['method_key']}_history.npy"), result.history)
        np.save(os.path.join(run_dir, f"{spec['method_key']}_history_steps.npy"), result.history_steps)
        np.save(os.path.join(run_dir, f"{spec['method_key']}_noise_values.npy"), result.noise_values)
        np.save(os.path.join(run_dir, f"{spec['method_key']}_final_params.npy"), np.array(result.final_params))

    clean_budget = max([int(rec["result"].total_steps) for rec in pauli_results], default=int(C.PAULI_TOTAL_STEPS))
    clean_run = run_clean(diag.init_params, steps=clean_budget)
    np.save(os.path.join(run_dir, "clean_eval_hist.npy"), clean_run.eval_hist)
    np.save(os.path.join(run_dir, "clean_final_params.npy"), np.array(clean_run.final_params))

    shot_runs = []
    if C.TARGET_INCLUDE_SHOTS:
        shot_runs = run_shot_experiments(diag.init_params, steps=clean_budget, clean_final=clean_run.final_eval)

    plot_path = _plot_targeted_standard(run_dir, diag.seed, clean_run, pauli_results, shot_runs)

    summary = {
        "config": _config_dict(ts, diag.seed),
        "seed_search_used": search_result is not None,
        "seed_diagnosis": {
            "seed": int(diag.seed),
            "kind": diag.kind,
            "gap": float(diag.gap),
            "clean_final_energy": float(diag.clean_final_energy),
            "grad_norm": float(diag.grad_norm),
            "lambda_min": float(diag.lambda_min),
            "trap_score": float(diag.trap_score),
        },
        "target_score_summary": {
            "score_steps": int(score_result.score_steps),
            "score_type": score_result.score_type,
            "total_parameters": int(flat_scores.size),
            "min_score": float(flat_scores.min()),
            "mean_score": float(flat_scores.mean()),
            "max_score": float(flat_scores.max()),
        },
        "clean_run": {
            "total_steps": int(clean_run.total_steps),
            "final_eval": float(clean_run.final_eval),
            "grad_norm": float(clean_run.grad_norm),
            "lambda_min": float(clean_run.lambda_min),
        },
        "targeted_runs": summary_methods,
        "plot_path": plot_path,
    }
    with open(os.path.join(run_dir, "targeted_standard_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 100)
    print("Targeted standard summary")
    print(f"{'method':<24} | {'selected':>9} | {'steps':>7} | {'final clean energy':>18}")
    print("-" * 100)
    print(f"{'clean':<24} | {'-':>9} | {clean_run.total_steps:>7} | {clean_run.final_eval:>18.8f}")
    for row in summary_methods:
        sel = f"{row['target_selected_count']}/{row['target_total_params']}"
        print(f"{row['method_key']:<24} | {sel:>9} | {row['total_steps']:>7} | {row['final_clean_eval']:>18.8f}")
    print(f"Saved outputs -> {run_dir}")
    print("=" * 100)

    tee.close()
    return {"run_dir": run_dir, "summary_path": os.path.join(run_dir, "targeted_standard_summary.json"), "plot_path": plot_path}
