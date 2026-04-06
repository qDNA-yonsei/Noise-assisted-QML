# =============================================================================
# main.py — top-level experiment runner
# =============================================================================

import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")


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


from src.args import parse_args, apply_args
apply_args(parse_args())

import src.config as C
from src.hamiltonian import hamiltonian_info_dict, hamiltonian_summary_string
from src.result import ExperimentResult
from src.training.seed_search import run_seed_search, diagnose_seed
from src.training.run import run_clean, run_pauli_annealing, run_shot_experiments



def _noise_axis_label(pauli_run) -> str:
    """Always plot the user-facing schedule variable as p."""
    return "p"



def _base_config_dict(ts: str, seed: int) -> dict:
    return {
        "timestamp": ts,
        "seed": seed,
        "hamiltonian": hamiltonian_info_dict(),
        "noise_mode": C.NOISE_MODE,
        "n_qubits": C.N_QUBITS,
        "n_layers": C.N_LAYERS,
        "ranges": C.RANGES,
        "pauli_optimizer": C.PAULI_OPTIMIZER,
        "clean_optimizer": C.CLEAN_OPTIMIZER,
        "shot_optimizer": C.SHOT_OPTIMIZER,
        "lr_pauli_adam": C.LR_PAULI_ADAM,
        "lr_clean_adam": C.LR_CLEAN_ADAM,
        "lr_shot_adam": C.LR_SHOT_ADAM,
        "schedule_mode": C.SCHEDULE_MODE,
        "pauli_schedule": C.PAULI_SCHEDULE,
        "pauli_min_steps": C.PAULI_MIN_STEPS,
        "pauli_max_steps": C.PAULI_MAX_STEPS,
        "pauli_check_every": C.PAULI_CHECK_EVERY,
        "pauli_window": C.PAULI_WINDOW,
        "pauli_std_tol": C.PAULI_STD_TOL,
        "pauli_rate_tol": C.PAULI_RATE_TOL,
        "clean_threshold": C.CLEAN_THRESHOLD,
        "p_max": C.P_MAX,
        "noise_decay_a": C.NOISE_DECAY_A,
        "pauli_total_steps": C.PAULI_TOTAL_STEPS,
        "shot_list": C.SHOT_LIST,
        "shot_repeats": C.SHOT_REPEATS,
        "run_seed_search": C.RUN_SEED_SEARCH,
        "search_steps": C.SEARCH_STEPS,
    }



def plot_results(r: ExperimentResult, prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))

    # Panel 1: learning curves
    ax = axes[0]
    ax.plot(np.arange(len(r.clean_run.eval_hist)), r.clean_run.eval_hist, lw=2.0, label=f"Clean ({r.clean_run.optimizer_type})")
    ax.plot(r.pauli_run.history_steps, r.pauli_run.history, lw=1.8, label=f"Regularized ({r.pauli_run.optimizer_type})")

    for sr in r.shot_runs:
        escaped = sr.escape_count
        total = sr.num_repeats
        if 0 < escaped < total:
            ax.plot(np.arange(len(sr.best_rep.eval_hist)), sr.best_rep.eval_hist, lw=1.6, label=f"Shot={sr.shots} best (rep {sr.best_rep.rep})")
            if sr.stuck_rep is not None:
                ax.plot(np.arange(len(sr.stuck_rep.eval_hist)), sr.stuck_rep.eval_hist, lw=1.2, ls="--", label=f"Shot={sr.shots} stuck (rep {sr.stuck_rep.rep})")
        else:
            ax.plot(np.arange(len(sr.best_rep.eval_hist)), sr.best_rep.eval_hist, lw=1.5, label=f"Shot={sr.shots} ({escaped}/{total} escaped, rep {sr.best_rep.rep})")

    ax.axhline(C.GLOBAL_MIN_ENERGY, color="red", ls="--", lw=1.2, label="Exact ground energy")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Clean exact energy")
    ax.set_title(f"Init seed={r.init_seed}\n{C.HAMILTONIAN_NAME}, layers={C.N_LAYERS}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 2: noise schedule
    ax = axes[1]
    ax.plot(np.arange(1, r.pauli_run.total_steps + 1), r.pauli_run.noise_values, lw=2.0)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(_noise_axis_label(r.pauli_run))
    ax.set_title(f"Regularization schedule (user-facing p, mode={r.pauli_run.noise_mode})")
    ax.grid(True, alpha=0.3)

    # Panel 3: final energy summary
    ax = axes[2]
    labels = [f"clean\n({r.clean_run.optimizer_type})", f"regularized\n({r.pauli_run.optimizer_type})"]
    values = [r.clean_run.final_eval, r.pauli_run.final_clean_eval]
    for sr in r.shot_runs:
        labels.append(f"shot={sr.shots}\nbest-min")
        values.append(sr.best_rep.min_clean_eval)
        if sr.stuck_rep is not None:
            labels.append(f"shot={sr.shots}\nstuck")
            values.append(sr.stuck_rep.final_clean_eval)

    xpos = np.arange(len(labels))
    ax.bar(xpos, values, alpha=0.8, edgecolor="black", lw=0.6)
    ax.axhline(C.GLOBAL_MIN_ENERGY, color="red", ls="--", lw=1.2, label="Exact ground energy")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Energy")
    ax.set_title("Reachability summary")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = f"{prefix}_comparison.png"
    plt.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Plot saved → {path}")



def save_results(r: ExperimentResult, prefix: str):
    np.save(f"{prefix}_clean_eval_hist.npy", r.clean_run.eval_hist)
    np.save(f"{prefix}_clean_final_params.npy", np.array(r.clean_run.final_params))
    np.save(f"{prefix}_regularized_hist.npy", r.pauli_run.history)
    np.save(f"{prefix}_regularized_hist_steps.npy", r.pauli_run.history_steps)
    np.save(f"{prefix}_regularized_noise_values.npy", r.pauli_run.noise_values)
    np.save(f"{prefix}_regularized_final_params.npy", np.array(r.pauli_run.final_params))
    np.save(f"{prefix}_init_params_seed{r.init_seed}.npy", np.array(r.seed_diagnosis.init_params))

    for sr in r.shot_runs:
        np.save(f"{prefix}_shot_{sr.shots}_best_eval_hist.npy", sr.best_rep.eval_hist)
        if sr.stuck_rep is not None:
            np.save(f"{prefix}_shot_{sr.shots}_stuck_eval_hist.npy", sr.stuck_rep.eval_hist)

    summary = {
        "config": _base_config_dict(ts="saved_with_result", seed=r.init_seed),
        "init_seed": r.init_seed,
        "seed_diagnosis": {
            "kind": r.seed_diagnosis.kind,
            "gap": r.seed_diagnosis.gap,
            "clean_final_energy": r.seed_diagnosis.clean_final_energy,
            "grad_norm": r.seed_diagnosis.grad_norm,
            "lambda_min": r.seed_diagnosis.lambda_min,
            "paper_trapped": r.seed_diagnosis.paper_trapped,
            "trap_score": r.seed_diagnosis.trap_score,
        },
        "clean_run": {
            "optimizer_type": r.clean_run.optimizer_type,
            "lr": r.clean_run.lr,
            "total_steps": r.clean_run.total_steps,
            "final_eval": r.clean_run.final_eval,
            "grad_norm": r.clean_run.grad_norm,
            "lambda_min": r.clean_run.lambda_min,
        },
        "regularized_run": {
            "optimizer_type": r.pauli_run.optimizer_type,
            "noise_mode": r.pauli_run.noise_mode,
            "schedule_mode": r.pauli_run.schedule_mode,
            "noise_label": r.pauli_run.noise_label,
            "total_steps": r.pauli_run.total_steps,
            "initial_noise_value": float(r.pauli_run.noise_values[0]),
            "final_noise_value": float(r.pauli_run.noise_values[-1]),
            "final_clean_eval": r.pauli_run.final_clean_eval,
            "final_grad_norm": r.pauli_run.final_grad_norm,
            "final_lambda_min": r.pauli_run.final_lambda_min,
        },
        "shot_runs": [
            {
                "shots": sr.shots,
                "optimizer_type": sr.optimizer_type,
                "escape_threshold": sr.escape_threshold,
                "escape_count": sr.escape_count,
                "num_repeats": sr.num_repeats,
                "mean_final_clean_eval": sr.mean_final_clean_eval,
                "std_final_clean_eval": sr.std_final_clean_eval,
                "best_rep_min_eval": sr.best_rep.min_clean_eval,
                "stuck_rep_final_eval": sr.stuck_rep.final_clean_eval if sr.stuck_rep else None,
            }
            for sr in r.shot_runs
        ],
    }
    path = f"{prefix}_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {path}")



def run_all() -> ExperimentResult:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_tag = C.PAULI_OPTIMIZER

    def _make_run_dir(seed: int):
        run_name = f"{ts}_seed{seed}_{opt_tag}"
        run_dir = os.path.join("outputs", "seed_training", run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(_base_config_dict(ts=ts, seed=seed), f, indent=2)
        return run_dir, os.path.join(run_dir, run_name)

    if C.RUN_SEED_SEARCH:
        search_result = run_seed_search()
        seed_search_out = search_result
        diag = search_result.selected
        run_dir, prefix = _make_run_dir(diag.seed)
    else:
        if C.INIT_SEED is None:
            raise ValueError("RUN_SEED_SEARCH=False but INIT_SEED is None")
        run_dir, prefix = _make_run_dir(int(C.INIT_SEED))
        diag = diagnose_seed(int(C.INIT_SEED))
        seed_search_out = None

    tee = _Tee(f"{prefix}.log")
    sys.stdout = tee

    print(f"CWD = {os.getcwd()}")
    print(f"Timestamp : {ts}")
    print(f"Seed      : {diag.seed}")
    print(f"Optimizer : {opt_tag}")
    print(f"Hamiltonian: {hamiltonian_summary_string()}")
    print(f"Noise mode: {C.NOISE_MODE}")
    print(f"Schedule mode: {C.SCHEDULE_MODE}")
    print(f"Ansatz layers: {C.N_LAYERS} | SEL ranges={C.RANGES}")
    if C.NOISE_MODE == "matched":
        print("Paper protocol: matched channel is reproduced exactly, using the user-facing p and internal qml.PauliError(p/2)")
    else:
        print("Legacy layerwise mode: direct PauliError(p) variant")

    if C.SCHEDULE_MODE == "manual":
        print(
            f"Manual schedule: p_schedule={C.PAULI_SCHEDULE}, min_steps={C.PAULI_MIN_STEPS}, "
            f"max_steps={C.PAULI_MAX_STEPS}, check_every={C.PAULI_CHECK_EVERY}, clean_threshold={C.CLEAN_THRESHOLD}"
        )
    else:
        print(f"Fixed-step schedule: p_max={C.P_MAX}, decay_a={C.NOISE_DECAY_A}, total_steps={C.PAULI_TOTAL_STEPS}")
    print(f"\nUsing seed={diag.seed} | kind={diag.kind} | gap={diag.gap:.6f}")

    pauli_run = run_pauli_annealing(diag.init_params)
    budget = pauli_run.total_steps
    clean_run = run_clean(diag.init_params, steps=budget)
    shot_runs = run_shot_experiments(init_params=diag.init_params, steps=budget, clean_final=clean_run.final_eval)

    result = ExperimentResult(
        init_seed=diag.seed,
        seed_search=seed_search_out,
        seed_diagnosis=diag,
        clean_run=clean_run,
        pauli_run=pauli_run,
        shot_runs=shot_runs,
    )

    save_results(result, prefix)
    plot_results(result, prefix)

    print("\n" + "=" * 120)
    print(f"{'mode':>20} | {'opt':>6} | {'steps':>7} | {'final energy':>14} | {'escape':>10}")
    print("-" * 120)
    print(f"{'clean':>20} | {clean_run.optimizer_type:>6} | {budget:>7} | {clean_run.final_eval:>14.8f} | {'—':>10}")
    print(f"{'regularized':>20} | {pauli_run.optimizer_type:>6} | {budget:>7} | {pauli_run.final_clean_eval:>14.8f} | {'—':>10}")
    for sr in shot_runs:
        esc = f"{sr.escape_count}/{sr.num_repeats}"
        print(f"{'shot-'+str(sr.shots):>20} | {sr.optimizer_type:>6} | {budget:>7} | {sr.mean_final_clean_eval:>14.8f} | {esc:>10}")
    print("=" * 120)

    tee.close()
    return result


if __name__ == "__main__":
    results = run_all()
