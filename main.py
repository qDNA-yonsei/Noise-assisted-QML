# =============================================================================
# main.py — top-level experiment runner
# =============================================================================

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class _Tee:
    """stdout을 화면과 파일 양쪽에 동시 출력."""
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

# args → config 패치를 먼저 해야 이후 모듈들이 올바른 값으로 초기화됨
from src.args import parse_args, apply_args
apply_args(parse_args())

import src.config as C
from src.result import ExperimentResult
from src.training.seed_search import run_seed_search, diagnose_seed
from src.training.run import run_clean, run_pauli_annealing, run_shot_experiments


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_results(r: ExperimentResult, prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))

    # --- Panel 1: learning curves ---
    ax = axes[0]
    x_clean = np.arange(len(r.clean_run.eval_hist))
    ax.plot(x_clean, r.clean_run.eval_hist, lw=2.0,
            label=f"Clean ({r.clean_run.optimizer_type})")
    ax.plot(r.pauli_run.history_steps, r.pauli_run.history, "o-",
            lw=1.8, ms=4, label=f"Adaptive Pauli ({r.pauli_run.optimizer_type})")

    for sr in r.shot_runs:
        escaped = sr.escape_count
        total   = sr.num_repeats
        if escaped > 0 and escaped < total:
            # show best + a stuck representative
            ax.plot(np.arange(len(sr.best_rep.eval_hist)), sr.best_rep.eval_hist,
                    lw=1.6, label=f"Shot={sr.shots} best (rep {sr.best_rep.rep})")
            if sr.stuck_rep is not None:
                ax.plot(np.arange(len(sr.stuck_rep.eval_hist)), sr.stuck_rep.eval_hist,
                        lw=1.2, ls="--", label=f"Shot={sr.shots} stuck (rep {sr.stuck_rep.rep})")
        else:
            label = (
                f"Shot={sr.shots} ({escaped}/{total} escaped, rep {sr.best_rep.rep})"
            )
            ax.plot(np.arange(len(sr.best_rep.eval_hist)), sr.best_rep.eval_hist,
                    lw=1.5, label=label)

    stage_ends = np.cumsum([s.steps_used for s in r.pauli_run.stages])
    for end in stage_ends[:-1]:
        ax.axvline(end, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(C.GLOBAL_MIN_ENERGY, color="red", ls="--", lw=1.2, label="Global min")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Clean exact energy")
    ax.set_title(f"Init seed={r.init_seed}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- Panel 2: Pauli stage steps ---
    ax = axes[1]
    x = np.arange(len(r.pauli_run.stages))
    colors = ["green" if s.converged else "red" for s in r.pauli_run.stages]
    bars = ax.bar(x, [s.steps_used for s in r.pauli_run.stages],
                  color=colors, alpha=0.75, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={s.noise_p}" for s in r.pauli_run.stages],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Steps used")
    ax.set_title("Adaptive Pauli stage steps\n(green=converged, red=max_steps hit)")
    for bar, s in zip(bars, r.pauli_run.stages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(s.steps_used), ha="center", va="bottom", fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 3: final energy bar chart ---
    ax = axes[2]
    labels = [
        f"clean\n({r.clean_run.optimizer_type})",
        f"pauli\n({r.pauli_run.optimizer_type})",
    ]
    values = [r.clean_run.final_eval, r.pauli_run.final_clean_eval]
    for sr in r.shot_runs:
        labels.append(f"shot={sr.shots}\nbest-min")
        values.append(sr.best_rep.min_clean_eval)
        if sr.stuck_rep is not None:
            labels.append(f"shot={sr.shots}\nstuck")
            values.append(sr.stuck_rep.final_clean_eval)

    xpos = np.arange(len(labels))
    ax.bar(xpos, values, alpha=0.8, edgecolor="black", lw=0.6)
    ax.axhline(C.GLOBAL_MIN_ENERGY, color="red", ls="--", lw=1.2, label="Global min")
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


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(r: ExperimentResult, prefix: str):
    np.save(f"{prefix}_clean_eval_hist.npy",     r.clean_run.eval_hist)
    np.save(f"{prefix}_clean_final_params.npy",  np.array(r.clean_run.final_params))
    np.save(f"{prefix}_pauli_hist.npy",           r.pauli_run.history)
    np.save(f"{prefix}_pauli_hist_steps.npy",     r.pauli_run.history_steps)
    np.save(f"{prefix}_pauli_final_params.npy",   np.array(r.pauli_run.final_params))
    np.save(f"{prefix}_init_params_seed{r.init_seed}.npy", np.array(r.seed_diagnosis.final_params))

    for sr in r.shot_runs:
        np.save(f"{prefix}_shot_{sr.shots}_best_eval_hist.npy",
                sr.best_rep.eval_hist)
        if sr.stuck_rep is not None:
            np.save(f"{prefix}_shot_{sr.shots}_stuck_eval_hist.npy",
                    sr.stuck_rep.eval_hist)

    # JSON summary (no numpy arrays)
    summary = {
        "init_seed": r.init_seed,
        "seed_diagnosis": {
            "kind":               r.seed_diagnosis.kind,
            "gap":                r.seed_diagnosis.gap,
            "clean_final_energy": r.seed_diagnosis.clean_final_energy,
            "grad_norm":          r.seed_diagnosis.grad_norm,
            "lambda_min":         r.seed_diagnosis.lambda_min,
            "paper_trapped":      r.seed_diagnosis.paper_trapped,
            "trap_score":         r.seed_diagnosis.trap_score,
        },
        "clean_run": {
            "optimizer_type":  r.clean_run.optimizer_type,
            "lr":              r.clean_run.lr,
            "total_steps":     r.clean_run.total_steps,
            "final_eval":      r.clean_run.final_eval,
            "grad_norm":       r.clean_run.grad_norm,
            "lambda_min":      r.clean_run.lambda_min,
        },
        "pauli_run": {
            "optimizer_type":   r.pauli_run.optimizer_type,
            "noise_schedule":   r.pauli_run.noise_schedule,
            "total_steps":      r.pauli_run.total_steps,
            "final_clean_eval": r.pauli_run.final_clean_eval,
            "final_grad_norm":  r.pauli_run.final_grad_norm,
            "final_lambda_min": r.pauli_run.final_lambda_min,
            "stages": [
                {
                    "noise_p":     s.noise_p,
                    "steps_used":  s.steps_used,
                    "converged":   s.converged,
                    "final_std":   s.final_std,
                    "final_delta": s.final_delta,
                }
                for s in r.pauli_run.stages
            ],
        },
        "shot_runs": [
            {
                "shots":                  sr.shots,
                "optimizer_type":         sr.optimizer_type,
                "escape_threshold":       sr.escape_threshold,
                "escape_count":           sr.escape_count,
                "num_repeats":            sr.num_repeats,
                "mean_final_clean_eval":  sr.mean_final_clean_eval,
                "std_final_clean_eval":   sr.std_final_clean_eval,
                "best_rep_min_eval":      sr.best_rep.min_clean_eval,
                "stuck_rep_final_eval":   sr.stuck_rep.final_clean_eval if sr.stuck_rep else None,
            }
            for sr in r.shot_runs
        ],
    }
    path = f"{prefix}_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all() -> ExperimentResult:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Choose initial seed
    if C.RUN_SEED_SEARCH:
        search_result   = run_seed_search()
        seed_search_out = search_result
        diag            = search_result.selected
    else:
        if C.INIT_SEED is None:
            raise ValueError("RUN_SEED_SEARCH=False but INIT_SEED is None")
        diag            = diagnose_seed(int(C.INIT_SEED))
        seed_search_out = None

    # prefix에 timestamp + seed + optimizer 포함
    opt_tag  = C.PAULI_OPTIMIZER
    run_name = f"{ts}_seed{diag.seed}_{opt_tag}"
    run_dir  = os.path.join("outputs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    prefix   = os.path.join(run_dir, run_name)

    # 이후 모든 print를 log 파일에도 저장
    tee = _Tee(f"{prefix}.log")
    sys.stdout = tee

    print(f"CWD = {os.getcwd()}")
    print(f"Timestamp : {ts}")
    print(f"Seed      : {diag.seed}")
    print(f"Optimizer : {opt_tag}")
    print(f"\nUsing seed={diag.seed} | kind={diag.kind} | gap={diag.gap:.6f}")

    # 2) Adaptive Pauli annealing (determines total step budget)
    # init_params = raw random init (before seed search steps), matching notebook behaviour
    pauli_run = run_pauli_annealing(diag.init_params)
    budget    = pauli_run.total_steps

    # 3) Clean baseline with same step budget
    clean_run = run_clean(diag.init_params, steps=budget)

    # 4) Shot-noise with same step budget
    shot_runs = run_shot_experiments(
        init_params=diag.init_params,
        steps=budget,
        clean_final=clean_run.final_eval,
    )

    # 5) Assemble
    result = ExperimentResult(
        init_seed=diag.seed,
        seed_search=seed_search_out,
        seed_diagnosis=diag,
        clean_run=clean_run,
        pauli_run=pauli_run,
        shot_runs=shot_runs,
    )

    # 6) Save & plot
    save_results(result, prefix)
    plot_results(result, prefix)

    # 7) Console summary
    print("\n" + "=" * 100)
    print(f"{'mode':>20} | {'opt':>6} | {'steps':>7} | {'final energy':>14} | {'escape':>10}")
    print("-" * 100)
    print(f"{'clean':>20} | {clean_run.optimizer_type:>6} | {budget:>7} | {clean_run.final_eval:>14.8f} | {'—':>10}")
    print(f"{'pauli':>20} | {pauli_run.optimizer_type:>6} | {budget:>7} | {pauli_run.final_clean_eval:>14.8f} | {'—':>10}")
    for sr in shot_runs:
        esc = f"{sr.escape_count}/{sr.num_repeats}"
        print(f"{'shot-'+str(sr.shots):>20} | {sr.optimizer_type:>6} | {budget:>7} | {sr.mean_final_clean_eval:>14.8f} | {esc:>10}")
    print("=" * 100)

    tee.close()
    return result


if __name__ == "__main__":
    results = run_all()
