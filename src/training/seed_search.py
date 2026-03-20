# =============================================================================
# training/seed_search.py — seed-search training mode
# =============================================================================
"""
Training mode: seed search
--------------------------
Runs `SEARCH_STEPS` of the clean optimizer over `SEARCH_SEEDS` seeds,
classifies each one, and returns the best trapped representative.

Checkpoint:
    Every 50 seeds, progress is saved to SEED_SEARCH_CKPT_DIR.
    If a checkpoint exists, the search resumes from where it left off.

Entry point:
    run_seed_search() -> SeedSearchResult
    diagnose_seed(seed) -> SeedDiagnosis       (single-seed variant)
"""

import json
import os
import time
from datetime import datetime

import numpy as np

from .. import config as C
from ..noise import CLEAN_COST
from ..result import SeedDiagnosis, SeedSearchResult
from ..utils import (
    init_params,
    clean_hessian_info,
    optimize_fixed_steps,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CKPT_FILE = "progress.json"


def _diag_to_dict(d: SeedDiagnosis) -> dict:
    return {
        "seed":               d.seed,
        "kind":               d.kind,
        "gap":                d.gap,
        "clean_final_energy": d.clean_final_energy,
        "grad_norm":          d.grad_norm,
        "lambda_min":         d.lambda_min,
        "lambda_max":         d.lambda_max,
        "paper_trapped":      d.paper_trapped,
        "paper_tail_std":     d.paper_tail_std,
        "paper_tail_drift":   d.paper_tail_drift,
        "trap_score":         d.trap_score,
    }


def _save_checkpoint(ckpt_dir: str, seeds_checked: list, problematic: list[SeedDiagnosis]):
    os.makedirs(ckpt_dir, exist_ok=True)
    for d in problematic:
        np.save(os.path.join(ckpt_dir, f"init_params_seed{d.seed}.npy"), np.array(d.init_params))
        np.save(os.path.join(ckpt_dir, f"final_params_seed{d.seed}.npy"), np.array(d.final_params))
    data = {
        "timestamp":     datetime.now().isoformat(),
        "n_checked":     len(seeds_checked),
        "seeds_checked": seeds_checked,
        "problematic":   [_diag_to_dict(d) for d in problematic],
    }
    with open(os.path.join(ckpt_dir, _CKPT_FILE), "w") as f:
        json.dump(data, f, indent=2)


def _load_checkpoint(ckpt_dir: str) -> tuple[set, list[SeedDiagnosis]]:
    path = os.path.join(ckpt_dir, _CKPT_FILE)
    if not os.path.exists(path):
        return set(), []
    with open(path) as f:
        data = json.load(f)
    seeds_done = set(data["seeds_checked"])
    problematic = []
    for rec in data["problematic"]:
        init_path  = os.path.join(ckpt_dir, f"init_params_seed{rec['seed']}.npy")
        final_path = os.path.join(ckpt_dir, f"final_params_seed{rec['seed']}.npy")
        init_params_loaded  = np.load(init_path)  if os.path.exists(init_path)  else np.array([])
        final_params_loaded = np.load(final_path) if os.path.exists(final_path) else np.array([])
        problematic.append(SeedDiagnosis(
            seed=rec["seed"],
            kind=rec["kind"],
            gap=rec["gap"],
            clean_final_energy=rec["clean_final_energy"],
            grad_norm=rec["grad_norm"],
            lambda_min=rec["lambda_min"],
            lambda_max=rec["lambda_max"],
            paper_trapped=rec["paper_trapped"],
            paper_tail_std=rec["paper_tail_std"],
            paper_tail_drift=rec["paper_tail_drift"],
            trap_score=rec["trap_score"],
            eval_hist=np.array([]),
            init_params=init_params_loaded,
            final_params=final_params_loaded,
        ))
    return seeds_done, problematic


# ---------------------------------------------------------------------------
# Single-seed diagnosis
# ---------------------------------------------------------------------------

def _classify(final_energy: float, hess: dict) -> tuple[str, float]:
    gap = final_energy - C.GLOBAL_MIN_ENERGY
    is_stat        = hess["grad_norm"] < C.GRAD_TOL
    is_saddle_like = is_stat and (hess["lambda_min"] < -C.HESS_NEG_TOL)

    if gap <= C.SUBOPT_GAP:
        kind = "not-suboptimal-or-too-close-to-ground"
    elif is_saddle_like:
        kind = "strict-saddle-like"
    elif is_stat:
        kind = "stationary-suboptimal"
    else:
        kind = "suboptimal-but-nonstationary"

    return kind, float(gap)


def _paper_trap(eval_hist: np.ndarray, gap: float) -> tuple[bool, float, float]:
    w     = min(C.PAPER_TAIL_WINDOW, len(eval_hist))
    tail  = eval_hist[-w:]
    std   = float(np.std(tail))
    drift = float(abs(tail[-1] - tail[0]))
    trapped = (gap > C.SUBOPT_GAP) and (std < C.PAPER_TAIL_STD_TOL) and (drift < C.PAPER_TAIL_DRIFT_TOL)
    return bool(trapped), std, drift


def _trap_score(eval_hist: np.ndarray, hess: dict, gap: float) -> float:
    w     = min(C.PAPER_TAIL_WINDOW, len(eval_hist))
    tail  = eval_hist[-w:]
    return (
        C.TRAP_SCORE_W_STD   * float(np.std(tail))
        + C.TRAP_SCORE_W_DRIFT * float(abs(tail[-1] - tail[0]))
        + C.TRAP_SCORE_W_GRAD  * float(hess["grad_norm"])
        - C.TRAP_SCORE_W_GAP   * float(gap)
    )


def diagnose_seed(seed: int) -> SeedDiagnosis:
    """Run clean optimiser for SEARCH_STEPS and diagnose the result."""
    p0  = init_params(seed)
    res = optimize_fixed_steps(
        cost_fn=CLEAN_COST,
        params0=p0,
        steps=C.SEARCH_STEPS,
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
    kind, gap = _classify(res["final_eval"], hess)
    trapped, tail_std, tail_drift = _paper_trap(res["eval_hist"], gap)
    score = _trap_score(res["eval_hist"], hess, gap)

    return SeedDiagnosis(
        seed=int(seed),
        kind=kind,
        gap=gap,
        clean_final_energy=float(res["final_eval"]),
        grad_norm=float(hess["grad_norm"]),
        lambda_min=float(hess["lambda_min"]),
        lambda_max=float(hess["lambda_max"]),
        paper_trapped=trapped,
        paper_tail_std=tail_std,
        paper_tail_drift=tail_drift,
        trap_score=score,
        eval_hist=res["eval_hist"],
        init_params=p0,
        final_params=res["final_params"],
    )


# ---------------------------------------------------------------------------
# Sorting / selection helpers
# ---------------------------------------------------------------------------

def _is_problematic(d: SeedDiagnosis) -> bool:
    return d.kind in {"strict-saddle-like", "stationary-suboptimal"} or d.paper_trapped


def _paper_only_score(d: SeedDiagnosis) -> float:
    return d.paper_tail_std + d.paper_tail_drift - 0.05 * d.gap


def _selection_key(d: SeedDiagnosis):
    if d.kind == "strict-saddle-like":
        return (0, d.seed)
    if d.kind == "stationary-suboptimal":
        return (1, d.trap_score, d.seed)
    if d.paper_trapped:
        return (2, _paper_only_score(d), d.seed)
    return (3, d.seed)


def _sort_problematic(lst: list[SeedDiagnosis]) -> list[SeedDiagnosis]:
    return sorted(lst, key=_selection_key)


def _select_best(problematic: list[SeedDiagnosis]) -> SeedDiagnosis:
    saddle   = [d for d in problematic if d.kind == "strict-saddle-like"]
    if saddle:
        return saddle[0]

    stationary = [d for d in problematic if d.kind == "stationary-suboptimal"]
    if stationary:
        return min(stationary, key=lambda d: d.trap_score)

    paper = [d for d in problematic if d.paper_trapped]
    if paper:
        return min(paper, key=_paper_only_score)

    raise RuntimeError(
        "No trapped seed found. Increase SEARCH_SEEDS or adjust thresholds."
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _print_table(problematic_sorted: list[SeedDiagnosis]):
    W = 195
    print("\n" + "=" * W)
    print("Problematic seeds sorted by selection rule")
    print("=" * W)
    hdr = (
        f"{'rank':>4} | {'seed':>4} | {'kind':>26} | {'paper':>5} | "
        f"{'trap_score':>11} | {'gap':>10} | {'tail_std':>10} | "
        f"{'tail_drift':>10} | {'grad_norm':>11} | {'lambda_min':>11}"
    )
    print(hdr)
    print("-" * W)
    for rank, d in enumerate(problematic_sorted, start=1):
        print(
            f"{rank:>4} | {d.seed:>4} | {d.kind:>26} | {str(d.paper_trapped):>5} | "
            f"{d.trap_score:>11.3e} | {d.gap:>10.6f} | {d.paper_tail_std:>10.3e} | "
            f"{d.paper_tail_drift:>10.3e} | {d.grad_norm:>11.3e} | {d.lambda_min:>11.3e}"
        )
    print("=" * W)


# ---------------------------------------------------------------------------
# Training mode entry points
# ---------------------------------------------------------------------------

def run_seed_search() -> SeedSearchResult:
    """
    Training mode: seed search.
    Scans SEARCH_SEEDS, identifies trapped seeds, returns SeedSearchResult.
    Saves checkpoint every 50 seeds; resumes automatically if checkpoint exists.
    """
    ckpt_dir = C.SEED_SEARCH_CKPT_DIR
    seeds_done, all_problematic = _load_checkpoint(ckpt_dir)

    print("=" * 80)
    print("Seed search mode")
    print(f"  Seeds  : {C.SEARCH_SEEDS[0]} … {C.SEARCH_SEEDS[-1]}")
    print(f"  Steps  : {C.SEARCH_STEPS}")
    print(f"  Opt    : {C.CLEAN_OPTIMIZER}")
    print(f"  Ckpt   : {ckpt_dir}")
    if seeds_done:
        print(f"  Resume : {len(seeds_done)}/{len(C.SEARCH_SEEDS)} seeds already done")
    print("=" * 80)

    remaining = [s for s in C.SEARCH_SEEDS if s not in seeds_done]
    seeds_checked_list = list(seeds_done)
    t0 = time.time()

    for idx, seed in enumerate(remaining, start=1):
        d = diagnose_seed(seed)
        seeds_checked_list.append(seed)
        if _is_problematic(d):
            all_problematic.append(d)
        if len(seeds_checked_list) % 50 == 0:
            print(f"  {len(seeds_checked_list):4d}/{len(C.SEARCH_SEEDS)} seeds checked …")
            _save_checkpoint(ckpt_dir, seeds_checked_list, all_problematic)

    _save_checkpoint(ckpt_dir, seeds_checked_list, all_problematic)

    n = len(C.SEARCH_SEEDS)
    n_prob  = len(all_problematic)
    n_paper = sum(1 for d in all_problematic if d.paper_trapped)
    n_saddle = sum(1 for d in all_problematic if d.kind == "strict-saddle-like")
    n_stat  = sum(1 for d in all_problematic if d.kind == "stationary-suboptimal")

    print(f"\nAll problematic     : {n_prob}/{n} ({100*n_prob/n:.2f}%)")
    print(f"  strict-saddle-like  : {n_saddle}")
    print(f"  stationary-subopt   : {n_stat}")
    print(f"  paper-trapped       : {n_paper}")
    print(f"Search time: {time.time()-t0:.1f}s")

    sorted_list = _sort_problematic(all_problematic)
    if sorted_list:
        _print_table(sorted_list)

    selected = _select_best(sorted_list)
    print(f"\nSelected seed = {selected.seed}  |  kind = {selected.kind}  |  gap = {selected.gap:.6f}")

    return SeedSearchResult(selected=selected, all_problematic=sorted_list)
