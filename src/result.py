# =============================================================================
# result.py — dataclasses for each training mode's output
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Seed search mode
# ---------------------------------------------------------------------------

@dataclass
class SeedDiagnosis:
    """Diagnosis of a single initialisation seed on the clean landscape."""
    seed: int
    kind: str           # "strict-saddle-like" | "stationary-suboptimal" |
                        # "not-suboptimal-or-too-close-to-ground" | "suboptimal-but-nonstationary"
    gap: float          # final_energy - global_min
    clean_final_energy: float
    grad_norm: float
    lambda_min: float
    lambda_max: float
    paper_trapped: bool
    paper_tail_std: float
    paper_tail_drift: float
    trap_score: float
    eval_hist: np.ndarray   # clean energy at each step during diagnosis
    init_params: np.ndarray  # raw random init (before any optimization)
    final_params: np.ndarray # params after SEARCH_STEPS of clean optimization


@dataclass
class SeedSearchResult:
    """Output of the seed-search training mode."""
    selected: SeedDiagnosis                 # representative trapped seed
    all_problematic: list[SeedDiagnosis]    # all candidates, sorted by selection rule


# ---------------------------------------------------------------------------
# Clean-baseline mode
# ---------------------------------------------------------------------------

@dataclass
class CleanRunResult:
    """Output of noiseless fixed-step optimisation."""
    optimizer_type: str
    lr: float
    total_steps: int
    final_eval: float
    eval_hist: np.ndarray
    final_params: np.ndarray
    grad_norm: float
    lambda_min: float
    lambda_max: float


# ---------------------------------------------------------------------------
# Adaptive Pauli annealing mode
# ---------------------------------------------------------------------------

@dataclass
class PauliStageResult:
    """Per-stage result inside adaptive Pauli annealing."""
    stage_idx: int
    noise_p: float
    steps_used: int
    converged: bool
    final_std: float
    final_delta: float
    eval_hist: np.ndarray   # clean-eval checkpoints (every check_every steps)
    hist_steps: np.ndarray  # global step indices of those checkpoints


@dataclass
class PauliAnnealingResult:
    """Output of adaptive Pauli annealing mode."""
    optimizer_type: str
    noise_schedule: list[float]
    stages: list[PauliStageResult]
    total_steps: int
    final_clean_eval: float
    final_params: np.ndarray
    final_grad_norm: float
    final_lambda_min: float
    history: np.ndarray        # all checkpoint clean-evals concatenated
    history_steps: np.ndarray  # corresponding global step indices


# ---------------------------------------------------------------------------
# Shot-noise mode
# ---------------------------------------------------------------------------

@dataclass
class ShotRepResult:
    """Result of a single shot-noise repetition."""
    rep: int
    device_seed: int
    eval_hist: np.ndarray   # clean-cost re-evaluation at each step
    final_clean_eval: float
    min_clean_eval: float
    escaped: bool


@dataclass
class ShotRunResult:
    """Output for one shots value (aggregated over repeats)."""
    shots: int
    optimizer_type: str
    escape_threshold: float
    repeats: list[ShotRepResult]
    # aggregates
    escape_count: int
    num_repeats: int
    mean_final_clean_eval: float
    std_final_clean_eval: float
    best_rep: ShotRepResult             # lowest min_clean_eval across repeats
    stuck_rep: Optional[ShotRepResult]  # closest-to-clean-final non-escaped rep
                                        # (None if all escaped or none escaped)


# ---------------------------------------------------------------------------
# Top-level experiment result
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """
    Full result for one experiment run.

    Fields map 1-to-1 to training modes:
      seed_search  → SeedSearchResult   (RUN_SEED_SEARCH=True) or None
      clean_run    → CleanRunResult
      pauli_run    → PauliAnnealingResult
      shot_runs    → list[ShotRunResult]
    """
    init_seed: int
    seed_search: Optional[SeedSearchResult]     # None when INIT_SEED is fixed
    seed_diagnosis: SeedDiagnosis               # diagnosis of the chosen seed
    clean_run: CleanRunResult
    pauli_run: PauliAnnealingResult
    shot_runs: list[ShotRunResult]
