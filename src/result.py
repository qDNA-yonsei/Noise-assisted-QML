# =============================================================================
# result.py — dataclasses for each training mode's output
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SeedDiagnosis:
    seed: int
    kind: str
    gap: float
    clean_final_energy: float
    grad_norm: float
    lambda_min: float
    lambda_max: float
    paper_trapped: bool
    paper_tail_std: float
    paper_tail_drift: float
    trap_score: float
    eval_hist: np.ndarray
    init_params: np.ndarray
    final_params: np.ndarray


@dataclass
class SeedSearchResult:
    selected: SeedDiagnosis
    all_problematic: list[SeedDiagnosis]


@dataclass
class CleanRunResult:
    optimizer_type: str
    lr: float
    total_steps: int
    final_eval: float
    eval_hist: np.ndarray
    final_params: np.ndarray
    grad_norm: float
    lambda_min: float
    lambda_max: float


@dataclass
class PauliAnnealingResult:
    optimizer_type: str
    noise_mode: str
    schedule_mode: str
    noise_label: str
    noise_values: np.ndarray
    total_steps: int
    final_clean_eval: float
    final_params: np.ndarray
    final_grad_norm: float
    final_lambda_min: float
    history: np.ndarray
    history_steps: np.ndarray


@dataclass
class ShotRepResult:
    rep: int
    device_seed: int
    eval_hist: np.ndarray
    final_clean_eval: float
    min_clean_eval: float
    escaped: bool


@dataclass
class ShotRunResult:
    shots: int
    optimizer_type: str
    escape_threshold: float
    repeats: list[ShotRepResult]
    escape_count: int
    num_repeats: int
    mean_final_clean_eval: float
    std_final_clean_eval: float
    best_rep: ShotRepResult
    stuck_rep: Optional[ShotRepResult]


@dataclass
class ExperimentResult:
    init_seed: int
    seed_search: Optional[SeedSearchResult]
    seed_diagnosis: SeedDiagnosis
    clean_run: CleanRunResult
    pauli_run: PauliAnnealingResult
    shot_runs: list[ShotRunResult]
