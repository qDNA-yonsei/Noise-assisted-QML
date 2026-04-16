# =============================================================================
# training/method_suite_parallel.py — parallel-safe multi-method Hamiltonian suite
# =============================================================================

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import config as C
from ..hamiltonian import build_hamiltonian_by_name
from ..noise import (
    make_clean_cost_for_hamiltonian,
    make_regularized_cost_for_hamiltonian,
    make_shot_cost_for_hamiltonian,
)
from ..utils import init_params, to_trainable, optimize_fixed_steps, make_stepper, converged_ckpt, hessian_info

_SECONDARY_KINDS = [
    "strict-saddle-like",
    "stationary-suboptimal",
    "suboptimal-but-nonstationary",
]


def _sanitize_thresholds(values: list[float]) -> list[float]:
    out = sorted({float(v) for v in values})
    bad = [v for v in out if v < 0.0 or v > 1.0]
    if bad:
        raise ValueError(f"All thresholds must lie in [0, 1]. Bad values: {bad}")
    if not out:
        raise ValueError("At least one threshold is required.")
    return out


def _method_specs() -> list[dict]:
    specs = [
        {"method_order": 1, "method_key": "clean", "method_label": "clean", "method_group": "clean", "schedule_mode": "", "shots": ""},
        {"method_order": 2, "method_key": "pauli_fixed", "method_label": "pauli_fixed", "method_group": "pauli", "schedule_mode": "fixed_step", "shots": ""},
        {"method_order": 3, "method_key": "pauli_manual", "method_label": "pauli_manual", "method_group": "pauli", "schedule_mode": "manual", "shots": ""},
    ]
    next_order = 4
    for shots in C.METHOD_SUITE_SHOT_LIST:
        shots = int(shots)
        specs.append({
            "method_order": next_order,
            "method_key": f"shot_{shots}",
            "method_label": f"shot_{shots}",
            "method_group": "shot",
            "schedule_mode": "",
            "shots": shots,
        })
        next_order += 1
    return specs


def _campaign_dir() -> Path:
    return Path(C.METHOD_SUITE_ROOT_DIR) / C.METHOD_SUITE_TAG


def _sample_id(sample_index: int) -> str:
    return f"H{int(sample_index):05d}"


def _sample_dir(sample_index: int) -> Path:
    return _campaign_dir() / _sample_id(sample_index)


def _completion_path(sample_index: int) -> Path:
    return _sample_dir(sample_index) / "_SUCCESS.json"


def _method_manifest_rows() -> list[dict]:
    return _method_specs()


def _write_csv(path: os.PathLike, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: os.PathLike, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_progress(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    _write_json(tmp, payload)
    tmp.replace(path)


def _coeff_rng_for_sample(sample_index: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(C.SEED_SWEEP_COEFF_SEED), int(sample_index), 917_431])
    return np.random.default_rng(ss)


def _sample_coefficients_for_index(sample_index: int) -> dict:
    rng = _coeff_rng_for_sample(sample_index)
    name = C.HAMILTONIAN_NAME
    if name == "single_Z":
        lo, hi = C.SINGLE_Z_COEFF_RANGE
        return {"single_z_coeff": float(rng.uniform(lo, hi))}
    if name == "tfim_longitudinal":
        return {
            "jzz": float(rng.uniform(*C.TFIM_JZZ_RANGE)),
            "hx": float(rng.uniform(*C.TFIM_HX_RANGE)),
            "hz": float(rng.uniform(*C.TFIM_HZ_RANGE)),
        }
    if name == "j1j2_heisenberg":
        return {
            "j1": float(rng.uniform(*C.J1_RANGE)),
            "j2": float(rng.uniform(*C.J2_RANGE)),
        }
    raise ValueError(f"Unknown HAMILTONIAN_NAME={name!r}")


def _coeff_text(params: dict) -> str:
    return ", ".join(f"{k}={v:.6f}" for k, v in params.items())


def _normalized_gap(final_energy: float, ground_energy: float, top_energy: float) -> tuple[float, float, float]:
    span = max(float(top_energy - ground_energy), float(C.SEED_SWEEP_SPAN_EPS))
    absolute_gap = float(final_energy - ground_energy)
    normalized = float(absolute_gap / span)
    return normalized, absolute_gap, span


def _secondary_kind(normalized_gap: float, hess: dict, threshold: float) -> str:
    if normalized_gap <= threshold:
        return "not-bad-final"
    is_stat = float(hess["grad_norm"]) < float(C.GRAD_TOL)
    is_saddle_like = is_stat and (float(hess["lambda_min"]) < -float(C.HESS_NEG_TOL))
    if is_saddle_like:
        return "strict-saddle-like"
    if is_stat:
        return "stationary-suboptimal"
    return "suboptimal-but-nonstationary"


def _paper_trap(eval_hist: np.ndarray, normalized_gap: float, threshold: float) -> tuple[bool, float, float]:
    w = min(C.PAPER_TAIL_WINDOW, len(eval_hist))
    tail = np.array(eval_hist[-w:], dtype=float)
    std = float(np.std(tail))
    drift = float(abs(tail[-1] - tail[0]))
    trapped = (normalized_gap > threshold) and (std < C.PAPER_TAIL_STD_TOL) and (drift < C.PAPER_TAIL_DRIFT_TOL)
    return bool(trapped), std, drift


def _trap_score(eval_hist: np.ndarray, hess: dict, normalized_gap: float) -> float:
    w = min(C.PAPER_TAIL_WINDOW, len(eval_hist))
    tail = np.array(eval_hist[-w:], dtype=float)
    return (
        C.TRAP_SCORE_W_STD * float(np.std(tail))
        + C.TRAP_SCORE_W_DRIFT * float(abs(tail[-1] - tail[0]))
        + C.TRAP_SCORE_W_GRAD * float(hess["grad_norm"])
        - C.TRAP_SCORE_W_GAP * float(normalized_gap)
    )


def _diagnose_from_result(*, seed: int, eval_hist, final_params, clean_cost, ground_energy: float, top_energy: float, main_threshold: float) -> dict:
    hess = hessian_info(clean_cost, final_params)
    final_energy = float(clean_cost(final_params))
    normalized_gap, absolute_gap, spectrum_span = _normalized_gap(final_energy, ground_energy, top_energy)
    kind = _secondary_kind(normalized_gap, hess, main_threshold)
    paper_trapped, tail_std, tail_drift = _paper_trap(np.array(eval_hist, dtype=float), normalized_gap, main_threshold)
    score = _trap_score(np.array(eval_hist, dtype=float), hess, normalized_gap)
    bad_final = bool(normalized_gap > main_threshold)
    return {
        "seed": int(seed),
        "secondary_kind": kind,
        "absolute_gap": float(absolute_gap),
        "normalized_gap": float(normalized_gap),
        "spectrum_span": float(spectrum_span),
        "bad_final": bool(bad_final),
        "clean_final_energy": final_energy,
        "grad_norm": float(hess["grad_norm"]),
        "lambda_min": float(hess["lambda_min"]),
        "lambda_max": float(hess["lambda_max"]),
        "paper_trapped": bool(paper_trapped),
        "paper_tail_std": float(tail_std),
        "paper_tail_drift": float(tail_drift),
        "trap_score": float(score),
    }


def _run_clean_seed_once(init_p, clean_cost) -> dict:
    res = optimize_fixed_steps(
        cost_fn=clean_cost,
        params0=init_p,
        steps=C.SEARCH_STEPS,
        optimizer_type=C.CLEAN_OPTIMIZER,
        lr_gd=C.LR_CLEAN_GD,
        lr_adam=C.LR_CLEAN_ADAM,
        spsa_alpha=C.CLEAN_SPSA_ALPHA,
        spsa_gamma=C.CLEAN_SPSA_GAMMA,
        spsa_c=C.CLEAN_SPSA_C,
        spsa_A=C.CLEAN_SPSA_A,
        spsa_a=C.CLEAN_SPSA_a,
        eval_fn=clean_cost,
    )
    return {
        "eval_hist": np.array(res["eval_hist"], dtype=float),
        "final_params": to_trainable(res["final_params"]),
        "final_eval": float(res["final_eval"]),
        "total_steps": int(C.SEARCH_STEPS),
    }


def _fixed_step_schedule(total_steps: int) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError(f"PAULI_TOTAL_STEPS must be positive, got {total_steps}.")
    max_value = float(C.P_MAX)
    if not (0.0 <= max_value <= 1.0):
        raise ValueError(f"Maximum p must lie in [0, 1], got {max_value}.")
    denom = max(total_steps - 1, 1)
    idx = np.arange(total_steps, dtype=float)
    values = max_value * np.exp(-C.NOISE_DECAY_A * idx / denom)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"Invalid pauli schedule values: {values.tolist()}")
    return values


def _run_pauli_fixed_seed_once(init_p, clean_cost, regularized_cost) -> dict:
    params = to_trainable(init_p)
    total_steps = int(C.PAULI_TOTAL_STEPS)
    noise_values = _fixed_step_schedule(total_steps)
    current_noise = {"value": float(noise_values[0])}

    def dynamic_cost(weights):
        return regularized_cost(weights, current_noise["value"])

    step_fn, _ = make_stepper(
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

    clean_hist = [float(clean_cost(params))]
    for step_idx in range(total_steps):
        current_noise["value"] = float(noise_values[step_idx])
        params = step_fn(params)
        clean_hist.append(float(clean_cost(params)))

    return {
        "eval_hist": np.array(clean_hist, dtype=float),
        "final_params": to_trainable(params),
        "final_eval": float(clean_hist[-1]),
        "total_steps": total_steps,
    }


def _run_pauli_manual_seed_once(init_p, clean_cost, regularized_cost) -> dict:
    params = to_trainable(init_p)
    schedule = np.array(C.PAULI_SCHEDULE, dtype=float)
    if schedule.size == 0:
        raise ValueError("PAULI_SCHEDULE must contain at least one value.")

    clean_hist = [float(clean_cost(params))]
    total_steps = 0
    win_ckpt = max(1, int(np.ceil(C.PAULI_WINDOW / C.PAULI_CHECK_EVERY)))

    for noise_value in schedule:
        use_clean = float(noise_value) <= float(C.CLEAN_THRESHOLD)
        if use_clean:
            cost_fn = clean_cost
        else:
            current_noise = {"value": float(noise_value)}

            def cost_fn(weights, _current_noise=current_noise):
                return regularized_cost(weights, _current_noise["value"])

        step_fn, _ = make_stepper(
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

        stage_ckpt = []
        for step_i in range(1, C.PAULI_MAX_STEPS + 1):
            params = step_fn(params)
            total_steps += 1
            e_clean = float(clean_cost(params))
            clean_hist.append(e_clean)

            if step_i % C.PAULI_CHECK_EVERY == 0:
                stage_ckpt.append(e_clean)
                conv, _, _ = converged_ckpt(stage_ckpt, win_ckpt, C.PAULI_STD_TOL, C.PAULI_RATE_TOL)
                if step_i >= C.PAULI_MIN_STEPS and conv:
                    break

    return {
        "eval_hist": np.array(clean_hist, dtype=float),
        "final_params": to_trainable(params),
        "final_eval": float(clean_hist[-1]),
        "total_steps": int(total_steps),
    }


def _run_shot_seed_once(init_p, clean_cost, shot_cost) -> dict:
    res = optimize_fixed_steps(
        cost_fn=shot_cost,
        params0=init_p,
        steps=C.SEARCH_STEPS,
        optimizer_type=C.SHOT_OPTIMIZER,
        lr_gd=C.LR_SHOT_GD,
        lr_adam=C.LR_SHOT_ADAM,
        spsa_alpha=C.SHOT_SPSA_ALPHA,
        spsa_gamma=C.SHOT_SPSA_GAMMA,
        spsa_c=C.SHOT_SPSA_C,
        spsa_A=C.SHOT_SPSA_A,
        spsa_a=C.SHOT_SPSA_a,
        eval_fn=clean_cost,
    )
    return {
        "eval_hist": np.array(res["eval_hist"], dtype=float),
        "final_params": to_trainable(res["final_params"]),
        "final_eval": float(res["final_eval"]),
        "total_steps": int(C.SEARCH_STEPS),
    }


def _failure_breakdown(rows: list[dict]) -> dict:
    counts = {key: 0 for key in _SECONDARY_KINDS}
    for row in rows:
        if row["bad_final"] and row["secondary_kind"] in counts:
            counts[row["secondary_kind"]] += 1
    counts["bad_final_total"] = int(sum(counts.values()))
    counts["paper_trapped_total"] = int(sum(1 for row in rows if row["paper_trapped"]))
    counts["mean_normalized_gap"] = float(np.mean([row["normalized_gap"] for row in rows])) if rows else 0.0
    counts["median_normalized_gap"] = float(np.median([row["normalized_gap"] for row in rows])) if rows else 0.0
    counts["mean_total_steps"] = float(np.mean([row["total_steps"] for row in rows])) if rows else 0.0
    return counts


def _rates_for_threshold(rows: list[dict], thresholds: list[float]) -> list[dict]:
    out = []
    n = max(1, len(rows))
    gaps = np.array([row["normalized_gap"] for row in rows], dtype=float)
    for threshold in thresholds:
        bad_count = int(np.sum(gaps > threshold))
        out.append({
            "threshold": float(threshold),
            "bad_count": bad_count,
            "bad_rate": float(bad_count / n),
        })
    return out


def _param_keys() -> list[str]:
    if C.HAMILTONIAN_NAME == "single_Z":
        return ["single_z_coeff"]
    if C.HAMILTONIAN_NAME == "tfim_longitudinal":
        return ["jzz", "hx", "hz"]
    return ["j1", "j2"]


def _build_campaign_manifest() -> dict:
    thresholds = _sanitize_thresholds([*C.METHOD_SUITE_THRESHOLD_SWEEP, C.METHOD_SUITE_BAD_NORM_THRESHOLD])
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "campaign_tag": C.METHOD_SUITE_TAG,
        "experiment_mode": C.EXPERIMENT_MODE,
        "hamiltonian_name": C.HAMILTONIAN_NAME,
        "periodic_boundary": bool(C.PERIODIC_BOUNDARY),
        "noise_mode": C.NOISE_MODE,
        "n_qubits": int(C.N_QUBITS),
        "n_layers": int(C.N_LAYERS),
        "ranges": list(C.RANGES),
        "search_seeds": list(C.SEARCH_SEEDS),
        "search_steps_clean_shot": int(C.SEARCH_STEPS),
        "pauli_total_steps": int(C.PAULI_TOTAL_STEPS),
        "pauli_schedule": list(C.PAULI_SCHEDULE),
        "method_suite_num_hamiltonians": int(C.METHOD_SUITE_NUM_HAMILTONIANS),
        "method_suite_sample_index": int(C.METHOD_SUITE_SAMPLE_INDEX),
        "method_suite_bad_threshold": float(C.METHOD_SUITE_BAD_NORM_THRESHOLD),
        "method_suite_threshold_sweep": thresholds,
        "method_suite_shot_list": [int(x) for x in C.METHOD_SUITE_SHOT_LIST],
        "seed_sweep_coeff_seed": int(C.SEED_SWEEP_COEFF_SEED),
        "single_z_coeff_range": list(C.SINGLE_Z_COEFF_RANGE),
        "tfim_jzz_range": list(C.TFIM_JZZ_RANGE),
        "tfim_hx_range": list(C.TFIM_HX_RANGE),
        "tfim_hz_range": list(C.TFIM_HZ_RANGE),
        "j1_range": list(C.J1_RANGE),
        "j2_range": list(C.J2_RANGE),
        "method_manifest": _method_manifest_rows(),
    }


def _ensure_campaign_scaffold() -> None:
    cdir = _campaign_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    manifest = _build_campaign_manifest()
    cfg_path = cdir / "campaign_config.json"
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        keys_to_check = [
            "campaign_tag",
            "hamiltonian_name",
            "periodic_boundary",
            "noise_mode",
            "n_qubits",
            "n_layers",
            "ranges",
            "search_seeds",
            "search_steps_clean_shot",
            "pauli_total_steps",
            "pauli_schedule",
            "method_suite_num_hamiltonians",
            "method_suite_bad_threshold",
            "method_suite_threshold_sweep",
            "method_suite_shot_list",
            "seed_sweep_coeff_seed",
            "single_z_coeff_range",
            "tfim_jzz_range",
            "tfim_hx_range",
            "tfim_hz_range",
            "j1_range",
            "j2_range",
            "method_manifest",
        ]
        mismatches = [k for k in keys_to_check if existing.get(k) != manifest.get(k)]
        if mismatches:
            raise ValueError(
                f"Existing campaign_config.json in {cdir} does not match the current settings. "
                f"Mismatched keys: {mismatches}"
            )
    else:
        _write_json(cfg_path, manifest)
    mm_path = cdir / "method_manifest.csv"
    if not mm_path.exists():
        _write_csv(mm_path, _method_manifest_rows(), ["method_order", "method_key", "method_label", "method_group", "schedule_mode", "shots"])


def run_method_suite_sample() -> dict:
    _ensure_campaign_scaffold()
    sample_index = int(C.METHOD_SUITE_SAMPLE_INDEX)
    if sample_index < 0 or sample_index >= int(C.METHOD_SUITE_NUM_HAMILTONIANS):
        raise ValueError(
            f"METHOD_SUITE_SAMPLE_INDEX must lie in [0, {int(C.METHOD_SUITE_NUM_HAMILTONIANS)-1}], got {sample_index}."
        )
    sdir = _sample_dir(sample_index)
    success_path = _completion_path(sample_index)
    if success_path.exists() and C.METHOD_SUITE_SKIP_EXISTING:
        print(f"Sample {_sample_id(sample_index)} already completed. Skipping because METHOD_SUITE_SKIP_EXISTING=True.")
        return {"sample_dir": str(sdir), "skipped": True, "success_path": str(success_path)}

    sdir.mkdir(parents=True, exist_ok=True)

    coeffs = _sample_coefficients_for_index(sample_index)
    hamiltonian, h_info = build_hamiltonian_by_name(
        C.HAMILTONIAN_NAME,
        n_qubits=C.N_QUBITS,
        periodic=bool(C.PERIODIC_BOUNDARY),
        params=coeffs,
    )
    clean_cost = make_clean_cost_for_hamiltonian(hamiltonian)
    regularized_cost = make_regularized_cost_for_hamiltonian(hamiltonian, noise_mode=C.NOISE_MODE)
    sample_id = _sample_id(sample_index)
    main_threshold = float(C.METHOD_SUITE_BAD_NORM_THRESHOLD)
    threshold_sweep = _sanitize_thresholds([*C.METHOD_SUITE_THRESHOLD_SWEEP, main_threshold])
    method_specs = _method_specs()
    n_methods = len(method_specs)
    n_seeds = len(C.SEARCH_SEEDS)
    progress_every = max(1, n_seeds // 10)

    sample_manifest = {
        "campaign_tag": C.METHOD_SUITE_TAG,
        "sample_id": sample_id,
        "sample_index": sample_index,
        "display_name": h_info.display_name,
        "hamiltonian_name": C.HAMILTONIAN_NAME,
        "periodic_boundary": bool(C.PERIODIC_BOUNDARY),
        "params": h_info.params,
        "ground_energy": float(h_info.exact_ground_energy),
        "top_energy": float(h_info.exact_top_energy),
        "spectrum_span": float(h_info.spectrum_span),
        "thresholds": threshold_sweep,
        "method_manifest": _method_manifest_rows(),
    }
    _write_json(sdir / "sample_manifest.json", sample_manifest)
    _write_csv(sdir / "method_manifest.csv", _method_manifest_rows(), ["method_order", "method_key", "method_label", "method_group", "schedule_mode", "shots"])

    progress_path = sdir / "sample_progress.json"
    method_progress = [
        {
            "method_order": int(spec["method_order"]),
            "method_key": spec["method_key"],
            "status": "pending",
            "seeds_done": 0,
            "seeds_total": n_seeds,
            "bad_final_count": 0,
        }
        for spec in method_specs
    ]

    def update_progress(**extra):
        payload = {
            "campaign_tag": C.METHOD_SUITE_TAG,
            "sample_id": sample_id,
            "sample_index": sample_index,
            "status": extra.pop("status", "running"),
            "hamiltonian_name": C.HAMILTONIAN_NAME,
            "coefficients": h_info.params,
            "ground_energy": float(h_info.exact_ground_energy),
            "top_energy": float(h_info.exact_top_energy),
            "bad_threshold": main_threshold,
            "n_methods": n_methods,
            "n_seeds": n_seeds,
            "methods": method_progress,
            **extra,
        }
        _write_progress(progress_path, payload)

    print("=" * 100)
    print("Method-suite sample mode")
    print(f"Campaign tag           : {C.METHOD_SUITE_TAG}")
    print(f"Sample                 : {sample_id} (index={sample_index})")
    print(f"Hamiltonian            : {h_info.display_name}")
    print(f"Coefficients           : {_coeff_text(h_info.params)}")
    print(f"E0 / Etop              : {h_info.exact_ground_energy:.8f} / {h_info.exact_top_energy:.8f}")
    print(f"Methods                : {[row['method_key'] for row in method_specs]}")
    print(f"Seeds per method       : {n_seeds}")
    print(f"Output dir             : {sdir}")
    print("=" * 100)

    overall_start = time.perf_counter()
    update_progress(status="running", started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    detailed_rows: list[dict] = []
    method_summary_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for method_idx, spec in enumerate(method_specs, start=1):
        method_rows = []
        method_start = time.perf_counter()
        method_progress[method_idx - 1]["status"] = "running"
        update_progress(
            current_method_key=spec["method_key"],
            current_method_order=int(spec["method_order"]),
            current_method_index=method_idx,
            current_seed_index=0,
            current_seed=None,
            elapsed_seconds=float(time.perf_counter() - overall_start),
        )
        print(f"\n[{sample_id}] method {method_idx}/{n_methods}: {spec['method_key']}")

        running_bad = 0
        for seed_idx, seed in enumerate(C.SEARCH_SEEDS, start=1):
            p0 = init_params(seed)
            if spec["method_key"] == "clean":
                res = _run_clean_seed_once(p0, clean_cost)
            elif spec["method_key"] == "pauli_fixed":
                res = _run_pauli_fixed_seed_once(p0, clean_cost, regularized_cost)
            elif spec["method_key"] == "pauli_manual":
                res = _run_pauli_manual_seed_once(p0, clean_cost, regularized_cost)
            elif spec["method_group"] == "shot":
                shots = int(spec["shots"])
                device_seed = int(C.SHOT_DEVICE_BASE_SEED + 100_000 * shots + 10_000 * sample_index + seed)
                shot_cost = make_shot_cost_for_hamiltonian(hamiltonian, shots=shots, device_seed=device_seed)
                res = _run_shot_seed_once(p0, clean_cost, shot_cost)
            else:
                raise ValueError(f"Unknown method spec: {spec}")

            diag = _diagnose_from_result(
                seed=seed,
                eval_hist=res["eval_hist"],
                final_params=res["final_params"],
                clean_cost=clean_cost,
                ground_energy=h_info.exact_ground_energy,
                top_energy=h_info.exact_top_energy,
                main_threshold=main_threshold,
            )
            if diag["bad_final"]:
                running_bad += 1
            row = {
                "campaign_tag": C.METHOD_SUITE_TAG,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "method_order": int(spec["method_order"]),
                "method_key": spec["method_key"],
                "method_label": spec["method_label"],
                "method_group": spec["method_group"],
                "schedule_mode": spec["schedule_mode"],
                "shots": spec["shots"],
                "seed": int(seed),
                "ground_energy": float(h_info.exact_ground_energy),
                "top_energy": float(h_info.exact_top_energy),
                **h_info.params,
                **diag,
                "total_steps": int(res["total_steps"]),
            }
            detailed_rows.append(row)
            method_rows.append(row)
            method_progress[method_idx - 1]["seeds_done"] = seed_idx
            method_progress[method_idx - 1]["bad_final_count"] = running_bad

            if seed_idx == 1 or seed_idx == n_seeds or (seed_idx % progress_every == 0):
                elapsed_method = time.perf_counter() - method_start
                mean_steps = float(np.mean([float(r["total_steps"]) for r in method_rows])) if method_rows else 0.0
                mean_gap = float(np.mean([float(r["normalized_gap"]) for r in method_rows])) if method_rows else 0.0
                print(
                    f"  progress {seed_idx:>4}/{n_seeds} | seed={seed:<5d} | "
                    f"bad={running_bad:>4} | mean_norm_gap={mean_gap:.4f} | mean_steps={mean_steps:.1f} | "
                    f"elapsed={elapsed_method/60.0:.2f} min"
                )
                update_progress(
                    current_method_key=spec["method_key"],
                    current_method_order=int(spec["method_order"]),
                    current_method_index=method_idx,
                    current_seed_index=seed_idx,
                    current_seed=int(seed),
                    elapsed_seconds=float(time.perf_counter() - overall_start),
                )

        breakdown = _failure_breakdown(method_rows)
        method_progress[method_idx - 1].update({
            "status": "done",
            "bad_final_count": int(breakdown["bad_final_total"]),
            "paper_trapped_total": int(breakdown["paper_trapped_total"]),
            "mean_normalized_gap": float(breakdown["mean_normalized_gap"]),
            "mean_total_steps": float(breakdown["mean_total_steps"]),
        })
        method_summary_rows.append({
            "campaign_tag": C.METHOD_SUITE_TAG,
            "sample_id": sample_id,
            "sample_index": sample_index,
            "method_order": int(spec["method_order"]),
            "method_key": spec["method_key"],
            "method_label": spec["method_label"],
            "method_group": spec["method_group"],
            "schedule_mode": spec["schedule_mode"],
            "shots": spec["shots"],
            "ground_energy": float(h_info.exact_ground_energy),
            "top_energy": float(h_info.exact_top_energy),
            **h_info.params,
            "bad_final_count": int(breakdown["bad_final_total"]),
            "bad_final_rate": float(breakdown["bad_final_total"] / max(1, n_seeds)),
            "mean_normalized_gap": float(breakdown["mean_normalized_gap"]),
            "median_normalized_gap": float(breakdown["median_normalized_gap"]),
            "mean_total_steps": float(breakdown["mean_total_steps"]),
            "strict_saddle_like": int(breakdown["strict-saddle-like"]),
            "stationary_suboptimal": int(breakdown["stationary-suboptimal"]),
            "suboptimal_but_nonstationary": int(breakdown["suboptimal-but-nonstationary"]),
            "paper_trapped_total": int(breakdown["paper_trapped_total"]),
        })
        for rec in _rates_for_threshold(method_rows, threshold_sweep):
            threshold_rows.append({
                "campaign_tag": C.METHOD_SUITE_TAG,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "method_order": int(spec["method_order"]),
                "method_key": spec["method_key"],
                "method_label": spec["method_label"],
                "method_group": spec["method_group"],
                "schedule_mode": spec["schedule_mode"],
                "shots": spec["shots"],
                **h_info.params,
                **rec,
            })
        elapsed_method = time.perf_counter() - method_start
        print(
            f"  finished {spec['method_key']:<13} -> bad={breakdown['bad_final_total']}/{n_seeds} "
            f"({breakdown['bad_final_total']/max(1, n_seeds):.3f}), "
            f"mean_norm_gap={breakdown['mean_normalized_gap']:.4f}, "
            f"mean_steps={breakdown['mean_total_steps']:.1f}, elapsed={elapsed_method/60.0:.2f} min"
        )
        update_progress(
            current_method_key=spec["method_key"],
            current_method_order=int(spec["method_order"]),
            current_method_index=method_idx,
            current_seed_index=n_seeds,
            current_seed=int(C.SEARCH_SEEDS[-1]) if C.SEARCH_SEEDS else None,
            elapsed_seconds=float(time.perf_counter() - overall_start),
        )

    param_keys = _param_keys()
    detailed_fieldnames = [
        "campaign_tag", "sample_id", "sample_index", "method_order", "method_key", "method_label", "method_group", "schedule_mode", "shots",
        "seed", "ground_energy", "top_energy", *param_keys,
        "secondary_kind", "absolute_gap", "normalized_gap", "spectrum_span", "bad_final", "clean_final_energy",
        "grad_norm", "lambda_min", "lambda_max", "paper_trapped", "paper_tail_std", "paper_tail_drift", "trap_score", "total_steps",
    ]
    _write_csv(sdir / "detailed_results.csv", detailed_rows, detailed_fieldnames)

    summary_fieldnames = [
        "campaign_tag", "sample_id", "sample_index", "method_order", "method_key", "method_label", "method_group", "schedule_mode", "shots",
        "ground_energy", "top_energy", *param_keys,
        "bad_final_count", "bad_final_rate", "mean_normalized_gap", "median_normalized_gap", "mean_total_steps",
        "strict_saddle_like", "stationary_suboptimal", "suboptimal_but_nonstationary", "paper_trapped_total",
    ]
    _write_csv(sdir / "method_summary.csv", method_summary_rows, summary_fieldnames)

    threshold_fieldnames = [
        "campaign_tag", "sample_id", "sample_index", "method_order", "method_key", "method_label", "method_group", "schedule_mode", "shots",
        *param_keys, "threshold", "bad_count", "bad_rate"
    ]
    _write_csv(sdir / "threshold_sweep_by_method.csv", threshold_rows, threshold_fieldnames)

    _write_json(sdir / "sample_summary.json", {
        "sample_manifest": sample_manifest,
        "method_summary": method_summary_rows,
    })
    total_elapsed = time.perf_counter() - overall_start
    _write_json(success_path, {
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "elapsed_seconds": float(total_elapsed),
        "method_bad_final_counts": [
            {
                "method_order": int(row["method_order"]),
                "method_key": row["method_key"],
                "bad_final_count": int(row["bad_final_count"]),
                "bad_final_rate": float(row["bad_final_rate"]),
            }
            for row in method_summary_rows
        ],
    })
    update_progress(
        status="completed",
        completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_seconds=float(total_elapsed),
        current_method_key=None,
        current_method_order=None,
        current_method_index=n_methods,
        current_seed_index=n_seeds,
        current_seed=int(C.SEARCH_SEEDS[-1]) if C.SEARCH_SEEDS else None,
    )

    print("\n" + "-" * 100)
    print(f"[{sample_id}] final method summary")
    for row in sorted(method_summary_rows, key=lambda x: int(x["method_order"])):
        print(
            f"  {row['method_order']}. {row['method_key']:<13} -> "
            f"bad_final={row['bad_final_count']}/{n_seeds} ({row['bad_final_rate']:.3f}), "
            f"mean_norm_gap={row['mean_normalized_gap']:.4f}, mean_steps={row['mean_total_steps']:.1f}"
        )
    print(f"Saved sample outputs -> {sdir}")
    print(f"Total elapsed          -> {total_elapsed/60.0:.2f} min")
    print("-" * 100)
    return {"sample_dir": str(sdir), "sample_id": sample_id, "success_path": str(success_path)}


def _load_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _method_order_mapping(rows: list[dict]) -> dict:
    order = {}
    for row in rows:
        order[row["method_key"]] = int(row["method_order"])
    return order


def _plot_rate_heatmap(method_summary_rows: list[dict], method_manifest: list[dict], out_path: Path) -> None:
    if not method_summary_rows:
        return
    method_keys = [row["method_key"] for row in sorted(method_manifest, key=lambda x: int(x["method_order"]))]
    sample_ids = sorted({row["sample_id"] for row in method_summary_rows})
    mat = np.full((len(sample_ids), len(method_keys)), np.nan, dtype=float)
    idx_sample = {sid: i for i, sid in enumerate(sample_ids)}
    idx_method = {mk: j for j, mk in enumerate(method_keys)}
    for row in method_summary_rows:
        mat[idx_sample[row["sample_id"]], idx_method[row["method_key"]]] = float(row["bad_final_rate"])
    fig_h = max(4.0, 0.45 * len(sample_ids) + 1.8)
    fig, ax = plt.subplots(figsize=(1.4 * len(method_keys) + 3.2, fig_h))
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(method_keys)))
    ax.set_xticklabels(method_keys, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(sample_ids)))
    ax.set_yticklabels(sample_ids)
    ax.set_xlabel("method")
    ax.set_ylabel("random Hamiltonian sample")
    ax.set_title("Bad-final rate by sample and method")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_threshold_sweep(overall_rows: list[dict], method_manifest: list[dict], out_path: Path) -> None:
    if not overall_rows:
        return
    thresholds = sorted({float(row["threshold"]) for row in overall_rows})
    method_keys = [row["method_key"] for row in sorted(method_manifest, key=lambda x: int(x["method_order"]))]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for mk in method_keys:
        vals = []
        for t in thresholds:
            match = next((row for row in overall_rows if row["method_key"] == mk and float(row["threshold"]) == t), None)
            vals.append(float(match["overall_bad_rate"]) if match else np.nan)
        ax.plot(thresholds, vals, marker="o", label=mk)
    ax.set_xlabel("normalized-gap threshold")
    ax.set_ylabel("overall bad-final rate")
    ax.set_title("Threshold sweep across all methods")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_ecdf(detailed_rows: list[dict], method_manifest: list[dict], out_path: Path) -> None:
    if not detailed_rows:
        return
    method_keys = [row["method_key"] for row in sorted(method_manifest, key=lambda x: int(x["method_order"]))]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for mk in method_keys:
        vals = np.sort(np.array([float(row["normalized_gap"]) for row in detailed_rows if row["method_key"] == mk], dtype=float))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1, dtype=float) / vals.size
        ax.step(vals, y, where="post", label=mk)
    ax.set_xlabel("normalized gap")
    ax.set_ylabel("ECDF")
    ax.set_title("Distribution of normalized gaps across methods")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_failure_breakdown(detailed_rows: list[dict], method_manifest: list[dict], main_threshold: float, out_path: Path) -> None:
    method_keys = [row["method_key"] for row in sorted(method_manifest, key=lambda x: int(x["method_order"]))]
    failed_by_method = {mk: [row for row in detailed_rows if row["method_key"] == mk and str(row["bad_final"]).lower() == "true"] for mk in method_keys}
    totals = [len(failed_by_method[mk]) for mk in method_keys]
    fig, ax = plt.subplots(figsize=(1.35 * len(method_keys) + 3.2, 4.8))
    bottoms = np.zeros(len(method_keys), dtype=float)
    for kind in _SECONDARY_KINDS:
        vals = []
        for mk in method_keys:
            rows = failed_by_method[mk]
            if not rows:
                vals.append(0.0)
            else:
                vals.append(sum(1 for row in rows if row["secondary_kind"] == kind) / len(rows))
        vals = np.array(vals, dtype=float)
        ax.bar(method_keys, vals, bottom=bottoms, label=kind)
        bottoms += vals
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("fraction within failed runs")
    ax.set_title(f"Failure-type breakdown among bad-final runs (threshold={main_threshold:.3f})")
    xticklabels = [f"{mk}\n(n_fail={totals[i]})" for i, mk in enumerate(method_keys)]
    ax.set_xticks(np.arange(len(method_keys)))
    ax.set_xticklabels(xticklabels)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def run_method_suite_aggregate() -> dict:
    cdir = _campaign_dir()
    if not cdir.exists():
        raise FileNotFoundError(f"Campaign directory does not exist: {cdir}")
    cfg_path = cdir / "campaign_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing campaign_config.json in {cdir}")
    campaign_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    method_manifest = campaign_config["method_manifest"]

    sample_dirs = sorted([p for p in cdir.iterdir() if p.is_dir() and (p / "_SUCCESS.json").exists()])
    if not sample_dirs:
        raise RuntimeError(f"No completed sample directories found in {cdir}")

    detailed_rows: list[dict] = []
    method_summary_rows: list[dict] = []
    threshold_rows: list[dict] = []
    sample_manifests: list[dict] = []
    for sdir in sample_dirs:
        sample_manifests.append(json.loads((sdir / "sample_manifest.json").read_text(encoding="utf-8")))
        detailed_rows.extend(_load_csv_rows(sdir / "detailed_results.csv"))
        method_summary_rows.extend(_load_csv_rows(sdir / "method_summary.csv"))
        threshold_rows.extend(_load_csv_rows(sdir / "threshold_sweep_by_method.csv"))

    method_keys = [row["method_key"] for row in sorted(method_manifest, key=lambda x: int(x["method_order"]))]
    overall_rows = []
    thresholds = sorted({float(x) for x in campaign_config["method_suite_threshold_sweep"]})
    for mk in method_keys:
        rows_m = [row for row in detailed_rows if row["method_key"] == mk]
        n = max(1, len(rows_m))
        gaps = np.array([float(row["normalized_gap"]) for row in rows_m], dtype=float)
        for t in thresholds:
            bad_count = int(np.sum(gaps > t))
            overall_rows.append({
                "method_key": mk,
                "threshold": float(t),
                "overall_bad_count": bad_count,
                "overall_bad_rate": float(bad_count / n),
            })

    overall_summary_rows = []
    for spec in sorted(method_manifest, key=lambda x: int(x["method_order"])):
        mk = spec["method_key"]
        rows_m = [row for row in detailed_rows if row["method_key"] == mk]
        n = max(1, len(rows_m))
        bad_total = int(sum(1 for row in rows_m if str(row["bad_final"]).lower() == "true"))
        overall_summary_rows.append({
            "method_order": int(spec["method_order"]),
            "method_key": mk,
            "method_label": spec["method_label"],
            "method_group": spec["method_group"],
            "schedule_mode": spec["schedule_mode"],
            "shots": spec["shots"],
            "bad_final_total": bad_total,
            "bad_final_rate": float(bad_total / n),
            "mean_normalized_gap": float(np.mean([float(row["normalized_gap"]) for row in rows_m])) if rows_m else 0.0,
            "median_normalized_gap": float(np.median([float(row["normalized_gap"]) for row in rows_m])) if rows_m else 0.0,
            "mean_total_steps": float(np.mean([float(row["total_steps"]) for row in rows_m])) if rows_m else 0.0,
            "strict_saddle_like": int(sum(1 for row in rows_m if row["secondary_kind"] == "strict-saddle-like" and str(row["bad_final"]).lower() == "true")),
            "stationary_suboptimal": int(sum(1 for row in rows_m if row["secondary_kind"] == "stationary-suboptimal" and str(row["bad_final"]).lower() == "true")),
            "suboptimal_but_nonstationary": int(sum(1 for row in rows_m if row["secondary_kind"] == "suboptimal-but-nonstationary" and str(row["bad_final"]).lower() == "true")),
            "paper_trapped_total": int(sum(1 for row in rows_m if str(row["paper_trapped"]).lower() == "true")),
        })

    out_dir = cdir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "method_summary_by_sample.csv", method_summary_rows, list(method_summary_rows[0].keys()))
    _write_csv(out_dir / "detailed_results.csv", detailed_rows, list(detailed_rows[0].keys()))
    _write_csv(out_dir / "threshold_sweep_by_sample.csv", threshold_rows, list(threshold_rows[0].keys()))
    _write_csv(out_dir / "threshold_sweep_overall.csv", overall_rows, ["method_key", "threshold", "overall_bad_count", "overall_bad_rate"])
    _write_csv(out_dir / "overall_summary.csv", overall_summary_rows, list(overall_summary_rows[0].keys()))
    _write_json(out_dir / "aggregate_summary.json", {
        "campaign_config": campaign_config,
        "num_completed_samples": len(sample_dirs),
        "completed_sample_ids": [m["sample_id"] for m in sample_manifests],
        "overall_summary": overall_summary_rows,
        "overall_threshold_sweep": overall_rows,
    })

    _plot_rate_heatmap(method_summary_rows, method_manifest, out_dir / "bad_final_rate_heatmap.png")
    _plot_threshold_sweep(overall_rows, method_manifest, out_dir / "threshold_sweep.png")
    _plot_ecdf(detailed_rows, method_manifest, out_dir / "normalized_gap_ecdf.png")
    _plot_failure_breakdown(detailed_rows, method_manifest, float(campaign_config["method_suite_bad_threshold"]), out_dir / "failure_type_breakdown.png")

    print("=" * 100)
    print("Method-suite aggregate mode")
    print(f"Campaign tag           : {C.METHOD_SUITE_TAG}")
    print(f"Completed samples      : {len(sample_dirs)}")
    print(f"Aggregate dir          : {out_dir}")
    for row in overall_summary_rows:
        print(
            f"  {row['method_key']:<14} -> bad={row['bad_final_total']} rate={row['bad_final_rate']:.3f} "
            f"mean_norm_gap={row['mean_normalized_gap']:.4f}"
        )
    print("=" * 100)

    return {
        "campaign_dir": str(cdir),
        "aggregate_dir": str(out_dir),
        "completed_samples": len(sample_dirs),
        "overall_summary_path": str(out_dir / "overall_summary.csv"),
        "threshold_sweep_path": str(out_dir / "threshold_sweep_overall.csv"),
        "heatmap_path": str(out_dir / "bad_final_rate_heatmap.png"),
        "ecdf_path": str(out_dir / "normalized_gap_ecdf.png"),
        "breakdown_path": str(out_dir / "failure_type_breakdown.png"),
    }
