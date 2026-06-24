# commeff_tfim_8qubit — Experiment Plan

## Overview

This experiment replicates the statistical structure of `targeted_tfim_8qubit` (jungyun)
on the same 5 Hamiltonians and 100 seeds, but replaces the importance scoring method:

- **jungyun**: `mean_abs_grad` (average gradient magnitude over 100 clean Adam steps)
- **YJ (this work)**: `commutator_eff` (effective Hamiltonian commutator, averaged over `n_avg=50` random parameter sets)

All other conditions (circuit, Hamiltonians, seeds, schedule structure, output format) are matched to enable direct comparison.

---

## Circuit

| Parameter | Value |
|-----------|-------|
| n_qubits  | 8 |
| n_layers  | 4 |
| ranges    | [1, 2, 3, 4] |
| total rotation gates | 96 |
| entangling | CNOT (StronglyEntanglingLayers) |

---

## Hamiltonians

5 fixed TFIM Hamiltonians from `targeted_tfim_8qubit` (open boundary, hz=0):

`H = -jzz * Σ Z_i Z_{i+1} - hx * Σ X_i`

| ID     | jzz      | hx       | ground energy |
|--------|----------|----------|---------------|
| H00000 | -1.02277 |  1.04293 | -10.18344 |
| H00001 |  0.91856 | -1.91341 | -16.08721 |
| H00002 | -1.92244 |  0.84161 | -14.39433 |
| H00003 |  1.32125 |  0.80994 | -10.54559 |
| H00004 | -1.71190 |  0.17443 | -12.02779 |

---

## Seeds

- 0 to 99 (100 seeds total)
- Initialization: `np.random.default_rng(seed).uniform(0.0, 2π, (n_layers, n_qubits, 3))`
  (identical to jungyun)

---

## Importance Scoring: commutator_eff

For each gate k in the circuit (traversed in reverse):

```
O ← H_mat
for gate k (reversed):
    score[k] = ||[O, G_k]||_F
    O ← U_k† O U_k
```

Averaged over `n_avg=50` random parameter sets to reduce sensitivity to initialization.

---

## Noise Model

- **Mode**: matched Pauli noise
- **Application**: `PauliError("Z", p)` before RZ gates, `PauliError("Y", p)` before RY gates
- **Convention**: YJ passes `p` directly to `qml.PauliError` (no p/2 conversion)
- **Equivalence**: YJ schedule × 2 = jungyun schedule (same physical noise strength)

---

## Annealing Schedule

| jungyun (user-facing p) | YJ (direct PauliError p) |
|-------------------------|--------------------------|
| 0.8 | **0.4** |
| 0.6 | 0.3 |
| 0.4 | 0.2 |
| 0.2 | 0.1 |
| 0.1 | 0.05 |
| 0.05 | 0.025 |
| 0.02 | 0.01 |
| 0.01 | 0.005 |
| 0.0 | 0.0 |

Per stage: max 1000 steps, early stop on convergence (std < 0.005 and rate < 0.005 over 20-step window).

---

## Methods (7 total)

| # | method_key | description |
|---|-----------|-------------|
| 1 | `clean` | No noise, plain Adam (baseline) |
| 2 | `top10_fixed` | commutator_eff top 10% (~10 gates) noisy |
| 3 | `top50_fixed` | commutator_eff top 50% (~48 gates) noisy |
| 4 | `top90_fixed` | commutator_eff top 90% (~86 gates) noisy |
| 5 | `bottom10_fixed` | commutator_eff bottom 10% noisy |
| 6 | `bottom50_fixed` | commutator_eff bottom 50% noisy |
| 7 | `bottom90_fixed` | commutator_eff bottom 90% noisy |

**Excluded vs jungyun**: random selection, shot_100, full/top100
- `full` == `top100` (identical mask when fraction=1.0) → redundant
- If `clean` ≈ `full` (verified on H00001 seed=54), both can serve as reference from jungyun data

---

## Comparison with targeted_tfim_8qubit

| Item | targeted_tfim_8qubit (jungyun) | commeff_tfim_8qubit (YJ) |
|------|-------------------------------|--------------------------|
| Hamiltonians | H00000–H00004 | identical |
| Seeds | 0–99 | identical |
| Circuit | 8-qubit, 4-layer, 96 gates | identical |
| Noise mode | matched (p/2 internally) | matched (p directly) |
| Schedule | [0.8 → 0.0] | [0.4 → 0.0] |
| Importance scoring | mean_abs_grad (100 steps) | commutator_eff (n_avg=50) |
| Methods | 9 (incl. random, shot) | 7 (top/bottom 10/50/90 + clean) |
| fractions | 10%, 50%, 100% | 10%, 50%, 90% |
| Output format | detailed_results.csv etc. | identical |

---

## Output Format

Matches `targeted_tfim_8qubit` structure:

```
results/commeff_tfim_8qubit/
├── campaign_config.json
├── H00000/
│   ├── sample_manifest.json
│   ├── detailed_results.csv
│   ├── method_summary.csv
│   ├── threshold_sweep_by_method.csv
│   └── target_masks/
│       ├── seed00000_target_score_hist.npy
│       ├── seed00000_commeff_top10_fixed_mask.npy
│       └── ...
├── H00001/ ...
└── aggregate/
```

### detailed_results.csv columns

Same as jungyun:
`campaign_tag, sample_id, method_key, seed, ground_energy, normalized_gap, spectrum_span, bad_final, clean_final_energy, total_steps, ...`

### normalized_gap formula

```
normalized_gap = (clean_final_energy - ground_energy) / spectrum_span
```

where `spectrum_span = top_energy - ground_energy ≈ 2 * abs(ground_energy)` for symmetric TFIM.

---

## Verification Status

| Check | Status | Notes |
|-------|--------|-------|
| Noise convention (YJ p = jungyun p/2) | ✅ confirmed | sweep_8qubit.py line 48 comment |
| Schedule equivalence (YJ 0.4 = jungyun 0.8) | ✅ confirmed | same physical PauliError prob |
| Hamiltonian formula | ✅ identical | open boundary, `-jzz*ZZ - hx*X` |
| Initialization | ✅ identical | `default_rng(seed).uniform(0, 2π)` |
| Optimizer | ✅ fixed | changed manual adam_step → qml.AdamOptimizer |
| normalized_gap formula | ✅ fixed | was `/abs(GROUND)`, now `/spectrum_span` |
| full noise (B1) == jungyun top100 | ✅ verified | H00001 seed=54: -15.41912 vs -15.41914 |
| clean == jungyun clean | ⏳ pending | need to run test_clean_vs_full.py on server |
