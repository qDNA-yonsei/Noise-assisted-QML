# Targeted Pauli-noise experiment modes

This version adds targeted Pauli-noise modes while keeping the original Pauli-noise schedules unchanged.

## Core behavior

1. Start from the original initial parameters `theta0`.
2. Run a short **clean warm-up copy** only to estimate gate importance.
3. Select a fixed mask using `ceil(fraction * num_params)`.
4. Reset to the original `theta0`.
5. Apply the existing Pauli-noise schedule only to selected matched Pauli-rotation gates.

For a selected mask `M_j in {0,1}`:

```text
p_j(t) = p(t) * M_j
```

Non-selected gates always receive `p_j(t)=0`.

The matched-mode convention is unchanged: user-facing `p` is internally passed to `qml.PauliError` as `p/2`.

## New experiment modes

```bash
--experiment-mode targeted_standard
--experiment-mode targeted_method_suite_sample
--experiment-mode targeted_method_suite_aggregate
```

`targeted_standard` is for a single Hamiltonian/seed trajectory check.  
`targeted_method_suite_sample` and `targeted_method_suite_aggregate` are for multi-seed / multi-Hamiltonian statistics.

## New target options

```bash
--target-score-steps 10
--target-score-type mean_abs_grad
--target-fractions 0.1 0.5 1.0
--target-selection-modes top bottom random
--target-suite-schedules fixed_step manual
--target-random-base-seed 2701
--target-no-save-masks
--target-include-shots
```

Defaults implement:

- top 10%, top 50%, top 100%
- bottom 10%, bottom 50%
- random 10%, random 50%

Bottom/random 100% are skipped because they duplicate all-gate noise.

## Recommended quick standard test

```bash
python main.py \
  --experiment-mode targeted_standard \
  --hamiltonian single_Z \
  --noise-mode matched \
  --no-seed-search \
  --init-seed 477 \
  --n-qubits 4 \
  --n-layers 2 \
  --schedule-mode fixed_step \
  --target-score-steps 10 \
  --target-fractions 0.1 0.5 1.0 \
  --target-selection-modes top bottom random \
  --p-max 0.9 \
  --noise-decay-a 10 \
  --pauli-total-steps 200 \
  --pauli-log-every 20
```

## Manual-schedule standard test

```bash
python main.py \
  --experiment-mode targeted_standard \
  --hamiltonian single_Z \
  --noise-mode matched \
  --no-seed-search \
  --init-seed 477 \
  --n-qubits 4 \
  --n-layers 2 \
  --schedule-mode manual \
  --target-score-steps 10 \
  --pauli-schedule 0.8 0.6 0.4 0.2 0.1 0.05 0.02 0.01 0.0 \
  --pauli-min-steps 30 \
  --pauli-max-steps 200 \
  --pauli-check-every 10
```

## Targeted method-suite sample

For one Hamiltonian sample:

```bash
python main.py \
  --experiment-mode targeted_method_suite_sample \
  --method-suite-tag targeted_tfim_v1 \
  --method-suite-sample-index 0 \
  --method-suite-num-hamiltonians 10 \
  --hamiltonian tfim_longitudinal \
  --n-qubits 4 \
  --n-layers 2 \
  --noise-mode matched \
  --search-n 100 \
  --search-steps 300 \
  --target-score-steps 10 \
  --target-fractions 0.1 0.5 1.0 \
  --target-selection-modes top bottom random \
  --target-suite-schedules fixed_step manual \
  --pauli-total-steps 300 \
  --pauli-max-steps 300
```

Run sample indices in parallel by changing `--method-suite-sample-index`.

## Aggregate targeted method-suite results

```bash
python main.py \
  --experiment-mode targeted_method_suite_aggregate \
  --method-suite-tag targeted_tfim_v1
```

## Notes

- Targeting currently supports only `--noise-mode matched`.
- `target_score_steps` default is 10.
- The clean warm-up is not used as pretraining. All targeted methods reset to the original initial parameters before Pauli-noise optimization.
- Random masks are deterministic per training seed and saved with their random seed.
