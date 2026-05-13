# Hamiltonian sweep notes

This patch adds three Hamiltonian families selectable from the terminal:

- `single_Z` (the original project Hamiltonian)
- `tfim_longitudinal`
- `j1j2_heisenberg`

It also keeps `--n-layers` as a first-class CLI option so deeper ansätze can be tested directly.

## Recommended sweep 1: 1D TFIM + longitudinal field

Suggested difficulty sweep:

- `n = 6, 8, 10`
- `Jzz = 1.0`
- `hx in {0.5, 1.0, 1.5}`
- `hz in {0.0, 0.1, 0.2, 0.5}`
- Start with `--n-layers 4` or `--n-layers 6`

Example:

```bash
python main.py \
  --hamiltonian tfim_longitudinal \
  --n-qubits 8 \
  --n-layers 4 \
  --tfim-jzz 1.0 \
  --tfim-hx 1.0 \
  --tfim-hz 0.2 \
  --no-seed-search \
  --init-seed 51 \
  --noise-mode matched
```

## Recommended sweep 2: 1D J1-J2 Heisenberg

Suggested difficulty sweep:

- `n = 6, 8, 10`
- `J1 = 1.0`
- `J2/J1 in {0.0, 0.2, 0.4, 0.5, 0.6}`
- Start with `--n-layers 4` or `--n-layers 6`

Example:

```bash
python main.py \
  --hamiltonian j1j2_heisenberg \
  --n-qubits 8 \
  --n-layers 6 \
  --j1 1.0 \
  --j2 0.5 \
  --no-seed-search \
  --init-seed 51 \
  --noise-mode matched
```

## Boundary conditions

Both `tfim_longitudinal` and `j1j2_heisenberg` accept:

- `--periodic`
- `--open-boundary`

The default is open boundary conditions.

## Notes

- The code now computes the exact ground-state energy of the selected Hamiltonian by dense diagonalization.
- This exact energy is saved into the config and summary files and is used consistently in the plots and diagnostics.
- For `matched` mode the user-facing `p` still matches the paper convention, while the internal PennyLane call uses `p/2`.
