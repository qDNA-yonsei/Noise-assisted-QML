"""
main.py — Effective-Hamiltonian Commutator Importance Scoring
--------------------------------------------------------------
Gate importance via effective (back-propagated) Hamiltonian:

    O  ← H_mat
    for gate in reversed(circuit):
        if gate is parameterized:
            scores[k] = ‖[O, G_k]‖_F
        O ← U_k† O U_k          # pull H back through this gate

Interpretation:  at position k, O = U_after† H U_after
so we score how much gate k can rotate the state toward lower energy
given everything that comes after it.

Compare with the naive baseline: scores_naive[k] = ‖[H, G_k]‖_F
which ignores the circuit position entirely.

Averaged effective scoring (--n_avg N):
  Instead of scoring at a single parameter point, draw N random parameter
  sets and average the effective scores across all of them.  This reduces
  sensitivity to the specific initial point chosen for scoring.

Usage
-----
  python main.py                          # seed 477, clean params (default)
  python main.py --seed 176               # seed 176
  python main.py --seed 477 --params init # random init params
  python main.py --seed 477 --params pauli
  python main.py --seed 477 --top_k 8 16 24
  python main.py --seed 477 --n_steps 500 --lr 0.01
  python main.py --seed 477 --no_train    # scoring + plot only, skip training
  python main.py --seed 477 --no_noise    # clean Adam only, skip noise annealing
  python main.py --seed 477 --n_avg 10   # average eff score over 10 random param sets
"""

import sys, os, json, argparse, time
from datetime import datetime
from math import cos, sin


# ---------------------------------------------------------------------------
# Tee: 콘솔과 파일에 동시 출력
# ---------------------------------------------------------------------------

class _Tee:
    """sys.stdout을 콘솔과 파일에 동시에 씁니다."""
    def __init__(self, path):
        self._file    = open(path, "w", buffering=1)
        self._stdout  = sys.stdout
        sys.stdout    = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE      = os.path.dirname(os.path.abspath(__file__))
_CODES     = os.path.join(_HERE, "..", "..", "Codes")
_CKPT_BASE = os.path.join(_CODES, "outputs", "seed_training", "adam")


def _find_ckpt(seed: int) -> dict:
    """
    Find the latest checkpoint directory for `seed` and return file paths.
    Raises FileNotFoundError if no matching directory exists.
    """
    import glob
    pattern = os.path.join(_CKPT_BASE, f"*_seed{seed}")
    dirs = sorted(glob.glob(pattern))   # sorted → latest is last
    if not dirs:
        raise FileNotFoundError(
            f"No checkpoint found for seed {seed} under {_CKPT_BASE}\n"
            f"Available: {os.listdir(_CKPT_BASE)}"
        )
    ckpt_dir = dirs[-1]                 # use most recent run for this seed
    ts_tag   = os.path.basename(ckpt_dir)   # e.g. "20260324_170741_seed477"
    prefix   = os.path.join(ckpt_dir, f"{ts_tag}_adam")
    return {
        "ckpt_dir":    ckpt_dir,
        "init_params": f"{prefix}_init_params_seed{seed}.npy",
        "clean_params":f"{prefix}_clean_final_params.npy",
        "pauli_params":f"{prefix}_pauli_final_params.npy",
        "clean_hist":  f"{prefix}_clean_eval_hist.npy",
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",    type=int, default=477,
                   help="Seed to load checkpoint for (default: 477)")
    p.add_argument("--params",  default="clean",
                   choices=["init", "clean", "pauli"],
                   help="Which parameter point to score at (default: clean-trained)")
    p.add_argument("--top_k",  nargs="+", type=int, default=[8, 16],
                   help="Pruning top-k values to run experiments for")
    p.add_argument("--n_steps", type=int, default=500)
    p.add_argument("--lr",      type=float, default=0.01)
    p.add_argument("--no_train", action="store_true",
                   help="Skip training, just compute and plot importance scores")
    p.add_argument("--no_noise", action="store_true",
                   help="Skip noise annealing experiments (run clean Adam only)")
    p.add_argument("--selective_only", action="store_true",
                   help="Skip clean Adam and pruned noise; run selective noise (B-3) only")
    p.add_argument("--inv_sel_only", action="store_true",
                   help="Like --selective_only but noise on NON-important gates (B-4)")
    p.add_argument("--n_avg", type=int, default=1,
                   help="Number of random param sets to average eff scores over (default: 1 = single point)")
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Multiplicative scale applied to PAULI_SCHEDULE (non-zero entries only). "
                        "E.g. 0.5 halves all noise levels. (default: 1.0)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Circuit config (must match checkpoint)
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_LAYERS = 2
RANGES   = [1, 2]
WIRES    = list(range(N_QUBITS))
DIM      = 2 ** N_QUBITS

# ---------------------------------------------------------------------------
# Matrix building blocks
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_PAULI = {
    "X": np.array([[0, 1],  [1, 0]],  dtype=complex),
    "Y": np.array([[0, -1j],[1j, 0]], dtype=complex),
    "Z": np.array([[1, 0],  [0, -1]], dtype=complex),
}


def _embed_single(mat2, qubit, n):
    """Embed a 2×2 matrix on `qubit` into n-qubit space via Kronecker product."""
    mats = [_I2] * n
    mats[qubit] = mat2
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _rot_U(theta, pauli, qubit, n):
    """n-qubit unitary for R_{pauli}(theta) on `qubit`: exp(-i theta/2 * sigma)."""
    c = cos(theta / 2)
    s = sin(theta / 2)
    U2 = c * _I2 - 1j * s * _PAULI[pauli]
    return _embed_single(U2, qubit, n)


def _rot_G(pauli, qubit, n):
    """Generator matrix: sigma_pauli/2 embedded in n-qubit space."""
    return _embed_single(_PAULI[pauli], qubit, n) / 2.0


def _cnot_U(ctrl, tgt, n):
    """n-qubit CNOT unitary (qubit 0 = MSB convention)."""
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        ctrl_bit = (col >> (n - 1 - ctrl)) & 1
        row = col ^ (1 << (n - 1 - tgt)) if ctrl_bit else col
        U[row, col] = 1.0
    return U


def _frob(M):
    """Frobenius norm: sqrt(Tr(M† M))."""
    return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))


# ---------------------------------------------------------------------------
# Build gate_specs list  (forward order)
# ---------------------------------------------------------------------------

def build_gate_specs(params):
    """
    Returns list of dicts in circuit execution order.
    Each dict has at minimum:
      {"type": "param"|"cnot", "U": ndarray}
    Param gates additionally have {"G": ndarray, "key": (l,q,pi), "label": str}.
    """
    specs = []
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            for pi, pauli in enumerate(["Z", "Y", "Z"]):
                theta = float(params[l, q, pi])
                specs.append({
                    "type":  "param",
                    "key":   (l, q, pi),
                    "label": f"L{l}Q{q}{'RY' if pauli=='Y' else 'RZ'}{pi}",
                    "pauli": pauli,
                    "G":     _rot_G(pauli, q, N_QUBITS),
                    "U":     _rot_U(theta, pauli, q, N_QUBITS),
                })
        r = RANGES[l]
        for q in range(N_QUBITS):
            tgt = (q + r) % N_QUBITS
            specs.append({
                "type": "cnot",
                "ctrl": q,
                "tgt":  tgt,
                "U":    _cnot_U(q, tgt, N_QUBITS),
            })
    return specs


# ---------------------------------------------------------------------------
# Importance scoring: effective Hamiltonian
# ---------------------------------------------------------------------------

H_mat = sum(_embed_single(_PAULI["Z"], i, N_QUBITS) for i in range(N_QUBITS))  # H = Σ Z_i


def compute_importance_eff(params):
    """
    Effective commutator:

        O  ← H_mat
        for gate in reversed(circuit):
            if gate is parameterized:
                scores[k] = ‖[O, G_k]‖_F
            O ← U_k† O U_k

    At the moment gate k is scored, O = U_after† H U_after,
    i.e. H pulled back through all subsequent unitaries.
    This reflects the actual gradient contribution of gate k.
    """
    gate_specs = build_gate_specs(params)

    O      = H_mat.astype(complex).copy()
    score_map = {}

    for spec in reversed(gate_specs):
        if spec["type"] == "param":
            G = spec["G"]
            comm = O @ G - G @ O
            score_map[spec["key"]] = _frob(comm)

        U = spec["U"]
        O = U.conj().T @ O @ U   # pull H back: O ← U† O U

    records = []
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            for pi, pauli in enumerate(["Z", "Y", "Z"]):
                key = (l, q, pi)
                records.append({
                    "key":       key,
                    "label":     f"L{l}Q{q}{'RY' if pauli=='Y' else 'RZ'}{pi}",
                    "pauli":     pauli,
                    "importance": float(score_map[key]),
                    "flat_idx":  l * N_QUBITS * 3 + q * 3 + pi,
                })
    return sorted(records, key=lambda x: x["importance"], reverse=True)


def compute_importance_eff_avg(params_list):
    """
    Average effective-Hamiltonian importance scores over multiple parameter sets.

    For each params in params_list, compute the effective commutator score for
    every gate, then return the element-wise mean across all parameter sets.
    This makes the pruning decision less sensitive to the specific initial point.

    Returns records in the same format as compute_importance_eff, plus an
    additional "std" field with the standard deviation across parameter sets.
    """
    # Collect per-gate scores across all parameter sets: key → list of scores
    score_lists = {}   # (l, q, pi) → [score_at_params0, score_at_params1, ...]
    label_map   = {}
    pauli_map   = {}
    fidx_map    = {}

    for params in params_list:
        records = compute_importance_eff(params)
        for r in records:
            k = r["key"]
            if k not in score_lists:
                score_lists[k] = []
                label_map[k]   = r["label"]
                pauli_map[k]   = r["pauli"]
                fidx_map[k]    = r["flat_idx"]
            score_lists[k].append(r["importance"])

    # Compute mean and std per gate
    averaged = []
    for k, scores in score_lists.items():
        arr = np.array(scores)
        averaged.append({
            "key":        k,
            "label":      label_map[k],
            "pauli":      pauli_map[k],
            "importance": float(arr.mean()),
            "std":        float(arr.std()),
            "flat_idx":   fidx_map[k],
            "n_samples":  len(scores),
        })

    return sorted(averaged, key=lambda x: x["importance"], reverse=True)


# ---------------------------------------------------------------------------
# PennyLane circuit for energy evaluation
# ---------------------------------------------------------------------------

_dev_clean = qml.device("default.qubit", wires=N_QUBITS)
_H_pl = qml.Hamiltonian([1.0] * N_QUBITS, [qml.PauliZ(i) for i in WIRES])


@qml.qnode(_dev_clean, interface="autograd", diff_method="best")
def energy_full(weights):
    qml.StronglyEntanglingLayers(weights, wires=WIRES, ranges=RANGES, imprimitive=qml.CNOT)
    return qml.expval(_H_pl)


def _make_pruned_cost(active_mask, p=0.0):
    """Return a QNode with only active gates.

    When p > 0, insert matched PauliError before each active rotation gate
    (same convention as apply_sel_with_matched_pauli in src/ansatz.py):
      RZ gate → PauliError("Z", p) before RZ
      RY gate → PauliError("Y", p) before RY
    CNOT entangling gates are kept noise-free.
    """
    use_noise = p > 1e-9
    dev  = qml.device("default.mixed" if use_noise else "default.qubit", wires=N_QUBITS)
    _H   = qml.Hamiltonian([1.0] * N_QUBITS, [qml.PauliZ(i) for i in WIRES])
    _msk = active_mask.copy()
    _p   = float(np.clip(p, 1e-12, 1.0 - 1e-12)) if use_noise else 0.0

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(active_params):
        idx = 0
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                for pi, gate_name in enumerate(["RZ", "RY", "RZ"]):
                    if _msk[l, q, pi]:
                        if gate_name == "RY":
                            if use_noise:
                                qml.PauliError("Y", _p, wires=q)
                            qml.RY(active_params[idx], wires=q)
                        else:
                            if use_noise:
                                qml.PauliError("Z", _p, wires=q)
                            qml.RZ(active_params[idx], wires=q)
                        idx += 1
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit


def _active_mask(flat_idxs):
    mask = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=bool)
    for idx in flat_idxs:
        mask[idx // (N_QUBITS * 3), (idx % (N_QUBITS * 3)) // 3, idx % 3] = True
    return mask


def _make_selective_noise_cost(noise_mask, p=0.0):
    """Full circuit (all gates kept) with PauliError only on noise_mask=True gates.

    Unlike _make_pruned_cost, all 24 rotation gates are present and all
    parameters are trained. noise_mask selects which gates receive a matched
    PauliError channel (top-k important gates); the rest are noise-free.
    """
    use_noise = p > 1e-9
    dev  = qml.device("default.mixed" if use_noise else "default.qubit", wires=N_QUBITS)
    _H   = qml.Hamiltonian([1.0] * N_QUBITS, [qml.PauliZ(i) for i in WIRES])
    _msk = noise_mask.copy()
    _p   = float(np.clip(p, 1e-12, 1.0 - 1e-12)) if use_noise else 0.0

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params):
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                for pi, gate_name in enumerate(["RZ", "RY", "RZ"]):
                    idx = l * N_QUBITS * 3 + q * 3 + pi
                    if gate_name == "RY":
                        if use_noise and _msk[l, q, pi]:
                            qml.PauliError("Y", _p, wires=q)
                        qml.RY(params[idx], wires=q)
                    else:
                        if use_noise and _msk[l, q, pi]:
                            qml.PauliError("Z", _p, wires=q)
                        qml.RZ(params[idx], wires=q)
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit


# ---------------------------------------------------------------------------
# Manual Adam optimizer
# ---------------------------------------------------------------------------

def _adam_step(params, m, v, t, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    return params - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps), m, v


# ---------------------------------------------------------------------------
# Src imports for noise annealing  (lazy — loaded inside functions)
# ---------------------------------------------------------------------------

def _load_src():
    """Import Codes/src modules. Called once inside noise runners."""
    import importlib
    sys.path.insert(0, _CODES)
    C  = importlib.import_module("src.config")
    N  = importlib.import_module("src.noise")
    U  = importlib.import_module("src.utils")
    return C, N, U


# ---------------------------------------------------------------------------
# Training runners
# ---------------------------------------------------------------------------

def run_pruned_clean(init_params, active_mask, n_steps, lr):
    params = np.array(init_params).flatten()[active_mask.flatten()].copy()
    n_act  = len(params)
    cost_fn = _make_pruned_cost(active_mask, p=0.0)
    gfn     = qml.grad(cost_fn)
    m = np.zeros(n_act); v = np.zeros(n_act)
    hist = [float(cost_fn(pnp.array(params, requires_grad=True)))]
    for step in range(n_steps):
        g = np.array(gfn(pnp.array(params, requires_grad=True)))
        params, m, v = _adam_step(params, m, v, step + 1, g, lr)
        hist.append(float(cost_fn(pnp.array(params, requires_grad=True))))
        if (step + 1) % 100 == 0:
            print(f"    step {step+1:>4} | E = {hist[-1]:.6f}")
    return np.array(hist)


def run_noise_annealing(init_params, lr, schedule=None):
    """Full circuit, adaptive Pauli noise annealing via src code."""
    from math import ceil
    _C, _N, _U = _load_src()
    win      = max(2, int(ceil(_C.PAULI_WINDOW / _C.PAULI_CHECK_EVERY)))
    params   = _U.to_trainable(init_params)
    schedule = schedule if schedule is not None else _C.PAULI_SCHEDULE
    all_e, all_steps, total = [], [], 0

    print(f"  schedule: {schedule}")
    for si, noise_p in enumerate(schedule, 1):
        use_clean = noise_p <= _C.CLEAN_THRESHOLD
        cost_fn   = _N.CLEAN_COST if use_clean else _N.make_pauli_cost(noise_p)
        step_fn, desc = _U.make_stepper(
            cost_fn, _C.PAULI_OPTIMIZER, _C.PAULI_MAX_STEPS,
            _C.LR_PAULI_GD, lr,
            _C.PAULI_SPSA_ALPHA, _C.PAULI_SPSA_GAMMA,
            _C.PAULI_SPSA_C, _C.PAULI_SPSA_A, _C.PAULI_SPSA_a,
        )
        stage_ckpt = []; step_i = 0; t0 = time.time()
        label = "clean" if use_clean else f"p={noise_p}"
        print(f"\n  Stage {si}/{len(schedule)} | {label} | {desc}")
        while step_i < _C.PAULI_MAX_STEPS:
            params = step_fn(params); step_i += 1
            if step_i % _C.PAULI_CHECK_EVERY == 0:
                e = float(_N.CLEAN_COST(params))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total + step_i)
                conv, std, delta = _U.converged_ckpt(stage_ckpt, win, _C.PAULI_STD_TOL, _C.PAULI_RATE_TOL)
                if step_i >= _C.PAULI_MIN_STEPS and conv:
                    print(f"    converged step {step_i} | E={e:.6f} std={std:.5f} Δ={delta:.5f}")
                    break
        total += step_i
        print(f"    done: steps={step_i} | E={all_e[-1]:.6f} | {time.time()-t0:.1f}s")

    print(f"\n  total_steps={total} | final_E={all_e[-1]:.6f}")
    return np.array(all_e), np.array(all_steps), total


def run_pruned_noise(init_params, active_mask, lr, schedule=None):
    """Pruned circuit, adaptive Pauli noise annealing."""
    from math import ceil
    _C, _N, _U = _load_src()
    win      = max(2, int(ceil(_C.PAULI_WINDOW / _C.PAULI_CHECK_EVERY)))
    params   = np.array(init_params).flatten()[active_mask.flatten()].copy()
    n_act    = len(params)
    _clean   = _make_pruned_cost(active_mask, p=0.0)
    schedule = schedule if schedule is not None else _C.PAULI_SCHEDULE
    all_e, all_steps, total = [], [], 0

    print(f"  active gates: {n_act} | schedule: {schedule}")
    for si, noise_p in enumerate(schedule, 1):
        use_clean = noise_p <= _C.CLEAN_THRESHOLD
        cost_fn   = _clean if use_clean else _make_pruned_cost(active_mask, noise_p)
        gfn       = qml.grad(cost_fn)
        m = np.zeros(n_act); v = np.zeros(n_act)
        stage_ckpt = []; step_i = 0; t0 = time.time()
        label = "clean" if use_clean else f"p={noise_p}"
        print(f"\n  Stage {si}/{len(schedule)} | {label}")
        while step_i < _C.PAULI_MAX_STEPS:
            g = np.array(gfn(pnp.array(params, requires_grad=True)))
            params, m, v = _adam_step(params, m, v, step_i + 1, g, lr)
            step_i += 1
            if step_i % _C.PAULI_CHECK_EVERY == 0:
                e = float(_clean(pnp.array(params, requires_grad=True)))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total + step_i)
                conv, std, delta = _U.converged_ckpt(stage_ckpt, win, _C.PAULI_STD_TOL, _C.PAULI_RATE_TOL)
                if step_i >= _C.PAULI_MIN_STEPS and conv:
                    print(f"    converged step {step_i} | E={e:.6f} std={std:.5f} Δ={delta:.5f}")
                    break
        total += step_i
        print(f"    done: steps={step_i} | E={all_e[-1]:.6f} | {time.time()-t0:.1f}s")

    print(f"\n  total_steps={total} | final_E={all_e[-1]:.6f}")
    return np.array(all_e), np.array(all_steps), total


def run_selective_noise(init_params, noise_mask, lr, schedule=None):
    """Full circuit with noise only on top-k (noise_mask=True) gates.

    All 24 parameters are trained together; only the important gates
    receive PauliError noise during annealing.
    """
    from math import ceil
    _C, _N, _U = _load_src()
    win      = max(2, int(ceil(_C.PAULI_WINDOW / _C.PAULI_CHECK_EVERY)))
    params   = np.array(init_params).flatten().copy()   # all 24 params
    n_total  = len(params)
    _clean   = _make_selective_noise_cost(noise_mask, p=0.0)
    schedule = schedule if schedule is not None else _C.PAULI_SCHEDULE
    all_e, all_steps, total = [], [], 0

    n_noisy = int(noise_mask.sum())
    print(f"  noisy gates: {n_noisy}/24 | schedule: {schedule}")
    for si, noise_p in enumerate(schedule, 1):
        use_clean = noise_p <= _C.CLEAN_THRESHOLD
        cost_fn   = _clean if use_clean else _make_selective_noise_cost(noise_mask, noise_p)
        gfn       = qml.grad(cost_fn)
        m = np.zeros(n_total); v = np.zeros(n_total)
        stage_ckpt = []; step_i = 0; t0 = time.time()
        label = "clean" if use_clean else f"p={noise_p}"
        print(f"\n  Stage {si}/{len(schedule)} | {label}")
        while step_i < _C.PAULI_MAX_STEPS:
            g = np.array(gfn(pnp.array(params, requires_grad=True)))
            params, m, v = _adam_step(params, m, v, step_i + 1, g, lr)
            step_i += 1
            if step_i % _C.PAULI_CHECK_EVERY == 0:
                e = float(_clean(pnp.array(params, requires_grad=True)))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total + step_i)
                conv, std, delta = _U.converged_ckpt(stage_ckpt, win, _C.PAULI_STD_TOL, _C.PAULI_RATE_TOL)
                if step_i >= _C.PAULI_MIN_STEPS and conv:
                    print(f"    converged step {step_i} | E={e:.6f} std={std:.5f} Δ={delta:.5f}")
                    break
        total += step_i
        print(f"    done: steps={step_i} | E={all_e[-1]:.6f} | {time.time()-t0:.1f}s")

    print(f"\n  total_steps={total} | final_E={all_e[-1]:.6f}")
    return np.array(all_e), np.array(all_steps), total


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _importance_comparison_plot(eff_records, ts_dir):
    """Bar chart of effective importance scores with optional std error bars."""
    eff_by_key = {r["key"]: r for r in eff_records}
    all_keys   = sorted(eff_by_key.keys(), key=lambda k: eff_by_key[k]["importance"], reverse=True)

    labels   = [eff_by_key[k]["label"]      for k in all_keys]
    eff_vals = [eff_by_key[k]["importance"] for k in all_keys]
    stds     = [eff_by_key[k].get("std", 0) for k in all_keys]
    paulis   = [eff_by_key[k]["pauli"]      for k in all_keys]
    colors   = ["tab:orange" if p == "Y" else "steelblue" for p in paulis]

    x = np.arange(len(all_keys))
    has_std = any(s > 0 for s in stds)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, eff_vals, color=colors, edgecolor="black", lw=0.5,
           label="Eff commutator  ‖[H_eff, G_k]‖_F")
    if has_std:
        ax.errorbar(x, eff_vals, yerr=stds, fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=3, capthick=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Importance score (Frobenius)")
    title = "Effective Hamiltonian importance  (orange=RY, blue=RZ)"
    if has_std:
        title += "\nerror bars = std over averaged param sets"
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(ts_dir, "importance_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: importance_comparison.png")
    return out


def _training_plot(summary, top_k_list, clean_ref_hist, ts_dir, global_min, seed):
    """
    summary keys:
      pruned_{k}_eff           → np.ndarray  (clean Adam, step-indexed)
      noise_full               → (e_arr, steps_arr)
      pruned_{k}_eff_noise     → (e_arr, steps_arr)
      selective_{k}_eff_noise  → (e_arr, steps_arr)
    """
    _PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
    _COLORS = {k: c for k, c in zip(top_k_list, _PALETTE)}
    has_noise      = "noise_full" in summary
    has_selective  = any(f"selective_{k}_eff_noise" in summary for k in top_k_list)
    has_inv_sel    = any(f"inv_sel_{k}_eff_noise" in summary for k in top_k_list)

    ncols = 2 + int(has_noise) + int(has_selective) + int(has_inv_sel)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    ax_idx = 0

    # ── Panel 1: Clean Adam ───────────────────────────────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    if clean_ref_hist is not None:
        ax.plot(np.arange(len(clean_ref_hist)), clean_ref_hist,
                lw=2.2, color="black", label="Clean full (checkpoint ref)")
    for k in top_k_list:
        h = summary.get(f"pruned_{k}_eff")
        if h is not None:
            ax.plot(np.arange(len(h)), h, lw=1.8, color=_COLORS[k],
                    label=f"eff pruned-{k}")
    ax.axhline(global_min, color="red", ls=":", lw=1.2, label="Global min")
    ax.set_xlabel("Step"); ax.set_ylabel("Energy")
    ax.set_title(f"Clean Adam — eff pruning  (seed {seed})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2: Pruned noise annealing ──────────────────────────────────────
    if has_noise:
        ax = axes[ax_idx]; ax_idx += 1
        e, s = summary["noise_full"]
        ax.plot(s, e, lw=2.2, color="black",
                marker="o", ms=2, markevery=5, label=f"Noise full ({int(s[-1])} steps)")
        for k in top_k_list:
            key = f"pruned_{k}_eff_noise"
            if key in summary:
                pe, ps = summary[key]
                ax.plot(ps, pe, lw=1.8, color=_COLORS[k],
                        marker="s", ms=2, markevery=5,
                        label=f"eff pruned-{k} ({int(ps[-1])} steps)")
        ax.axhline(global_min, color="red", ls=":", lw=1.2, label="Global min")
        ax.set_xlabel("Cumulative step"); ax.set_ylabel("Energy (clean eval)")
        ax.set_title(f"Noise annealing — eff pruning  (seed {seed})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 3: Selective noise (full circuit, noise on top-k only) ──────────
    if has_selective:
        ax = axes[ax_idx]; ax_idx += 1
        if has_noise:
            e, s = summary["noise_full"]
            ax.plot(s, e, lw=2.2, color="black", ls="--",
                    marker="o", ms=2, markevery=5, label=f"Noise full (ref)")
        for k in top_k_list:
            key = f"selective_{k}_eff_noise"
            if key in summary:
                pe, ps = summary[key]
                ax.plot(ps, pe, lw=1.8, color=_COLORS[k],
                        marker="^", ms=2, markevery=5,
                        label=f"selective-{k} ({int(ps[-1])} steps)")
        ax.axhline(global_min, color="red", ls=":", lw=1.2, label="Global min")
        ax.set_xlabel("Cumulative step"); ax.set_ylabel("Energy (clean eval)")
        ax.set_title(f"Selective noise — full circuit, noise on top-k  (seed {seed})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 4: Inverse selective noise (noise on non-important gates) ──────
    if has_inv_sel:
        ax = axes[ax_idx]; ax_idx += 1
        if has_noise:
            e, s = summary["noise_full"]
            ax.plot(s, e, lw=2.2, color="black", ls="--",
                    marker="o", ms=2, markevery=5, label=f"Noise full (ref)")
        for k in top_k_list:
            key = f"inv_sel_{k}_eff_noise"
            if key in summary:
                ie, is_ = summary[key]
                ax.plot(is_, ie, lw=1.8, color=_COLORS[k],
                        marker="v", ms=2, markevery=5,
                        label=f"inv-sel-{k} ({int(is_[-1])} steps)")
        ax.axhline(global_min, color="red", ls=":", lw=1.2, label="Global min")
        ax.set_xlabel("Cumulative step"); ax.set_ylabel("Energy (clean eval)")
        ax.set_title(f"Inv-selective noise — noise on non-important gates  (seed {seed})")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel last: Final energy bar chart ────────────────────────────────────
    ax = axes[ax_idx]
    bar_keys, bar_vals, bar_cols = [], [], []
    ref_final = float(clean_ref_hist[-1]) if clean_ref_hist is not None else None
    if ref_final is not None:
        bar_keys.append("clean_full_ref"); bar_vals.append(ref_final); bar_cols.append("black")
    for k in top_k_list:
        key = f"pruned_{k}_eff"
        if key in summary:
            bar_keys.append(key); bar_vals.append(float(summary[key][-1])); bar_cols.append(_COLORS[k])
    if has_noise:
        e, _ = summary["noise_full"]
        bar_keys.append("noise_full"); bar_vals.append(float(e[-1])); bar_cols.append("crimson")
        for k in top_k_list:
            key = f"pruned_{k}_eff_noise"
            if key in summary:
                pe, _ = summary[key]
                bar_keys.append(key); bar_vals.append(float(pe[-1])); bar_cols.append(_COLORS[k])
    if has_selective:
        for k in top_k_list:
            key = f"selective_{k}_eff_noise"
            if key in summary:
                pe, _ = summary[key]
                bar_keys.append(key); bar_vals.append(float(pe[-1])); bar_cols.append(_COLORS[k])
    if has_inv_sel:
        for k in top_k_list:
            key = f"inv_sel_{k}_eff_noise"
            if key in summary:
                ie, _ = summary[key]
                bar_keys.append(key); bar_vals.append(float(ie[-1])); bar_cols.append(_COLORS[k])
    ax.barh(bar_keys, bar_vals, color=bar_cols, edgecolor="black", lw=0.5)
    ax.axvline(global_min, color="red", ls=":", lw=1.2, label="Global min")
    ax.set_xlabel("Final energy"); ax.set_title("Final energy summary")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle(f"seed={seed} | commutator_eff pruning", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(ts_dir, "training_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: training_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse()
    SEED        = args.seed
    TOP_K_LIST  = args.top_k
    N_STEPS     = args.n_steps
    LR          = args.lr
    N_AVG       = args.n_avg
    NOISE_SCALE = args.noise_scale

    # ── Resolve checkpoint paths for this seed ────────────────────────────────
    ckpt = _find_ckpt(SEED)
    params_map = {
        "init":  ckpt["init_params"],
        "clean": ckpt["clean_params"],
        "pauli": ckpt["pauli_params"],
    }

    # ── Load params ──────────────────────────────────────────────────────────
    score_params = np.load(params_map[args.params])
    # Regenerate init_params from seed using the same RNG as the original
    # importance_pruning.py — the saved checkpoint file was generated by a
    # different code path and does NOT match np.random.default_rng(seed).
    rng         = np.random.default_rng(SEED)
    init_params = rng.uniform(0.0, 2 * np.pi, (N_LAYERS, N_QUBITS, 3))
    clean_hist  = np.load(ckpt["clean_hist"])
    GLOBAL_MIN   = -float(N_QUBITS)

    # ── Build averaged-scoring param list ─────────────────────────────────────
    # When N_AVG > 1: draw N_AVG independent random parameter sets (each via a
    # deterministic sub-seed derived from SEED) and average the eff scores.
    # When N_AVG == 1: fall back to the single score_params point (original behavior).
    if N_AVG > 1:
        avg_params_list = [
            np.random.default_rng(SEED * 10000 + i).uniform(
                0.0, 2 * np.pi, (N_LAYERS, N_QUBITS, 3)
            )
            for i in range(N_AVG)
        ]
    else:
        avg_params_list = [score_params]

    # ── Output directory + 로그 파일 (모든 출력을 처음부터 캡처) ─────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    avg_tag    = f"_avg{N_AVG}" if N_AVG > 1 else ""
    scale_tag  = f"_ns{NOISE_SCALE:.2g}".replace(".", "p") if NOISE_SCALE != 1.0 else ""
    sel_tag    = "_sel" if args.selective_only else ("_invsel" if args.inv_sel_only else "")
    ts_dir     = os.path.join(_HERE, "results", f"{ts}_seed{SEED}{avg_tag}{scale_tag}{sel_tag}")
    os.makedirs(ts_dir, exist_ok=True)
    _tee   = _Tee(os.path.join(ts_dir, "run.log"))

    print("=" * 60)
    print(f"  commutator_eff importance scoring")
    print(f"  seed        : {SEED}  ({os.path.basename(ckpt['ckpt_dir'])})")
    if N_AVG > 1:
        print(f"  scoring at  : {N_AVG} random param sets (averaged eff)")
    else:
        print(f"  scoring at  : {args.params} params  (shape {score_params.shape})")
    print(f"  n_avg       : {N_AVG}")
    print(f"  noise_scale : {NOISE_SCALE}")
    print(f"  init energy : {float(energy_full(pnp.array(init_params, requires_grad=True))):.6f}")
    print(f"  global min  : {GLOBAL_MIN:.6f}")
    print("=" * 60)

    # ── Compute importance scores ────────────────────────────────────────────
    if N_AVG > 1:
        print(f"\n[1] Computing averaged effective-Hamiltonian importance scores ({N_AVG} param sets)...")
        eff_records = compute_importance_eff_avg(avg_params_list)
    else:
        print("\n[1] Computing effective-Hamiltonian importance scores...")
        eff_records = compute_importance_eff(score_params)

    print("\n  Effective score ranking:")
    if N_AVG > 1:
        print(f"  {'rank':>4} | {'label':>12} | {'eff score':>10} | {'std':>8}")
        print("  " + "-" * 42)
        for rank, r in enumerate(eff_records, 1):
            bar   = "█" * int(r["importance"] * 5)
            std_s = f"{r['std']:.4f}" if "std" in r else "N/A"
            print(f"  {rank:>4} | {r['label']:>12} | {r['importance']:>10.4f} | {std_s:>8}  {bar}")
    else:
        print(f"  {'rank':>4} | {'label':>12} | {'eff score':>10}")
        print("  " + "-" * 30)
        for rank, r in enumerate(eff_records, 1):
            bar = "█" * int(r["importance"] * 5)
            print(f"  {rank:>4} | {r['label']:>12} | {r['importance']:>10.4f}  {bar}")

    # ── Importance comparison plot ────────────────────────────────────────────
    print("\n[2] Plotting importance comparison...")
    _importance_comparison_plot(eff_records, ts_dir)

    if args.no_train:
        print("\n--no_train: skipping training experiments.")
        return

    # ── Pruning experiments ───────────────────────────────────────────────────
    summary   = {}
    cfg_runs  = {}
    t_global  = time.time()

    # [A] Clean Adam (skipped with --selective_only)
    if not args.selective_only:
        print(f"\n{'='*55}")
        print(f"  Clean Adam — eff scoring")
        print(f"{'='*55}")
    _clean_k_list = [] if (args.selective_only or args.inv_sel_only) else TOP_K_LIST
    for k in _clean_k_list:
        top_recs   = eff_records[:k]
        flat_idxs  = [r["flat_idx"] for r in top_recs]
        top_labels = [r["label"] for r in top_recs]
        mask_k     = _active_mask(flat_idxs)
        key        = f"pruned_{k}_eff"
        print(f"\n[{key}] top-{k}: {top_labels[:6]}{'...' if k>6 else ''}")
        t0   = time.time()
        hist = run_pruned_clean(init_params, mask_k, N_STEPS, LR)
        dt   = time.time() - t0
        summary[key] = hist
        np.save(os.path.join(ts_dir, f"{key}.npy"), hist)
        cfg_runs[key] = {
            "type": "clean", "scoring": "eff",
            "selected_params": top_labels,
            "final_energy": float(hist[-1]),
            "n_steps": len(hist) - 1, "time_s": round(dt, 1),
        }
        print(f"  → final E = {hist[-1]:.6f}  |  {dt:.1f}s")

    # [B] Noise annealing
    if not args.no_noise:
        print(f"\n{'='*55}")
        print(f"  Noise annealing experiments")
        print(f"{'='*55}")

        # 스케줄 계산 (0.0은 그대로 유지, _load_src()로 경로 포함해서 import)
        _C_tmp, _, _ = _load_src()
        noise_schedule = [p * NOISE_SCALE if p > 0 else 0.0 for p in _C_tmp.PAULI_SCHEDULE]

        if not args.selective_only and not args.inv_sel_only:
            # B-1: full circuit (24 gates, all noisy)
            print(f"\n[noise_full]")
            t0 = time.time()
            ne, ns, ntot = run_noise_annealing(init_params, LR, schedule=noise_schedule)
            dt = time.time() - t0
            summary["noise_full"] = (ne, ns)
            np.save(os.path.join(ts_dir, "noise_full_e.npy"),     ne)
            np.save(os.path.join(ts_dir, "noise_full_steps.npy"), ns)
            cfg_runs["noise_full"] = {
                "type": "noise", "scoring": "full",
                "final_energy": float(ne[-1]),
                "total_steps": ntot, "time_s": round(dt, 1),
            }
            print(f"  → final E = {ne[-1]:.6f}  |  {dt:.1f}s")

            # B-2: pruned circuits (top-k gates only, noise on those)
            for k in TOP_K_LIST:
                top_recs   = eff_records[:k]
                flat_idxs  = [r["flat_idx"] for r in top_recs]
                top_labels = [r["label"] for r in top_recs]
                mask_k     = _active_mask(flat_idxs)
                key        = f"pruned_{k}_eff_noise"
                print(f"\n[{key}] top-{k}: {top_labels[:6]}{'...' if k>6 else ''}")
                t0 = time.time()
                pe, ps, ptot = run_pruned_noise(init_params, mask_k, LR, schedule=noise_schedule)
                dt = time.time() - t0
                summary[key] = (pe, ps)
                np.save(os.path.join(ts_dir, f"{key}_e.npy"),     pe)
                np.save(os.path.join(ts_dir, f"{key}_steps.npy"), ps)
                cfg_runs[key] = {
                    "type": "noise", "scoring": "eff",
                    "selected_params": top_labels,
                    "final_energy": float(pe[-1]),
                    "total_steps": ptot, "time_s": round(dt, 1),
                }
                print(f"  → final E = {pe[-1]:.6f}  |  {dt:.1f}s")

        # B-3: selective noise (full circuit, noise on top-k important gates)
        if not args.inv_sel_only:
            print("\n[B-3] Selective noise experiments...")
            for k in TOP_K_LIST:
                top_recs   = eff_records[:k]
                flat_idxs  = [r["flat_idx"] for r in top_recs]
                top_labels = [r["label"] for r in top_recs]
                noise_mask = _active_mask(flat_idxs)   # True on top-k gates
                key        = f"selective_{k}_eff_noise"
                print(f"\n[{key}] noise on top-{k}: {top_labels[:6]}{'...' if k>6 else ''}")
                t0 = time.time()
                se, ss, stot = run_selective_noise(init_params, noise_mask, LR, schedule=noise_schedule)
                dt = time.time() - t0
                summary[key] = (se, ss)
                np.save(os.path.join(ts_dir, f"{key}_e.npy"),     se)
                np.save(os.path.join(ts_dir, f"{key}_steps.npy"), ss)
                cfg_runs[key] = {
                    "type": "selective_noise", "scoring": "eff",
                    "noisy_params": top_labels,
                    "final_energy": float(se[-1]),
                    "total_steps": stot, "time_s": round(dt, 1),
                }
                print(f"  → final E = {se[-1]:.6f}  |  {dt:.1f}s")

        # B-4: inverse selective noise (full circuit, noise on NON-important gates)
        if args.inv_sel_only:
            print("\n[B-4] Inverse selective noise (noise on non-important gates)...")
            for k in TOP_K_LIST:
                top_recs    = eff_records[:k]
                flat_idxs   = [r["flat_idx"] for r in top_recs]
                top_labels  = [r["label"] for r in top_recs]
                important_mask = _active_mask(flat_idxs)
                noise_mask  = ~important_mask   # noise on the OTHER (non-important) gates
                bot_labels  = [r["label"] for r in eff_records[k:]]
                key         = f"inv_sel_{k}_eff_noise"
                print(f"\n[{key}] noise on bottom-{24-k} (non-top-{k}): {bot_labels[:4]}{'...' if len(bot_labels)>4 else ''}")
                t0 = time.time()
                ie, is_, itot = run_selective_noise(init_params, noise_mask, LR, schedule=noise_schedule)
                dt = time.time() - t0
                summary[key] = (ie, is_)
                np.save(os.path.join(ts_dir, f"{key}_e.npy"),     ie)
                np.save(os.path.join(ts_dir, f"{key}_steps.npy"), is_)
                cfg_runs[key] = {
                    "type": "inv_selective_noise", "scoring": "eff",
                    "noisy_params": bot_labels,
                    "final_energy": float(ie[-1]),
                    "total_steps": itot, "time_s": round(dt, 1),
                }
                print(f"  → final E = {ie[-1]:.6f}  |  {dt:.1f}s")

    # ── Training plot ─────────────────────────────────────────────────────────
    print("\n[3] Plotting training comparison...")
    _training_plot(summary, TOP_K_LIST, clean_hist, ts_dir, GLOBAL_MIN, SEED)

    total_time = time.time() - t_global

    # ── Save config ───────────────────────────────────────────────────────────
    cfg = {
        "timestamp":    ts,
        "seed":         SEED,
        "ckpt_dir":     os.path.basename(ckpt["ckpt_dir"]),
        "score_params": args.params,
        "n_qubits":     N_QUBITS,
        "n_layers":     N_LAYERS,
        "ranges":       RANGES,
        "top_k_list":   TOP_K_LIST,
        "n_steps":      N_STEPS,
        "lr":           LR,
        "no_noise":     args.no_noise,
        "n_avg":        N_AVG,
        "noise_scale":  NOISE_SCALE,
        "global_min":   GLOBAL_MIN,
        "total_time_s": round(total_time, 1),
        "eff_ranking":  [{"label": r["label"], "score": round(r["importance"], 6),
                          **({"std": round(r["std"], 6)} if "std" in r else {})}
                         for r in eff_records],
        "runs": cfg_runs,
    }
    with open(os.path.join(ts_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'experiment':>30} | {'final E':>10} | {'time':>7}")
    print("-" * 65)
    all_exp_keys = (
        [f"pruned_{k}_eff" for k in TOP_K_LIST]
        + (["noise_full"]
           + [f"pruned_{k}_eff_noise" for k in TOP_K_LIST]
           + [f"selective_{k}_eff_noise" for k in TOP_K_LIST]
           + [f"inv_sel_{k}_eff_noise" for k in TOP_K_LIST]
           if not args.no_noise else [])
    )
    for key in all_exp_keys:
        if key not in cfg_runs:
            continue
        r  = cfg_runs[key]
        fe = r["final_energy"]
        ts_str = f"{r['time_s']:>6.1f}s"
        print(f"  {key:>30} | {fe:>10.6f} | {ts_str}")
    print("-" * 65)
    print(f"  {'global min':>30} | {GLOBAL_MIN:>10.6f} |")
    print(f"  total: {total_time:.0f}s")
    print("=" * 65)
    print(f"\nResults saved → {ts_dir}/")

    _tee.close()


if __name__ == "__main__":
    main()
