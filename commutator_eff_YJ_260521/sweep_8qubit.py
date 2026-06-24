"""
sweep_8qubit.py — B1/B2/B3/B4 noise placement study for 8-qubit TFIM
----------------------------------------------------------------------
Commutator-eff importance scoring on 8 qubits (4 layers, ranges=[1,2,3,4]).
Top-k as percentages: 20,40,60,80,100% of 96 total rotation gates.

Methods:
  B1 — noise_full      : all 96 gates kept & trained, all noisy
  B2 — pruned_noise    : top-k gates kept & trained, those k are noisy
  B3 — selective       : all 96 gates kept & trained, noise only on top-k
  B4 — inv_selective   : all 96 gates kept & trained, noise on bottom-(96-k)

Usage:
  python sweep_8qubit.py --h_idx 0 --seed 0
  python sweep_8qubit.py --h_idx 1 --seed 42 --n_avg 5
  python sweep_8qubit.py --h_idx 0 --seed 0 --methods b3 b4
  python sweep_8qubit.py --h_idx 0 --seed 0 --noise_scale 0.5
"""

import sys, os, json, argparse, time, csv
from math import cos, sin
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Circuit & study constants
# ---------------------------------------------------------------------------

N_QUBITS  = 8
N_LAYERS  = 4
RANGES    = [1, 2, 3, 4]
WIRES     = list(range(N_QUBITS))
DIM       = 2 ** N_QUBITS
N_GATES   = N_QUBITS * N_LAYERS * 3   # 96 rotation gates

TOP_K_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
TOP_K_LIST  = [max(1, round(N_GATES * f)) for f in TOP_K_FRACS]
# [19, 38, 58, 77, 96]

# Convergence constants
# Schedule: gradient_JY_260514/src/config.py (0.8 start, 9 stages)
# PauliError range: 0~1 (직접 전달, JY 코드의 0~0.5 변환 없이)
PAULI_SCHEDULE   = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]
PAULI_MIN_STEPS  = 50
PAULI_MAX_STEPS  = 1000
PAULI_CHECK_EVERY = 10
PAULI_WINDOW     = 20
PAULI_STD_TOL    = 0.005
PAULI_RATE_TOL   = 0.005
CLEAN_THRESHOLD  = 0.0   # p=0.0 only → use clean device
DEFAULT_LR       = 0.01

# 5 Hamiltonians from targeted_tfim_8qubit experiment
HAMILTONIANS = [
    {"h_id": "H00000", "jzz": -1.022768208338797,  "hx":  1.0429285161955684, "hz": 0.0, "ground": -10.183437124401689},
    {"h_id": "H00001", "jzz":  0.9185630992485563, "hx": -1.9134118252881938, "hz": 0.0, "ground": -16.087213925340926},
    {"h_id": "H00002", "jzz": -1.922442538910711,  "hx":  0.8416061333504223, "hz": 0.0, "ground": -14.394334375222694},
    {"h_id": "H00003", "jzz":  1.3212546470614317, "hx":  0.8099413834553117, "hz": 0.0, "ground": -10.545593734894037},
    {"h_id": "H00004", "jzz": -1.7119030790166239, "hx":  0.17443229811114325,"hz": 0.0, "ground": -12.02779019891294},
]

# ---------------------------------------------------------------------------
# Matrix building blocks (generalized; 256×256 for N_QUBITS=8)
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_PAULI = {
    "X": np.array([[0, 1],   [1,  0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0],   [0, -1]], dtype=complex),
}


def _embed_single(mat2, qubit, n=N_QUBITS):
    mats = [_I2] * n
    mats[qubit] = mat2
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _embed_multi(qubit_mat_dict, n=N_QUBITS):
    """Build n-qubit matrix: qubit_mat_dict = {q: 2×2 mat}, rest = I."""
    mats = [qubit_mat_dict.get(q, _I2) for q in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _rot_U(theta, pauli, qubit, n=N_QUBITS):
    c = cos(theta / 2); s = sin(theta / 2)
    U2 = c * _I2 - 1j * s * _PAULI[pauli]
    return _embed_single(U2, qubit, n)


def _rot_G(pauli, qubit, n=N_QUBITS):
    return _embed_single(_PAULI[pauli], qubit, n) / 2.0


def _cnot_U(ctrl, tgt, n=N_QUBITS):
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        ctrl_bit = (col >> (n - 1 - ctrl)) & 1
        row = col ^ (1 << (n - 1 - tgt)) if ctrl_bit else col
        U[row, col] = 1.0
    return U


def _frob(M):
    return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))


def build_tfim_matrix(jzz, hx, hz=0.0, n=N_QUBITS):
    """Full 2^n × 2^n TFIM Hamiltonian matrix (open boundary)."""
    H = np.zeros((2**n, 2**n), dtype=complex)
    Z = _PAULI["Z"]; X = _PAULI["X"]
    for i in range(n - 1):  # open boundary ZZ
        H += -jzz * _embed_multi({i: Z, i+1: Z}, n)
    for i in range(n):
        H += -hx * _embed_single(X, i, n)
    if hz != 0.0:
        for i in range(n):
            H += -hz * _embed_single(Z, i, n)
    return H


# ---------------------------------------------------------------------------
# Gate specs (forward circuit order)
# ---------------------------------------------------------------------------

def build_gate_specs(params):
    """Returns ordered list of gate dicts for N_QUBITS=8, N_LAYERS=4 circuit."""
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
                    "G":     _rot_G(pauli, q),
                    "U":     _rot_U(theta, pauli, q),
                })
        r = RANGES[l]
        for q in range(N_QUBITS):
            tgt = (q + r) % N_QUBITS
            specs.append({"type": "cnot", "ctrl": q, "tgt": tgt, "U": _cnot_U(q, tgt)})
    return specs


# ---------------------------------------------------------------------------
# Importance scoring: effective Hamiltonian commutator
# ---------------------------------------------------------------------------

def compute_importance_eff(params, H_mat):
    gate_specs = build_gate_specs(params)
    O = H_mat.astype(complex).copy()
    score_map = {}
    for spec in reversed(gate_specs):
        if spec["type"] == "param":
            G = spec["G"]
            score_map[spec["key"]] = _frob(O @ G - G @ O)
        U = spec["U"]
        O = U.conj().T @ O @ U
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


def compute_importance_eff_avg(H_mat, n_avg, base_seed):
    """Average eff scores over n_avg random param sets."""
    score_lists = {}; label_map = {}; pauli_map = {}; fidx_map = {}
    for i in range(n_avg):
        params = np.random.default_rng(base_seed * 10000 + i).uniform(
            0.0, 2 * np.pi, (N_LAYERS, N_QUBITS, 3))
        records = compute_importance_eff(params, H_mat)
        for r in records:
            k = r["key"]
            if k not in score_lists:
                score_lists[k] = []; label_map[k] = r["label"]
                pauli_map[k] = r["pauli"]; fidx_map[k] = r["flat_idx"]
            score_lists[k].append(r["importance"])
    averaged = []
    for k, scores in score_lists.items():
        arr = np.array(scores)
        averaged.append({
            "key": k, "label": label_map[k], "pauli": pauli_map[k],
            "importance": float(arr.mean()), "std": float(arr.std()),
            "flat_idx": fidx_map[k], "n_samples": len(scores),
        })
    return sorted(averaged, key=lambda x: x["importance"], reverse=True)


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def flat_to_mask(flat_idxs):
    mask = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=bool)
    for idx in flat_idxs:
        l  = idx // (N_QUBITS * 3)
        q  = (idx % (N_QUBITS * 3)) // 3
        pi = idx % 3
        mask[l, q, pi] = True
    return mask


# ---------------------------------------------------------------------------
# PennyLane cost function factories
# ---------------------------------------------------------------------------

def _build_tfim_pl(jzz, hx, hz=0.0, n=N_QUBITS):
    """Build PennyLane TFIM Hamiltonian observable."""
    coeffs, obs = [], []
    for i in range(n - 1):
        coeffs.append(-jzz); obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
    for i in range(n):
        coeffs.append(-hx); obs.append(qml.PauliX(i))
    if hz != 0.0:
        for i in range(n):
            coeffs.append(-hz); obs.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, obs)


def make_clean_cost(H_pl):
    """Full circuit, no noise (default.qubit). Used for clean eval at each checkpoint."""
    dev = qml.device("default.qubit", wires=N_QUBITS)
    _H  = H_pl

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params):
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.RZ(params[l, q, 0], wires=q)
                qml.RY(params[l, q, 1], wires=q)
                qml.RZ(params[l, q, 2], wires=q)
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit


def make_full_noise_cost(H_pl, p):
    """B1: all 96 gates, all noisy."""
    use_noise = p > 1e-9
    dev = qml.device("default.mixed" if use_noise else "default.qubit", wires=N_QUBITS)
    _H  = H_pl
    _p  = float(np.clip(p, 1e-12, 1 - 1e-12)) if use_noise else 0.0

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params):
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                if use_noise:
                    qml.PauliError("Z", _p, wires=q)
                qml.RZ(params[l, q, 0], wires=q)
                if use_noise:
                    qml.PauliError("Y", _p, wires=q)
                qml.RY(params[l, q, 1], wires=q)
                if use_noise:
                    qml.PauliError("Z", _p, wires=q)
                qml.RZ(params[l, q, 2], wires=q)
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit


def make_pruned_cost(H_pl, active_mask, p):
    """B2: only top-k gates kept, those k gates noisy if p>0."""
    use_noise = p > 1e-9
    dev  = qml.device("default.mixed" if use_noise else "default.qubit", wires=N_QUBITS)
    _H   = H_pl
    _msk = active_mask.copy()
    _p   = float(np.clip(p, 1e-12, 1 - 1e-12)) if use_noise else 0.0

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(active_params):
        idx = 0
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                for pi, gate_name in enumerate(["RZ", "RY", "RZ"]):
                    if _msk[l, q, pi]:
                        if gate_name == "RY":
                            if use_noise: qml.PauliError("Y", _p, wires=q)
                            qml.RY(active_params[idx], wires=q)
                        else:
                            if use_noise: qml.PauliError("Z", _p, wires=q)
                            qml.RZ(active_params[idx], wires=q)
                        idx += 1
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit


def make_selective_cost(H_pl, noise_mask, p):
    """B3/B4: all 96 gates kept & trained, noise only where noise_mask=True."""
    use_noise = p > 1e-9
    dev  = qml.device("default.mixed" if use_noise else "default.qubit", wires=N_QUBITS)
    _H   = H_pl
    _msk = noise_mask.copy()
    _p   = float(np.clip(p, 1e-12, 1 - 1e-12)) if use_noise else 0.0

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params):
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                for pi, gate_name in enumerate(["RZ", "RY", "RZ"]):
                    noisy = use_noise and _msk[l, q, pi]
                    if gate_name == "RY":
                        if noisy: qml.PauliError("Y", _p, wires=q)
                        qml.RY(params[l, q, pi], wires=q)
                    else:
                        if noisy: qml.PauliError("Z", _p, wires=q)
                        qml.RZ(params[l, q, pi], wires=q)
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit


# ---------------------------------------------------------------------------
# Adam optimizer — use qml.AdamOptimizer (matches jungyun implementation)


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def check_converged(hist, win, std_tol, rate_tol):
    if len(hist) < win:
        return False, 0.0, 0.0
    tail  = np.array(hist[-win:])
    std   = float(tail.std())
    delta = float(abs(tail[-1] - tail[0]) / max(1, win))
    return (std < std_tol and delta < rate_tol), std, delta


# ---------------------------------------------------------------------------
# Generic noise annealing runner
# ---------------------------------------------------------------------------

def run_anneal(cost_factory, clean_cost, init_params_flat, n_params,
               schedule, lr, label=""):
    """
    cost_factory(p) -> QNode that takes flattened params of shape (n_params,).
    clean_cost     -> QNode taking same params, noise-free (for eval).
    Returns: (energies_arr, steps_arr, total_steps, elapsed_s)
    """
    params = np.array(init_params_flat).flatten()[:n_params].copy()
    all_e, all_steps, total = [], [], 0
    t_start = time.time()
    win = max(2, PAULI_WINDOW // PAULI_CHECK_EVERY)

    for si, noise_p in enumerate(schedule, 1):
        use_clean = noise_p <= CLEAN_THRESHOLD
        cost_fn   = clean_cost if use_clean else cost_factory(noise_p)
        opt       = qml.AdamOptimizer(stepsize=lr)
        stage_ckpt = []; step_i = 0; t0 = time.time()
        stage_lbl = "clean" if use_clean else f"p={noise_p:.3f}"

        while step_i < PAULI_MAX_STEPS:
            params = np.array(opt.step(cost_fn, pnp.array(params, requires_grad=True)))
            step_i += 1
            if step_i % PAULI_CHECK_EVERY == 0:
                e = float(clean_cost(pnp.array(params, requires_grad=True)))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total + step_i)
                conv, std, delta = check_converged(stage_ckpt, win, PAULI_STD_TOL, PAULI_RATE_TOL)
                if step_i >= PAULI_MIN_STEPS and conv:
                    print(f"    [{label}] stage {si} converged at step {step_i}  E={e:.5f}")
                    break

        total += step_i
        e_last = all_e[-1] if all_e else float("nan")
        print(f"  [{label}] stage {si}/{len(schedule)} {stage_lbl}  "
              f"steps={step_i}  E={e_last:.5f}  ({time.time()-t0:.1f}s)")

    return (np.array(all_e), np.array(all_steps), total, time.time() - t_start)


# ---------------------------------------------------------------------------
# B1: full circuit, all gates noisy
# ---------------------------------------------------------------------------

def run_b1(init_params, H_pl, schedule, lr):
    clean_cost = make_clean_cost(H_pl)
    flat = init_params.flatten()

    def _factory(p):
        return make_full_noise_cost(H_pl, p)

    # cost_factory takes flat (N_LAYERS,N_QUBITS,3)-shaped array reshaped inside
    # but clean_cost / factory expect (N_LAYERS, N_QUBITS, 3) shaped
    # wrap to handle reshape

    class _CleanWrapper:
        def __call__(self, p):
            return clean_cost(p.reshape(N_LAYERS, N_QUBITS, 3))
        def __call__(self, p):
            return clean_cost(p.reshape(N_LAYERS, N_QUBITS, 3))

    # Actually let's just use the original shape (N_LAYERS, N_QUBITS, 3) params
    # and adapt run_anneal accordingly
    return _run_anneal_shaped(make_clean_cost(H_pl),
                              lambda p: make_full_noise_cost(H_pl, p),
                              init_params, schedule, lr, "B1-full")


def _run_anneal_shaped(clean_cost_fn, noisy_factory, init_params, schedule, lr, label=""):
    """
    Noise annealing where params are (N_LAYERS, N_QUBITS, 3) shaped.
    clean_cost_fn(params) and noisy_factory(p)(params) take that shape.
    """
    params = np.array(init_params).copy()
    all_e, all_steps, total = [], [], 0
    t_start = time.time()
    win = max(2, PAULI_WINDOW // PAULI_CHECK_EVERY)

    for si, noise_p in enumerate(schedule, 1):
        use_clean = noise_p <= CLEAN_THRESHOLD
        if use_clean:
            cost_fn = clean_cost_fn
        else:
            cost_fn = noisy_factory(noise_p)
        opt = qml.AdamOptimizer(stepsize=lr)
        stage_ckpt = []; step_i = 0; t0 = time.time()
        stage_lbl = "clean" if use_clean else f"p={noise_p:.3f}"

        while step_i < PAULI_MAX_STEPS:
            params = np.array(opt.step(cost_fn, pnp.array(params, requires_grad=True)))
            step_i += 1
            if step_i % PAULI_CHECK_EVERY == 0:
                e = float(clean_cost_fn(pnp.array(params, requires_grad=True)))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total + step_i)
                conv, std, delta = check_converged(stage_ckpt, win, PAULI_STD_TOL, PAULI_RATE_TOL)
                if step_i >= PAULI_MIN_STEPS and conv:
                    print(f"    [{label}] stage {si} converged step {step_i}  E={e:.5f}")
                    break

        total += step_i
        e_last = all_e[-1] if all_e else float("nan")
        print(f"  [{label}] stage {si}/{len(schedule)} {stage_lbl}  "
              f"steps={step_i}  E={e_last:.5f}  ({time.time()-t0:.1f}s)")

    return np.array(all_e), np.array(all_steps), total, time.time() - t_start


# ---------------------------------------------------------------------------
# B2: pruned circuit (flat active_params)
# ---------------------------------------------------------------------------

def _run_anneal_flat(clean_cost_fn, noisy_factory, init_active_params, label="",
                     schedule=None, lr=DEFAULT_LR):
    """Annealing with flat active_params (for B2 pruned circuit)."""
    if schedule is None:
        schedule = PAULI_SCHEDULE
    params = np.array(init_active_params).flatten().copy()
    all_e, all_steps, total = [], [], 0
    t_start = time.time()
    win = max(2, PAULI_WINDOW // PAULI_CHECK_EVERY)

    for si, noise_p in enumerate(schedule, 1):
        use_clean = noise_p <= CLEAN_THRESHOLD
        cost_fn   = clean_cost_fn if use_clean else noisy_factory(noise_p)
        opt       = qml.AdamOptimizer(stepsize=lr)
        stage_ckpt = []; step_i = 0; t0 = time.time()
        stage_lbl = "clean" if use_clean else f"p={noise_p:.3f}"

        while step_i < PAULI_MAX_STEPS:
            params = np.array(opt.step(cost_fn, pnp.array(params, requires_grad=True)))
            step_i += 1
            if step_i % PAULI_CHECK_EVERY == 0:
                e = float(clean_cost_fn(pnp.array(params, requires_grad=True)))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total + step_i)
                conv, _, _ = check_converged(stage_ckpt, win, PAULI_STD_TOL, PAULI_RATE_TOL)
                if step_i >= PAULI_MIN_STEPS and conv:
                    print(f"    [{label}] stage {si} converged step {step_i}  E={e:.5f}")
                    break

        total += step_i
        e_last = all_e[-1] if all_e else float("nan")
        print(f"  [{label}] stage {si}/{len(schedule)} {stage_lbl}  "
              f"steps={step_i}  E={e_last:.5f}  ({time.time()-t0:.1f}s)")

    return np.array(all_e), np.array(all_steps), total, time.time() - t_start


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="8-qubit TFIM noise placement study")
    p.add_argument("--h_idx",       type=int,   default=0,
                   help="Hamiltonian index 0-4 (default: 0)")
    p.add_argument("--seed",        type=int,   default=0,
                   help="Random seed for initial params (default: 0)")
    p.add_argument("--lr",          type=float, default=DEFAULT_LR)
    p.add_argument("--n_avg",       type=int,   default=50,
                   help="Number of random param sets to average eff scores over (default: 50)")
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="Scale factor on noise schedule (default: 1.0)")
    p.add_argument("--methods",     nargs="+",
                   choices=["a", "b1", "b2", "b3", "b4"],
                   default=["a", "b1", "b2", "b3", "b4"],
                   help="Which methods to run (default: all)")
    p.add_argument("--out_base",    type=str,   default=None,
                   help="Override output base directory")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    H_cfg = HAMILTONIANS[args.h_idx]
    H_ID  = H_cfg["h_id"]
    jzz   = H_cfg["jzz"]; hx = H_cfg["hx"]; hz = H_cfg["hz"]
    GROUND = H_cfg["ground"]
    SEED  = args.seed
    LR    = args.lr
    METHODS = set(args.methods)

    # Output directory
    if args.out_base is not None:
        out_base = args.out_base
    else:
        ts_study = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        out_base = os.path.join(results_dir, f"noise_placement_study3_8qubit_{ts_study}")

    run_dir = os.path.join(out_base, H_ID, f"seed{SEED:04d}")
    os.makedirs(run_dir, exist_ok=True)

    # Log to file + stdout
    log_path = os.path.join(run_dir, "run.log")
    log_f    = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self): self._real = sys.stdout; sys.stdout = self
        def write(self, s): self._real.write(s); log_f.write(s)
        def flush(self): self._real.flush(); log_f.flush()
        def close(self): sys.stdout = self._real; log_f.close()

    tee = _Tee()

    print("=" * 65)
    print(f"  sweep_8qubit.py — 8-qubit TFIM noise placement study")
    print(f"  H      : {H_ID}  jzz={jzz:.4f}  hx={hx:.4f}  hz={hz}")
    print(f"  ground : {GROUND:.5f}")
    print(f"  seed   : {SEED}  lr={LR}  n_avg={args.n_avg}  ns={args.noise_scale}")
    print(f"  methods: {sorted(METHODS)}")
    print(f"  top_k  : {TOP_K_LIST}  (fracs: {TOP_K_FRACS})")
    print(f"  out    : {run_dir}")
    print("=" * 65)

    # Build H
    H_mat = build_tfim_matrix(jzz, hx, hz)
    H_pl  = _build_tfim_pl(jzz, hx, hz)

    # Initial params (random)
    rng         = np.random.default_rng(SEED)
    init_params = rng.uniform(0.0, 2 * np.pi, (N_LAYERS, N_QUBITS, 3))

    clean_cost = make_clean_cost(H_pl)
    init_e = float(clean_cost(pnp.array(init_params, requires_grad=True)))
    print(f"\n  init energy : {init_e:.5f}  (ground={GROUND:.5f}  global_min/{abs(GROUND):.4f})")

    # Importance scoring
    print(f"\n[1] Eff commutator scoring (n_avg={args.n_avg})...")
    t0 = time.time()
    if args.n_avg > 1:
        eff_records = compute_importance_eff_avg(H_mat, args.n_avg, SEED)
    else:
        eff_records = compute_importance_eff(init_params, H_mat)
    print(f"  scoring done in {time.time()-t0:.1f}s")

    # Top-k mask lookup
    def _mask_for_k(k, invert=False):
        flat_idxs = [r["flat_idx"] for r in eff_records[:k]]
        mask = flat_to_mask(flat_idxs)
        return ~mask if invert else mask

    # Noise schedule
    sched = [p * args.noise_scale if p > CLEAN_THRESHOLD else 0.0
             for p in PAULI_SCHEDULE]

    # Save eff scores
    scores_path = os.path.join(run_dir, "eff_scores.json")
    with open(scores_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "key"} for r in eff_records], f, indent=2)

    print(f"\n  Top-10 eff importance scores:")
    for i, r in enumerate(eff_records[:10], 1):
        std_s = f"±{r['std']:.3f}" if "std" in r else ""
        print(f"    {i:>2}. {r['label']:>10}  {r['importance']:.4f}{std_s}")

    summary = {}
    cfg_runs = {}
    t_global = time.time()

    # ── Block A: clean Adam (no noise), pruned top-k gates ────────────────────
    if "a" in METHODS:
        print(f"\n{'='*55}\n  Block A — clean Adam, top-k gates only (no noise)\n{'='*55}")
        CLEAN_N_STEPS = 1000
        for k in TOP_K_LIST:
            mask_k       = _mask_for_k(k)
            active_init  = init_params.flatten()[mask_k.flatten()]
            clean_k      = make_pruned_cost(H_pl, mask_k, 0.0)
            params_a     = np.array(active_init).flatten().copy()
            n_a          = len(params_a)
            opt_a        = qml.AdamOptimizer(stepsize=LR)
            hist = [float(clean_k(pnp.array(params_a, requires_grad=True)))]
            t0 = time.time()
            print(f"\n  [A] k={k} ({100*k//N_GATES}%)  active_params={n_a}")
            for step in range(CLEAN_N_STEPS):
                params_a = np.array(opt_a.step(clean_k, pnp.array(params_a, requires_grad=True)))
                hist.append(float(clean_k(pnp.array(params_a, requires_grad=True))))
                if (step + 1) % 200 == 0:
                    print(f"    step {step+1:>4} | E={hist[-1]:.5f}")
            dt  = time.time() - t0
            key = f"pruned_{k}_clean"
            summary[key] = np.array(hist)
            np.save(os.path.join(run_dir, f"{key}.npy"), np.array(hist))
            cfg_runs[key] = {"final_e": float(hist[-1]), "steps": CLEAN_N_STEPS,
                             "time_s": round(dt, 1), "k": k}
            print(f"  → A-k{k} final E = {hist[-1]:.5f}  |  {dt:.1f}s")

    # ── B1: noise_full ─────────────────────────────────────────────────────────
    if "b1" in METHODS:
        print(f"\n{'='*55}\n  B1 — noise_full (all 96 gates noisy)\n{'='*55}")
        ne, ns, ntot, dt = _run_anneal_shaped(
            clean_cost,
            lambda p: make_full_noise_cost(H_pl, p),
            init_params, sched, LR, label="B1-full")
        summary["noise_full"] = (ne, ns)
        np.save(os.path.join(run_dir, "noise_full_e.npy"),     ne)
        np.save(os.path.join(run_dir, "noise_full_steps.npy"), ns)
        cfg_runs["noise_full"] = {"final_e": float(ne[-1]), "steps": ntot, "time_s": round(dt,1)}
        print(f"  → B1 final E = {ne[-1]:.5f}  |  {dt/60:.1f}min")

    # ── B2: pruned noise ───────────────────────────────────────────────────────
    if "b2" in METHODS:
        print(f"\n{'='*55}\n  B2 — pruned noise (top-k gates only, noisy)\n{'='*55}")
        for k in TOP_K_LIST:
            mask_k = _mask_for_k(k)
            active_init = init_params.flatten()[mask_k.flatten()]
            clean_k = make_pruned_cost(H_pl, mask_k, 0.0)
            key = f"pruned_{k}_noise"
            print(f"\n  [B2] k={k} ({k}/{N_GATES} = {100*k/N_GATES:.0f}%)")
            ne, ns, ntot, dt = _run_anneal_flat(
                clean_k,
                lambda p, m=mask_k: make_pruned_cost(H_pl, m, p),
                active_init, label=f"B2-k{k}", schedule=sched, lr=LR)
            summary[key] = (ne, ns)
            np.save(os.path.join(run_dir, f"{key}_e.npy"),     ne)
            np.save(os.path.join(run_dir, f"{key}_steps.npy"), ns)
            cfg_runs[key] = {"final_e": float(ne[-1]), "steps": ntot, "time_s": round(dt,1), "k": k}
            print(f"  → B2-k{k} final E = {ne[-1]:.5f}  |  {dt/60:.1f}min")

    # ── B3: selective noise ────────────────────────────────────────────────────
    if "b3" in METHODS:
        print(f"\n{'='*55}\n  B3 — selective noise (all gates, noise on top-k)\n{'='*55}")
        for k in TOP_K_LIST:
            noise_mask = _mask_for_k(k)
            clean_sel  = make_selective_cost(H_pl, noise_mask, 0.0)
            key = f"selective_{k}_noise"
            print(f"\n  [B3] k={k} noisy ({100*k/N_GATES:.0f}%)")
            ne, ns, ntot, dt = _run_anneal_shaped(
                clean_sel,
                lambda p, m=noise_mask: make_selective_cost(H_pl, m, p),
                init_params, sched, LR, label=f"B3-k{k}")
            summary[key] = (ne, ns)
            np.save(os.path.join(run_dir, f"{key}_e.npy"),     ne)
            np.save(os.path.join(run_dir, f"{key}_steps.npy"), ns)
            cfg_runs[key] = {"final_e": float(ne[-1]), "steps": ntot, "time_s": round(dt,1), "k": k}
            print(f"  → B3-k{k} final E = {ne[-1]:.5f}  |  {dt/60:.1f}min")

    # ── B4: inv-selective noise ────────────────────────────────────────────────
    if "b4" in METHODS:
        print(f"\n{'='*55}\n  B4 — inv-selective noise (all gates, noise on bottom-96-k)\n{'='*55}")
        for k in TOP_K_LIST:
            noise_mask = _mask_for_k(k, invert=True)   # noise on non-important
            clean_inv  = make_selective_cost(H_pl, noise_mask, 0.0)
            key = f"inv_sel_{k}_noise"
            n_noisy = int(noise_mask.sum())
            print(f"\n  [B4] k={k} important, noise on {n_noisy} non-important gates")
            ne, ns, ntot, dt = _run_anneal_shaped(
                clean_inv,
                lambda p, m=noise_mask: make_selective_cost(H_pl, m, p),
                init_params, sched, LR, label=f"B4-k{k}")
            summary[key] = (ne, ns)
            np.save(os.path.join(run_dir, f"{key}_e.npy"),     ne)
            np.save(os.path.join(run_dir, f"{key}_steps.npy"), ns)
            cfg_runs[key] = {"final_e": float(ne[-1]), "steps": ntot, "time_s": round(dt,1), "k": k}
            print(f"  → B4-k{k} final E = {ne[-1]:.5f}  |  {dt/60:.1f}min")

    total_time = time.time() - t_global

    # ── CSV summary ────────────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "final_e", "normalized_gap", "steps", "time_s",
                    "k", "frac", "h_id", "seed"])
        span = abs(GROUND) * 2  # spectrum_span ≈ top - ground (symmetric TFIM)
        for key, rec in cfg_runs.items():
            k_val = rec.get("k", N_GATES)
            frac  = k_val / N_GATES
            norm_gap = (rec["final_e"] - GROUND) / span
            w.writerow([key, f"{rec['final_e']:.5f}", f"{norm_gap:.4f}",
                        rec["steps"], rec["time_s"],
                        k_val, f"{frac:.2f}", H_ID, SEED])

    # ── Config JSON ───────────────────────────────────────────────────────────
    cfg = {
        "timestamp": datetime.now().isoformat(),
        "h_id": H_ID, "jzz": jzz, "hx": hx, "hz": hz, "ground": GROUND,
        "seed": SEED, "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
        "ranges": RANGES, "n_gates": N_GATES,
        "top_k_fracs": TOP_K_FRACS, "top_k_list": TOP_K_LIST,
        "lr": LR, "n_avg": args.n_avg, "noise_scale": args.noise_scale,
        "pauli_schedule": sched, "methods": sorted(METHODS),
        "total_time_s": round(total_time, 1),
        "runs": cfg_runs,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'key':>30}  {'final E':>9}  {'norm gap':>9}  {'time':>7}")
    print("-" * 65)
    for key, rec in cfg_runs.items():
        norm_gap = (rec["final_e"] - GROUND) / (abs(GROUND) * 2)
        ok = "✓" if norm_gap < 0.05 else ("△" if norm_gap < 0.2 else "✗")
        print(f"  {key:>30}  {rec['final_e']:>9.5f}  {norm_gap:>9.4f} {ok}  {rec['time_s']:>6.1f}s")
    print("-" * 65)
    print(f"  {'ground energy':>30}  {GROUND:>9.5f}")
    print(f"  total wall time: {total_time/60:.1f} min")
    print("=" * 65)
    print(f"\nResults → {run_dir}")

    tee.close()


if __name__ == "__main__":
    main()
