"""
importance_pruning.py
---------------------
Runs and compares 8 experiments, all starting from the same raw random init
(seed 176, energy ≈ 0.15):

  clean              : full circuit, clean Adam (loaded from saved run)
  noise              : full circuit, adaptive Pauli noise annealing
  pruned_{k}         : top-k params only, clean Adam      (k = 8, 16, 24)
  pruned_{k}_noise   : top-k params only, noise annealing (k = 8, 16, 24)

Incremental saving: each experiment is written to disk immediately after it
finishes. Nothing is held in memory beyond the current run's history array.

Directory layout:
  param_pruning/results/{timestamp}/
    config.json          ← updated after every experiment
    plot.png
    clean.npy
    noise_e.npy, noise_steps.npy
    pruned_{k}.npy
    pruned_{k}_noise_e.npy, pruned_{k}_noise_steps.npy
  param_pruning/results/runs_index.json  ← accumulates across all runs
"""

import sys
import os
import json
import time
from math import ceil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE  = os.path.dirname(os.path.abspath(__file__))
_CODES = os.path.join(_HERE, "..", "Codes")
_S477  = os.path.join(_CODES, "outputs", "seed_training", "adam",
                       "20260324_170741_seed477")

CLEAN_HIST_PATH = os.path.join(_S477, "20260324_170741_seed477_adam_clean_eval_hist.npy")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED             = 477
N_QUBITS         = 4
N_LAYERS         = 2
RANGES           = [1, 2]
WIRES            = list(range(N_QUBITS))
N_STEPS          = 500
LR               = 0.01
TOP_K_LIST       = [8, 16]
WEIGHT_BY_COEFF  = True
GLOBAL_MIN       = -float(N_QUBITS)

# ---------------------------------------------------------------------------
# Import original src  (for exact noise annealing reproduction)
# ---------------------------------------------------------------------------

sys.path.insert(0, _CODES)
import src.config as _C
from src.noise import CLEAN_COST as _CLEAN_COST, make_pauli_cost as _make_pauli_cost
from src.utils import to_trainable as _to_trainable, make_stepper as _make_stepper, \
                      converged_ckpt as _converged_ckpt

# ---------------------------------------------------------------------------
# Importance scores
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_PAULI = {
    "I": _I,
    "X": np.array([[0,1],[1,0]], dtype=complex),
    "Y": np.array([[0,-1j],[1j,0]], dtype=complex),
    "Z": np.array([[1,0],[0,-1]], dtype=complex),
}

def _embed(gate, qubit, n):
    mats = [_I]*n; mats[qubit] = _PAULI[gate]
    r = mats[0]
    for m in mats[1:]: r = np.kron(r, m)
    return r

def _frob(M):
    return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))

H_mat   = sum(_embed("Z", i, N_QUBITS) for i in range(N_QUBITS))
H_TERMS = [(1.0, i, "Z") for i in range(N_QUBITS)]

def compute_importance(weight_by_coeff=True):
    records = []
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            for p, pauli in enumerate(["Z","Y","Z"]):
                G    = _embed(pauli, q, N_QUBITS) / 2.0
                comm = H_mat @ G - G @ H_mat
                imp  = _frob(comm)
                if weight_by_coeff:
                    cw = sum(abs(c) for c,qi,pi in H_TERMS
                             if _frob(_embed(pi,qi,N_QUBITS)@G - G@_embed(pi,qi,N_QUBITS)) > 1e-12)
                    imp *= cw if cw > 0 else 1.0
                records.append({
                    "layer": l, "qubit": q, "param": p,
                    "gate": pauli, "gate_name": "RY" if pauli=="Y" else "RZ",
                    "label": f"L{l}Q{q}{'RY' if pauli=='Y' else 'RZ'}{p}",
                    "importance": float(imp),
                    "flat_idx": l*N_QUBITS*3 + q*3 + p,
                })
    return sorted(records, key=lambda x: x["importance"], reverse=True)

param_info = compute_importance(WEIGHT_BY_COEFF)

print("=" * 55)
print(f"Importance ranking  (H=ΣZ_i, weight_by_coeff={WEIGHT_BY_COEFF})")
print("=" * 55)
for rank, r in enumerate(param_info, 1):
    bar = "#" * int(r["importance"] * 3)
    print(f"  {rank:>2} | {r['label']:>10} | {r['importance']:.4f}  {bar}")
print()

# ---------------------------------------------------------------------------
# PennyLane energy (local, for pruned runs)
# ---------------------------------------------------------------------------

_dev   = qml.device("default.qubit", wires=N_QUBITS)
_H_pl  = qml.Hamiltonian([1.0]*N_QUBITS, [qml.PauliZ(i) for i in range(N_QUBITS)])

@qml.qnode(_dev, interface="autograd", diff_method="best")
def energy(weights):
    qml.StronglyEntanglingLayers(weights, wires=WIRES, ranges=RANGES, imprimitive=qml.CNOT)
    return qml.expval(_H_pl)

# ---------------------------------------------------------------------------
# Active-param mask helper
# ---------------------------------------------------------------------------

def _active_mask(active_flat_idx):
    mask = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=bool)
    for idx in active_flat_idx:
        mask[idx//(N_QUBITS*3), (idx%(N_QUBITS*3))//3, idx%3] = True
    return mask

# ---------------------------------------------------------------------------
# Manual Adam
# ---------------------------------------------------------------------------

def _adam(params, m, v, t, grad, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = b1*m + (1-b1)*grad
    v = b2*v + (1-b2)*grad**2
    return params - lr*(m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps), m, v

# ---------------------------------------------------------------------------
# Gate-removal cost function
# ---------------------------------------------------------------------------

def _make_pruned_cost(active_mask, p=0.0):
    """QNode containing only the active gates (gate removal, not masking).
    active_params: 1-D array of length active_mask.sum(), ordered by (l,q,pi).
    """
    dev  = qml.device("default.mixed" if p > 1e-9 else "default.qubit", wires=N_QUBITS)
    _H   = qml.Hamiltonian([1.0]*N_QUBITS, [qml.PauliZ(i) for i in WIRES])
    _msk = active_mask.copy()

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(active_params):
        idx = 0
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                for pi, gate_name in enumerate(["RZ", "RY", "RZ"]):
                    if _msk[l, q, pi]:
                        if p > 1e-9:
                            qml.PauliError("Y" if gate_name == "RY" else "Z", p, wires=q)
                        if gate_name == "RY":
                            qml.RY(active_params[idx], wires=q)
                        else:
                            qml.RZ(active_params[idx], wires=q)
                        idx += 1
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(_H)

    return circuit

# ---------------------------------------------------------------------------
# Incremental save helpers
# ---------------------------------------------------------------------------

def _save_npy(ts_dir, name, arr):
    np.save(os.path.join(ts_dir, f"{name}.npy"), arr)

def _update_cfg(ts_dir, key, val):
    """Append / update a key in config.json."""
    cfg_path = os.path.join(ts_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg.setdefault("runs", {})[key] = val
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_pruned(init_np, active_mask, n_steps=N_STEPS, lr=LR):
    """Clean Adam on circuit with only active gates (gate removal)."""
    params = np.array(init_np).flatten()[active_mask.flatten()].copy()
    n_act  = len(params)
    cost_fn = _make_pruned_cost(active_mask, p=0.0)
    gfn     = qml.grad(cost_fn)
    m = np.zeros(n_act); v = np.zeros(n_act)
    hist = [float(cost_fn(pnp.array(params, requires_grad=True)))]
    for step in range(n_steps):
        g = np.array(gfn(pnp.array(params, requires_grad=True)))
        params, m, v = _adam(params, m, v, step+1, g, lr)
        hist.append(float(cost_fn(pnp.array(params, requires_grad=True))))
        if (step+1) % 100 == 0:
            print(f"    step {step+1:>4} | E = {hist[-1]:.6f}")
    return np.array(hist)


def run_noise_annealing(init_np, lr=LR):
    """Full noise annealing via original src code (exact reproduction)."""
    win   = max(2, int(ceil(_C.PAULI_WINDOW / _C.PAULI_CHECK_EVERY)))
    params = _to_trainable(init_np)
    all_e, all_steps, total = [], [], 0

    print(f"\n[Noise annealing | schedule={_C.PAULI_SCHEDULE}]")
    for si, noise_p in enumerate(_C.PAULI_SCHEDULE, 1):
        use_clean = noise_p <= _C.CLEAN_THRESHOLD
        cost_fn   = _CLEAN_COST if use_clean else _make_pauli_cost(noise_p)
        step_fn, desc = _make_stepper(
            cost_fn, _C.PAULI_OPTIMIZER, _C.PAULI_MAX_STEPS,
            _C.LR_PAULI_GD, lr,
            _C.PAULI_SPSA_ALPHA, _C.PAULI_SPSA_GAMMA,
            _C.PAULI_SPSA_C, _C.PAULI_SPSA_A, _C.PAULI_SPSA_a,
        )
        stage_ckpt = []; step_i = 0; t0 = time.time()
        label = "clean" if use_clean else f"p={noise_p}"
        print(f"\n  Stage {si}/{len(_C.PAULI_SCHEDULE)} | {label} | {desc}")
        while step_i < _C.PAULI_MAX_STEPS:
            params = step_fn(params); step_i += 1
            if step_i % _C.PAULI_CHECK_EVERY == 0:
                e = float(_CLEAN_COST(params))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total+step_i)
                conv, std, delta = _converged_ckpt(stage_ckpt, win, _C.PAULI_STD_TOL, _C.PAULI_RATE_TOL)
                if step_i >= _C.PAULI_MIN_STEPS and conv:
                    print(f"    converged step {step_i} | E={e:.6f} std={std:.5f} Δ={delta:.5f}")
                    break
        total += step_i
        print(f"    done: steps={step_i} | E={all_e[-1]:.6f} | {time.time()-t0:.1f}s")

    print(f"\n  total_steps={total} | final_E={all_e[-1]:.6f}")
    return np.array(all_e), np.array(all_steps), total


def run_pruned_noise(init_np, active_mask, lr=LR):
    """Noise annealing on circuit with only active gates (gate removal)."""
    win    = max(2, int(ceil(_C.PAULI_WINDOW / _C.PAULI_CHECK_EVERY)))
    params = np.array(init_np).flatten()[active_mask.flatten()].copy()
    n_act  = len(params)
    _clean = _make_pruned_cost(active_mask, p=0.0)
    all_e, all_steps, total = [], [], 0

    print(f"\n[Pruned noise annealing | {n_act} active gates]")
    for si, noise_p in enumerate(_C.PAULI_SCHEDULE, 1):
        use_clean = noise_p <= _C.CLEAN_THRESHOLD
        cost_fn   = _clean if use_clean else _make_pruned_cost(active_mask, noise_p)
        gfn       = qml.grad(cost_fn)
        m = np.zeros(n_act); v = np.zeros(n_act)
        stage_ckpt = []; step_i = 0; t0 = time.time()
        label = "clean" if use_clean else f"p={noise_p}"
        print(f"\n  Stage {si}/{len(_C.PAULI_SCHEDULE)} | {label}")
        while step_i < _C.PAULI_MAX_STEPS:
            g = np.array(gfn(pnp.array(params, requires_grad=True)))
            params, m, v = _adam(params, m, v, step_i+1, g, lr)
            step_i += 1
            if step_i % _C.PAULI_CHECK_EVERY == 0:
                e = float(_clean(pnp.array(params, requires_grad=True)))
                stage_ckpt.append(e); all_e.append(e); all_steps.append(total+step_i)
                conv, std, delta = _converged_ckpt(stage_ckpt, win, _C.PAULI_STD_TOL, _C.PAULI_RATE_TOL)
                if step_i >= _C.PAULI_MIN_STEPS and conv:
                    print(f"    converged step {step_i} | E={e:.6f} std={std:.5f} Δ={delta:.5f}")
                    break
        total += step_i
        print(f"    done: steps={step_i} | E={all_e[-1]:.6f} | {time.time()-t0:.1f}s")

    print(f"\n  total_steps={total} | final_E={all_e[-1]:.6f}")
    return np.array(all_e), np.array(all_steps), total

# ---------------------------------------------------------------------------
# Setup: raw init + timestamp directory
# ---------------------------------------------------------------------------

raw_rng        = np.random.default_rng(SEED)
init_params_np = raw_rng.uniform(0.0, 2.0*np.pi, (N_LAYERS, N_QUBITS, 3))
clean_hist     = np.load(CLEAN_HIST_PATH)

init_energy = float(energy(pnp.array(init_params_np, requires_grad=True)))
assert abs(init_energy - float(clean_hist[0])) < 1e-5, \
    f"Init energy mismatch: {init_energy:.6f} vs clean_hist[0]={clean_hist[0]:.6f}"

print(f"Seed {SEED} raw init energy  : {init_energy:.6f}  ✓ matches clean_hist[0]")
print(f"Clean full final energy       : {clean_hist[-1]:.6f}")
print(f"Global minimum                : {GLOBAL_MIN:.6f}\n")

ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
ts_dir = os.path.join(_HERE, "results", ts)
os.makedirs(ts_dir, exist_ok=True)

# Write base config immediately
_base_cfg = {
    "timestamp": ts, "seed": SEED, "init_source": "raw_random_init_from_seed",
    "n_qubits": N_QUBITS, "n_layers": N_LAYERS, "ranges": RANGES,
    "n_steps": N_STEPS, "lr": LR, "top_k_list": TOP_K_LIST,
    "weight_by_coeff": WEIGHT_BY_COEFF, "pauli_schedule": list(_C.PAULI_SCHEDULE),
    "init_energy": init_energy, "global_min": GLOBAL_MIN,
    "clean_full_final": float(clean_hist[-1]),
    "runs": {},
}
with open(os.path.join(ts_dir, "config.json"), "w") as f:
    json.dump(_base_cfg, f, indent=2)

t_global = time.time()

# ---------------------------------------------------------------------------
# Experiment 1: clean  (load, save reference copy)
# ---------------------------------------------------------------------------

_save_npy(ts_dir, "clean", clean_hist)
_update_cfg(ts_dir, "clean", {
    "source": "loaded", "final_energy": float(clean_hist[-1]),
    "n_steps": len(clean_hist)-1,
})
print(f"[clean] loaded → {ts_dir}/clean.npy")
summary = {}   # key → (final_E, time_s, n_steps)
summary["clean"] = (float(clean_hist[-1]), None, len(clean_hist)-1)

# ---------------------------------------------------------------------------
# Experiment 2-4: pruned top-k  (clean Adam)
# ---------------------------------------------------------------------------

for k in TOP_K_LIST:
    top_recs  = param_info[:k]
    labels    = [r["label"] for r in top_recs]
    mask_k    = _active_mask([r["flat_idx"] for r in top_recs])
    print(f"\n{'='*50}\n[pruned_{k}] top-{k} gates: {labels[:8]}{'...' if k>8 else ''}")

    t0   = time.time()
    hist = run_pruned(init_params_np, mask_k)
    elapsed = time.time() - t0

    _save_npy(ts_dir, f"pruned_{k}", hist)
    _update_cfg(ts_dir, f"pruned_{k}", {
        "selected_params": labels, "final_energy": float(hist[-1]),
        "n_steps": len(hist)-1, "time_s": round(elapsed, 1),
    })
    summary[f"pruned_{k}"] = (float(hist[-1]), elapsed, len(hist)-1)
    print(f"  → final E={hist[-1]:.6f} | {elapsed:.1f}s | saved pruned_{k}.npy")
    del hist

# ---------------------------------------------------------------------------
# Experiment 5: noise annealing  (full, using src code)
# ---------------------------------------------------------------------------

print(f"\n{'='*50}\n[noise]")
t0 = time.time()
noise_e, noise_steps, noise_total = run_noise_annealing(init_params_np)
elapsed = time.time() - t0

_save_npy(ts_dir, "noise_e",     noise_e)
_save_npy(ts_dir, "noise_steps", noise_steps)
_update_cfg(ts_dir, "noise", {
    "source": "re-run via src.training", "final_energy": float(noise_e[-1]),
    "total_steps": noise_total, "time_s": round(elapsed, 1),
})
summary["noise"] = (float(noise_e[-1]), elapsed, noise_total)
print(f"  → final E={noise_e[-1]:.6f} | {elapsed:.1f}s | saved noise_e/steps.npy")
del noise_e, noise_steps

# ---------------------------------------------------------------------------
# Experiments 6-8: pruned top-k + noise annealing
# ---------------------------------------------------------------------------

for k in TOP_K_LIST:
    labels  = [r["label"] for r in param_info[:k]]
    mask_k  = _active_mask([r["flat_idx"] for r in param_info[:k]])
    print(f"\n{'='*50}\n[pruned_{k}_noise]")

    t0 = time.time()
    p_e, p_steps, p_total = run_pruned_noise(init_params_np, mask_k)
    elapsed = time.time() - t0

    _save_npy(ts_dir, f"pruned_{k}_noise_e",     p_e)
    _save_npy(ts_dir, f"pruned_{k}_noise_steps",  p_steps)
    _update_cfg(ts_dir, f"pruned_{k}_noise", {
        "selected_params": labels, "final_energy": float(p_e[-1]),
        "total_steps": p_total, "time_s": round(elapsed, 1),
    })
    summary[f"pruned_{k}_noise"] = (float(p_e[-1]), elapsed, p_total)
    print(f"  → final E={p_e[-1]:.6f} | {elapsed:.1f}s | saved pruned_{k}_noise_*.npy")
    del p_e, p_steps

total_time = time.time() - t_global

# ---------------------------------------------------------------------------
# Plot  (load from saved files)
# ---------------------------------------------------------------------------

def _load(name):
    return np.load(os.path.join(ts_dir, f"{name}.npy"))

_COLORS = {8: "tab:blue", 16: "tab:orange", 24: "tab:green"}

fig, axes = plt.subplots(1, 3, figsize=(21, 5))

# --- Panel 1: Clean training comparison ---
ax = axes[0]
_ch = _load("clean")
ax.plot(np.arange(len(_ch)), _ch, lw=2.2, color="black",
        label="Clean full (24 params)")
for k in TOP_K_LIST:
    _h = _load(f"pruned_{k}")
    ax.plot(np.arange(len(_h)), _h, lw=1.6, color=_COLORS[k],
            ls="--", label=f"Pruned top-{k}")
ax.axhline(GLOBAL_MIN, color="red", ls=":", lw=1.2, label="Global min")
ax.set_xlabel("Step"); ax.set_ylabel("Energy"); ax.legend(fontsize=8)
ax.set_title("Clean Adam  (full vs pruned)")
ax.grid(True, alpha=0.3)

# --- Panel 2: Noise annealing comparison ---
ax = axes[1]
_ne = _load("noise_e"); _ns = _load("noise_steps")
ax.plot(_ns, _ne, lw=2.2, color="crimson",
        marker="o", ms=2, markevery=5,
        label=f"Noise full ({int(_ns[-1])} steps)")
for k in TOP_K_LIST:
    _pe = _load(f"pruned_{k}_noise_e"); _ps = _load(f"pruned_{k}_noise_steps")
    ax.plot(_ps, _pe, lw=1.6, color=_COLORS[k],
            ls="--", marker="s", ms=2, markevery=5,
            label=f"Pruned top-{k} noise ({int(_ps[-1])} steps)")
ax.axhline(GLOBAL_MIN, color="red", ls=":", lw=1.2, label="Global min")
ax.set_xlabel("Cumulative step"); ax.set_ylabel("Energy (clean eval)")
ax.legend(fontsize=8)
ax.set_title("Noise annealing  (full vs pruned)")
ax.grid(True, alpha=0.3)

# --- Panel 3: Importance bar chart ---
ax = axes[2]
bar_c = ["tab:orange" if r["gate"]=="Y" else "tab:gray" for r in param_info]
x = np.arange(len(param_info))
ax.bar(x, [r["importance"] for r in param_info],
       color=bar_c, edgecolor="black", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([r["label"] for r in param_info], rotation=90, fontsize=7)
ax.set_ylabel(r"$\|[H,G_k]\|_F$")
ax.set_title("Param importance  (orange=RY, gray=RZ)")
for k, (col, ls) in zip(TOP_K_LIST, [("tab:blue","-"),("tab:orange","--"),("tab:green",":")]):
    ax.axvline(k-0.5, color=col, ls=ls, lw=1.2, label=f"Top-{k}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

plt.suptitle(f"seed={SEED}  |  ts={ts}  |  total={total_time:.0f}s", fontsize=10)
plt.tight_layout()
plot_path = os.path.join(ts_dir, "plot.png")
plt.savefig(plot_path, dpi=150); plt.show()
print(f"\nPlot saved → {plot_path}")

# update plot path in config
cfg_path = os.path.join(ts_dir, "config.json")
cfg = json.load(open(cfg_path))
cfg["total_time_s"] = round(total_time, 1)
cfg["plot_file"] = "plot.png"
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)

# ---------------------------------------------------------------------------
# runs_index.json  (accumulate across all runs)
# ---------------------------------------------------------------------------

idx_path   = os.path.join(_HERE, "results", "runs_index.json")
runs_index = json.load(open(idx_path)) if os.path.exists(idx_path) else []
runs_index.append({
    "timestamp":   ts,
    "run_dir":     ts,
    "seed":        SEED,
    "n_steps":     N_STEPS,
    "lr":          LR,
    "top_k_list":  TOP_K_LIST,
    "init_energy": init_energy,
    "finals":      {k: round(v[0], 6) for k, v in summary.items()},
    "times_s":     {k: (round(v[1],1) if v[1] else None) for k,v in summary.items()},
    "total_time_s": round(total_time, 1),
})
with open(idx_path, "w") as f:
    json.dump(runs_index, f, indent=2)
print(f"Index updated → {idx_path}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print(f"{'experiment':>25} | {'final E':>10} | {'steps':>7} | {'time':>7}")
print("-" * 65)
order = ["clean", "noise"] + \
        [f"pruned_{k}" for k in TOP_K_LIST] + \
        [f"pruned_{k}_noise" for k in TOP_K_LIST]
for key in order:
    fe, t, ns = summary[key]
    t_str  = f"{t:.1f}s"  if t  else "—"
    ns_str = str(ns) if ns else "—"
    print(f"  {key:>23} | {fe:>10.6f} | {ns_str:>7} | {t_str:>7}")
print("-" * 65)
print(f"  {'global min':>23} | {GLOBAL_MIN:>10.6f} | {'—':>7} | {'—':>7}")
print(f"  {'TOTAL':>23}   {'':>10}   {'':>7}   {total_time:.1f}s")
print("=" * 65)
