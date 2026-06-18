"""
qfim_analysis.py
----------------
Compute Quantum Fisher Information Matrix (QFIM) and compare with the
Hessian-based effective dimension already computed in circuit_analysis.py.

QFIM = 4 * g_ij  where g_ij is the Fubini-Study metric tensor.
Computed via qml.adjoint_metric_tensor (exact, no shots).

Outputs saved to results/{latest_ts}/:
    qfim_matrix.npy
    qfim_eigvals.npy
    qfim_vs_hessian.png
"""

import sys
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE  = os.path.dirname(os.path.abspath(__file__))
_CODES = os.path.join(_HERE, "..", "Codes")
sys.path.insert(0, _CODES)

import src.config as _C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED     = 176
N_QUBITS = 4
N_LAYERS = 2
RANGES   = [1, 2]
WIRES    = list(range(N_QUBITS))

# ---------------------------------------------------------------------------
# Find latest results folder  (same as circuit_analysis.py)
# ---------------------------------------------------------------------------

_results_base = os.path.join(_HERE, "results")
_subdirs = [d for d in glob.glob(os.path.join(_results_base, "2*"))
            if os.path.isdir(d)]
if not _subdirs:
    raise RuntimeError(f"No timestamp subdirs found under {_results_base}")
TS_DIR = max(_subdirs, key=os.path.basename)
print(f"Using results dir: {TS_DIR}")

# ---------------------------------------------------------------------------
# Raw init  (same seed as circuit_analysis.py)
# ---------------------------------------------------------------------------

raw_rng        = np.random.default_rng(SEED)
init_params_np = raw_rng.uniform(0.0, 2.0 * np.pi, (N_LAYERS, N_QUBITS, 3))

# ---------------------------------------------------------------------------
# Circuit (same structure as circuit_analysis.py / importance_pruning.py)
# ---------------------------------------------------------------------------

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd")
def circuit(weights):
    qml.StronglyEntanglingLayers(weights, wires=WIRES, ranges=RANGES, imprimitive=qml.CNOT)
    return qml.expval(qml.Hamiltonian([1.0] * N_QUBITS, [qml.PauliZ(i) for i in WIRES]))

# ---------------------------------------------------------------------------
# QFIM via adjoint metric tensor
# QFIM = 4 * metric_tensor  (pure state)
# ---------------------------------------------------------------------------

print("Computing metric tensor (adjoint method)...")
mt_fn = qml.adjoint_metric_tensor(circuit)
params_pnp = pnp.array(init_params_np, requires_grad=True)
metric = np.array(mt_fn(params_pnp), dtype=float)   # shape (24, 24)
qfim   = 4.0 * metric                                # QFIM = 4g

qfim = qfim.reshape(24, 24)
np.save(os.path.join(TS_DIR, "qfim_matrix.npy"), qfim)
print(f"QFIM shape: {qfim.shape}")

qfim_eigvals = np.linalg.eigvalsh(qfim)
np.save(os.path.join(TS_DIR, "qfim_eigvals.npy"), qfim_eigvals)

# ---------------------------------------------------------------------------
# Load Hessian eigenvalues computed by circuit_analysis.py
# ---------------------------------------------------------------------------

# Recompute Hessian at same init (circuit_analysis used SEED=176 too)
@qml.qnode(dev, interface="autograd", diff_method="best")
def _energy(weights):
    qml.StronglyEntanglingLayers(weights, wires=WIRES, ranges=RANGES, imprimitive=qml.CNOT)
    return qml.expval(qml.Hamiltonian([1.0] * N_QUBITS, [qml.PauliZ(i) for i in WIRES]))

def _flat_cost(x):
    return _energy(x.reshape(N_LAYERS, N_QUBITS, 3))

x0    = pnp.array(init_params_np.flatten(), requires_grad=True)
g_fn  = qml.grad(_flat_cost)
H_fn  = qml.jacobian(g_fn)

print("Recomputing Hessian...")
Hmat     = np.array(H_fn(x0), dtype=float)
Hmat     = 0.5 * (Hmat + Hmat.T)
h_eigvals = np.linalg.eigvalsh(Hmat)

# ---------------------------------------------------------------------------
# Effective rank helpers
# ---------------------------------------------------------------------------

def eff_rank_pr(eigvals):
    a = np.abs(eigvals)
    denom = (a ** 2).sum()
    return float(a.sum() ** 2 / denom) if denom > 0 else 0.0

def eff_rank_thresh(eigvals, pct=0.01):
    a = np.abs(eigvals)
    return int((a > pct * a.max()).sum())

qfim_pr     = eff_rank_pr(qfim_eigvals)
qfim_thresh = eff_rank_thresh(qfim_eigvals)
hess_pr     = eff_rank_pr(h_eigvals)
hess_thresh = eff_rank_thresh(h_eigvals)

print(f"\nEffective dimension summary")
print(f"  QFIM  : eff.rank(PR)={qfim_pr:.1f},  eff.rank(1%)={qfim_thresh}")
print(f"  Hessian: eff.rank(PR)={hess_pr:.1f}, eff.rank(1%)={hess_thresh}")

# ---------------------------------------------------------------------------
# Parameter type labels  (RY=index%3==1, RZ otherwise)
# ---------------------------------------------------------------------------

param_labels = [
    f"L{i//(N_QUBITS*3)}Q{(i%(N_QUBITS*3))//3}{'RY' if i%3==1 else 'RZ'}{i%3}"
    for i in range(24)
]
param_types = ["RY" if i % 3 == 1 else "RZ" for i in range(24)]
bar_colors  = ["tab:orange" if t == "RY" else "tab:blue" for t in param_types]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(20, 10))

# --- Row 0: QFIM ---

# Panel 0,0: QFIM eigenvalue spectrum
ax = axes[0, 0]
_, qfim_vecs = np.linalg.eigh(qfim)
ry_frac_q = np.sum(qfim_vecs[np.array(param_types) == "RY", :] ** 2, axis=0)
colors_q  = ["tab:orange" if f > 0.5 else "tab:blue" for f in ry_frac_q]
ax.bar(np.arange(24), qfim_eigvals, color=colors_q, edgecolor="black", lw=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"QFIM eigenspectrum\neff.rank(PR)={qfim_pr:.1f}, eff.rank(1%)={qfim_thresh}")
ax.set_xlabel("Eigenvalue index (ascending)")
ax.set_ylabel("Eigenvalue")
ax.grid(True, alpha=0.3, axis="y")

# Panel 0,1: sorted |QFIM eigenvalues|
ax = axes[0, 1]
sorted_q = np.sort(np.abs(qfim_eigvals))[::-1]
thr_q    = 0.01 * sorted_q[0]
ax.semilogy(np.arange(24), sorted_q + 1e-12, "ko-", ms=5, lw=1.5)
ax.axhline(thr_q, color="red", ls="--", lw=1.2, label=f"1% = {thr_q:.4f}")
ax.set_title("Sorted |QFIM eigenvalues|")
ax.set_xlabel("Rank index")
ax.set_ylabel("|eigenvalue| (log)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 0,2: QFIM diagonal per parameter
ax = axes[0, 2]
qfim_diag = np.diag(qfim)
ax.bar(np.arange(24), qfim_diag, color=bar_colors, edgecolor="black", lw=0.4, alpha=0.85)
ax.set_xticks(np.arange(24))
ax.set_xticklabels(param_labels, rotation=90, fontsize=6)
ax.set_ylabel("QFIM diagonal")
ax.set_title("Per-param QFIM diagonal\n(sensitivity of quantum state to each parameter)")
ax.grid(True, alpha=0.3, axis="y")

# --- Row 1: Hessian (reference) + comparison ---

# Panel 1,0: Hessian eigenvalue spectrum
ax = axes[1, 0]
_, h_vecs = np.linalg.eigh(Hmat)
ry_frac_h = np.sum(h_vecs[np.array(param_types) == "RY", :] ** 2, axis=0)
colors_h  = ["tab:orange" if f > 0.5 else "tab:blue" for f in ry_frac_h]
ax.bar(np.arange(24), h_eigvals, color=colors_h, edgecolor="black", lw=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_title(f"Hessian eigenspectrum\neff.rank(PR)={hess_pr:.1f}, eff.rank(1%)={hess_thresh}")
ax.set_xlabel("Eigenvalue index (ascending)")
ax.set_ylabel("Eigenvalue")
ax.grid(True, alpha=0.3, axis="y")

# Panel 1,1: sorted |Hessian eigenvalues|
ax = axes[1, 1]
sorted_h = np.sort(np.abs(h_eigvals))[::-1]
thr_h    = 0.01 * sorted_h[0]
ax.semilogy(np.arange(24), sorted_h + 1e-12, "ko-", ms=5, lw=1.5)
ax.axhline(thr_h, color="red", ls="--", lw=1.2, label=f"1% = {thr_h:.4f}")
ax.set_title("Sorted |Hessian eigenvalues|")
ax.set_xlabel("Rank index")
ax.set_ylabel("|eigenvalue| (log)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 1,2: QFIM diagonal vs Hessian diagonal per parameter
ax = axes[1, 2]
x = np.arange(24)
w = 0.35
ax.bar(x - w/2, np.abs(qfim_diag),    width=w, color=bar_colors, edgecolor="black",
       lw=0.4, alpha=0.9, label="QFIM diag")
ax.bar(x + w/2, np.abs(np.diag(Hmat)), width=w, color=bar_colors, edgecolor="black",
       lw=0.4, alpha=0.45, hatch="//", label="Hessian diag")
ax.set_xticks(x)
ax.set_xticklabels(param_labels, rotation=90, fontsize=6)
ax.set_ylabel("|diagonal|")
ax.set_title("QFIM vs Hessian diagonal per parameter\n(solid=QFIM, hatched=Hessian)")
ax.grid(True, alpha=0.3, axis="y")

ry_p = mpatches.Patch(color="tab:orange", label="RY params")
rz_p = mpatches.Patch(color="tab:blue",   label="RZ params")
ax.legend(handles=[ry_p, rz_p], fontsize=8, loc="upper right")

fig.suptitle(
    f"QFIM vs Hessian — seed={SEED} | raw init\n"
    f"QFIM eff.rank: PR={qfim_pr:.1f}, 1%={qfim_thresh}   "
    f"Hessian eff.rank: PR={hess_pr:.1f}, 1%={hess_thresh}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
out = os.path.join(TS_DIR, "qfim_vs_hessian.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

# ---------------------------------------------------------------------------
# Console summary: which RZ params appear non-trivial in QFIM?
# ---------------------------------------------------------------------------

print("\nPer-param QFIM diagonal (sorted by value):")
qfim_diag_full = [(param_labels[i], param_types[i], float(qfim_diag[i])) for i in range(24)]
for label, ptype, val in sorted(qfim_diag_full, key=lambda x: -abs(x[2])):
    bar = "#" * int(abs(val) * 20)
    print(f"  {label:>12} ({ptype}) | {val:>8.4f}  {bar}")

# RZ params with non-negligible QFIM diagonal
rz_nonzero = [(l, v) for l, t, v in qfim_diag_full if t == "RZ" and abs(v) > 1e-4]
if rz_nonzero:
    print(f"\nRZ params with non-trivial QFIM diagonal (>1e-4):")
    for l, v in rz_nonzero:
        print(f"  {l}: {v:.6f}")
else:
    print("\nAll RZ params have negligible QFIM diagonal (<1e-4)")
