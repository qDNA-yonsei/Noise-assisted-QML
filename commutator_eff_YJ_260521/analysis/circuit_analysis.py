"""
circuit_analysis.py
-------------------
1. Draw 8 circuit schematics (clean/noisy × full/pruned-k) as individual PNGs.
   Active gates: orange (RY), blue (RZ); frozen gates: light gray.
   Noisy circuits include PauliError symbols.

2. Compute Hessian eigenspectrum at raw_init → effective dimension analysis.
   Uses qml.jacobian(qml.grad(flat_cost)) — same as src/utils.py clean_hessian_info.

3. Saves all PNGs to results/{latest_ts}/  (most-recently-modified subfolder).
"""

import sys
import os
import glob
import json

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
from src.noise import CLEAN_COST as _CLEAN_COST

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED      = 176
N_QUBITS  = 4
N_LAYERS  = 2
RANGES    = [1, 2]
WIRES     = list(range(N_QUBITS))

# ---------------------------------------------------------------------------
# Find latest results folder
# ---------------------------------------------------------------------------

_results_base = os.path.join(_HERE, "results")
_subdirs = [d for d in glob.glob(os.path.join(_results_base, "2*"))
            if os.path.isdir(d)]
if not _subdirs:
    raise RuntimeError(f"No timestamp subdirs found under {_results_base}")
TS_DIR = max(_subdirs, key=os.path.basename)
print(f"Using results dir: {TS_DIR}")

# Load config to recover param info
with open(os.path.join(TS_DIR, "config.json")) as f:
    _cfg = json.load(f)

# ---------------------------------------------------------------------------
# Importance ranking (reproduce same order as importance_pruning.py)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_PAULI_MAT = {
    "I": _I,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _embed(gate, qubit, n):
    mats = [_I] * n
    mats[qubit] = _PAULI_MAT[gate]
    r = mats[0]
    for m in mats[1:]:
        r = np.kron(r, m)
    return r


def _frob(M):
    return float(np.sqrt(np.real(np.trace(M.conj().T @ M))))


H_mat   = sum(_embed("Z", i, N_QUBITS) for i in range(N_QUBITS))
H_TERMS = [(1.0, i, "Z") for i in range(N_QUBITS)]


def compute_importance():
    records = []
    for l in range(N_LAYERS):
        for q in range(N_QUBITS):
            for p, pauli in enumerate(["Z", "Y", "Z"]):
                G    = _embed(pauli, q, N_QUBITS) / 2.0
                comm = H_mat @ G - G @ H_mat
                imp  = _frob(comm)
                cw   = sum(abs(c) for c, qi, pi in H_TERMS
                           if _frob(_embed(pi, qi, N_QUBITS) @ G
                                    - G @ _embed(pi, qi, N_QUBITS)) > 1e-12)
                imp *= cw if cw > 0 else 1.0
                records.append({
                    "layer": l, "qubit": q, "param": p,
                    "gate": pauli,
                    "gate_name": "RY" if pauli == "Y" else "RZ",
                    "label": f"L{l}Q{q}{'RY' if pauli=='Y' else 'RZ'}{p}",
                    "importance": float(imp),
                    "flat_idx": l * N_QUBITS * 3 + q * 3 + p,
                })
    return sorted(records, key=lambda x: x["importance"], reverse=True)


param_info = compute_importance()

# ---------------------------------------------------------------------------
# Active masks per experiment
# ---------------------------------------------------------------------------

TOP_K_LIST = _cfg.get("top_k_list", [8, 16, 24])


def _active_mask(top_k):
    """bool mask shape (N_LAYERS, N_QUBITS, 3) for top-k params."""
    mask = np.zeros((N_LAYERS, N_QUBITS, 3), dtype=bool)
    for rec in param_info[:top_k]:
        l, q, p = rec["layer"], rec["qubit"], rec["param"]
        mask[l, q, p] = True
    return mask


FULL_MASK = np.ones((N_LAYERS, N_QUBITS, 3), dtype=bool)
MASKS = {k: _active_mask(k) for k in TOP_K_LIST}

# ---------------------------------------------------------------------------
# Raw init
# ---------------------------------------------------------------------------

raw_rng        = np.random.default_rng(SEED)
init_params_np = raw_rng.uniform(0.0, 2.0 * np.pi, (N_LAYERS, N_QUBITS, 3))

# ---------------------------------------------------------------------------
# Circuit schematic drawing
# ---------------------------------------------------------------------------

# Layout constants
_COL  = 1.0   # column width
_ROW  = 0.85  # row height (between qubit lines)
_GW   = 0.72  # gate box width
_GH   = 0.45  # gate box height
_NW   = 0.50  # noise symbol width

# Gate colors
_C_RY   = ("#FF8C00", "#CC6600")    # active RY: orange fill / dark-orange edge
_C_RZ   = ("#4682B4", "#285680")    # active RZ: steel-blue fill / dark-blue edge
_C_FRZ  = ("#D9D9D9", "#999999")    # frozen: light-gray fill / gray edge
_C_NZ   = ("#FFB6C1", "#E07090")    # Z-noise: pink
_C_NY   = ("#B0EEB0", "#40A040")    # Y-noise: green


def _qy(q):
    """y-coordinate of qubit q (q0 at top)."""
    return (N_QUBITS - 1 - q) * _ROW


def _gate_box(ax, x, y, label, fc, ec, fontsize=7.5, italic=False):
    rect = mpatches.FancyBboxPatch(
        (x - _GW / 2, y - _GH / 2), _GW, _GH,
        boxstyle="round,pad=0.04",
        facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3, clip_on=False,
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, zorder=4,
            fontstyle="italic" if italic else "normal",
            color="#555555" if italic else "black")


def _noise_sym(ax, x, y, kind, p):
    """Draw a small noise symbol (⚡ approximation = zigzag box)."""
    color, label = (_C_NZ, f"E_Z") if kind == "Z" else (_C_NY, f"E_Y")
    rect = mpatches.FancyBboxPatch(
        (x - _NW / 2, y - _GH / 2), _NW, _GH,
        boxstyle="round,pad=0.03",
        facecolor=color[0], edgecolor=color[1], linewidth=1.0, zorder=3,
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=6.0, zorder=4,
            color="#333333")


def _cnot(ax, x, q_ctrl, q_tgt):
    """Draw a CNOT gate between q_ctrl and q_tgt."""
    yc = _qy(q_ctrl)
    yt = _qy(q_tgt)
    r  = _GH * 0.45
    ax.plot([x, x], [yc, yt], "k-", lw=0.9, zorder=2)
    ax.plot(x, yc, "ko", ms=7, zorder=4)
    circ = plt.Circle((x, yt), r, fill=True, facecolor="white",
                       edgecolor="black", linewidth=1.2, zorder=3)
    ax.add_patch(circ)
    ax.plot([x - r, x + r], [yt, yt], "k-", lw=0.8, zorder=4)
    ax.plot([x, x], [yt - r, yt + r], "k-", lw=0.8, zorder=4)


def draw_circuit(ax, active_mask, noisy=False, p_noise=0.4, title=""):
    """
    Draw a 2-layer, 4-qubit SEL circuit on ax.
    active_mask : (N_LAYERS, N_QUBITS, 3) bool — which params are being trained.
    noisy        : if True, insert PauliError symbols before RZ and RY.
    """
    # Columns per layer:
    #   clean : RZ(0) | RY(1) | RZ(2) | CNOT(×NQ)
    #   noisy : E_Z | RZ(0) | E_Y | RY(1) | RZ(2) | CNOT(×NQ)
    # CNOT block: we draw NQ CNOTs stacked in 1 visual column but they are
    # in practice sequential — that is fine for a schematic.
    gate_cols = 6 if noisy else 3   # rotation+noise columns per layer
    cnot_col  = gate_cols           # CNOT column offset
    cols_per_layer = gate_cols + 1  # +1 for CNOT block

    total_cols = N_LAYERS * cols_per_layer
    x_max = (total_cols + 0.5) * _COL
    y_max = (N_QUBITS - 1) * _ROW

    # Qubit wire labels + lines
    for q in range(N_QUBITS):
        y = _qy(q)
        ax.hlines(y, 0, x_max, color="black", lw=0.7, zorder=1)
        ax.text(-0.25, y, f"q{q}", ha="right", va="center", fontsize=8,
                fontfamily="monospace")

    for l in range(N_LAYERS):
        x0 = l * cols_per_layer * _COL + _COL * 0.5  # first gate column x

        if noisy:
            # E_Z + RZ(0)  —  only for qubits where RZ(0) is active
            x_ez = x0; x_rz0 = x0 + _COL
            for q in range(N_QUBITS):
                if active_mask[l, q, 0]:
                    _noise_sym(ax, x_ez,  _qy(q), "Z", p_noise)
                    _gate_box(ax,  x_rz0, _qy(q), "RZ", *_C_RZ)
            # E_Y + RY(1)  —  only for active qubits
            x_ey = x0 + 2 * _COL; x_ry = x0 + 3 * _COL
            for q in range(N_QUBITS):
                if active_mask[l, q, 1]:
                    _noise_sym(ax, x_ey, _qy(q), "Y", p_noise)
                    _gate_box(ax,  x_ry, _qy(q), "RY", *_C_RY)
            # E_Z + RZ(2)  —  only for active qubits
            x_ez2 = x0 + 4 * _COL; x_rz2 = x0 + 5 * _COL
            for q in range(N_QUBITS):
                if active_mask[l, q, 2]:
                    _noise_sym(ax, x_ez2, _qy(q), "Z", p_noise)
                    _gate_box(ax, x_rz2, _qy(q), "RZ", *_C_RZ)
        else:
            # RZ(0)
            x = x0
            for q in range(N_QUBITS):
                if active_mask[l, q, 0]:
                    _gate_box(ax, x, _qy(q), "RZ", *_C_RZ)
            # RY(1)
            x = x0 + _COL
            for q in range(N_QUBITS):
                if active_mask[l, q, 1]:
                    _gate_box(ax, x, _qy(q), "RY", *_C_RY)
            # RZ(2)
            x = x0 + 2 * _COL
            for q in range(N_QUBITS):
                if active_mask[l, q, 2]:
                    _gate_box(ax, x, _qy(q), "RZ", *_C_RZ)

        # CNOT block
        x = x0 + cnot_col * _COL
        r = RANGES[l]
        # Draw them spread slightly to avoid overlap
        cnot_offset = [-0.15 * _COL * (N_QUBITS // 2 - 0.5) + 0.15 * _COL * i
                       for i in range(N_QUBITS)]
        for i, q in enumerate(range(N_QUBITS)):
            tgt = (q + r) % N_QUBITS
            _cnot(ax, x + cnot_offset[i], q, tgt)

        # Layer separator
        if l < N_LAYERS - 1:
            x_sep = x0 + (cnot_col + 0.6) * _COL
            ax.axvline(x_sep, ymin=0.05, ymax=0.95,
                       color="slategray", lw=0.7, ls=":", zorder=0, alpha=0.6)
            ax.text(x_sep, y_max + 0.35, f"L{l}",
                    ha="center", va="bottom", fontsize=7, color="slategray")

    # Last layer label
    x_lbl = (N_LAYERS - 1) * cols_per_layer * _COL + _COL * 0.5 + (cnot_col) * _COL
    ax.text(x_lbl, y_max + 0.35, f"L{N_LAYERS-1}",
            ha="center", va="bottom", fontsize=7, color="slategray")

    ax.set_xlim(-0.6, x_max + 0.3)
    ax.set_ylim(-0.6, y_max + 0.65)
    ax.set_title(title, fontsize=9, pad=6, fontweight="bold")
    ax.axis("off")


def _legend_patches(active_mask, noisy=False):
    patches = []
    if np.any(active_mask[:, :, 1]):
        patches.append(mpatches.Patch(fc=_C_RY[0], ec=_C_RY[1], lw=1.2,
                                       label=f"RY ({int(active_mask[:,:,1].sum())})"))
    n_rz = int(active_mask[:, :, 0].sum()) + int(active_mask[:, :, 2].sum())
    if n_rz > 0:
        patches.append(mpatches.Patch(fc=_C_RZ[0], ec=_C_RZ[1], lw=1.2,
                                       label=f"RZ ({n_rz})"))
    if noisy:
        n_ry = int(active_mask[:, :, 1].sum())
        n_rz_act = n_rz
        if n_ry > 0:
            patches.append(mpatches.Patch(fc=_C_NY[0], ec=_C_NY[1], lw=1.0, ls="--",
                                           label=f"E_Y ({n_ry})"))
        if n_rz_act > 0:
            patches.append(mpatches.Patch(fc=_C_NZ[0], ec=_C_NZ[1], lw=1.0, ls="--",
                                           label=f"E_Z ({n_rz_act})"))
    return patches


# Circuit configs: (name, active_mask, noisy, title)
_p_repr = f"p∈{[round(x,3) for x in _C.PAULI_SCHEDULE]}"
CIRCUITS = [
    ("clean",
     FULL_MASK,
     False,
     f"Clean full — 24 active params (all RZ+RY)\nseed={SEED}, clean Adam"),
    ("pruned_8",
     MASKS[8],
     False,
     f"Pruned top-8 — RY only (8 gates)\nseed={SEED}, clean Adam"),
    ("pruned_16",
     MASKS[16],
     False,
     f"Pruned top-16 — RY + L0-RZ (16 gates)\nseed={SEED}, clean Adam"),
    ("noise",
     FULL_MASK,
     True,
     f"Noise annealing full — 24 active\nseed={SEED}, adaptive Pauli {_p_repr}"),
    ("pruned_8_noise",
     MASKS[8],
     True,
     f"Noise annealing pruned-8 — RY only (8 gates)\nseed={SEED}"),
    ("pruned_16_noise",
     MASKS[16],
     True,
     f"Noise annealing pruned-16 — RY + L0-RZ (16 gates)\nseed={SEED}"),
]

print("\n=== Drawing circuit schematics ===")
for name, amask, noisy, title in CIRCUITS:
    fig, ax = plt.subplots(figsize=(11 if noisy else 8, 3.5))
    draw_circuit(ax, amask, noisy=noisy, p_noise=0.4, title=title)
    handles = _legend_patches(amask, noisy=noisy)
    ax.legend(handles=handles, loc="lower right",
              fontsize=8, framealpha=0.9, edgecolor="gray",
              bbox_to_anchor=(1.0, -0.05))
    plt.tight_layout(pad=0.8)
    out = os.path.join(TS_DIR, f"circuit_{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: circuit_{name}.png")

# ---------------------------------------------------------------------------
# Combined 8-circuit overview
# ---------------------------------------------------------------------------

print("\n=== Drawing combined 6-circuit overview ===")
fig, axes = plt.subplots(2, 3, figsize=(30, 8))
for ax, (name, amask, noisy, title) in zip(axes.ravel(), CIRCUITS):
    draw_circuit(ax, amask, noisy=noisy, p_noise=0.4,
                 title=title.split("\n")[0])   # short title for combined
    handles = _legend_patches(amask, noisy=noisy)
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              framealpha=0.85, edgecolor="gray")

fig.suptitle(f"6 Circuit Variants — seed={SEED} | results: {os.path.basename(TS_DIR)}",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout(pad=1.2)
out_combined = os.path.join(TS_DIR, "circuits_overview.png")
plt.savefig(out_combined, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: circuits_overview.png")

# ---------------------------------------------------------------------------
# Hessian eigenspectrum (effective dimension)
# ---------------------------------------------------------------------------

print("\n=== Computing Hessian eigenspectrum at raw_init ===")

_dev_clean = qml.device("default.qubit", wires=N_QUBITS)
_H_pl = qml.Hamiltonian([1.0] * N_QUBITS, [qml.PauliZ(i) for i in WIRES])


@qml.qnode(_dev_clean, interface="autograd", diff_method="best")
def _energy(weights):
    qml.StronglyEntanglingLayers(weights, wires=WIRES,
                                 ranges=RANGES, imprimitive=qml.CNOT)
    return qml.expval(_H_pl)


def _flat_cost(x):
    return _energy(x.reshape(N_LAYERS, N_QUBITS, 3))


x0    = pnp.array(init_params_np.flatten(), requires_grad=True)
g_fn  = qml.grad(_flat_cost)
H_fn  = qml.jacobian(g_fn)

print("  Computing gradient ...")
grad0 = np.array(g_fn(x0), dtype=float)
print(f"  |∇E| = {np.linalg.norm(grad0):.4f}")

print("  Computing Hessian (24×24) ...")
Hmat  = np.array(H_fn(x0), dtype=float)
Hmat  = 0.5 * (Hmat + Hmat.T)
eigvals = np.linalg.eigvalsh(Hmat)

# Label each eigenvalue by its dominant parameter (RY=1 vs RZ=0,2) via eigenvectors
_, eigvecs = np.linalg.eigh(Hmat)

# Classify params by gate type (index in flattened 24-dim: 0,2 mod 3 → RZ; 1 mod 3 → RY)
param_types = np.array(["RY" if (i % 3 == 1) else "RZ" for i in range(24)])
# For each eigenvector, compute participation from RY vs RZ params
eigvec_ry_frac = np.sum(eigvecs[param_types == "RY", :] ** 2, axis=0)  # shape (24,)

# Effective rank via participation ratio (flat spectrum = max rank)
abs_vals = np.abs(eigvals)
eff_rank_pr = abs_vals.sum() ** 2 / (abs_vals ** 2).sum() if (abs_vals ** 2).sum() > 0 else 0.0

# Count eigenvalues above 1% of max
threshold = 0.01 * abs_vals.max()
eff_rank_thresh = int((abs_vals > threshold).sum())

print(f"  Eigenvalue range : [{eigvals[0]:.4f}, {eigvals[-1]:.4f}]")
print(f"  Effective rank (participation ratio) : {eff_rank_pr:.2f}")
print(f"  Effective rank (>1% of max)          : {eff_rank_thresh}")

# ---------------------------------------------------------------------------
# Hessian plots
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Eigenvalue spectrum
ax = axes[0]
colors_ev = ["tab:orange" if (1 - f) < f else "tab:blue"
             for f in eigvec_ry_frac]   # orange if majority-RY, blue if majority-RZ
ax.bar(np.arange(24), eigvals, color=colors_ev, edgecolor="black", lw=0.4)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Eigenvalue index (ascending)")
ax.set_ylabel("Eigenvalue")
ax.set_title(f"Hessian eigenspectrum at raw init\n"
             f"eff.rank(PR)={eff_rank_pr:.1f}, eff.rank(1%)={eff_rank_thresh}")
ax.grid(True, alpha=0.3, axis="y")
legend_patches = [
    mpatches.Patch(color="tab:orange", label="RY-dominant eigenvector"),
    mpatches.Patch(color="tab:blue",   label="RZ-dominant eigenvector"),
]
ax.legend(handles=legend_patches, fontsize=8)

# Panel 2: |eigenvalue| log scale
ax = axes[1]
sorted_abs = np.sort(abs_vals)[::-1]
ax.semilogy(np.arange(24), sorted_abs + 1e-10, "ko-", ms=5, lw=1.5)
ax.axhline(threshold, color="red", ls="--", lw=1.2, label=f"1% threshold = {threshold:.4f}")
ax.set_xlabel("Rank index")
ax.set_ylabel("|eigenvalue| (log scale)")
ax.set_title("Sorted |eigenvalues| — effective dimension")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

# Panel 3: Per-parameter gradient and Hessian diagonal
ax = axes[2]
hess_diag = np.diag(Hmat)
x_idx     = np.arange(24)
param_labels = [f"L{i//(N_QUBITS*3)}Q{(i%(N_QUBITS*3))//3}{'RY' if i%3==1 else 'RZ'}{i%3}"
                for i in range(24)]
bar_c = ["tab:orange" if pt == "RY" else "tab:blue" for pt in param_types]
ax.bar(x_idx, hess_diag, color=bar_c, edgecolor="black", lw=0.4, alpha=0.8, label="Hess diagonal")
ax2 = ax.twinx()
ax2.plot(x_idx, np.abs(grad0), "k^--", ms=5, lw=1.2, label="|gradient|")
ax.set_xticks(x_idx)
ax.set_xticklabels(param_labels, rotation=90, fontsize=6)
ax.set_ylabel("Hessian diagonal", color="tab:blue")
ax2.set_ylabel("|gradient|", color="black")
ax.set_title("Per-param Hessian diagonal + |gradient|")
ax.grid(True, alpha=0.3, axis="y")
lines1, lab1 = ax.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right")

# Importance score overlay (right y-axis on panel 3 is busy; put importance in title)
imp_scores = np.array([param_info[i]["importance"] if param_info[i] else 0.0
                       for i in range(24)])
# Sort param_info by flat_idx to match bar order
imp_by_idx = {r["flat_idx"]: r["importance"] for r in param_info}
imp_sorted  = np.array([imp_by_idx.get(i, 0.0) for i in range(24)])

fig.suptitle(
    f"Effective Dimension Analysis — seed={SEED} | raw init (E={_cfg['init_energy']:.4f})\n"
    f"Hessian at init_params. Orange=RY params (||[H,G]||_F={imp_by_idx.get(1,0):.2f}), "
    f"Blue=RZ params (||[H,G]||_F=0.00)",
    fontsize=10, y=1.02
)

plt.tight_layout()
out_hess = os.path.join(TS_DIR, "effective_dimension.png")
plt.savefig(out_hess, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: effective_dimension.png")

# ---------------------------------------------------------------------------
# Importance vs Hessian comparison panel
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
x_idx = np.arange(24)
bar_c = ["tab:orange" if pt == "RY" else "tab:blue" for pt in param_types]
ax.bar(x_idx, imp_sorted, color=bar_c, edgecolor="black", lw=0.5)
ax.set_xticks(x_idx)
ax.set_xticklabels(param_labels, rotation=90, fontsize=6)
ax.set_ylabel(r"$\|[H, G_k]\|_F$  (importance score)")
ax.set_title("Parameter importance scores (by flat index)")
ax.grid(True, alpha=0.3, axis="y")
for k, ls in zip(TOP_K_LIST, ["-", "--", ":"]):
    # mark top-k boundary by index after sorting
    top_flat_idxs = [r["flat_idx"] for r in param_info[:k]]
    # shade active params
    for fi in top_flat_idxs:
        ax.axvspan(fi - 0.5, fi + 0.5, alpha=0.08, color="tab:green" if k == 8
                   else ("tab:orange" if k == 16 else "gray"))
patches = [
    mpatches.Patch(color="tab:green",  alpha=0.3, label="Top-8 (pruned_8)"),
    mpatches.Patch(color="tab:orange", alpha=0.3, label="Top-16 (pruned_16)"),
    mpatches.Patch(color="gray",       alpha=0.3, label="Top-24 (pruned_24)"),
]
ax.legend(handles=patches, fontsize=8)

ax = axes[1]
ax.scatter(imp_sorted, np.abs(grad0), c=bar_c, s=60, edgecolors="black", lw=0.6)
ax.scatter(imp_sorted, np.abs(hess_diag), c=bar_c, marker="^",
           s=60, edgecolors="black", lw=0.6)
ax.set_xlabel(r"$\|[H, G_k]\|_F$  (importance)")
ax.set_ylabel("Value")
ax.set_title("Importance vs |gradient| (●) and |Hess diag| (▲)")
ax.grid(True, alpha=0.3)
legend_p = [
    mpatches.Patch(color="tab:orange", label="RY params"),
    mpatches.Patch(color="tab:blue",   label="RZ params"),
    plt.Line2D([0], [0], marker="o", color="gray", ms=7, lw=0, label="|gradient|"),
    plt.Line2D([0], [0], marker="^", color="gray", ms=7, lw=0, label="|Hess diag|"),
]
ax.legend(handles=legend_p, fontsize=8)

fig.suptitle(f"Importance Score vs Hessian Diagnostics — seed={SEED}",
             fontsize=11, fontweight="bold")
plt.tight_layout()
out_cmp = os.path.join(TS_DIR, "importance_vs_hessian.png")
plt.savefig(out_cmp, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: importance_vs_hessian.png")

# ---------------------------------------------------------------------------
# Per-gate side-by-side: importance score vs |Hessian diagonal|
# ---------------------------------------------------------------------------

# 중요도 순으로 정렬된 gate 순서 사용
sorted_labels = [r["label"] for r in param_info]          # importance 내림차순
sorted_imp    = np.array([r["importance"] for r in param_info])
sorted_hdiag  = np.array([abs(hess_diag[r["flat_idx"]]) for r in param_info])
sorted_grad   = np.array([abs(grad0[r["flat_idx"]])      for r in param_info])
sorted_colors = ["tab:orange" if r["gate"] == "Y" else "tab:blue" for r in param_info]

n = len(param_info)
x = np.arange(n)
w = 0.28

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# --- 상단: importance score vs |Hessian diagonal| ---
ax = axes[0]
bars_imp  = ax.bar(x - w, sorted_imp,   width=w, label=r"Importance $\|[H,G_k]\|_F$",
                   color=sorted_colors, edgecolor="black", lw=0.5, alpha=0.9)
bars_hd   = ax.bar(x,     sorted_hdiag, width=w, label=r"|Hess diag| $|\partial^2 E/\partial\theta_k^2|$",
                   color=sorted_colors, edgecolor="black", lw=0.5, alpha=0.45, hatch="//")
bars_grad = ax.bar(x + w, sorted_grad,  width=w, label=r"|gradient| $|\partial E/\partial\theta_k|$",
                   color=sorted_colors, edgecolor="black", lw=0.5, alpha=0.45, hatch="xx")

# top-8 구분선
ax.axvline(7.5, color="red", ls="--", lw=1.5, label="Top-8 boundary")
ax.axvline(15.5, color="darkorange", ls=":", lw=1.2, label="Top-16 boundary")
ax.set_ylabel("Value")
ax.set_title("Per-gate: Importance score  vs  |Hessian diagonal|  vs  |gradient|\n"
             "(sorted by importance, left=high importance)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, axis="y")

ry_patch = mpatches.Patch(color="tab:orange", label="RY gate")
rz_patch = mpatches.Patch(color="tab:blue",   label="RZ gate")
ax.legend(handles=[ry_patch, rz_patch,
                   mpatches.Patch(color="gray", alpha=0.9,  label=r"$\|[H,G_k]\|_F$ (solid)"),
                   mpatches.Patch(color="gray", alpha=0.45, hatch="//", label="|Hess diag| (hatched)"),
                   mpatches.Patch(color="gray", alpha=0.45, hatch="xx", label="|gradient| (hatched)"),
                   plt.Line2D([0],[0], color="red",        ls="--", lw=1.5, label="top-8"),
                   plt.Line2D([0],[0], color="darkorange",  ls=":",  lw=1.2, label="top-16"),
                   ],
          fontsize=8, loc="upper right", ncol=2)

# --- 하단: 두 metric의 비율 (Hess diag / importance, 이론상 proportional해야 함) ---
ax = axes[1]
# RY: both non-zero → ratio 의미있음
# RZ: importance=0, hess≈0 → 비율 대신 |hess diag| 절대값만 표시 (수치 노이즈 확인용)
ratio = np.where(sorted_imp > 0,
                 sorted_hdiag / (sorted_imp + 1e-10),
                 np.nan)
ax.bar(x, sorted_hdiag, width=0.6, color=sorted_colors,
       edgecolor="black", lw=0.5, alpha=0.7)
ax.axvline(7.5,  color="red",        ls="--", lw=1.5)
ax.axvline(15.5, color="darkorange", ls=":",  lw=1.2)
ax.axhline(0,    color="black",      lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
ax.set_ylabel(r"|Hessian diagonal|")
ax.set_title("Per-gate |Hessian diagonal|  (RZ 값이 0에 가까울수록 수치 노이즈만 있는 것)")
ax.grid(True, alpha=0.3, axis="y")

# RY: ratio 오버레이
ax2 = ax.twinx()
ax2.plot(x[:8], ratio[:8], "r^-", ms=7, lw=1.5,
         label="|Hess diag| / importance (RY only)")
ax2.set_ylabel("Hess diag / importance (RY)", color="red")
ax2.tick_params(axis="y", colors="red")
ax2.legend(fontsize=8, loc="upper left")

fig.suptitle(f"Gate-level comparison: Importance Score vs Effective Dimension metric\n"
             f"seed={SEED} | raw init | theoretical: RZ importance=0 → Hess diag=0",
             fontsize=11, fontweight="bold")
plt.tight_layout()
out_pergate = os.path.join(TS_DIR, "per_gate_comparison.png")
plt.savefig(out_pergate, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: per_gate_comparison.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n=== Analysis complete ===")
print(f"Output directory : {TS_DIR}")
print(f"Files generated  :")
for f in sorted(os.listdir(TS_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")

print(f"\nHessian summary:")
print(f"  Effective rank (participation ratio) : {eff_rank_pr:.2f} / 24")
print(f"  Effective rank (eigenvalue > 1% max) : {eff_rank_thresh} / 24")
print(f"  Non-zero importance params (RY only) : {int((imp_sorted > 0).sum())} / 24")
print(f"  Max |eigenvalue|  (RY-dominated)     : {abs_vals.max():.4f}")
print(f"  Conclusion: importance-pruning identifies the {int((imp_sorted>0).sum())}")
print(f"    RY params that drive the landscape curvature; the {24 - int((imp_sorted>0).sum())}")
print(f"    RZ params have importance=0 (Z commutes with H=ΣZ_i) and")
print(f"    correspondingly small Hessian contributions.")
