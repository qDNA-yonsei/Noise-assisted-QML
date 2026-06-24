"""
test_clean_vs_full.py
---------------------
목적: YJ 구현이 jungyun과 동일한 결과를 내는지 검증.

  YJ clean  == jungyun clean  (H00000, seed=0)
  YJ full   == jungyun top100 (H00000, seed=0)

jungyun 기준값 (targeted_tfim_8qubit):
  clean:  E=-8.29078  normalized_gap=0.09293
  top100: E=-8.86469  normalized_gap=0.06475

Usage:
  python test_clean_vs_full.py
  python test_clean_vs_full.py --max_steps 200
"""

import argparse, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as pnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_QUBITS = 8
N_LAYERS = 4
RANGES   = [1, 2, 3, 4]
WIRES    = list(range(N_QUBITS))

YJ_SCHEDULE = [0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0]

H00000 = {
    "jzz": -1.022768208338797, "hx": 1.0429285161955684, "hz": 0.0,
    "ground": -10.183437124401689, "top": 10.183437124401669,
    "spectrum_span": 20.366874248803356,
}

# jungyun 기준값 (targeted_tfim_8qubit, H00000, seed=0)
JUNGYUN_REF = {
    "clean": {
        "clean_final_energy": -8.290776601466016,
        "normalized_gap":      0.09292837476260625,
    },
    "pauli_top100_fixed": {
        "clean_final_energy": -8.864692238529534,
        "normalized_gap":      0.0647494981194592,
    },
}

CHECK_EVERY = 10
WIN         = 20
STD_TOL     = 0.005
RATE_TOL    = 0.005
MIN_STEPS   = 50

# ---------------------------------------------------------------------------
# Hamiltonian  (open boundary, matches sweep_8qubit._build_tfim_pl)
# ---------------------------------------------------------------------------

def make_H_pl(cfg):
    jzz, hx, hz = cfg["jzz"], cfg["hx"], cfg["hz"]
    coeffs, obs = [], []
    for i in range(N_QUBITS - 1):
        coeffs.append(-jzz); obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    for i in range(N_QUBITS):
        coeffs.append(-hx); obs.append(qml.PauliX(i))
        if hz != 0.0:
            coeffs.append(-hz); obs.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeffs, obs)

# ---------------------------------------------------------------------------
# QNodes
# ---------------------------------------------------------------------------

def make_clean_cost(H_pl):
    dev = qml.device("default.qubit", wires=N_QUBITS)
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
        return qml.expval(H_pl)
    return circuit


def make_full_noise_cost(H_pl, p):
    dev = qml.device("default.mixed", wires=N_QUBITS)
    _p  = float(np.clip(p, 1e-12, 1 - 1e-12))
    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(params):
        for l in range(N_LAYERS):
            for q in range(N_QUBITS):
                qml.PauliError("Z", _p, wires=q); qml.RZ(params[l, q, 0], wires=q)
                qml.PauliError("Y", _p, wires=q); qml.RY(params[l, q, 1], wires=q)
                qml.PauliError("Z", _p, wires=q); qml.RZ(params[l, q, 2], wires=q)
            r = RANGES[l]
            for q in range(N_QUBITS):
                qml.CNOT(wires=[q, (q + r) % N_QUBITS])
        return qml.expval(H_pl)
    return circuit

# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def check_converged(hist):
    if len(hist) < WIN:
        return False
    tail = np.array(hist[-WIN:])
    return tail.std() < STD_TOL and abs(tail[-1] - tail[0]) / max(1, WIN) < RATE_TOL

# ---------------------------------------------------------------------------
# Runners  (qml.AdamOptimizer, matches jungyun)
# ---------------------------------------------------------------------------

def run_clean(clean_cost, init_params, max_steps, lr=0.01):
    """YJ clean: plain Adam, no noise. Compare with jungyun clean."""
    params = pnp.array(np.array(init_params).copy(), requires_grad=True)
    opt    = qml.AdamOptimizer(stepsize=lr)
    all_e, all_steps, hist = [], [], []
    t0 = time.time()
    print("  [YJ clean] training...")
    for step in range(1, max_steps + 1):
        params = opt.step(clean_cost, params)
        if step % CHECK_EVERY == 0:
            e = float(clean_cost(pnp.array(params, requires_grad=True)))
            all_e.append(e); all_steps.append(step); hist.append(e)
            if step >= MIN_STEPS and check_converged(hist):
                print(f"    converged at step {step}  E={e:.5f}  ({time.time()-t0:.1f}s)")
                break
    print(f"  [YJ clean] done  E_final={all_e[-1]:.5f}  ({time.time()-t0:.1f}s)")
    return np.array(all_e), np.array(all_steps)


def run_full(clean_cost, H_pl, init_params, schedule, max_steps, lr=0.01):
    """YJ full: all-gate noise annealing. Compare with jungyun pauli_top100_fixed."""
    params = np.array(init_params).copy()
    all_e, all_steps, cumul = [], [], 0
    t_start = time.time()
    print("  [YJ full] annealing...")
    for si, noise_p in enumerate(schedule, 1):
        cost_fn = clean_cost if noise_p == 0.0 else make_full_noise_cost(H_pl, noise_p)
        opt     = qml.AdamOptimizer(stepsize=lr)
        hist    = []; step_i = 0; t0 = time.time()
        while step_i < max_steps:
            params  = np.array(opt.step(cost_fn, pnp.array(params, requires_grad=True)))
            step_i += 1
            if step_i % CHECK_EVERY == 0:
                e = float(clean_cost(pnp.array(params, requires_grad=True)))
                all_e.append(e); all_steps.append(cumul + step_i); hist.append(e)
                if step_i >= MIN_STEPS and check_converged(hist):
                    print(f"    stage {si} (p={noise_p}) converged step {step_i}  E={e:.5f}")
                    break
        cumul += step_i
        print(f"  [YJ full] stage {si}/{len(schedule)} p={noise_p}  E={all_e[-1]:.5f}  ({time.time()-t0:.1f}s)")
    print(f"  [YJ full] done  E_final={all_e[-1]:.5f}  ({time.time()-t_start:.1f}s)")
    return np.array(all_e), np.array(all_steps)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",      type=int,   default=0)
    parser.add_argument("--max_steps", type=int,   default=500)
    parser.add_argument("--lr",        type=float, default=0.01)
    args = parser.parse_args()

    # 초기화 (jungyun과 동일)
    rng = np.random.default_rng(args.seed)
    init_params = rng.uniform(0.0, 2.0 * np.pi, size=(N_LAYERS, N_QUBITS, 3))

    H_pl       = make_H_pl(H00000)
    clean_cost = make_clean_cost(H_pl)
    ground     = H00000["ground"]
    span       = H00000["spectrum_span"]

    print(f"H00000  ground={ground:.4f}  seed={args.seed}  max_steps/stage={args.max_steps}")
    print(f"YJ schedule: {YJ_SCHEDULE}\n")

    # YJ clean
    e_clean, s_clean = run_clean(clean_cost, init_params, args.max_steps, args.lr)
    print()

    # YJ full (동일한 init_params에서 시작)
    e_full, s_full = run_full(clean_cost, H_pl, init_params, YJ_SCHEDULE, args.max_steps, args.lr)
    print()

    # --- 비교 결과 ---
    def norm_gap(e): return (e - ground) / span

    ref_clean  = JUNGYUN_REF["clean"]
    ref_top100 = JUNGYUN_REF["pauli_top100_fixed"]

    print("=== 검증 결과 (H00000, seed=0) ===")
    print(f"{'':6}  {'YJ E_final':>12}  {'YJ gap':>8}  {'JY E_final':>12}  {'JY gap':>8}  {'gap diff':>9}  match?")
    print("-" * 75)

    for label, e_yj, ref in [
        ("clean",  e_clean[-1], ref_clean),
        ("full",   e_full[-1],  ref_top100),
    ]:
        yj_gap  = norm_gap(e_yj)
        jy_gap  = ref["normalized_gap"]
        jy_e    = ref["clean_final_energy"]
        diff    = abs(yj_gap - jy_gap)
        match   = "✓" if diff < 0.005 else ("△" if diff < 0.02 else "✗")
        print(f"{label:6}  {e_yj:>12.5f}  {yj_gap:>8.5f}  {jy_e:>12.5f}  {jy_gap:>8.5f}  {diff:>9.5f}  {match}")

    print(f"\n  ground: {ground:.5f}  spectrum_span: {span:.5f}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s_clean, e_clean, label=f"YJ clean (E={e_clean[-1]:.4f})", color="steelblue", lw=1.5)
    ax.plot(s_full,  e_full,  label=f"YJ full  (E={e_full[-1]:.4f})",  color="tomato",    lw=1.5, ls="--")
    ax.axhline(ref_clean["clean_final_energy"],  color="steelblue", ls=":", lw=1,
               label=f"JY clean  ref (E={ref_clean['clean_final_energy']:.4f})")
    ax.axhline(ref_top100["clean_final_energy"], color="tomato",    ls=":", lw=1,
               label=f"JY top100 ref (E={ref_top100['clean_final_energy']:.4f})")
    ax.axhline(ground, color="black", ls=":", lw=1, label=f"ground ({ground:.3f})")
    ax.set_xlabel("Step"); ax.set_ylabel("Energy (clean eval)")
    ax.set_title(f"H00000  seed={args.seed} — YJ vs JY reference")
    ax.legend(fontsize=8); fig.tight_layout()

    out = "test_clean_vs_full.png"
    fig.savefig(out, dpi=120)
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
