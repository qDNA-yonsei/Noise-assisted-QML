"""
plot_summary_B1B2B3B4.py
------------------------
Generate summary_B1B2B3B4_8qubit_H<id>_seed<s>.png from a study3/study4 run directory.

Usage:
  python plot_summary_B1B2B3B4.py --run_dir results/noise_placement_study4_8qubit_20260618_133447
  python plot_summary_B1B2B3B4.py --run_dir results/noise_placement_study3_8qubit_20260617_162439
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOP_K_LIST  = [19, 38, 58, 77, 96]
TOP_K_FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
N_GATES     = 96

COLORS_K = {
    19: "#1f77b4",
    38: "#ff7f0e",
    58: "#2ca02c",
    77: "#d62728",
    96: "#9467bd",
}
COLOR_B1   = "#8c564b"
COLOR_GND  = "black"


def load_npy(run_dir, name):
    path = os.path.join(run_dir, name + ".npy")
    if os.path.exists(path):
        return np.load(path, allow_pickle=False)
    return None


def make_label(k, frac):
    return f"k={k} ({int(frac*100)}%)"


def plot_clean(ax, run_dir, ground):
    """Panel 1: Clean pruning — step vs energy."""
    ax.set_title("① Clean pruning\n(gate removal, no noise)", fontsize=9)
    for k, frac in zip(TOP_K_LIST, TOP_K_FRACS):
        arr = load_npy(run_dir, f"pruned_{k}_clean")
        if arr is None:
            continue
        steps = np.arange(len(arr))
        ax.plot(steps, arr, color=COLORS_K[k], lw=1.2, label=make_label(k, frac))
    ax.axhline(ground, color=COLOR_GND, ls="--", lw=0.8, label=f"Ground ({ground:.2f})")
    ax.set_xlabel("Step", fontsize=8)
    ax.set_ylabel("Energy / clean mean", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)


def plot_noise_block(ax, run_dir, prefix, title, ground, label_fn=None, show_noise_full=False):
    """Panels 2-4: noise blocks with cumulative steps."""
    ax.set_title(title, fontsize=9)
    if show_noise_full:
        ne = load_npy(run_dir, "noise_full_e")
        ns = load_npy(run_dir, "noise_full_steps")
        if ne is not None and ns is not None:
            ax.plot(ns, ne, color=COLOR_B1, lw=1.5, ls="-",
                    label="noise full (96 noisy)", zorder=3)
    for k, frac in zip(TOP_K_LIST, TOP_K_FRACS):
        ne = load_npy(run_dir, f"{prefix}_{k}_noise_e")
        ns = load_npy(run_dir, f"{prefix}_{k}_noise_steps")
        if ne is None or ns is None:
            continue
        label = label_fn(k, frac) if label_fn else (
            f"k={k} ({int(frac*100)}%)" + (" noise full" if k == N_GATES else ""))
        mask = ~np.isnan(ne)
        if mask.sum() == 0:
            ax.plot([], [], color="gray", lw=1.2, ls="--",
                    label=f"k={k} ({int(frac*100)}%) — not converged")
            continue
        ax.plot(ns[mask], ne[mask], color=COLORS_K[k], lw=1.2, label=label)
    ax.axhline(ground, color=COLOR_GND, ls="--", lw=0.8, label=f"Ground ({ground:.2f})")
    ax.set_xlabel("Cumulative step", fontsize=8)
    ax.set_ylabel("Energy / clean mean", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)


def plot_bar_summary(ax, cfg, ground):
    """Panel 5: Final energy bar chart."""
    ax.set_title("Final energy summary", fontsize=9)

    runs = cfg.get("runs", {})
    labels, values, colors = [], [], []

    def _add(key, label, color):
        rec = runs.get(key)
        if rec is None:
            return
        fe = rec.get("final_e")
        if fe is None or (isinstance(fe, float) and np.isnan(fe)):
            return
        labels.append(label)
        values.append(float(fe))
        colors.append(color)

    # B1 (= B2 k=96 noise full)
    _add("noise_full", "k=96 (100%) noise full", COLORS_K[96])

    # B4 inv_sel (bottom to top = largest k to smallest)
    for k, frac in reversed(list(zip(TOP_K_LIST, TOP_K_FRACS))):
        _add(f"inv_sel_{k}_noise",
             f"B4 inv_sel, k={k} ({int(frac*100)}%)", COLORS_K[k])

    # B3 selective
    for k, frac in reversed(list(zip(TOP_K_LIST, TOP_K_FRACS))):
        _add(f"selective_{k}_noise",
             f"B3 sel, k={k} ({int(frac*100)}%)", COLORS_K[k])

    # B2 pruned
    for k, frac in reversed(list(zip(TOP_K_LIST, TOP_K_FRACS))):
        _add(f"pruned_{k}_noise",
             f"B2 pruned, k={k} ({int(frac*100)}%)", COLORS_K[k])


    if not labels:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor="white", height=0.7)
    ax.axvline(ground, color=COLOR_GND, ls="--", lw=1.0, label=f"Ground ({ground:.2f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Final energy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)


def build_title(cfg):
    h_id  = cfg.get("h_id", "?")
    jzz   = cfg.get("jzz", 0)
    hx    = cfg.get("hx", 0)
    seed  = cfg.get("seed", "?")
    n_avg = cfg.get("n_avg", "?")
    sched = cfg.get("pauli_schedule", [])
    ns    = cfg.get("noise_scale", 1.0)
    sched_str = f"[{sched[0]:.2f}→{sched[-2]:.3f}→0.0]" if len(sched) >= 2 else str(sched)
    return (f"8-qubit TFIM noise placement study  |  commutator-eff importance scoring\n"
            f"{h_id}: jzz={jzz:.3f}, hx={hx:.3f}  |  seed={seed}, n_avg={n_avg}  |  "
            f"noise_scale={ns}  |  schedule: {sched_str}  |  PauliError range [0,1]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Path to the study directory (containing H<id>/seed<s>/ subdirs)")
    args = ap.parse_args()

    run_dir_base = args.run_dir

    # Auto-detect first H*/seed* subdirectory
    run_dir = None
    cfg = {}
    h_id = seed = None

    for h_subdir in sorted(os.listdir(run_dir_base)):
        h_path = os.path.join(run_dir_base, h_subdir)
        if not (os.path.isdir(h_path) and h_subdir.startswith("H")):
            continue
        for s_subdir in sorted(os.listdir(h_path)):
            s_path = os.path.join(h_path, s_subdir)
            cfg_path = os.path.join(s_path, "config.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                run_dir = s_path
                h_id = h_subdir
                seed = cfg.get("seed", "?")
                break
        if run_dir:
            break

    if run_dir is None:
        raise RuntimeError(f"No config.json found under {run_dir_base}")

    print(f"  run_dir : {run_dir}")
    ground = cfg.get("ground", 0.0)
    n_qubits = cfg.get("n_qubits", 8)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(build_title(cfg), fontsize=10, y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.98, top=0.88, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1:])

    plot_clean(ax1, run_dir, ground)

    plot_noise_block(ax2, run_dir, "pruned",
                     "② Pruned noise (B2)\n(gate removal + noise on kept gates)",
                     ground)

    plot_noise_block(ax3, run_dir, "selective",
                     "③ Selective noise (B3)\n(all 96 gates trained, noise on top-k)",
                     ground)

    plot_noise_block(ax4, run_dir, "inv_sel",
                     "④ Inv-selective noise (B4)\n(all 96 gates trained, noise on bottom-(96-k))",
                     ground,
                     label_fn=lambda k, frac: f"k={k} ({int(frac*100)}%, {N_GATES-k} noisy)",
                     show_noise_full=True)

    plot_bar_summary(ax5, cfg, ground)

    out_name = f"summary_B1B2B3B4_{n_qubits}qubit_{h_id}_seed{seed}.png"
    out_path = os.path.join(run_dir_base, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
