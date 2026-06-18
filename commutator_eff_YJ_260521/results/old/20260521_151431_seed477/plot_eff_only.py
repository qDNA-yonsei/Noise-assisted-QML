import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))

TOP_K_LIST  = [8, 16]
GLOBAL_MIN  = -4.0
SEED        = 477
COLORS      = {8: "tab:blue", 16: "tab:orange"}

# ── Load data ─────────────────────────────────────────────────────────────────
CLEAN_REF = "/home/yujin/noise_assisted_annealing/Codes/outputs/seed_training/adam/20260324_170741_seed477/20260324_170741_seed477_adam_clean_eval_hist.npy"
clean_ref_hist = np.load(CLEAN_REF)

clean_hist = {
    k: np.load(os.path.join(BASE, f"pruned_{k}_eff.npy")) for k in TOP_K_LIST
}
noise_full_e = np.load(os.path.join(BASE, "noise_full_e.npy"))
noise_full_s = np.load(os.path.join(BASE, "noise_full_steps.npy"))
noise_eff = {
    k: (
        np.load(os.path.join(BASE, f"pruned_{k}_eff_noise_e.npy")),
        np.load(os.path.join(BASE, f"pruned_{k}_eff_noise_steps.npy")),
    )
    for k in TOP_K_LIST
}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# Panel 1: Clean Adam
ax = axes[0]
ax.plot(np.arange(len(clean_ref_hist)), clean_ref_hist,
        lw=2.2, color="black", label="Clean full (checkpoint ref)")
for k in TOP_K_LIST:
    h = clean_hist[k]
    ax.plot(np.arange(len(h)), h, lw=2.0, color=COLORS[k],
            label=f"eff pruned {k}")
ax.axhline(GLOBAL_MIN, color="red", ls=":", lw=1.2, label="Global min")
ax.set_xlabel("Step")
ax.set_ylabel("Energy")
ax.set_title(f"Clean Adam — eff pruning  (seed {SEED})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Noise annealing
ax = axes[1]
ax.plot(noise_full_s, noise_full_e, lw=2.2, color="black",
        label=f"Noise full ({len(noise_full_s)} steps)")
for k in TOP_K_LIST:
    pe, ps = noise_eff[k]
    ax.plot(ps, pe, lw=2.0, color=COLORS[k],
            label=f"eff pruned {k} ({len(ps)} steps)")
ax.axhline(GLOBAL_MIN, color="red", ls=":", lw=1.2, label="Global min")
ax.set_xlabel("Cumulative step")
ax.set_ylabel("Energy (clean eval)")
ax.set_title(f"Noise annealing — eff pruning  (seed {SEED})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Final energy bar
ax = axes[2]
bar_keys, bar_vals, bar_cols = [], [], []
for k in TOP_K_LIST:
    bar_keys.append(f"pruned_{k}_eff")
    bar_vals.append(float(clean_hist[k][-1]))
    bar_cols.append(COLORS[k])
bar_keys.append("noise_full")
bar_vals.append(float(noise_full_e[-1]))
bar_cols.append("crimson")
for k in TOP_K_LIST:
    pe, _ = noise_eff[k]
    bar_keys.append(f"pruned_{k}_eff_noise")
    bar_vals.append(float(pe[-1]))
    bar_cols.append("tab:pink" if k == 8 else "tab:red")

ax.barh(bar_keys, bar_vals, color=bar_cols, edgecolor="black", lw=0.5)
ax.axvline(GLOBAL_MIN, color="red", ls=":", lw=1.2, label="Global min")
ax.set_xlabel("Final energy")
ax.set_title("Final energy summary")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="x")

plt.suptitle(f"seed={SEED} | commutator_eff pruning (eff only)", fontsize=11, fontweight="bold")
plt.tight_layout()

out = os.path.join(BASE, "training_comparison_eff_only.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
