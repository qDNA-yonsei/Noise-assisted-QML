# =============================================================================
# args.py — CLI argument parser
# =============================================================================

import argparse
from . import config as C


_AVAILABLE_HAMILTONIANS = ["single_Z", "tfim_longitudinal", "j1j2_heisenberg"]


def _default_ranges(n_qubits: int, n_layers: int) -> list[int]:
    """
    Build a sensible SEL range list for arbitrary layer counts.

    StronglyEntanglingLayers expects one range per layer. For deep circuits we
    cycle through the valid interaction distances 1, 2, ..., n_qubits-1 instead
    of letting the range grow without bound.
    """
    max_range = max(1, n_qubits - 1)
    return [1 + (layer_idx % max_range) for layer_idx in range(n_layers)]



def parse_args():
    p = argparse.ArgumentParser(
        description="Noise-Assisted Annealing experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Circuit ---
    p.add_argument("--n-qubits", type=int, default=C.N_QUBITS, help="number of qubits")
    p.add_argument("--n-layers", type=int, default=C.N_LAYERS, help="number of SEL layers")
    p.add_argument(
        "--noise-mode",
        type=str,
        default=C.NOISE_MODE,
        choices=["matched", "layerwise"],
        help=(
            "Noise insertion method. 'matched' reproduces the paper exactly, but still uses the "
            "single user-facing name p; internally it passes p/2 to qml.PauliError. "
            "'layerwise' is the legacy experimental variant and uses p directly."
        ),
    )

    # --- Hamiltonian selection ---
    p.add_argument(
        "--hamiltonian",
        type=str,
        default=C.HAMILTONIAN_NAME,
        choices=_AVAILABLE_HAMILTONIANS,
        help=(
            "Hamiltonian family to use. 'single_Z' is the original baseline from the project. "
            "'tfim_longitudinal' is the 1D transverse-field Ising model with an extra longitudinal field. "
            "'j1j2_heisenberg' is the 1D J1-J2 Heisenberg chain."
        ),
    )
    p.add_argument(
        "--periodic",
        dest="periodic",
        action="store_true",
        default=C.PERIODIC_BOUNDARY,
        help="use periodic boundary conditions for the 1D chain Hamiltonians",
    )
    p.add_argument(
        "--open-boundary",
        dest="periodic",
        action="store_false",
        help="use open boundary conditions for the 1D chain Hamiltonians",
    )
    p.add_argument("--single-z-coeff", type=float, default=C.SINGLE_Z_COEFF, help="coefficient for H = coeff * Σ_i Z_i")
    p.add_argument("--tfim-jzz", type=float, default=C.TFIM_JZZ, help="nearest-neighbour ZZ coupling for tfim_longitudinal")
    p.add_argument("--tfim-hx", type=float, default=C.TFIM_HX, help="transverse X-field strength for tfim_longitudinal")
    p.add_argument("--tfim-hz", type=float, default=C.TFIM_HZ, help="longitudinal Z-field strength for tfim_longitudinal")
    p.add_argument("--j1", type=float, default=C.J1, help="nearest-neighbour coupling for j1j2_heisenberg")
    p.add_argument("--j2", type=float, default=C.J2, help="next-nearest-neighbour coupling for j1j2_heisenberg")

    # --- Seed search ---
    p.add_argument("--no-seed-search", action="store_true", help="skip seed search, use --init-seed")
    p.add_argument("--init-seed", type=int, default=C.INIT_SEED, help="fixed init seed (used when --no-seed-search)")
    p.add_argument("--search-n", type=int, default=len(C.SEARCH_SEEDS), help="number of seeds to search (range(n))")
    p.add_argument("--search-steps", type=int, default=C.SEARCH_STEPS, help="optimizer steps per seed during search")
    p.add_argument(
        "--seed-search-ckpt-dir",
        type=str,
        default=C.SEED_SEARCH_CKPT_DIR,
        help="directory to save/load seed search checkpoints",
    )

    # --- Optimizer ---
    p.add_argument("--optimizer", type=str, default=None, choices=["gd", "adam", "spsa"], help="set all three modes to the same optimizer")
    p.add_argument("--lr-adam", type=float, default=None, help="set Adam lr for all modes")

    p.add_argument("--clean-optimizer", type=str, default=C.CLEAN_OPTIMIZER, choices=["gd", "adam", "spsa"])
    p.add_argument("--pauli-optimizer", type=str, default=C.PAULI_OPTIMIZER, choices=["gd", "adam", "spsa"])
    p.add_argument("--shot-optimizer", type=str, default=C.SHOT_OPTIMIZER, choices=["gd", "adam", "spsa"])

    p.add_argument("--lr-clean-adam", type=float, default=C.LR_CLEAN_ADAM)
    p.add_argument("--lr-pauli-adam", type=float, default=C.LR_PAULI_ADAM)
    p.add_argument("--lr-shot-adam", type=float, default=C.LR_SHOT_ADAM)
    p.add_argument("--lr-clean-gd", type=float, default=C.LR_CLEAN_GD)
    p.add_argument("--lr-pauli-gd", type=float, default=C.LR_PAULI_GD)
    p.add_argument("--lr-shot-gd", type=float, default=C.LR_SHOT_GD)

    # --- Regularized optimisation schedule ---
    p.add_argument(
        "--schedule-mode",
        type=str,
        default=C.SCHEDULE_MODE,
        choices=["manual", "fixed_step"],
        help=(
            "'manual': use a user-specified stage schedule with convergence-based stage transitions. "
            "'fixed_step': use the paper-style exponentially decaying schedule over a fixed number of steps."
        ),
    )
    p.add_argument(
        "--pauli-schedule",
        type=float,
        nargs="+",
        default=C.PAULI_SCHEDULE,
        help=(
            "directly specified stage schedule. These values are always the user-facing p in [0, 1]. "
            "In matched mode each value is converted internally to qml.PauliError(p/2). "
            "In layerwise mode each value is used directly as qml.PauliError(p)."
        ),
    )
    p.add_argument("--pauli-min-steps", type=int, default=C.PAULI_MIN_STEPS)
    p.add_argument("--pauli-max-steps", type=int, default=C.PAULI_MAX_STEPS)
    p.add_argument("--pauli-check-every", type=int, default=C.PAULI_CHECK_EVERY)
    p.add_argument("--pauli-window", type=int, default=C.PAULI_WINDOW)
    p.add_argument("--pauli-std-tol", type=float, default=C.PAULI_STD_TOL)
    p.add_argument("--pauli-rate-tol", type=float, default=C.PAULI_RATE_TOL)
    p.add_argument(
        "--clean-threshold",
        type=float,
        default=C.CLEAN_THRESHOLD,
        help=(
            "for manual mode only: if the scheduled user-facing p is <= this threshold, use "
            "CLEAN_COST instead of density-matrix simulation"
        ),
    )

    # --- Paper-style fixed-step regularized optimisation ---
    p.add_argument(
        "--p-max", "--mu-max", "--layerwise-p-max",
        dest="p_max",
        type=float,
        default=C.P_MAX,
        help=(
            "maximum user-facing p for the fixed-step run. Preferred flag: --p-max. "
            "For backward compatibility, --mu-max and --layerwise-p-max are also accepted. "
            "In matched mode this is converted internally to qml.PauliError(p/2); "
            "in layerwise mode it is used directly."
        ),
    )
    p.add_argument("--noise-decay-a", type=float, default=C.NOISE_DECAY_A, help="exponential decay parameter a in exp(-a * i / i_max)")
    p.add_argument("--pauli-total-steps", type=int, default=C.PAULI_TOTAL_STEPS, help="total optimizer steps for the fixed-step regularized run")
    p.add_argument("--pauli-log-every", type=int, default=C.PAULI_LOG_EVERY, help="log clean energy every k steps during the fixed-step run")

    # --- Shot noise ---
    p.add_argument("--shot-list", type=int, nargs="+", default=C.SHOT_LIST, help="shots values to test")
    p.add_argument("--shot-repeats", type=int, default=C.SHOT_REPEATS)

    # --- I/O ---
    p.add_argument("--save-prefix", type=str, default=C.SAVE_PREFIX)

    return p.parse_args()



def apply_args(args):
    """Apply parsed args to the config module."""

    # Circuit
    C.N_QUBITS = args.n_qubits
    C.N_LAYERS = args.n_layers
    C.WIRES = list(range(args.n_qubits))
    C.RANGES = _default_ranges(args.n_qubits, args.n_layers)
    C.NOISE_MODE = args.noise_mode

    # Hamiltonian
    C.HAMILTONIAN_NAME = args.hamiltonian
    C.PERIODIC_BOUNDARY = bool(args.periodic)
    C.SINGLE_Z_COEFF = args.single_z_coeff
    C.TFIM_JZZ = args.tfim_jzz
    C.TFIM_HX = args.tfim_hx
    C.TFIM_HZ = args.tfim_hz
    C.J1 = args.j1
    C.J2 = args.j2

    # Seed search
    C.RUN_SEED_SEARCH = not args.no_seed_search
    C.INIT_SEED = args.init_seed
    C.SEARCH_SEEDS = list(range(args.search_n))
    C.SEARCH_STEPS = args.search_steps
    C.SEED_SEARCH_CKPT_DIR = args.seed_search_ckpt_dir

    # Optimizer
    if args.optimizer:
        C.CLEAN_OPTIMIZER = args.optimizer
        C.PAULI_OPTIMIZER = args.optimizer
        C.SHOT_OPTIMIZER = args.optimizer
    else:
        C.CLEAN_OPTIMIZER = args.clean_optimizer
        C.PAULI_OPTIMIZER = args.pauli_optimizer
        C.SHOT_OPTIMIZER = args.shot_optimizer

    if args.lr_adam is not None:
        C.LR_CLEAN_ADAM = args.lr_adam
        C.LR_PAULI_ADAM = args.lr_adam
        C.LR_SHOT_ADAM = args.lr_adam

    C.LR_CLEAN_ADAM = args.lr_clean_adam
    C.LR_PAULI_ADAM = args.lr_pauli_adam
    C.LR_SHOT_ADAM = args.lr_shot_adam
    C.LR_CLEAN_GD = args.lr_clean_gd
    C.LR_PAULI_GD = args.lr_pauli_gd
    C.LR_SHOT_GD = args.lr_shot_gd

    # Regularized optimisation schedule
    C.SCHEDULE_MODE = args.schedule_mode
    C.PAULI_SCHEDULE = args.pauli_schedule
    C.PAULI_MIN_STEPS = args.pauli_min_steps
    C.PAULI_MAX_STEPS = args.pauli_max_steps
    C.PAULI_CHECK_EVERY = args.pauli_check_every
    C.PAULI_WINDOW = args.pauli_window
    C.PAULI_STD_TOL = args.pauli_std_tol
    C.PAULI_RATE_TOL = args.pauli_rate_tol
    C.CLEAN_THRESHOLD = args.clean_threshold

    C.P_MAX = args.p_max
    C.NOISE_DECAY_A = args.noise_decay_a
    C.PAULI_TOTAL_STEPS = args.pauli_total_steps
    C.PAULI_LOG_EVERY = args.pauli_log_every

    # Shot noise
    C.SHOT_LIST = args.shot_list
    C.SHOT_REPEATS = args.shot_repeats

    # IO
    C.SAVE_PREFIX = args.save_prefix
