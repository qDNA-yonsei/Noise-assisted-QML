# =============================================================================
# args.py — CLI argument parser
# config.py 값을 기본값으로 사용하고, 인자로 override
# =============================================================================

import argparse
from . import config as C


def parse_args():
    p = argparse.ArgumentParser(
        description="Noise-Assisted Annealing experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Circuit ---
    p.add_argument("--n-qubits",  type=int, default=C.N_QUBITS,  help="number of qubits")
    p.add_argument("--n-layers",  type=int, default=C.N_LAYERS,  help="number of SEL layers")
    p.add_argument("--noise-mode", type=str, default=C.NOISE_MODE,
                   choices=["matched", "layerwise"],
                   help="Pauli noise insertion method: "
                        "'matched' (PauliError per RZ/RY/RZ axis, default) or "
                        "'layerwise' (X/Z/Y before each SEL layer)")

    # --- Seed search ---
    p.add_argument("--no-seed-search", action="store_true",       help="skip seed search, use --init-seed")
    p.add_argument("--init-seed",      type=int, default=C.INIT_SEED, help="fixed init seed (used when --no-seed-search)")
    p.add_argument("--search-n",       type=int, default=len(C.SEARCH_SEEDS), help="number of seeds to search (range(n))")
    p.add_argument("--search-steps",   type=int, default=C.SEARCH_STEPS, help="optimizer steps per seed during search")

    # --- Optimizer ---
    p.add_argument("--optimizer", type=str, default=None,
                   choices=["gd", "adam", "spsa"],
                   help="set all three modes (clean/pauli/shot) to same optimizer")
    p.add_argument("--lr-adam",   type=float, default=None,
                   help="set adam lr for all modes (overrides individual lr settings)")

    p.add_argument("--clean-optimizer", type=str, default=C.CLEAN_OPTIMIZER, choices=["gd", "adam", "spsa"])
    p.add_argument("--pauli-optimizer", type=str, default=C.PAULI_OPTIMIZER, choices=["gd", "adam", "spsa"])
    p.add_argument("--shot-optimizer",  type=str, default=C.SHOT_OPTIMIZER,  choices=["gd", "adam", "spsa"])

    p.add_argument("--lr-clean-adam",   type=float, default=C.LR_CLEAN_ADAM)
    p.add_argument("--lr-pauli-adam",   type=float, default=C.LR_PAULI_ADAM)
    p.add_argument("--lr-shot-adam",    type=float, default=C.LR_SHOT_ADAM)
    p.add_argument("--lr-clean-gd",     type=float, default=C.LR_CLEAN_GD)
    p.add_argument("--lr-pauli-gd",     type=float, default=C.LR_PAULI_GD)
    p.add_argument("--lr-shot-gd",      type=float, default=C.LR_SHOT_GD)

    # --- Pauli annealing ---
    p.add_argument("--pauli-schedule",  type=float, nargs="+", default=C.PAULI_SCHEDULE,
                   help="noise annealing schedule (space-separated floats)")
    p.add_argument("--pauli-min-steps", type=int,   default=C.PAULI_MIN_STEPS)
    p.add_argument("--pauli-max-steps", type=int,   default=C.PAULI_MAX_STEPS)

    # --- Shot noise ---
    p.add_argument("--shot-list",    type=int, nargs="+", default=C.SHOT_LIST,
                   help="shots values to test (space-separated ints)")
    p.add_argument("--shot-repeats", type=int, default=C.SHOT_REPEATS)

    # --- I/O ---
    p.add_argument("--save-prefix", type=str, default=C.SAVE_PREFIX)

    return p.parse_args()


def apply_args(args):
    """파싱된 args를 config 모듈에 반영."""

    # Circuit
    C.N_QUBITS = args.n_qubits
    C.N_LAYERS = args.n_layers
    C.WIRES    = list(range(args.n_qubits))
    C.RANGES   = list(range(1, args.n_layers + 1))[:args.n_layers]
    C.GLOBAL_MIN_ENERGY = -float(args.n_qubits)
    C.NOISE_MODE = args.noise_mode

    # Seed search
    C.RUN_SEED_SEARCH = not args.no_seed_search
    C.INIT_SEED       = args.init_seed
    C.SEARCH_SEEDS    = list(range(args.search_n))
    C.SEARCH_STEPS    = args.search_steps

    # Optimizer (--optimizer shortcut이 개별 설정보다 우선)
    if args.optimizer:
        C.CLEAN_OPTIMIZER = args.optimizer
        C.PAULI_OPTIMIZER = args.optimizer
        C.SHOT_OPTIMIZER  = args.optimizer
    else:
        C.CLEAN_OPTIMIZER = args.clean_optimizer
        C.PAULI_OPTIMIZER = args.pauli_optimizer
        C.SHOT_OPTIMIZER  = args.shot_optimizer

    # LR (--lr-adam shortcut이 개별 설정보다 우선)
    if args.lr_adam is not None:
        C.LR_CLEAN_ADAM = args.lr_adam
        C.LR_PAULI_ADAM = args.lr_adam
        C.LR_SHOT_ADAM  = args.lr_adam
    else:
        C.LR_CLEAN_ADAM = args.lr_clean_adam
        C.LR_PAULI_ADAM = args.lr_pauli_adam
        C.LR_SHOT_ADAM  = args.lr_shot_adam

    C.LR_CLEAN_GD = args.lr_clean_gd
    C.LR_PAULI_GD = args.lr_pauli_gd
    C.LR_SHOT_GD  = args.lr_shot_gd

    # Pauli annealing
    C.PAULI_SCHEDULE  = args.pauli_schedule
    C.PAULI_MIN_STEPS = args.pauli_min_steps
    C.PAULI_MAX_STEPS = args.pauli_max_steps

    # Shot noise
    C.SHOT_LIST    = args.shot_list
    C.SHOT_REPEATS = args.shot_repeats

    # I/O
    C.SAVE_PREFIX = args.save_prefix
