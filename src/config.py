# =============================================================================
# config.py — all hyperparameters in one place
# =============================================================================

# --- Circuit ---
N_QUBITS = 4
N_LAYERS = 2
WIRES = list(range(N_QUBITS))
RANGES = [1, 2]

# --- Noise mode ---
# "matched"   : PauliError(Z)->RZ, PauliError(Y)->RY, PauliError(Z)->RZ  (default)
# "layerwise" : PauliError(X/Z/Y) before each SEL layer (original)
NOISE_MODE = "matched"   # len == N_LAYERS

# --- Optimizer choice per mode: "gd" | "adam" | "spsa" ---
CLEAN_OPTIMIZER = "adam"
SHOT_OPTIMIZER  = "adam"
PAULI_OPTIMIZER = "adam"

# --- Learning rates ---
LR_CLEAN_GD   = 0.1
LR_CLEAN_ADAM = 0.01
LR_SHOT_GD    = 0.1
LR_SHOT_ADAM  = 0.01
LR_PAULI_GD   = 0.1
LR_PAULI_ADAM = 0.01

# --- SPSA hyperparameters (per mode) ---
CLEAN_SPSA_ALPHA = 1
CLEAN_SPSA_GAMMA = 1 / 6
CLEAN_SPSA_C     = 0.2
CLEAN_SPSA_A     = 0
CLEAN_SPSA_a     = 3

SHOT_SPSA_ALPHA = 1
SHOT_SPSA_GAMMA = 1 / 6
SHOT_SPSA_C     = 0.2
SHOT_SPSA_A     = 0
SHOT_SPSA_a     = 3

PAULI_SPSA_ALPHA = 1
PAULI_SPSA_GAMMA = 1 / 6
PAULI_SPSA_C     = 0.2
PAULI_SPSA_A     = 0
PAULI_SPSA_a     = 3

# --- Seed search ---
RUN_SEED_SEARCH      = True
INIT_SEED            = 1          # used only when RUN_SEED_SEARCH=False
SEARCH_SEEDS         = list(range(1000))
SEARCH_STEPS         = 500
SEED_SEARCH_CKPT_DIR = "outputs/seed_search_ckpt"  # checkpoint dir for seed search

# --- Trap diagnosis ---
SUBOPT_GAP        = 0.25
GRAD_TOL          = 1e-4
HESS_NEG_TOL      = 1e-5

PAPER_TAIL_WINDOW    = 20
PAPER_TAIL_STD_TOL   = 1e-2
PAPER_TAIL_DRIFT_TOL = 1e-2

TRAP_SCORE_W_STD   = 1.0
TRAP_SCORE_W_DRIFT = 1.0
TRAP_SCORE_W_GRAD  = 0.5
TRAP_SCORE_W_GAP   = 0.05   # subtracted: larger gap → slightly preferred

# --- Adaptive Pauli annealing ---
PAULI_SCHEDULE  = [0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.007, 0.005, 0.0]
PAULI_MIN_STEPS = 50
PAULI_MAX_STEPS = 1000
PAULI_CHECK_EVERY   = 10
PAULI_WINDOW        = 20
PAULI_STD_TOL       = 0.005
PAULI_RATE_TOL      = 0.005
CLEAN_THRESHOLD     = 0.005   # p ≤ this → use noiseless device

# --- Shot-noise ---
SHOT_LIST            = [1000, 500, 100, 50]
SHOT_REPEATS         = 10
SHOT_DEVICE_BASE_SEED = 12345
SHOT_ESCAPE_THRESHOLD = None   # None → clean_final - SHOT_ESCAPE_DELTA
SHOT_ESCAPE_DELTA     = 0.5

# --- I/O ---
SAVE_PREFIX = "adaptive_pauli_vs_shot_sel"

# --- Ground truth ---
GLOBAL_MIN_ENERGY = -float(N_QUBITS)
