# =============================================================================
# config.py — all hyperparameters in one place
# =============================================================================

# --- Experiment mode ---
# "standard"               : existing single-Hamiltonian training flow
# "method_suite_sample"    : run one random-Hamiltonian sample with multiple methods
#                              and save into its own sample directory (parallel-safe)
# "method_suite_aggregate" : merge finished method_suite_sample directories and
#                              build aggregate tables/plots
EXPERIMENT_MODE = "standard"

# --- Circuit ---
N_QUBITS = 4
N_LAYERS = 2
WIRES = list(range(N_QUBITS))
# RANGES is set at runtime from args.py.
RANGES = [1 + (i % max(1, N_QUBITS - 1)) for i in range(N_LAYERS)]

# --- Hamiltonian selection ---
# Available choices:
#   - "single_Z"
#   - "tfim_longitudinal"
#   - "j1j2_heisenberg"
HAMILTONIAN_NAME = "single_Z"

# Boundary condition used by the 1D chain models.
PERIODIC_BOUNDARY = False

# single_Z parameters
SINGLE_Z_COEFF = 1.0

# 1D transverse-field Ising model with an additional longitudinal field:
#   H = -Jzz * Σ Z_i Z_{i+1} - hx * Σ X_i - hz * Σ Z_i
TFIM_JZZ = 1.0
TFIM_HX = 1.0
TFIM_HZ = 0.0

# 1D J1-J2 Heisenberg chain:
#   H = J1 * Σ_nn (XX + YY + ZZ) + J2 * Σ_nnn (XX + YY + ZZ)
J1 = 1.0
J2 = 0.0

# This value is overwritten at runtime by src.hamiltonian after the selected
# Hamiltonian has been built and exact diagonalization has been performed.
GLOBAL_MIN_ENERGY = None

# --- Noise mode ---
# "matched"   : exact protocol from the paper, but the USER-FACING parameter
#                is still called p for convenience. Internally we pass p/2 to
#                qml.PauliError so that the paper channel is reproduced exactly.
# "layerwise" : legacy experimental variant; here the same user-facing p is
#                used directly as the PauliError probability.
NOISE_MODE = "matched"

# --- Optimizer choice per mode: "gd" | "adam" | "spsa" ---
CLEAN_OPTIMIZER = "adam"
SHOT_OPTIMIZER = "adam"
PAULI_OPTIMIZER = "adam"

# --- Learning rates ---
# Paper default for Adam is 0.5 x 10^-2 = 0.005.
LR_CLEAN_GD = 0.1
LR_CLEAN_ADAM = 0.01
LR_SHOT_GD = 0.1
LR_SHOT_ADAM = 0.01
LR_PAULI_GD = 0.1
LR_PAULI_ADAM = 0.01

# --- SPSA hyperparameters (per mode) ---
CLEAN_SPSA_ALPHA = 1
CLEAN_SPSA_GAMMA = 1 / 6
CLEAN_SPSA_C = 0.2
CLEAN_SPSA_A = 0
CLEAN_SPSA_a = 3

SHOT_SPSA_ALPHA = 1
SHOT_SPSA_GAMMA = 1 / 6
SHOT_SPSA_C = 0.2
SHOT_SPSA_A = 0
SHOT_SPSA_a = 3

PAULI_SPSA_ALPHA = 1
PAULI_SPSA_GAMMA = 1 / 6
PAULI_SPSA_C = 0.2
PAULI_SPSA_A = 0
PAULI_SPSA_a = 3

# --- Seed search ---
RUN_SEED_SEARCH = True
INIT_SEED = 1
SEARCH_SEEDS = list(range(1000))
SEARCH_STEPS = 1000
SEED_SEARCH_CKPT_DIR = "outputs/seed_search_ckpt"

# --- Trap diagnosis ---
SUBOPT_GAP = 0.25
GRAD_TOL = 1e-4
HESS_NEG_TOL = 1e-5

PAPER_TAIL_WINDOW = 20
PAPER_TAIL_STD_TOL = 1e-2
PAPER_TAIL_DRIFT_TOL = 1e-2

TRAP_SCORE_W_STD = 1.0
TRAP_SCORE_W_DRIFT = 1.0
TRAP_SCORE_W_GRAD = 0.5
TRAP_SCORE_W_GAP = 0.05

# --- Regularized optimisation schedule ---
# "manual"     : legacy stage schedule that you specify directly via PAULI_SCHEDULE
#                and that advances adaptively based on convergence checks.
# "fixed_step" : paper-style continuously decaying schedule over PAULI_TOTAL_STEPS.
SCHEDULE_MODE = "manual"

# Directly specified stage schedule.
# IMPORTANT: these are always USER-FACING p values in [0, 1].
#
# - matched mode  : each value is interpreted as the paper's regularization
#                   strength p, then converted internally to qml.PauliError(p/2).
# - layerwise mode: each value is used directly as qml.PauliError(p).
PAULI_SCHEDULE = [0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]

# Manual/adaptive schedule controls
PAULI_MIN_STEPS = 30
PAULI_MAX_STEPS = 500
PAULI_CHECK_EVERY = 10
PAULI_WINDOW = 20
PAULI_STD_TOL = 0.005
PAULI_RATE_TOL = 0.005
CLEAN_THRESHOLD = 0.0

# Paper-style fixed-step schedule controls
# IMPORTANT: P_MAX is always the USER-FACING p in [0, 1].
#
# - matched mode  : internal qml.PauliError probability is P_MAX / 2.
# - layerwise mode: internal qml.PauliError probability is P_MAX.
P_MAX = 0.9
# Exponential decay value(i) = value_max * exp(-a * i / i_max)
NOISE_DECAY_A = 10.0
PAULI_TOTAL_STEPS = 2000
# Log clean energy every k optimisation steps during the fixed-step run.
PAULI_LOG_EVERY = 10

# --- Shot-noise ---
SHOT_LIST = [1000, 500, 100, 50]
SHOT_REPEATS = 5
SHOT_DEVICE_BASE_SEED = 12345
SHOT_ESCAPE_THRESHOLD = None
SHOT_ESCAPE_DELTA = 0.5

# --- Shared random-Hamiltonian sampling helpers ---
SEED_SWEEP_NUM_HAMILTONIANS = 8
SEED_SWEEP_COEFF_SEED = 0
# Random coefficient ranges used by random-Hamiltonian experiments.
# The selected HAMILTONIAN_NAME determines which range set is used.
SINGLE_Z_COEFF_RANGE = (-3.0, 3.0)
TFIM_JZZ_RANGE = (-2.0, 2.0)
TFIM_HX_RANGE = (-2.0, 2.0)
TFIM_HZ_RANGE = (-2.0, 2.0)
J1_RANGE = (-2.0, 2.0)
J2_RANGE = (-2.0, 2.0)

# Shared normalized-gap helper settings used by random-Hamiltonian experiments:
#     normalized_gap = (E_final - E_ground) / (E_top - E_ground)
# A run is counted as a main failure when normalized_gap exceeds the threshold.
SEED_SWEEP_BAD_NORM_THRESHOLD = 0.10
SEED_SWEEP_THRESHOLD_SWEEP = [0.01, 0.05, 0.10, 0.20, 0.30]
SEED_SWEEP_SPAN_EPS = 1e-12


# --- Parallel-safe multi-method suite over random Hamiltonians ---
# Fixed method order used everywhere in this mode:
#   1. clean
#   2. pauli_fixed
#   3. pauli_manual
#   4. shot_100
#   5. shot_1000
METHOD_SUITE_ROOT_DIR = "outputs/seed_search_compare"
METHOD_SUITE_TAG = "default"
METHOD_SUITE_NUM_HAMILTONIANS = 8
METHOD_SUITE_SAMPLE_INDEX = 0
METHOD_SUITE_SKIP_EXISTING = True
METHOD_SUITE_BAD_NORM_THRESHOLD = 0.10
METHOD_SUITE_THRESHOLD_SWEEP = [0.01, 0.05, 0.10, 0.20, 0.30]
METHOD_SUITE_SHOT_LIST = [100, 1000]

# --- I/O ---
SAVE_PREFIX = "regularized_vs_clean_vs_shot_sel"
