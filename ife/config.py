"""
Physical and algorithmic constants for the IFE laser-beam engagement problem.
All SI units unless noted.
"""

# ── Facility / target parameters ────────────────────────────────────────────
TARGET_SPEED_NOM   = 450.0       # m/s  nominal injection speed
CHAMBER_RADIUS     = 5.0         # m    reactor chamber inner radius
MIRROR_RADIUS      = 6.0         # m    mirror stand-off from chamber centre
TARGET_CAPSULE_R   = 2.2e-3      # m    fuel-capsule radius
ACCURACY_3SIGMA    = 20e-6       # m    20 µm 3σ engagement-error requirement

# ── Timing ───────────────────────────────────────────────────────────────────
DT              = 1e-3           # s    inter-measurement interval
TAU_LATENCY     = 3e-3           # s    system latency (mid-range of [1, 5] ms)
N_MEASUREMENTS  = 5              # m    number of tracking measurements (range 3–8)
SHOT_RATE       = 5.0            # Hz   repetition rate

# ── Beamlet / mirror parameters ──────────────────────────────────────────────
N_BEAMLETS      = 4              # N    number of beamlets
THETA_DDOT_MAX   = 1000.0        # rad/s²  per-axis mirror slew-rate limit
THETA_MIN       = -0.5           # rad  global lower bound on pointing angle
THETA_MAX       =  0.5           # rad  global upper bound

# ── Process-noise parameters ─────────────────────────────────────────────────
SIGMA_A_PROCESS = 20           # m/s²  process noise in target acceleration (per axis)
# ── Measurement-noise parameters ─────────────────────────────────────────────
SIGMA_TRANSVERSE = 5e-6          # m    transverse (x, y) measurement std
SIGMA_AXIAL      = 10e-6         # m    axial (z) measurement std

# ── Stochastic / GP hyperparameters ──────────────────────────────────────────
N_SAMPLES_SAA   = 50             # S    MC draws for SAA objective
N_MC_EVAL       = 1_000          # trajectories for final evaluation
GP_SIGMA_F      = 1.0            # GP signal standard deviation
GP_LENGTH_SCALE = 0.05           # m    GP spatial length-scale
N_GP_HISTORY    = 20             # I    previous trajectories used for GP fit

# ── Robust OUU parameters ────────────────────────────────────────────────────
CAL_DRIFT_STD   = 10e-6          # m    1σ calibration drift per beamlet axis
CHI2_95_3DOF    = 7.815          # χ²_{3, 0.95} critical value
