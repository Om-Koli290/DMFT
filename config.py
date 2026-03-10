# config.py
# Central configuration for all hyperparameters, physical constants, and experiment settings.
# Change everything from here — nothing is hardcoded elsewhere.

import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Energy axis ───────────────────────────────────────────────────────────────
OMEGA_MIN   = -6.0      # Lower bound of energy window (units of eV / hopping t)
OMEGA_MAX   =  6.0      # Upper bound
N_OMEGA     =  512      # Number of discretisation points

# ── Physical parameter ranges ─────────────────────────────────────────────────
# These span metallic, correlated metallic, and Mott insulating regimes
U_MIN       =  0.5      # Minimum Hubbard U (correlation strength)
U_MAX       =  4.0      # Maximum Hubbard U — above ~3.0 enters Mott insulating regime
W_MIN       =  1.0      # Minimum bandwidth
W_MAX       =  4.0      # Maximum bandwidth
N_MIN       =  0.3      # Minimum electron filling
N_MAX       =  1.0      # Maximum electron filling (half-filling = 1.0 for single orbital)

# ── Dataset ───────────────────────────────────────────────────────────────────
N_SAMPLES        = 8000    # Total number of generated spectral functions
TRAIN_FRAC       = 0.7
VAL_FRAC         = 0.15
TEST_FRAC        = 0.15
RANDOM_SEED      = 42

# Stratified sampling: fraction of dataset drawn near the Mott transition (U/W in [1.0, 1.6])
# This region is physically most important and hardest to learn
TRANSITION_OVERSAMPLE_FRAC = 0.30

# ── Model architecture ────────────────────────────────────────────────────────
INPUT_DIM        = 3        # (U, W, n)

# MLP
MLP_HIDDEN_DIMS  = [128, 256, 512, 256, 128]
MLP_DROPOUT      = 0.1

# DeepONet — branch/trunk dims widened to match DEEPONET_BASIS (avoids 2× expansion bottleneck)
BRANCH_HIDDEN    = [128, 256, 256]   # Encodes input parameters
TRUNK_HIDDEN     = [128, 256, 256]   # Encodes omega axis
DEEPONET_BASIS   = 256              # Dimension of shared latent space

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE       = 64
LEARNING_RATE    = 1e-3
DEEPONET_LR      = 3e-4             # Separate LR for DeepONet (inner-product arch is more sensitive)
N_EPOCHS         = 300
LR_PATIENCE      = 15       # Epochs before learning rate reduction
LR_FACTOR        = 0.5
EARLY_STOP       = 50       # Epochs without improvement before stopping

# KK loss warmup: ramp LAMBDA_KK from 0 to full value over this many epochs.
# Lets the model first fit the data before being constrained by causality.
KK_WARMUP_EPOCHS = 50

# ── Physics constraint weights ────────────────────────────────────────────────
# These scale the contribution of each constraint term in the loss function.
# Set to 0.0 to disable a constraint.
LAMBDA_NORM      = 5.0      # Normalisation (sum rule)
LAMBDA_SMOOTH    = 0.5      # Smoothness (second derivative penalty)
LAMBDA_KK        = 1.0      # Kramers-Kronig causality constraint
USE_KK           = True     # Toggle KK constraint on/off

# ── Evaluation ────────────────────────────────────────────────────────────────
# U/W thresholds for regime-separated evaluation
METALLIC_THRESHOLD      = 0.8    # U/W < this → metallic
CORRELATED_THRESHOLD    = 1.5    # U/W < this → correlated metal, else Mott insulating

# QP peak window: |omega| < this used to locate the quasiparticle peak
QP_PEAK_WINDOW  = 0.5

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH             = "data/spectral_dataset.pt"
MLP_CHECKPOINT        = "checkpoints/mlp_best.pt"
DEEPONET_CHECKPOINT   = "checkpoints/deeponet_best.pt"
RESULTS_DIR           = "results/"
