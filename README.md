# Physics-Informed ML for DMFT Spectral Function Prediction

Two physics-informed neural networks — a residual **MLP** and a **DeepONet** operator learning model — trained to predict electronic spectral functions A(ω) from Dynamical Mean-Field Theory (DMFT) input parameters.

---

## Background

Computing spectral functions via full DMFT is computationally expensive. This project trains neural networks as fast surrogates, learning the mapping:

```
(U, W, n)  →  A(ω)
```

where **U** is the Hubbard interaction strength, **W** is the electronic bandwidth, and **n** is the electron filling fraction. The spectral function A(ω) encodes the full single-particle excitation spectrum and is the central observable in correlated electron physics.

### Physical regimes

The parameter space spans three distinct phases of matter, classified by the ratio U/W:

| Regime | U/W range | Physical picture |
|---|---|---|
| **Metallic** | < 0.8 | Sharp quasiparticle peak at ω = 0; large spectral weight at Fermi level |
| **Correlated metal** | 0.8 – 1.5 | Quasiparticle peak narrows as Z → 0; incoherent Hubbard bands emerge |
| **Mott insulating** | ≥ 1.5 | Gap opens at ω = 0; spectral weight fully transferred to Hubbard bands |

The boundary U/W = 1.5 is set by the **Brinkman-Rice formula**: the quasiparticle weight Z = 1 − (U/W · U_c)² vanishes exactly at U/W = 1.5, marking the Mott critical point.

---

## Models

### MLP — Residual Spectral Network (`mlp.py`)

```
Input (3) → Linear projection → Residual blocks → Output (512) → PhysicsConstraintLayer

Residual blocks: Linear → LayerNorm → GELU → Dropout(0.1) → Linear + skip connection
Hidden dims: [128, 256, 512, 256, 128]
```

- Treats A(ω) as a fixed-length vector output
- Residual connections with LayerNorm stabilise training across the wide hidden dimensions
- **397,952 parameters**

### DeepONet — Operator Learning Model (`deeponet.py`)

```
Branch net: (U,W,n) → [128 → 256 → 256 → 256]   (encodes physical parameters)
Trunk net:  ω       → [128 → 256 → 256 → 256]   (encodes energy axis)
Output:     A_raw[batch, ω] = Σ_k branch_k · trunk_k(ω) + bias(ω)
          → PhysicsConstraintLayer
```

- Treats A(ω) as a **continuous function** of ω, not a fixed vector
- The trunk net learns a set of basis functions {φ_k(ω)} — a data-driven spectral decomposition
- Can in principle be queried at any ω, not just the training grid
- **331,776 parameters**

> Reference: Lu et al., *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence (2021)

---

## Physics Constraints

Three constraints are enforced on every prediction:

### Hard constraints — enforced by architecture
Applied as a `PhysicsConstraintLayer` at the output of both models:

1. **Positivity** — `A(ω) ≥ 0` via softplus activation; no spectral weight can be negative
2. **Sum rule** — `∫A(ω)dω = 1` via trapezoid normalisation; total spectral weight is conserved

### Soft constraints — enforced via auxiliary loss terms

3. **Smoothness** — second-derivative penalty penalises unphysical sharp kinks in A(ω)
4. **Kramers-Kronig causality** — the retarded Green's function must be analytic in the upper half complex plane, forcing Re[G] and Im[G] = −πA(ω) to be Hilbert transform pairs. Computed via FFT Hilbert transform vs. direct principal value integral over 128 reference ω points

---

## Loss Function

```
L_total = L_reconstruction + L_physics

L_reconstruction = Σ_ω  w(ω) · (A_pred(ω) − A_ref(ω))²
    w(ω) = 1 + 12 · exp(−0.5 · (ω/0.3)²)    ← upweights quasiparticle peak region

L_physics = λ_norm · L_norm  +  λ_smooth · L_smooth  +  λ_KK(t) · L_KK
    λ_KK(t) = min(1, t / 50)                 ← linear KK warmup over 50 epochs
```

Checkpointing and early stopping use `L_reconstruction` only. Using total loss caused models to checkpoint at epoch 1 (when KK contributed near-zero), so this separation is essential.

---

## Results

Test set: 1,200 held-out spectral functions.

### Overall metrics

| Metric | MLP | DeepONet |
|---|---|---|
| **MSE** | **0.0522** ★ | 0.0522 |
| **MAE** | 0.0709 | **0.0699** ★ |
| **R²** | **0.291** ★ | 0.290 |
| QP Peak Height MAE | 1.815 | **1.752** ★ |
| Normalisation error | **0.000** | **0.000** |
| Positivity violations | **0.000** | **0.000** |

### Per-regime breakdown

| Regime | MLP MSE | DeepONet MSE | MLP R² | DeepONet R² |
|---|---|---|---|---|
| **Metallic** | **0.1359** ★ | 0.1374 | **0.260** ★ | 0.252 |
| **Correlated** | **0.0168** ★ | 0.0172 | **0.435** ★ | 0.423 |
| **Mott insulating** | 0.0132 | **0.0098** ★ | 0.021 | **0.272** ★ |

### Key findings

- **Overall performance is nearly identical** (~0.052 MSE for both) — the bottleneck is data and physics, not architecture
- **MLP leads on metallic and correlated regimes** — residual blocks efficiently learn the broader, smoother spectral features that dominate these phases
- **DeepONet leads decisively on Mott insulating** (R² 0.272 vs 0.021) — the operator learning framework better captures sharp gap formation and Hubbard band separation; MLP's near-zero Mott R² reveals it cannot reproduce the full diversity of truly gapped spectra
- **DeepONet recovers quasiparticle peak height more accurately** (QP MAE 1.752 vs 1.815), which is the physically most important feature in the correlated metal phase
- **All physics constraints satisfied exactly** by both models — zero normalisation error and zero positivity violations across the entire test set

---

## Training Summary

| | MLP | DeepONet |
|---|---|---|
| Epochs run | 300 (full budget) | 115 (early stopped) |
| Best checkpoint | Epoch 300 | Epoch 65 |
| Best val recon loss | 0.3662 | 0.3556 |
| Learning rate | 1e-3 | 3e-4 |

The MLP's reconstruction loss improved monotonically through all 300 epochs. DeepONet converged faster but stopped improving around epoch 65 in reconstruction terms.

---

## Result Plots

All generated plots are saved to `results/`:

| File | Description |
|---|---|
| `dataset_examples.png` | Sample A(ω) curves from each physical regime |
| `training_curves.png` | Train and validation loss (total, reconstruction, KK) per model |
| `predictions.png` | Predicted vs reference A(ω) for representative test samples across regimes |
| `error_distribution.png` | Pointwise mean absolute error ± std across the ω axis |
| `mott_transition.png` | A(ω) evolution as U/W sweeps from metallic → Mott insulating (W=2.0, n=1.0 fixed) |

---

## Installation

```bash
pip install -r requirements.txt
```

Pinned dependencies:
```
torch==2.10.0
numpy==1.26.4
matplotlib==3.9.2
scipy==1.13.1
```

---

## Usage

```bash
# Regenerate dataset and train both models from scratch
python main.py --regen

# Train on existing dataset
python main.py

# Skip training, load saved checkpoints and evaluate
python main.py --eval-only

# Train a single model only (other loaded from checkpoint)
python main.py --model mlp
python main.py --model deeponet

# Disable Kramers-Kronig constraint (ablation)
python main.py --no-kk
```

Checkpoints are saved to `checkpoints/`, all plots to `results/`.

---

## Project Structure

```
├── main.py            Entry point — orchestrates the full pipeline
├── config.py          All hyperparameters and paths (single source of truth)
├── generator.py       Synthetic DMFT spectral function generation (Brinkman-Rice model)
├── data.py            SpectralDataset, DataLoader setup, train/val/test split
├── mlp.py             Residual MLP model
├── deeponet.py        DeepONet operator learning model
├── constraints.py     PhysicsConstraintLayer, KK loss, smoothness and norm losses
├── trainer.py         Training loop — early stopping, KK warmup, checkpointing
├── metrics.py         MSE, MAE, R², QP peak height MAE, per-regime evaluation
├── plots.py           All visualisation functions
└── requirements.txt
```

---

## Configuration Reference

All settings are in `config.py`. Notable values:

```python
# Dataset
N_SAMPLES                  = 8000
TRANSITION_OVERSAMPLE_FRAC = 0.30    # 30% of samples drawn near Mott transition

# Regime boundaries (consistent with Brinkman-Rice Z = 0 at U/W = 1.5)
METALLIC_THRESHOLD         = 0.8
CORRELATED_THRESHOLD       = 1.5

# Architecture
MLP_HIDDEN_DIMS            = [128, 256, 512, 256, 128]
BRANCH_HIDDEN              = [128, 256, 256]   # DeepONet branch net
TRUNK_HIDDEN               = [128, 256, 256]   # DeepONet trunk net
DEEPONET_BASIS             = 256               # Shared latent dimension

# Training
N_EPOCHS                   = 300
EARLY_STOP                 = 50
LEARNING_RATE              = 1e-3    # MLP
DEEPONET_LR                = 3e-4    # DeepONet (inner-product arch more sensitive)
KK_WARMUP_EPOCHS           = 50      # Epochs to ramp λ_KK from 0 → 1.0

# Physics loss weights
LAMBDA_NORM                = 5.0
LAMBDA_SMOOTH              = 0.5
LAMBDA_KK                  = 1.0
```

---

## Data Generation

Spectral functions are generated analytically using a three-peak Lorentzian model:

| Feature | Centre | Width | Weight |
|---|---|---|---|
| Quasiparticle peak | ω ≈ 0 | W · 0.15 · (1 − 0.8Z) | Z |
| Lower Hubbard band | −U/2 | W · 0.6 | (1−Z)/2 · (2−n) |
| Upper Hubbard band | +U/2 | W · 0.6 | (1−Z)/2 · n |

where Z = max(0, 1 − (U/W / 1.5)²) is the quasiparticle weight (Brinkman-Rice). Particle-hole asymmetry shifts peak positions with filling n. Small Gaussian noise (σ=0.005) is added to prevent exact memorisation of Lorentzian shapes. All spectra are normalised to satisfy the sum rule.

Dataset sampling uses 70% uniform coverage of (U, W, n) space plus 30% targeted sampling with U/W ∈ [1.0, 1.6] to improve coverage of the physically critical Mott transition region.
