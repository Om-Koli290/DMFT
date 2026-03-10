# Physics-Informed ML for DMFT Spectral Function Prediction

Predicting electronic spectral functions A(ω) from Dynamical Mean-Field Theory (DMFT) parameters using two physics-informed neural networks: a residual MLP and a DeepONet operator learning model.

---

## Overview

In strongly correlated electron systems, computing the electronic spectral function A(ω) — which encodes the full single-particle excitation spectrum — via DMFT is computationally expensive. This project trains neural networks to learn the mapping:

```
(U, W, n)  →  A(ω)
```

where **U** is the Hubbard interaction strength, **W** is the electronic bandwidth, and **n** is the electron filling. The models are trained across three physical regimes:

| Regime | Condition | Spectral signature |
|---|---|---|
| **Metallic** | U/W < 0.8 | Sharp quasiparticle peak at ω=0, high spectral weight |
| **Correlated metal** | 0.8 < U/W < 1.2 | Reduced quasiparticle peak, emergent Hubbard bands |
| **Mott insulating** | U/W > 1.2 | Gap opens at ω=0, spectral weight redistributed to Hubbard bands |

Both models enforce three hard physical constraints via a `PhysicsConstraintLayer` applied as the final network layer, plus a Kramers-Kronig causality loss during training.

---

## Physics Constraints

### Hard constraints (enforced exactly by architecture)
1. **Positivity** — A(ω) ≥ 0 at all ω, via softplus activation
2. **Sum rule** — ∫A(ω)dω = 1, via trapezoid normalisation

### Soft constraints (enforced via auxiliary loss terms)
3. **Smoothness** — second-derivative penalty on A(ω); physical spectral functions have no sharp kinks
4. **Kramers-Kronig causality** — Re[G] and Im[G] = −πA(ω) are Hilbert transform pairs; violation implies an acausal Green's function. Computed via FFT Hilbert transform vs. direct principal value integral

---

## Models

### MLP — Residual Spectral Network
```
Input (3) → Linear projection → Residual blocks [128→256→512→256→128]
          → Output projection (512) → PhysicsConstraintLayer
```
- Residual blocks: `Linear → LayerNorm → GELU → Dropout(0.1) → Linear + skip`
- Treats A(ω) as a fixed-length vector; no inductive bias for continuity
- **397,952 parameters**

### DeepONet — Operator Learning
```
Branch net: (3) → [128→256→256→256]  ← encodes physical parameters
Trunk net:  (1) → [128→256→256→256]  ← encodes ω as a continuous variable
Output: A_raw[i,j] = Σ_k branch[i,k] · trunk[j,k] + bias[j]
      → PhysicsConstraintLayer
```
- Treats A(ω) as a **function**, not a vector
- Trunk net learns a set of basis functions {φ_k(ω)} — a learned spectral decomposition
- Can in principle evaluate at any ω point, not just training grid
- **331,776 parameters**

Reference: Lu et al., *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence, 2021.

---

## Results

All metrics computed on a held-out test set of 1,200 spectral functions.

### Overall performance

| Metric | MLP | DeepONet |
|---|---|---|
| **MSE** | **0.0522** ★ | 0.0522 |
| **MAE** | 0.0709 | **0.0699** ★ |
| **R²** | **0.291** ★ | 0.290 |
| QP Peak Height MAE | 1.815 | **1.752** ★ |
| Normalisation error | **0.000** | **0.000** |
| Negative fraction | **0.000** | **0.000** |

### Per-regime performance

| Regime | MLP MSE | DeepONet MSE | MLP R² | DeepONet R² |
|---|---|---|---|---|
| **Metallic** | **0.1359** ★ | 0.1374 | **0.260** ★ | 0.252 |
| **Correlated** | **0.0226** ★ | 0.0235 | **0.446** ★ | 0.423 |
| **Mott insulating** | 0.0103 | **0.0085** ★ | 0.221 | **0.355** ★ |

**Key findings:**
- Overall MSE is nearly identical (~0.052) — both models reach the same performance floor
- **MLP** is stronger on metallic and correlated regimes, where broader spectral features dominate
- **DeepONet** is stronger on Mott insulating (R²=0.355 vs 0.221), better capturing the sharp gap opening and Hubbard band structure at strong correlation
- DeepONet recovers the quasiparticle peak height more accurately (QP MAE 1.752 vs 1.815)
- Both models satisfy all physical constraints exactly (zero normalisation error, zero positivity violations)

---

## Result Plots

| File | Description |
|---|---|
| `dataset_examples.png` | Sample A(ω) curves from each physical regime |
| `training_curves.png` | Train and validation loss (total, reconstruction, KK) per model |
| `predictions.png` | Predicted vs reference A(ω) for representative test samples |
| `error_distribution.png` | Pointwise mean absolute error across the ω axis |
| `mott_transition.png` | A(ω) evolution as U/W sweeps from metallic → Mott insulating |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies** (pinned):
```
torch==2.10.0
numpy==1.26.4
matplotlib==3.9.2
scipy==1.13.1
```

---

## Usage

```bash
# Full run: regenerate dataset and train both models from scratch
python main.py --regen

# Train on existing dataset (no regeneration)
python main.py

# Load saved checkpoints and evaluate only (no training)
python main.py --eval-only

# Train a single model only
python main.py --model mlp
python main.py --model deeponet

# Disable Kramers-Kronig constraint for ablation
python main.py --no-kk
```

Checkpoints are saved to `checkpoints/`, results and plots to `results/`.

---

## Project Structure

```
├── main.py              # Entry point — full pipeline
├── config.py            # All hyperparameters and paths (single source of truth)
├── generator.py         # Synthetic DMFT spectral function generation
├── data.py              # Dataset class, DataLoader setup, train/val/test split
├── mlp.py               # Residual MLP model
├── deeponet.py          # DeepONet operator learning model
├── constraints.py       # PhysicsConstraintLayer, KK loss, smoothness/norm losses
├── trainer.py           # Unified training loop (early stopping, checkpointing)
├── metrics.py           # MSE, MAE, R², QP peak height, per-regime evaluation
├── plots.py             # All visualisation functions
└── requirements.txt
```

---

## Configuration

All hyperparameters are in `config.py`. Key settings:

```python
# Dataset
N_SAMPLES                = 8000
TRANSITION_OVERSAMPLE_FRAC = 0.30   # 30% of data sampled near Mott transition

# Architecture
MLP_HIDDEN_DIMS          = [128, 256, 512, 256, 128]
BRANCH_HIDDEN            = [128, 256, 256]
TRUNK_HIDDEN             = [128, 256, 256]
DEEPONET_BASIS           = 256

# Training
N_EPOCHS                 = 300
EARLY_STOP               = 50
LEARNING_RATE            = 1e-3      # MLP
DEEPONET_LR              = 3e-4      # DeepONet
KK_WARMUP_EPOCHS         = 50        # Ramp KK loss from 0 → 1.0

# Physics constraint weights
LAMBDA_NORM              = 5.0
LAMBDA_SMOOTH            = 0.5
LAMBDA_KK                = 1.0
```

---

## Data Generation

Synthetic spectral functions are generated using a Brinkman-Rice-inspired model:

- **Quasiparticle peak** — Lorentzian at ω≈0, weight Z = 1−(U/W·U_c)², width ∝ W(1−Z)
- **Lower Hubbard band** — Lorentzian centred at −U/2, weight (1−Z)/2
- **Upper Hubbard band** — Lorentzian centred at +U/2, weight (1−Z)/2
- **Particle-hole asymmetry** — peak positions shift with filling n
- Small Gaussian noise added to prevent exact memorisation of Lorentzian shapes

Sampling uses stratified oversampling near the Mott transition (U/W ∈ [1.0, 1.6]) to improve coverage of the most physically challenging region.

---

## Loss Function

```
L_total = L_reconstruction + L_physics

L_reconstruction = Σ_ω  ω_weight(ω) · (A_pred(ω) − A_ref(ω))²
    where ω_weight = 1 + 12·exp(−0.5·(ω/0.3)²)   ← upweights quasiparticle peak

L_physics = λ_norm · L_norm + λ_smooth · L_smooth + λ_KK(t) · L_KK
    λ_KK(t) = min(1, t / KK_WARMUP_EPOCHS)         ← linear warmup
```

Checkpointing and early stopping use `L_reconstruction` only, so the rising KK loss during warmup does not cause the model to checkpoint at epoch 1.
