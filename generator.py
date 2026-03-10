# data/generator.py
# Generates synthetic spectral functions representative of strongly correlated materials.
#
# Physics basis:
#   - Quasiparticle peaks are Lorentzian (Fermi liquid theory)
#   - Hubbard bands appear at +/- U/2 as correlations grow
#   - Near the Mott transition, quasiparticle weight Z → 0 and the peak collapses
#   - Spectral functions are parametrised by (U, W, n): interaction strength,
#     bandwidth, and electron filling — the natural DMFT input parameters

import numpy as np
import torch
from typing import Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config




def lorentzian(omega: np.ndarray, centre: float, width: float, weight: float) -> np.ndarray:
    """Single Lorentzian peak — the lineshape of a quasiparticle with finite lifetime."""
    return weight * (width / (2 * np.pi)) / ((omega - centre) ** 2 + (width / 2) ** 2)


def compute_quasiparticle_weight(U: float, W: float) -> float:
    """
    Quasiparticle weight Z as a function of correlation strength U/W.
    Z = 1 in the non-interacting limit, Z → 0 at the Mott transition.
    Uses a Brinkman-Rice-inspired interpolation.
    Z vanishes at U/W ≈ 1.5 (approximate Mott critical point).
    """
    ratio = U / W
    mott_critical = 1.5
    if ratio >= mott_critical:
        return 0.0
    Z = 1.0 - (ratio / mott_critical) ** 2
    return float(np.clip(Z, 0.0, 1.0))


def generate_spectral_function(
    U: float,
    W: float,
    n: float,
    omega: np.ndarray,
    noise_level: float = 0.005
) -> np.ndarray:
    """
    Generate a single spectral function for given physical parameters.

    Structure:
    - Quasiparticle peak at the Fermi level (omega=0), weight Z, width ~ W * (1-Z)
    - Lower Hubbard band centred near -U/2, weight ~ (1-Z)/2
    - Upper Hubbard band centred near +U/2, weight ~ (1-Z)/2
    - Asymmetry controlled by filling n (particle-hole symmetry broken away from n=1)

    Args:
        U: Hubbard interaction strength
        W: Electronic bandwidth
        n: Electron filling (0 to 2, half-filling = 1)
        omega: Energy axis array
        noise_level: Small Gaussian noise to add (mimics numerical noise in real DMFT)

    Returns:
        A(omega): normalised spectral function array
    """
    Z = compute_quasiparticle_weight(U, W)

    # Particle-hole asymmetry from filling (n=1 is symmetric)
    asymmetry = (n - 1.0) * 0.3

    # ── Quasiparticle peak ─────────────────────────────────────────────────────
    qp_centre = asymmetry * W * 0.1
    qp_width  = max(0.05, W * 0.15 * (1.0 - Z * 0.8))
    qp_weight = Z

    A = lorentzian(omega, qp_centre, qp_width, qp_weight)

    # ── Hubbard bands (incoherent spectral weight) ────────────────────────────
    hubbard_weight = (1.0 - Z) / 2.0

    if hubbard_weight > 0.01:
        # Lower Hubbard band — shifts with filling asymmetry
        lhb_centre = -U / 2.0 - asymmetry * 0.2
        lhb_width  = W * 0.6
        A += lorentzian(omega, lhb_centre, lhb_width, hubbard_weight * (2.0 - n))

        # Upper Hubbard band
        uhb_centre =  U / 2.0 - asymmetry * 0.2
        uhb_width  = W * 0.6
        A += lorentzian(omega, uhb_centre, uhb_width, hubbard_weight * n)

    # ── Small noise to prevent the model memorising exact Lorentzians ─────────
    A += np.random.normal(0, noise_level, size=omega.shape)
    A = np.clip(A, 0.0, None)  # Enforce positivity

    # ── Normalise to satisfy the sum rule ─────────────────────────────────────
    d_omega = omega[1] - omega[0]
    norm = np.trapz(A, dx=d_omega)
    if norm > 1e-8:
        A = A / norm

    return A.astype(np.float32)


def classify_regime(U: float, W: float) -> str:
    """Label the physical regime for a given (U, W) pair."""
    ratio = U / W
    if ratio < config.METALLIC_THRESHOLD:
        return "metallic"
    elif ratio < config.CORRELATED_THRESHOLD:
        return "correlated"
    else:
        return "mott_insulating"


def generate_dataset(
    n_samples: int = config.N_SAMPLES,
    seed: int = config.RANDOM_SEED
) -> Dict[str, torch.Tensor]:
    """
    Generate the full dataset of (parameters, spectral function) pairs.

    Returns a dict with:
        'params'  : (N, 3) tensor of (U, W, n) values
        'spectra' : (N, N_OMEGA) tensor of spectral functions
        'omega'   : (N_OMEGA,) energy axis
        'regimes' : list of regime labels (for evaluation)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    omega = np.linspace(config.OMEGA_MIN, config.OMEGA_MAX, config.N_OMEGA)

    params_list  = []
    spectra_list = []
    regimes      = []

    # Stratified sampling: oversample near the Mott transition (U/W in [1.0, 1.6])
    # This region is physically critical and hardest to learn, but underrepresented
    # in uniform sampling.
    n_transition = int(n_samples * config.TRANSITION_OVERSAMPLE_FRAC)
    n_uniform    = n_samples - n_transition

    # Uniform samples
    U_uniform = np.random.uniform(config.U_MIN, config.U_MAX, n_uniform)
    W_uniform = np.random.uniform(config.W_MIN, config.W_MAX, n_uniform)
    f_uniform = np.random.uniform(config.N_MIN, config.N_MAX, n_uniform)

    # Transition-region samples: draw W uniformly, set U = ratio * W, ratio in [1.0, 1.6]
    W_trans     = np.random.uniform(config.W_MIN, config.W_MAX, n_transition)
    ratio_trans = np.random.uniform(1.0, 1.6, n_transition)
    U_trans     = np.clip(ratio_trans * W_trans, config.U_MIN, config.U_MAX)
    f_trans     = np.random.uniform(config.N_MIN, config.N_MAX, n_transition)

    # Combine and shuffle
    U_vals = np.concatenate([U_uniform, U_trans])
    W_vals = np.concatenate([W_uniform, W_trans])
    n_vals = np.concatenate([f_uniform, f_trans])
    shuffle_idx = np.random.permutation(n_samples)
    U_vals = U_vals[shuffle_idx]
    W_vals = W_vals[shuffle_idx]
    n_vals = n_vals[shuffle_idx]

    print(f"Generating {n_samples} spectral functions...")
    for i in range(n_samples):
        U, W, n = U_vals[i], W_vals[i], n_vals[i]
        A = generate_spectral_function(U, W, n, omega)

        params_list.append([U, W, n])
        spectra_list.append(A)
        regimes.append(classify_regime(U, W))

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_samples} generated")

    params  = torch.tensor(np.array(params_list), dtype=torch.float32)
    spectra = torch.tensor(np.array(spectra_list), dtype=torch.float32)
    omega_t = torch.tensor(omega, dtype=torch.float32)

    # Normalise input parameters to [0, 1] for stable training
    params_min = params.min(dim=0).values
    params_max = params.max(dim=0).values
    params_norm = (params - params_min) / (params_max - params_min + 1e-8)

    print(f"\nDataset summary:")
    print(f"  Total samples    : {n_samples}")
    print(f"  Metallic         : {regimes.count('metallic')}")
    print(f"  Correlated metal : {regimes.count('correlated')}")
    print(f"  Mott insulating  : {regimes.count('mott_insulating')}")
    print(f"  Omega range      : [{omega.min():.1f}, {omega.max():.1f}]")
    print(f"  Spectral shape   : {spectra.shape}")

    return {
        "params":      params_norm,
        "params_raw":  params,
        "spectra":     spectra,
        "omega":       omega_t,
        "regimes":     regimes,
        "params_min":  params_min,
        "params_max":  params_max,
    }
