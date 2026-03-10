# models/constraints.py
# Physics constraint enforcement layer and loss terms.
#
# Three hard constraints on any physical spectral function:
#   1. Positivity:    A(omega) >= 0
#   2. Sum rule:      integral A(omega) d(omega) = 1
#   3. Smoothness:    A(omega) is continuous and slowly varying
#
# One deep constraint from causality:
#   4. Kramers-Kronig: Re[G] and Im[G] are related by the Hilbert transform.
#      A spectral function violating KK is acausal — cannot correspond to any
#      physical system.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PhysicsConstraintLayer(nn.Module):
    """
    Wraps any model's raw output and enforces positivity + normalisation.
    Applied as the final layer of both MLP and DeepONet.
    """
    def __init__(self, omega: torch.Tensor):
        super().__init__()
        self.register_buffer("omega", omega)
        d_omega = omega[1] - omega[0]
        self.register_buffer("d_omega", d_omega)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw: (batch, N_OMEGA) — unconstrained model output
        Returns:
            A: (batch, N_OMEGA) — positivity-enforced and normalised spectral function
        """
        # Positivity via softplus (smooth, differentiable, always > 0)
        A = F.softplus(raw)

        # Normalisation — enforce sum rule integral A d(omega) = 1
        norm = torch.trapezoid(A, dx=self.d_omega, dim=-1).unsqueeze(-1)
        A = A / (norm + 1e-8)

        return A


def smoothness_loss(A: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Penalises non-smooth spectral functions via second derivative.
    Physical spectral functions do not have sharp kinks or oscillations.

    Args:
        A:     (batch, N_OMEGA) spectral function
        omega: (N_OMEGA,) energy axis
    Returns:
        Scalar smoothness penalty
    """
    d_omega = omega[1] - omega[0]
    # Second derivative via finite differences
    d2A = (A[:, 2:] - 2 * A[:, 1:-1] + A[:, :-2]) / (d_omega ** 2)
    return torch.mean(d2A ** 2)


def normalisation_loss(A: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Residual normalisation penalty — extra enforcement beyond the constraint layer.
    Penalises deviation of integral(A) from 1.

    Args:
        A:     (batch, N_OMEGA) spectral function
        omega: (N_OMEGA,) energy axis
    Returns:
        Scalar normalisation penalty
    """
    d_omega = omega[1] - omega[0]
    norms = torch.trapezoid(A, dx=d_omega, dim=-1)
    return torch.mean((norms - 1.0) ** 2)


def hilbert_transform(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hilbert transform of A(omega) using FFT.
    This gives Re[G(omega)] from Im[G(omega)] = -pi * A(omega).

    The Hilbert transform H[f](t) = (1/pi) P.V. integral f(t')/(t-t') dt'
    is computed efficiently as: H[f] = IFFT(-i * sign(freq) * FFT(f))

    Args:
        A: (batch, N_OMEGA) spectral function
    Returns:
        ReG: (batch, N_OMEGA) real part of Green's function implied by A
    """
    # FFT along frequency axis
    F_A = torch.fft.rfft(A, dim=-1)

    n = A.shape[-1]
    freqs = torch.fft.rfftfreq(n, device=A.device)

    # Multiply by -i * sign(freq) in frequency domain = Hilbert transform
    sign = torch.sign(freqs)
    sign[0] = 0.0  # Zero DC component

    # Apply: H[A] = IFFT(-i * sign * FFT(A))
    F_H = F_A * (-1j * sign)
    hilbert_A = torch.fft.irfft(F_H, n=n, dim=-1)

    return hilbert_A


def kramers_kronig_loss(A: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Kramers-Kronig consistency loss.

    The retarded Green's function G(omega) must be analytic in the upper half
    complex plane (causality). This forces Re[G] and Im[G] to be Hilbert
    transform pairs. Since Im[G] = -pi * A(omega), Re[G] is fully determined
    by A. We enforce this by computing Re[G] two ways and penalising disagreement:

      Method 1: Re[G](omega) = H[-pi * A](omega)  (direct Hilbert transform)
      Method 2: Re[G](omega) = P.V. integral A(omega') / (omega - omega') d(omega')
                             (principal value integral, discretised)

    In practice we use the FFT Hilbert transform (Method 1) for efficiency
    and compute the principal value integral (Method 2) as the reference.
    The loss penalises their difference.

    Args:
        A:     (batch, N_OMEGA) spectral function
        omega: (N_OMEGA,) energy axis
    Returns:
        Scalar KK penalty
    """
    d_omega = omega[1] - omega[0]

    # Method 1: Re[G] via FFT Hilbert transform of A
    # Im[G] = -pi * A => Re[G] = H[Im[G]] = -pi * H[A]  (wait: KK says Re[G] = -(1/pi)*H[Im[G]])
    # Re[G](omega) = (1/pi) P.V. integral Im[G(omega')]/(omega-omega') domega'
    #              = (1/pi) P.V. integral (-pi*A(omega'))/(omega-omega') domega'
    #              = -P.V. integral A(omega')/(omega-omega') domega'
    #              = -H[A](omega)  (definition of Hilbert transform)
    ReG_fft = -hilbert_transform(A)

    # Method 2: Direct principal value integral (reference, batched matrix form)
    # Only computed on a subset of omega points for efficiency
    step = max(1, config.N_OMEGA // 128)  # 128 reference points (was 64) for better KK accuracy
    omega_sub = omega[::step]  # (M,)

    # Difference matrix: omega_i - omega_j, with diagonal regularised
    diff = omega_sub.unsqueeze(1) - omega.unsqueeze(0)  # (M, N_OMEGA)
    eps  = d_omega * 0.5
    diff_reg = torch.where(diff.abs() < eps, torch.full_like(diff, float('inf')), diff)

    # P.V. integral: sum_j A(omega_j) / (omega_i - omega_j) * d_omega
    # A shape: (batch, N_OMEGA), diff_reg: (M, N_OMEGA)
    ReG_pv = -torch.matmul(A, (1.0 / diff_reg).T) * d_omega  # (batch, M)
    ReG_fft_sub = ReG_fft[:, ::step]  # (batch, M)

    return F.mse_loss(ReG_fft_sub, ReG_pv.detach())


def physics_loss(
    A: torch.Tensor,
    omega: torch.Tensor,
    use_kk: bool = config.USE_KK,
    kk_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute all physics constraint loss terms.

    Args:
        A:        (batch, N_OMEGA) predicted spectral function (post constraint layer)
        omega:    (N_OMEGA,) energy axis
        use_kk:   Whether to include KK loss (can be toggled in config)
        kk_scale: Multiplicative scale on the KK term (used for warmup scheduling)
    Returns:
        Dict of individual loss terms (for logging) and 'total' weighted sum
    """
    losses: Dict[str, torch.Tensor] = {}

    losses["normalisation"] = normalisation_loss(A, omega) * config.LAMBDA_NORM
    losses["smoothness"]    = smoothness_loss(A, omega)    * config.LAMBDA_SMOOTH

    if use_kk:
        losses["kramers_kronig"] = kramers_kronig_loss(A, omega) * config.LAMBDA_KK * kk_scale
    else:
        losses["kramers_kronig"] = torch.tensor(0.0, device=A.device)

    losses["total"] = sum(losses.values())
    return losses

