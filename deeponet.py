# models/deeponet.py
# DeepONet-style operator learning model.
#
# Physics motivation:
#   A(omega) is a function, not a vector. DeepONet treats it as such by
#   learning a decomposition:
#
#       A(omega; params) ≈ sum_k  branch_k(params) * trunk_k(omega)
#
#   The branch net encodes the input parameters into a latent space.
#   The trunk net encodes the query point omega into the same space.
#   Their inner product gives the output at any omega.
#
#   This has two advantages over MLP:
#     1. It treats omega as a continuous variable — the model can in principle
#        evaluate at any energy, not just training grid points.
#     2. The trunk net learns a physically motivated basis for spectral functions,
#        analogous to a learned spectral decomposition.
#
#   Reference: Lu et al., "Learning nonlinear operators via DeepONet",
#              Nature Machine Intelligence, 2021.

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from constraints import PhysicsConstraintLayer


class BranchNet(nn.Module):
    """
    Encodes physical parameters (U, W, n) into the latent basis space.
    Output dimension must match TrunkNet output dimension.
    """
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: (batch, input_dim)
        Returns:
            b: (batch, output_dim) — latent encoding of parameters
        """
        return self.net(params)


class TrunkNet(nn.Module):
    """
    Encodes the query energy omega into the latent basis space.
    Learns a set of basis functions {phi_k(omega)} for spectral functions.
    Input is a single scalar omega; output is a vector of basis function values.
    """
    def __init__(self, hidden_dims: list, output_dim: int):
        super().__init__()
        dims = [1] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Args:
            omega: (N_OMEGA,) or (batch, N_OMEGA) energy axis
        Returns:
            t: (N_OMEGA, output_dim) — basis function values at each omega point
        """
        if omega.dim() == 1:
            omega = omega.unsqueeze(-1)  # (N_OMEGA, 1)
        return self.net(omega)


class SpectralDeepONet(nn.Module):
    """
    DeepONet operator learning model for spectral function prediction.

    Forward pass:
        1. Branch net encodes params -> b (batch, p)
        2. Trunk net encodes omega  -> t (N_OMEGA, p)
        3. Output: A_raw = t @ b.T + bias  -> (batch, N_OMEGA)
        4. Physics constraint layer: positivity + normalisation

    The model learns A(omega; params) as an inner product in a learned
    function space, giving it better inductive bias for smooth functional outputs.
    """
    def __init__(
        self,
        omega: torch.Tensor,
        input_dim:    int  = config.INPUT_DIM,
        branch_hidden: list = config.BRANCH_HIDDEN,
        trunk_hidden:  list = config.TRUNK_HIDDEN,
        basis_dim:     int  = config.DEEPONET_BASIS,
    ):
        super().__init__()

        self.branch = BranchNet(input_dim, branch_hidden, basis_dim)
        self.trunk  = TrunkNet(trunk_hidden, basis_dim)
        self.bias   = nn.Parameter(torch.zeros(config.N_OMEGA))

        # Register omega as a buffer so it moves to device with the model
        self.register_buffer("omega", omega)

        # Physics constraint layer
        self.constraints = PhysicsConstraintLayer(omega)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: (batch, 3) — normalised (U, W, n)
        Returns:
            A: (batch, N_OMEGA) — physically constrained spectral function
        """
        # Branch: encode parameters
        b = self.branch(params)          # (batch, basis_dim)

        # Trunk: evaluate basis functions at all omega points
        t = self.trunk(self.omega)       # (N_OMEGA, basis_dim)

        # Inner product: A_raw[i, j] = sum_k b[i,k] * t[j,k] + bias[j]
        A_raw = torch.matmul(b, t.T) + self.bias   # (batch, N_OMEGA)

        # Physics constraints
        A = self.constraints(A_raw)

        return A

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
