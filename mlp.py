# models/mlp.py
# Baseline MLP model: maps parameter vector (U, W, n) -> A(omega).
#
# Architecture: fully connected network with residual connections.
# Treats A(omega) as a fixed-length vector — straightforward but effective.
# Serves as the baseline against which DeepONet is compared.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from constraints import PhysicsConstraintLayer


class ResidualBlock(nn.Module):
    """Single residual block: Linear -> LayerNorm -> GELU -> Dropout -> Linear + skip."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class SpectralMLP(nn.Module):
    """
    MLP that predicts A(omega) as a fixed-length vector from (U, W, n).

    Architecture:
        Input (3) -> Projection -> Residual blocks -> Output (N_OMEGA)
        Physics constraint layer enforces positivity and normalisation.

    Strengths: simple, fast, interpretable failure modes.
    Limitations: treats A(omega) as a vector, not a function — no inductive
                 bias for continuity across the omega axis.
    """
    def __init__(
        self,
        omega: torch.Tensor,
        input_dim:   int       = config.INPUT_DIM,
        hidden_dims: List[int] = config.MLP_HIDDEN_DIMS,
        output_dim:  int       = config.N_OMEGA,
        dropout:     float     = config.MLP_DROPOUT,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
        )

        # Hidden layers with residual connections where dimensions match
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_dim  = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            if in_dim == out_dim:
                layers.append(ResidualBlock(in_dim, dropout))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
        self.hidden = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], output_dim)

        # Physics constraint layer (positivity + normalisation)
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
        x = self.input_proj(params)
        x = self.hidden(x)
        x = self.output_proj(x)
        A = self.constraints(x)
        return A

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
