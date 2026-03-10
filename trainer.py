# training/trainer.py
# Unified training loop for both MLP and DeepONet.
# Both models are trained identically so comparisons are fair.
#
# Loss function:
#   L_total = L_reconstruction + L_physics
#   L_reconstruction = MSE(predicted A, reference A)
#   L_physics = weighted sum of normalisation, smoothness, KK losses
#
# KK warmup: LAMBDA_KK is ramped from 0 to its full value over
# KK_WARMUP_EPOCHS epochs, letting the model first fit the data before
# being constrained by causality.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from constraints import physics_loss


class Trainer:
    """
    Handles training, validation, checkpointing, and early stopping
    for any spectral function model.
    """
    def __init__(
        self,
        model:          nn.Module,
        omega:          torch.Tensor,
        checkpoint_path: str,
        model_name:     str = "model",
        lr:             float = config.LEARNING_RATE,
    ):
        self.model      = model.to(config.DEVICE)
        self.omega      = omega.to(config.DEVICE)
        self.ckpt_path  = checkpoint_path
        self.name       = model_name

        self.optimiser  = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            patience=config.LR_PATIENCE,
            factor=config.LR_FACTOR,
        )
        self.criterion  = nn.MSELoss()

        self.train_history = []
        self.val_history   = []
        self.best_val_recon = float("inf")  # checkpoint on reconstruction, not total
        self.epochs_no_improve = 0

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def _step(
        self,
        params: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 1,
    ) -> Tuple[torch.Tensor, Dict]:
        """Single forward pass and loss computation."""
        params  = params.to(config.DEVICE)
        targets = targets.to(config.DEVICE)

        predictions = self.model(params)

        # Reconstruction loss with quasiparticle-peak-focused weighting
        omega_weights = 1.0 + 12.0 * torch.exp(-0.5 * (self.omega / 0.3) ** 2)
        omega_weights = omega_weights / omega_weights.mean()
        recon_loss = torch.mean(omega_weights * (predictions - targets) ** 2)

        # KK warmup: ramp the KK scale from 0 → 1 over KK_WARMUP_EPOCHS
        kk_scale = min(1.0, epoch / max(1, config.KK_WARMUP_EPOCHS))

        # Physics constraint losses
        phys_losses = physics_loss(
            predictions, self.omega,
            use_kk=config.USE_KK,
            kk_scale=kk_scale,
        )

        total_loss = recon_loss + phys_losses["total"]

        loss_dict = {
            "reconstruction":  recon_loss.item(),
            "normalisation":   phys_losses["normalisation"].item(),
            "smoothness":      phys_losses["smoothness"].item(),
            "kramers_kronig":  phys_losses["kramers_kronig"].item(),
            "total":           total_loss.item(),
        }

        return total_loss, loss_dict

    def train_epoch(self, loader: DataLoader, epoch: int = 1) -> Dict:
        self.model.train()
        epoch_losses = {k: 0.0 for k in ["reconstruction", "normalisation", "smoothness", "kramers_kronig", "total"]}

        for params, targets in loader:
            self.optimiser.zero_grad()
            loss, loss_dict = self._step(params, targets, epoch=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]

        n = len(loader)
        return {k: v / n for k, v in epoch_losses.items()}

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader, epoch: int = 1) -> Dict:
        self.model.eval()
        epoch_losses = {k: 0.0 for k in ["reconstruction", "normalisation", "smoothness", "kramers_kronig", "total"]}

        for params, targets in loader:
            _, loss_dict = self._step(params, targets, epoch=epoch)
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k]

        n = len(loader)
        return {k: v / n for k, v in epoch_losses.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop with early stopping and checkpointing."""
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"  Parameters : {self.model.count_parameters():,}")
        print(f"  Device     : {config.DEVICE}")
        print(f"  Epochs     : {config.N_EPOCHS}")
        print(f"  KK loss    : {'ON' if config.USE_KK else 'OFF'}")
        print(f"  KK warmup  : {config.KK_WARMUP_EPOCHS} epochs")
        print(f"{'='*60}")

        for epoch in range(1, config.N_EPOCHS + 1):
            train_losses = self.train_epoch(train_loader, epoch=epoch)
            val_losses   = self.val_epoch(val_loader, epoch=epoch)

            self.train_history.append(train_losses)
            self.val_history.append(val_losses)

            self.scheduler.step(val_losses["reconstruction"])

            # Checkpoint on reconstruction loss only — KK warmup inflates total loss
            # over time, so using total would always pick the earliest epoch.
            if val_losses["reconstruction"] < self.best_val_recon:
                self.best_val_recon    = val_losses["reconstruction"]
                self.epochs_no_improve = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "val_loss":    val_losses["reconstruction"],
                }, self.ckpt_path)
            else:
                self.epochs_no_improve += 1

            # Logging
            if epoch % 10 == 0 or epoch == 1:
                kk_scale = min(1.0, epoch / max(1, config.KK_WARMUP_EPOCHS))
                print(
                    f"Epoch {epoch:4d}/{config.N_EPOCHS} | "
                    f"Train: {train_losses['total']:.5f} | "
                    f"Val: {val_losses['total']:.5f} | "
                    f"Recon: {val_losses['reconstruction']:.5f} | "
                    f"KK: {val_losses['kramers_kronig']:.5f} "
                    f"(scale={kk_scale:.2f})"
                )

            # Early stopping
            if self.epochs_no_improve >= config.EARLY_STOP:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {config.EARLY_STOP} epochs)")
                break

        print(f"\nTraining complete. Best val recon loss: {self.best_val_recon:.6f}")
        self.load_best()

    def load_best(self):
        """Load the best checkpoint."""
        ckpt = torch.load(self.ckpt_path, map_location=config.DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")
