# visualisation/plots.py
# All plotting functions for the project.
#
# Key plots:
#   1. Example spectral functions across physical regimes
#   2. Predicted vs reference for representative test samples
#   3. Error distribution across test set
#   4. Mott transition evolution — the physically most important plot
#   5. Training loss curves (train + val) for all models
#
# All multi-model functions accept dicts keyed by model name so they
# scale cleanly to any number of models.

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

os.makedirs(config.RESULTS_DIR, exist_ok=True)

COLOURS = {
    "MLP":             "#2E75B6",
    "DeepONet":        "#E05A2B",
    "FourierNet":      "#7B2FBE",
    "reference":       "#2C2C2C",
    "metallic":        "#1A7A3C",
    "correlated":      "#E8A020",
    "mott_insulating": "#B81C1C",
}

# Fallback palette for any additional models
_EXTRA_COLOURS = ["#1B998B", "#FF6B35", "#FFBE0B", "#3A86FF"]


def _model_colour(name: str, idx: int = 0) -> str:
    if name in COLOURS:
        return COLOURS[name]
    return _EXTRA_COLOURS[idx % len(_EXTRA_COLOURS)]


def plot_dataset_examples(omega: torch.Tensor, data: Dict, save: bool = True):
    """
    Plot example spectral functions from each physical regime
    to visualise what the model is learning.
    """
    omega_np  = omega.numpy()
    spectra   = data["spectra"].numpy()
    params    = data["params_raw"].numpy()
    regimes   = data["regimes"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    regime_list = ["metallic", "correlated", "mott_insulating"]
    titles      = ["Metallic (U/W < 0.8)", "Correlated Metal (0.8 < U/W < 1.2)", "Mott Insulating (U/W > 1.2)"]

    for ax, regime, title in zip(axes, regime_list, titles):
        indices = [i for i, r in enumerate(regimes) if r == regime][:5]
        for idx in indices:
            U, W, n = params[idx]
            ax.plot(omega_np, spectra[idx], alpha=0.7, lw=1.5,
                    label=f"U={U:.1f}, W={W:.1f}, n={n:.2f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Energy ω")
        ax.set_ylabel("A(ω)")
        ax.legend(fontsize=7)
        ax.set_xlim(config.OMEGA_MIN, config.OMEGA_MAX)
        ax.grid(alpha=0.3)

    plt.suptitle("Synthetic Spectral Functions by Physical Regime", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(f"{config.RESULTS_DIR}dataset_examples.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {config.RESULTS_DIR}dataset_examples.png")
        plt.close(fig)


def plot_predictions(
    omega: torch.Tensor,
    results: Dict[str, Dict],
    n_examples: int = 6,
    save: bool = True,
):
    """
    Predicted vs reference spectral functions for representative test samples.
    Args:
        results: {model_name: results_dict}
    """
    omega_np = omega.numpy()
    # Use regimes from the first model (all share the same test set)
    first = next(iter(results.values()))
    regimes = first["regimes"]

    examples = []
    for regime in ["metallic", "correlated", "mott_insulating"]:
        idxs = [i for i, r in enumerate(regimes) if r == regime][:2]
        examples.extend(idxs)
    examples = examples[:n_examples]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, examples):
        regime = regimes[idx]
        ref    = first["targets"][idx].numpy()
        ax.plot(omega_np, ref, color=COLOURS["reference"], lw=2.0, label="Reference", zorder=3)

        for i, (name, res) in enumerate(results.items()):
            pred = res["predictions"][idx].numpy()
            ax.plot(omega_np, pred, color=_model_colour(name, i),
                    lw=1.5, linestyle=["--", ":", "-."][i % 3],
                    label=name, alpha=0.9)

        ax.set_title(f"Regime: {regime.replace('_', ' ').title()}", fontsize=10)
        ax.set_xlabel("Energy ω")
        ax.set_ylabel("A(ω)")
        ax.legend(fontsize=8)
        ax.set_xlim(config.OMEGA_MIN, config.OMEGA_MAX)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    plt.suptitle("Predicted vs Reference Spectral Functions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(f"{config.RESULTS_DIR}predictions.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {config.RESULTS_DIR}predictions.png")
        plt.close(fig)


def plot_mott_transition(
    models: Dict,
    omega: torch.Tensor,
    params_min: torch.Tensor,
    params_max: torch.Tensor,
    save: bool = True,
):
    """
    THE key physics plot: evolution of A(omega) across the Mott transition.
    Args:
        models: {model_name: nn.Module}
    """
    for m in models.values():
        m.eval()

    W_fixed = 2.0
    n_fixed = 1.0
    U_vals  = np.linspace(config.U_MIN, config.U_MAX * 0.95, 10)
    omega_np = omega.numpy()

    def normalise(U, W, n):
        p      = torch.tensor([[U, W, n]], dtype=torch.float32)
        p_norm = (p - params_min) / (params_max - params_min + 1e-8)
        return p_norm.to(config.DEVICE)

    from generator import generate_spectral_function
    ref_spectra = [generate_spectral_function(U, W_fixed, n_fixed, omega_np) for U in U_vals]

    n_cols  = 1 + len(models)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    cmap    = plt.cm.coolwarm
    colours = [cmap(i / (len(U_vals) - 1)) for i in range(len(U_vals))]

    with torch.no_grad():
        for i, (U, ref) in enumerate(zip(U_vals, ref_spectra)):
            ratio = U / W_fixed
            label = f"U/W={ratio:.2f}"
            p_norm = normalise(U, W_fixed, n_fixed)

            axes[0].plot(omega_np, ref, color=colours[i], lw=1.5, label=label)

            for j, (name, model) in enumerate(models.items()):
                pred = model(p_norm).cpu().numpy()[0]
                axes[j + 1].plot(omega_np, pred, color=colours[i], lw=1.5, label=label)

    col_titles = ["Reference (Ground Truth)"] + list(models.keys())
    for ax, title in zip(axes, col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Energy ω")
        ax.set_ylabel("A(ω)")
        ax.set_xlim(config.OMEGA_MIN, config.OMEGA_MAX)
        ax.set_ylim(bottom=0)
        ax.axvline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(U_vals[0] / W_fixed, U_vals[-1] / W_fixed))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Correlation strength U/W", fontsize=10)

    plt.suptitle("Spectral Function Evolution Across the Mott Transition\n(W=2.0, n=1.0 fixed)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(f"{config.RESULTS_DIR}mott_transition.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {config.RESULTS_DIR}mott_transition.png")
        plt.close(fig)


def plot_training_curves(
    histories: Dict[str, Dict[str, List[Dict]]],
    save: bool = True,
):
    """
    Training and validation loss curves for all models, showing both train and val.
    Args:
        histories: {model_name: {"train": [...], "val": [...]}}
    """
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (i, (name, hist)) in zip(axes, enumerate(histories.items())):
        colour    = _model_colour(name, i)
        train_h   = hist["train"]
        val_h     = hist["val"]
        epochs    = range(1, len(val_h) + 1)

        ax.plot(epochs, [h["total"]          for h in val_h],   color=colour,    lw=2.0, label="Val total")
        ax.plot(epochs, [h["total"]          for h in train_h], color=colour,    lw=1.5, linestyle="--", alpha=0.6, label="Train total")
        ax.plot(epochs, [h["kramers_kronig"] for h in val_h],   color="purple",  lw=1.2, linestyle=":", alpha=0.8, label="KK (val)")
        ax.plot(epochs, [h["reconstruction"] for h in val_h],   color="#888888", lw=1.2, linestyle="-.", alpha=0.7, label="Recon (val)")

        ax.set_title(f"{name} Training History", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    if save:
        plt.savefig(f"{config.RESULTS_DIR}training_curves.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {config.RESULTS_DIR}training_curves.png")
        plt.close(fig)


def plot_error_distribution(
    omega: torch.Tensor,
    results: Dict[str, Dict],
    save: bool = True,
):
    """
    Pointwise mean absolute error distribution across the test set for all models.
    Args:
        results: {model_name: results_dict}
    """
    omega_np = omega.numpy()

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, res) in enumerate(results.items()):
        err  = torch.abs(res["predictions"] - res["targets"]).numpy()
        mean = err.mean(axis=0)
        std  = err.std(axis=0)
        colour = _model_colour(name, i)
        ax.plot(omega_np, mean, color=colour, lw=2, label=f"{name} mean |error|")
        ax.fill_between(omega_np, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("Energy ω")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Pointwise Error Distribution Across Test Set", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(config.OMEGA_MIN, config.OMEGA_MAX)

    plt.tight_layout()
    if save:
        plt.savefig(f"{config.RESULTS_DIR}error_distribution.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {config.RESULTS_DIR}error_distribution.png")
        plt.close(fig)
