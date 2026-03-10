# evaluation/metrics.py
# Quantitative evaluation of reconstruction accuracy and physical validity.
#
# Key design: metrics are computed per physical regime (metallic, correlated,
# Mott insulating) in addition to overall — this demonstrates physical
# understanding of where models succeed and struggle.
#
# New metrics vs original:
#   - R² score per regime (more interpretable than MSE for physics audiences)
#   - Quasiparticle peak height error (directly tracks the peak-underestimation problem)

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R² (coefficient of determination) between predictions and targets."""
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    if ss_tot.item() < 1e-12:
        return 1.0
    return (1.0 - ss_res / ss_tot).item()


def qp_peak_height(A: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    Estimate quasiparticle peak height: maximum of A(omega) within
    |omega| < QP_PEAK_WINDOW, close to the Fermi level.
    Returns shape (batch,).
    """
    mask = omega.abs() < config.QP_PEAK_WINDOW
    if mask.sum() == 0:
        return A.max(dim=-1).values
    return A[:, mask].max(dim=-1).values


@torch.no_grad()
def evaluate_model(
    model,
    test_loader: DataLoader,
    omega: torch.Tensor,
    regimes: List[str],
    test_indices: List[int],
) -> Dict:
    """
    Full evaluation of a trained model on the test set.

    Returns:
        Dict containing:
            - overall MSE, MAE, R²
            - MSE and R² per physical regime
            - mean normalisation error
            - mean positivity violation (should be 0 with constraint layer)
            - quasiparticle peak height MAE
            - predictions and targets for plotting
    """
    model.eval()
    omega = omega.to(config.DEVICE)
    d_omega = omega[1] - omega[0]

    all_preds   = []
    all_targets = []

    for params, targets in test_loader:
        params  = params.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        preds   = model(params)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    preds     = torch.cat(all_preds,   dim=0)  # (N_test, N_OMEGA)
    targets   = torch.cat(all_targets, dim=0)
    omega_cpu = omega.cpu()

    # ── Overall metrics ───────────────────────────────────────────────────────
    mse     = torch.mean((preds - targets) ** 2).item()
    mae     = torch.mean(torch.abs(preds - targets)).item()
    max_err = torch.max(torch.abs(preds - targets)).item()
    r2      = r2_score(preds, targets)

    # ── Normalisation error ───────────────────────────────────────────────────
    norms      = torch.trapezoid(preds, dx=d_omega.cpu(), dim=-1)
    norm_error = torch.mean(torch.abs(norms - 1.0)).item()

    # ── Positivity violations ─────────────────────────────────────────────────
    neg_frac = (preds < 0).float().mean().item()

    # ── Quasiparticle peak height error ──────────────────────────────────────
    pred_qp_h       = qp_peak_height(preds,   omega_cpu)
    target_qp_h     = qp_peak_height(targets, omega_cpu)
    qp_height_error = torch.mean(torch.abs(pred_qp_h - target_qp_h)).item()

    # ── Per-regime MSE and R² ─────────────────────────────────────────────────
    test_regimes = [regimes[i] for i in test_indices]
    regime_mse   = {}
    regime_r2    = {}
    for regime in ["metallic", "correlated", "mott_insulating"]:
        mask = torch.tensor([r == regime for r in test_regimes])
        if mask.sum() > 0:
            regime_mse[regime] = torch.mean((preds[mask] - targets[mask]) ** 2).item()
            regime_r2[regime]  = r2_score(preds[mask], targets[mask])
        else:
            regime_mse[regime] = float("nan")
            regime_r2[regime]  = float("nan")

    results = {
        "mse":             mse,
        "mae":             mae,
        "r2":              r2,
        "max_error":       max_err,
        "norm_error":      norm_error,
        "neg_fraction":    neg_frac,
        "qp_height_error": qp_height_error,
        "regime_mse":      regime_mse,
        "regime_r2":       regime_r2,
        "predictions":     preds,
        "targets":         targets,
        "regimes":         test_regimes,
    }

    return results


def print_results(name: str, results: Dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  Results: {name}")
    print(f"{'='*50}")
    print(f"  MSE (overall)        : {results['mse']:.6f}")
    print(f"  MAE (overall)        : {results['mae']:.6f}")
    print(f"  R² (overall)         : {results['r2']:.4f}")
    print(f"  Max error            : {results['max_error']:.6f}")
    print(f"  Normalisation error  : {results['norm_error']:.6f}")
    print(f"  Neg fraction         : {results['neg_fraction']:.2e}")
    print(f"  QP peak height MAE   : {results['qp_height_error']:.6f}")
    print(f"\n  Per-regime MSE / R²:")
    for regime in ["metallic", "correlated", "mott_insulating"]:
        mse = results["regime_mse"].get(regime, float("nan"))
        r2  = results["regime_r2"].get(regime, float("nan"))
        print(f"    {regime:<20s} : MSE={mse:.6f}  R²={r2:.4f}")


def compare_models(results: Dict[str, Dict]):
    """
    Side-by-side comparison of N models.
    Args:
        results: {model_name: results_dict}
    """
    names = list(results.keys())
    col_w = 13

    print(f"\n{'='*75}")
    print(f"  Model Comparison: {' vs '.join(names)}")
    print(f"{'='*75}")
    header = f"  {'Metric':<25}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print(f"  {'-'*72}")

    scalar_metrics = [
        ("MSE",           "mse"),
        ("MAE",           "mae"),
        ("R²",            "r2"),
        ("Max Error",     "max_error"),
        ("Norm Error",    "norm_error"),
        ("QP Height MAE", "qp_height_error"),
    ]
    for label, key in scalar_metrics:
        vals = [results[n][key] for n in names]
        best = max(vals) if key == "r2" else min(vals)
        row = f"  {label:<25}" + "".join(
            f"{v:>{col_w}.6f}" + ("*" if abs(v - best) < 1e-9 else " ")
            for v in vals
        )
        print(row)

    print(f"\n  Per-regime MSE:")
    for regime in ["metallic", "correlated", "mott_insulating"]:
        vals = [results[n]["regime_mse"].get(regime, float("nan")) for n in names]
        print(f"    {regime:<22}" + "".join(f"{v:>{col_w}.6f} " for v in vals))

    print(f"\n  Per-regime R²:")
    for regime in ["metallic", "correlated", "mott_insulating"]:
        vals = [results[n]["regime_r2"].get(regime, float("nan")) for n in names]
        print(f"    {regime:<22}" + "".join(f"{v:>{col_w}.4f} " for v in vals))
