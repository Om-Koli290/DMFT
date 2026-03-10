# main.py
# Runs the full pipeline end to end.
# Usage: python main.py
#        python main.py --regen          (force regenerate dataset)
#        python main.py --no-kk          (disable KK constraint for comparison)
#        python main.py --eval-only      (skip training, load saved checkpoints)
#        python main.py --model mlp      (train only the MLP)
#        python main.py --model deeponet (train only the DeepONet)

import argparse
import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from data import get_dataloaders, get_test_indices
from mlp import SpectralMLP
from deeponet import SpectralDeepONet
from trainer import Trainer
from metrics import evaluate_model, print_results, compare_models
from plots import (
    plot_dataset_examples,
    plot_predictions,
    plot_mott_transition,
    plot_training_curves,
    plot_error_distribution,
)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)


def load_checkpoint(model, path):
    """Load a model checkpoint with the correct device mapping."""
    ckpt = torch.load(path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(config.DEVICE)


def main(args):
    print("\n" + "="*60)
    print("  Physics-Informed ML for Spectral Function Prediction")
    print("="*60)
    print(f"  Device    : {config.DEVICE}")
    print(f"  KK loss   : {'ON' if config.USE_KK else 'OFF'}")
    print(f"  Samples   : {config.N_SAMPLES}")
    print(f"  Epochs    : {config.N_EPOCHS}")
    print(f"  KK warmup : {config.KK_WARMUP_EPOCHS} epochs")
    print("="*60 + "\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, data = get_dataloaders(
        force_regenerate=args.regen
    )
    omega = data["omega"].to(config.DEVICE)

    # Derive test indices once — shared by all models
    test_indices = get_test_indices(len(data["params"]))

    # Visualise dataset examples
    plot_dataset_examples(data["omega"], data)

    # ── 2. Build models ───────────────────────────────────────────────────────
    mlp = SpectralMLP(omega)
    don = SpectralDeepONet(omega)

    print(f"\nModel sizes:")
    print(f"  MLP      : {mlp.count_parameters():,} parameters")
    print(f"  DeepONet : {don.count_parameters():,} parameters")

    train_all = args.model is None
    histories = {}

    # ── 3. Train ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        if train_all or args.model == "mlp":
            trainer_mlp = Trainer(mlp, omega, config.MLP_CHECKPOINT, "MLP",
                                  lr=config.LEARNING_RATE)
            trainer_mlp.fit(train_loader, val_loader)
            histories["MLP"] = {"train": trainer_mlp.train_history, "val": trainer_mlp.val_history}
        else:
            mlp = load_checkpoint(mlp, config.MLP_CHECKPOINT)

        if train_all or args.model == "deeponet":
            trainer_don = Trainer(don, omega, config.DEEPONET_CHECKPOINT, "DeepONet",
                                  lr=config.DEEPONET_LR)
            trainer_don.fit(train_loader, val_loader)
            histories["DeepONet"] = {"train": trainer_don.train_history, "val": trainer_don.val_history}
        else:
            don = load_checkpoint(don, config.DEEPONET_CHECKPOINT)

        if histories:
            plot_training_curves(histories)

    else:
        print("Skipping training — loading saved checkpoints...")
        mlp = load_checkpoint(mlp, config.MLP_CHECKPOINT)
        don = load_checkpoint(don, config.DEEPONET_CHECKPOINT)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    all_results = {
        "MLP":      evaluate_model(mlp, test_loader, omega, data["regimes"], test_indices),
        "DeepONet": evaluate_model(don, test_loader, omega, data["regimes"], test_indices),
    }

    for name, res in all_results.items():
        print_results(name, res)

    compare_models(all_results)

    # ── 5. Visualise ──────────────────────────────────────────────────────────
    plot_predictions(data["omega"], all_results)
    plot_error_distribution(data["omega"], all_results)
    plot_mott_transition(
        {"MLP": mlp, "DeepONet": don},
        data["omega"],
        data["params_min"],
        data["params_max"],
    )

    print(f"\nAll results saved to {config.RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen",     action="store_true", help="Force regenerate dataset")
    parser.add_argument("--no-kk",     action="store_true", help="Disable KK constraint")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, load checkpoints")
    parser.add_argument("--model",     type=str, choices=["mlp", "deeponet"],
                        help="Train a single model only (other loaded from checkpoint)")
    args = parser.parse_args()

    if args.no_kk:
        config.USE_KK = False

    main(args)
