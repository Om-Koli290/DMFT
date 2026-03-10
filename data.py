# data/dataset.py
# Wraps generated data into PyTorch Dataset/DataLoader objects.
# Handles train/val/test splitting.

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from typing import List, Tuple, Dict
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from generator import generate_dataset


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for (parameters, spectral function) pairs.
    Each item is (param_vector, A(omega)).
    """
    def __init__(self, params: torch.Tensor, spectra: torch.Tensor):
        assert params.shape[0] == spectra.shape[0]
        self.params  = params
        self.spectra = spectra

    def __len__(self) -> int:
        return self.params.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.params[idx], self.spectra[idx]


def load_or_generate_data(force_regenerate: bool = False) -> Dict:
    """
    Load dataset from disk if it exists, otherwise generate and save it.
    Caches to config.DATA_PATH to avoid regenerating on every run.
    """
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)

    if os.path.exists(config.DATA_PATH) and not force_regenerate:
        print(f"Loading cached dataset from {config.DATA_PATH}")
        data = torch.load(config.DATA_PATH, weights_only=False)
    else:
        print("Generating new dataset...")
        data = generate_dataset()
        torch.save(data, config.DATA_PATH)
        print(f"Dataset saved to {config.DATA_PATH}")

    return data


def get_test_indices(n_total: int) -> List[int]:
    """
    Reproduce the identical train/val/test split used in get_dataloaders.
    Returns the test set indices — avoids duplicating split logic in main.py.
    """
    n_train = int(config.TRAIN_FRAC * n_total)
    n_val   = int(config.VAL_FRAC * n_total)
    n_test  = n_total - n_train - n_val
    dummy   = TensorDataset(torch.zeros(n_total))
    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    _, _, test_ds = random_split(dummy, [n_train, n_val, n_test], generator=generator)
    return list(test_ds.indices)


def get_dataloaders(
    force_regenerate: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Returns (train_loader, val_loader, test_loader, data_dict).
    data_dict contains omega axis and regime labels for evaluation.
    """
    data = load_or_generate_data(force_regenerate)

    dataset = SpectralDataset(data["params"], data["spectra"])
    n       = len(dataset)

    n_train = int(config.TRAIN_FRAC * n)
    n_val   = int(config.VAL_FRAC * n)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nDataloaders ready:")
    print(f"  Train : {len(train_ds)} samples")
    print(f"  Val   : {len(val_ds)} samples")
    print(f"  Test  : {len(test_ds)} samples")

    return train_loader, val_loader, test_loader, data
