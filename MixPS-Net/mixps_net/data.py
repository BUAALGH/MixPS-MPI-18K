"""HDF5 input pipeline for the public MixPS-MPI-18K release."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SUBSETS = (
    "MixPS-Simulation-7K",
    "MixPS-Synthetic-10K",
    "MixPS-Realworld-41",
)


def _read_harmonics(h5: h5py.File, group: str) -> np.ndarray:
    real = h5[f"{group}/image/real"][...]
    imag = h5[f"{group}/image/imag"][...]
    if real.shape != (6, 64, 64) or imag.shape != (6, 64, 64):
        raise ValueError(
            f"expected real/imag [6,64,64], got {real.shape}/{imag.shape}"
        )
    return np.concatenate((real, imag), axis=0).astype(np.float32, copy=False)


class MixPSDataset(Dataset):
    """Load paired M/P/S samples and apply mixture-derived MaxAbs scaling."""

    def __init__(
        self,
        data_root: str | Path,
        subset: str,
        split: str,
        augment: bool = False,
        realworld_transpose: bool = True,
        include_simulation_stress_test: bool = False,
        eps: float = 1e-8,
    ) -> None:
        if subset not in SUBSETS:
            raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSETS}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.subset = subset
        self.split = split
        self.directory = Path(data_root) / subset / split
        if not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        self.files = sorted(self.directory.glob("mix_*.h5"))
        if (
            subset == "MixPS-Simulation-7K"
            and split == "test"
            and not include_simulation_stress_test
        ):
            self.files = [
                path for path in self.files if path.name != "mix_00701.h5"
            ]
        if not self.files:
            raise RuntimeError(f"no mix_*.h5 files found in {self.directory}")
        self.augment = bool(augment and split == "train")
        self.transpose = bool(
            realworld_transpose and subset == "MixPS-Realworld-41"
        )
        self.eps = eps

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _augment(arrays: Iterable[np.ndarray]) -> list[np.ndarray]:
        arrays = list(arrays)
        turns = random.randrange(4)
        arrays = [np.rot90(x, turns, axes=(-2, -1)).copy() for x in arrays]
        if random.random() < 0.5:
            arrays = [np.flip(x, axis=-1).copy() for x in arrays]
        if random.random() < 0.5:
            arrays = [np.flip(x, axis=-2).copy() for x in arrays]
        return arrays

    def __getitem__(self, index: int) -> dict:
        path = self.files[index]
        with h5py.File(path, "r") as h5:
            mixed = _read_harmonics(h5, "mix")
            perimag = _read_harmonics(h5, "P")
            synomag = _read_harmonics(h5, "S")
        if self.transpose:
            mixed = mixed.transpose(0, 2, 1).copy()
            perimag = perimag.transpose(0, 2, 1).copy()
            synomag = synomag.transpose(0, 2, 1).copy()
        if self.augment:
            mixed, perimag, synomag = self._augment(
                (mixed, perimag, synomag)
            )
        scale = max(float(np.max(np.abs(mixed))), self.eps)
        return {
            "M": torch.from_numpy(mixed / scale),
            "P": torch.from_numpy(perimag / scale),
            "S": torch.from_numpy(synomag / scale),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "sample_id": path.stem,
            "subset": self.subset,
        }


def seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python from the deterministic PyTorch worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
