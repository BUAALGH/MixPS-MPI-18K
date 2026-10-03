"""Configuration and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import MixPSNet


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["training"]["epochs"] != 100:
        raise ValueError("the public reference protocol uses exactly 100 epochs")
    return config


def build_model(config: dict) -> MixPSNet:
    return MixPSNet(**config["model"])


def set_reproducibility(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(payload: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
