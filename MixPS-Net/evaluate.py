#!/usr/bin/env python3
"""Evaluate a MixPS-Net checkpoint on all three public test subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mixps_net.data import MixPSDataset
from mixps_net.metrics import sample_metrics, summarize
from mixps_net.runtime import build_model, save_json, set_reproducibility


@torch.inference_mode()
def evaluate_subset(model, loader, device) -> dict:
    records = {"Perimag": [], "Synomag": []}
    for batch in tqdm(loader, desc=loader.dataset.subset):
        mixed = batch["M"].to(device, non_blocking=True)
        outputs = model(mixed)
        records["Perimag"].extend(sample_metrics(outputs["P_hat"], batch["P"]))
        records["Synomag"].extend(sample_metrics(outputs["S_hat"], batch["S"]))
    return {
        particle: summarize(values) for particle, values in records.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    set_reproducibility(config["training"]["seed"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    results = {}
    for subset in config["data"]["test_subsets"]:
        dataset = MixPSDataset(
            args.data_root,
            subset,
            "test",
            augment=False,
            realworld_transpose=config["data"]["realworld_transpose"],
            include_simulation_stress_test=config["data"][
                "include_simulation_stress_test"
            ],
        )
        loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            pin_memory=device.type == "cuda",
        )
        results[subset] = evaluate_subset(model, loader, device)
    save_json(results, args.output)
    for subset, particles in results.items():
        print(subset)
        for particle, metrics in particles.items():
            psnr, ssim = metrics["psnr"], metrics["ssim"]
            print(
                f"  {particle:8s} PSNR {psnr['mean']:.4f} +/- {psnr['std']:.4f}  "
                f"SSIM {ssim['mean']:.4f} +/- {ssim['std']:.4f}"
            )


if __name__ == "__main__":
    main()
