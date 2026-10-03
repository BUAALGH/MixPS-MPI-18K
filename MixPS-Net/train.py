#!/usr/bin/env python3
"""Train MixPS-Net for the 100-epoch public reference protocol."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from mixps_net.data import MixPSDataset, seed_worker
from mixps_net.losses import reconstruction_loss
from mixps_net.runtime import build_model, load_config, save_json, set_reproducibility


def make_loader(
    data_root: Path,
    subsets: list[str],
    split: str,
    config: dict,
    shuffle: bool,
) -> DataLoader:
    data_config = config["data"]
    datasets = [
        MixPSDataset(
            data_root,
            subset,
            split,
            augment=data_config["augmentation"],
            realworld_transpose=data_config["realworld_transpose"],
            include_simulation_stress_test=data_config[
                "include_simulation_stress_test"
            ],
        )
        for subset in subsets
    ]
    generator = torch.Generator().manual_seed(config["training"]["seed"])
    return DataLoader(
        ConcatDataset(datasets),
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config["training"]["num_workers"] > 0,
    )


def move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("M", "P", "S")
    }


def run_epoch(
    model,
    loader,
    device,
    loss_config,
    optimizer=None,
    scaler=None,
    gradient_clip=1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "P", "S", "M")}
    samples = 0
    progress = tqdm(loader, desc="train" if training else "validation")
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in progress:
            batch = move_batch(raw_batch, device)
            batch_size = batch["M"].shape[0]
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = scaler is not None and scaler.is_enabled()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(batch["M"])
                loss, components = reconstruction_loss(
                    outputs, batch, **loss_config
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            for key in totals:
                totals[key] += components[key] * batch_size
            samples += batch_size
            progress.set_postfix(loss=f"{components['total']:.5f}")
    return {key: value / samples for key, value in totals.items()}


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mixps_net.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mixps_net"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, args.output_dir / "config.yaml")
    set_reproducibility(config["training"]["seed"])
    device = torch.device(args.device)

    train_loader = make_loader(
        args.data_root,
        config["data"]["train_subsets"],
        "train",
        config,
        shuffle=True,
    )
    val_loader = make_loader(
        args.data_root,
        config["data"]["val_subsets"],
        "val",
        config,
        shuffle=False,
    )
    model = build_model(config).to(device)
    training = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training["epochs"],
        eta_min=training["min_learning_rate"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(training["amp"] and device.type == "cuda")
    )
    start_epoch, best_validation, history = 1, float("inf"), []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation"]
        history = checkpoint.get("history", [])

    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(start_epoch, training["epochs"] + 1):
        print(f"\nEpoch {epoch}/{training['epochs']}")
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config["loss"],
            optimizer,
            scaler,
            training["gradient_clip"],
        )
        val_metrics = run_epoch(
            model, val_loader, device, config["loss"]
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        save_json(history, args.output_dir / "history.json")
        improved = val_metrics["total"] < best_validation
        best_validation = min(best_validation, val_metrics["total"])
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation": best_validation,
            "config": config,
            "history": history,
        }
        save_checkpoint(args.output_dir / "last.pt", state)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", state)
        print(
            f"train={train_metrics['total']:.6f} "
            f"val={val_metrics['total']:.6f} best={best_validation:.6f}"
        )


if __name__ == "__main__":
    main()
