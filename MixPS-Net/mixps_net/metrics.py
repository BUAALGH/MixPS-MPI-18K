"""Per-sample, per-harmonic PSNR and SSIM."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity


def sample_metrics(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> list[dict[str, float]]:
    """Average channel-wise metrics for every sample in a batch.

    Each harmonic uses the corresponding target channel's dynamic range.
    This definition is scale invariant under the shared M-derived scaling.
    """
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    results = []
    for predicted_sample, target_sample in zip(prediction_np, target_np):
        psnr_values, ssim_values = [], []
        for predicted_channel, target_channel in zip(
            predicted_sample, target_sample
        ):
            data_range = max(
                float(target_channel.max() - target_channel.min()), eps
            )
            mse = float(np.mean((predicted_channel - target_channel) ** 2))
            psnr_values.append(
                10.0 * np.log10((data_range * data_range) / max(mse, eps))
            )
            ssim_values.append(
                structural_similarity(
                    target_channel,
                    predicted_channel,
                    data_range=data_range,
                )
            )
        results.append(
            {
                "psnr": float(np.mean(psnr_values)),
                "ssim": float(np.mean(ssim_values)),
            }
        )
    return results


def summarize(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("cannot summarize an empty record list")
    summary = {}
    for key in ("psnr", "ssim"):
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary
