"""Training objective used by the public MixPS-Net implementation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def reconstruction_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    alpha_p: float = 1.0,
    alpha_s: float = 1.0,
    alpha_m: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_p = F.l1_loss(outputs["P_hat"], targets["P"])
    loss_s = F.l1_loss(outputs["S_hat"], targets["S"])
    loss_m = F.l1_loss(outputs["M_hat"], targets["M"])
    total = alpha_p * loss_p + alpha_s * loss_s + alpha_m * loss_m
    return total, {
        "total": float(total.detach()),
        "P": float(loss_p.detach()),
        "S": float(loss_s.detach()),
        "M": float(loss_m.detach()),
    }
