"""MixPS-Net: particle separation with operator--concentration factorization.

Tensor flow (C is the base width):
    M [B,K,H,W]
    -> HFE F [B,2C,H,W]
    -> particle-specific MHA FP/FS [B,2C,H,W]
    -> OC-Factorizer ZC/ZH [B,C,H,W]
    -> concentration c [B,1,H,W] and operators H [B,K,L,L]
    -> HCMM rendering P/S/M [B,K,H,W].
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SpatialResidualBlock(nn.Module):
    """Reflection-padded spatial residual block preserving signed features."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                padding_mode="reflect",
            ),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
            ),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PointwiseResidualBlock(nn.Module):
    """Channel-only residual block for the concentration factor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HarmonicFeatureEncoder(nn.Module):
    """Stage 1: K -> C -> 2C pointwise mixing and two residual blocks."""

    def __init__(self, harmonics: int, base_channels: int) -> None:
        super().__init__()
        width = 2 * base_channels
        self.harmonic_mixer = nn.Sequential(
            nn.Conv2d(harmonics, base_channels, kernel_size=1),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, width, kernel_size=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
        )
        self.refinement = nn.Sequential(
            SpatialResidualBlock(width),
            SpatialResidualBlock(width),
        )

    def forward(self, mixed: torch.Tensor) -> torch.Tensor:
        return self.refinement(self.harmonic_mixer(mixed))


class ParticleAttention(nn.Module):
    """One standard MHA branch assigned to one particle identity."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by attention heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.q = nn.Linear(channels, channels, bias=True)
        self.k = nn.Linear(channels, channels, bias=True)
        self.v = nn.Linear(channels, channels, bias=True)
        self.output = nn.Linear(channels, channels, bias=True)
        self.norm = nn.LayerNorm(channels)

    def _split(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        return tokens.reshape(
            batch, count, self.heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(tokens))
        k = self._split(self.k(tokens))
        v = self._split(self.v(tokens))
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[1], self.channels
        )
        value_residual = v.transpose(1, 2).reshape_as(attended)
        return self.norm(self.output(attended) + value_residual)


class ParticleFeatureSeparation(nn.Module):
    """Stage 2: independent P- and S-specific multi-head attention."""

    def __init__(self, channels: int, heads: int = 2) -> None:
        super().__init__()
        self.p_attention = ParticleAttention(channels, heads)
        self.s_attention = ParticleAttention(channels, heads)

    @staticmethod
    def _restore(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], height, width
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        fp = self._restore(self.p_attention(tokens), height, width)
        fs = self._restore(self.s_attention(tokens), height, width)
        return fp, fs


class OCFactorizer(nn.Module):
    """Stage 3: learn independent concentration and operator subspaces."""

    def __init__(self, in_channels: int, latent_channels: int) -> None:
        super().__init__()

        def adapter() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    latent_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                ),
                nn.GroupNorm(_groups(latent_channels), latent_channels),
                nn.GELU(),
            )

        self.concentration_adapter = adapter()
        self.operator_adapter = adapter()
        self.concentration_encoder = nn.Sequential(
            PointwiseResidualBlock(latent_channels),
            PointwiseResidualBlock(latent_channels),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
        )
        self.operator_encoder = nn.Sequential(
            SpatialResidualBlock(latent_channels, dilation=2),
            SpatialResidualBlock(latent_channels, dilation=4),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zc = self.concentration_encoder(self.concentration_adapter(features))
        zh = self.operator_encoder(self.operator_adapter(features))
        return zc, zh


class ConcentrationDecoder(nn.Module):
    """Decode one nonnegative concentration map shared by all harmonics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refinement = SpatialResidualBlock(channels)
        self.projection = nn.Conv2d(channels, 1, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.activation(self.projection(self.refinement(latent)))


class HarmonicOperatorDecoder(nn.Module):
    """Decode a global descriptor into K signed LxL spatial operators."""

    def __init__(
        self,
        channels: int,
        harmonics: int,
        operator_size: int = 21,
        seed_size: int = 5,
    ) -> None:
        super().__init__()
        if operator_size % 2 != 1:
            raise ValueError("operator_size must be odd")
        self.harmonics = harmonics
        self.operator_size = operator_size
        self.seed_size = seed_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(
            channels, harmonics * seed_size * seed_size
        )
        self.refinement = nn.Conv2d(
            harmonics,
            harmonics,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            groups=harmonics,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        descriptor = self.pool(latent).flatten(1)
        seeds = self.projection(descriptor).reshape(
            batch, self.harmonics, self.seed_size, self.seed_size
        )
        operators = F.interpolate(
            seeds,
            size=(self.operator_size, self.operator_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.refinement(operators)


class HCMMRenderer(nn.Module):
    """Stage 4: sample-dependent grouped convolution H_t^k * c_t."""

    def __init__(
        self, harmonics: int, operator_size: int, padding_mode: str = "reflect"
    ) -> None:
        super().__init__()
        if padding_mode not in {"reflect", "zero"}:
            raise ValueError("padding_mode must be 'reflect' or 'zero'")
        self.harmonics = harmonics
        self.operator_size = operator_size
        self.padding_mode = padding_mode

    def forward(
        self, concentration: torch.Tensor, operators: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width = concentration.shape
        if operators.shape[:2] != (batch, self.harmonics):
            raise ValueError("operator batch/harmonic dimensions do not match")
        pad = self.operator_size // 2
        fields = concentration.expand(-1, self.harmonics, -1, -1)
        fields = fields.reshape(1, batch * self.harmonics, height, width)
        mode = "constant" if self.padding_mode == "zero" else "reflect"
        fields = F.pad(fields, (pad, pad, pad, pad), mode=mode)
        kernels = operators.reshape(
            batch * self.harmonics, 1, self.operator_size, self.operator_size
        )
        rendered = F.conv2d(fields, kernels, groups=batch * self.harmonics)
        return rendered.reshape(batch, self.harmonics, height, width)


class ParticleBranch(nn.Module):
    """Tracer-specific OC-Factorizer and physical decoders."""

    def __init__(
        self,
        feature_channels: int,
        latent_channels: int,
        harmonics: int,
        operator_size: int,
        seed_size: int,
        renderer_padding: str,
    ) -> None:
        super().__init__()
        self.factorizer = OCFactorizer(feature_channels, latent_channels)
        self.concentration_decoder = ConcentrationDecoder(latent_channels)
        self.operator_decoder = HarmonicOperatorDecoder(
            latent_channels, harmonics, operator_size, seed_size
        )
        self.renderer = HCMMRenderer(
            harmonics, operator_size, renderer_padding
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        zc, zh = self.factorizer(features)
        concentration = self.concentration_decoder(zc)
        operators = self.operator_decoder(zh)
        reconstruction = self.renderer(concentration, operators)
        return {
            "zc": zc,
            "zh": zh,
            "concentration": concentration,
            "operators": operators,
            "reconstruction": reconstruction,
        }


class MixPSNet(nn.Module):
    """Four-stage MixPS-Net shown in the public architecture figure."""

    def __init__(
        self,
        num_harmonics: int = 12,
        base_channels: int = 32,
        attention_heads: int = 2,
        operator_size: int = 21,
        operator_seed_size: int = 5,
        renderer_padding: str = "reflect",
    ) -> None:
        super().__init__()
        particle_channels = 2 * base_channels
        self.hfe = HarmonicFeatureEncoder(num_harmonics, base_channels)
        self.pfd = ParticleFeatureSeparation(
            particle_channels, attention_heads
        )
        branch_args = (
            particle_channels,
            base_channels,
            num_harmonics,
            operator_size,
            operator_seed_size,
            renderer_padding,
        )
        self.perimag_branch = ParticleBranch(*branch_args)
        self.synomag_branch = ParticleBranch(*branch_args)

    def forward(self, mixed: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.hfe(mixed)
        fp, fs = self.pfd(features)
        p = self.perimag_branch(fp)
        s = self.synomag_branch(fs)
        p_hat = p["reconstruction"]
        s_hat = s["reconstruction"]
        return {
            "P_hat": p_hat,
            "S_hat": s_hat,
            "M_hat": p_hat + s_hat,
            "F": features,
            "FP": fp,
            "FS": fs,
            "ZcP": p["zc"],
            "ZhP": p["zh"],
            "ZcS": s["zc"],
            "ZhS": s["zh"],
            "cP": p["concentration"],
            "cS": s["concentration"],
            "HP": p["operators"],
            "HS": s["operators"],
        }
