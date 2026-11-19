"""Modern external neural operator baselines (2021-2026) for 2D RVE homogenization.

Includes:
  - FNO2dSmall (Li et al., ICLR 2021)
  - FFNO2d (Factorized FNO, Tran et al., ICLR 2023)
  - UNO2d wrapper (Rahman et al., TMLR 2023)
  - CNO2d (Convolutional Neural Operator w/ anti-aliased resampling, Raonic et al., NeurIPS 2023)
  - MgNO2d (Multigrid Neural Operator, He et al., ICLR 2024; reused from workspace)
  - Transolver2d (Physics-Attention Transformer, Wu et al., ICML 2024)
  - ResUNet2d (Strong multi-scale residual U-Net)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from uno_2d import UNO2d


def _apply_dirichlet_zero_v(v: torch.Tensor, lift: torch.Tensor) -> torch.Tensor:
    interior = torch.ones_like(v)
    interior[:, 0, :] = 0.0
    interior[:, -1, :] = 0.0
    interior[:, :, 0] = 0.0
    interior[:, :, -1] = 0.0
    return lift + v * interior


def _make_input_channels(kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    beta = (kappa - 1.0) / (kappa + 1.0)
    log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
    return torch.stack([beta, log_k, lift, ortho, dist], dim=1)


class FNO2dSmall(nn.Module):
    def __init__(self, in_ch: int = 5, width: int = 32, modes: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.convs = nn.ParameterList()
        self.ws = nn.ModuleList()
        scale = 1.0 / (width * width)
        for _ in range(depth):
            self.convs.append(nn.Parameter(scale * torch.randn(width, width, modes, modes, 2)))
            self.ws.append(nn.Conv2d(width, width, 1))
        self.fc1 = nn.Conv2d(width, 64, 1)
        self.fc2 = nn.Conv2d(64, 1, 1)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def _spec_conv(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        m = min(self.modes, H // 2, W // 2)
        wc = torch.complex(w[..., :m, :m, 0], w[..., :m, :m, 1])
        out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], wc)
        out_ft[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], wc)
        return torch.fft.irfft2(out_ft, s=(H, W))

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = _make_input_channels(kappa, lift, ortho, dist)
        h = self.fc0(x)
        for w, lin in zip(self.convs, self.ws):
            h = F.gelu(self._spec_conv(h, w) + lin(h))
        v = self.fc2(F.gelu(self.fc1(h)))[:, 0]
        return _apply_dirichlet_zero_v(v, lift)


class FactorizedSpectralLayer2d(nn.Module):
    """Factorized 1D+1D spectral convolution (Tran et al., ICLR 2023)."""

    def __init__(self, width: int, modes: int = 16) -> None:
        super().__init__()
        self.width = width
        self.modes = modes
        scale = 1.0 / (width * width)
        self.wx = nn.Parameter(scale * torch.randn(width, width, modes, 2))
        self.wy = nn.Parameter(scale * torch.randn(width, width, modes, 2))
        self.mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mx = min(self.modes, H // 2 + 1)
        my = min(self.modes, W // 2 + 1)
        xf_x = torch.fft.rfft(x, dim=-2)
        out_x = torch.zeros_like(xf_x)
        wxc = torch.complex(self.wx[:, :, :mx, 0], self.wx[:, :, :mx, 1])
        out_x[:, :, :mx, :] = torch.einsum("bixw,iox->boxw", xf_x[:, :, :mx, :], wxc)
        x_spec = torch.fft.irfft(out_x, n=H, dim=-2)

        xf_y = torch.fft.rfft(x, dim=-1)
        out_y = torch.zeros_like(xf_y)
        wyc = torch.complex(self.wy[:, :, :my, 0], self.wy[:, :, :my, 1])
        out_y[:, :, :, :my] = torch.einsum("bihy,ioy->bohy", xf_y[:, :, :, :my], wyc)
        y_spec = torch.fft.irfft(out_y, n=W, dim=-1)

        return x + self.mlp(x_spec + y_spec)


class FFNO2d(nn.Module):
    def __init__(self, in_ch: int = 5, width: int = 48, modes: int = 20, depth: int = 6) -> None:
        super().__init__()
        self.fc0 = nn.Conv2d(in_ch, width, 1)
        self.layers = nn.ModuleList([FactorizedSpectralLayer2d(width, modes=modes) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        h = self.fc0(_make_input_channels(kappa, lift, ortho, dist))
        for layer in self.layers:
            h = layer(h)
        v = self.head(h)[:, 0]
        return _apply_dirichlet_zero_v(v, lift)


class ResUNetSmall(nn.Module):
    """4-level residual U-Net with full 128x128 global receptive field."""

    def __init__(self, in_ch: int = 5, width: int = 32) -> None:
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(min(8, co), co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(min(8, co), co), nn.GELU(),
            )
        self.e1 = blk(in_ch, width)
        self.e2 = blk(width, width * 2)
        self.e3 = blk(width * 2, width * 4)
        self.bot = blk(width * 4, width * 4)
        self.d3 = blk(width * 8, width * 2)
        self.d2 = blk(width * 4, width)
        self.d1 = blk(width * 2, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = _make_input_channels(kappa, lift, ortho, dist)
        h1 = self.e1(x)
        h2 = self.e2(F.avg_pool2d(h1, 2))
        h3 = self.e3(F.avg_pool2d(h2, 2))
        hb = self.bot(F.avg_pool2d(h3, 2))
        u3 = self.d3(torch.cat([F.interpolate(hb, size=h3.shape[-2:], mode="bilinear", align_corners=False), h3], dim=1))
        u2 = self.d2(torch.cat([F.interpolate(u3, size=h2.shape[-2:], mode="bilinear", align_corners=False), h2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="bilinear", align_corners=False), h1], dim=1))
        v = self.head(u1)[:, 0]
        return _apply_dirichlet_zero_v(v, lift)


class UNOWrapper(nn.Module):
    def __init__(self, width: int = 32, modes: int = 12) -> None:
        super().__init__()
        self.net = UNO2d(in_channels=5, out_channels=1, width=width, modes1=modes, modes2=modes, pad=4, factor=1)

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = _make_input_channels(kappa, lift, ortho, dist).permute(0, 2, 3, 1)  # NHWC
        v = self.net(x)[..., 0]
        return _apply_dirichlet_zero_v(v, lift)


class ResidualSmoother2D(nn.Module):
    def __init__(self, channels: int, steps: int = 2, groups: int = 8) -> None:
        super().__init__()
        groups = max(1, min(groups, channels))
        self.steps = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            ) for _ in range(steps)
        ])
        self.norms = nn.ModuleList([nn.GroupNorm(groups, channels) for _ in self.steps])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk, norm in zip(self.steps, self.norms):
            x = F.gelu(norm(x + blk(x)))
        return x


class MgNO2dWrapper(nn.Module):
    """Multigrid Neural Operator (He et al., ICLR 2024; workspace implementation)."""

    def __init__(self, in_ch: int = 5, width: int = 48, levels: int = 4, smooth_steps: int = 2) -> None:
        super().__init__()
        self.levels = levels
        self.lift_proj = nn.Conv2d(in_ch, width, 1)
        self.down_smoothers = nn.ModuleList([ResidualSmoother2D(width, smooth_steps) for _ in range(levels)])
        self.restrict = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU()) for _ in range(levels - 1)
        ])
        self.coarse = ResidualSmoother2D(width, smooth_steps + 2)
        self.prolong = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(levels - 1)])
        self.up_smoothers = nn.ModuleList([ResidualSmoother2D(width, smooth_steps) for _ in range(levels - 1)])
        self.head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, 1, 1))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = self.lift_proj(_make_input_channels(kappa, lift, ortho, dist))
        skips = []
        for lvl in range(self.levels - 1):
            x = self.down_smoothers[lvl](x)
            skips.append(x)
            x = self.restrict[lvl](x)
        x = self.coarse(self.down_smoothers[-1](x))
        for lvl, (prolong, smoother) in enumerate(zip(self.prolong, self.up_smoothers)):
            skip = skips[-1 - lvl]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = smoother(skip + prolong(x))
        v = self.head(x)[:, 0]
        return _apply_dirichlet_zero_v(v, lift)


class AntiAliasedActivation(nn.Module):
    """Smooth upsample -> pointwise SiLU -> smooth downsample (CNO, Raonic et al., NeurIPS 2023)."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        up = F.interpolate(x, size=(2 * H, 2 * W), mode="bilinear", align_corners=False)
        up = self.act(up)
        return F.avg_pool2d(up, kernel_size=2, stride=2)


class CNOBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = AntiAliasedActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CNO2dWrapper(nn.Module):
    """Convolutional Neural Operator (Raonic et al., NeurIPS 2023)."""

    def __init__(self, in_ch: int = 5, width: int = 32) -> None:
        super().__init__()
        self.lift = CNOBlock(in_ch, width)
        self.e1 = CNOBlock(width, width)
        self.e2 = CNOBlock(width, width * 2)
        self.bot = CNOBlock(width * 2, width * 2)
        self.inv1 = CNOBlock(width, width)
        self.inv2 = CNOBlock(width * 2, width * 2)
        self.d2 = CNOBlock(width * 4, width)
        self.d1 = CNOBlock(width * 2, width)
        self.proj = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.proj.weight, std=1e-3)
        nn.init.zeros_(self.proj.bias)

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        x = self.lift(_make_input_channels(kappa, lift, ortho, dist))
        h1 = self.e1(x)
        h2 = self.e2(F.avg_pool2d(h1, 2))
        hb = self.bot(F.avg_pool2d(h2, 2))
        u2 = F.interpolate(hb, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.d2(torch.cat([u2, self.inv2(h2)], dim=1))
        u1 = F.interpolate(u2, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.d1(torch.cat([u1, self.inv1(h1)], dim=1))
        v = self.proj(u1)[:, 0]
        return _apply_dirichlet_zero_v(v, lift)


class PhysicsAttentionBlock(nn.Module):
    """Slice-based Physics Attention block (Transolver, Wu et al., ICML 2024)."""

    def __init__(self, dim: int, heads: int = 4, n_slices: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.n_slices = n_slices

        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.slice_weights = nn.Linear(dim, heads * n_slices)
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        B, N, C = x.shape
        h_in = self.norm1(x)
        fx = self.in_proj(h_in).view(B, N, self.heads, self.head_dim)
        # Slice assignment weights (B, N, H, G) -> softmax over G slices
        sw = F.softmax(self.slice_weights(h_in).view(B, N, self.heads, self.n_slices), dim=-1)
        # Aggregate spatial tokens into G physical slice tokens: (B, H, G, D)
        slice_norm = sw.sum(dim=1).clamp_min(1e-6).unsqueeze(-1)  # (B, H, G, 1)
        z = torch.einsum("bnhg,bnhd->bhgd", sw, fx) / slice_norm

        q = self.to_q(z)
        k = self.to_k(z)
        v = self.to_v(z)
        attn = F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        z_out = attn @ v  # (B, H, G, D)

        # Deslice back to spatial tokens
        out = torch.einsum("bnhg,bhgd->bnhd", sw, z_out).reshape(B, N, C)
        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm2(x))
        return x


class Transolver2d(nn.Module):
    """2D Physics-Attention Transformer (Transolver, Wu et al., ICML 2024)."""

    def __init__(self, in_ch: int = 5, dim: int = 64, depth: int = 4, heads: int = 4, n_slices: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.blocks = nn.ModuleList([PhysicsAttentionBlock(dim, heads=heads, n_slices=n_slices) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, 1, 1))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        B, H, W = kappa.shape
        x = self.stem(_make_input_channels(kappa, lift, ortho, dist))
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        for blk in self.blocks:
            tokens = blk(tokens)
        x = tokens.transpose(1, 2).view(B, -1, H, W)
        v = self.head(x)[:, 0]
        return _apply_dirichlet_zero_v(v, lift)
