"""True 3D Boundary-Capacitance Neural Operator (BC-NO 3D) + 3D baselines + 3D PCG Warm-Start.

Extends BC-NO identically to 3D scalar RVE homogenization:
  1. Exact 3D (-Delta_h)^{-1} via orthonormal 3D DST-I in O(N^3 log N).
  2. 1-voxel 3D interface strip + bounded residual weights w in [0, 1].
  3. 3D floating conductive component capacitance pooling (6-connected).
  4. 3D percolating-aware dipole readout for K_eff tensor columns.
  5. 3D Jacobi-PCG solver and BC-NO Warm-Started PCG benchmark (32^3 and 64^3).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from rve_discrete_operator_3d import face_conductivities_3d


def dst1_3d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 3D DST-I along last three dims (D, H, W)."""
    *lead, D, H, W = x.shape
    dtype, device = x.dtype, x.device

    z1 = torch.zeros(*lead, D, H, 1, dtype=dtype, device=device)
    e1 = torch.cat([z1, x, z1, -torch.flip(x, dims=[-1])], dim=-1)
    y1 = (-0.5 * torch.fft.fft(e1, dim=-1)[..., 1 : W + 1].imag) * math.sqrt(2.0 / (W + 1))

    z2 = torch.zeros(*lead, D, 1, W, dtype=dtype, device=device)
    e2 = torch.cat([z2, y1, z2, -torch.flip(y1, dims=[-2])], dim=-2)
    y2 = (-0.5 * torch.fft.fft(e2, dim=-2)[..., 1 : H + 1, :].imag) * math.sqrt(2.0 / (H + 1))

    z3 = torch.zeros(*lead, 1, H, W, dtype=dtype, device=device)
    e3 = torch.cat([z3, y2, z3, -torch.flip(y2, dims=[-3])], dim=-3)
    y3 = (-0.5 * torch.fft.fft(e3, dim=-3)[..., 1 : D + 1, :, :].imag) * math.sqrt(2.0 / (D + 1))
    return y3


class DirichletGreen3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[int, int, int, str, torch.dtype], torch.Tensor] = {}

    def _inv_eig(self, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (D, H, W, str(device), dtype)
        if key not in self._cache:
            md, mh, mw = D - 2, H - 2, W - 2
            hd, hh, hw = 1.0 / (D - 1), 1.0 / (H - 1), 1.0 / (W - 1)
            kd = torch.arange(1, md + 1, dtype=torch.float64, device=device)
            kh = torch.arange(1, mh + 1, dtype=torch.float64, device=device)
            kw = torch.arange(1, mw + 1, dtype=torch.float64, device=device)
            ed = 2.0 * (1.0 - torch.cos(math.pi * kd / (D - 1))) / (hd * hd)
            eh = 2.0 * (1.0 - torch.cos(math.pi * kh / (H - 1))) / (hh * hh)
            ew = 2.0 * (1.0 - torch.cos(math.pi * kw / (W - 1))) / (hw * hw)
            eig = ed[:, None, None] + eh[None, :, None] + ew[None, None, :]
            self._cache[key] = (1.0 / eig).to(dtype=dtype)[None]
        return self._cache[key]

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        _, D, H, W = source.shape
        s_in = source[:, 1:-1, 1:-1, 1:-1]
        hat = dst1_3d(s_in)
        u_in = dst1_3d(hat * self._inv_eig(D, H, W, source.device, source.dtype))
        out = torch.zeros_like(source)
        out[:, 1:-1, 1:-1, 1:-1] = u_in
        return out


def boundary_strip_mask_3d(kappa: torch.Tensor) -> torch.Tensor:
    dx = (kappa[:, 1:] - kappa[:, :-1]).abs() > 1e-6 * torch.maximum(kappa[:, 1:], kappa[:, :-1])
    dy = (kappa[:, :, 1:] - kappa[:, :, :-1]).abs() > 1e-6 * torch.maximum(kappa[:, :, 1:], kappa[:, :, :-1])
    dz = (kappa[:, :, :, 1:] - kappa[:, :, :, :-1]).abs() > 1e-6 * torch.maximum(kappa[:, :, :, 1:], kappa[:, :, :, :-1])
    nbr = torch.zeros_like(kappa, dtype=torch.bool)
    nbr[:, :-1] |= dx; nbr[:, 1:] |= dx
    nbr[:, :, :-1] |= dy; nbr[:, :, 1:] |= dy
    nbr[:, :, :, :-1] |= dz; nbr[:, :, :, 1:] |= dz
    interior = torch.zeros_like(nbr)
    interior[:, 1:-1, 1:-1, 1:-1] = True
    return (nbr & interior).to(kappa.dtype)


def precompute_floating_components_3d(kappa_batch: torch.Tensor, threshold: float = 1.5):
    B, D, H, W = kappa_batch.shape
    structure = ndimage.generate_binary_structure(3, 1)
    labels_out = np.zeros((B, D, H, W), dtype=np.int64)
    max_comp = 0
    k_np = kappa_batch.detach().cpu().numpy()
    for b in range(B):
        inc = k_np[b] > threshold
        lbl, nc = ndimage.label(inc, structure=structure)
        if nc == 0:
            continue
        bdry_ids = (
            set(np.unique(lbl[0]).tolist()) | set(np.unique(lbl[-1]).tolist())
            | set(np.unique(lbl[:, 0]).tolist()) | set(np.unique(lbl[:, -1]).tolist())
            | set(np.unique(lbl[:, :, 0]).tolist()) | set(np.unique(lbl[:, :, -1]).tolist())
        )
        bdry_ids.discard(0)
        remap = np.zeros(nc + 1, dtype=np.int64)
        nxt = 0
        for m in range(1, nc + 1):
            if m not in bdry_ids:
                nxt += 1
                remap[m] = nxt
        labels_out[b] = remap[lbl]
        max_comp = max(max_comp, nxt)
    return torch.from_numpy(labels_out).to(kappa_batch.device), max_comp


def dipole_keff_columns_3d(
    u: torch.Tensor,
    kappa: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    zg: torch.Tensor,
) -> torch.Tensor:
    """Percolating-aware 3D dipole readout for [K0, K1, K2]."""
    B, D, H, W = u.shape
    h = 1.0 / max(W - 1, 1)
    inc_all = kappa > 1.5
    nonperc = inc_all.clone()
    k_np = inc_all.detach().cpu().numpy()
    structure = ndimage.generate_binary_structure(3, 1)
    for b in range(B):
        if not bool(k_np[b].any()):
            continue
        lbl, nc = ndimage.label(k_np[b], structure=structure)
        for m in range(1, nc + 1):
            cm = lbl == m
            if (cm[0].any() and cm[-1].any()) or (cm[:, 0].any() and cm[:, -1].any()) or (cm[:, :, 0].any() and cm[:, :, -1].any()):
                nonperc[b][torch.from_numpy(cm).to(u.device)] = False

    kx, ky, kz = face_conductivities_3d(kappa, kappa_channel=0)
    kxm, kxp = kx[:, :-1, 1:-1, 1:-1], kx[:, 1:, 1:-1, 1:-1]
    kym, kyp = ky[:, 1:-1, :-1, 1:-1], ky[:, 1:-1, 1:, 1:-1]
    kzm, kzp = kz[:, 1:-1, 1:-1, :-1], kz[:, 1:-1, 1:-1, 1:]
    uc = u[:, 1:-1, 1:-1, 1:-1]

    c_inc = nonperc[:, 1:-1, 1:-1, 1:-1]
    inc_int = torch.zeros_like(nonperc)
    inc_int[:, 1:-1, 1:-1, 1:-1] = c_inc

    qx_n = kx[:, :, 1:-1, 1:-1] * (u[:, 1:, 1:-1, 1:-1] - u[:, :-1, 1:-1, 1:-1]) / h
    qy_n = ky[:, 1:-1, :, 1:-1] * (u[:, 1:-1, 1:, 1:-1] - u[:, 1:-1, :-1, 1:-1]) / h
    qz_n = kz[:, 1:-1, 1:-1, :] * (u[:, 1:-1, 1:-1, 1:] - u[:, 1:-1, 1:-1, :-1]) / h

    kx_ext = torch.where(inc_int[:, :-1, 1:-1, 1:-1] & inc_int[:, 1:, 1:-1, 1:-1], torch.zeros_like(qx_n), qx_n).sum((1, 2, 3)) / float((D - 1) * (H - 2) * (W - 2))
    ky_ext = torch.where(inc_int[:, 1:-1, :-1, 1:-1] & inc_int[:, 1:-1, 1:, 1:-1], torch.zeros_like(qy_n), qy_n).sum((1, 2, 3)) / float((D - 2) * (H - 1) * (W - 2))
    kz_ext = torch.where(inc_int[:, 1:-1, 1:-1, :-1] & inc_int[:, 1:-1, 1:-1, 1:], torch.zeros_like(qz_n), qz_n).sum((1, 2, 3)) / float((D - 2) * (H - 2) * (W - 1))

    val = (
        torch.where(~inc_int[:, :-2, 1:-1, 1:-1], kxm * (uc - u[:, :-2, 1:-1, 1:-1]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 2:, 1:-1, 1:-1], kxp * (uc - u[:, 2:, 1:-1, 1:-1]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 1:-1, :-2, 1:-1], kym * (uc - u[:, 1:-1, :-2, 1:-1]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 1:-1, 2:, 1:-1], kyp * (uc - u[:, 1:-1, 2:, 1:-1]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 1:-1, 1:-1, :-2], kzm * (uc - u[:, 1:-1, 1:-1, :-2]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 1:-1, 1:-1, 2:], kzp * (uc - u[:, 1:-1, 1:-1, 2:]), torch.zeros_like(uc))
    ) / (h * h)
    ap = torch.zeros_like(u)
    ap[:, 1:-1, 1:-1, 1:-1] = torch.where(c_inc, val, torch.zeros_like(val))
    kd0 = -(xg * ap).sum((1, 2, 3)) * h / float((H - 2) * (W - 2))
    kd1 = -(yg * ap).sum((1, 2, 3)) * h / float((D - 2) * (W - 2))
    kd2 = -(zg * ap).sum((1, 2, 3)) * h / float((D - 2) * (H - 2))
    return torch.stack([kx_ext + kd0, ky_ext + kd1, kz_ext + kd2], dim=-1)


class BoundaryBlock3D(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:, 0]


class BCNO3D(nn.Module):
    """3D Boundary-Capacitance Neural Operator."""

    def __init__(self, steps: int = 3, hidden: int = 20) -> None:
        super().__init__()
        self.steps = int(steps)
        self.green = DirichletGreen3D()
        # [beta, log_k, strip, inc_cond, inc_insul, floating, lift, o1, o2, dist, u, r_flux, u_dev*10, phi_map, sigma] -> 15 ch
        self.blocks = nn.ModuleList([BoundaryBlock3D(15, hidden=hidden) for _ in range(self.steps)])
        self.alpha_phi = nn.Parameter(torch.zeros(self.steps, dtype=torch.float32))
        self.gamma_r = nn.Parameter(torch.full((self.steps,), -2.0, dtype=torch.float32))

    def forward(self, kappa, lift, o1, o2, dist, labels, max_comp: int):
        B, D, H, W = kappa.shape
        h = 1.0 / max(W - 1, 1)
        scale_h = float(max(W - 1, 1))
        beta = (kappa - 1.0) / (kappa + 1.0)
        log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
        strip = boundary_strip_mask_3d(kappa)
        inc_cond = (kappa > 1.5).to(kappa.dtype)
        inc_insul = (kappa < 0.67).to(kappa.dtype)
        floating = (labels > 0).to(kappa.dtype)

        kx, ky, kz = face_conductivities_3d(kappa, kappa_channel=0)
        kxm, kxp = kx[:, :-1, 1:-1, 1:-1], kx[:, 1:, 1:-1, 1:-1]
        kym, kyp = ky[:, 1:-1, :-1, 1:-1], ky[:, 1:-1, 1:, 1:-1]
        kzm, kzp = kz[:, 1:-1, 1:-1, :-1], kz[:, 1:-1, 1:-1, 1:]
        diag = (kxm + kxp + kym + kyp + kzm + kzp).clamp_min(1e-12)
        wxm, wxp, wym, wyp, wzm, wzp = kxm / diag, kxp / diag, kym / diag, kyp / diag, kzm / diag, kzp / diag

        inc_b = kappa > 1.5
        c_inc = inc_b[:, 1:-1, 1:-1, 1:-1]
        m_xm, m_xp = ~inc_b[:, :-2, 1:-1, 1:-1], ~inc_b[:, 2:, 1:-1, 1:-1]
        m_ym, m_yp = ~inc_b[:, 1:-1, :-2, 1:-1], ~inc_b[:, 1:-1, 2:, 1:-1]
        m_zm, m_zp = ~inc_b[:, 1:-1, 1:-1, :-2], ~inc_b[:, 1:-1, 1:-1, 2:]

        sigma = torch.zeros_like(kappa)
        u = lift.clone()
        for t, block in enumerate(self.blocks):
            uc = u[:, 1:-1, 1:-1, 1:-1]
            r_tilde = torch.zeros_like(u)
            r_tilde[:, 1:-1, 1:-1, 1:-1] = (
                uc - wxm * u[:, :-2, 1:-1, 1:-1] - wxp * u[:, 2:, 1:-1, 1:-1]
                - wym * u[:, 1:-1, :-2, 1:-1] - wyp * u[:, 1:-1, 2:, 1:-1]
                - wzm * u[:, 1:-1, 1:-1, :-2] - wzp * u[:, 1:-1, 1:-1, 2:]
            )
            r_flux = r_tilde * scale_h

            val = (
                torch.where(m_xm, kxm * (uc - u[:, :-2, 1:-1, 1:-1]), torch.zeros_like(uc))
                + torch.where(m_xp, kxp * (uc - u[:, 2:, 1:-1, 1:-1]), torch.zeros_like(uc))
                + torch.where(m_ym, kym * (uc - u[:, 1:-1, :-2, 1:-1]), torch.zeros_like(uc))
                + torch.where(m_yp, kyp * (uc - u[:, 1:-1, 2:, 1:-1]), torch.zeros_like(uc))
                + torch.where(m_zm, kzm * (uc - u[:, 1:-1, 1:-1, :-2]), torch.zeros_like(uc))
                + torch.where(m_zp, kzp * (uc - u[:, 1:-1, 1:-1, 2:]), torch.zeros_like(uc))
            ) / (h * h)
            ap_iface = torch.zeros_like(u)
            ap_iface[:, 1:-1, 1:-1, 1:-1] = torch.where(c_inc, val, torch.zeros_like(val))

            if max_comp > 0:
                flat_lbl = labels.reshape(B, -1)
                flat_u = u.reshape(B, -1)
                flat_r = ap_iface.reshape(B, -1)
                flat_s = strip.reshape(B, -1)
                K = max_comp + 1
                cnt = torch.zeros(B, K, dtype=u.dtype, device=u.device).scatter_add_(1, flat_lbl, torch.ones_like(flat_u)).clamp_min(1.0)
                mean_u = torch.zeros(B, K, dtype=u.dtype, device=u.device).scatter_add_(1, flat_lbl, flat_u) / cnt
                mean_u[:, 0] = 0.0
                perim = torch.zeros(B, K, dtype=u.dtype, device=u.device).scatter_add_(1, flat_lbl, flat_s).clamp_min(1.0)
                sum_r = torch.zeros(B, K, dtype=u.dtype, device=u.device).scatter_add_(1, flat_lbl, flat_r)
                fd = (sum_r * h) / (perim * h)
                fd[:, 0] = 0.0
                u_dev = (u - torch.gather(mean_u, 1, flat_lbl).view(B, D, H, W)) * floating
                phi_map = torch.gather(fd, 1, flat_lbl).view(B, D, H, W) * floating
            else:
                u_dev = torch.zeros_like(u)
                phi_map = torch.zeros_like(u)

            feats = torch.stack(
                [beta, log_k, strip, inc_cond, inc_insul, floating, lift, o1, o2, dist, u, r_flux, u_dev * 10.0, phi_map, sigma],
                dim=1,
            )
            dsigma = (block(feats) + self.gamma_r[t] * r_flux + self.alpha_phi[t] * phi_map) * strip
            sigma = sigma + dsigma
            u = lift + self.green(sigma * scale_h)
        return u, sigma


def _zero_bdry_3d(v: torch.Tensor, lift: torch.Tensor) -> torch.Tensor:
    m = torch.zeros_like(v)
    m[:, 1:-1, 1:-1, 1:-1] = 1.0
    return lift + v * m


class FNO3dSmall(nn.Module):
    def __init__(self, in_ch: int = 6, width: int = 20, modes: int = 8, depth: int = 4) -> None:
        super().__init__()
        self.modes = modes
        self.fc0 = nn.Conv3d(in_ch, width, 1)
        scale = 1.0 / (width * width)
        self.convs = nn.ParameterList([nn.Parameter(scale * torch.randn(width, width, modes, modes, modes, 2)) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv3d(width, 32, 1), nn.GELU(), nn.Conv3d(32, 1, 1))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def _spec(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out = torch.zeros(B, C, D, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        m = min(self.modes, D // 2, H // 2, W // 2)
        wc = torch.complex(w[..., :m, :m, :m, 0], w[..., :m, :m, :m, 1])
        out[:, :, :m, :m, :m] = torch.einsum("bixyz,ioxyz->boxyz", xf[:, :, :m, :m, :m], wc)
        out[:, :, -m:, :m, :m] = torch.einsum("bixyz,ioxyz->boxyz", xf[:, :, -m:, :m, :m], wc)
        out[:, :, :m, -m:, :m] = torch.einsum("bixyz,ioxyz->boxyz", xf[:, :, :m, -m:, :m], wc)
        out[:, :, -m:, -m:, :m] = torch.einsum("bixyz,ioxyz->boxyz", xf[:, :, -m:, -m:, :m], wc)
        return torch.fft.irfftn(out, s=(D, H, W), dim=(-3, -2, -1))

    def forward(self, kappa, lift, o1, o2, dist):
        beta = (kappa - 1.0) / (kappa + 1.0)
        log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
        h = self.fc0(torch.stack([beta, log_k, lift, o1, o2, dist], dim=1))
        for w, lin in zip(self.convs, self.ws):
            h = F.gelu(self._spec(h, w) + lin(h))
        return _zero_bdry_3d(self.head(h)[:, 0], lift)


class ResUNet3dSmall(nn.Module):
    def __init__(self, in_ch: int = 6, width: int = 20) -> None:
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, padding=1), nn.GroupNorm(min(4, co), co), nn.GELU(),
                nn.Conv3d(co, co, 3, padding=1), nn.GroupNorm(min(4, co), co), nn.GELU(),
            )
        self.e1 = blk(in_ch, width)
        self.e2 = blk(width, width * 2)
        self.bot = blk(width * 2, width * 2)
        self.d2 = blk(width * 4, width)
        self.d1 = blk(width * 2, width)
        self.head = nn.Conv3d(width, 1, 1)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, kappa, lift, o1, o2, dist):
        beta = (kappa - 1.0) / (kappa + 1.0)
        log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
        x = torch.stack([beta, log_k, lift, o1, o2, dist], dim=1)
        h1 = self.e1(x)
        h2 = self.e2(F.avg_pool3d(h1, 2))
        hb = self.bot(F.avg_pool3d(h2, 2))
        u2 = self.d2(torch.cat([F.interpolate(hb, size=h2.shape[-3:], mode="trilinear", align_corners=False), h2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=h1.shape[-3:], mode="trilinear", align_corners=False), h1], dim=1))
        return _zero_bdry_3d(self.head(u1)[:, 0], lift)


@torch.no_grad()
def jacobi_pcg_3d(
    kappa: torch.Tensor,
    lift: torch.Tensor,
    u_init: torch.Tensor | None = None,
    *,
    tol: float = 1e-6,
    max_iters: int = 2000,
) -> tuple[torch.Tensor, int, float]:
    """GPU/CPU Jacobi-preconditioned Conjugate Gradient for 3D harmonic-face PDE."""
    t0 = time.time()
    h = 1.0 / max(kappa.shape[-1] - 1, 1)
    kx, ky, kz = face_conductivities_3d(kappa, kappa_channel=0)
    kxm, kxp = kx[:, :-1, 1:-1, 1:-1] / (h * h), kx[:, 1:, 1:-1, 1:-1] / (h * h)
    kym, kyp = ky[:, 1:-1, :-1, 1:-1] / (h * h), ky[:, 1:-1, 1:, 1:-1] / (h * h)
    kzm, kzp = kz[:, 1:-1, 1:-1, :-1] / (h * h), kz[:, 1:-1, 1:-1, 1:] / (h * h)
    diag = (kxm + kxp + kym + kyp + kzm + kzp).clamp_min(1e-12)

    def apply_A(v_in: torch.Tensor) -> torch.Tensor:
        vp = F.pad(v_in, (1, 1, 1, 1, 1, 1))
        return (
            diag * v_in
            - kxm * vp[:, :-2, 1:-1, 1:-1] - kxp * vp[:, 2:, 1:-1, 1:-1]
            - kym * vp[:, 1:-1, :-2, 1:-1] - kyp * vp[:, 1:-1, 2:, 1:-1]
            - kzm * vp[:, 1:-1, 1:-1, :-2] - kzp * vp[:, 1:-1, 1:-1, 2:]
        )

    # RHS = -A_full(lift)
    b = -(
        diag * lift[:, 1:-1, 1:-1, 1:-1]
        - kxm * lift[:, :-2, 1:-1, 1:-1] - kxp * lift[:, 2:, 1:-1, 1:-1]
        - kym * lift[:, 1:-1, :-2, 1:-1] - kyp * lift[:, 1:-1, 2:, 1:-1]
        - kzm * lift[:, 1:-1, 1:-1, :-2] - kzp * lift[:, 1:-1, 1:-1, 2:]
    )
    b_norm = float(torch.linalg.vector_norm(b))
    v = torch.zeros_like(b) if u_init is None else (u_init[:, 1:-1, 1:-1, 1:-1] - lift[:, 1:-1, 1:-1, 1:-1])
    r = b - apply_A(v)
    z = r / diag
    p = z.clone()
    rz = float((r * z).sum())
    iters = 0
    for it in range(1, max_iters + 1):
        if float(torch.linalg.vector_norm(r)) <= tol * max(b_norm, 1e-12):
            break
        Ap = apply_A(p)
        alpha = rz / max(float((p * Ap).sum()), 1e-30)
        v = v + alpha * p
        r = r - alpha * Ap
        z = r / diag
        rz_new = float((r * z).sum())
        p = z + (rz_new / max(rz, 1e-30)) * p
        rz = rz_new
        iters = it
    out = lift.clone()
    out[:, 1:-1, 1:-1, 1:-1] += v
    if kappa.device.type == "cuda":
        torch.cuda.synchronize()
    return out, iters, (time.time() - t0) * 1000.0
