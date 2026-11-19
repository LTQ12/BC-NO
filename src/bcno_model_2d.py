"""Boundary-Capacitance Neural Operator (BC-NO) and baseline architectures.

Flat Colab-ready module (zero relative parent imports).
Includes clean ablation toggles:
  - use_green: replace 2D DST-I G0 with local conv readout (ablation w/o Green)
  - use_capacitance: disable floating-component u_dev and phi_map (ablation w/o Cap)
  - use_strip_mask: predict sigma everywhere instead of 1-cell jump strip (ablation w/o Strip)
  - use_physics_skip: disable -gamma_r * r_flux physics skip
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from rve_discrete_operator_2d import face_conductivities


def dst1_2d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D DST-I along last two dims using odd extension + FFT."""
    *lead, H, W = x.shape
    dtype, device = x.dtype, x.device
    zero1 = torch.zeros(*lead, H, 1, dtype=dtype, device=device)
    ext1 = torch.cat([zero1, x, zero1, -torch.flip(x, dims=[-1])], dim=-1)
    f1 = torch.fft.fft(ext1, dim=-1)
    y1 = (-0.5 * f1[..., 1 : W + 1].imag) * math.sqrt(2.0 / (W + 1))

    zero2 = torch.zeros(*lead, 1, W, dtype=dtype, device=device)
    ext2 = torch.cat([zero2, y1, zero2, -torch.flip(y1, dims=[-2])], dim=-2)
    f2 = torch.fft.fft(ext2, dim=-2)
    y2 = (-0.5 * f2[..., 1 : H + 1, :].imag) * math.sqrt(2.0 / (H + 1))
    return y2


class DirichletGreen2D(nn.Module):
    """Exact (-Delta_h)^{-1} on [0,1]^2 interior grid; supports arbitrary resolution N."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def _inv_eig(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, str(device), dtype)
        if key not in self._cache:
            mh, mw = H - 2, W - 2
            hh = 1.0 / max(H - 1, 1)
            hw = 1.0 / max(W - 1, 1)
            kh = torch.arange(1, mh + 1, dtype=torch.float64, device=device)
            kw = torch.arange(1, mw + 1, dtype=torch.float64, device=device)
            eigh = 2.0 * (1.0 - torch.cos(math.pi * kh / max(H - 1, 1))) / (hh * hh)
            eigw = 2.0 * (1.0 - torch.cos(math.pi * kw / max(W - 1, 1))) / (hw * hw)
            inv = (1.0 / (eigh[:, None] + eigw[None, :])).to(dtype=dtype)[None]
            self._cache[key] = inv
        return self._cache[key]

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        _, H, W = source.shape
        s_in = source[:, 1:-1, 1:-1]
        hat = dst1_2d(s_in)
        u_in = dst1_2d(hat * self._inv_eig(H, W, source.device, source.dtype))
        out = torch.zeros_like(source)
        out[:, 1:-1, 1:-1] = u_in
        return out


def boundary_strip_mask(kappa: torch.Tensor) -> torch.Tensor:
    """1-cell strip on both sides of conductivity jumps."""
    mask = (kappa - 1.0).abs() > 1e-6
    nbr_diff = torch.zeros_like(mask)
    diff_x = (kappa[:, 1:, :] - kappa[:, :-1, :]).abs() > 1e-6 * torch.maximum(kappa[:, 1:, :], kappa[:, :-1, :])
    diff_y = (kappa[:, :, 1:] - kappa[:, :, :-1]).abs() > 1e-6 * torch.maximum(kappa[:, :, 1:], kappa[:, :, :-1])
    nbr_diff[:, :-1, :] |= diff_x
    nbr_diff[:, 1:, :] |= diff_x
    nbr_diff[:, :, :-1] |= diff_y
    nbr_diff[:, :, 1:] |= diff_y
    interior = torch.zeros_like(mask)
    interior[:, 1:-1, 1:-1] = True
    return (nbr_diff & interior).to(kappa.dtype)


def precompute_floating_components(kappa_batch: torch.Tensor, threshold: float = 1.5) -> tuple[torch.Tensor, int]:
    """Label 4-connected conductive components (kappa > threshold) that do NOT touch outer boundary."""
    B, H, W = kappa_batch.shape
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    labels_out = np.zeros((B, H, W), dtype=np.int64)
    max_comp = 0
    k_np = kappa_batch.detach().cpu().numpy()
    for b in range(B):
        inc = k_np[b] > threshold
        lbl, ncomp = ndimage.label(inc, structure=structure)
        if ncomp == 0:
            continue
        bdry_ids = (
            set(np.unique(lbl[0, :]).tolist())
            | set(np.unique(lbl[-1, :]).tolist())
            | set(np.unique(lbl[:, 0]).tolist())
            | set(np.unique(lbl[:, -1]).tolist())
        )
        bdry_ids.discard(0)
        remap = np.zeros(ncomp + 1, dtype=np.int64)
        nxt = 0
        for m in range(1, ncomp + 1):
            if m not in bdry_ids:
                nxt += 1
                remap[m] = nxt
        labels_out[b] = remap[lbl]
        max_comp = max(max_comp, nxt)
    return torch.from_numpy(labels_out).to(kappa_batch.device), max_comp


def precompute_operator_weights(kappa: torch.Tensor):
    kx, ky = face_conductivities(kappa)
    kxm = kx[:, :-1, 1:-1]
    kxp = kx[:, 1:, 1:-1]
    kym = ky[:, 1:-1, :-1]
    kyp = ky[:, 1:-1, 1:]
    diag = (kxm + kxp + kym + kyp).clamp_min(1e-12)
    wxm = kxm / diag
    wxp = kxp / diag
    wym = kym / diag
    wyp = kyp / diag

    inc = kappa > 1.5
    c_inc = inc[:, 1:-1, 1:-1]
    m_xm = ~inc[:, :-2, 1:-1]
    m_xp = ~inc[:, 2:, 1:-1]
    m_ym = ~inc[:, 1:-1, :-2]
    m_yp = ~inc[:, 1:-1, 2:]
    return wxm, wxp, wym, wyp, kx, ky, c_inc, m_xm, m_xp, m_ym, m_yp


def normalized_residual_from_weights(u: torch.Tensor, wxm, wxp, wym, wyp) -> torch.Tensor:
    uc = u[:, 1:-1, 1:-1]
    out = torch.zeros_like(u)
    out[:, 1:-1, 1:-1] = (
        uc
        - wxm * u[:, :-2, 1:-1]
        - wxp * u[:, 2:, 1:-1]
        - wym * u[:, 1:-1, :-2]
        - wyp * u[:, 1:-1, 2:]
    )
    return out


def interface_flux_from_weights(u: torch.Tensor, kx, ky, c_inc, m_xm, m_xp, m_ym, m_yp) -> torch.Tensor:
    h = 1.0 / max(u.shape[-1] - 1, 1)
    uc = u[:, 1:-1, 1:-1]
    kxm = kx[:, :-1, 1:-1]
    kxp = kx[:, 1:, 1:-1]
    kym = ky[:, 1:-1, :-1]
    kyp = ky[:, 1:-1, 1:]
    val = (
        torch.where(m_xm, kxm * (uc - u[:, :-2, 1:-1]), torch.zeros_like(uc))
        + torch.where(m_xp, kxp * (uc - u[:, 2:, 1:-1]), torch.zeros_like(uc))
        + torch.where(m_ym, kym * (uc - u[:, 1:-1, :-2]), torch.zeros_like(uc))
        + torch.where(m_yp, kyp * (uc - u[:, 1:-1, 2:]), torch.zeros_like(uc))
    ) / (h * h)
    ap_inc = torch.zeros_like(u)
    ap_inc[:, 1:-1, 1:-1] = torch.where(c_inc, val, torch.zeros_like(val))
    return ap_inc


def floating_component_stats(
    field: torch.Tensor,
    ap_iface: torch.Tensor,
    strip: torch.Tensor,
    labels: torch.Tensor,
    max_comp: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, W = field.shape
    if max_comp == 0:
        return torch.zeros_like(field), torch.zeros_like(field)
    flat_lbl = labels.reshape(B, -1)
    flat_u = field.reshape(B, -1)
    flat_r = ap_iface.reshape(B, -1)
    flat_s = strip.reshape(B, -1)

    K = max_comp + 1
    ones = torch.ones_like(flat_u)
    cnt = torch.zeros(B, K, dtype=field.dtype, device=field.device).scatter_add_(1, flat_lbl, ones).clamp_min(1.0)
    sum_u = torch.zeros(B, K, dtype=field.dtype, device=field.device).scatter_add_(1, flat_lbl, flat_u)
    mean_u = sum_u / cnt
    mean_u[:, 0] = 0.0

    perim = torch.zeros(B, K, dtype=field.dtype, device=field.device).scatter_add_(1, flat_lbl, flat_s).clamp_min(1.0)
    sum_r = torch.zeros(B, K, dtype=field.dtype, device=field.device).scatter_add_(1, flat_lbl, flat_r)
    h = 1.0 / max(W - 1, 1)
    flux_density = (sum_r * h) / (perim * h)
    flux_density[:, 0] = 0.0

    gathered_mean = torch.gather(mean_u, 1, flat_lbl).view(B, H, W)
    gathered_phi = torch.gather(flux_density, 1, flat_lbl).view(B, H, W)
    active = (labels > 0).to(field.dtype)
    return (field - gathered_mean) * active, gathered_phi * active


def dipole_keff_columns(
    u: torch.Tensor,
    kappa: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
) -> torch.Tensor:
    """Generalized percolating-aware contrast-invariant dipole readout for [K_col_0, K_col_1].

    Excludes wall-to-wall percolating clusters (which carry O(kappa) direct conduction)
    from dipole replacement, while eliminating O(kappa) amplification on all
    non-percolating conductive inclusions.
    """
    B, H, W = u.shape
    h = 1.0 / max(W - 1, 1)
    inc_all = kappa > 1.5
    nonperc_0 = inc_all.clone()
    nonperc_1 = inc_all.clone()
    k_np = inc_all.detach().cpu().numpy()
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    for b in range(B):
        if not bool(k_np[b].any()):
            continue
        lbl, nc = ndimage.label(k_np[b], structure=structure)
        for m in range(1, nc + 1):
            cm = lbl == m
            if (cm[0, :].any() and cm[-1, :].any()) or (cm[:, 0].any() and cm[:, -1].any()):
                cm_t = torch.from_numpy(cm).to(u.device)
                nonperc_0[b][cm_t] = False
                nonperc_1[b][cm_t] = False

    kx, ky = face_conductivities(kappa)
    kxm = kx[:, :-1, 1:-1]
    kxp = kx[:, 1:, 1:-1]
    kym = ky[:, 1:-1, :-1]
    kyp = ky[:, 1:-1, 1:]
    uc = u[:, 1:-1, 1:-1]

    def _one_axis(inc_m: torch.Tensor, is_x: bool) -> torch.Tensor:
        c_inc = inc_m[:, 1:-1, 1:-1]
        inc_int = torch.zeros_like(inc_m)
        inc_int[:, 1:-1, 1:-1] = c_inc
        if is_x:
            q_neg = kx[:, :, 1:-1] * (u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / h
            inc_f = inc_int[:, :-1, 1:-1] & inc_int[:, 1:, 1:-1]
            k_ext = torch.where(inc_f, torch.zeros_like(q_neg), q_neg).sum(dim=(1, 2)) / float((H - 1) * (W - 2))
            coord, denom = x_grid, float(W - 2)
        else:
            q_neg = ky[:, 1:-1, :] * (u[:, 1:-1, 1:] - u[:, 1:-1, :-1]) / h
            inc_f = inc_int[:, 1:-1, :-1] & inc_int[:, 1:-1, 1:]
            k_ext = torch.where(inc_f, torch.zeros_like(q_neg), q_neg).sum(dim=(1, 2)) / float((H - 2) * (W - 1))
            coord, denom = y_grid, float(H - 2)
        val = (
            torch.where(~inc_int[:, :-2, 1:-1], kxm * (uc - u[:, :-2, 1:-1]), torch.zeros_like(uc))
            + torch.where(~inc_int[:, 2:, 1:-1], kxp * (uc - u[:, 2:, 1:-1]), torch.zeros_like(uc))
            + torch.where(~inc_int[:, 1:-1, :-2], kym * (uc - u[:, 1:-1, :-2]), torch.zeros_like(uc))
            + torch.where(~inc_int[:, 1:-1, 2:], kyp * (uc - u[:, 1:-1, 2:]), torch.zeros_like(uc))
        ) / (h * h)
        ap = torch.zeros_like(u)
        ap[:, 1:-1, 1:-1] = torch.where(c_inc, val, torch.zeros_like(val))
        k_dip = -(coord * ap).sum(dim=(1, 2)) * h / denom
        return k_ext + k_dip

    return torch.stack([_one_axis(nonperc_0, True), _one_axis(nonperc_1, False)], dim=-1)


class BoundaryBlock(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:, 0]


class BCNO2D(nn.Module):
    """Boundary-Capacitance Neural Operator with ablation switches."""

    def __init__(
        self,
        steps: int = 4,
        hidden: int = 32,
        *,
        use_green: bool = True,
        use_capacitance: bool = True,
        use_strip_mask: bool = True,
        use_physics_skip: bool = True,
    ) -> None:
        super().__init__()
        self.steps = int(steps)
        self.use_green = bool(use_green)
        self.use_capacitance = bool(use_capacitance)
        self.use_strip_mask = bool(use_strip_mask)
        self.use_physics_skip = bool(use_physics_skip)
        self.green = DirichletGreen2D()
        self.blocks = nn.ModuleList([BoundaryBlock(14, hidden=hidden) for _ in range(self.steps)])
        if not self.use_green:
            for b in self.blocks:
                nn.init.normal_(b.net[-1].weight, std=1e-4)
        self.alpha_phi = nn.Parameter(torch.zeros(self.steps, dtype=torch.float32))
        self.gamma_r = nn.Parameter(torch.full((self.steps,), -2.0, dtype=torch.float32))

    def forward(
        self,
        kappa: torch.Tensor,
        lift: torch.Tensor,
        ortho: torch.Tensor,
        dist: torch.Tensor,
        labels: torch.Tensor,
        max_comp: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = (kappa - 1.0) / (kappa + 1.0)
        log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
        strip = boundary_strip_mask(kappa)
        inc_cond = (kappa > 1.5).to(kappa.dtype)
        inc_insul = (kappa < 0.67).to(kappa.dtype)
        floating = (labels > 0).to(kappa.dtype)

        wxm, wxp, wym, wyp, kx, ky, c_inc, m_xm, m_xp, m_ym, m_yp = precompute_operator_weights(kappa)

        interior = torch.ones_like(kappa)
        interior[:, 0, :] = 0.0
        interior[:, -1, :] = 0.0
        interior[:, :, 0] = 0.0
        interior[:, :, -1] = 0.0
        active_mask = strip if self.use_strip_mask else interior

        sigma = torch.zeros_like(kappa)
        u = lift.clone()
        scale_h = float(max(kappa.shape[-1] - 1, 1))
        for t, block in enumerate(self.blocks):
            r_flux = normalized_residual_from_weights(u, wxm, wxp, wym, wyp) * scale_h
            ap_iface = interface_flux_from_weights(u, kx, ky, c_inc, m_xm, m_xp, m_ym, m_yp)
            if self.use_capacitance:
                u_dev, phi_map = floating_component_stats(u, ap_iface, strip, labels, max_comp)
            else:
                u_dev = torch.zeros_like(u)
                phi_map = torch.zeros_like(u)
            feats = torch.stack(
                [beta, log_k, strip, inc_cond, inc_insul, floating, lift, ortho, dist, u, r_flux, u_dev * 10.0, phi_map, sigma],
                dim=1,
            )
            delta = block(feats)
            if self.use_physics_skip and self.use_green:
                delta = delta + self.gamma_r[t] * r_flux
            if self.use_capacitance:
                delta = delta + self.alpha_phi[t] * phi_map
            if self.use_green:
                sigma = sigma + delta * active_mask
                u = lift + self.green(sigma * scale_h)
            else:
                sigma = sigma + delta * interior
                u = lift + sigma * interior
        return u, sigma


class PeriodicGreen2D(nn.Module):
    """Exact (-Delta_per)^{-1} on periodic [0,1)^2 grid via 2D rfft2."""

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[tuple[int, int, str, torch.dtype], torch.Tensor] = {}

    def _inv_eig(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (H, W, str(device), dtype)
        if key not in self._cache:
            hh, hw = 1.0 / H, 1.0 / W
            kh = torch.fft.fftfreq(H, d=1.0 / H, device=device, dtype=torch.float64)
            kw = torch.fft.rfftfreq(W, d=1.0 / W, device=device, dtype=torch.float64)
            eigh = 2.0 * (1.0 - torch.cos(2.0 * math.pi * kh / H)) / (hh * hh)
            eigw = 2.0 * (1.0 - torch.cos(2.0 * math.pi * kw / W)) / (hw * hw)
            eig = eigh[:, None] + eigw[None, :]
            inv = torch.where(eig > 1e-12, 1.0 / eig.clamp_min(1e-12), torch.zeros_like(eig))
            self._cache[key] = inv.to(dtype=dtype)[None]
        return self._cache[key]

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        _, H, W = source.shape
        hat = torch.fft.rfft2(source)
        out = torch.fft.irfft2(hat * self._inv_eig(H, W, source.device, source.dtype), s=(H, W))
        return out - out.mean(dim=(1, 2), keepdim=True)


def periodic_boundary_strip_mask(kappa: torch.Tensor) -> torch.Tensor:
    diff_x = (torch.roll(kappa, -1, dims=-2) - kappa).abs() > 1e-6 * torch.maximum(torch.roll(kappa, -1, dims=-2), kappa)
    diff_y = (torch.roll(kappa, -1, dims=-1) - kappa).abs() > 1e-6 * torch.maximum(torch.roll(kappa, -1, dims=-1), kappa)
    strip = diff_x | torch.roll(diff_x, 1, dims=-2) | diff_y | torch.roll(diff_y, 1, dims=-1)
    return strip.to(kappa.dtype)


def precompute_periodic_components(kappa_batch: torch.Tensor, threshold: float = 1.5):
    """Label periodic 4-connected conductive components (torus union-find) + wrapped center coords."""
    B, H, W = kappa_batch.shape
    k_np = (kappa_batch > threshold).detach().cpu().numpy()
    labels_out = np.zeros((B, H, W), dtype=np.int64)
    dx_wrap = np.zeros((B, H, W), dtype=np.float32)
    dy_wrap = np.zeros((B, H, W), dtype=np.float32)
    is_nonperc = np.zeros((B, H, W), dtype=bool)
    max_comp = 0
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    ax_x = np.arange(H) / float(H)
    ax_y = np.arange(W) / float(W)
    gx, gy = np.meshgrid(ax_x, ax_y, indexing="ij")

    for b in range(B):
        lbl, nc = ndimage.label(k_np[b], structure=structure)
        if nc == 0:
            continue
        parent = list(range(nc + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        def union(a, c):
            ra, rc = find(a), find(c)
            if ra != rc:
                parent[max(ra, rc)] = min(ra, rc)
        for j in range(W):
            if lbl[0, j] > 0 and lbl[-1, j] > 0:
                union(int(lbl[0, j]), int(lbl[-1, j]))
        for i in range(H):
            if lbl[i, 0] > 0 and lbl[i, -1] > 0:
                union(int(lbl[i, 0]), int(lbl[i, -1]))
        roots = {find(m) for m in range(1, nc + 1)}
        root_map = {r: idx + 1 for idx, r in enumerate(sorted(roots))}
        remap = np.zeros(nc + 1, dtype=np.int64)
        for m in range(1, nc + 1):
            remap[m] = root_map[find(m)]
        merged = remap[lbl]
        labels_out[b] = merged
        n_merged = len(roots)
        max_comp = max(max_comp, n_merged)

        for m in range(1, n_merged + 1):
            cm = merged == m
            # Percolation check: if component covers all rows or all cols -> percolates!
            if cm.any(axis=1).all() or cm.any(axis=0).all():
                continue
            is_nonperc[b] |= cm
            # Circular mean center on torus [0, 1)^2
            thx = 2.0 * np.pi * gx[cm]
            thy = 2.0 * np.pi * gy[cm]
            cx = (np.arctan2(np.sin(thx).mean(), np.cos(thx).mean()) / (2.0 * np.pi)) % 1.0
            cy = (np.arctan2(np.sin(thy).mean(), np.cos(thy).mean()) / (2.0 * np.pi)) % 1.0
            dx_wrap[b, cm] = ((gx[cm] - cx + 0.5) % 1.0) - 0.5
            dy_wrap[b, cm] = ((gy[cm] - cy + 0.5) % 1.0) - 0.5

    dev = kappa_batch.device
    return (
        torch.from_numpy(labels_out).to(dev),
        max_comp,
        torch.from_numpy(is_nonperc).to(dev),
        torch.from_numpy(dx_wrap).to(dev),
        torch.from_numpy(dy_wrap).to(dev),
    )


def periodic_dipole_keff(
    v: torch.Tensor,
    kappa: torch.Tensor,
    load_axis: torch.Tensor,
    nonperc_mask: torch.Tensor,
    dx_wrap: torch.Tensor,
    dy_wrap: torch.Tensor,
) -> torch.Tensor:
    """Exact periodic contrast-invariant dipole readout for [K_col_0, K_col_1]."""
    B, H, W = v.shape
    h = 1.0 / float(H)
    kx = 2.0 * kappa * torch.roll(kappa, -1, dims=-2) / (kappa + torch.roll(kappa, -1, dims=-2)).clamp_min(1e-12)
    ky = 2.0 * kappa * torch.roll(kappa, -1, dims=-1) / (kappa + torch.roll(kappa, -1, dims=-1)).clamp_min(1e-12)
    kxm = torch.roll(kx, 1, dims=-2)
    kym = torch.roll(ky, 1, dims=-1)

    e0 = (load_axis == 0).to(v.dtype).view(B, 1, 1)
    e1 = (load_axis == 1).to(v.dtype).view(B, 1, 1)

    gx_tot = e0 + (torch.roll(v, -1, dims=-2) - v) / h
    gy_tot = e1 + (torch.roll(v, -1, dims=-1) - v) / h

    inc_fx = nonperc_mask & torch.roll(nonperc_mask, -1, dims=-2)
    inc_fy = nonperc_mask & torch.roll(nonperc_mask, -1, dims=-1)
    k_ext_0 = torch.where(inc_fx, torch.zeros_like(gx_tot), kx * gx_tot).mean(dim=(1, 2))
    k_ext_1 = torch.where(inc_fy, torch.zeros_like(gy_tot), ky * gy_tot).mean(dim=(1, 2))

    # Matrix -> inclusion boundary flux (in finite-volume units / h)
    dv_xm = torch.where(~torch.roll(nonperc_mask, 1, dims=-2), kxm * ((v - torch.roll(v, 1, dims=-2)) / h + e0), torch.zeros_like(v))
    dv_xp = torch.where(~torch.roll(nonperc_mask, -1, dims=-2), kx * ((v - torch.roll(v, -1, dims=-2)) / h - e0), torch.zeros_like(v))
    dv_ym = torch.where(~torch.roll(nonperc_mask, 1, dims=-1), kym * ((v - torch.roll(v, 1, dims=-1)) / h + e1), torch.zeros_like(v))
    dv_yp = torch.where(~torch.roll(nonperc_mask, -1, dims=-1), ky * ((v - torch.roll(v, -1, dims=-1)) / h - e1), torch.zeros_like(v))
    ap_per = torch.where(nonperc_mask, (dv_xm + dv_xp + dv_ym + dv_yp) / h, torch.zeros_like(v))
    k_dip_0 = -(dx_wrap * ap_per).sum(dim=(1, 2)) / float(H * W)
    k_dip_1 = -(dy_wrap * ap_per).sum(dim=(1, 2)) / float(H * W)
    return torch.stack([k_ext_0 + k_dip_0, k_ext_1 + k_dip_1], dim=-1)


class PeriodicBoundaryBlock(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[:, 0]


class BCNO_PBC_2D(nn.Module):
    """2D Periodic Boundary-Condition BC-NO via rfft2 Green + wrapped capacitance pooling."""

    def __init__(self, steps: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.steps = int(steps)
        self.green = PeriodicGreen2D()
        # [beta, log_k, strip, inc_cond, inc_insul, nonperc, e0, e1, v, r_flux, v_dev*10, phi_map, sigma] -> 13 ch
        self.blocks = nn.ModuleList([PeriodicBoundaryBlock(13, hidden=hidden) for _ in range(self.steps)])
        self.alpha_phi = nn.Parameter(torch.zeros(self.steps, dtype=torch.float32))
        self.gamma_r = nn.Parameter(torch.full((self.steps,), -2.0, dtype=torch.float32))

    def forward(
        self,
        kappa: torch.Tensor,
        load_axis: torch.Tensor,
        labels: torch.Tensor,
        max_comp: int,
        nonperc_mask: torch.Tensor,
        dx_wrap: torch.Tensor,
        dy_wrap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, W = kappa.shape
        h = 1.0 / float(H)
        beta = (kappa - 1.0) / (kappa + 1.0)
        log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
        strip = periodic_boundary_strip_mask(kappa)
        inc_cond = (kappa > 1.5).to(kappa.dtype)
        inc_insul = (kappa < 0.67).to(kappa.dtype)
        nonperc_f = nonperc_mask.to(kappa.dtype)

        kx = 2.0 * kappa * torch.roll(kappa, -1, dims=-2) / (kappa + torch.roll(kappa, -1, dims=-2)).clamp_min(1e-12)
        ky = 2.0 * kappa * torch.roll(kappa, -1, dims=-1) / (kappa + torch.roll(kappa, -1, dims=-1)).clamp_min(1e-12)
        kxm = torch.roll(kx, 1, dims=-2)
        kym = torch.roll(ky, 1, dims=-1)
        diag = (kxm + kx + kym + ky).clamp_min(1e-12)
        wxm, wxp, wym, wyp = kxm / diag, kx / diag, kym / diag, ky / diag

        e0 = (load_axis == 0).to(kappa.dtype).view(B, 1, 1).expand(B, H, W)
        e1 = (load_axis == 1).to(kappa.dtype).view(B, 1, 1).expand(B, H, W)

        sigma = torch.zeros_like(kappa)
        v = torch.zeros_like(kappa)
        scale_h = float(H)
        for t, block in enumerate(self.blocks):
            # Normalized periodic residual of u = E*x + v: (A u) / diag
            dv_xm = wxm * ((v - torch.roll(v, 1, dims=-2)) + h * e0)
            dv_xp = wxp * ((v - torch.roll(v, -1, dims=-2)) - h * e0)
            dv_ym = wym * ((v - torch.roll(v, 1, dims=-1)) + h * e1)
            dv_yp = wyp * ((v - torch.roll(v, -1, dims=-1)) - h * e1)
            r_flux = (dv_xm + dv_xp + dv_ym + dv_yp) * scale_h

            # Wrapped interface flux + rigid-potential (u - c_m) pooling via torus coords dx_wrap, dy_wrap
            ap_xm = torch.where(~torch.roll(nonperc_mask, 1, dims=-2), kxm * ((v - torch.roll(v, 1, dims=-2)) / h + e0), torch.zeros_like(v))
            ap_xp = torch.where(~torch.roll(nonperc_mask, -1, dims=-2), kx * ((v - torch.roll(v, -1, dims=-2)) / h - e0), torch.zeros_like(v))
            ap_ym = torch.where(~torch.roll(nonperc_mask, 1, dims=-1), kym * ((v - torch.roll(v, 1, dims=-1)) / h + e1), torch.zeros_like(v))
            ap_yp = torch.where(~torch.roll(nonperc_mask, -1, dims=-1), ky * ((v - torch.roll(v, -1, dims=-1)) / h - e1), torch.zeros_like(v))
            ap_iface = torch.where(nonperc_mask, (ap_xm + ap_xp + ap_ym + ap_yp) / h, torch.zeros_like(v))

            # Local u relative to wrapped inclusion center: E_0*dx + E_1*dy + v
            u_rel = e0 * dx_wrap + e1 * dy_wrap + v
            v_dev, phi_map = floating_component_stats(u_rel, ap_iface, strip, labels * nonperc_mask.long(), max_comp)
            feats = torch.stack(
                [beta, log_k, strip, inc_cond, inc_insul, nonperc_f, e0, e1, v, r_flux, v_dev * 10.0, phi_map, sigma],
                dim=1,
            )
            dsigma = (block(feats) + self.gamma_r[t] * r_flux + self.alpha_phi[t] * phi_map) * strip
            sigma = sigma + dsigma
            v = self.green(sigma * scale_h)
        return v, sigma
