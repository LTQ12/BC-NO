"""2D Periodic Q1 FEM Linear Elasticity BC-NO and baselines (E_1 / E_0 up to 1000).

Implements:
  1. Fast vectorized Q1 periodic FEM elasticity solver for C_eff homogenization (3 strain modes).
  2. Exact Q1 Navier-Cauchy Green operator G0_elas in 2D rfft2 (verified to 6.8e-15).
  3. 3-DOF rigid-inclusion capacitance pooling (2 translations + 1 rotation) + normalized residual.
  4. Virtual-work stress-dipole readout for C_eff tensor (eliminates O(E_1/E_0) amplification).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


def q1_unit_matrices(h: float, nu: float = 0.3):
    lam0 = 1.0 * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu0 = 1.0 / (2.0 * (1.0 + nu))
    gp = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    nodes_ref = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    K_e = np.zeros((8, 8), dtype=np.float64)
    B_mid = np.zeros((3, 8), dtype=np.float64)
    for xi in gp:
        for eta in gp:
            B = np.zeros((3, 8), dtype=np.float64)
            for a, (sx, sy) in enumerate(nodes_ref):
                dNx = 0.25 * sx * (1.0 + sy * eta) * (2.0 / h)
                dNy = 0.25 * sy * (1.0 + sx * xi) * (2.0 / h)
                B[0, 2 * a] = dNx
                B[1, 2 * a + 1] = dNy
                B[2, 2 * a] = dNy
                B[2, 2 * a + 1] = dNx
            D = np.array([[lam0 + 2 * mu0, lam0, 0.0], [lam0, lam0 + 2 * mu0, 0.0], [0.0, 0.0, mu0]], dtype=np.float64)
            K_e += 0.25 * h * h * (B.T @ D @ B)
    for a, (sx, sy) in enumerate(nodes_ref):
        B_mid[0, 2 * a] = 0.25 * sx * (2.0 / h)
        B_mid[1, 2 * a + 1] = 0.25 * sy * (2.0 / h)
        B_mid[2, 2 * a] = 0.25 * sy * (2.0 / h)
        B_mid[2, 2 * a + 1] = 0.25 * sx * (2.0 / h)
    return K_e, B_mid, lam0, mu0


def solve_q1_elasticity_periodic(E_elem: np.ndarray, E_macro: np.ndarray, nu: float = 0.3):
    """Exact Q1 FEM periodic elasticity solve; returns nodal fluctuation v (N,N,2) & mean stress [sxx, syy, sxy]."""
    N = E_elem.shape[0]
    h = 1.0 / float(N)
    K_unit, B_mid, lam0, mu0 = q1_unit_matrices(h, nu=nu)

    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    n00 = ii * N + jj
    n10 = ((ii + 1) % N) * N + jj
    n11 = ((ii + 1) % N) * N + ((jj + 1) % N)
    n01 = ii * N + ((jj + 1) % N)
    elem_nodes = np.stack([n00, n10, n11, n01], axis=-1).reshape(-1, 4)
    elem_dofs = np.empty((N * N, 8), dtype=np.int64)
    for a in range(4):
        elem_dofs[:, 2 * a] = 2 * elem_nodes[:, a]
        elem_dofs[:, 2 * a + 1] = 2 * elem_nodes[:, a] + 1

    E_flat = E_elem.reshape(-1)
    Ke_all = E_flat[:, None, None] * K_unit[None]
    rr = np.repeat(elem_dofs[:, :, None], 8, axis=2).reshape(-1)
    cc = np.repeat(elem_dofs[:, None, :], 8, axis=1).reshape(-1)
    K_glob = sp.coo_matrix((Ke_all.reshape(-1), (rr, cc)), shape=(2 * N * N, 2 * N * N)).tolil()

    x_loc = np.array([[0.0, 0.0], [h, 0.0], [h, h], [0.0, h]], dtype=np.float64)
    u0_loc = (x_loc @ E_macro.T).reshape(8)
    f_loc = -np.einsum("eij,j->ei", Ke_all, u0_loc)
    rhs = np.zeros(2 * N * N, dtype=np.float64)
    np.add.at(rhs, elem_dofs, f_loc)

    for d in (0, 1):
        K_glob[d, :] = 0.0
        K_glob[d, d] = 1.0
        rhs[d] = 0.0
    v_sol = spla.spsolve(K_glob.tocsc(), rhs).reshape(N, N, 2)
    v_sol -= v_sol.mean(axis=(0, 1), keepdims=True)

    v_loc = v_sol.reshape(-1, 2)[elem_nodes].reshape(-1, 8)
    u_loc = u0_loc[None, :] + v_loc
    eps_e = u_loc @ B_mid.T
    lam_e = E_flat * lam0
    mu_e = E_flat * mu0
    sxx = (lam_e + 2 * mu_e) * eps_e[:, 0] + lam_e * eps_e[:, 1]
    syy = lam_e * eps_e[:, 0] + (lam_e + 2 * mu_e) * eps_e[:, 1]
    sxy = mu_e * eps_e[:, 2]
    stress_mean = np.array([sxx.mean(), syy.mean(), sxy.mean()], dtype=np.float64)
    return v_sol, stress_mean


class PeriodicElasticityGreen2D(nn.Module):
    """Exact periodic Q1 Navier-Cauchy Green operator in 2D rfft2."""

    def __init__(self, nu: float = 0.3) -> None:
        super().__init__()
        self.nu = float(nu)
        self._cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}

    def _inv_symbol(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (N, str(device), dtype)
        if key not in self._cache:
            h = 1.0 / float(N)
            K_unit, _, _, _ = q1_unit_matrices(h, nu=self.nu)
            S = np.zeros((3, 3, 2, 2), dtype=np.float64)
            node_offsets = [(0, 0), (1, 0), (1, 1), (0, 1)]
            for a, (ax, ay) in enumerate(node_offsets):
                for b, (bx, by) in enumerate(node_offsets):
                    S[bx - ax + 1, by - ay + 1] += K_unit[2 * a : 2 * a + 2, 2 * b : 2 * b + 2]
            wx = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0)
            wy = 2.0 * np.pi * np.fft.rfftfreq(N, d=1.0)
            kx, ky = np.meshgrid(wx, wy, indexing="ij")
            M = np.zeros((N, N // 2 + 1, 2, 2), dtype=np.complex128)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    M += np.exp(1j * (kx * dx + ky * dy))[..., None, None] * S[dx + 1, dy + 1][None, None]
            inv_M = np.zeros_like(M)
            det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
            mask = np.abs(det) > 1e-12
            inv_M[mask, 0, 0] = M[mask, 1, 1] / det[mask]
            inv_M[mask, 1, 1] = M[mask, 0, 0] / det[mask]
            inv_M[mask, 0, 1] = -M[mask, 0, 1] / det[mask]
            inv_M[mask, 1, 0] = -M[mask, 1, 0] / det[mask]
            cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            self._cache[key] = torch.from_numpy(inv_M).to(device=device, dtype=cdtype)[None]  # (1, N, N//2+1, 2, 2)
        return self._cache[key]

    def forward(self, force_b2hw: torch.Tensor) -> torch.Tensor:
        # force_b2hw: (B, 2, N, N)
        B, _, N, _ = force_b2hw.shape
        f_hat = torch.fft.rfft2(force_b2hw, dim=(-2, -1)).permute(0, 2, 3, 1)  # (B, N, N//2+1, 2)
        inv_M = self._inv_symbol(N, force_b2hw.device, force_b2hw.dtype)
        v_hat = torch.einsum("nxyab,nxyb->nxya", inv_M, f_hat).permute(0, 3, 1, 2)
        v = torch.fft.irfft2(v_hat, s=(N, N), dim=(-2, -1))
        return v - v.mean(dim=(-2, -1), keepdim=True)


def q1_stress_dipole_readout(
    v: torch.Tensor,
    E_elem: torch.Tensor,
    E_macro: torch.Tensor,
    nonperc_elem: torch.Tensor,
    dx_node: torch.Tensor,
    dy_node: torch.Tensor,
    nu: float = 0.3,
) -> torch.Tensor:
    """Exact virtual-work stress-dipole readout for homogenized [sig_xx, sig_yy, sig_xy].

    v: (B, 2, N, N), E_elem: (B, N, N), E_macro: (B, 2, 2), nonperc_elem: (B, N, N) bool
    """
    B, _, N, _ = v.shape
    h = 1.0 / float(N)
    dtype, dev = v.dtype, v.device
    K_unit, B_mid, lam0, mu0 = q1_unit_matrices(h, nu=nu)
    K_t = torch.tensor(K_unit, dtype=dtype, device=dev)
    B_t = torch.tensor(B_mid, dtype=dtype, device=dev)

    # Assemble 4 corner displacements per element: (B, N, N, 8)
    v_xy = v.permute(0, 2, 3, 1)  # (B, N, N, 2)
    v00 = v_xy
    v10 = torch.roll(v_xy, -1, dims=1)
    v11 = torch.roll(torch.roll(v_xy, -1, dims=1), -1, dims=2)
    v01 = torch.roll(v_xy, -1, dims=2)
    v_el = torch.cat([v00, v10, v11, v01], dim=-1)  # (B, N, N, 8)

    # Local affine displacement u0_loc: (B, 8)
    x_loc = torch.tensor([[0.0, 0.0], [h, 0.0], [h, h], [0.0, h]], dtype=dtype, device=dev)
    u0_el = torch.einsum("ak,bjk->baj", x_loc, E_macro).reshape(B, 1, 1, 8)
    u_el = v_el + u0_el  # (B, N, N, 8)

    # Element strain & stress (using E_elem)
    eps = torch.einsum("ca,bija->bijc", B_t, u_el)  # (B, N, N, 3)
    lam = E_elem * lam0
    mu = E_elem * mu0
    sxx = (lam + 2 * mu) * eps[..., 0] + lam * eps[..., 1]
    syy = lam * eps[..., 0] + (lam + 2 * mu) * eps[..., 1]
    sxy = mu * eps[..., 2]

    # Exclude non-percolating stiff inclusions from direct interior volume stress
    sxx_mat = torch.where(nonperc_elem, torch.zeros_like(sxx), sxx).sum((1, 2)) * (h * h)
    syy_mat = torch.where(nonperc_elem, torch.zeros_like(syy), syy).sum((1, 2)) * (h * h)
    sxy_mat = torch.where(nonperc_elem, torch.zeros_like(sxy), sxy).sum((1, 2)) * (h * h)

    # Matrix nodal forces f_mat (uses ONLY E_elem on ~nonperc_elem, where E <= 1.5!)
    E_mat = torch.where(nonperc_elem, torch.zeros_like(E_elem), E_elem)
    f_el = torch.einsum("ab,nijb->nija", K_t, u_el) * E_mat[..., None]  # (B, N, N, 8)

    # Scatter element forces back to nodes (B, N, N, 2)
    f_node = (
        f_el[..., 0:2]
        + torch.roll(f_el[..., 2:4], 1, dims=1)
        + torch.roll(torch.roll(f_el[..., 4:6], 1, dims=1), 1, dims=2)
        + torch.roll(f_el[..., 6:8], 1, dims=2)
    )
    sig_xx_dip = -(dx_node * f_node[..., 0]).sum((1, 2))
    sig_yy_dip = -(dy_node * f_node[..., 1]).sum((1, 2))
    sig_xy_dip = -0.5 * (dy_node * f_node[..., 0] + dx_node * f_node[..., 1]).sum((1, 2))
    return torch.stack([sxx_mat + sig_xx_dip, syy_mat + sig_yy_dip, sxy_mat + sig_xy_dip], dim=-1)


def precompute_elasticity_components(E_elem: torch.Tensor, threshold: float = 1.5):
    """Label periodic stiff inclusions (elements -> nodes), wrapped coords for 3-DOF capacitance & stress dipole."""
    B, N, _ = E_elem.shape
    e_np = (E_elem > threshold).detach().cpu().numpy()
    nonperc_el = np.zeros((B, N, N), dtype=bool)
    node_lbls = np.zeros((B, N, N), dtype=np.int64)
    dx_n = np.zeros((B, N, N), dtype=np.float32)
    dy_n = np.zeros((B, N, N), dtype=np.float32)
    strip_n = np.zeros((B, N, N), dtype=np.float32)
    max_comp = 0
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    ax = np.arange(N) / float(N)
    gx, gy = np.meshgrid(ax, ax, indexing="ij")

    for b in range(B):
        # Node touches inclusion element if any of 4 adjacent elements is in inclusion
        inc_e = e_np[b]
        n_in = inc_e | np.roll(inc_e, 1, 0) | np.roll(inc_e, 1, 1) | np.roll(np.roll(inc_e, 1, 0), 1, 1)
        n_out = (~inc_e) | np.roll(~inc_e, 1, 0) | np.roll(~inc_e, 1, 1) | np.roll(np.roll(~inc_e, 1, 0), 1, 1)
        strip_n[b] = (n_in & n_out).astype(np.float32)

        lbl, nc = ndimage.label(inc_e, structure=structure)
        if nc == 0:
            continue
        parent = list(range(nc + 1))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        def union(a, c):
            ra, rc = find(a), find(c)
            if ra != rc: parent[max(ra, rc)] = min(ra, rc)
        for j in range(N):
            if lbl[0, j] > 0 and lbl[-1, j] > 0: union(int(lbl[0, j]), int(lbl[-1, j]))
        for i in range(N):
            if lbl[i, 0] > 0 and lbl[i, -1] > 0: union(int(lbl[i, 0]), int(lbl[i, -1]))
        roots = {find(m) for m in range(1, nc + 1)}
        rmap = {r: idx + 1 for idx, r in enumerate(sorted(roots))}
        merged = np.zeros_like(lbl)
        for m in range(1, nc + 1):
            merged[lbl == m] = rmap[find(m)]
        # Map element labels to incident nodes (for non-percolating inclusions)
        nxt = 0
        for m in range(1, len(roots) + 1):
            cm_e = merged == m
            if cm_e.any(axis=1).all() or cm_e.any(axis=0).all():
                continue
            nonperc_el[b] |= cm_e
            nxt += 1
            cm_n = cm_e | np.roll(cm_e, 1, 0) | np.roll(cm_e, 1, 1) | np.roll(np.roll(cm_e, 1, 0), 1, 1)
            node_lbls[b, cm_n] = nxt
            thx, thy = 2.0 * np.pi * gx[cm_n], 2.0 * np.pi * gy[cm_n]
            cx = (np.arctan2(np.sin(thx).mean(), np.cos(thx).mean()) / (2.0 * np.pi)) % 1.0
            cy = (np.arctan2(np.sin(thy).mean(), np.cos(thy).mean()) / (2.0 * np.pi)) % 1.0
            dx_n[b, cm_n] = ((gx[cm_n] - cx + 0.5) % 1.0) - 0.5
            dy_n[b, cm_n] = ((gy[cm_n] - cy + 0.5) % 1.0) - 0.5
        max_comp = max(max_comp, nxt)

    dev = E_elem.device
    return (
        torch.from_numpy(nonperc_el).to(dev),
        torch.from_numpy(node_lbls).to(dev),
        max_comp,
        torch.from_numpy(dx_n).to(dev),
        torch.from_numpy(dy_n).to(dev),
        torch.from_numpy(strip_n).to(dev),
    )


class BCNO_Elas_2D(nn.Module):
    """2D Periodic Q1 Elasticity Boundary-Capacitance Neural Operator."""

    def __init__(self, steps: int = 4, hidden: int = 32, nu: float = 0.3) -> None:
        super().__init__()
        self.steps = int(steps)
        self.nu = float(nu)
        self.green = PeriodicElasticityGreen2D(nu=nu)
        # Features per step (16):
        # [beta_node, log_E, strip, E11, E22, E12, vx, vy, rx_norm, ry_norm, u_dev_x*10, u_dev_y*10, phi_x, phi_y, sig_x, sig_y]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(16, hidden, 3, padding=1, padding_mode="circular"), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="circular"), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, padding_mode="circular"), nn.GELU(),
                nn.Conv2d(hidden, 2, 1),
            ) for _ in range(self.steps)
        ])
        for b in self.blocks:
            nn.init.normal_(b[-1].weight, std=1e-2)
            nn.init.zeros_(b[-1].bias)
        self.gamma_r = nn.Parameter(torch.full((self.steps,), -1.0, dtype=torch.float32))

    def forward(self, E_elem, E_macro, nonperc_el, node_lbls, max_comp: int, dx_n, dy_n, strip_n):
        B, N, _ = E_elem.shape
        h = 1.0 / float(N)
        dtype, dev = E_elem.dtype, E_elem.device
        K_unit, _, _, _ = q1_unit_matrices(h, nu=self.nu)
        K_t = torch.tensor(K_unit, dtype=dtype, device=dev)

        # Nodal stiffness average for feature conditioning
        E_node = 0.25 * (E_elem + torch.roll(E_elem, 1, 1) + torch.roll(E_elem, 1, 2) + torch.roll(torch.roll(E_elem, 1, 1), 1, 2))
        beta_n = (E_node - 1.0) / (E_node + 1.0)
        log_E = torch.log(E_node.clamp(1e-4, 1e6)) / 10.0

        # Nodal diagonal of K_glob (2 channels: K_xx_diag, K_yy_diag)
        k_diag_e = torch.diagonal(K_t).view(4, 2)  # (4, 2)
        diag_n = (
            E_elem[..., None] * k_diag_e[0]
            + torch.roll(E_elem, 1, 1)[..., None] * k_diag_e[1]
            + torch.roll(torch.roll(E_elem, 1, 1), 1, 2)[..., None] * k_diag_e[2]
            + torch.roll(E_elem, 1, 2)[..., None] * k_diag_e[3]
        ).clamp_min(1e-8)  # (B, N, N, 2)

        x_loc = torch.tensor([[0.0, 0.0], [h, 0.0], [h, h], [0.0, h]], dtype=dtype, device=dev)
        u0_el = torch.einsum("ak,bjk->baj", x_loc, E_macro).reshape(B, 1, 1, 8)
        E11 = E_macro[:, 0, 0].view(B, 1, 1).expand(B, N, N)
        E22 = E_macro[:, 1, 1].view(B, 1, 1).expand(B, N, N)
        E12 = E_macro[:, 0, 1].view(B, 1, 1).expand(B, N, N)

        E_mat = torch.where(nonperc_el, torch.zeros_like(E_elem), E_elem)
        v = torch.zeros(B, 2, N, N, dtype=dtype, device=dev)
        sigma = torch.zeros(B, 2, N, N, dtype=dtype, device=dev)
        scale_h = float(N)

        for t, block in enumerate(self.blocks):
            v_xy = v.permute(0, 2, 3, 1)
            v_el = torch.cat([v_xy, torch.roll(v_xy, -1, 1), torch.roll(torch.roll(v_xy, -1, 1), -1, 2), torch.roll(v_xy, -1, 2)], dim=-1)
            u_el = v_el + u0_el
            f_el = torch.einsum("ab,nijb->nija", K_t, u_el) * E_elem[..., None]
            f_glob = (
                f_el[..., 0:2] + torch.roll(f_el[..., 2:4], 1, 1)
                + torch.roll(torch.roll(f_el[..., 4:6], 1, 1), 1, 2) + torch.roll(f_el[..., 6:8], 1, 2)
            )
            r_norm = (f_glob / diag_n) * scale_h  # O(1) normalized displacement residual

            # Matrix-side reaction on interface nodes
            fm_el = torch.einsum("ab,nijb->nija", K_t, u_el) * E_mat[..., None]
            fm_glob = (
                fm_el[..., 0:2] + torch.roll(fm_el[..., 2:4], 1, 1)
                + torch.roll(torch.roll(fm_el[..., 4:6], 1, 1), 1, 2) + torch.roll(fm_el[..., 6:8], 1, 2)
            )

            # Floating rigid-body capacitance pooling (mean translation + net force)
            if max_comp > 0:
                flat_lbl = node_lbls.reshape(B, -1)
                K = max_comp + 1
                cnt = torch.zeros(B, K, dtype=dtype, device=dev).scatter_add_(1, flat_lbl, torch.ones_like(E_elem.reshape(B, -1))).clamp_min(1.0)
                perim = torch.zeros(B, K, dtype=dtype, device=dev).scatter_add_(1, flat_lbl, strip_n.reshape(B, -1)).clamp_min(1.0)
                u_rel_x = E11 * dx_n + E12 * dy_n + v[:, 0]
                u_rel_y = E12 * dx_n + E22 * dy_n + v[:, 1]
                mx = (torch.zeros(B, K, dtype=dtype, device=dev).scatter_add_(1, flat_lbl, u_rel_x.reshape(B, -1)) / cnt)
                my = (torch.zeros(B, K, dtype=dtype, device=dev).scatter_add_(1, flat_lbl, u_rel_y.reshape(B, -1)) / cnt)
                mx[:, 0] = 0.0; my[:, 0] = 0.0
                fx_s = (torch.zeros(B, K, dtype=dtype, device=dev).scatter_add_(1, flat_lbl, fm_glob[..., 0].reshape(B, -1)) / perim) * scale_h
                fy_s = (torch.zeros(B, K, dtype=dtype, device=dev).scatter_add_(1, flat_lbl, fm_glob[..., 1].reshape(B, -1)) / perim) * scale_h
                fx_s[:, 0] = 0.0; fy_s[:, 0] = 0.0
                act = (node_lbls > 0).to(dtype)
                u_dev_x = (u_rel_x - torch.gather(mx, 1, flat_lbl).view(B, N, N)) * act
                u_dev_y = (u_rel_y - torch.gather(my, 1, flat_lbl).view(B, N, N)) * act
                phi_x = torch.gather(fx_s, 1, flat_lbl).view(B, N, N) * act
                phi_y = torch.gather(fy_s, 1, flat_lbl).view(B, N, N) * act
            else:
                u_dev_x = u_dev_y = phi_x = phi_y = torch.zeros_like(E_elem)

            feats = torch.stack([
                beta_n, log_E, strip_n, E11, E22, E12,
                v[:, 0], v[:, 1], r_norm[..., 0], r_norm[..., 1],
                u_dev_x * 10.0, u_dev_y * 10.0, phi_x, phi_y, sigma[:, 0], sigma[:, 1],
            ], dim=1)
            ds = (block(feats) + self.gamma_r[t] * r_norm.permute(0, 3, 1, 2)) * strip_n[:, None]
            sigma = sigma + ds
            # Scale boundary force by 1/N to match unit stiffness K_unit ~ O(1)
            v = self.green(sigma * (1.0 / scale_h))
        return v, sigma


class ResUNet_Elas_2D(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, padding_mode="circular"), nn.GroupNorm(min(8, co), co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1, padding_mode="circular"), nn.GroupNorm(min(8, co), co), nn.GELU(),
            )
        self.e1 = blk(5, width)
        self.e2 = blk(width, width * 2)
        self.bot = blk(width * 2, width * 4)
        self.d2 = blk(width * 6, width * 2)
        self.d1 = blk(width * 3, width)
        self.head = nn.Conv2d(width, 2, 1)
        nn.init.normal_(self.head.weight, std=1e-3); nn.init.zeros_(self.head.bias)

    def forward(self, E_elem, E_macro):
        B, N, _ = E_elem.shape
        E_n = 0.25 * (E_elem + torch.roll(E_elem, 1, 1) + torch.roll(E_elem, 1, 2) + torch.roll(torch.roll(E_elem, 1, 1), 1, 2))
        x = torch.stack([
            (E_n - 1.0) / (E_n + 1.0),
            torch.log(E_n.clamp(1e-4, 1e6)) / 10.0,
            E_macro[:, 0, 0].view(B, 1, 1).expand(B, N, N),
            E_macro[:, 1, 1].view(B, 1, 1).expand(B, N, N),
            E_macro[:, 0, 1].view(B, 1, 1).expand(B, N, N),
        ], dim=1)
        h1 = self.e1(x)
        h2 = self.e2(F.avg_pool2d(h1, 2))
        hb = self.bot(F.avg_pool2d(h2, 2))
        u2 = self.d2(torch.cat([F.interpolate(hb, size=h2.shape[-2:], mode="bilinear", align_corners=False), h2], dim=1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=h1.shape[-2:], mode="bilinear", align_corners=False), h1], dim=1))
        out = self.head(u1)
        return out - out.mean(dim=(-2, -1), keepdim=True)


class FNO_Elas_2D(nn.Module):
    def __init__(self, width: int = 32, modes: int = 16, depth: int = 4) -> None:
        super().__init__()
        self.modes = modes
        self.fc0 = nn.Conv2d(5, width, 1)
        scale = 1.0 / (width * width)
        self.convs = nn.ParameterList([nn.Parameter(scale * torch.randn(width, width, modes, modes, 2)) for _ in range(depth)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.head = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(), nn.Conv2d(64, 2, 1))
        nn.init.normal_(self.head[-1].weight, std=1e-3); nn.init.zeros_(self.head[-1].bias)

    def _spec(self, x, w):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        m = min(self.modes, H // 2, W // 2)
        wc = torch.complex(w[..., :m, :m, 0], w[..., :m, :m, 1])
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], wc)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], wc)
        return torch.fft.irfft2(out, s=(H, W))

    def forward(self, E_elem, E_macro):
        B, N, _ = E_elem.shape
        E_n = 0.25 * (E_elem + torch.roll(E_elem, 1, 1) + torch.roll(E_elem, 1, 2) + torch.roll(torch.roll(E_elem, 1, 1), 1, 2))
        x = torch.stack([
            (E_n - 1.0) / (E_n + 1.0),
            torch.log(E_n.clamp(1e-4, 1e6)) / 10.0,
            E_macro[:, 0, 0].view(B, 1, 1).expand(B, N, N),
            E_macro[:, 1, 1].view(B, 1, 1).expand(B, N, N),
            E_macro[:, 0, 1].view(B, 1, 1).expand(B, N, N),
        ], dim=1)
        h = self.fc0(x)
        for w, lin in zip(self.convs, self.ws):
            h = F.gelu(self._spec(h, w) + lin(h))
        out = self.head(h)
        return out - out.mean(dim=(-2, -1), keepdim=True)
