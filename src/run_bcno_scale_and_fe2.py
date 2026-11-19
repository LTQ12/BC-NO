"""Scale & Hard-Physics CMAME Suite (Exp 10-13):
  Exp 10: Constitutive Admissibility Audit + 16x16 Two-Scale FE^2 Composite Plate
  Exp 11: Large-Sample N=128 Benchmark (1000 train / 200 ID / 200 OOD)
  Exp 12: Large 3D 64^3 Composites (120 train / 60 test) + 64^3 Real Ruhr Sandstone CT
  Exp 13: Nonlinear High-Contrast PDE + Continuous Log-Normal Matrix + Non-Convex Star Inclusions (400 train / 200 test)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_2d import CNO2dWrapper, FFNO2d, FNO2dSmall, MgNO2dWrapper, ResUNetSmall, Transolver2d
from bcno_model_2d import BCNO2D, DirichletGreen2D, dipole_keff_columns, precompute_floating_components
from bcno_model_3d import (
    BCNO3D,
    FNO3dSmall,
    ResUNet3dSmall,
    boundary_strip_mask_3d,
    dipole_keff_columns_3d,
    jacobi_pcg_3d,
    precompute_floating_components_3d,
)
from colab_data_gen import fast_interior_solve, generate_synthetic_benchmarks, make_fiber_columns
from rve_discrete_operator_2d import conservative_flux_columns, face_conductivities, harmonic_mean
from rve_discrete_operator_3d import conservative_flux_columns_3d, face_conductivities_3d
from run_bcno_fe2_audit import build_fe2_micro_suite, constitutive_audit, predict_keff_cols, solve_macro_q1_plate
from run_bcno_paper_suite import evaluate_pack, pack_from_pt, train_model


# ---------- Exp 11: Large-Sample 2D Dataset (1000 train + 200 ID + 200 OOD, N=128) ----------
def build_large_2d_pack(N: int, n_cases: int, seed: int, c_lo: float, c_hi: float, vf_lo: float, vf_hi: float, device: torch.device) -> dict:
    rng = np.random.default_rng(seed)
    ax1 = np.linspace(0.0, 1.0, N, dtype=np.float64)
    xg, yg = np.meshgrid(ax1, ax1, indexing="ij")
    kaps, vs, lifts, axs = [], [], [], []
    t0 = time.time()
    for s in range(n_cases):
        c = float(np.exp(rng.uniform(np.log(c_lo), np.log(c_hi))))
        if rng.random() < 0.25:
            c = 1.0 / c
        vf = float(rng.uniform(vf_lo, vf_hi))
        kap = make_fiber_columns(N, int(rng.integers(1, 10_000_000)), c, vf)
        for ax_id in (0, 1):
            v, lft = fast_interior_solve(kap, ax_id)
            kaps.append(kap.astype(np.float32))
            vs.append(v.astype(np.float32))
            lifts.append(lft.astype(np.float32))
            axs.append(ax_id)
    kap_t = torch.from_numpy(np.stack(kaps))
    v_t = torch.from_numpy(np.stack(vs))
    lift_t = torch.from_numpy(np.stack(lifts))
    k_cols = conservative_flux_columns((lift_t + v_t).double(), kap_t.double(), kappa_channel=0).float()

    B = kap_t.shape[0]
    xg_t = torch.from_numpy(xg.astype(np.float32))
    yg_t = torch.from_numpy(yg.astype(np.float32))
    ortho_t = torch.stack([yg_t if a == 0 else xg_t for a in axs], dim=0)
    idx = torch.arange(N, dtype=torch.float32)
    ii, jj = idx[:, None].expand(N, N), idx[None, :].expand(N, N)
    dist_t = (torch.minimum(torch.minimum(ii, jj), torch.minimum(N - 1 - ii, N - 1 - jj)) / (N / 2.0))[None].expand(B, N, N)
    lbl, mc = precompute_floating_components(kap_t)
    print(f"  Built 2D pack ({B} samples, {N}x{N}) in {time.time()-t0:.1f}s", flush=True)
    return {
        "kappa": kap_t, "kappa_raw": kap_t, "v_true": v_t, "u_true": lift_t + v_t,
        "lift": lift_t, "ortho": ortho_t, "dist": dist_t, "k_cols": k_cols,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc, "x_grid": xg_t[None], "y_grid": yg_t[None],
    }


# ---------- Exp 12: Large 3D 64^3 Fast Solver (GPU PCG or Sparse LU fallback) ----------
def solve_3d_fast(kappa_np: np.ndarray, device: torch.device) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    N = kappa_np.shape[0]
    ax1 = np.linspace(0.0, 1.0, N, dtype=np.float64)
    xg, yg, zg = np.meshgrid(ax1, ax1, ax1, indexing="ij")
    grids = (xg, yg, zg)
    kap_t = torch.from_numpy(kappa_np).to(device)[None]
    vs, lifts, ks = [], [], []
    for ax_id in (0, 1, 2):
        lft = grids[ax_id]
        lft_t = torch.from_numpy(lft).to(device)[None]
        u_t, _, _ = jacobi_pcg_3d(kap_t, lft_t, tol=1e-7, max_iters=2500)
        u_cpu = u_t.cpu()
        k_c = conservative_flux_columns_3d(u_cpu, torch.from_numpy(kappa_np)[None], kappa_channel=0)[0].numpy()
        vs.append((u_cpu[0].numpy() - lft).astype(np.float32))
        lifts.append(lft.astype(np.float32))
        ks.append(k_c.astype(np.float32))
    return vs, lifts, np.stack(ks)


def build_3d_64_synth_pack(N: int, n_cases: int, seed: int, c_lo: float, c_hi: float, device: torch.device) -> dict:
    rng = np.random.default_rng(seed)
    ax1 = np.linspace(0.0, 1.0, N)
    xg, yg, zg = np.meshgrid(ax1, ax1, ax1, indexing="ij")
    kaps, vs, lifts, o1s, o2s, ks, axs = [], [], [], [], [], [], []
    t0 = time.time()
    for s in range(n_cases):
        c = float(np.exp(rng.uniform(np.log(c_lo), np.log(c_hi))))
        if rng.random() < 0.25:
            c = 1.0 / c
        mask = np.zeros((N, N, N), dtype=bool)
        if s % 2 == 0:
            for _ in range(int(rng.integers(5, 11))):
                ctr = rng.uniform(0.14, 0.86, size=3)
                rad = rng.uniform(0.07, 0.17, size=3)
                mask |= ((xg - ctr[0]) / rad[0]) ** 2 + ((yg - ctr[1]) / rad[1]) ** 2 + ((zg - ctr[2]) / rad[2]) ** 2 <= 1.0
        else:
            f_ax = int(rng.integers(0, 3))
            t_pair = [(yg, zg), (xg, zg), (xg, yg)][f_ax]
            for _ in range(int(rng.integers(4, 8))):
                ctr = rng.uniform(0.14, 0.86, size=2)
                r = float(rng.uniform(0.05, 0.12))
                mask |= (t_pair[0] - ctr[0]) ** 2 + (t_pair[1] - ctr[1]) ** 2 <= r ** 2
        kap = np.where(mask, c, 1.0)
        v_list, l_list, k_mat = solve_3d_fast(kap, device)
        for ax_id in (0, 1, 2):
            others = [(yg, zg), (xg, zg), (xg, yg)][ax_id]
            kaps.append(kap.astype(np.float32))
            vs.append(v_list[ax_id]); lifts.append(l_list[ax_id])
            o1s.append(others[0].astype(np.float32)); o2s.append(others[1].astype(np.float32))
            ks.append(k_mat[ax_id]); axs.append(ax_id)
    kap_t = torch.from_numpy(np.stack(kaps))
    idx = torch.arange(N, dtype=torch.float32)
    ii, jj, kk = idx[:, None, None].expand(N, N, N), idx[None, :, None].expand(N, N, N), idx[None, None, :].expand(N, N, N)
    dist = (torch.minimum(torch.minimum(torch.minimum(ii, jj), kk), torch.minimum(torch.minimum(N - 1 - ii, N - 1 - jj), N - 1 - kk)) / (N / 2.0))[None].expand(kap_t.shape[0], N, N, N)
    lbl, mc = precompute_floating_components_3d(kap_t)
    print(f"  Built 3D {N}^3 synth pack ({kap_t.shape[0]} samples) in {time.time()-t0:.1f}s", flush=True)
    return {
        "kappa": kap_t, "v_true": torch.from_numpy(np.stack(vs)), "lift": torch.from_numpy(np.stack(lifts)),
        "o1": torch.from_numpy(np.stack(o1s)), "o2": torch.from_numpy(np.stack(o2s)), "dist": dist,
        "k_cols": torch.from_numpy(np.stack(ks)), "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc,
        "xg": torch.from_numpy(xg.astype(np.float32))[None],
        "yg": torch.from_numpy(yg.astype(np.float32))[None],
        "zg": torch.from_numpy(zg.astype(np.float32))[None],
    }


def load_3d_ruhr64_pack(npz_path: Path, key_prefix: str, device: torch.device) -> dict:
    z = np.load(npz_path)
    shape = tuple(int(v) for v in z[f"{key_prefix}_shape"])
    bits = np.unpackbits(z[f"{key_prefix}_bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    N = shape[1]
    ax1 = np.linspace(0.0, 1.0, N)
    xg, yg, zg = np.meshgrid(ax1, ax1, ax1, indexing="ij")
    kaps, vs, lifts, o1s, o2s, ks, axs = [], [], [], [], [], [], []
    t0 = time.time()
    for i in range(shape[0]):
        kap = np.where(bits[i], 1000.0, 1.0)
        v_list, l_list, k_mat = solve_3d_fast(kap, device)
        for ax_id in (0, 1, 2):
            others = [(yg, zg), (xg, zg), (xg, yg)][ax_id]
            kaps.append(kap.astype(np.float32))
            vs.append(v_list[ax_id]); lifts.append(l_list[ax_id])
            o1s.append(others[0].astype(np.float32)); o2s.append(others[1].astype(np.float32))
            ks.append(k_mat[ax_id]); axs.append(ax_id)
    kap_t = torch.from_numpy(np.stack(kaps))
    idx = torch.arange(N, dtype=torch.float32)
    ii, jj, kk = idx[:, None, None].expand(N, N, N), idx[None, :, None].expand(N, N, N), idx[None, None, :].expand(N, N, N)
    dist = (torch.minimum(torch.minimum(torch.minimum(ii, jj), kk), torch.minimum(torch.minimum(N - 1 - ii, N - 1 - jj), N - 1 - kk)) / (N / 2.0))[None].expand(kap_t.shape[0], N, N, N)
    lbl, mc = precompute_floating_components_3d(kap_t)
    print(f"  Built 3D Ruhr {N}^3 ({key_prefix}, {kap_t.shape[0]} samples) in {time.time()-t0:.1f}s", flush=True)
    return {
        "kappa": kap_t, "v_true": torch.from_numpy(np.stack(vs)), "lift": torch.from_numpy(np.stack(lifts)),
        "o1": torch.from_numpy(np.stack(o1s)), "o2": torch.from_numpy(np.stack(o2s)), "dist": dist,
        "k_cols": torch.from_numpy(np.stack(ks)), "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc,
        "xg": torch.from_numpy(xg.astype(np.float32))[None],
        "yg": torch.from_numpy(yg.astype(np.float32))[None],
        "zg": torch.from_numpy(zg.astype(np.float32))[None],
    }


# ---------- Exp 13: Nonlinear PDE + Log-Normal Matrix + Non-Convex Star Inclusions ----------
def make_nonlinear_pack(N: int, n_cases: int, seed: int, c_lo: float, c_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    ax = np.linspace(0.0, 1.0, N, dtype=np.float64)
    xg, yg = np.meshgrid(ax, ax, indexing="ij")
    freqs = np.fft.fftfreq(N, d=1.0 / N)
    fx, fy = np.meshgrid(freqs, freqs, indexing="ij")
    filt = np.exp(-0.5 * ((fx ** 2 + fy ** 2) / 4.0 ** 2))
    m = N - 2; h = 1.0 / (N - 1); s2 = 1.0 / (h * h)
    idx = np.arange(m * m).reshape(m, m)

    k_bases, incs, us, lifts, orthos, axs, ks = [], [], [], [], [], [], []
    t0 = time.time()
    for s in range(n_cases):
        c = float(np.exp(rng.uniform(np.log(c_lo), np.log(c_hi))))
        gf = np.fft.ifft2(np.fft.fft2(rng.standard_normal((N, N))) * filt).real
        gf = (gf - gf.mean()) / (gf.std() + 1e-8)
        k_mat = np.exp(0.45 * gf)
        inc = np.zeros((N, N), dtype=bool)
        for _ in range(int(rng.integers(3, 6))):
            cx, cy = rng.uniform(0.18, 0.82, size=2)
            r0 = float(rng.uniform(0.08, 0.14))
            amp = float(rng.uniform(0.20, 0.38))
            km = int(rng.integers(3, 7))
            ph = float(rng.uniform(0, 2 * math.pi))
            dx, dy = xg - cx, yg - cy
            inc |= np.hypot(dx, dy) <= r0 * (1.0 + amp * np.cos(km * np.arctan2(dy, dx) + ph))
        k_base = np.where(inc, c, k_mat)
        for ax_id in (0, 1):
            lft = xg if ax_id == 0 else yg
            ort = yg if ax_id == 0 else xg
            u = lft.copy()
            for _ in range(18):
                kap = k_base * (1.0 + 3.0 * u * u)
                kx = harmonic_mean(kap[:-1, :], kap[1:, :]) * s2
                ky = harmonic_mean(kap[:, :-1], kap[:, 1:]) * s2
                kxm, kxp, kym, kyp = kx[:-1, 1:-1], kx[1:, 1:-1], ky[1:-1, :-1], ky[1:-1, 1:]
                diag = (kxm + kxp + kym + kyp).reshape(-1)
                A = sp.coo_matrix((
                    np.concatenate([diag, -kxm[1:].ravel(), -kxp[:-1].ravel(), -kym[:, 1:].ravel(), -kyp[:, :-1].ravel()]),
                    (np.concatenate([idx.ravel(), idx[1:].ravel(), idx[:-1].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()]),
                     np.concatenate([idx.ravel(), idx[:-1].ravel(), idx[1:].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()]))
                ), shape=(m * m, m * m)).tocsc()
                rhs = -((kxm + kxp + kym + kyp) * lft[1:-1, 1:-1] - kxm * lft[:-2, 1:-1] - kxp * lft[2:, 1:-1] - kym * lft[1:-1, :-2] - kyp * lft[1:-1, 2:])
                u_new = lft.copy()
                u_new[1:-1, 1:-1] += spla.spsolve(A, rhs.ravel()).reshape(m, m)
                if np.max(np.abs(u_new - u)) < 1e-8:
                    u = u_new; break
                u = u_new
            kap_f = k_base * (1.0 + 3.0 * u * u)
            kc = conservative_flux_columns(torch.from_numpy(u)[None], torch.from_numpy(kap_f)[None], kappa_channel=0)[0].numpy()
            k_bases.append(k_base.astype(np.float32))
            incs.append(inc)
            us.append(u.astype(np.float32))
            lifts.append(lft.astype(np.float32))
            orthos.append(ort.astype(np.float32))
            axs.append(ax_id)
            ks.append(kc.astype(np.float32))
    idx_g = torch.arange(N, dtype=torch.float32)
    ii, jj = idx_g[:, None].expand(N, N), idx_g[None, :].expand(N, N)
    dist = (torch.minimum(torch.minimum(ii, jj), torch.minimum(N - 1 - ii, N - 1 - jj)) / (N / 2.0))[None].expand(len(k_bases), N, N)
    print(f"  Built Nonlinear+LogNormal+Star pack ({len(k_bases)} samples, {N}x{N}) in {time.time()-t0:.1f}s", flush=True)
    return {
        "k_base": torch.from_numpy(np.stack(k_bases)),
        "inc_mask": torch.from_numpy(np.stack(incs)),
        "u_true": torch.from_numpy(np.stack(us)),
        "lift": torch.from_numpy(np.stack(lifts)),
        "ortho": torch.from_numpy(np.stack(orthos)),
        "dist": dist,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "k_cols": torch.from_numpy(np.stack(ks)),
        "x_grid": torch.from_numpy(xg.astype(np.float32))[None],
        "y_grid": torch.from_numpy(yg.astype(np.float32))[None],
    }


def nonlinear_dipole_keff(u: torch.Tensor, k_base: torch.Tensor, inc_mask: torch.Tensor, xg: torch.Tensor, yg: torch.Tensor) -> torch.Tensor:
    B, H, W = u.shape
    h = 1.0 / max(W - 1, 1)
    kap = k_base * (1.0 + 3.0 * u * u)
    kx, ky = face_conductivities(kap, kappa_channel=0)
    c_inc = inc_mask[:, 1:-1, 1:-1]
    inc_int = torch.zeros_like(inc_mask); inc_int[:, 1:-1, 1:-1] = c_inc
    qx_n = kx[:, :, 1:-1] * (u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / h
    qy_n = ky[:, 1:-1, :] * (u[:, 1:-1, 1:] - u[:, 1:-1, :-1]) / h
    kx_ext = torch.where(inc_int[:, :-1, 1:-1] & inc_int[:, 1:, 1:-1], torch.zeros_like(qx_n), qx_n).sum((1, 2)) / float((H - 1) * (W - 2))
    ky_ext = torch.where(inc_int[:, 1:-1, :-1] & inc_int[:, 1:-1, 1:], torch.zeros_like(qy_n), qy_n).sum((1, 2)) / float((H - 2) * (W - 1))
    kxm, kxp = kx[:, :-1, 1:-1], kx[:, 1:, 1:-1]
    kym, kyp = ky[:, 1:-1, :-1], ky[:, 1:-1, 1:]
    uc = u[:, 1:-1, 1:-1]
    val = (
        torch.where(~inc_int[:, :-2, 1:-1], kxm * (uc - u[:, :-2, 1:-1]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 2:, 1:-1], kxp * (uc - u[:, 2:, 1:-1]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 1:-1, :-2], kym * (uc - u[:, 1:-1, :-2]), torch.zeros_like(uc))
        + torch.where(~inc_int[:, 1:-1, 2:], kyp * (uc - u[:, 1:-1, 2:]), torch.zeros_like(uc))
    ) / (h * h)
    ap = torch.zeros_like(u)
    ap[:, 1:-1, 1:-1] = torch.where(c_inc, val, torch.zeros_like(val))
    kd0 = -(xg * ap).sum((1, 2)) * h / float(W - 2)
    kd1 = -(yg * ap).sum((1, 2)) * h / float(H - 2)
    return torch.stack([kx_ext + kd0, ky_ext + kd1], dim=-1)


class NonlinearBCNO(nn.Module):
    def __init__(self, steps: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.green = DirichletGreen2D()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(10, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            ) for _ in range(steps)
        ])
        for b in self.blocks:
            nn.init.normal_(b[-1].weight, std=1e-2); nn.init.zeros_(b[-1].bias)
        self.gamma = nn.Parameter(torch.full((steps,), -2.0))

    def forward(self, k_base, inc_mask, lift, ortho, dist):
        scale_h = float(k_base.shape[-1] - 1)
        interior = torch.ones_like(lift)
        interior[:, 0, :] = 0; interior[:, -1, :] = 0; interior[:, :, 0] = 0; interior[:, :, -1] = 0
        sigma = torch.zeros_like(lift)
        u = lift.clone()
        beta0 = (k_base - 1.0) / (k_base + 1.0)
        for t, blk in enumerate(self.blocks):
            kap = k_base * (1.0 + 3.0 * u * u)
            kx, ky = face_conductivities(kap, kappa_channel=0)
            kxm, kxp, kym, kyp = kx[:, :-1, 1:-1], kx[:, 1:, 1:-1], ky[:, 1:-1, :-1], ky[:, 1:-1, 1:]
            diag = (kxm + kxp + kym + kyp).clamp_min(1e-12)
            r_tilde = torch.zeros_like(u)
            uc = u[:, 1:-1, 1:-1]
            r_tilde[:, 1:-1, 1:-1] = uc - (kxm * u[:, :-2, 1:-1] + kxp * u[:, 2:, 1:-1] + kym * u[:, 1:-1, :-2] + kyp * u[:, 1:-1, 2:]) / diag
            r_flux = r_tilde * scale_h
            beta = (kap - 1.0) / (kap + 1.0)
            log_k = torch.log(kap.clamp(1e-4, 1e6)) / 10.0
            feats = torch.stack([beta, beta0, log_k, inc_mask.float(), lift, ortho, dist, u, r_flux, sigma], dim=1)
            sigma = sigma + (blk(feats)[:, 0] + self.gamma[t] * r_flux) * interior
            u = lift + self.green(sigma * scale_h)
        return u


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip_exp10", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parent
    generate_synthetic_benchmarks(root)
    ep_fe2 = 2 if args.smoke else 80
    ep_big = 2 if args.smoke else args.epochs
    ep_3d = 2 if args.smoke else 35
    ep_nl = 2 if args.smoke else 60
    results: dict[str, object] = {"device": str(device), "smoke": args.smoke}

    # ==== Exp 10: Constitutive Admissibility + 16x16 Two-Scale FE^2 ====
    if not args.skip_exp10:
        print("\n=== Exp 10: Constitutive Admissibility & Two-Scale FE^2 ===", flush=True)
        M_macro = 6 if args.smoke else 16
        f_tr = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "train")
        f_id = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "test")
        f_od = pack_from_pt(root / "rve_fiber_columns_ood_N128_f64.pt", "test")
        fe2_pk = build_fe2_micro_suite(M_macro=M_macro, N_micro=128, seed=2026)
        n_el = M_macro * M_macro
        K_true_macro = fe2_pk["k_cols"].view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_macro, M_macro, 2, 2)
        T_ref, flux_ref, _ = solve_macro_q1_plate(K_true_macro, M=M_macro)
        exp10 = {"micro_solve_s": round(fe2_pk["micro_solve_s"], 2), "models": {}}
        for name, is_b, use_k, use_sig, ctor in [
            ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
            ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
            ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
            ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
            ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
            ("FNO2d", False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
        ]:
            torch.manual_seed(args.seed)
            mdl = ctor().float()
            train_model(mdl, is_b, f_tr, epochs=ep_fe2, batch_size=8, lr=(3e-3 if is_b else 2e-3), seed=args.seed, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
            k_od_p, _, _, _ = predict_keff_cols(mdl, f_od, is_b, device)
            aud = constitutive_audit(k_od_p, f_od["kappa_raw"], f_od["load_axis"])
            k_fe2_p, _, _, inf_s = predict_keff_cols(mdl, fe2_pk, is_b, device)
            K_surr = k_fe2_p.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_macro, M_macro, 2, 2)
            T_surr, flux_surr, spd_ok = solve_macro_q1_plate(K_surr, M=M_macro)
            t_err = float(np.linalg.norm(T_surr - T_ref) / np.linalg.norm(T_ref)) if np.isfinite(T_surr).all() else float("nan")
            f_err = float(abs(flux_surr[0] - flux_ref[0]) / abs(flux_ref[0])) if np.isfinite(flux_surr).all() else float("nan")
            rve_err = float((torch.linalg.vector_norm(k_fe2_p - fe2_pk["k_cols"], dim=1) / torch.linalg.vector_norm(fe2_pk["k_cols"], dim=1)).mean())
            exp10["models"][name] = {"audit_ood": aud, "fe2": {"inf_s": round(inf_s, 3), "speedup": round(fe2_pk["micro_solve_s"]/max(inf_s,1e-4), 1), "rve_err": rve_err, "macro_T_L2": t_err, "macro_flux_err": f_err, "spd_all": spd_ok}}
            print(f"  [Exp10 {name:8s}] Sym={aud['OnsagerSymErr']:.3f} SPD={aud['SPDRate']:.2f} VR={aud['VoigtReussRate']:.2f} | FE^2 inf={inf_s:.3f}s rveErr={rve_err:.3f} macroT={t_err:.4f} flux={f_err:.4f}", flush=True)
        results["exp10_fe2_audit"] = exp10

    # ==== Exp 11: Large-Sample N=128 Benchmark (1000 train / 200 ID / 200 OOD) ====
    print("\n=== Exp 11: Large-Sample N=128 (1000 train / 200 ID / 200 OOD) ===", flush=True)
    n_tr11 = 8 if args.smoke else 500   # x2 axes = 1000 train!
    n_te11 = 4 if args.smoke else 100   # x2 axes = 200 ID / 200 OOD!
    big_tr = build_large_2d_pack(128, n_tr11, 1101, 10.0, 80.0, 0.30, 0.58, device)
    big_id = build_large_2d_pack(128, n_te11, 1102, 10.0, 80.0, 0.30, 0.58, device)
    big_od = build_large_2d_pack(128, n_te11, 1103, 100.0, 400.0, 0.45, 0.72, device)
    exp11 = {}
    for name, is_b, use_k, use_sig, ctor in [
        ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
        ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("Transolver2d", False, False, False, lambda: Transolver2d(5, dim=64, depth=4, heads=4, n_slices=32)),
        ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
        ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ("FNO2d", False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
    ]:
        torch.manual_seed(args.seed)
        mdl = ctor().float()
        train_model(mdl, is_b, big_tr, epochs=ep_big, batch_size=16, lr=(3e-3 if is_b else 2e-3), seed=args.seed, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
        m_id = evaluate_pack(mdl, big_id, is_b, device, use_dipole=is_b)
        m_od = evaluate_pack(mdl, big_od, is_b, device, use_dipole=is_b)
        exp11[name] = {"id": m_id, "ood": m_od}
        print(f"  [Exp11 {name:12s}] ID Keff={m_id['KeffRelErr']:.3f} Fluct={m_id['Fluctuation_L2']:.3f} | OOD Keff={m_od['KeffRelErr']:.3f} Fluct={m_od['Fluctuation_L2']:.3f}", flush=True)
    results["exp11_large_sample_1000"] = exp11

    # ==== Exp 12: Large 3D 64^3 (120 train / 30 ID / 30 OOD + 64^3 Real Ruhr CT) ====
    print("\n=== Exp 12: True 3D 64^3 Composites & Real 64^3 Ruhr CT ===", flush=True)
    s64_tr = build_3d_64_synth_pack(64, 2 if args.smoke else 40, 2101, 10.0, 80.0, device)
    s64_id = build_3d_64_synth_pack(64, 1 if args.smoke else 10, 2102, 10.0, 80.0, device)
    s64_od = build_3d_64_synth_pack(64, 1 if args.smoke else 10, 2103, 200.0, 1000.0, device)
    r64_tr = load_3d_ruhr64_pack(root / "rve_3d_ruhr64_compact.npz", "tr64", device)
    r64_te = load_3d_ruhr64_pack(root / "rve_3d_ruhr64_compact.npz", "te64", device)
    if args.smoke:
        for pk in (r64_tr, r64_te):
            for k in ("kappa", "v_true", "lift", "o1", "o2", "dist", "k_cols", "load_axis", "labels"):
                pk[k] = pk[k][:3]

    def run_3d_64(tag, tr_p, te_dict):
        res = {}
        for name, is_b, ctor in [
            ("BCNO-3D", True, lambda: BCNO3D(steps=3, hidden=16)),
            ("ResUNet-3D", False, lambda: ResUNet3dSmall(6, width=16)),
            ("FNO-3D", False, lambda: FNO3dSmall(6, width=16, modes=8, depth=4)),
        ]:
            torch.manual_seed(args.seed)
            mdl = ctor().float().to(device)
            opt = torch.optim.AdamW(mdl.parameters(), lr=(3e-3 if is_b else 2e-3), weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_3d)
            B = tr_p["kappa"].shape[0]; h3 = 1.0 / 63.0
            xg, yg, zg = tr_p["xg"].to(device), tr_p["yg"].to(device), tr_p["zg"].to(device)
            for _ in range(ep_3d):
                mdl.train(); perm = torch.randperm(B)
                for s in range(0, B, 3):
                    idx = perm[s : min(s + 3, B)]
                    kap = tr_p["kappa"][idx].to(device); lft = tr_p["lift"][idx].to(device)
                    o1 = tr_p["o1"][idx].to(device); o2 = tr_p["o2"][idx].to(device); dst = tr_p["dist"][idx].to(device)
                    vt = tr_p["v_true"][idx].to(device)
                    up, sig = mdl(kap, lft, o1, o2, dst, tr_p["labels"][idx].to(device), tr_p["max_comp"]) if is_b else (mdl(kap, lft, o1, o2, dst), None)
                    vp = up - lft
                    loss = (torch.sqrt(((vp - vt) ** 2).sum((1, 2, 3)) + 1e-8) / torch.sqrt((vt ** 2).sum((1, 2, 3)) + 1e-6).clamp_min(1e-2)).mean()
                    if is_b:
                        kp = dipole_keff_columns_3d(up, kap, xg, yg, zg)
                        bi = torch.arange(idx.shape[0], device=device)
                        ax_b = tr_p["load_axis"][idx].to(device)
                        kt_d = tr_p["k_cols"][idx].to(device)[bi, ax_b]
                        loss = loss + 0.25 * ((kp[bi, ax_b] - kt_d).abs() / kt_d.abs().clamp_min(0.1)).mean()
                        strip = boundary_strip_mask_3d(kap)
                        sig_t = torch.zeros_like(vt)
                        sig_t[:, 1:-1, 1:-1, 1:-1] = (6.0 * vt[:, 1:-1, 1:-1, 1:-1] - vt[:, :-2, 1:-1, 1:-1] - vt[:, 2:, 1:-1, 1:-1] - vt[:, 1:-1, :-2, 1:-1] - vt[:, 1:-1, 2:, 1:-1] - vt[:, 1:-1, 1:-1, :-2] - vt[:, 1:-1, 1:-1, 2:]) / h3
                        loss = loss + 0.25 * ((((sig - sig_t) * strip) ** 2).sum((1, 2, 3)) / strip.sum((1, 2, 3)).clamp_min(1.0)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            mdl.eval()
            res[name] = {}
            for sk, spk in te_dict.items():
                ups = []; t0 = time.time()
                with torch.no_grad():
                    for s in range(0, spk["kappa"].shape[0], 3):
                        e = min(s + 3, spk["kappa"].shape[0])
                        kap = spk["kappa"][s:e].to(device); lft = spk["lift"][s:e].to(device)
                        o1 = spk["o1"][s:e].to(device); o2 = spk["o2"][s:e].to(device); dst = spk["dist"][s:e].to(device)
                        up = mdl(kap, lft, o1, o2, dst, spk["labels"][s:e].to(device), spk["max_comp"])[0] if is_b else mdl(kap, lft, o1, o2, dst)
                        ups.append(up.cpu())
                if device.type == "cuda": torch.cuda.synchronize()
                inf_ms = (time.time() - t0) * 1000.0 / spk["kappa"].shape[0]
                u_p = torch.cat(ups, 0); v_p = u_p - spk["lift"]
                k_p = dipole_keff_columns_3d(u_p.double(), spk["kappa"].double(), spk["xg"].double(), spk["yg"].double(), spk["zg"].double()).float() if is_b else conservative_flux_columns_3d(u_p.double(), spk["kappa"].double(), kappa_channel=0).float()
                bi = torch.arange(spk["kappa"].shape[0]); kt = spk["k_cols"][bi, spk["load_axis"]]
                diag_err = float(((k_p[bi, spk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean())
                vn = torch.sqrt((spk["v_true"] ** 2).sum((1, 2, 3))); vv = vn > 1e-4
                fl = float((torch.sqrt(((v_p[vv] - spk["v_true"][vv]) ** 2).sum((1, 2, 3)) + 1e-12) / vn[vv]).mean()) if bool(vv.any()) else 0.0
                res[name][sk] = {"DiagKeff": diag_err, "FluctL2": fl, "InferMs": inf_ms}
                print(f"  [Exp12 {tag:8s} | {sk:4s} | {name:10s}] DiagKeff={diag_err:.3f} FluctL2={fl:.3f} ms={inf_ms:.2f}", flush=True)
        return res

    results["exp12_3d_64_synth"] = run_3d_64("Synth64", s64_tr, {"id": s64_id, "ood": s64_od})
    results["exp12_3d_64_ruhr"] = run_3d_64("Ruhr64", r64_tr, {"test": r64_te})

    # ==== Exp 13: Nonlinear PDE + Log-Normal Matrix + Non-Convex Star Inclusions ====
    print("\n=== Exp 13: Nonlinear PDE + Continuous Log-Normal Matrix + Non-Convex Star ===", flush=True)
    nl_tr = make_nonlinear_pack(64, 4 if args.smoke else 200, 3101, 20.0, 500.0)
    nl_id = make_nonlinear_pack(64, 2 if args.smoke else 50, 3102, 20.0, 500.0)
    nl_od = make_nonlinear_pack(64, 2 if args.smoke else 50, 3103, 1000.0, 10000.0)
    exp13 = {}
    for name, is_nl_bcno, ctor in [
        ("BCNO-NL", True, lambda: NonlinearBCNO(steps=4, hidden=32)),
        ("MgNO2d", False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("FFNO2d", False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("CNO2d", False, lambda: CNO2dWrapper(5, width=32)),
        ("ResUNet", False, lambda: ResUNetSmall(5, width=32)),
        ("FNO2d", False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
    ]:
        torch.manual_seed(args.seed)
        mdl = ctor().float().to(device)
        opt = torch.optim.AdamW(mdl.parameters(), lr=(3e-3 if is_nl_bcno else 2e-3), weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_nl)
        B = nl_tr["k_base"].shape[0]
        xg, yg = nl_tr["x_grid"].to(device), nl_tr["y_grid"].to(device)
        for _ in range(ep_nl):
            mdl.train(); perm = torch.randperm(B)
            for s in range(0, B, 16):
                idx = perm[s : min(s + 16, B)]
                kb = nl_tr["k_base"][idx].to(device); inc = nl_tr["inc_mask"][idx].to(device)
                lft = nl_tr["lift"][idx].to(device); ort = nl_tr["ortho"][idx].to(device); dst = nl_tr["dist"][idx].to(device)
                ut = nl_tr["u_true"][idx].to(device); vt = ut - lft
                up = mdl(kb, inc, lft, ort, dst) if is_nl_bcno else mdl(kb, lft, ort, dst)
                vp = up - lft
                loss = (torch.sqrt(((vp - vt) ** 2).sum((1, 2)) + 1e-8) / torch.sqrt((vt ** 2).sum((1, 2)) + 1e-6).clamp_min(1e-2)).mean()
                if is_nl_bcno:
                    kp = nonlinear_dipole_keff(up, kb, inc, xg, yg)
                    bi = torch.arange(idx.shape[0], device=device); ax_b = nl_tr["load_axis"][idx].to(device)
                    kt_d = nl_tr["k_cols"][idx].to(device)[bi, ax_b]
                    loss = loss + 0.25 * ((kp[bi, ax_b] - kt_d).abs() / kt_d.abs().clamp_min(0.1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        mdl.eval()
        nl_out = {}
        for sk, spk in [("id", nl_id), ("ood", nl_od)]:
            ups = []
            with torch.no_grad():
                for s in range(0, spk["k_base"].shape[0], 16):
                    e = min(s + 16, spk["k_base"].shape[0])
                    kb = spk["k_base"][s:e].to(device); inc = spk["inc_mask"][s:e].to(device)
                    lft = spk["lift"][s:e].to(device); ort = spk["ortho"][s:e].to(device); dst = spk["dist"][s:e].to(device)
                    up = mdl(kb, inc, lft, ort, dst) if is_nl_bcno else mdl(kb, lft, ort, dst)
                    ups.append(up.cpu())
            u_p = torch.cat(ups, 0); v_p = u_p - spk["lift"]; vt = spk["u_true"] - spk["lift"]
            kap_p = spk["k_base"].double() * (1.0 + 3.0 * u_p.double() ** 2)
            k_dip = nonlinear_dipole_keff(u_p.double(), spk["k_base"].double(), spk["inc_mask"], spk["x_grid"].double(), spk["y_grid"].double()).float()
            k_nv = conservative_flux_columns(u_p.double(), kap_p, kappa_channel=0).float()
            bi = torch.arange(spk["k_base"].shape[0]); kt = spk["k_cols"][bi, spk["load_axis"]]
            kd_e = float(((k_dip[bi, spk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean())
            kn_e = float(((k_nv[bi, spk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean())
            fl = float((torch.sqrt(((v_p - vt) ** 2).sum((1, 2)) + 1e-8) / torch.sqrt((vt ** 2).sum((1, 2)) + 1e-6).clamp_min(1e-2)).mean())
            nl_out[sk] = {"DiagDip": kd_e, "DiagNaive": kn_e, "FluctL2": fl}
        exp13[name] = nl_out
        print(f"  [Exp13 {name:10s}] ID Keff={nl_out['id']['DiagDip' if is_nl_bcno else 'DiagNaive']:.3f} Fluct={nl_out['id']['FluctL2']:.3f} | OOD Dip={nl_out['ood']['DiagDip']:.3f} Naive={nl_out['ood']['DiagNaive']:.3f} Fluct={nl_out['ood']['FluctL2']:.3f}", flush=True)
    results["exp13_nonlinear_lognormal_star"] = exp13

    (root / "bcno_scale_and_fe2_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: bcno_scale_and_fe2_results.json", flush=True)


if __name__ == "__main__":
    main()
