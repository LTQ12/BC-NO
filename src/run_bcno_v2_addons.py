"""BC-NO v2 Add-On Suite: 3D RVE + 3D Real CT + 3D PCG Warm-Start + 2D PBC + 2D Elasticity + Percolating Adaptive Readout.

Runs cleanly on Colab GPU (flat /content directory).
Outputs:
  bcno_v2_addons_results.json
  bcno_v2_addons_tables.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_2d import FFNO2d, FNO2dSmall, MgNO2dWrapper, ResUNetSmall
from bcno_elasticity_2d import (
    BCNO_Elas_2D,
    FNO_Elas_2D,
    ResUNet_Elas_2D,
    precompute_elasticity_components,
    q1_stress_dipole_readout,
    solve_q1_elasticity_periodic,
)
from bcno_model_2d import (
    BCNO2D,
    BCNO_PBC_2D,
    dipole_keff_columns,
    periodic_dipole_keff,
    precompute_periodic_components,
)
from bcno_model_3d import (
    BCNO3D,
    FNO3dSmall,
    ResUNet3dSmall,
    boundary_strip_mask_3d,
    dipole_keff_columns_3d,
    jacobi_pcg_3d,
    precompute_floating_components_3d,
)
from colab_data_gen import generate_synthetic_benchmarks, make_fiber_columns
from rve_discrete_operator_2d import harmonic_mean
from rve_discrete_operator_3d import conservative_flux_columns_3d, face_conductivities_3d
from run_bcno_paper_suite import agg_seeds, evaluate_pack, pack_from_pt, relative_l2, train_model


# ---------- Helpers for 2D Periodic Scalar Solver ----------
def solve_2d_periodic_scalar(kappa: np.ndarray, load_axis: int) -> tuple[np.ndarray, np.ndarray]:
    N = kappa.shape[0]
    hp = 1.0 / float(N)
    kxp = harmonic_mean(kappa, np.roll(kappa, -1, axis=0)) / (hp * hp)
    kyp = harmonic_mean(kappa, np.roll(kappa, -1, axis=1)) / (hp * hp)
    kxm = np.roll(kxp, 1, axis=0)
    kym = np.roll(kyp, 1, axis=1)
    diagp = kxm + kxp + kym + kyp
    idx = np.arange(N * N).reshape(N, N)
    rr = [idx.reshape(-1), idx.reshape(-1), idx.reshape(-1), idx.reshape(-1), idx.reshape(-1)]
    cc = [idx.reshape(-1), np.roll(idx, 1, 0).reshape(-1), np.roll(idx, -1, 0).reshape(-1), np.roll(idx, 1, 1).reshape(-1), np.roll(idx, -1, 1).reshape(-1)]
    vv = [diagp.reshape(-1), -kxm.reshape(-1), -kxp.reshape(-1), -kym.reshape(-1), -kyp.reshape(-1)]
    A = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(N * N, N * N)).tolil()
    rhs = ((kxp - kxm) if load_axis == 0 else (kyp - kym)) * hp
    A[0, :] = 0.0; A[0, 0] = 1.0; rhs[0, 0] = 0.0
    v = spla.spsolve(A.tocsc(), rhs.reshape(-1)).reshape(N, N)
    v -= v.mean()
    gx_tot = (1.0 if load_axis == 0 else 0.0) + (np.roll(v, -1, 0) - v) / hp
    gy_tot = (1.0 if load_axis == 1 else 0.0) + (np.roll(v, -1, 1) - v) / hp
    k_cols = np.array([float(np.mean(kxp * (hp * hp) * gx_tot)), float(np.mean(kyp * (hp * hp) * gy_tot))], dtype=np.float32)
    return v.astype(np.float32), k_cols


def build_periodic_2d_pack(N: int, n_cases: int, seed: int, c_lo: float, c_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    kaps, vs, ks, axs = [], [], [], []
    for s in range(n_cases):
        c = float(np.exp(rng.uniform(np.log(c_lo), np.log(c_hi))))
        if rng.random() < 0.25:
            c = 1.0 / c
        vf = float(rng.uniform(0.30, 0.55))
        # Periodic wrapped ellipses
        ax = np.arange(N) / float(N)
        gx, gy = np.meshgrid(ax, ax, indexing="ij")
        nc = int(rng.integers(3, 6))
        nr = int(rng.integers(3, 6))
        rad = math.sqrt(vf / (math.pi * nc * nr))
        mask = np.zeros((N, N), dtype=bool)
        for ci in range(nc):
            for ri in range(nr):
                cx = ((ci + rng.uniform(0.1, 0.9)) / nc) % 1.0
                cy = ((ri + rng.uniform(0.1, 0.9)) / nr) % 1.0
                rx = rad * float(rng.uniform(0.75, 1.35))
                ry = rad * float(rng.uniform(0.75, 1.35))
                dx = ((gx - cx + 0.5) % 1.0) - 0.5
                dy = ((gy - cy + 0.5) % 1.0) - 0.5
                th = float(rng.uniform(-0.35, 0.35))
                xp = math.cos(th) * dx + math.sin(th) * dy
                yp = -math.sin(th) * dx + math.cos(th) * dy
                mask |= (xp / rx) ** 2 + (yp / ry) ** 2 <= 1.0
        kap = np.where(mask, c, 1.0)
        for ax_id in (0, 1):
            v, k_c = solve_2d_periodic_scalar(kap, ax_id)
            kaps.append(kap.astype(np.float32))
            vs.append(v)
            ks.append(k_c)
            axs.append(ax_id)
    kap_t = torch.from_numpy(np.stack(kaps))
    lbl, mc, nonperc, dx_w, dy_w = precompute_periodic_components(kap_t)
    return {
        "kappa": kap_t,
        "v_true": torch.from_numpy(np.stack(vs)),
        "k_cols": torch.from_numpy(np.stack(ks)),
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc, "nonperc": nonperc, "dx_wrap": dx_w, "dy_wrap": dy_w,
    }


# ---------- Helpers for 3D Scalar Solver ----------
def solve_3d_dirichlet_all_axes(kappa: np.ndarray, axes: tuple[int, ...] = (0, 1, 2)) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Factorize 3D sparse matrix ONCE per geometry (or GPU PCG if CUDA available) and solve all 3 axes."""
    n = kappa.shape[0]
    m = n - 2
    h = 1.0 / (n - 1)
    s2 = 1.0 / (h * h)
    ax = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xg, yg, zg = np.meshgrid(ax, ax, ax, indexing="ij")
    grids = (xg, yg, zg)

    if torch.cuda.is_available():
        kap_t = torch.from_numpy(kappa).cuda()[None]
        out = []
        for load_axis in axes:
            lift = grids[load_axis]
            lft_t = torch.from_numpy(lift).cuda()[None]
            u_t, _, _ = jacobi_pcg_3d(kap_t, lft_t, tol=1e-9, max_iters=3000)
            u_t_cpu = u_t.cpu()
            k_cols = conservative_flux_columns_3d(u_t_cpu, torch.from_numpy(kappa)[None], kappa_channel=0)[0].numpy()
            out.append(((u_t_cpu[0].numpy() - lift).astype(np.float32), lift.astype(np.float32), k_cols.astype(np.float32)))
        return out

    kx, ky, kz = [t[0].numpy() * s2 for t in face_conductivities_3d(torch.from_numpy(kappa)[None], kappa_channel=0)]
    kxm, kxp = kx[:-1, 1:-1, 1:-1], kx[1:, 1:-1, 1:-1]
    kym, kyp = ky[1:-1, :-1, 1:-1], ky[1:-1, 1:, 1:-1]
    kzm, kzp = kz[1:-1, 1:-1, :-1], kz[1:-1, 1:-1, 1:]
    diag = (kxm + kxp + kym + kyp + kzm + kzp).reshape(-1)
    idx = np.arange(m ** 3).reshape(m, m, m)
    rr = [idx.reshape(-1), idx[1:].reshape(-1), idx[:-1].reshape(-1), idx[:, 1:].reshape(-1), idx[:, :-1].reshape(-1), idx[:, :, 1:].reshape(-1), idx[:, :, :-1].reshape(-1)]
    cc = [idx.reshape(-1), idx[:-1].reshape(-1), idx[1:].reshape(-1), idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1), idx[:, :, :-1].reshape(-1), idx[:, :, 1:].reshape(-1)]
    vv = [diag, -kxm[1:].reshape(-1), -kxp[:-1].reshape(-1), -kym[:, 1:].reshape(-1), -kyp[:, :-1].reshape(-1), -kzm[:, :, 1:].reshape(-1), -kzp[:, :, :-1].reshape(-1)]
    A = sp.coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))), shape=(m ** 3, m ** 3)).tocsc()
    lu = spla.splu(A)
    out = []
    for load_axis in axes:
        lift = grids[load_axis]
        rhs = -(
            (kxm + kxp + kym + kyp + kzm + kzp) * lift[1:-1, 1:-1, 1:-1]
            - kxm * lift[:-2, 1:-1, 1:-1] - kxp * lift[2:, 1:-1, 1:-1]
            - kym * lift[1:-1, :-2, 1:-1] - kyp * lift[1:-1, 2:, 1:-1]
            - kzm * lift[1:-1, 1:-1, :-2] - kzp * lift[1:-1, 1:-1, 2:]
        )
        v_in = lu.solve(rhs.reshape(-1)).reshape(m, m, m)
        u = lift.copy()
        u[1:-1, 1:-1, 1:-1] += v_in
        k_cols = conservative_flux_columns_3d(torch.from_numpy(u)[None], torch.from_numpy(kappa)[None], kappa_channel=0)[0].numpy()
        out.append(((u - lift).astype(np.float32), lift.astype(np.float32), k_cols.astype(np.float32)))
    return out


def make_3d_synthetic_pack(N: int, n_cases: int, seed: int, c_lo: float, c_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    ax1 = np.linspace(0.0, 1.0, N)
    xg, yg, zg = np.meshgrid(ax1, ax1, ax1, indexing="ij")
    kaps, vs, lifts, o1s, o2s, ks, axs = [], [], [], [], [], [], []
    for s in range(n_cases):
        c = float(np.exp(rng.uniform(np.log(c_lo), np.log(c_hi))))
        if rng.random() < 0.25:
            c = 1.0 / c
        mask = np.zeros((N, N, N), dtype=bool)
        if s % 2 == 0:  # ellipsoids
            for _ in range(int(rng.integers(4, 9))):
                ctr = rng.uniform(0.12, 0.88, size=3)
                rad = rng.uniform(0.08, 0.20, size=3)
                mask |= ((xg - ctr[0]) / rad[0]) ** 2 + ((yg - ctr[1]) / rad[1]) ** 2 + ((zg - ctr[2]) / rad[2]) ** 2 <= 1.0
        else:  # 3D fibers
            f_ax = int(rng.integers(0, 3))
            t_pair = [(yg, zg), (xg, zg), (xg, yg)][f_ax]
            for _ in range(int(rng.integers(4, 8))):
                ctr = rng.uniform(0.12, 0.88, size=2)
                r = float(rng.uniform(0.06, 0.13))
                mask |= (t_pair[0] - ctr[0]) ** 2 + (t_pair[1] - ctr[1]) ** 2 <= r ** 2
        kap = np.where(mask, c, 1.0)
        for ax_id, (v, lft, k_c) in enumerate(solve_3d_dirichlet_all_axes(kap)):
            others = [(yg, zg), (xg, zg), (xg, yg)][ax_id]
            kaps.append(kap.astype(np.float32))
            vs.append(v); lifts.append(lft)
            o1s.append(others[0].astype(np.float32)); o2s.append(others[1].astype(np.float32))
            ks.append(k_c); axs.append(ax_id)
    kap_t = torch.from_numpy(np.stack(kaps))
    idx = torch.arange(N, dtype=torch.float32)
    ii, jj, kk = idx[:, None, None].expand(N, N, N), idx[None, :, None].expand(N, N, N), idx[None, None, :].expand(N, N, N)
    dist = (torch.minimum(torch.minimum(torch.minimum(ii, jj), kk), torch.minimum(torch.minimum(N - 1 - ii, N - 1 - jj), N - 1 - kk)) / (N / 2.0))[None].expand(kap_t.shape[0], N, N, N)
    lbl, mc = precompute_floating_components_3d(kap_t)
    xg_t, yg_t, zg_t = torch.from_numpy(xg.astype(np.float32))[None], torch.from_numpy(yg.astype(np.float32))[None], torch.from_numpy(zg.astype(np.float32))[None]
    return {
        "kappa": kap_t, "v_true": torch.from_numpy(np.stack(vs)), "lift": torch.from_numpy(np.stack(lifts)),
        "o1": torch.from_numpy(np.stack(o1s)), "o2": torch.from_numpy(np.stack(o2s)), "dist": dist,
        "k_cols": torch.from_numpy(np.stack(ks)), "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc, "xg": xg_t, "yg": yg_t, "zg": zg_t,
    }


def load_3d_ruhr_pack(npz_path: Path, key_prefix: str, solve_labels: bool = True) -> dict:
    z = np.load(npz_path)
    shape = tuple(int(v) for v in z[f"{key_prefix}_shape"])
    bits = np.unpackbits(z[f"{key_prefix}_bits"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    N = shape[1]
    ax1 = np.linspace(0.0, 1.0, N)
    xg, yg, zg = np.meshgrid(ax1, ax1, ax1, indexing="ij")
    kaps, vs, lifts, o1s, o2s, ks, axs = [], [], [], [], [], [], []
    for i in range(shape[0]):
        kap = np.where(bits[i], 1000.0, 1.0)
        solved_all = solve_3d_dirichlet_all_axes(kap) if solve_labels else None
        for ax_id in (0, 1, 2):
            others = [(yg, zg), (xg, zg), (xg, yg)][ax_id]
            if solve_labels:
                v, lft, k_c = solved_all[ax_id]
            else:
                lft = (xg, yg, zg)[ax_id].astype(np.float32)
                v = np.zeros((N, N, N), dtype=np.float32)
                k_c = np.zeros(3, dtype=np.float32)
            kaps.append(kap.astype(np.float32))
            vs.append(v); lifts.append(lft)
            o1s.append(others[0].astype(np.float32)); o2s.append(others[1].astype(np.float32))
            ks.append(k_c); axs.append(ax_id)
    kap_t = torch.from_numpy(np.stack(kaps))
    idx = torch.arange(N, dtype=torch.float32)
    ii, jj, kk = idx[:, None, None].expand(N, N, N), idx[None, :, None].expand(N, N, N), idx[None, None, :].expand(N, N, N)
    dist = (torch.minimum(torch.minimum(torch.minimum(ii, jj), kk), torch.minimum(torch.minimum(N - 1 - ii, N - 1 - jj), N - 1 - kk)) / (N / 2.0))[None].expand(kap_t.shape[0], N, N, N)
    lbl, mc = precompute_floating_components_3d(kap_t)
    return {
        "kappa": kap_t, "v_true": torch.from_numpy(np.stack(vs)), "lift": torch.from_numpy(np.stack(lifts)),
        "o1": torch.from_numpy(np.stack(o1s)), "o2": torch.from_numpy(np.stack(o2s)), "dist": dist,
        "k_cols": torch.from_numpy(np.stack(ks)), "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc,
        "xg": torch.from_numpy(xg.astype(np.float32))[None],
        "yg": torch.from_numpy(yg.astype(np.float32))[None],
        "zg": torch.from_numpy(zg.astype(np.float32))[None],
    }


# ---------- Helpers for 2D Elasticity ----------
def build_elasticity_pack(N: int, n_cases: int, seed: int, E_lo: float, E_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    ax = (np.arange(N) + 0.5) / float(N)
    ex, ey = np.meshgrid(ax, ax, indexing="ij")
    E_macros = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
    ]
    Es, Emacs, vs, sigs = [], [], [], []
    for s in range(n_cases):
        E_val = float(np.exp(rng.uniform(np.log(E_lo), np.log(E_hi))))
        mask = np.zeros((N, N), dtype=bool)
        for _ in range(int(rng.integers(3, 8))):
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            rx, ry = rng.uniform(0.08, 0.18, size=2)
            th = float(rng.uniform(-0.5, 0.5))
            dx = ((ex - cx + 0.5) % 1.0) - 0.5
            dy = ((ey - cy + 0.5) % 1.0) - 0.5
            xp = math.cos(th) * dx + math.sin(th) * dy
            yp = -math.sin(th) * dx + math.cos(th) * dy
            mask |= (xp / rx) ** 2 + (yp / ry) ** 2 <= 1.0
        E_el = np.where(mask, E_val, 1.0)
        for Em in E_macros:
            v_sol, s_mean = solve_q1_elasticity_periodic(E_el, Em)
            Es.append(E_el.astype(np.float32))
            Emacs.append(Em.astype(np.float32))
            vs.append(v_sol.transpose(2, 0, 1).astype(np.float32))  # (2, N, N)
            sigs.append(s_mean.astype(np.float32))
    E_t = torch.from_numpy(np.stack(Es))
    nonperc, nlbl, mc, dx_n, dy_n, strip_n = precompute_elasticity_components(E_t)
    return {
        "E_elem": E_t, "E_macro": torch.from_numpy(np.stack(Emacs)),
        "v_true": torch.from_numpy(np.stack(vs)), "stress_true": torch.from_numpy(np.stack(sigs)),
        "nonperc_el": nonperc, "node_lbls": nlbl, "max_comp": mc,
        "dx_n": dx_n, "dy_n": dy_n, "strip_n": strip_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    ep = 2 if args.smoke else args.epochs
    root = Path(__file__).resolve().parent
    generate_synthetic_benchmarks(root)

    out: dict[str, object] = {"device": str(device), "seeds": seeds, "epochs": ep}

    # ==== Exp 4b: Re-verify wavy_laminate + hard_bridge_1e6 with Percolating-Aware Adaptive Dipole Readout ====
    print("\n=== Exp 4b: Percolating-Aware Adaptive Readout on wavy_laminate & hard_bridge_1e6 ===", flush=True)
    exp4b = {}
    for fam_name, pt_file in [("wavy_laminate", "wavy_laminate_N128.pt"), ("hard_bridge_1e6", "hard_bridge_N128.pt")]:
        tr_p = pack_from_pt(root / pt_file, "train")
        id_p = pack_from_pt(root / pt_file, "test")
        od_p = pack_from_pt(root / pt_file, "ood")
        id_s, od_s = [], []
        for sd in seeds:
            torch.manual_seed(sd)
            mdl = BCNO2D(steps=4, hidden=32).float()
            train_model(mdl, True, tr_p, epochs=ep, batch_size=8, lr=3e-3, seed=sd, device=device, use_keff_loss=True, use_sigma_loss=True)
            id_s.append(evaluate_pack(mdl, id_p, True, device))
            od_s.append(evaluate_pack(mdl, od_p, True, device))
        exp4b[fam_name] = {"id": agg_seeds(id_s), "ood": agg_seeds(od_s)}
        print(f"  [{fam_name:15s}] ID DiagKeff={exp4b[fam_name]['id']['DiagRelErr']['mean']:.3f} | OOD DiagKeff={exp4b[fam_name]['ood']['DiagRelErr']['mean']:.3f} | OOD Fluct={exp4b[fam_name]['ood']['Fluctuation_L2']['mean']:.3f}", flush=True)
    out["exp4b_adaptive_readout"] = exp4b

    # ==== Exp 7: 2D Periodic Boundary Conditions (PBC) ====
    print("\n=== Exp 7: 2D Periodic Boundary Conditions (PBC, N=64, wrapped inclusions) ===", flush=True)
    pbc_tr = build_periodic_2d_pack(64, 16 if args.smoke else 48, 101, 10.0, 80.0)
    pbc_id = build_periodic_2d_pack(64, 8 if args.smoke else 16, 202, 10.0, 80.0)
    pbc_od = build_periodic_2d_pack(64, 8 if args.smoke else 16, 303, 100.0, 300.0)

    def eval_pbc(mdl, pk, is_bcno: bool):
        mdl.eval()
        B = pk["kappa"].shape[0]
        vps = []
        with torch.no_grad():
            for s in range(0, B, 16):
                e = min(s + 16, B)
                kap = pk["kappa"][s:e].to(device); ax = pk["load_axis"][s:e].to(device)
                if is_bcno:
                    vp, _ = mdl(kap, ax, pk["labels"][s:e].to(device), pk["max_comp"], pk["nonperc"][s:e].to(device), pk["dx_wrap"][s:e].to(device), pk["dy_wrap"][s:e].to(device))
                else:
                    # Feed constant lift indicator via load_axis grids
                    H = kap.shape[-1]
                    ax1 = torch.linspace(0.0, 1.0, H, device=device)
                    xg, yg = torch.meshgrid(ax1, ax1, indexing="ij")
                    lft = torch.stack([xg if int(a) == 0 else yg for a in ax])
                    ort = torch.stack([yg if int(a) == 0 else xg for a in ax])
                    dist = torch.zeros_like(lft)
                    vp = mdl(kap, lft, ort, dist) - lft
                    vp = vp - vp.mean((1, 2), keepdim=True)
                vps.append(vp.cpu())
        vp_all = torch.cat(vps, 0)
        kd = periodic_dipole_keff(vp_all.double(), pk["kappa"].double(), pk["load_axis"], pk["nonperc"], pk["dx_wrap"].double(), pk["dy_wrap"].double()).float()
        kn = periodic_dipole_keff(vp_all.double(), pk["kappa"].double(), pk["load_axis"], torch.zeros_like(pk["nonperc"]), pk["dx_wrap"].double(), pk["dy_wrap"].double()).float()
        idx = torch.arange(B)
        kt = pk["k_cols"][idx, pk["load_axis"]]
        return {
            "DiagDip": float(((kd[idx, pk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean()),
            "DiagNaive": float(((kn[idx, pk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean()),
            "FluctL2": float(relative_l2(vp_all, pk["v_true"]).mean()),
        }

    exp7 = {}
    for name, is_b, ctor in [
        ("BCNO-PBC", True, lambda: BCNO_PBC_2D(steps=4, hidden=32)),
        ("MgNO2d", False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("FFNO2d", False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("FNO2d", False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
        ("ResUNet", False, lambda: ResUNetSmall(5, width=32)),
    ]:
        id_s, od_s = [], []
        for sd in seeds:
            torch.manual_seed(sd)
            mdl = ctor().float().to(device)
            opt = torch.optim.AdamW(mdl.parameters(), lr=(3e-3 if is_b else 2e-3), weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
            B = pbc_tr["kappa"].shape[0]
            H = pbc_tr["kappa"].shape[-1]
            ax1 = torch.linspace(0.0, 1.0, H, device=device)
            xg, yg = torch.meshgrid(ax1, ax1, indexing="ij")
            for _ in range(ep):
                mdl.train()
                perm = torch.randperm(B)
                for s in range(0, B, 8):
                    idx = perm[s : min(s + 8, B)]
                    kap = pbc_tr["kappa"][idx].to(device); ax = pbc_tr["load_axis"][idx].to(device); vt = pbc_tr["v_true"][idx].to(device)
                    if is_b:
                        vp, _ = mdl(kap, ax, pbc_tr["labels"][idx].to(device), pbc_tr["max_comp"], pbc_tr["nonperc"][idx].to(device), pbc_tr["dx_wrap"][idx].to(device), pbc_tr["dy_wrap"][idx].to(device))
                    else:
                        lft = torch.stack([xg if int(a) == 0 else yg for a in ax])
                        ort = torch.stack([yg if int(a) == 0 else xg for a in ax])
                        vp = mdl(kap, lft, ort, torch.zeros_like(lft)) - lft
                        vp = vp - vp.mean((1, 2), keepdim=True)
                    num = torch.sqrt(((vp - vt) ** 2).sum((1, 2)) + 1e-8)
                    den = torch.sqrt((vt ** 2).sum((1, 2)) + 1e-6).clamp_min(1e-2)
                    loss = (num / den).mean()
                    if is_b:
                        kp = periodic_dipole_keff(vp, kap, ax, pbc_tr["nonperc"][idx].to(device), pbc_tr["dx_wrap"][idx].to(device), pbc_tr["dy_wrap"][idx].to(device))
                        bi = torch.arange(idx.shape[0], device=device)
                        kt_d = pbc_tr["k_cols"][idx].to(device)[bi, ax]
                        loss = loss + 0.25 * ((kp[bi, ax] - kt_d).abs() / kt_d.abs().clamp_min(0.1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            id_s.append(eval_pbc(mdl, pbc_id, is_b))
            od_s.append(eval_pbc(mdl, pbc_od, is_b))
        exp7[name] = {"id": agg_seeds(id_s), "ood": agg_seeds(od_s)}
        print(f"  [PBC {name:9s}] ID Diag={exp7[name]['id']['DiagDip' if is_b else 'DiagNaive']['mean']:.3f} Fluct={exp7[name]['id']['FluctL2']['mean']:.3f} | OOD Dip={exp7[name]['ood']['DiagDip']['mean']:.3f} Naive={exp7[name]['ood']['DiagNaive']['mean']:.3f} Fluct={exp7[name]['ood']['FluctL2']['mean']:.3f}", flush=True)
    out["exp7_pbc_2d"] = exp7

    # ==== Exp 8: True 3D RVE (Synthetic 32^3 + Real 3D Ruhr Sandstone CT 32^3 + 64^3 PCG Warm-Start) ====
    print("\n=== Exp 8: True 3D RVE + Real 3D Ruhr CT + 3D PCG Warm-Start ===", flush=True)
    s3_tr = make_3d_synthetic_pack(32, 6 if args.smoke else 16, 11, 10.0, 80.0)
    s3_id = make_3d_synthetic_pack(32, 3 if args.smoke else 6, 22, 10.0, 80.0)
    s3_od = make_3d_synthetic_pack(32, 3 if args.smoke else 6, 33, 200.0, 1000.0)
    r3_tr = load_3d_ruhr_pack(root / "rve_3d_ct_compact.npz", "tr32", solve_labels=True)
    r3_te = load_3d_ruhr_pack(root / "rve_3d_ct_compact.npz", "te32", solve_labels=True)
    if args.smoke:
        for pk in (r3_tr, r3_te):
            for k in ("kappa", "v_true", "lift", "o1", "o2", "dist", "k_cols", "load_axis", "labels"):
                pk[k] = pk[k][:6]

    def eval_3d(mdl, pk, is_b: bool):
        mdl.eval()
        B = pk["kappa"].shape[0]
        ups = []
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, B, 6):
                e = min(s + 6, B)
                kap = pk["kappa"][s:e].to(device); lft = pk["lift"][s:e].to(device)
                o1 = pk["o1"][s:e].to(device); o2 = pk["o2"][s:e].to(device); dst = pk["dist"][s:e].to(device)
                if is_b:
                    up, _ = mdl(kap, lft, o1, o2, dst, pk["labels"][s:e].to(device), pk["max_comp"])
                else:
                    up = mdl(kap, lft, o1, o2, dst)
                ups.append(up.cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        inf_ms = (time.time() - t0) * 1000.0 / max(B, 1)
        u_p = torch.cat(ups, 0)
        v_p = u_p - pk["lift"]
        kd = dipole_keff_columns_3d(u_p.double(), pk["kappa"].double(), pk["xg"].double(), pk["yg"].double(), pk["zg"].double()).float()
        kn = conservative_flux_columns_3d(u_p.double(), pk["kappa"].double(), kappa_channel=0).float()
        idx = torch.arange(B)
        kt = pk["k_cols"][idx, pk["load_axis"]]
        v_norm = torch.sqrt((pk["v_true"] ** 2).sum((1, 2, 3)))
        valid_v = v_norm > 1e-4
        if bool(valid_v.any()):
            fluct = float((torch.sqrt(((v_p[valid_v] - pk["v_true"][valid_v]) ** 2).sum((1, 2, 3)) + 1e-12) / v_norm[valid_v]).mean())
        else:
            fluct = 0.0
        return {
            "DiagDip": float(((kd[idx, pk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean()),
            "DiagNaive": float(((kn[idx, pk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean()),
            "FluctL2": fluct,
            "InferMs": inf_ms,
        }

    def run_3d_bench(tag, tr_p, te_packs):
        res = {}
        ep3 = 2 if args.smoke else max(40, int(0.6 * ep))
        for name, is_b, ctor in [
            ("BCNO-3D", True, lambda: BCNO3D(steps=3, hidden=20)),
            ("ResUNet-3D", False, lambda: ResUNet3dSmall(6, width=20)),
            ("FNO-3D", False, lambda: FNO3dSmall(6, width=20, modes=8, depth=4)),
        ]:
            per_split = {k: [] for k in te_packs}
            trained_mdl = None
            for sd in seeds:
                torch.manual_seed(sd)
                mdl = ctor().float().to(device)
                opt = torch.optim.AdamW(mdl.parameters(), lr=(3e-3 if is_b else 2e-3), weight_decay=1e-5)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep3)
                B = tr_p["kappa"].shape[0]
                H = tr_p["kappa"].shape[-1]
                h3 = 1.0 / (H - 1)
                xg, yg, zg = tr_p["xg"].to(device), tr_p["yg"].to(device), tr_p["zg"].to(device)
                for _ in range(ep3):
                    mdl.train()
                    perm = torch.randperm(B)
                    for s in range(0, B, 6):
                        idx = perm[s : min(s + 6, B)]
                        kap = tr_p["kappa"][idx].to(device); lft = tr_p["lift"][idx].to(device)
                        o1 = tr_p["o1"][idx].to(device); o2 = tr_p["o2"][idx].to(device); dst = tr_p["dist"][idx].to(device)
                        vt = tr_p["v_true"][idx].to(device)
                        if is_b:
                            up, sig = mdl(kap, lft, o1, o2, dst, tr_p["labels"][idx].to(device), tr_p["max_comp"])
                        else:
                            up = mdl(kap, lft, o1, o2, dst)
                        vp = up - lft
                        num = torch.sqrt(((vp - vt) ** 2).sum((1, 2, 3)) + 1e-8)
                        den = torch.sqrt((vt ** 2).sum((1, 2, 3)) + 1e-6).clamp_min(1e-2)
                        loss = (num / den).mean()
                        if is_b:
                            kp = dipole_keff_columns_3d(up, kap, xg, yg, zg)
                            bi = torch.arange(idx.shape[0], device=device)
                            kt_d = tr_p["k_cols"][idx].to(device)[bi, tr_p["load_axis"][idx].to(device)]
                            loss = loss + 0.25 * ((kp[bi, tr_p["load_axis"][idx].to(device)] - kt_d).abs() / kt_d.abs().clamp_min(0.1)).mean()
                            strip = boundary_strip_mask_3d(kap)
                            sig_t = torch.zeros_like(vt)
                            sig_t[:, 1:-1, 1:-1, 1:-1] = (
                                6.0 * vt[:, 1:-1, 1:-1, 1:-1] - vt[:, :-2, 1:-1, 1:-1] - vt[:, 2:, 1:-1, 1:-1]
                                - vt[:, 1:-1, :-2, 1:-1] - vt[:, 1:-1, 2:, 1:-1] - vt[:, 1:-1, 1:-1, :-2] - vt[:, 1:-1, 1:-1, 2:]
                            ) / h3
                            loss = loss + 0.25 * ((((sig - sig_t) * strip) ** 2).sum((1, 2, 3)) / strip.sum((1, 2, 3)).clamp_min(1.0)).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
                    sched.step()
                for sk, spk in te_packs.items():
                    per_split[sk].append(eval_3d(mdl, spk, is_b))
                if sd == seeds[0]:
                    trained_mdl = mdl
            res[name] = {sk: agg_seeds(per_split[sk]) for sk in te_packs}
            res[name]["params"] = sum(p.numel() for p in trained_mdl.parameters())
            for sk in te_packs:
                print(f"  [3D {tag:10s} | {sk:3s} | {name:10s}] Dip={res[name][sk]['DiagDip']['mean']:.3f} Naive={res[name][sk]['DiagNaive']['mean']:.3f} Fluct={res[name][sk]['FluctL2']['mean']:.3f} ms={res[name][sk]['InferMs']['mean']:.2f}", flush=True)
            if name == "BCNO-3D" and tag in ("Synth32", "RuhrCT32"):
                # Run 3D PCG warm-start benchmark on high-contrast 32^3 (and 64^3 Ruhr CT!)
                pcg_bench = {}
                test_list = [("32_cubed_synth_ood", s3_od)] if tag == "Synth32" else [("32_cubed_ruhr", r3_te), ("64_cubed_ruhr", load_3d_ruhr_pack(root / "rve_3d_ct_compact.npz", "te64", solve_labels=False))]
                for sz_name, pk_s in test_list:
                    kap_1 = pk_s["kappa"][0:1].to(device); lft_1 = pk_s["lift"][0:1].to(device)
                    o1_1 = pk_s["o1"][0:1].to(device); o2_1 = pk_s["o2"][0:1].to(device); dst_1 = pk_s["dist"][0:1].to(device)
                    lbl_1 = pk_s["labels"][0:1].to(device)
                    with torch.no_grad():
                        if device.type == "cuda": torch.cuda.synchronize()
                        tb = time.time()
                        u_bc, _ = trained_mdl(kap_1, lft_1, o1_1, o2_1, dst_1, lbl_1, pk_s["max_comp"])
                        if device.type == "cuda": torch.cuda.synchronize()
                        bc_ms = (time.time() - tb) * 1000.0
                    _, it_cold, ms_cold = jacobi_pcg_3d(kap_1, lft_1, None, tol=1e-6)
                    _, it_warm, ms_warm = jacobi_pcg_3d(kap_1, lft_1, u_bc, tol=1e-6)
                    lu_ms = float("nan")
                    if "32_cubed" in sz_name:
                        # Measure genuine CPU Sparse LU time (even when CUDA is active)
                        kap_lu = pk_s["kappa"][0].numpy().astype(np.float64)
                        n_lu = kap_lu.shape[0]; m_lu = n_lu - 2; h_lu = 1.0 / (n_lu - 1); s2_lu = 1.0 / (h_lu * h_lu)
                        tl = time.time()
                        kx_, ky_, kz_ = [t[0].numpy() * s2_lu for t in face_conductivities_3d(torch.from_numpy(kap_lu)[None], kappa_channel=0)]
                        kxm_, kxp_ = kx_[:-1,1:-1,1:-1], kx_[1:,1:-1,1:-1]; kym_, kyp_ = ky_[1:-1,:-1,1:-1], ky_[1:-1,1:,1:-1]; kzm_, kzp_ = kz_[1:-1,1:-1,:-1], kz_[1:-1,1:-1,1:]
                        idx_ = np.arange(m_lu**3).reshape(m_lu, m_lu, m_lu)
                        A_lu = sp.coo_matrix((np.concatenate([(kxm_+kxp_+kym_+kyp_+kzm_+kzp_).reshape(-1), -kxm_[1:].reshape(-1), -kxp_[:-1].reshape(-1), -kym_[:,1:].reshape(-1), -kyp_[:,:-1].reshape(-1), -kzm_[:,:,1:].reshape(-1), -kzp_[:,:,:-1].reshape(-1)]), (np.concatenate([idx_.reshape(-1), idx_[1:].reshape(-1), idx_[:-1].reshape(-1), idx_[:,1:].reshape(-1), idx_[:,:-1].reshape(-1), idx_[:,:,1:].reshape(-1), idx_[:,:,:-1].reshape(-1)]), np.concatenate([idx_.reshape(-1), idx_[:-1].reshape(-1), idx_[1:].reshape(-1), idx_[:,:-1].reshape(-1), idx_[:,1:].reshape(-1), idx_[:,:,:-1].reshape(-1), idx_[:,:,1:].reshape(-1)]))), shape=(m_lu**3, m_lu**3)).tocsc()
                        spla.spsolve(A_lu, np.ones(m_lu**3))
                        lu_ms = (time.time() - tl) * 1000.0
                    pcg_bench[sz_name] = {
                        "bcno_ms": round(bc_ms, 2), "lu_ms": round(lu_ms, 2),
                        "pcg_cold_iters": it_cold, "pcg_cold_ms": round(ms_cold, 2),
                        "pcg_warm_iters": it_warm, "pcg_warm_ms": round(ms_warm + bc_ms, 2),
                    }
                    print(f"  [3D Solver {sz_name}] BC-NO={bc_ms:.2f}ms | LU={lu_ms:.1f}ms | PCG cold={it_cold} ({ms_cold:.1f}ms) -> BCNO+PCG={it_warm} ({ms_warm+bc_ms:.1f}ms)", flush=True)
                res["pcg_warmstart"] = pcg_bench
        return res

    out["exp8_3d_synth"] = run_3d_bench("Synth32", s3_tr, {"id": s3_id, "ood": s3_od})
    out["exp8_3d_ruhr"] = run_3d_bench("RuhrCT32", r3_tr, {"test": r3_te})

    # ==== Exp 9: 2D Periodic Q1 Linear Elasticity (C_eff at E1/E0 = 10..200 -> 1000 OOD) ====
    print("\n=== Exp 9: 2D Periodic Linear Elasticity (E1/E0=1000 OOD) ===", flush=True)
    el_tr = build_elasticity_pack(48, 8 if args.smoke else 32, 401, 10.0, 200.0)
    el_id = build_elasticity_pack(48, 4 if args.smoke else 12, 402, 10.0, 200.0)
    el_od = build_elasticity_pack(48, 4 if args.smoke else 12, 403, 500.0, 1000.0)

    def eval_elas(mdl, pk, is_b: bool):
        mdl.eval()
        B = pk["E_elem"].shape[0]
        vps = []
        with torch.no_grad():
            for s in range(0, B, 12):
                e = min(s + 12, B)
                Ee = pk["E_elem"][s:e].to(device); Em = pk["E_macro"][s:e].to(device)
                if is_b:
                    vp, _ = mdl(Ee, Em, pk["nonperc_el"][s:e].to(device), pk["node_lbls"][s:e].to(device), pk["max_comp"], pk["dx_n"][s:e].to(device), pk["dy_n"][s:e].to(device), pk["strip_n"][s:e].to(device))
                else:
                    vp = mdl(Ee, Em)
                vps.append(vp.cpu())
        vp_all = torch.cat(vps, 0)
        sd = q1_stress_dipole_readout(vp_all.double(), pk["E_elem"].double(), pk["E_macro"].double(), pk["nonperc_el"], pk["dx_n"].double(), pk["dy_n"].double()).float()
        sn = q1_stress_dipole_readout(vp_all.double(), pk["E_elem"].double(), pk["E_macro"].double(), torch.zeros_like(pk["nonperc_el"]), pk["dx_n"].double(), pk["dy_n"].double()).float()
        st = pk["stress_true"]
        c_dip = float((torch.linalg.vector_norm(sd - st, dim=1) / torch.linalg.vector_norm(st, dim=1).clamp_min(1e-4)).mean())
        c_naive = float((torch.linalg.vector_norm(sn - st, dim=1) / torch.linalg.vector_norm(st, dim=1).clamp_min(1e-4)).mean())
        fluct = float((torch.sqrt(((vp_all - pk["v_true"]) ** 2).sum((1, 2, 3)) + 1e-12) / torch.sqrt((pk["v_true"] ** 2).sum((1, 2, 3)) + 1e-12).clamp_min(1e-6)).mean())
        return {"CeffDip": c_dip, "CeffNaive": c_naive, "FluctL2": fluct}

    exp9 = {}
    for name, is_b, ctor in [
        ("BCNO-Elas", True, lambda: BCNO_Elas_2D(steps=4, hidden=32)),
        ("ResUNet-Elas", False, lambda: ResUNet_Elas_2D(width=32)),
        ("FNO-Elas", False, lambda: FNO_Elas_2D(width=32, modes=16, depth=4)),
    ]:
        id_s, od_s = [], []
        for sd in seeds:
            torch.manual_seed(sd)
            mdl = ctor().float().to(device)
            opt = torch.optim.AdamW(mdl.parameters(), lr=(3e-3 if is_b else 2e-3), weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
            B = el_tr["E_elem"].shape[0]
            for _ in range(ep):
                mdl.train()
                perm = torch.randperm(B)
                for s in range(0, B, 9):
                    idx = perm[s : min(s + 9, B)]
                    Ee = el_tr["E_elem"][idx].to(device); Em = el_tr["E_macro"][idx].to(device); vt = el_tr["v_true"][idx].to(device)
                    if is_b:
                        vp, _ = mdl(Ee, Em, el_tr["nonperc_el"][idx].to(device), el_tr["node_lbls"][idx].to(device), el_tr["max_comp"], el_tr["dx_n"][idx].to(device), el_tr["dy_n"][idx].to(device), el_tr["strip_n"][idx].to(device))
                    else:
                        vp = mdl(Ee, Em)
                    num = torch.sqrt(((vp - vt) ** 2).sum((1, 2, 3)) + 1e-8)
                    den = torch.sqrt((vt ** 2).sum((1, 2, 3)) + 1e-6).clamp_min(1e-2)
                    loss = (num / den).mean()
                    if is_b:
                        sp_pred = q1_stress_dipole_readout(vp, Ee, Em, el_tr["nonperc_el"][idx].to(device), el_tr["dx_n"][idx].to(device), el_tr["dy_n"][idx].to(device))
                        st_t = el_tr["stress_true"][idx].to(device)
                        loss = loss + 0.25 * (torch.linalg.vector_norm(sp_pred - st_t, dim=1) / torch.linalg.vector_norm(st_t, dim=1).clamp_min(0.1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
            id_s.append(eval_elas(mdl, el_id, is_b))
            od_s.append(eval_elas(mdl, el_od, is_b))
        exp9[name] = {"id": agg_seeds(id_s), "ood": agg_seeds(od_s)}
        print(f"  [Elas {name:12s}] ID Ceff={exp9[name]['id']['CeffDip' if is_b else 'CeffNaive']['mean']:.3f} Fluct={exp9[name]['id']['FluctL2']['mean']:.3f} | OOD Dip={exp9[name]['ood']['CeffDip']['mean']:.3f} Naive={exp9[name]['ood']['CeffNaive']['mean']:.3f} Fluct={exp9[name]['ood']['FluctL2']['mean']:.3f}", flush=True)
    out["exp9_elasticity_2d"] = exp9

    (root / "bcno_v2_addons_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: bcno_v2_addons_results.json", flush=True)


if __name__ == "__main__":
    main()
