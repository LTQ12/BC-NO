"""CMAME Rigor Add-ons Suite (Exp 5b, Exp 10b, Exp 14):
  Exp 5b : Experimental Root-Cause Proof for Cross-Resolution (Single-Res N=128 vs Multi-Res N={64,128}, zero N=256 training)
  Exp 10b: Thermal FE^2 SPD Eigenvalue Projection Verification (Raw Sym vs SPD-Projected Macro Accuracy)
  Exp 14 : 2D Q1 Linear Elasticity Two-Scale FE^2 Holed Composite Cantilever Plate (E1/E0 in [30, 1000] OOD)
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

from baselines_2d import CNO2dWrapper, FFNO2d, FNO2dSmall, MgNO2dWrapper, ResUNetSmall
from bcno_elasticity_2d import (
    BCNO_Elas_2D,
    FNO_Elas_2D,
    ResUNet_Elas_2D,
    precompute_elasticity_components,
    q1_stress_dipole_readout,
    solve_q1_elasticity_periodic,
)
from bcno_model_2d import BCNO2D, boundary_strip_mask, dipole_keff_columns, precompute_floating_components
from colab_data_gen import fast_interior_solve, generate_synthetic_benchmarks, make_fiber_columns
from rve_discrete_operator_2d import conservative_flux_columns
from run_bcno_fe2_audit import build_fe2_micro_suite, constitutive_audit, predict_keff_cols, solve_macro_q1_plate
from run_bcno_paper_suite import agg_seeds, evaluate_pack, pack_from_pt, train_model


def spd_project_2x2(K_tensors: np.ndarray, min_eig: float = 1.0) -> np.ndarray:
    """Symmetric positive-definite projection of 2x2 conductivity tensors via eigenvalue clipping."""
    K_sym = 0.5 * (K_tensors + np.swapaxes(K_tensors, -1, -2))
    shape = K_sym.shape[:-2]
    flat = K_sym.reshape(-1, 2, 2)
    vals, vecs = np.linalg.eigh(flat)
    vals_c = np.maximum(vals, min_eig)
    proj = np.einsum("eik,ek,ejk->eij", vecs, vals_c, vecs)
    return proj.reshape(*shape, 2, 2)


def spd_project_3x3(C_tensors: np.ndarray, min_eig: float = 1.0) -> np.ndarray:
    """Symmetric positive-definite projection of 3x3 Voigt elasticity tensors via eigenvalue clipping."""
    C_sym = 0.5 * (C_tensors + np.swapaxes(C_tensors, -1, -2))
    shape = C_sym.shape[:-2]
    flat = C_sym.reshape(-1, 3, 3)
    vals, vecs = np.linalg.eigh(flat)
    vals_c = np.maximum(vals, min_eig)
    proj = np.einsum("eik,ek,ejk->eij", vecs, vals_c, vecs)
    return proj.reshape(*shape, 3, 3)


def build_fiber_pack_custom(N: int, n_cases: int, seed: int, c_lo: float, c_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    ax1 = np.linspace(0.0, 1.0, N, dtype=np.float64)
    xg, yg = np.meshgrid(ax1, ax1, indexing="ij")
    kaps, vs, lifts, axs = [], [], [], []
    for _ in range(n_cases):
        c = float(np.exp(rng.uniform(np.log(c_lo), np.log(c_hi))))
        if rng.random() < 0.25:
            c = 1.0 / c
        vf = float(rng.uniform(0.25, 0.58))
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
    return {
        "kappa": kap_t, "kappa_raw": kap_t, "v_true": v_t, "u_true": lift_t + v_t,
        "lift": lift_t, "ortho": ortho_t, "dist": dist_t, "k_cols": k_cols,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": lbl, "max_comp": mc, "x_grid": xg_t[None], "y_grid": yg_t[None],
    }


@torch.no_grad()
def evaluate_sigma_strip_err(model: nn.Module, pack: dict, device: torch.device, batch_size: int = 8) -> float:
    """Measure surface density error ||(sigma_pred - sigma_true) * 1_{dI}||_2 / ||sigma_true * 1_{dI}||_2."""
    model.eval()
    B, N, _ = pack["kappa"].shape
    h = 1.0 / float(N - 1)
    errs = []
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        kap = pack["kappa"][s:e].to(device)
        lft = pack["lift"][s:e].to(device)
        ort = pack["ortho"][s:e].to(device)
        dst = pack["dist"][s:e].to(device)
        lbl = pack["labels"][s:e].to(device)
        vt = pack["v_true"][s:e].to(device)
        _, sig_p = model(kap, lft, ort, dst, lbl, pack["max_comp"])
        strip = boundary_strip_mask(kap)
        sig_t = torch.zeros_like(vt)
        sig_t[:, 1:-1, 1:-1] = (
            4.0 * vt[:, 1:-1, 1:-1] - vt[:, :-2, 1:-1] - vt[:, 2:, 1:-1] - vt[:, 1:-1, :-2] - vt[:, 1:-1, 2:]
        ) / h
        num = torch.sqrt((((sig_p - sig_t) * strip) ** 2).sum(dim=(1, 2)) + 1e-12)
        den = torch.sqrt(((sig_t * strip) ** 2).sum(dim=(1, 2)) + 1e-12).clamp_min(1e-4)
        errs.extend((num / den).cpu().tolist())
    return float(statistics.mean(errs))


def train_multires(model: nn.Module, is_bcno: bool, packs: list[dict], epochs: int, batch_size: int, lr: float, seed: int, device: torch.device):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        model.train()
        for pk in packs:
            B, _, N = pk["kappa"].shape
            h_grid = 1.0 / max(N - 1, 1)
            perm = torch.randperm(B)
            xg = pk["x_grid"].to(device)
            yg = pk["y_grid"].to(device)
            for s in range(0, B, batch_size):
                idx = perm[s : min(s + batch_size, B)]
                kap = pk["kappa"][idx].to(device)
                kap_raw = pk["kappa_raw"][idx].to(device)
                vt = pk["v_true"][idx].to(device)
                lift = pk["lift"][idx].to(device)
                ortho = pk["ortho"][idx].to(device)
                dist = pk["dist"][idx].to(device)
                kt = pk["k_cols"][idx].to(device)
                ax = pk["load_axis"][idx].to(device)
                if is_bcno:
                    up, sig = model(kap, lift, ortho, dist, pk["labels"][idx].to(device), pk["max_comp"])
                else:
                    up = model(kap, lift, ortho, dist)
                vp = up - lift
                num_v = torch.sqrt(((vp - vt) ** 2).sum(dim=(1, 2)) + 1e-8)
                den_v = torch.sqrt((vt ** 2).sum(dim=(1, 2)) + 1e-6).clamp_min(1e-2)
                loss = (num_v / den_v).mean()
                if is_bcno:
                    kp = dipole_keff_columns(up, kap_raw, xg, yg)
                    b_idx = torch.arange(idx.shape[0], device=device)
                    loss = loss + 0.25 * ((kp[b_idx, ax] - kt[b_idx, ax]).abs() / kt[b_idx, ax].abs().clamp_min(0.1)).mean()
                    strip = boundary_strip_mask(kap)
                    sig_t = torch.zeros_like(vt)
                    sig_t[:, 1:-1, 1:-1] = (
                        4.0 * vt[:, 1:-1, 1:-1] - vt[:, :-2, 1:-1] - vt[:, 2:, 1:-1] - vt[:, 1:-1, :-2] - vt[:, 1:-1, 2:]
                    ) / h_grid
                    loss_sig = (((sig - sig_t) * strip) ** 2).sum(dim=(1, 2)) / strip.sum(dim=(1, 2)).clamp_min(1.0)
                    loss = loss + 0.25 * loss_sig.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        sched.step()


# ---------- Exp 14: 2D Q1 Linear Elasticity FE^2 Holed Composite Cantilever Plate ----------
def build_elas_rve_pack(N: int, n_cases: int, seed: int, E_lo: float, E_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    ax = (np.arange(N) + 0.5) / float(N)
    ex, ey = np.meshgrid(ax, ax, indexing="ij")
    E_macros = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
    ]
    Es, Emacs, vs, sigs = [], [], [], []
    for _ in range(n_cases):
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
            vs.append(v_sol.transpose(2, 0, 1).astype(np.float32))
            sigs.append(s_mean.astype(np.float32))
    E_t = torch.from_numpy(np.stack(Es))
    nonperc, nlbl, mc, dx_n, dy_n, strip_n = precompute_elasticity_components(E_t)
    return {
        "E_elem": E_t, "E_macro": torch.from_numpy(np.stack(Emacs)),
        "v_true": torch.from_numpy(np.stack(vs)), "stress_true": torch.from_numpy(np.stack(sigs)),
        "nonperc_el": nonperc, "node_lbls": nlbl, "max_comp": mc,
        "dx_n": dx_n, "dy_n": dy_n, "strip_n": strip_n,
    }


def build_holed_plate_fe2_micro(Mx: int, My: int, N_micro: int, seed: int = 2027) -> dict:
    """Build active elements of a Mx x My composite cantilever [0, 2] x [0, 1] with central circular cutout."""
    rng = np.random.default_rng(seed)
    Lx, Ly = 2.0, 1.0
    hx, hy = Lx / Mx, Ly / My
    ax = (np.arange(N_micro) + 0.5) / float(N_micro)
    ex, ey = np.meshgrid(ax, ax, indexing="ij")
    E_macros = [
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
    ]

    active_mask = np.ones((Mx, My), dtype=bool)
    Es, Emacs, vs, sigs = [], [], [], []
    t0 = time.time()
    for ei in range(Mx):
        for ej in range(My):
            xm = (ei + 0.5) * hx
            ym = (ej + 0.5) * hy
            dist_hole = math.hypot(xm - 1.0, ym - 0.5)
            if dist_hole < 0.24:
                active_mask[ei, ej] = False
                continue
            # Grade E_inc from 30 near clamped root up to 1000 OOD near cutout / tip!
            r_stress = math.exp(-((dist_hole - 0.24) / 0.35) ** 2)
            E_inc = 30.0 + 970.0 * max(r_stress, (xm / Lx) ** 2)
            n_inc = int(3 + round(4 * (0.5 + 0.5 * math.sin(math.pi * xm) * math.cos(math.pi * ym))))
            mask = np.zeros((N_micro, N_micro), dtype=bool)
            for _ in range(n_inc):
                cx, cy = rng.uniform(0.0, 1.0, size=2)
                rx, ry = rng.uniform(0.08, 0.17, size=2)
                th = float(rng.uniform(-0.5, 0.5))
                dx = ((ex - cx + 0.5) % 1.0) - 0.5
                dy = ((ey - cy + 0.5) % 1.0) - 0.5
                xp = math.cos(th) * dx + math.sin(th) * dy
                yp = -math.sin(th) * dx + math.cos(th) * dy
                mask |= (xp / rx) ** 2 + (yp / ry) ** 2 <= 1.0
            E_el = np.where(mask, E_inc, 1.0)
            for Em in E_macros:
                v_sol, s_mean = solve_q1_elasticity_periodic(E_el, Em)
                Es.append(E_el.astype(np.float32))
                Emacs.append(Em.astype(np.float32))
                vs.append(v_sol.transpose(2, 0, 1).astype(np.float32))
                sigs.append(s_mean.astype(np.float32))
    micro_solve_s = time.time() - t0
    E_t = torch.from_numpy(np.stack(Es))
    nonperc, nlbl, mc, dx_n, dy_n, strip_n = precompute_elasticity_components(E_t)
    return {
        "E_elem": E_t, "E_macro": torch.from_numpy(np.stack(Emacs)),
        "v_true": torch.from_numpy(np.stack(vs)), "stress_true": torch.from_numpy(np.stack(sigs)),
        "nonperc_el": nonperc, "node_lbls": nlbl, "max_comp": mc,
        "dx_n": dx_n, "dy_n": dy_n, "strip_n": strip_n,
        "active_mask": active_mask, "Mx": Mx, "My": My, "Lx": Lx, "Ly": Ly,
        "micro_solve_s": micro_solve_s,
    }


def solve_macro_holed_elasticity(C_active: np.ndarray, active_mask: np.ndarray, Lx: float = 2.0, Ly: float = 1.0):
    """Solve 2D Q1 linear elasticity on holed plate: left x=0 clamped, right x=Lx traction t=[1.0, -0.4]."""
    Mx, My = active_mask.shape
    hx, hy = Lx / Mx, Ly / My
    Nx, Ny = Mx + 1, My + 1
    gp = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    nodes_ref = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    C_sym = 0.5 * (C_active + np.swapaxes(C_active, -1, -2))
    min_eig = float(np.min(np.linalg.eigvalsh(C_sym)))
    spd_ok = min_eig > 1e-6

    # Build active node set
    elem_list = np.argwhere(active_mask)  # (n_act, 2)
    n_act = elem_list.shape[0]
    elem_dofs = np.empty((n_act, 8), dtype=np.int64)
    active_nodes = np.zeros(Nx * Ny, dtype=bool)
    for k, (ei, ej) in enumerate(elem_list):
        n00 = ei * Ny + ej
        n10 = (ei + 1) * Ny + ej
        n11 = (ei + 1) * Ny + (ej + 1)
        n01 = ei * Ny + (ej + 1)
        ns = [n00, n10, n11, n01]
        active_nodes[ns] = True
        for a, nd in enumerate(ns):
            elem_dofs[k, 2 * a] = 2 * nd
            elem_dofs[k, 2 * a + 1] = 2 * nd + 1

    Ke_all = np.zeros((n_act, 8, 8), dtype=np.float64)
    B_mid = np.zeros((3, 8), dtype=np.float64)
    for xi in gp:
        for eta in gp:
            B = np.zeros((3, 8), dtype=np.float64)
            for a, (sx, sy) in enumerate(nodes_ref):
                dNx = 0.25 * sx * (1.0 + sy * eta) * (2.0 / hx)
                dNy = 0.25 * sy * (1.0 + sx * xi) * (2.0 / hy)
                B[0, 2 * a] = dNx
                B[1, 2 * a + 1] = dNy
                B[2, 2 * a] = dNy
                B[2, 2 * a + 1] = dNx
            w = 0.25 * hx * hy
            Ke_all += w * np.einsum("ca,ecd,db->eab", B, C_sym, B)
    for a, (sx, sy) in enumerate(nodes_ref):
        B_mid[0, 2 * a] = 0.25 * sx * (2.0 / hx)
        B_mid[1, 2 * a + 1] = 0.25 * sy * (2.0 / hy)
        B_mid[2, 2 * a] = 0.25 * sy * (2.0 / hy)
        B_mid[2, 2 * a + 1] = 0.25 * sx * (2.0 / hx)

    n_dof = 2 * Nx * Ny
    rr = np.repeat(elem_dofs[:, :, None], 8, axis=2).reshape(-1)
    cc = np.repeat(elem_dofs[:, None, :], 8, axis=1).reshape(-1)
    K_glob = sp.coo_matrix((Ke_all.reshape(-1), (rr, cc)), shape=(n_dof, n_dof)).tocsr()

    f_glob = np.zeros(n_dof, dtype=np.float64)
    # Right boundary traction t = [1.0, -0.4] on x = Lx
    for ej in range(My):
        if active_mask[Mx - 1, ej]:
            nb = (Mx) * Ny + ej
            nt = (Mx) * Ny + (ej + 1)
            for nd in (nb, nt):
                f_glob[2 * nd] += 0.5 * hy * 1.0
                f_glob[2 * nd + 1] += 0.5 * hy * (-0.4)

    fixed_dofs = np.zeros(n_dof, dtype=bool)
    for nd in range(Nx * Ny):
        if not active_nodes[nd]:
            fixed_dofs[2 * nd] = True
            fixed_dofs[2 * nd + 1] = True
    for ej in range(Ny):
        nd = 0 * Ny + ej
        fixed_dofs[2 * nd] = True
        fixed_dofs[2 * nd + 1] = True
    free = ~fixed_dofs
    try:
        u_free = spla.spsolve(K_glob[free][:, free].tocsc(), f_glob[free])
    except Exception:
        return np.full(n_dof, np.nan), np.nan, np.nan, spd_ok, min_eig
    u = np.zeros(n_dof, dtype=np.float64)
    u[free] = u_free

    # Strain energy and tip vertical deflection at (Lx, Ly/2)
    u_el = u[elem_dofs]
    eps_mid = u_el @ B_mid.T
    sig_mid = np.einsum("ecd,ed->ec", C_sym, eps_mid)
    strain_energy = 0.5 * float(np.sum(eps_mid * sig_mid) * hx * hy)
    tip_nd = Mx * Ny + (My // 2)
    tip_uy = float(u[2 * tip_nd + 1])
    return u[free], strain_energy, tip_uy, spd_ok, min_eig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    ep = 2 if args.smoke else args.epochs
    root = Path(__file__).resolve().parent
    generate_synthetic_benchmarks(root)

    results: dict[str, object] = {"device": str(device), "seeds": seeds, "epochs": ep}

    # ==== Exp 10b: Thermal FE^2 SPD Eigenvalue Projection Audit ====
    print("\n=== Exp 10b: Thermal FE^2 SPD Eigenvalue Projection Audit ===", flush=True)
    f_tr = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "train")
    M_mac = 6 if args.smoke else 16
    fe2_pk = build_fe2_micro_suite(M_macro=M_mac, N_micro=128, seed=2026)
    n_el = M_mac * M_mac
    K_true_mac = fe2_pk["k_cols"].view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_mac, M_mac, 2, 2)
    T_ref, flux_ref, _ = solve_macro_q1_plate(K_true_mac, M=M_mac)

    exp10b = {"micro_solve_s": round(fe2_pk["micro_solve_s"], 2), "M_macro": M_mac, "models": {}}
    for name, is_b, use_k, use_sig, ctor in [
        ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
        ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
        ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ("FNO2d", False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
    ]:
        torch.manual_seed(0)
        mdl = ctor().float()
        train_model(mdl, is_b, f_tr, epochs=ep, batch_size=8, lr=(3e-3 if is_b else 2e-3), seed=0, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
        k_fe2_p, _, _, inf_s = predict_keff_cols(mdl, fe2_pk, is_b, device)
        K_surr = k_fe2_p.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_mac, M_mac, 2, 2)
        K_sym = 0.5 * (K_surr + np.swapaxes(K_surr, -1, -2))
        eigs_raw = np.linalg.eigvalsh(K_sym.reshape(-1, 2, 2))[:, 0]
        spd_rate_raw = float(np.mean(eigs_raw > 1e-6))
        min_eig_raw = float(np.min(eigs_raw))

        T_raw, f_raw, spd_all_raw = solve_macro_q1_plate(K_surr, M=M_mac)
        K_proj = spd_project_2x2(K_surr, min_eig=1.0)
        T_proj, f_proj, spd_all_proj = solve_macro_q1_plate(K_proj, M=M_mac)

        exp10b["models"][name] = {
            "inf_s": round(inf_s, 3),
            "spd_rate_raw": spd_rate_raw,
            "min_eig_raw": min_eig_raw,
            "spd_all_raw": spd_all_raw,
            "macro_T_L2_raw": float(np.linalg.norm(T_raw - T_ref) / np.linalg.norm(T_ref)) if np.isfinite(T_raw).all() else float("nan"),
            "macro_flux_err_raw": float(abs(f_raw[0] - flux_ref[0]) / abs(flux_ref[0])) if np.isfinite(f_raw).all() else float("nan"),
            "spd_all_proj": spd_all_proj,
            "macro_T_L2_proj": float(np.linalg.norm(T_proj - T_ref) / np.linalg.norm(T_ref)),
            "macro_flux_err_proj": float(abs(f_proj[0] - flux_ref[0]) / abs(flux_ref[0])),
        }
        m = exp10b["models"][name]
        print(
            f"  [10b {name:8s}] SPD_raw={m['spd_rate_raw']:.3f} (minEig={m['min_eig_raw']:.3f}) | "
            f"Raw T={m['macro_T_L2_raw']:.4f} Flux={m['macro_flux_err_raw']:.4f} -> "
            f"SPD-Proj(all={m['spd_all_proj']}) T={m['macro_T_L2_proj']:.4f} Flux={m['macro_flux_err_proj']:.4f}",
            flush=True,
        )
    results["exp10b_spd_projection"] = exp10b

    # ==== Exp 5b: Experimental Cross-Resolution Root-Cause Proof (Single-Res N=128 vs Multi-Res N={64,128}) ====
    print("\n=== Exp 5b: Cross-Resolution Root-Cause Proof (Zero N=256 Training) ===", flush=True)
    f_n64_tr = build_fiber_pack_custom(64, 8 if args.smoke else 32, seed=5051, c_lo=10.0, c_hi=80.0)
    f_n64_te = pack_from_pt(root / "fiber_N64_eval.pt", "test")
    f_n128_te = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "test")
    f_n256_te = pack_from_pt(root / "fiber_N256_eval.pt", "test")

    exp5b = {}
    for label, is_b, is_multi, ctor in [
        ("BC-NO_SingleRes128", True, False, lambda: BCNO2D(steps=4, hidden=32)),
        ("BC-NO_MultiRes64_128", True, True, lambda: BCNO2D(steps=4, hidden=32)),
        ("FFNO2d_MultiRes64_128", False, True, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
    ]:
        s64, s128, s256, sig64, sig128, sig256 = [], [], [], [], [], []
        for sd in seeds:
            mdl = ctor().float()
            if is_multi:
                train_multires(mdl, is_b, [f_n64_tr, f_tr], epochs=ep, batch_size=8, lr=(3e-3 if is_b else 2e-3), seed=sd, device=device)
            else:
                train_model(mdl, is_b, f_tr, epochs=ep, batch_size=8, lr=(3e-3 if is_b else 2e-3), seed=sd, device=device, use_keff_loss=is_b, use_sigma_loss=is_b)
            s64.append(evaluate_pack(mdl, f_n64_te, is_b, device))
            s128.append(evaluate_pack(mdl, f_n128_te, is_b, device))
            s256.append(evaluate_pack(mdl, f_n256_te, is_b, device, batch_size=4))
            if is_b:
                sig64.append(evaluate_sigma_strip_err(mdl, f_n64_te, device))
                sig128.append(evaluate_sigma_strip_err(mdl, f_n128_te, device))
                sig256.append(evaluate_sigma_strip_err(mdl, f_n256_te, device, batch_size=4))
        exp5b[label] = {
            "N64": agg_seeds(s64), "N128": agg_seeds(s128), "N256": agg_seeds(s256),
            "sigma_strip_rel_err": {
                "N64": round(statistics.mean(sig64), 4) if sig64 else None,
                "N128": round(statistics.mean(sig128), 4) if sig128 else None,
                "N256": round(statistics.mean(sig256), 4) if sig256 else None,
            },
        }
        e = exp5b[label]
        print(
            f"  [5b {label:22s}] Keff(64/128/256)={e['N64']['KeffRelErr']['mean']:.3f}/{e['N128']['KeffRelErr']['mean']:.3f}/{e['N256']['KeffRelErr']['mean']:.3f} | "
            f"Fluct(256)={e['N256']['Fluctuation_L2']['mean']:.3f} | SigStrip={e['sigma_strip_rel_err']}",
            flush=True,
        )
    results["exp5b_crossres_ablation"] = exp5b

    # ==== Exp 14: 2D Q1 Linear Elasticity Two-Scale FE^2 Holed Composite Cantilever Plate ====
    print("\n=== Exp 14: 2D Q1 Linear Elasticity FE^2 Holed Composite Plate (E1/E0 in [30, 1000] OOD) ===", flush=True)
    N_mic_el = 32 if args.smoke else 64
    el_tr = build_elas_rve_pack(N_mic_el, 8 if args.smoke else 48, seed=1401, E_lo=10.0, E_hi=200.0)
    Mx_el, My_el = (8, 4) if args.smoke else (20, 10)
    holed_pk = build_holed_plate_fe2_micro(Mx=Mx_el, My=My_el, N_micro=N_mic_el, seed=2027)
    n_act_el = int(holed_pk["active_mask"].sum())

    # Assemble exact 3x3 symmetric Voigt stiffness tensor C_true[e, i, j]
    C_true_raw = holed_pk["stress_true"].view(n_act_el, 3, 3).transpose(1, 2).numpy()
    u_ref, se_ref, tip_ref, _, _ = solve_macro_holed_elasticity(C_true_raw, holed_pk["active_mask"], holed_pk["Lx"], holed_pk["Ly"])
    print(f"  Holed Plate FE^2 ({n_act_el} active RVEs x 3 modes, {N_mic_el}x{N_mic_el}): exact micro={holed_pk['micro_solve_s']:.2f}s | tip_uy={tip_ref:.5f}", flush=True)

    exp14 = {"micro_solve_s": round(holed_pk["micro_solve_s"], 2), "n_active_elems": n_act_el, "N_micro": N_mic_el, "models": {}}
    ep_el = 2 if args.smoke else max(60, int(0.8 * ep))
    for name, is_b, ctor in [
        ("BCNO-Elas", True, lambda: BCNO_Elas_2D(steps=4, hidden=32)),
        ("ResUNet-Elas", False, lambda: ResUNet_Elas_2D(width=32)),
        ("FNO-Elas", False, lambda: FNO_Elas_2D(width=32, modes=16, depth=4)),
    ]:
        torch.manual_seed(0)
        mdl = ctor().float().to(device)
        opt = torch.optim.AdamW(mdl.parameters(), lr=(3e-3 if is_b else 2e-3), weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep_el)
        B_tr = el_tr["E_elem"].shape[0]
        for _ in range(ep_el):
            mdl.train()
            perm = torch.randperm(B_tr)
            for s in range(0, B_tr, 9):
                idx = perm[s : min(s + 9, B_tr)]
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

        # Predict on all active holed-plate elements
        mdl.eval()
        B_h = holed_pk["E_elem"].shape[0]
        vps = []
        if device.type == "cuda": torch.cuda.synchronize()
        t_inf = time.time()
        with torch.no_grad():
            for s in range(0, B_h, 12):
                e = min(s + 12, B_h)
                Ee = holed_pk["E_elem"][s:e].to(device); Em = holed_pk["E_macro"][s:e].to(device)
                if is_b:
                    vp, _ = mdl(Ee, Em, holed_pk["nonperc_el"][s:e].to(device), holed_pk["node_lbls"][s:e].to(device), holed_pk["max_comp"], holed_pk["dx_n"][s:e].to(device), holed_pk["dy_n"][s:e].to(device), holed_pk["strip_n"][s:e].to(device))
                else:
                    vp = mdl(Ee, Em)
                vps.append(vp.cpu())
        if device.type == "cuda": torch.cuda.synchronize()
        inf_s = time.time() - t_inf
        vp_all = torch.cat(vps, 0)
        if is_b:
            s_pred = q1_stress_dipole_readout(vp_all.double(), holed_pk["E_elem"].double(), holed_pk["E_macro"].double(), holed_pk["nonperc_el"], holed_pk["dx_n"].double(), holed_pk["dy_n"].double()).float()
        else:
            s_pred = q1_stress_dipole_readout(vp_all.double(), holed_pk["E_elem"].double(), holed_pk["E_macro"].double(), torch.zeros_like(holed_pk["nonperc_el"]), holed_pk["dx_n"].double(), holed_pk["dy_n"].double()).float()

        C_pred = s_pred.view(n_act_el, 3, 3).transpose(1, 2).numpy()
        ceff_rel_err = float(np.mean(np.linalg.norm(C_pred - C_true_raw, axis=(1, 2)) / np.linalg.norm(C_true_raw, axis=(1, 2))))
        sym_err = float(np.mean(np.linalg.norm(C_pred - np.swapaxes(C_pred, 1, 2), axis=(1, 2)) / np.maximum(np.linalg.norm(C_pred, axis=(1, 2)), 1e-6)))
        C_sym = 0.5 * (C_pred + np.swapaxes(C_pred, 1, 2))
        eigs_c = np.linalg.eigvalsh(C_sym)[:, 0]
        spd_rate_c = float(np.mean(eigs_c > 1e-6))

        u_raw, se_raw, tip_raw, spd_all_raw, min_eig_raw = solve_macro_holed_elasticity(C_pred, holed_pk["active_mask"], holed_pk["Lx"], holed_pk["Ly"])
        C_proj = spd_project_3x3(C_pred, min_eig=0.5)
        u_proj, se_proj, tip_proj, spd_all_proj, _ = solve_macro_holed_elasticity(C_proj, holed_pk["active_mask"], holed_pk["Lx"], holed_pk["Ly"])

        exp14["models"][name] = {
            "params": sum(p.numel() for p in mdl.parameters()),
            "inf_s": round(inf_s, 3),
            "speedup": round(holed_pk["micro_solve_s"] / max(inf_s, 1e-4), 1),
            "ceff_rel_err": ceff_rel_err,
            "onsager_sym_err": sym_err,
            "spd_rate_raw": spd_rate_c,
            "min_eig_raw": float(np.min(eigs_c)),
            "macro_u_L2_raw": float(np.linalg.norm(u_raw - u_ref) / np.linalg.norm(u_ref)) if np.isfinite(u_raw).all() else float("nan"),
            "macro_se_err_raw": float(abs(se_raw - se_ref) / abs(se_ref)) if np.isfinite(se_raw) else float("nan"),
            "macro_tip_err_raw": float(abs(tip_raw - tip_ref) / abs(tip_ref)) if np.isfinite(tip_raw) else float("nan"),
            "spd_all_proj": spd_all_proj,
            "macro_u_L2_proj": float(np.linalg.norm(u_proj - u_ref) / np.linalg.norm(u_ref)),
            "macro_se_err_proj": float(abs(se_proj - se_ref) / abs(se_ref)),
            "macro_tip_err_proj": float(abs(tip_proj - tip_ref) / abs(tip_ref)),
        }
        m14 = exp14["models"][name]
        print(
            f"  [14 {name:12s}] inf={inf_s:.3f}s ({m14['speedup']}x) | CeffErr={ceff_rel_err:.3f} Sym={sym_err:.3f} SPD={spd_rate_c:.2f} | "
            f"MacroU(raw/proj)={m14['macro_u_L2_raw']:.4f}/{m14['macro_u_L2_proj']:.4f} TipErr={m14['macro_tip_err_proj']:.4f}",
            flush=True,
        )
    results["exp14_elasticity_fe2"] = exp14

    out_path = root / "bcno_v5_rigor_addons_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
