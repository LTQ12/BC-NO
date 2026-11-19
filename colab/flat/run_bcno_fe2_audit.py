"""Exp 10: CMAME Constitutive Admissibility Audit + Two-Scale FE^2 Engineering Case Study.

Part A (Constitutive Admissibility on N=128 fiber_columns ID + OOD):
  - Onsager symmetry error |K_12 - K_21| / ||K||_F
  - Positive-definiteness (SPD) pass rate: lambda_min(sym(K)) > 0
  - Voigt-Reuss physical bound satisfaction rate along active load axis

Part B (Two-Scale FE^2 Macro-Micro Composite Plate Homogenization):
  - Macroscopic 16x16 Q1 finite-element thermal plate with spatially varying fiber RVE
    at each element (256 heterogeneous 128x128 RVEs, kappa in [20, 250], vf in [0.28, 0.62]).
  - Compares Exact FE^2 (256 float64 micro-solves + macro FEM) against BC-NO FE^2 and
    baseline FE^2 surrogates (MgNO2d, FFNO2d, CNO2d, ResUNet, FNO2d).
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

from baselines_2d import CNO2dWrapper, FFNO2d, FNO2dSmall, MgNO2dWrapper, ResUNetSmall
from bcno_model_2d import BCNO2D, dipole_keff_columns, precompute_floating_components
from colab_data_gen import fast_interior_solve, generate_synthetic_benchmarks, make_fiber_columns
from rve_discrete_operator_2d import conservative_flux_columns
from run_bcno_paper_suite import pack_from_pt, train_model


def constitutive_audit(k_cols: torch.Tensor, kappa: torch.Tensor, load_axis: torch.Tensor) -> dict[str, float]:
    """Audit Onsager symmetry, SPD, and Voigt-Reuss bounds on paired [axis0, axis1] tensor rows."""
    n_pairs = k_cols.shape[0] // 2
    T = k_cols[: 2 * n_pairs].view(n_pairs, 2, 2).transpose(1, 2)  # K_eff[i, j] = col j[i]
    # Note: col 0 is [K00, K10], col 1 is [K01, K11]
    sym_err = (T[:, 0, 1] - T[:, 1, 0]).abs() / torch.linalg.vector_norm(T, dim=(1, 2)).clamp_min(1e-6)
    T_sym = 0.5 * (T + T.transpose(1, 2))
    eigs = torch.linalg.eigvalsh(T_sym)
    spd_rate = float((eigs[:, 0] > 0.0).float().mean().item())

    # Voigt (arithmetic mean) & Reuss (harmonic mean) bounds on diagonal entries K00 and K11
    kap_pair = kappa[0 : 2 * n_pairs : 2]
    k_voigt = kap_pair.mean(dim=(1, 2))
    k_reuss = 1.0 / (1.0 / kap_pair.clamp_min(1e-6)).mean(dim=(1, 2))
    tol = 1e-3
    in_bounds = (
        (T[:, 0, 0] >= k_reuss * (1.0 - tol)) & (T[:, 0, 0] <= k_voigt * (1.0 + tol))
        & (T[:, 1, 1] >= k_reuss * (1.0 - tol)) & (T[:, 1, 1] <= k_voigt * (1.0 + tol))
    )
    return {
        "OnsagerSymErr": float(sym_err.mean().item()),
        "OnsagerSymMax": float(sym_err.max().item()),
        "SPDRate": spd_rate,
        "VoigtReussRate": float(in_bounds.float().mean().item()),
        "MinEig": float(eigs[:, 0].min().item()),
    }


def solve_macro_q1_plate(K_tensors: np.ndarray, M: int = 16) -> tuple[np.ndarray, np.ndarray, bool]:
    """Solve macroscopic -div(K(x) grad T) = q(x) on [0,1]^2 with 2x2 Q1 FEM (M x M elements).

    BC: T(0, y) = 0, T(1, y) = 1 + 0.3*sin(pi*y), T(x, 0) = x, T(x, 1) = x^2 (non-affine thermal load).
    Source q(x, y) = 2.0 * sin(pi*x) * sin(pi*y).
    Returns: nodal temperature T (M+1, M+1), right-boundary heat flux integral, spd_ok flag.
    """
    h = 1.0 / float(M)
    gp = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    nodes_ref = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    N = M + 1

    # Check positive-definiteness of symmetrized K_tensors before macro assembly
    K_sym = 0.5 * (K_tensors + np.swapaxes(K_tensors, -1, -2))
    min_eig = float(np.min(np.linalg.eigvalsh(K_sym)))
    spd_ok = min_eig > 1e-6

    ii, jj = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    n00 = ii * N + jj
    n10 = (ii + 1) * N + jj
    n11 = (ii + 1) * N + (jj + 1)
    n01 = ii * N + (jj + 1)
    elem_nodes = np.stack([n00, n10, n11, n01], axis=-1).reshape(-1, 4)

    Ke_all = np.zeros((M * M, 4, 4), dtype=np.float64)
    fe_all = np.zeros((M * M, 4), dtype=np.float64)
    K_flat = K_sym.reshape(M * M, 2, 2)

    for xi in gp:
        for eta in gp:
            N_val = np.array([0.25 * (1 + sx * xi) * (1 + sy * eta) for sx, sy in nodes_ref])
            dN = np.array([
                [0.25 * sx * (1 + sy * eta) * (2.0 / h), 0.25 * sy * (1 + sx * xi) * (2.0 / h)]
                for sx, sy in nodes_ref
            ])  # (4, 2)
            w = 0.25 * h * h
            Ke_all += w * np.einsum("ai,eij,bj->eab", dN, K_flat, dN)
            # Macro element physical center + gp offset
            ex_c = (ii.ravel() + 0.5 + 0.5 * xi) * h
            ey_c = (jj.ravel() + 0.5 + 0.5 * eta) * h
            q_val = 2.0 * np.sin(math.pi * ex_c) * np.sin(math.pi * ey_c)
            fe_all += w * q_val[:, None] * N_val[None, :]

    rr = np.repeat(elem_nodes[:, :, None], 4, axis=2).reshape(-1)
    cc = np.repeat(elem_nodes[:, None, :], 4, axis=1).reshape(-1)
    A_glob = sp.coo_matrix((Ke_all.reshape(-1), (rr, cc)), shape=(N * N, N * N)).tocsr()
    b_glob = np.zeros(N * N, dtype=np.float64)
    np.add.at(b_glob, elem_nodes, fe_all)

    # Boundary values
    ax = np.linspace(0.0, 1.0, N)
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    T_bc = np.zeros((N, N), dtype=np.float64)
    T_bc[0, :] = 0.0
    T_bc[-1, :] = 1.0 + 0.3 * np.sin(math.pi * gy[-1, :])
    T_bc[:, 0] = gx[:, 0]
    T_bc[:, -1] = gx[:, -1] ** 2

    is_bdry = np.zeros((N, N), dtype=bool)
    is_bdry[0, :] = True; is_bdry[-1, :] = True; is_bdry[:, 0] = True; is_bdry[:, -1] = True
    free = ~is_bdry.ravel()

    rhs_free = b_glob[free] - A_glob[free][:, ~free] @ T_bc.ravel()[~free]
    A_free = A_glob[free][:, free].tocsc()
    try:
        T_free = spla.spsolve(A_free, rhs_free)
    except Exception:
        return np.full((N, N), np.nan), np.nan, False
    T = T_bc.copy().ravel()
    T[free] = T_free
    T = T.reshape(N, N)
    # Right boundary flux q_out = mean_y(K_xx * dT/dx at x=1)
    dTdx_right = (T[-1, :-1] - T[-2, :-1]) / h
    flux_right = float(np.mean(K_sym[-1, :, 0, 0] * dTdx_right))
    return T, np.array([flux_right]), spd_ok


def build_fe2_micro_suite(M_macro: int = 16, N_micro: int = 128, seed: int = 2026) -> dict:
    """Build M_macro x M_macro spatially graded microstructures and exact float64 micro-solves."""
    rng = np.random.default_rng(seed)
    B_geom = M_macro * M_macro
    kaps, vs, lifts, ks, axs = [], [], [], [], []
    ax_1d = np.linspace(0.0, 1.0, N_micro, dtype=np.float64)
    xg, yg = np.meshgrid(ax_1d, ax_1d, indexing="ij")
    t0 = time.time()
    for ei in range(M_macro):
        for ej in range(M_macro):
            xm = (ei + 0.5) / M_macro
            ym = (ej + 0.5) / M_macro
            # Spatially graded contrast & volume fraction across the macro plate!
            vf = 0.28 + 0.34 * (0.5 * (1.0 + math.sin(math.pi * xm) * math.cos(math.pi * ym)))
            c = 20.0 + 230.0 * (xm ** 2)  # ranges from 20 at x=0 to 250 at x=1 (spans ID->OOD!)
            kap = make_fiber_columns(N_micro, int(rng.integers(10000, 999999)), c, vf)
            for ax_id in (0, 1):
                v, lift = fast_interior_solve(kap, ax_id)
                kaps.append(kap.astype(np.float32))
                vs.append(v.astype(np.float32))
                lifts.append(lift.astype(np.float32))
                axs.append(ax_id)
    solve_s = time.time() - t0
    kap_t = torch.from_numpy(np.stack(kaps))
    v_t = torch.from_numpy(np.stack(vs))
    lift_t = torch.from_numpy(np.stack(lifts))
    k_cols = conservative_flux_columns((lift_t + v_t).double(), kap_t.double(), kappa_channel=0).float()

    B = kap_t.shape[0]
    xg_t = torch.from_numpy(xg.astype(np.float32))
    yg_t = torch.from_numpy(yg.astype(np.float32))
    ortho_t = torch.stack([yg_t if a == 0 else xg_t for a in axs], dim=0)
    idx = torch.arange(N_micro, dtype=torch.float32)
    ii, jj = idx[:, None].expand(N_micro, N_micro), idx[None, :].expand(N_micro, N_micro)
    dist_t = (torch.minimum(torch.minimum(ii, jj), torch.minimum(N_micro - 1 - ii, N_micro - 1 - jj)) / (N_micro / 2.0))[None].expand(B, N_micro, N_micro)
    labels, mc = precompute_floating_components(kap_t)
    return {
        "kappa": kap_t, "kappa_raw": kap_t, "v_true": v_t, "u_true": lift_t + v_t,
        "lift": lift_t, "ortho": ortho_t, "dist": dist_t, "k_cols": k_cols,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": labels, "max_comp": mc, "x_grid": xg_t[None], "y_grid": yg_t[None],
        "micro_solve_s": solve_s, "M_macro": M_macro,
    }


@torch.no_grad()
def predict_keff_cols(mdl, pk: dict, is_bcno: bool, device: torch.device, batch_size: int = 16):
    mdl.eval()
    B = pk["kappa"].shape[0]
    ups = []
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        kap = pk["kappa"][s:e].to(device); lift = pk["lift"][s:e].to(device)
        ortho = pk["ortho"][s:e].to(device); dist = pk["dist"][s:e].to(device)
        if is_bcno:
            up, _ = mdl(kap, lift, ortho, dist, pk["labels"][s:e].to(device), pk["max_comp"])
        else:
            up = mdl(kap, lift, ortho, dist)
        ups.append(up.cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    inf_s = time.time() - t0
    u_pred = torch.cat(ups, 0)
    k_dip = dipole_keff_columns(u_pred.double(), pk["kappa_raw"].double(), pk["x_grid"].double(), pk["y_grid"].double()).float()
    k_naive = conservative_flux_columns(u_pred.double(), pk["kappa_raw"].double(), kappa_channel=0).float()
    return (k_dip if is_bcno else k_naive), k_dip, k_naive, inf_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ep = 2 if args.smoke else args.epochs
    M_macro = 6 if args.smoke else 16
    root = Path(__file__).resolve().parent
    generate_synthetic_benchmarks(root)

    f_tr = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "train")
    f_id = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "test")
    f_od = pack_from_pt(root / "rve_fiber_columns_ood_N128_f64.pt", "test")
    fe2_pk = build_fe2_micro_suite(M_macro=M_macro, N_micro=128, seed=2026)

    # Exact FE^2 reference macro solution
    n_el = M_macro * M_macro
    K_true_macro = fe2_pk["k_cols"].view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_macro, M_macro, 2, 2)
    T_ref, flux_ref, _ = solve_macro_q1_plate(K_true_macro, M=M_macro)
    print(f"FE^2 micro suite ({n_el} RVEs x 2 axes): exact micro solve = {fe2_pk['micro_solve_s']:.2f}s | macro flux_ref = {float(flux_ref[0]):.4f}", flush=True)

    models_spec = [
        ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
        ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
        ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ("FNO2d", False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
    ]

    results = {"micro_solve_s": round(fe2_pk["micro_solve_s"], 2), "M_macro": M_macro, "models": {}}
    # Reference truth audit
    results["truth_audit"] = {
        "id": constitutive_audit(f_id["k_cols"], f_id["kappa_raw"], f_id["load_axis"]),
        "ood": constitutive_audit(f_od["k_cols"], f_od["kappa_raw"], f_od["load_axis"]),
    }

    for name, is_b, use_k, use_sig, ctor in models_spec:
        torch.manual_seed(args.seed)
        mdl = ctor().float()
        train_model(mdl, is_b, f_tr, epochs=ep, batch_size=8, lr=(3e-3 if is_b else 2e-3), seed=args.seed, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)

        k_id_p, _, _, _ = predict_keff_cols(mdl, f_id, is_b, device)
        k_od_p, _, _, _ = predict_keff_cols(mdl, f_od, is_b, device)
        audit_id = constitutive_audit(k_id_p, f_id["kappa_raw"], f_id["load_axis"])
        audit_od = constitutive_audit(k_od_p, f_od["kappa_raw"], f_od["load_axis"])

        k_fe2_p, k_fe2_dip, k_fe2_naive, inf_s = predict_keff_cols(mdl, fe2_pk, is_b, device)
        K_surr = k_fe2_p.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_macro, M_macro, 2, 2)
        T_surr, flux_surr, spd_ok = solve_macro_q1_plate(K_surr, M=M_macro)
        t_err = float(np.linalg.norm(T_surr - T_ref) / np.linalg.norm(T_ref)) if np.isfinite(T_surr).all() else float("nan")
        f_err = float(abs(flux_surr[0] - flux_ref[0]) / abs(flux_ref[0])) if np.isfinite(flux_surr).all() else float("nan")
        rve_err = float((torch.linalg.vector_norm(k_fe2_p - fe2_pk["k_cols"], dim=1) / torch.linalg.vector_norm(fe2_pk["k_cols"], dim=1)).mean())

        results["models"][name] = {
            "audit_id": audit_id,
            "audit_ood": audit_od,
            "fe2": {
                "micro_infer_s": round(inf_s, 3),
                "speedup_vs_lu": round(fe2_pk["micro_solve_s"] / max(inf_s, 1e-4), 1),
                "micro_rve_rel_err": rve_err,
                "macro_T_rel_L2": t_err,
                "macro_flux_rel_err": f_err,
                "macro_spd_all": spd_ok,
            },
        }
        print(
            f"[{name:8s}] OOD Audit: Sym={audit_od['OnsagerSymErr']:.3f} SPD={audit_od['SPDRate']:.2f} VR={audit_od['VoigtReussRate']:.2f} | "
            f"FE^2: inf={inf_s:.3f}s ({fe2_pk['micro_solve_s']/max(inf_s,1e-4):.0f}x) microErr={rve_err:.3f} macroT_L2={t_err:.4f} macroFluxErr={f_err:.4f} SPD={spd_ok}",
            flush=True,
        )

    (root / "bcno_fe2_audit_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: bcno_fe2_audit_results.json", flush=True)


if __name__ == "__main__":
    main()
