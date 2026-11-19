"""CMAME v9 — fix Part C, reframe Part B, add CT-in-loop FE2.

Parts:
  B) FE2 fix: PRIMARY 16x16 thermal + 20x10 elastic with diagnostics;
     STRESS 24x24 / 30x15 kept as degradation probe (not headline).
     Elastic adds Ceff vs tip/energy to expose tip-only luck.
  C) Fixed 64^3 3-seed (BCNO3D gets labels/max_comp in train).
  E) Engineering: Ruhr CT morphologies in the FE2 loop (from rve_specs_compact.npz).

Upload flat into /content (same pack as v8) plus:
  rve_specs_compact.npz          (required for E)
  rve_3d_ruhr64_compact.npz      (optional for C Ruhr)

Outputs: bcno_v9_engineering_fix_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baselines_2d import FNO2dSmall, MgNO2dWrapper, ResUNetSmall
from bcno_elasticity_2d import (
    BCNO_Elas_2D,
    FNO_Elas_2D,
    ResUNet_Elas_2D,
    q1_stress_dipole_readout,
)
from bcno_model_2d import BCNO2D, precompute_floating_components
from bcno_model_3d import BCNO3D, FNO3dSmall, ResUNet3dSmall
from colab_data_gen import fast_interior_solve
from rve_discrete_operator_2d import conservative_flux_columns
from run_bcno_fe2_audit import build_fe2_micro_suite, predict_keff_cols, solve_macro_q1_plate
from run_bcno_paper_suite import agg_seeds, paired_tensor_error, train_model
from run_bcno_scale_and_fe2 import build_3d_64_synth_pack, load_3d_ruhr64_pack
from run_bcno_v5_rigor_addons import (
    build_elas_rve_pack,
    build_fiber_pack_custom,
    build_holed_plate_fe2_micro,
    solve_macro_holed_elasticity,
    spd_project_2x2,
    spd_project_3x3,
)
from run_bcno_v6_audit_fixes import eval_3d_dual, reinit_baseline_properly
from run_bcno_v8_scale_engineering import train_3d_any


def _rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / max(np.linalg.norm(np.asarray(b)), 1e-12))


def _train_2d(factory, is_b, tr, epochs, lr, seed, device):
    torch.manual_seed(seed)
    mdl = factory()
    train_model(
        mdl, is_b, tr, epochs=epochs, batch_size=8, lr=lr, seed=seed, device=device,
        use_keff_loss=is_b, use_sigma_loss=is_b, use_energy_loss=False,
    )
    return mdl


def plate_dissipation(T: np.ndarray, K: np.ndarray, M: int) -> float:
    """Stable engineering scalar: sum_e gradT^T K_e gradT * area (Q1 element centers)."""
    h = 1.0 / float(M)
    diss = 0.0
    for ei in range(M):
        for ej in range(M):
            dTx = (T[ei + 1, ej] + T[ei + 1, ej + 1] - T[ei, ej] - T[ei, ej + 1]) / (2.0 * h)
            dTy = (T[ei, ej + 1] + T[ei + 1, ej + 1] - T[ei, ej] - T[ei + 1, ej]) / (2.0 * h)
            g = np.array([dTx, dTy], dtype=np.float64)
            diss += float(g @ K[ei, ej] @ g) * (h * h)
    return diss


def train_eval_thermal_fe2(M, Nmic, tr, fe2, seeds, ep, device, tag: str) -> dict:
    n_el = M * M
    K_true = fe2["k_cols"].view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M, M, 2, 2)
    T_ref, flux_ref, _ = solve_macro_q1_plate(K_true, M=M)
    diss_ref = plate_dissipation(T_ref, 0.5 * (K_true + np.swapaxes(K_true, -1, -2)), M)
    out = {
        "tag": tag, "M_macro": M, "N_micro": Nmic,
        "micro_solve_s": round(fe2["micro_solve_s"], 2),
        "diss_ref": diss_ref, "models": {},
    }
    for name, factory, is_b, lr in [
        ("BC-NO", lambda: BCNO2D(), True, 3e-3),
        ("ResUNet", lambda: ResUNetSmall(), False, 1e-3),
        ("FNO2d", lambda: FNO2dSmall(), False, 1e-3),
        ("MgNO2d", lambda: MgNO2dWrapper(), False, 1e-3),
    ]:
        rows = []
        for sd in seeds:
            print(f"  [{tag}] thermal {name} seed={sd}", flush=True)
            mdl = _train_2d(factory, is_b, tr, ep, lr, sd, device)
            _, k_dip, _, inf_s = predict_keff_cols(mdl, fe2, is_b, device)
            K_surr = k_dip.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M, M, 2, 2)
            K_spd = spd_project_2x2(K_surr, min_eig=1.0)
            T_spd, flux_spd, _ = solve_macro_q1_plate(K_spd, M=M)
            diss = plate_dissipation(T_spd, 0.5 * (K_spd + np.swapaxes(K_spd, -1, -2)), M)
            rows.append({
                "infer_s": round(float(inf_s), 3),
                "K_fro": round(paired_tensor_error(k_dip, fe2["k_cols"]), 4),
                "T_spd": round(_rel(T_spd, T_ref), 4),
                "flux_edge": round(_rel(np.asarray(flux_spd), np.asarray(flux_ref)), 4),
                "diss_spd": round(abs(diss - diss_ref) / max(abs(diss_ref), 1e-12), 4),
            })
        out["models"][name] = agg_seeds(rows) if len(rows) > 1 else rows[0]
        print(f"    {name}:", out["models"][name], flush=True)
    return out


def train_eval_elastic_fe2(Mx, My, Nmic, el_tr, seeds, ep, device, tag: str) -> dict:
    holed = build_holed_plate_fe2_micro(Mx=Mx, My=My, N_micro=Nmic, seed=2027)
    n_act = int(holed["active_mask"].sum())
    C_true = holed["stress_true"].view(n_act, 3, 3).transpose(1, 2).numpy()
    u_ref, se_ref, tip_ref, _, _ = solve_macro_holed_elasticity(C_true, holed["active_mask"], holed["Lx"], holed["Ly"])
    out = {
        "tag": tag, "Mx": Mx, "My": My, "N_micro": Nmic, "n_active": n_act,
        "micro_solve_s": round(holed["micro_solve_s"], 2),
        "tip_ref": float(tip_ref), "models": {},
    }
    for name, is_b, factory, lr in [
        ("BCNO-Elas", True, lambda: BCNO_Elas_2D(steps=4, hidden=32), 3e-3),
        ("ResUNet-Elas", False, lambda: ResUNet_Elas_2D(width=32), 2e-3),
        ("FNO-Elas", False, lambda: FNO_Elas_2D(width=32, modes=16, depth=4), 2e-3),
    ]:
        rows = []
        for sd in seeds:
            print(f"  [{tag}] elastic {name} seed={sd}", flush=True)
            torch.manual_seed(sd)
            mdl = factory().float().to(device)
            opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-5)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
            B_tr = el_tr["E_elem"].shape[0]
            for _ in range(ep):
                mdl.train()
                perm = torch.randperm(B_tr)
                for s in range(0, B_tr, 9):
                    idx = perm[s:s + 9]
                    Ee = el_tr["E_elem"][idx].to(device)
                    Em = el_tr["E_macro"][idx].to(device)
                    vt = el_tr["v_true"][idx].to(device)
                    if is_b:
                        vp, _ = mdl(
                            Ee, Em, el_tr["nonperc_el"][idx].to(device), el_tr["node_lbls"][idx].to(device),
                            el_tr["max_comp"], el_tr["dx_n"][idx].to(device), el_tr["dy_n"][idx].to(device),
                            el_tr["strip_n"][idx].to(device),
                        )
                    else:
                        vp = mdl(Ee, Em)
                    loss = (torch.sqrt(((vp - vt) ** 2).sum((1, 2, 3)) + 1e-8) / torch.sqrt((vt ** 2).sum((1, 2, 3)) + 1e-6).clamp_min(1e-2)).mean()
                    if is_b:
                        sp = q1_stress_dipole_readout(
                            vp, Ee, Em, el_tr["nonperc_el"][idx].to(device),
                            el_tr["dx_n"][idx].to(device), el_tr["dy_n"][idx].to(device),
                        )
                        st = el_tr["stress_true"][idx].to(device)
                        loss = loss + 0.25 * (torch.linalg.vector_norm(sp - st, dim=1) / torch.linalg.vector_norm(st, dim=1).clamp_min(0.1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                sch.step()
            mdl.eval()
            vps = []
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                for s in range(0, holed["E_elem"].shape[0], 12):
                    e = min(s + 12, holed["E_elem"].shape[0])
                    Ee = holed["E_elem"][s:e].to(device)
                    Em = holed["E_macro"][s:e].to(device)
                    if is_b:
                        vp, _ = mdl(
                            Ee, Em, holed["nonperc_el"][s:e].to(device), holed["node_lbls"][s:e].to(device),
                            holed["max_comp"], holed["dx_n"][s:e].to(device), holed["dy_n"][s:e].to(device),
                            holed["strip_n"][s:e].to(device),
                        )
                    else:
                        vp = mdl(Ee, Em)
                    vps.append(vp.cpu())
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_s = time.time() - t0
            vp_all = torch.cat(vps, 0)
            nonperc = holed["nonperc_el"] if is_b else torch.zeros_like(holed["nonperc_el"])
            s_pred = q1_stress_dipole_readout(
                vp_all.double(), holed["E_elem"].double(), holed["E_macro"].double(),
                nonperc, holed["dx_n"].double(), holed["dy_n"].double(),
            ).float()
            C_pred = s_pred.view(n_act, 3, 3).transpose(1, 2).numpy()
            ceff = float(np.mean(np.linalg.norm(C_pred - C_true, axis=(1, 2)) / np.linalg.norm(C_true, axis=(1, 2))))
            C_proj = spd_project_3x3(C_pred, min_eig=0.5)
            u_proj, se_proj, tip_proj, _, _ = solve_macro_holed_elasticity(C_proj, holed["active_mask"], holed["Lx"], holed["Ly"])
            tip_e = abs(float(tip_proj) - float(tip_ref)) / max(abs(float(tip_ref)), 1e-12)
            rows.append({
                "infer_s": round(infer_s, 3),
                "Ceff_ood": round(ceff, 4),
                "U_spd": round(_rel(u_proj, u_ref), 4),
                "energy_spd": round(_rel(se_proj, se_ref), 4),
                "tip_spd": round(tip_e, 4),
                # tip can look good while Ceff is wrong; report the gap explicitly
                "tip_over_ceff": round(tip_e / max(ceff, 1e-6), 4),
            })
        out["models"][name] = agg_seeds(rows) if len(rows) > 1 else rows[0]
        print(f"    {name}:", out["models"][name], flush=True)
    return out


def part_b_fix(device: torch.device, smoke: bool) -> dict:
    print("\n=== Part B-fix: primary FE2 + stress probe + diagnostics ===", flush=True)
    seeds = [0] if smoke else [0, 1, 2]
    ep = 4 if smoke else 80
    out = {}

    # PRIMARY (paper headline grids)
    M0, N0 = (6, 64) if smoke else (16, 128)
    print(f"  PRIMARY thermal M={M0}...", flush=True)
    tr0 = build_fiber_pack_custom(N0, 8 if smoke else 128, seed=9101, c_lo=10.0, c_hi=250.0)
    fe2_0 = build_fe2_micro_suite(M_macro=M0, N_micro=N0, seed=2026)
    out["thermal_primary"] = train_eval_thermal_fe2(M0, N0, tr0, fe2_0, seeds, ep, device, "primary16")

    Mx0, My0, Ne0 = ((8, 4, 32) if smoke else (20, 10, 64))
    print(f"  PRIMARY elastic {Mx0}x{My0}...", flush=True)
    el0 = build_elas_rve_pack(Ne0, 8 if smoke else 80, seed=1401, E_lo=10.0, E_hi=200.0)
    out["elastic_primary"] = train_eval_elastic_fe2(Mx0, My0, Ne0, el0, seeds, ep, device, "primary20x10")

    # STRESS (enlarged; do not replace primary claims)
    M1, N1 = (8, 64) if smoke else (24, 128)
    print(f"  STRESS thermal M={M1}...", flush=True)
    # reuse tr0 if same N
    tr1 = tr0 if N1 == N0 else build_fiber_pack_custom(N1, 8 if smoke else 128, seed=9101, c_lo=10.0, c_hi=250.0)
    fe2_1 = build_fe2_micro_suite(M_macro=M1, N_micro=N1, seed=2026)
    out["thermal_stress"] = train_eval_thermal_fe2(M1, N1, tr1, fe2_1, seeds, ep, device, "stress24")

    Mx1, My1, Ne1 = ((10, 5, 32) if smoke else (30, 15, 64))
    print(f"  STRESS elastic {Mx1}x{My1}...", flush=True)
    el1 = el0 if Ne1 == Ne0 else build_elas_rve_pack(Ne1, 8 if smoke else 80, seed=1401, E_lo=10.0, E_hi=200.0)
    out["elastic_stress"] = train_eval_elastic_fe2(Mx1, My1, Ne1, el1, seeds, ep, device, "stress30x15")
    out["policy"] = {
        "primary": "16x16 thermal / 20x10 elastic are manuscript headline numbers",
        "stress": "24x24 / 30x15 are degradation probes; report edge-flux carefully; prefer T_spd + diss_spd",
        "elastic_read": "rank by Ceff_ood (+ energy); tip alone can favor wrong constitutive tensors",
    }
    return out


def part_c_fixed(root: Path, device: torch.device, smoke: bool) -> dict:
    print("\n=== Part C-fixed: 64^3 Synth (+ Ruhr), 3 seeds ===", flush=True)
    n_tr = 2 if smoke else 40
    n_te = 1 if smoke else 10
    ep = 2 if smoke else 50
    seeds = [0] if smoke else [0, 1, 2]
    s_tr = build_3d_64_synth_pack(64, n_tr, 2101, 10.0, 80.0, device)
    s_id = build_3d_64_synth_pack(64, n_te, 2102, 10.0, 80.0, device)
    s_od = build_3d_64_synth_pack(64, n_te, 2103, 200.0, 1000.0, device)
    out = {"synth64": {}}
    for name, ctor, is_b in [
        ("BCNO-3D", lambda: BCNO3D(), True),
        ("ResUNet-3D", lambda: reinit_baseline_properly(ResUNet3dSmall(6, width=16)), False),
        ("FNO-3D", lambda: reinit_baseline_properly(FNO3dSmall(6, width=16, modes=8, depth=4)), False),
    ]:
        rows = []
        for sd in seeds:
            print(f"  {name} seed={sd}", flush=True)
            mdl = train_3d_any(ctor().float(), s_tr, ep, 2 if smoke else 3, 3e-3, sd, device, is_bcno=is_b)
            rows.append({
                "id": eval_3d_dual(mdl, s_id, is_b, device, bs=2),
                "ood": eval_3d_dual(mdl, s_od, is_b, device, bs=2),
            })
        out["synth64"][name] = {
            "id": agg_seeds([r["id"] for r in rows]) if len(rows) > 1 else rows[0]["id"],
            "ood": agg_seeds([r["ood"] for r in rows]) if len(rows) > 1 else rows[0]["ood"],
        }
        print(f"    {name} OOD:", out["synth64"][name]["ood"], flush=True)

    npz = root / "rve_3d_ruhr64_compact.npz"
    if npz.exists():
        r_tr = load_3d_ruhr64_pack(npz, "tr64", device)
        r_te = load_3d_ruhr64_pack(npz, "te64", device)
        out["ruhr64"] = {}
        for name, ctor, is_b in [
            ("BCNO-3D", lambda: BCNO3D(), True),
            ("ResUNet-3D", lambda: reinit_baseline_properly(ResUNet3dSmall(6, width=16)), False),
            ("FNO-3D", lambda: reinit_baseline_properly(FNO3dSmall(6, width=16, modes=8, depth=4)), False),
        ]:
            rows = []
            for sd in seeds:
                print(f"  Ruhr64 {name} seed={sd}", flush=True)
                mdl = train_3d_any(ctor().float(), r_tr, ep, 2 if smoke else 3, 3e-3, sd, device, is_bcno=is_b)
                rows.append(eval_3d_dual(mdl, r_te, is_b, device, bs=2))
            out["ruhr64"][name] = agg_seeds(rows) if len(rows) > 1 else rows[0]
            print(f"  Ruhr64 {name}:", out["ruhr64"][name], flush=True)
    else:
        out["ruhr64"] = {"skipped": "upload rve_3d_ruhr64_compact.npz"}
    return out


def _unpack_ruhr_masks(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    masks, cs = [], []
    for sp in ("train", "test"):
        shape = tuple(int(v) for v in z[f"ruhr_{sp}_shape"])
        n_bits = int(np.prod(shape))
        m = np.unpackbits(z[f"ruhr_{sp}_bits"])[:n_bits].reshape(shape).astype(bool)
        c = z[f"ruhr_{sp}_c"].astype(np.float64)
        masks.append(m)
        cs.append(c)
    return np.concatenate(masks, 0), np.concatenate(cs, 0)


def build_ct_fe2_suite(masks: np.ndarray, M: int, seed: int = 3030) -> dict:
    """Tile real CT binary morphologies onto an M x M FE2 plate with graded contrast."""
    rng = np.random.default_rng(seed)
    N = int(masks.shape[-1])
    kaps, vs, lifts, axs = [], [], [], []
    ax_1d = np.linspace(0.0, 1.0, N, dtype=np.float64)
    xg, yg = np.meshgrid(ax_1d, ax_1d, indexing="ij")
    t0 = time.time()
    for ei in range(M):
        for ej in range(M):
            xm = (ei + 0.5) / M
            # Intermediate→high engineering contrast, graded in x
            c = 50.0 + 450.0 * (xm ** 2)  # [50, 500]
            mi = int(rng.integers(0, masks.shape[0]))
            kap = np.where(masks[mi], c, 1.0).astype(np.float64)
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
    idx = torch.arange(N, dtype=torch.float32)
    ii, jj = idx[:, None].expand(N, N), idx[None, :].expand(N, N)
    dist_t = (torch.minimum(torch.minimum(ii, jj), torch.minimum(N - 1 - ii, N - 1 - jj)) / (N / 2.0))[None].expand(B, N, N)
    labels, mc = precompute_floating_components(kap_t)
    return {
        "kappa": kap_t, "kappa_raw": kap_t, "v_true": v_t, "u_true": lift_t + v_t,
        "lift": lift_t, "ortho": ortho_t, "dist": dist_t, "k_cols": k_cols,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": labels, "max_comp": mc, "x_grid": xg_t[None], "y_grid": yg_t[None],
        "micro_solve_s": solve_s, "M_macro": M, "N_micro": N,
        "contrast_range": [50.0, 500.0], "n_ct_bank": int(masks.shape[0]),
    }


def build_ct_train_pack(masks: np.ndarray, n_geom: int, seed: int, c_lo: float, c_hi: float) -> dict:
    rng = np.random.default_rng(seed)
    N = int(masks.shape[-1])
    kaps, vs, lifts, axs = [], [], [], []
    ax_1d = np.linspace(0.0, 1.0, N, dtype=np.float64)
    xg, yg = np.meshgrid(ax_1d, ax_1d, indexing="ij")
    for i in range(n_geom):
        mi = int(rng.integers(0, masks.shape[0]))
        c = float(rng.uniform(c_lo, c_hi))
        kap = np.where(masks[mi], c, 1.0).astype(np.float64)
        for ax_id in (0, 1):
            v, lift = fast_interior_solve(kap, ax_id)
            kaps.append(kap.astype(np.float32))
            vs.append(v.astype(np.float32))
            lifts.append(lift.astype(np.float32))
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
    labels, mc = precompute_floating_components(kap_t)
    return {
        "kappa": kap_t, "kappa_raw": kap_t, "v_true": v_t, "u_true": lift_t + v_t,
        "lift": lift_t, "ortho": ortho_t, "dist": dist_t, "k_cols": k_cols,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": labels, "max_comp": mc, "x_grid": xg_t[None], "y_grid": yg_t[None],
    }


def part_e_ct_fe2(root: Path, device: torch.device, smoke: bool) -> dict:
    print("\n=== Part E: Ruhr CT morphologies in FE2 loop ===", flush=True)
    npz = root / "rve_specs_compact.npz"
    if not npz.exists():
        return {"skipped": "upload rve_specs_compact.npz to /content"}
    masks, _ = _unpack_ruhr_masks(npz)
    M = 6 if smoke else 16
    seeds = [0] if smoke else [0, 1, 2]
    ep = 4 if smoke else 80
    n_tr = 8 if smoke else 96  # 96 geoms x 2 axes
    print(f"  CT bank={masks.shape[0]} masks@{masks.shape[-1]}, FE2 M={M}", flush=True)
    tr = build_ct_train_pack(masks, n_tr, seed=4041, c_lo=40.0, c_hi=400.0)
    # Mix a little synthetic fiber so models see both families
    fib = build_fiber_pack_custom(int(masks.shape[-1]), 4 if smoke else 64, seed=4042, c_lo=10.0, c_hi=250.0)
    # concatenate packs
    def _cat(a, b, keys):
        out = {}
        for k in keys:
            if k == "max_comp":
                out[k] = max(int(a[k]), int(b[k]))
            elif k in ("x_grid", "y_grid"):
                out[k] = a[k]
            else:
                out[k] = torch.cat([a[k], b[k]], 0)
        return out
    keys = ["kappa", "kappa_raw", "v_true", "u_true", "lift", "ortho", "dist", "k_cols", "load_axis", "labels", "max_comp", "x_grid", "y_grid"]
    tr_mix = _cat(tr, fib, keys)
    fe2 = build_ct_fe2_suite(masks, M=M, seed=3030)
    result = train_eval_thermal_fe2(M, int(masks.shape[-1]), tr_mix, fe2, seeds, ep, device, "ct_fe2")
    result["ct_bank"] = int(masks.shape[0])
    result["train_ct_fields"] = int(tr["kappa"].shape[0])
    result["train_fiber_fields"] = int(fib["kappa"].shape[0])
    result["contrast_range"] = fe2["contrast_range"]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="BCE", help="B|C|E|BC|BE|CE|BCE|ALL")
    args = ap.parse_args()
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    only = args.only.upper()
    print("Device:", device, "smoke:", args.smoke, "only:", only, flush=True)
    out: dict = {"device": str(device), "smoke": args.smoke, "only": only}

    if only in ("ALL", "B", "BC", "BE", "BCE"):
        out["part_b_fix"] = part_b_fix(device, args.smoke)
    if only in ("ALL", "C", "BC", "CE", "BCE"):
        out["part_c_64"] = part_c_fixed(root, device, args.smoke)
    if only in ("ALL", "E", "BE", "CE", "BCE"):
        out["part_e_ct_fe2"] = part_e_ct_fe2(root, device, args.smoke)

    path = root / "bcno_v9_engineering_fix_results.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved", path, flush=True)


if __name__ == "__main__":
    main()
