"""CMAME v8 — scale + engineering (flat /content; generate data on the fly).

Four deliverables:
  A) Exp.11 narrative already elevated in main.tex; optional re-run 1000-train 3-seed fiber
  B) Enlarged FE2: thermal 24x24 (576 RVE) + elastic 30x15 holed cantilever, 3 model seeds
  C) 64^3 Ruhr CT + synth: BCNO-3D / ResUNet-3D / FNO-3D, 3 seeds (needs compact npz if present)
  D) Expanded Exp.1 (64 train shapes) + Exp.4b-scale hard/wavy packs (256 train)

No large .pt upload required. Optional upload for Part C:
  rve_3d_ruhr64_compact.npz  (small; if missing, Ruhr64 is skipped and Synth64 still runs)

Outputs:
  bcno_v8_scale_engineering_results.json
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
from colab_data_gen import fast_interior_solve, make_hard_topology
from rve_discrete_operator_2d import conservative_flux_columns
from run_bcno_fe2_audit import build_fe2_micro_suite, predict_keff_cols, solve_macro_q1_plate
from run_bcno_paper_suite import agg_seeds, evaluate_pack, paired_tensor_error, train_model
from run_bcno_scale_and_fe2 import build_3d_64_synth_pack, build_large_2d_pack, load_3d_ruhr64_pack
from run_bcno_v5_rigor_addons import (
    build_elas_rve_pack,
    build_fiber_pack_custom,
    build_holed_plate_fe2_micro,
    solve_macro_holed_elasticity,
    spd_project_2x2,
    spd_project_3x3,
)
from run_bcno_v6_audit_fixes import eval_3d_dual, reinit_baseline_properly


def train_3d_any(mdl: nn.Module, pack: dict, epochs: int, batch_size: int, lr: float, seed: int, device: torch.device, is_bcno: bool):
    """Train 3D models; BCNO3D needs labels/max_comp (train_3d_fixed in v6 omits them)."""
    torch.manual_seed(seed)
    mdl = reinit_baseline_properly(mdl).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    B = pack["kappa"].shape[0]
    v_rms = float(torch.sqrt((pack["v_true"] ** 2).mean()).clamp_min(1e-3))
    for _ in range(epochs):
        mdl.train()
        perm = torch.randperm(B)
        for s in range(0, B, batch_size):
            idx = perm[s : min(s + batch_size, B)]
            kap = pack["kappa"][idx].to(device)
            lft = pack["lift"][idx].to(device)
            o1 = pack["o1"][idx].to(device)
            o2 = pack["o2"][idx].to(device)
            dst = pack["dist"][idx].to(device)
            vt = pack["v_true"][idx].to(device)
            if is_bcno:
                up = mdl(kap, lft, o1, o2, dst, pack["labels"][idx].to(device), pack["max_comp"])[0]
            else:
                up = mdl(kap, lft, o1, o2, dst)
            vp = up - lft
            mse_norm = ((vp - vt) / v_rms).pow(2).mean()
            vt_n = torch.sqrt((vt ** 2).sum((1, 2, 3)) + 1e-10)
            nz = vt_n > 1e-4
            rel = (torch.sqrt(((vp[nz] - vt[nz]) ** 2).sum((1, 2, 3)) + 1e-10) / vt_n[nz]).mean() if bool(nz.any()) else 0.0
            loss = 0.5 * mse_norm + 0.5 * rel
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 2.0)
            opt.step()
        sched.step()
    return mdl


def _train_2d(name, factory, is_b, tr, epochs, lr, seed, device):
    torch.manual_seed(seed)
    mdl = factory()
    train_model(
        mdl, is_b, tr, epochs=epochs, batch_size=8, lr=lr, seed=seed, device=device,
        use_keff_loss=is_b, use_sigma_loss=is_b, use_energy_loss=False,
    )
    return mdl


# ---------- D: Expanded Exp.1 & Exp.4b-scale ----------
def make_train_hard_shapes(n: int, n_shapes: int, seed: int, contrast: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    kinds = ("near_wall_box", "u_bridge", "l_shape")
    for i in range(n_shapes):
        kind = kinds[i % 3]
        out.append(make_hard_topology(n, int(rng.integers(1, 1_000_000_000)), contrast, kind))
    return out


def pack_from_kappas(kaps: list[np.ndarray], axes=(0, 1)) -> dict:
    vs, lifts, klist, axs, kaps_out = [], [], [], [], []
    for kap in kaps:
        for ax in axes:
            v, lift = fast_interior_solve(kap, ax)
            kaps_out.append(kap.astype(np.float32))
            vs.append(v.astype(np.float32))
            lifts.append(lift.astype(np.float32))
            axs.append(ax)
    kap_t = torch.from_numpy(np.stack(kaps_out))
    v_t = torch.from_numpy(np.stack(vs))
    lift_t = torch.from_numpy(np.stack(lifts))
    k_cols = conservative_flux_columns((lift_t + v_t).double(), kap_t.double(), kappa_channel=0).float()
    B, H, W = kap_t.shape
    ax1 = torch.linspace(0, 1, H)
    xg, yg = torch.meshgrid(ax1, ax1, indexing="ij")
    ortho = torch.stack([yg if a == 0 else xg for a in axs], 0)
    idx = torch.arange(H, dtype=torch.float32)
    ii, jj = idx[:, None].expand(H, W), idx[None, :].expand(H, W)
    dist = (torch.minimum(torch.minimum(ii, jj), torch.minimum(H - 1 - ii, W - 1 - jj)) / (H / 2.0))[None].expand(B, H, W)
    labels, mc = precompute_floating_components(kap_t)
    return {
        "kappa": kap_t, "kappa_raw": kap_t, "v_true": v_t, "u_true": lift_t + v_t,
        "lift": lift_t, "ortho": ortho, "dist": dist, "k_cols": k_cols,
        "load_axis": torch.tensor(axs, dtype=torch.long),
        "labels": labels, "max_comp": mc, "x_grid": xg[None], "y_grid": yg[None],
    }


def heldout_48_eval(mdl, is_b, device, contrasts=(1e2, 1e4, 1e6)) -> dict:
    from run_bcno_v6_audit_fixes import _exp1_48x48_geoms
    rows = {}
    for c in contrasts:
        rows[str(int(c))] = {}
        for tag, kap in _exp1_48x48_geoms(c):
            pk = pack_from_kappas([kap], axes=(0, 1))
            m = evaluate_pack(mdl, pk, is_b, device, batch_size=2, use_dipole=True)
            rows[str(int(c))][tag] = {
                "Dip": round(m["KeffRelErr"], 4),
                "Naive": round(m["KeffRelErr_Naive"], 4),
                "DiagDip": round(m["DiagRelErr"], 4),
                "Fluct": round(m["Fluctuation_L2"], 4),
            }
    return rows


def part_d_expanded_hard(root: Path, device: torch.device, smoke: bool) -> dict:
    print("\n=== Part D: Expanded Exp.1 (64 shapes) + Exp.4b-scale packs ===", flush=True)
    n_shapes = 8 if smoke else 64
    # multi-contrast train shapes
    kaps_tr = []
    for c in ([20.0, 100.0] if smoke else [20.0, 100.0, 1e3, 1e4]):
        kaps_tr.extend(make_train_hard_shapes(48, max(2, n_shapes // 4), seed=int(c) + 11, contrast=c))
    tr = pack_from_kappas(kaps_tr)
    out = {"n_train_fields": tr["kappa"].shape[0], "models": {}}
    for name, factory, is_b, lr in [
        ("BC-NO", lambda: BCNO2D(), True, 3e-3),
        ("ResUNet", lambda: ResUNetSmall(), False, 1e-3),
        ("FNO2d", lambda: FNO2dSmall(), False, 1e-3),
    ]:
        print(f"  Exp1-scale train {name}...", flush=True)
        mdl = _train_2d(name, factory, is_b, tr, epochs=(5 if smoke else 80), lr=lr, seed=0, device=device)
        out["models"][name] = heldout_48_eval(mdl, is_b, device)
        print(f"    {name} bridge@1e6 Dip={out['models'][name]['1000000']['bridge']['Dip']}", flush=True)

    # Exp.4b-scale: hard + wavy-like via hard_topology families at N=128
    n_tr = 16 if smoke else 256
    n_te = 4 if smoke else 64
    hard_tr = pack_from_kappas(make_train_hard_shapes(128, n_tr, 401, 1e3))
    hard_id = pack_from_kappas(make_train_hard_shapes(128, n_te, 402, 1e3))
    hard_od = pack_from_kappas(make_train_hard_shapes(128, n_te, 403, 1e6))
    out["exp4b_scale"] = {"n_train": hard_tr["kappa"].shape[0], "models": {}}
    for name, factory, is_b, lr in [
        ("BC-NO", lambda: BCNO2D(), True, 3e-3),
        ("MgNO2d", lambda: MgNO2dWrapper(), False, 1e-3),
        ("ResUNet", lambda: ResUNetSmall(), False, 1e-3),
    ]:
        print(f"  Exp4b-scale train {name}...", flush=True)
        mdl = _train_2d(name, factory, is_b, hard_tr, epochs=(5 if smoke else 75), lr=lr, seed=1, device=device)
        out["exp4b_scale"]["models"][name] = {
            "id": evaluate_pack(mdl, hard_id, is_b, device, use_dipole=True),
            "ood_1e6": evaluate_pack(mdl, hard_od, is_b, device, use_dipole=True),
        }
        m = out["exp4b_scale"]["models"][name]
        print(f"    {name} ID Dip={m['id']['KeffRelErr']:.3f} OOD Dip={m['ood_1e6']['KeffRelErr']:.3f}", flush=True)
    return out


# ---------- B: Enlarged FE2 ----------
def part_b_fe2(device: torch.device, smoke: bool) -> dict:
    print("\n=== Part B: Enlarged FE2 (thermal 24x24, elastic 30x15), 3 seeds ===", flush=True)
    M = 8 if smoke else 24
    Mx, My = (8, 4) if smoke else (30, 15)
    Nmic_el = 32 if smoke else 64
    Nmic_th = 64 if smoke else 128
    seeds = [0] if smoke else [0, 1, 2]
    ep = 4 if smoke else 80
    out: dict = {}

    print(f"  Building thermal FE2 micro suite M={M}, N={Nmic_th}...", flush=True)
    fe2 = build_fe2_micro_suite(M_macro=M, N_micro=Nmic_th, seed=2026)
    tr = build_fiber_pack_custom(Nmic_th, 8 if smoke else 128, seed=9101, c_lo=10.0, c_hi=250.0)
    n_el = M * M
    K_true_mac = fe2["k_cols"].view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M, M, 2, 2)
    T_ref, flux_ref, _ = solve_macro_q1_plate(K_true_mac, M=M)
    thermal = {
        "M_macro": M, "N_micro": Nmic_th,
        "micro_solve_s": round(fe2["micro_solve_s"], 2),
        "n_micro_fields": int(fe2["kappa"].shape[0]),
        "models": {},
    }

    def _rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

    for name, factory, is_b, lr in [
        ("BC-NO", lambda: BCNO2D(), True, 3e-3),
        ("ResUNet", lambda: ResUNetSmall(), False, 1e-3),
        ("FNO2d", lambda: FNO2dSmall(), False, 1e-3),
        ("MgNO2d", lambda: MgNO2dWrapper(), False, 1e-3),
    ]:
        seed_rows = []
        for sd in seeds:
            print(f"  thermal {name} seed={sd}", flush=True)
            mdl = _train_2d(name, factory, is_b, tr, epochs=ep, lr=lr, seed=sd, device=device)
            _, k_dip, _, inf_s = predict_keff_cols(mdl, fe2, is_b, device)
            K_surr = k_dip.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M, M, 2, 2)
            K_spd = spd_project_2x2(K_surr, min_eig=1.0)
            T_raw, flux_raw, _ = solve_macro_q1_plate(K_surr, M=M)
            T_spd, flux_spd, _ = solve_macro_q1_plate(K_spd, M=M)
            seed_rows.append({
                "infer_s": round(float(inf_s), 3),
                "T_raw": round(_rel(T_raw, T_ref), 4),
                "T_spd": round(_rel(T_spd, T_ref), 4),
                "flux_raw": round(_rel(np.asarray(flux_raw), np.asarray(flux_ref)), 4),
                "flux_spd": round(_rel(np.asarray(flux_spd), np.asarray(flux_ref)), 4),
                "micro_dip": round(paired_tensor_error(k_dip, fe2["k_cols"]), 4),
            })
        thermal["models"][name] = agg_seeds(seed_rows) if len(seed_rows) > 1 else seed_rows[0]
        print(f"    {name}:", thermal["models"][name], flush=True)
    out["thermal_fe2"] = thermal

    print(f"  Building elastic FE2 Mx={Mx}, My={My}, N={Nmic_el}...", flush=True)
    el_tr = build_elas_rve_pack(Nmic_el, 8 if smoke else 80, seed=1401, E_lo=10.0, E_hi=200.0)
    holed = build_holed_plate_fe2_micro(Mx=Mx, My=My, N_micro=Nmic_el, seed=2027)
    n_act = int(holed["active_mask"].sum())
    C_true = holed["stress_true"].view(n_act, 3, 3).transpose(1, 2).numpy()
    u_ref, se_ref, tip_ref, _, _ = solve_macro_holed_elasticity(C_true, holed["active_mask"], holed["Lx"], holed["Ly"])
    elastic = {
        "Mx": Mx, "My": My, "N_micro": Nmic_el, "n_active": n_act,
        "micro_solve_s": round(holed["micro_solve_s"], 2),
        "tip_ref": float(tip_ref),
        "models": {},
    }
    for name, is_b, factory, lr in [
        ("BCNO-Elas", True, lambda: BCNO_Elas_2D(steps=4, hidden=32), 3e-3),
        ("ResUNet-Elas", False, lambda: ResUNet_Elas_2D(width=32), 2e-3),
        ("FNO-Elas", False, lambda: FNO_Elas_2D(width=32, modes=16, depth=4), 2e-3),
    ]:
        seed_rows = []
        for sd in seeds:
            print(f"  elastic {name} seed={sd}", flush=True)
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
            u_raw, se_raw, tip_raw, _, _ = solve_macro_holed_elasticity(C_pred, holed["active_mask"], holed["Lx"], holed["Ly"])
            C_proj = spd_project_3x3(C_pred, min_eig=0.5)
            u_proj, se_proj, tip_proj, _, _ = solve_macro_holed_elasticity(C_proj, holed["active_mask"], holed["Lx"], holed["Ly"])
            seed_rows.append({
                "infer_s": round(infer_s, 3),
                "Ceff_ood": round(ceff, 4),
                "U_raw": round(_rel(u_raw, u_ref), 4),
                "U_spd": round(_rel(u_proj, u_ref), 4),
                "energy_spd": round(_rel(se_proj, se_ref), 4),
                "tip_spd": round(abs(float(tip_proj) - float(tip_ref)) / max(abs(float(tip_ref)), 1e-12), 4),
            })
        elastic["models"][name] = agg_seeds(seed_rows) if len(seed_rows) > 1 else seed_rows[0]
        print(f"    {name}:", elastic["models"][name], flush=True)
    out["elastic_fe2"] = elastic
    return out


# ---------- C: 64^3 3-seed ----------
def part_c_64_3seed(root: Path, device: torch.device, smoke: bool) -> dict:
    print("\n=== Part C: 64^3 Synth (+ Ruhr if npz present), 3 seeds ===", flush=True)
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
        # flatten for agg
        id_rows = [r["id"] for r in rows]
        od_rows = [r["ood"] for r in rows]
        out["synth64"][name] = {
            "id": agg_seeds(id_rows) if len(id_rows) > 1 else id_rows[0],
            "ood": agg_seeds(od_rows) if len(od_rows) > 1 else od_rows[0],
        }
        print(f"    {name} OOD Dip~", out["synth64"][name]["ood"], flush=True)

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
                mdl = train_3d_any(ctor().float(), r_tr, ep, 2 if smoke else 3, 3e-3, sd, device, is_bcno=is_b)
                rows.append(eval_3d_dual(mdl, r_te, is_b, device, bs=2))
            out["ruhr64"][name] = agg_seeds(rows) if len(rows) > 1 else rows[0]
            print(f"  Ruhr64 {name}:", out["ruhr64"][name], flush=True)
    else:
        out["ruhr64"] = {"skipped": "upload rve_3d_ruhr64_compact.npz to /content to enable"}
    return out


# ---------- A: Exp.11 3-seed reaffirm ----------
def part_a_exp11(device: torch.device, smoke: bool) -> dict:
    print("\n=== Part A: Exp.11 1000-train reaffirm (3 seeds, key models) ===", flush=True)
    n_tr = 8 if smoke else 500  # 500 geoms * 2 axes ~ 1000 fields
    n_te = 4 if smoke else 100
    tr = build_large_2d_pack(128, n_tr, 1101, 10.0, 80.0, 0.30, 0.58, device)
    idp = build_large_2d_pack(128, n_te, 1102, 10.0, 80.0, 0.30, 0.58, device)
    od2 = build_large_2d_pack(128, n_te, 1104, 100.0, 300.0, 0.40, 0.65, device)
    odh = build_large_2d_pack(128, n_te, 1103, 100.0, 400.0, 0.45, 0.72, device)
    ep = 2 if smoke else 50
    seeds = [0] if smoke else [0, 1, 2]
    out = {}
    for name, factory, is_b, lr in [
        ("BC-NO", lambda: BCNO2D(), True, 3e-3),
        ("MgNO2d", lambda: MgNO2dWrapper(), False, 2e-3),
        ("ResUNet", lambda: ResUNetSmall(), False, 2e-3),
        ("FNO2d", lambda: FNO2dSmall(), False, 2e-3),
    ]:
        rows = []
        for sd in seeds:
            print(f"  Exp11 {name} seed={sd}", flush=True)
            mdl = _train_2d(name, factory, is_b, tr, ep, lr, sd, device)
            rows.append({
                "id": evaluate_pack(mdl, idp, is_b, device, use_dipole=True)["KeffRelErr"],
                "ood_exp2": evaluate_pack(mdl, od2, is_b, device, use_dipole=True)["KeffRelErr"],
                "ood_hard": evaluate_pack(mdl, odh, is_b, device, use_dipole=True)["KeffRelErr"],
            })
        out[name] = agg_seeds(rows) if len(rows) > 1 else rows[0]
        print(f"    {name}:", out[name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="all", help="all|A|B|C|D|AB|CD")
    args = ap.parse_args()
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "smoke:", args.smoke, "only:", args.only, flush=True)
    only = args.only.upper()
    out = {"device": str(device), "smoke": args.smoke}

    if only in ("ALL", "A", "AB"):
        out["part_a_exp11"] = part_a_exp11(device, args.smoke)
    if only in ("ALL", "B", "AB"):
        out["part_b_fe2"] = part_b_fe2(device, args.smoke)
    if only in ("ALL", "C", "CD"):
        out["part_c_64"] = part_c_64_3seed(root, device, args.smoke)
    if only in ("ALL", "D", "CD"):
        out["part_d_hard"] = part_d_expanded_hard(root, device, args.smoke)

    path = root / "bcno_v8_scale_engineering_results.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved", path, flush=True)


if __name__ == "__main__":
    main()
