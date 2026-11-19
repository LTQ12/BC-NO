"""Full CMAME / JCP experimental suite for Boundary-Capacitance Neural Operator (BC-NO).

Runs cleanly on Colab GPU (flat /content directory):
  Exp 1: n=48 hard-topology (wall, center, bridge, Lhold) + zero-shot contrast OOD (10..1e6) [3 seeds]
  Exp 2: N=128 fiber_columns main benchmark (ID + extreme OOD) vs FNO2d, ResUNet2D, UNO2d (+Keff-loss variants) [3 seeds]
  Exp 3: N=128 fiber_columns ablation study (Full vs w/o Green, w/o Cap, w/o Strip, w/o Dipole) [3 seeds]
  Exp 4: N=128 multi-family geometry & topology (wavy_laminate + hard_bridge_N128 w/ 1e6 OOD) [3 seeds]
  Exp 5: Zero-shot cross-resolution generalization (trained on N=128 -> tested on N=64 and N=256) [3 seeds]
  Exp 6: Real 3D-sliced CT microstructures (32x32 multi-phase locked_test + 128x128 high-contrast Ruhr sandstone) [3 seeds]

Outputs:
  bcno_paper_results.json
  bcno_paper_tables.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baselines_2d import (
    CNO2dWrapper,
    FFNO2d,
    FNO2dSmall,
    MgNO2dWrapper,
    ResUNetSmall,
    Transolver2d,
    UNOWrapper,
)
from bcno_model_2d import (
    BCNO2D,
    boundary_strip_mask,
    dipole_keff_columns,
    precompute_floating_components,
)
from colab_data_gen import fast_interior_solve, generate_synthetic_benchmarks, make_hard_topology
from rve_discrete_operator_2d import conservative_flux_columns, harmonic_mean


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = torch.sqrt(torch.sum((pred - target) ** 2, dim=(1, 2)) + 1e-12)
    den = torch.sqrt(torch.sum(target ** 2, dim=(1, 2)) + 1e-12)
    return num / den.clamp_min(1e-8)


def paired_tensor_error(pred_cols: torch.Tensor, target_cols: torch.Tensor) -> float:
    n_pairs = pred_cols.shape[0] // 2
    if n_pairs <= 0:
        return float("nan")
    pred = pred_cols[: 2 * n_pairs].view(n_pairs, 2, 2).transpose(1, 2)
    target = target_cols[: 2 * n_pairs].view(n_pairs, 2, 2).transpose(1, 2)
    num = torch.linalg.vector_norm(pred - target, dim=(1, 2))
    den = torch.linalg.vector_norm(target, dim=(1, 2)).clamp_min(1e-8)
    return float((num / den).mean().item())


def energy_rel_error(pred_u: torch.Tensor, target_u: torch.Tensor, kappa: torch.Tensor) -> float:
    h = 1.0 / max(pred_u.shape[-1] - 1, 1)
    kx = harmonic_mean(kappa[:, :-1, 1:-1], kappa[:, 1:, 1:-1])
    ky = harmonic_mean(kappa[:, 1:-1, :-1], kappa[:, 1:-1, 1:])
    pqx = -kx * (pred_u[:, 1:, 1:-1] - pred_u[:, :-1, 1:-1]) / h
    pqy = -ky * (pred_u[:, 1:-1, 1:] - pred_u[:, 1:-1, :-1]) / h
    tqx = -kx * (target_u[:, 1:, 1:-1] - target_u[:, :-1, 1:-1]) / h
    tqy = -ky * (target_u[:, 1:-1, 1:] - target_u[:, 1:-1, :-1]) / h
    num = torch.sqrt(((pqx - tqx) ** 2 / kx).sum((1, 2)) + ((pqy - tqy) ** 2 / ky).sum((1, 2)) + 1e-12)
    den = torch.sqrt((tqx ** 2 / kx).sum((1, 2)) + (tqy ** 2 / ky).sum((1, 2)) + 1e-12).clamp_min(1e-8)
    return float((num / den).mean().item())


def pack_from_pt(pt_path: Path, split: str, normalize_median: bool = False) -> dict:
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "splits" in d:  # real_ct_32.pt format
        sp = d["splits"][split]
        kappa_raw = sp["kappa"].float()
        v = sp["fluctuation"].float()
        lift = sp["affine_lift"].float()
        k_cols = sp["keff_column"].float()
        load_axis = sp["load_axis"].long()
        B, H, W = kappa_raw.shape
    else:
        a = d[f"a_{split}"].float()
        v = d[f"u_{split}"][..., 0].float()
        lift = d[f"affine_lift_{split}"][..., 0].float()
        k_cols = d[f"k_eff_columns_{split}"].float()
        load_axis = d[f"load_axis_{split}"].long()
        kappa_raw = a[..., 2]
        B, H, W = kappa_raw.shape

    if normalize_median:
        k_ref = kappa_raw.flatten(1).median(dim=1).values.view(B, 1, 1).clamp_min(1e-4)
        kappa = kappa_raw / k_ref
    else:
        kappa = kappa_raw

    axis_1d = torch.linspace(0.0, 1.0, H, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(axis_1d, axis_1d, indexing="ij")
    ortho = torch.stack([y_grid if int(load_axis[i]) == 0 else x_grid for i in range(B)], dim=0)
    idx = torch.arange(H, dtype=torch.float32)
    ii = idx[:, None].expand(H, W)
    jj = idx[None, :].expand(H, W)
    dist = (torch.minimum(torch.minimum(ii, jj), torch.minimum(H - 1 - ii, W - 1 - jj)) / (H / 2.0))[None].expand(B, H, W)
    labels, max_comp = precompute_floating_components(kappa)
    return {
        "kappa_raw": kappa_raw,
        "kappa": kappa,
        "lift": lift,
        "ortho": ortho,
        "dist": dist,
        "u_true": lift + v,
        "v_true": v,
        "k_cols": k_cols,
        "load_axis": load_axis,
        "labels": labels,
        "max_comp": max_comp,
        "x_grid": x_grid[None],
        "y_grid": y_grid[None],
    }


@torch.no_grad()
def evaluate_pack(model, pack: dict, is_bcno: bool, device: torch.device, batch_size: int = 16, use_dipole: bool = True) -> dict[str, float]:
    model.eval()
    B = pack["kappa"].shape[0]
    xg = pack["x_grid"].to(device)
    yg = pack["y_grid"].to(device)
    u_preds = []
    t_inf = 0.0
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        kap = pack["kappa"][s:e].to(device)
        lift = pack["lift"][s:e].to(device)
        ortho = pack["ortho"][s:e].to(device)
        dist = pack["dist"][s:e].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        if is_bcno:
            lbl = pack["labels"][s:e].to(device)
            up, _ = model(kap, lift, ortho, dist, lbl, pack["max_comp"])
        else:
            up = model(kap, lift, ortho, dist)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.time() - t0
        u_preds.append(up.cpu())

    u_pred = torch.cat(u_preds, dim=0)
    v_pred = u_pred - pack["lift"]

    # Compute Keff in float64 using raw physical kappa (handles both standard and normalized median)
    k_dip = dipole_keff_columns(u_pred.double(), pack["kappa_raw"].double(), pack["x_grid"].double(), pack["y_grid"].double()).float()
    k_naive = conservative_flux_columns(u_pred.double(), pack["kappa_raw"].double()).float()
    k_use = k_dip if use_dipole else k_naive
    k_true = pack["k_cols"]

    idx = torch.arange(B)
    kt_d = k_true[idx, pack["load_axis"]]
    ku_d = k_use[idx, pack["load_axis"]]
    kn_d = k_naive[idx, pack["load_axis"]]

    diag_rel = float(((ku_d - kt_d).abs() / kt_d.abs().clamp_min(1e-4)).mean())
    diag_naive = float(((kn_d - kt_d).abs() / kt_d.abs().clamp_min(1e-4)).mean())
    pair_rel = paired_tensor_error(k_use, k_true)
    pair_naive = paired_tensor_error(k_naive, k_true)

    # Filter non-trivial fluctuation samples for relative L2 (keep all for Temp L2)
    v_norm = torch.sqrt((pack["v_true"] ** 2).sum((1, 2)))
    valid_v = v_norm > 1e-4
    fluct_l2 = float(relative_l2(v_pred[valid_v], pack["v_true"][valid_v]).mean()) if bool(valid_v.any()) else 0.0
    temp_l2 = float(relative_l2(u_pred, pack["u_true"]).mean())
    e_rel = energy_rel_error(u_pred.double(), pack["u_true"].double(), pack["kappa_raw"].double())

    return {
        "KeffRelErr": pair_rel,
        "KeffRelErr_Naive": pair_naive,
        "DiagRelErr": diag_rel,
        "DiagRelErr_Naive": diag_naive,
        "Fluctuation_L2": fluct_l2,
        "Temperature_L2": temp_l2,
        "EnergyRelErr": e_rel,
        "InferMsPerSample": 1000.0 * t_inf / max(B, 1),
    }


def energy_loss_term(pred_u: torch.Tensor, target_u: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Variational energy inner-product loss int kappa |grad(u_pred - u_true)|^2 (CGiNS / Haltmayer 2026)."""
    kx = harmonic_mean(kappa[:, :-1, 1:-1], kappa[:, 1:, 1:-1])
    ky = harmonic_mean(kappa[:, 1:-1, :-1], kappa[:, 1:-1, 1:])
    dp_x = (pred_u[:, 1:, 1:-1] - pred_u[:, :-1, 1:-1]) - (target_u[:, 1:, 1:-1] - target_u[:, :-1, 1:-1])
    dp_y = (pred_u[:, 1:-1, 1:] - pred_u[:, 1:-1, :-1]) - (target_u[:, 1:-1, 1:] - target_u[:, 1:-1, :-1])
    dt_x = target_u[:, 1:, 1:-1] - target_u[:, :-1, 1:-1]
    dt_y = target_u[:, 1:-1, 1:] - target_u[:, 1:-1, :-1]
    num = (kx * (dp_x ** 2)).sum((1, 2)) + (ky * (dp_y ** 2)).sum((1, 2))
    den = (kx * (dt_x ** 2)).sum((1, 2)) + (ky * (dt_y ** 2)).sum((1, 2)) + 1e-6
    return torch.sqrt(num / den + 1e-8).mean()


def train_model(
    model: torch.nn.Module,
    is_bcno: bool,
    train_pack: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    use_keff_loss: bool,
    use_sigma_loss: bool,
    use_energy_loss: bool = False,
) -> float:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    B = train_pack["kappa"].shape[0]
    H = train_pack["kappa"].shape[-1]
    h_grid = 1.0 / max(H - 1, 1)
    xg = train_pack["x_grid"].to(device)
    yg = train_pack["y_grid"].to(device)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(B)
        for s in range(0, B, batch_size):
            idx = perm[s : min(s + batch_size, B)]
            kap = train_pack["kappa"][idx].to(device)
            kap_raw = train_pack["kappa_raw"][idx].to(device)
            lift = train_pack["lift"][idx].to(device)
            ortho = train_pack["ortho"][idx].to(device)
            dist = train_pack["dist"][idx].to(device)
            vt = train_pack["v_true"][idx].to(device)
            kt = train_pack["k_cols"][idx].to(device)
            ax = train_pack["load_axis"][idx].to(device)

            if is_bcno:
                lbl = train_pack["labels"][idx].to(device)
                up, sig = model(kap, lift, ortho, dist, lbl, train_pack["max_comp"])
            else:
                up = model(kap, lift, ortho, dist)

            vp = up - lift
            num_v = torch.sqrt(((vp - vt) ** 2).sum(dim=(1, 2)) + 1e-8)
            den_v = torch.sqrt((vt ** 2).sum(dim=(1, 2)) + 1e-6).clamp_min(1e-2)
            loss = (num_v / den_v).mean()

            if use_keff_loss:
                kp = dipole_keff_columns(up, kap_raw, xg, yg)
                b_idx = torch.arange(idx.shape[0], device=device)
                kt_d = kt[b_idx, ax]
                kp_d = kp[b_idx, ax]
                loss = loss + 0.25 * ((kp_d - kt_d).abs() / kt_d.abs().clamp_min(0.1)).mean()

            if use_energy_loss:
                loss = loss + 0.5 * energy_loss_term(up, lift + vt, kap_raw)

            if is_bcno and use_sigma_loss:
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
    return time.time() - t0


def agg_seeds(seed_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = seed_dicts[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in seed_dicts if not math.isnan(d[k])]
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan")}
        else:
            out[k] = {
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Quick 2-epoch 1-seed verification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running BC-NO Paper Suite on device={device} (smoke={args.smoke})", flush=True)
    seeds = [0] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]
    ep_main = 2 if args.smoke else args.epochs
    ep_sec = 2 if args.smoke else max(40, int(0.75 * args.epochs))

    root = Path(__file__).resolve().parent
    generate_synthetic_benchmarks(root)

    results: dict[str, object] = {"device": str(device), "seeds": seeds, "epochs_main": ep_main}

    # Benchmark sparse LU solver time on N=128 once
    t_lu = time.time()
    _kap_sample = make_hard_topology(128, 1, 1000.0, "u_bridge")
    fast_interior_solve(_kap_sample, 0)
    results["sparse_lu_n128_ms"] = round((time.time() - t_lu) * 1000.0, 2)
    print(f"Sparse LU N=128 reference time: {results['sparse_lu_n128_ms']} ms/sample", flush=True)

    # ---- Exp 2 & Exp 5: N=128 fiber_columns + Cross-resolution (N=64, N=256) ----
    print("\n=== Exp 2 & 5: N=128 fiber_columns + Cross-Resolution (64, 256) ===", flush=True)
    f_tr = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "train")
    f_id = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "test")
    f_ood = pack_from_pt(root / "rve_fiber_columns_ood_N128_f64.pt", "test")
    f_n64 = pack_from_pt(root / "fiber_N64_eval.pt", "test")
    f_n256 = pack_from_pt(root / "fiber_N256_eval.pt", "test")

    exp2_specs = [
        ("BC-NO", True, True, True, False, lambda: BCNO2D(steps=4, hidden=32)),
        ("MgNO2d", False, False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("MgNO2d+Energy", False, True, False, True, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("Transolver2d", False, False, False, False, lambda: Transolver2d(5, dim=64, depth=4, heads=4, n_slices=32)),
        ("CNO2d", False, False, False, False, lambda: CNO2dWrapper(5, width=32)),
        ("FFNO2d", False, False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("UNO2d", False, False, False, False, lambda: UNOWrapper(width=32, modes=12)),
        ("FNO2d", False, False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
        ("ResUNet", False, False, False, False, lambda: ResUNetSmall(5, width=32)),
        ("ResUNet+Energy", False, True, False, True, lambda: ResUNetSmall(5, width=32)),
    ]
    exp2_res = {}
    exp5_res = {}
    for name, is_bcno, use_k, use_sig, use_en, ctor in exp2_specs:
        id_s, ood_s, n64_s, n256_s = [], [], [], []
        n_params = 0
        train_times = []
        for sd in seeds:
            torch.manual_seed(sd)
            mdl = ctor().float()
            n_params = sum(p.numel() for p in mdl.parameters())
            dt = train_model(mdl, is_bcno, f_tr, epochs=ep_main, batch_size=args.batch_size, lr=(3e-3 if is_bcno else 2e-3), seed=sd, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig, use_energy_loss=use_en)
            train_times.append(dt)
            id_s.append(evaluate_pack(mdl, f_id, is_bcno, device))
            ood_s.append(evaluate_pack(mdl, f_ood, is_bcno, device))
            # Cross-resolution N=64 and N=256 (all baselines support arbitrary resolution except Transolver runs on N=256 w/ batch 4)
            n64_s.append(evaluate_pack(mdl, f_n64, is_bcno, device))
            n256_s.append(evaluate_pack(mdl, f_n256, is_bcno, device, batch_size=4))
        exp2_res[name] = {"params": n_params, "train_s": round(statistics.mean(train_times), 1), "id": agg_seeds(id_s), "ood": agg_seeds(ood_s)}
        exp5_res[name] = {"N64": agg_seeds(n64_s), "N256": agg_seeds(n256_s)}
        print(
            f"  [{name:12s}] params={n_params:7d} | "
            f"ID Keff={exp2_res[name]['id']['KeffRelErr']['mean']:.3f} Fluct={exp2_res[name]['id']['Fluctuation_L2']['mean']:.3f} | "
            f"OOD Keff={exp2_res[name]['ood']['KeffRelErr']['mean']:.3f} Naive={exp2_res[name]['ood']['KeffRelErr_Naive']['mean']:.3f} Fluct={exp2_res[name]['ood']['Fluctuation_L2']['mean']:.3f} | "
            f"N256 Keff={exp5_res[name]['N256']['KeffRelErr']['mean']:.3f}",
            flush=True,
        )
    results["exp2_fiber_n128"] = exp2_res
    results["exp5_cross_res"] = exp5_res

    # ---- Exp 3: Ablation Study on N=128 fiber_columns ----
    print("\n=== Exp 3: Ablation Study (N=128 fiber_columns) ===", flush=True)
    ab_specs = [
        ("BC-NO (Full)", True, True, True, True, True),
        ("w/o DST Green", False, True, False, False, False),
        ("w/o Cap Pooling", True, False, True, True, True),
        ("w/o Strip Mask", True, True, False, True, True),
        ("w/o Dipole Readout", True, True, True, True, False),
    ]
    exp3_res = {}
    for name, ug, uc, us, up_skip, use_dip in ab_specs:
        id_s, ood_s = [], []
        for sd in seeds:
            torch.manual_seed(sd)
            mdl = BCNO2D(steps=4, hidden=32, use_green=ug, use_capacitance=uc, use_strip_mask=us, use_physics_skip=up_skip).float()
            train_model(mdl, True, f_tr, epochs=ep_sec, batch_size=args.batch_size, lr=3e-3, seed=sd, device=device, use_keff_loss=True, use_sigma_loss=ug)
            id_s.append(evaluate_pack(mdl, f_id, True, device, use_dipole=use_dip))
            ood_s.append(evaluate_pack(mdl, f_ood, True, device, use_dipole=use_dip))
        exp3_res[name] = {"id": agg_seeds(id_s), "ood": agg_seeds(ood_s)}
        print(
            f"  [{name:18s}] ID Keff={exp3_res[name]['id']['KeffRelErr']['mean']:.3f} Fluct={exp3_res[name]['id']['Fluctuation_L2']['mean']:.3f} | "
            f"OOD Keff={exp3_res[name]['ood']['KeffRelErr']['mean']:.3f} Fluct={exp3_res[name]['ood']['Fluctuation_L2']['mean']:.3f}",
            flush=True,
        )
    results["exp3_ablation"] = exp3_res

    # ---- Exp 4: Multi-Family (wavy_laminate + hard_bridge_N128 w/ 1e6 OOD) ----
    print("\n=== Exp 4: Multi-Family N=128 (wavy_laminate + hard_bridge 1e6 OOD) ===", flush=True)
    exp4_res = {}
    for fam_name, pt_file in [("wavy_laminate", "wavy_laminate_N128.pt"), ("hard_bridge_1e6", "hard_bridge_N128.pt")]:
        tr_p = pack_from_pt(root / pt_file, "train")
        id_p = pack_from_pt(root / pt_file, "test")
        od_p = pack_from_pt(root / pt_file, "ood")
        fam_out = {}
        for name, is_bcno, use_k, use_sig, ctor in [
            ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
            ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
            ("Transolver2d", False, False, False, lambda: Transolver2d(5, dim=64, depth=4, heads=4, n_slices=32)),
            ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
            ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
            ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ]:
            id_s, od_s = [], []
            for sd in seeds:
                torch.manual_seed(sd)
                mdl = ctor().float()
                train_model(mdl, is_bcno, tr_p, epochs=ep_sec, batch_size=args.batch_size, lr=3e-3, seed=sd, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
                id_s.append(evaluate_pack(mdl, id_p, is_bcno, device))
                od_s.append(evaluate_pack(mdl, od_p, is_bcno, device))
            fam_out[name] = {"id": agg_seeds(id_s), "ood": agg_seeds(od_s)}
            print(
                f"  [{fam_name:15s} | {name:7s}] ID Keff={fam_out[name]['id']['DiagRelErr']['mean']:.3f} | "
                f"OOD Dip={fam_out[name]['ood']['DiagRelErr']['mean']:.3f} Naive={fam_out[name]['ood']['DiagRelErr_Naive']['mean']:.3f} Fluct={fam_out[name]['ood']['Fluctuation_L2']['mean']:.3f}",
                flush=True,
            )
        exp4_res[fam_name] = fam_out
    results["exp4_multifamily"] = exp4_res

    # ---- Exp 6: Real CT (32x32 multi-phase locked_test + 128x128 high-contrast Ruhr sandstone) ----
    print("\n=== Exp 6: Real CT Microstructures ===", flush=True)
    exp6_res = {}
    for ct_name, pt_file, sp_test, norm_med, modes_f in [
        ("real_ct_32", "real_ct_32.pt", "locked_test", True, 10),
        ("ruhr_sandstone_N128_hc", "real_ct_ruhr_N128_hc.pt", "test", False, 16),
    ]:
        tr_p = pack_from_pt(root / pt_file, "train", normalize_median=norm_med)
        te_p = pack_from_pt(root / pt_file, sp_test, normalize_median=norm_med)
        ct_out = {}
        for name, is_bcno, use_k, use_sig, ctor in [
            ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
            ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
            ("Transolver2d", False, False, False, lambda: Transolver2d(5, dim=64, depth=4, heads=4, n_slices=32)),
            ("FFNO2d", False, False, False, lambda mf=modes_f: FFNO2d(5, width=48, modes=mf, depth=6)),
            ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
            ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ]:
            te_s = []
            for sd in seeds:
                torch.manual_seed(sd)
                mdl = ctor().float()
                train_model(mdl, is_bcno, tr_p, epochs=ep_sec, batch_size=args.batch_size, lr=3e-3, seed=sd, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
                te_s.append(evaluate_pack(mdl, te_p, is_bcno, device, use_dipole=not norm_med))
            ct_out[name] = agg_seeds(te_s)
            print(
                f"  [{ct_name:22s} | {name:7s}] DiagKeff={ct_out[name]['DiagRelErr']['mean']:.3f} TempL2={ct_out[name]['Temperature_L2']['mean']:.4f} FluctL2={ct_out[name]['Fluctuation_L2']['mean']:.3f}",
                flush=True,
            )
        exp6_res[ct_name] = ct_out
    results["exp6_real_ct"] = exp6_res

    (root / "bcno_paper_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved: bcno_paper_results.json", flush=True)

    # Write clean Markdown summary tables for direct paper insertion
    lines = ["# BC-NO Paper Suite Results\n", f"Device: `{device}` | Seeds: `{seeds}` | Main epochs: `{ep_main}` | Sparse LU N=128: `{results['sparse_lu_n128_ms']} ms`\n"]
    lines.append("\n## Exp 2: N=128 fiber_columns (mean +- std)\n")
    lines.append("| Model | Params | ID KeffRelErr | ID FluctL2 | OOD KeffRelErr | OOD NaiveKeff | OOD FluctL2 | OOD TempL2 | Infer ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mname, d in exp2_res.items():
        def fmt(dd, k): return f"{dd[k]['mean']:.3f}+-{dd[k]['std']:.3f}"
        lines.append(f"| {mname} | {d['params']} | {fmt(d['id'],'KeffRelErr')} | {fmt(d['id'],'Fluctuation_L2')} | {fmt(d['ood'],'KeffRelErr')} | {fmt(d['ood'],'KeffRelErr_Naive')} | {fmt(d['ood'],'Fluctuation_L2')} | {fmt(d['ood'],'Temperature_L2')} | {d['ood']['InferMsPerSample']['mean']:.2f} |")
    lines.append("\n## Exp 3: Ablation Study (N=128 fiber_columns)\n")
    lines.append("| Variant | ID KeffRelErr | ID FluctL2 | OOD KeffRelErr | OOD FluctL2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mname, d in exp3_res.items():
        def fmt(dd, k): return f"{dd[k]['mean']:.3f}+-{dd[k]['std']:.3f}"
        lines.append(f"| {mname} | {fmt(d['id'],'KeffRelErr')} | {fmt(d['id'],'Fluctuation_L2')} | {fmt(d['ood'],'KeffRelErr')} | {fmt(d['ood'],'Fluctuation_L2')} |")
    lines.append("\n## Exp 4: Multi-Family N=128\n")
    lines.append("| Family | Model | ID DiagKeff | OOD DiagKeff (Dipole) | OOD DiagKeff (Naive) | OOD FluctL2 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for fam, fd in exp4_res.items():
        for mname, d in fd.items():
            lines.append(f"| {fam} | {mname} | {d['id']['DiagRelErr']['mean']:.3f} | {d['ood']['DiagRelErr']['mean']:.3f} | {d['ood']['DiagRelErr_Naive']['mean']:.3f} | {d['ood']['Fluctuation_L2']['mean']:.3f} |")
    lines.append("\n## Exp 5: Cross-Resolution Zero-Shot (trained on N=128)\n")
    lines.append("| Model | N=64 Keff | N=64 FluctL2 | N=256 Keff | N=256 FluctL2 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mname, d in exp5_res.items():
        lines.append(f"| {mname} | {d['N64']['KeffRelErr']['mean']:.3f} | {d['N64']['Fluctuation_L2']['mean']:.3f} | {d['N256']['KeffRelErr']['mean']:.3f} | {d['N256']['Fluctuation_L2']['mean']:.3f} |")
    lines.append("\n## Exp 6: Real CT Microstructures\n")
    lines.append("| Dataset | Model | DiagKeff | TempL2 | FluctL2 |")
    lines.append("|---|---|---:|---:|---:|")
    for ct, cd in exp6_res.items():
        for mname, d in cd.items():
            lines.append(f"| {ct} | {mname} | {d['DiagRelErr']['mean']:.3f} | {d['Temperature_L2']['mean']:.4f} | {d['Fluctuation_L2']['mean']:.3f} |")
    (root / "bcno_paper_tables.md").write_text("\n".join(lines))
    print("Saved: bcno_paper_tables.md", flush=True)


if __name__ == "__main__":
    main()
