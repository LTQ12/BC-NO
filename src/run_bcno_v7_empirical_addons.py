"""CMAME v7 empirical add-ons (flat /content only; NO large .pt upload).

1) Empirical mechanism figure from real U-bridge solves (replaces cartoon Scheme-1)
2) Matched-accuracy early-stop: Willot CG-FFT (PBC64) & Jacobi-PCG (32^3)
3) Energy/variational readout vs Naive/Dipole on on-the-fly fiber packs
4) Training curves (loss + val dipole Keff) for BC-NO / FNO / ResUNet / MgNO

Fiber/PBC/3D data are generated in /content via LU/PCG — do not upload .pt files.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines_2d import FNO2dSmall, MgNO2dWrapper, ResUNetSmall
from bcno_model_2d import BCNO2D, dipole_keff_columns
from bcno_model_3d import BCNO3D, jacobi_pcg_3d
from make_fig_empirical_mechanism import main as make_empirical_fig
from rve_discrete_operator_2d import conservative_flux_columns, energy_inner_product
from rve_discrete_operator_3d import conservative_flux_columns_3d, face_conductivities_3d
from run_bcno_paper_suite import evaluate_pack, paired_tensor_error, relative_l2, train_model
from run_bcno_v2_addons import build_periodic_2d_pack, make_3d_synthetic_pack
from run_bcno_v5_rigor_addons import build_fiber_pack_custom
from run_bcno_v6_audit_fixes import willot_cgfft_2d


def evaluate_three_readouts(model, pack: dict, is_bcno: bool, device: torch.device) -> dict:
    """Naive / Dipole / Energy readouts on the same predicted fields."""
    model.eval()
    B = pack["kappa"].shape[0]
    preds = []
    with torch.no_grad():
        for s in range(0, B, 8):
            e = min(s + 8, B)
            kap = pack["kappa"][s:e].to(device)
            lift = pack["lift"][s:e].to(device)
            ortho = pack["ortho"][s:e].to(device)
            dist = pack["dist"][s:e].to(device)
            if is_bcno:
                up, _ = model(kap, lift, ortho, dist, pack["labels"][s:e].to(device), pack["max_comp"])
            else:
                up = model(kap, lift, ortho, dist)
            preds.append(up.cpu())
    u = torch.cat(preds, 0)
    kap = pack["kappa_raw"]
    k_nv = conservative_flux_columns(u.double(), kap.double()).float()
    k_dp = dipole_keff_columns(u.double(), kap.double(), pack["x_grid"].double(), pack["y_grid"].double()).float()
    e = energy_inner_product(u.double(), u.double(), kap.double(), kappa_channel=0).float()
    k_en = k_nv.clone()
    la = pack["load_axis"]
    idx = torch.arange(B)
    k_en[idx, la] = e
    fl = float(relative_l2(u - pack["lift"], pack["v_true"]).mean())
    return {
        "Fluct_L2": round(fl, 4),
        "Naive": round(paired_tensor_error(k_nv, pack["k_cols"]), 4),
        "Dipole": round(paired_tensor_error(k_dp, pack["k_cols"]), 4),
        "Energy": round(paired_tensor_error(k_en, pack["k_cols"]), 4),
    }


def true_field_three_readouts(pack: dict) -> dict:
    u = pack["u_true"]
    kap = pack["kappa_raw"]
    B = u.shape[0]
    k_nv = conservative_flux_columns(u.double(), kap.double()).float()
    k_dp = dipole_keff_columns(u.double(), kap.double(), pack["x_grid"].double(), pack["y_grid"].double()).float()
    e = energy_inner_product(u.double(), u.double(), kap.double(), kappa_channel=0).float()
    k_en = k_nv.clone()
    la = pack["load_axis"]
    idx = torch.arange(B)
    k_en[idx, la] = e
    return {
        "Naive": round(paired_tensor_error(k_nv, pack["k_cols"]), 6),
        "Dipole": round(paired_tensor_error(k_dp, pack["k_cols"]), 6),
        "Energy": round(paired_tensor_error(k_en, pack["k_cols"]), 6),
        "note": "On exact fields, Naive/Dipole/Energy should all be ~0 (Energy uses loaded diagonal only)",
    }


def willot_matched(kap: torch.Tensor, k_ref_22: torch.Tensor, thresholds=(0.30, 0.10, 0.05, 0.01), max_iter=400) -> dict:
    B, N, _ = kap.shape
    dev = kap.device
    k64 = kap.double()
    wx = 2.0 * math.pi * torch.fft.fftfreq(N, d=1.0, device=dev, dtype=torch.float64)
    wy = 2.0 * math.pi * torch.fft.fftfreq(N, d=1.0, device=dev, dtype=torch.float64)
    kx, ky = torch.meshgrid(wx, wy, indexing="ij")
    ex = torch.exp(1j * kx) - 1.0
    ey = torch.exp(1j * ky) - 1.0
    px = 0.5 * ex * (1.0 + torch.exp(1j * ky))
    py = 0.5 * ey * (1.0 + torch.exp(1j * kx))
    k2 = px.abs() ** 2 + py.abs() ** 2
    inv = torch.where(k2 > 1e-12, 1.0 / k2.clamp_min(1e-12), torch.zeros_like(k2))
    Gxx, Gxy = (px * px.conj()) * inv, (px * py.conj()) * inv
    Gyx, Gyy = (py * px.conj()) * inv, (py * py.conj()) * inv

    def apply_G(jx, jy):
        Jx, Jy = torch.fft.fftn(jx, dim=(-2, -1)), torch.fft.fftn(jy, dim=(-2, -1))
        return (
            torch.fft.ifftn(Gxx * Jx + Gxy * Jy, dim=(-2, -1)).real,
            torch.fft.ifftn(Gyx * Jx + Gyy * Jy, dim=(-2, -1)).real,
        )

    if dev.type == "cuda":
        willot_cgfft_2d(kap[:1], tol=1e-2, max_iter=5)
        torch.cuda.synchronize()

    states = []
    for ax_id in (0, 1):
        E0x = torch.ones(B, N, N, dtype=torch.float64, device=dev) if ax_id == 0 else torch.zeros(B, N, N, dtype=torch.float64, device=dev)
        E0y = torch.zeros(B, N, N, dtype=torch.float64, device=dev) if ax_id == 0 else torch.ones(B, N, N, dtype=torch.float64, device=dev)
        bx, by = apply_G(-k64 * E0x, -k64 * E0y)
        states.append({"E0x": E0x, "E0y": E0y, "ex": torch.zeros_like(E0x), "ey": torch.zeros_like(E0y),
                       "rx": bx.clone(), "ry": by.clone(), "px": bx.clone(), "py": by.clone(),
                       "rr": (bx * bx + by * by).sum((1, 2))})

    hits = {th: None for th in thresholds}
    rem = set(thresholds)
    hist = []
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for it in range(1, max_iter + 1):
        for st in states:
            Apx, Apy = apply_G(k64 * st["px"], k64 * st["py"])
            pAp = (st["px"] * Apx + st["py"] * Apy).sum((1, 2)).clamp_min(1e-30)
            a = (st["rr"] / pAp).view(B, 1, 1)
            st["ex"] = st["ex"] + a * st["px"]
            st["ey"] = st["ey"] + a * st["py"]
            st["rx"] = st["rx"] - a * Apx
            st["ry"] = st["ry"] - a * Apy
            rr_n = (st["rx"] * st["rx"] + st["ry"] * st["ry"]).sum((1, 2))
            b = (rr_n / st["rr"].clamp_min(1e-30)).view(B, 1, 1)
            st["px"] = st["rx"] + b * st["px"]
            st["py"] = st["ry"] + b * st["py"]
            st["rr"] = rr_n
        K = torch.zeros(B, 2, 2, dtype=torch.float64, device=dev)
        for ax_id, st in enumerate(states):
            K[:, 0, ax_id] = (k64 * (st["E0x"] + st["ex"])).mean((1, 2))
            K[:, 1, ax_id] = (k64 * (st["E0y"] + st["ey"])).mean((1, 2))
        err = float((torch.linalg.matrix_norm(K.float() - k_ref_22) / torch.linalg.matrix_norm(k_ref_22).clamp_min(1e-8)).mean())
        ms = (time.time() - t0) * 1000.0 / max(B, 1)
        hist.append({"iter": it, "keff_err": round(err, 5), "ms": round(ms, 3)})
        for th in list(rem):
            if err <= th:
                hits[th] = {"iters": it, "ms": round(ms, 3), "keff_err": round(err, 5)}
                rem.remove(th)
        if not rem:
            break
    return {
        "thresholds": {str(k): v for k, v in hits.items()},
        "history_head": hist[:6],
        "history_tail": hist[-3:],
        "note": "Error vs periodic FVM reference in this pack; residual discretization gap may floor >0",
    }


def pcg3d_matched(kappa, lift, k_ref_cols, load_axis, thresholds=(0.30, 0.10, 0.05, 0.01), max_iters=2000) -> dict:
    if kappa.device.type == "cuda":
        jacobi_pcg_3d(kappa[:1], lift[:1], tol=1e-2, max_iters=5)
        torch.cuda.synchronize()
    h = 1.0 / max(kappa.shape[-1] - 1, 1)
    kx, ky, kz = face_conductivities_3d(kappa, kappa_channel=0)
    kxm, kxp = kx[:, :-1, 1:-1, 1:-1] / (h * h), kx[:, 1:, 1:-1, 1:-1] / (h * h)
    kym, kyp = ky[:, 1:-1, :-1, 1:-1] / (h * h), ky[:, 1:-1, 1:, 1:-1] / (h * h)
    kzm, kzp = kz[:, 1:-1, 1:-1, :-1] / (h * h), kz[:, 1:-1, 1:-1, 1:] / (h * h)
    diag = (kxm + kxp + kym + kyp + kzm + kzp).clamp_min(1e-12)

    def apply_A(v_in):
        vp = F.pad(v_in, (1, 1, 1, 1, 1, 1))
        return (
            diag * v_in
            - kxm * vp[:, :-2, 1:-1, 1:-1] - kxp * vp[:, 2:, 1:-1, 1:-1]
            - kym * vp[:, 1:-1, :-2, 1:-1] - kyp * vp[:, 1:-1, 2:, 1:-1]
            - kzm * vp[:, 1:-1, 1:-1, :-2] - kzp * vp[:, 1:-1, 1:-1, 2:]
        )

    b = -(
        diag * lift[:, 1:-1, 1:-1, 1:-1]
        - kxm * lift[:, :-2, 1:-1, 1:-1] - kxp * lift[:, 2:, 1:-1, 1:-1]
        - kym * lift[:, 1:-1, :-2, 1:-1] - kyp * lift[:, 1:-1, 2:, 1:-1]
        - kzm * lift[:, 1:-1, 1:-1, :-2] - kzp * lift[:, 1:-1, 1:-1, 2:]
    )
    v = torch.zeros_like(b)
    r = b.clone()
    z = r / diag
    p = z.clone()
    rz = float((r * z).sum())
    hits = {th: None for th in thresholds}
    rem = set(thresholds)
    hist = []
    if kappa.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for it in range(1, max_iters + 1):
        Ap = apply_A(p)
        alpha = rz / max(float((p * Ap).sum()), 1e-30)
        v = v + alpha * p
        r = r - alpha * Ap
        z = r / diag
        rz_n = float((r * z).sum())
        p = z + (rz_n / max(rz, 1e-30)) * p
        rz = rz_n
        if it % 2 == 0 or it <= 8:
            out = lift.clone()
            out[:, 1:-1, 1:-1, 1:-1] += v
            kp = conservative_flux_columns_3d(out, kappa, kappa_channel=0)
            idx = torch.arange(out.shape[0], device=kappa.device)
            la = load_axis.to(kappa.device)
            err = float(((kp[idx, la] - k_ref_cols.to(kappa.device)[idx, la]).abs() / k_ref_cols.to(kappa.device)[idx, la].abs().clamp_min(1e-4)).mean())
            ms = (time.time() - t0) * 1000.0 / max(out.shape[0], 1)
            hist.append({"iter": it, "keff_err": round(err, 5), "ms": round(ms, 3)})
            for th in list(rem):
                if err <= th:
                    hits[th] = {"iters": it, "ms": round(ms, 3), "keff_err": round(err, 5)}
                    rem.remove(th)
            if not rem:
                break
    return {"thresholds": {str(k): v for k, v in hits.items()}, "history_head": hist[:6], "history_tail": hist[-3:]}


def train_curves(model, is_bcno, tr, va, epochs, bs, lr, seed, device) -> dict:
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = tr["kappa"].shape[0]
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(B)
        losses = []
        for s in range(0, B, bs):
            idx = perm[s : s + bs]
            kap = tr["kappa"][idx].to(device)
            lift = tr["lift"][idx].to(device)
            ortho = tr["ortho"][idx].to(device)
            dist = tr["dist"][idx].to(device)
            vt = tr["v_true"][idx].to(device)
            if is_bcno:
                up, _ = model(kap, lift, ortho, dist, tr["labels"][idx].to(device), tr["max_comp"])
            else:
                up = model(kap, lift, ortho, dist)
            vp = up - lift
            loss = (torch.sqrt(((vp - vt) ** 2).sum((1, 2)) + 1e-8) / torch.sqrt((vt ** 2).sum((1, 2)) + 1e-6).clamp_min(1e-2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sch.step()
        va_m = evaluate_pack(model, va, is_bcno, device, batch_size=8, use_dipole=True)
        row = {"epoch": ep, "train_relL2": round(float(np.mean(losses)), 5),
               "val_dip": round(va_m["KeffRelErr"], 4), "val_fluct": round(va_m["Fluctuation_L2"], 4)}
        hist.append(row)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"    ep {ep}/{epochs} loss={row['train_relL2']:.4f} val_dip={row['val_dip']:.4f}", flush=True)
    return {"history": hist, "final": hist[-1]}


def plot_curves(curves: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5))
    cols = {"BC-NO": "#4c72b0", "FNO2d": "#dd8452", "ResUNet": "#c44e52", "MgNO2d": "#55a868"}
    for name, col in cols.items():
        if name not in curves:
            continue
        h = curves[name]["history"]
        axes[0].plot([r["epoch"] for r in h], [r["train_relL2"] for r in h], color=col, lw=1.8, label=name)
        axes[1].plot([r["epoch"] for r in h], [r["val_dip"] for r in h], color=col, lw=1.8, label=name)
    axes[0].set_title("(a) Train field rel-L2"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("(b) Val dipole $K_{eff}$ error"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Rel. error")
    for ax in axes:
        ax.grid(True, ls=":", alpha=0.45); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_training_curves_fiber128.{ext}")
    plt.close(fig)


def plot_matched(matched: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5))
    ths = [0.30, 0.10, 0.05, 0.01]
    for ax, key, title, color in [
        (axes[0], "willot_pbc64", "(a) Willot CG-FFT early-stop", "#4c72b0"),
        (axes[1], "jacobi_pcg_32", "(b) Jacobi-PCG early-stop", "#c44e52"),
    ]:
        block = matched.get(key, {})
        ms, its = [], []
        for th in ths:
            h = (block.get("thresholds") or {}).get(str(th))
            ms.append(None if h is None else h.get("ms"))
            its.append(None if h is None else h.get("iters"))
        xs = [100 * t for t in ths]
        ax.plot(xs, [m if m is not None else np.nan for m in ms], "o-", color=color, lw=2)
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xlabel("Matched Keff error (%)"); ax.set_ylabel("GPU ms / sample"); ax.set_title(title)
        for x, y, it in zip(xs, ms, its):
            if y is not None:
                ax.annotate(f"{it} it", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.grid(True, ls=":", alpha=0.45)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_matched_precision_timing.{ext}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content")
    ap.add_argument("--epochs-curves", type=int, default=60)
    ap.add_argument("--skip-curves", action="store_true")
    ap.add_argument("--skip-3d", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"device": str(device)}
    print("Device:", device, flush=True)

    print("\n=== Part 1: Empirical mechanism figure ===", flush=True)
    out["empirical_mechanism"] = make_empirical_fig(root)

    print("\n=== Part 2: Matched-accuracy early-stop solvers ===", flush=True)
    pbc = build_periodic_2d_pack(64, 8, 0, 10.0, 80.0)
    # Interleaved axis pairs -> 2x2 tensors; take first 4 geometries
    n_pairs = 4
    kap = pbc["kappa"][0 : 2 * n_pairs : 2].to(device)
    k_ref = []
    for i in range(n_pairs):
        c0 = pbc["k_cols"][2 * i]
        c1 = pbc["k_cols"][2 * i + 1]
        k_ref.append(torch.stack([c0, c1], dim=1))
    k_ref = torch.stack(k_ref, 0).to(device)
    out["matched_precision"] = {"willot_pbc64": willot_matched(kap, k_ref)}

    if not args.skip_3d:
        s3 = make_3d_synthetic_pack(32, 4, 0, 10.0, 80.0)
        # pack fields
        kap3 = s3["kappa"][:3].to(device)
        lift3 = s3["lift"][:3].to(device)
        out["matched_precision"]["jacobi_pcg_32"] = pcg3d_matched(
            kap3, lift3, s3["k_cols"][:3], s3["load_axis"][:3]
        )
        # BCNO-3D reference forward time
        m3 = BCNO3D().to(device).eval()
        with torch.no_grad():
            for _ in range(3):
                _ = m3(
                    s3["kappa"][:1].to(device), s3["lift"][:1].to(device),
                    s3["o1"][:1].to(device), s3["o2"][:1].to(device), s3["dist"][:1].to(device),
                    s3["labels"][:1].to(device), int(s3["max_comp"]),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                _ = m3(
                    s3["kappa"][:1].to(device), s3["lift"][:1].to(device),
                    s3["o1"][:1].to(device), s3["o2"][:1].to(device), s3["dist"][:1].to(device),
                    s3["labels"][:1].to(device), int(s3["max_comp"]),
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            out["matched_precision"]["bcno3d_ms_per_forward"] = round((time.time() - t0) * 100.0, 3)
    plot_matched(out["matched_precision"], root)
    print("  willot hits:", out["matched_precision"]["willot_pbc64"]["thresholds"], flush=True)

    print("\n=== Part 3: Energy readout baseline (on-the-fly fiber packs, no .pt upload) ===", flush=True)
    # Small regenerated packs: enough to compare Naive/Dipole/Energy; not the paper's 128-train ledger
    print("  Building fiber packs in /content (CPU LU)...", flush=True)
    tr = build_fiber_pack_custom(128, 24, seed=701, c_lo=10.0, c_hi=80.0)
    te = build_fiber_pack_custom(128, 8, seed=702, c_lo=10.0, c_hi=80.0)
    od = build_fiber_pack_custom(128, 8, seed=703, c_lo=100.0, c_hi=300.0)
    energy = {
        "data_note": "On-the-fly packs (24 train / 8 ID / 8 OOD geometries); no large .pt upload",
        "true_field_id": true_field_three_readouts(te),
        "true_field_ood": true_field_three_readouts(od),
    }
    for name, factory, is_b, lr in [
        ("BC-NO", BCNO2D, True, 3e-3),
        ("FNO2d", FNO2dSmall, False, 1e-3),
        ("ResUNet", ResUNetSmall, False, 1e-3),
        ("MgNO2d", MgNO2dWrapper, False, 1e-3),
    ]:
        print(f"  train {name} 40ep for readout table...", flush=True)
        mdl = factory()
        train_model(mdl, is_b, tr, epochs=40, batch_size=8, lr=lr, seed=0, device=device,
                    use_keff_loss=False, use_sigma_loss=is_b, use_energy_loss=False)
        energy[name] = {
            "id": evaluate_three_readouts(mdl, te, is_b, device),
            "ood": evaluate_three_readouts(mdl, od, is_b, device),
        }
        print(f"    {name} OOD Naive/Dipole/Energy:", energy[name]["ood"], flush=True)
    out["energy_readout"] = energy

    if not args.skip_curves:
        print("\n=== Part 4: Training curves (same on-the-fly packs) ===", flush=True)
        curves = {}
        for name, factory, is_b, lr in [
            ("BC-NO", BCNO2D, True, 3e-3),
            ("FNO2d", FNO2dSmall, False, 1e-3),
            ("ResUNet", ResUNetSmall, False, 1e-3),
            ("MgNO2d", MgNO2dWrapper, False, 1e-3),
        ]:
            print(f"  curves {name}...", flush=True)
            curves[name] = train_curves(factory(), is_b, tr, te, args.epochs_curves, 8, lr, 1, device)
        out["training_curves"] = curves
        plot_curves(curves, root)
    else:
        out["training_curves"] = {"skipped": True}

    path = root / "bcno_v7_empirical_addons_results.json"
    path.write_text(json.dumps(out, indent=2))
    print("\nSaved", path, flush=True)


if __name__ == "__main__":
    main()
