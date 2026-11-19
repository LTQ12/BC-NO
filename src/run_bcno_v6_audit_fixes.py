"""CMAME Audit Remediation Suite (v6):
  1. Equipotential-Clamping Baseline (Exp 1 48^2 + Exp 2 N=128) to empirically measure clamping underprediction.
  2. Classical CG-FFT (Willot 2015 rotated-grid Green) & Same-GPU Jacobi-PCG reference (accuracy + GPU/CPU timing).
  3. Homogenization-Specific Learned Baseline: Lippmann-Schwinger Unrolled FNO (LS-FNO, Nguyen & Schneider style)
     evaluated on N=128 fiber_columns & 2D PBC N=64 (both Naive and Dipole readouts).
  4. Fixed Non-Collapsed Baselines on 2D CT (Exp 6) and 3D 32^3 / 64^3 (Exp 8 & Exp 12) with target RMS scaling,
     standard Kaiming head init, and both Naive & Dipole readouts.
  5. Unified Exp 11 (1000-train evaluated with Dipole+Naive on BOTH Exp2-OOD and Harder-OOD) &
     Unified 3-seed Exp 10 (Thermal FE^2 Raw vs SPD-Projected from the exact same run).
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
import torch.nn as nn
import torch.nn.functional as F

from baselines_2d import CNO2dWrapper, FFNO2d, FNO2dSmall, MgNO2dWrapper, ResUNetSmall, Transolver2d
from bcno_model_2d import BCNO2D, DirichletGreen2D, PeriodicGreen2D, dipole_keff_columns, periodic_dipole_keff, precompute_floating_components
from bcno_model_3d import BCNO3D, FNO3dSmall, ResUNet3dSmall, dipole_keff_columns_3d, jacobi_pcg_3d, precompute_floating_components_3d
from colab_data_gen import fast_interior_solve, generate_synthetic_benchmarks, make_fiber_columns, make_hard_topology
from rve_discrete_operator_2d import conservative_flux_columns, harmonic_mean
from rve_discrete_operator_3d import conservative_flux_columns_3d
from run_bcno_fe2_audit import build_fe2_micro_suite, constitutive_audit, predict_keff_cols, solve_macro_q1_plate
from run_bcno_paper_suite import agg_seeds, evaluate_pack, pack_from_pt, paired_tensor_error, relative_l2, train_model
from run_bcno_scale_and_fe2 import build_3d_64_synth_pack, build_large_2d_pack, load_3d_ruhr64_pack
from run_bcno_v2_addons import build_periodic_2d_pack, load_3d_ruhr_pack, make_3d_synthetic_pack


def spd_project_2x2(K_tensors: np.ndarray, min_eig: float = 1.0) -> np.ndarray:
    K_sym = 0.5 * (K_tensors + np.swapaxes(K_tensors, -1, -2))
    shape = K_sym.shape[:-2]
    vals, vecs = np.linalg.eigh(K_sym.reshape(-1, 2, 2))
    vals_c = np.maximum(vals, min_eig)
    return np.einsum("eik,ek,ejk->eij", vecs, vals_c, vecs).reshape(*shape, 2, 2)


# ---------- 1. Equipotential-Clamping Baseline (Remark 1 empirical verification) ----------
def _exp1_48x48_geoms(contrast: float) -> list[tuple[str, np.ndarray]]:
    n = 48
    geoms = []
    # wall: box(48, c, 16, 4, 16, 12)
    kw = np.ones((n, n), dtype=np.float64); kw[16:32, 4:16] = contrast; geoms.append(("wall", kw))
    # center: box(48, c, 16, 18, 16, 12)
    kc = np.ones((n, n), dtype=np.float64); kc[16:32, 18:30] = contrast; geoms.append(("center", kc))
    # bridge: u_kappa(48, c, 8, 8, 30, 4, 18, 6)
    kb = np.ones((n, n), dtype=np.float64); kb[8:12, 8:38] = contrast; kb[26:30, 8:38] = contrast; kb[8:30, 8:14] = contrast; geoms.append(("bridge", kb))
    # Lhold: lshape_kappa(48, c, 0.3, 0.12, 0.45)
    kl = np.ones((n, n), dtype=np.float64)
    p = int(round(0.3 * n)); w = max(2, int(round(0.12 * n))); arm = int(round(0.45 * n))
    kl[p : min(n - 1, p + arm), p : min(n - 1, p + w)] = contrast
    kl[p : min(n - 1, p + w), p : min(n - 1, p + arm)] = contrast
    geoms.append(("Lhold", kl))
    return geoms


def run_equipotential_clamping_audit(root: Path) -> dict:
    out = {"exp1_48x48": {}, "exp2_fiber_n128": {}}
    ax1 = np.linspace(0.0, 1.0, 48, dtype=np.float64)
    xg, yg = np.meshgrid(ax1, ax1, indexing="ij")
    xg_t = torch.from_numpy(xg)[None]; yg_t = torch.from_numpy(yg)[None]

    for tag, _ in _exp1_48x48_geoms(100.0):
        out["exp1_48x48"][tag] = {}
    for c in [100.0, 10000.0, 1000000.0]:
        for tag, kap in _exp1_48x48_geoms(c):
            v, lft = fast_interior_solve(kap, 1)  # horizontal x-load matching Exp 1
            u_true = torch.from_numpy(lft + v)[None]
            kap_t = torch.from_numpy(kap)[None]
            k_true = float(conservative_flux_columns(u_true, kap_t, kappa_channel=0)[0, 1])

            inc = kap > 1.5
            u_clamp = u_true.clone()
            u_clamp[0][torch.from_numpy(inc)] = float(u_true[0][torch.from_numpy(inc)].mean())
            k_clamp_naive = float(conservative_flux_columns(u_clamp, kap_t, kappa_channel=0)[0, 1])
            k_clamp_dip = float(dipole_keff_columns(u_clamp, kap_t, xg_t, yg_t)[0, 1])
            out["exp1_48x48"][tag][str(int(c))] = {
                "K_true": round(k_true, 5),
                "K_clamp_naive": round(k_clamp_naive, 5),
                "rel_underpred_naive": round((k_clamp_naive - k_true) / k_true, 4),
                "rel_err_dip": round(abs(k_clamp_dip - k_true) / k_true, 4),
            }

    for sp_name, pt_file, split in [("id", "rve_fiber_columns_N128_f64.pt", "test"), ("ood", "rve_fiber_columns_ood_N128_f64.pt", "test")]:
        pk = pack_from_pt(root / pt_file, split)
        u_clamp = pk["u_true"].double().clone()
        lbl = pk["labels"]
        for b in range(u_clamp.shape[0]):
            for m in range(1, int(lbl[b].max()) + 1):
                cm = lbl[b] == m
                if cm.any():
                    u_clamp[b][cm] = u_clamp[b][cm].mean()
        k_clamp_nv = conservative_flux_columns(u_clamp, pk["kappa_raw"].double(), kappa_channel=0).float()
        k_clamp_dp = dipole_keff_columns(u_clamp, pk["kappa_raw"].double(), pk["x_grid"].double(), pk["y_grid"].double()).float()
        out["exp2_fiber_n128"][sp_name] = {
            "KeffRelErr_ClampNaive": round(paired_tensor_error(k_clamp_nv, pk["k_cols"]), 4),
            "KeffRelErr_ClampDip": round(paired_tensor_error(k_clamp_dp, pk["k_cols"]), 4),
        }
    return out


# ---------- 2. Classical CG-FFT (Willot 2015 rotated-grid Green) ----------
def willot_cgfft_2d(kap: torch.Tensor, tol: float = 1e-6, max_iter: int = 400) -> tuple[torch.Tensor, float, int]:
    """2D periodic CG-FFT with Willot (2015) rotated-grid discrete Green operator on GPU/CPU."""
    B, N, _ = kap.shape
    dev = kap.device
    k64 = kap.double()

    # Willot (2015) rotated-difference Fourier derivatives (zero out DC & Nyquist checkerboard modes)
    wx = 2.0 * math.pi * torch.fft.fftfreq(N, d=1.0, device=dev, dtype=torch.float64)
    wy = 2.0 * math.pi * torch.fft.fftfreq(N, d=1.0, device=dev, dtype=torch.float64)
    kx, ky = torch.meshgrid(wx, wy, indexing="ij")
    ex = torch.exp(1j * kx) - 1.0
    ey = torch.exp(1j * ky) - 1.0
    px = 0.5 * ex * (1.0 + torch.exp(1j * ky))
    py = 0.5 * ey * (1.0 + torch.exp(1j * kx))
    k2 = px.abs() ** 2 + py.abs() ** 2
    inv_k2 = torch.where(k2 > 1e-12, 1.0 / k2.clamp_min(1e-12), torch.zeros_like(k2))
    Gamma_xx = (px * px.conj()) * inv_k2
    Gamma_xy = (px * py.conj()) * inv_k2
    Gamma_yx = (py * px.conj()) * inv_k2
    Gamma_yy = (py * py.conj()) * inv_k2

    def apply_Gamma(jx: torch.Tensor, jy: torch.Tensor):
        Jx = torch.fft.fftn(jx, dim=(-2, -1))
        Jy = torch.fft.fftn(jy, dim=(-2, -1))
        Gx = torch.fft.ifftn(Gamma_xx * Jx + Gamma_xy * Jy, dim=(-2, -1)).real
        Gy = torch.fft.ifftn(Gamma_yx * Jx + Gamma_yy * Jy, dim=(-2, -1)).real
        return Gx, Gy

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    K_out = torch.zeros(B, 2, 2, dtype=torch.float64, device=dev)
    iters_list = []
    for ax_id in (0, 1):
        E0x = torch.ones(B, N, N, dtype=torch.float64, device=dev) if ax_id == 0 else torch.zeros(B, N, N, dtype=torch.float64, device=dev)
        E0y = torch.zeros(B, N, N, dtype=torch.float64, device=dev) if ax_id == 0 else torch.ones(B, N, N, dtype=torch.float64, device=dev)
        bx, by = apply_Gamma(-k64 * E0x, -k64 * E0y)
        ex_f = torch.zeros_like(E0x); ey_f = torch.zeros_like(E0y)
        rx, ry = bx.clone(), by.clone()
        px_c, py_c = rx.clone(), ry.clone()
        rr = (rx * rx + ry * ry).sum((1, 2))
        b_norm = torch.sqrt((bx * bx + by * by).sum((1, 2))).clamp_min(1e-12)
        it_used = 0
        for it in range(1, max_iter + 1):
            it_used = it
            if bool((torch.sqrt(rr) <= tol * b_norm).all()):
                break
            Apx, Apy = apply_Gamma(k64 * px_c, k64 * py_c)
            pAp = (px_c * Apx + py_c * Apy).sum((1, 2)).clamp_min(1e-30)
            alpha = (rr / pAp).view(B, 1, 1)
            ex_f = ex_f + alpha * px_c
            ey_f = ey_f + alpha * py_c
            rx = rx - alpha * Apx
            ry = ry - alpha * Apy
            rr_new = (rx * rx + ry * ry).sum((1, 2))
            beta = (rr_new / rr.clamp_min(1e-30)).view(B, 1, 1)
            px_c = rx + beta * px_c
            py_c = ry + beta * py_c
            rr = rr_new
        iters_list.append(it_used)
        K_out[:, 0, ax_id] = (k64 * (E0x + ex_f)).mean((1, 2))
        K_out[:, 1, ax_id] = (k64 * (E0y + ey_f)).mean((1, 2))
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ms_per_sample = (time.time() - t0) * 1000.0 / max(2 * B, 1)
    return K_out.float(), ms_per_sample, int(round(statistics.mean(iters_list)))


# ---------- 3. Homogenization-Specific Learned Baseline: LS-FNO (Lippmann-Schwinger Unrolled FNO) ----------
class LSFNO2d(nn.Module):
    """Lippmann-Schwinger Unrolled Fourier Neural Operator (Nguyen & Schneider 2026 / PINO-style physics-iterative FNO)."""

    def __init__(self, width: int = 32, modes: int = 16, steps: int = 4, periodic: bool = False) -> None:
        super().__init__()
        self.steps = steps
        self.modes = modes
        self.periodic = periodic
        self.green = PeriodicGreen2D() if periodic else DirichletGreen2D()
        self.fc0 = nn.Conv2d(7, width, 1)
        scale = 1.0 / width
        self.convs = nn.ParameterList([nn.Parameter(scale * torch.randn(width, width, modes, modes, 2) * 0.1) for _ in range(steps)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(steps)])
        self.proj = nn.ModuleList([nn.Conv2d(width, 1, 1) for _ in range(steps)])
        for p in self.proj:
            nn.init.normal_(p.weight, std=0.02); nn.init.zeros_(p.bias)
        self.alpha = nn.Parameter(torch.full((steps,), 0.5))

    def _spec(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.complex64, device=x.device)
        m = min(self.modes, H // 2, W // 2)
        wc = torch.complex(w[..., :m, :m, 0], w[..., :m, :m, 1])
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], wc)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], wc)
        return torch.fft.irfft2(out, s=(H, W))

    def forward(self, kappa: torch.Tensor, lift: torch.Tensor, ortho: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        beta = (kappa - 1.0) / (kappa + 1.0)
        log_k = torch.log(kappa.clamp(1e-4, 1e6)) / 10.0
        u = lift.clone()
        for t in range(self.steps):
            if self.periodic:
                gx = 0.5 * (torch.roll(u, -1, 1) - torch.roll(u, 1, 1))
                gy = 0.5 * (torch.roll(u, -1, 2) - torch.roll(u, 1, 2))
            else:
                gx = F.pad(0.5 * (u[:, 2:, :] - u[:, :-2, :]), (0, 0, 1, 1))
                gy = F.pad(0.5 * (u[:, :, 2:] - u[:, :, :-2]), (1, 1, 0, 0))
            h = F.gelu(self.fc0(torch.stack([beta, log_k, lift, ortho, dist, gx, gy], dim=1)))
            h = F.gelu(self._spec(h, self.convs[t]) + self.ws[t](h))
            tau = self.proj[t](h)[:, 0]
            u = (lift if not self.periodic else 0.0) + self.alpha[t] * self.green(tau) + (1.0 - self.alpha[t]) * u
            if not self.periodic:
                u[:, 0, :] = lift[:, 0, :]; u[:, -1, :] = lift[:, -1, :]
                u[:, :, 0] = lift[:, :, 0]; u[:, :, -1] = lift[:, :, -1]
        return u


# ---------- 4. Fixed Non-Collapsed Baselines (Proper Init + Target Normalization + Dual Loss) ----------
def reinit_baseline_properly(mdl: nn.Module) -> nn.Module:
    """Fix collapsed baselines: replace std=1e-3 head & tiny 1/w^2 FNO init with standard Kaiming/Xavier scaling."""
    for m in mdl.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if hasattr(mdl, "convs"):
        for p in mdl.convs:
            with torch.no_grad():
                p.normal_(0.0, 0.05)
    return mdl


def train_2d_fixed(mdl: nn.Module, pack: dict, epochs: int, batch_size: int, lr: float, seed: int, device: torch.device):
    torch.manual_seed(seed)
    mdl = reinit_baseline_properly(mdl).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    B = pack["kappa"].shape[0]
    vt_all = pack["v_true"]
    v_rms = float(torch.sqrt((vt_all ** 2).mean()).clamp_min(1e-3))

    for _ in range(epochs):
        mdl.train()
        perm = torch.randperm(B)
        for s in range(0, B, batch_size):
            idx = perm[s : min(s + batch_size, B)]
            kap = pack["kappa"][idx].to(device); lft = pack["lift"][idx].to(device)
            ort = pack["ortho"][idx].to(device); dst = pack["dist"][idx].to(device)
            vt = pack["v_true"][idx].to(device)
            up = mdl(kap, lft, ort, dst)
            vp = up - lft
            mse_norm = ((vp - vt) / v_rms).pow(2).mean()
            vt_n = torch.sqrt((vt ** 2).sum((1, 2)) + 1e-10)
            nz = vt_n > 1e-4
            if bool(nz.any()):
                rel = (torch.sqrt(((vp[nz] - vt[nz]) ** 2).sum((1, 2)) + 1e-10) / vt_n[nz]).mean()
                loss = 0.5 * mse_norm + 0.5 * rel
            else:
                loss = mse_norm
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 2.0)
            opt.step()
        sched.step()
    return mdl


def train_3d_fixed(mdl: nn.Module, pack: dict, epochs: int, batch_size: int, lr: float, seed: int, device: torch.device):
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
            kap = pack["kappa"][idx].to(device); lft = pack["lift"][idx].to(device)
            o1 = pack["o1"][idx].to(device); o2 = pack["o2"][idx].to(device); dst = pack["dist"][idx].to(device)
            vt = pack["v_true"][idx].to(device)
            up = mdl(kap, lft, o1, o2, dst)
            vp = up - lft
            mse_norm = ((vp - vt) / v_rms).pow(2).mean()
            vt_n = torch.sqrt((vt ** 2).sum((1, 2, 3)) + 1e-10)
            nz = vt_n > 1e-4
            rel = (torch.sqrt(((vp[nz] - vt[nz]) ** 2).sum((1, 2, 3)) + 1e-10) / vt_n[nz]).mean() if bool(nz.any()) else 0.0
            loss = 0.5 * mse_norm + 0.5 * rel
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 2.0)
            opt.step()
        sched.step()
    return mdl


@torch.no_grad()
def eval_3d_dual(mdl, pk: dict, is_b: bool, device: torch.device, bs: int = 4) -> dict[str, float]:
    mdl.eval()
    B = pk["kappa"].shape[0]
    ups = []
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for s in range(0, B, bs):
        e = min(s + bs, B)
        kap = pk["kappa"][s:e].to(device); lft = pk["lift"][s:e].to(device)
        o1 = pk["o1"][s:e].to(device); o2 = pk["o2"][s:e].to(device); dst = pk["dist"][s:e].to(device)
        up = mdl(kap, lft, o1, o2, dst, pk["labels"][s:e].to(device), pk["max_comp"])[0] if is_b else mdl(kap, lft, o1, o2, dst)
        ups.append(up.cpu())
    if device.type == "cuda": torch.cuda.synchronize()
    inf_ms = (time.time() - t0) * 1000.0 / max(B, 1)
    u_p = torch.cat(ups, 0); v_p = u_p - pk["lift"]
    kd = dipole_keff_columns_3d(u_p.double(), pk["kappa"].double(), pk["xg"].double(), pk["yg"].double(), pk["zg"].double()).float()
    kn = conservative_flux_columns_3d(u_p.double(), pk["kappa"].double(), kappa_channel=0).float()
    idx = torch.arange(B); kt = pk["k_cols"][idx, pk["load_axis"]]
    vn = torch.sqrt((pk["v_true"] ** 2).sum((1, 2, 3))); vv = vn > 1e-4
    fl = float((torch.sqrt(((v_p[vv] - pk["v_true"][vv]) ** 2).sum((1, 2, 3)) + 1e-12) / vn[vv]).mean()) if bool(vv.any()) else 0.0
    return {
        "DiagDip": float(((kd[idx, pk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean()),
        "DiagNaive": float(((kn[idx, pk["load_axis"]] - kt).abs() / kt.abs().clamp_min(1e-4)).mean()),
        "FluctL2": fl,
        "InferMs": inf_ms,
    }


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
    # Verify any cached .pt files aren't corrupted before skipping generation
    for pt_name in ["rve_fiber_columns_N128_f64.pt", "rve_fiber_columns_ood_N128_f64.pt", "real_ct_32.pt", "real_ct_ruhr_N128_hc.pt", "wavy_laminate_N128.pt", "hard_bridge_N128.pt", "fiber_N64_eval.pt", "fiber_N256_eval.pt"]:
        p = root / pt_name
        if p.exists():
            try:
                torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                print(f"Removing corrupted cache file: {pt_name}", flush=True)
                p.unlink()
    generate_synthetic_benchmarks(root)

    out: dict[str, object] = {"device": str(device), "seeds": seeds, "epochs": ep}

    # ==== 1. Equipotential Clamping Empirical Check ====
    print("\n=== Part 1: Equipotential Clamping Verification (Exp 1 & Exp 2) ===", flush=True)
    out["equipotential_clamping"] = run_equipotential_clamping_audit(root)
    print("  48^2 Bridge clamping naive rel err:", out["equipotential_clamping"]["exp1_48x48"]["bridge"]["1000000"], flush=True)
    print("  N=128 Fiber clamping naive:", out["equipotential_clamping"]["exp2_fiber_n128"], flush=True)

    # ==== 2. CG-FFT (Willot) & Same-GPU Jacobi-PCG Timing & Accuracy ====
    print("\n=== Part 2: Classical CG-FFT (Willot) & Same-GPU Iterative Solver Timing ===", flush=True)
    pbc_od = build_periodic_2d_pack(64, 4 if args.smoke else 16, 303, 100.0, 300.0)
    pbc_id = build_periodic_2d_pack(64, 4 if args.smoke else 16, 202, 10.0, 80.0)
    # Run Willot CG-FFT on same device as BC-NO
    k_cg_id, ms_cg_id, it_id = willot_cgfft_2d(pbc_id["kappa"][0::2].to(device), tol=1e-6)
    k_cg_od, ms_cg_od, it_od = willot_cgfft_2d(pbc_od["kappa"][0::2].to(device), tol=1e-6)
    k_t_id = pbc_id["k_cols"].view(-1, 2, 2).transpose(1, 2)
    k_t_od = pbc_od["k_cols"].view(-1, 2, 2).transpose(1, 2)
    err_cg_id = float((torch.linalg.vector_norm(k_cg_id.cpu() - k_t_id, dim=(1, 2)) / torch.linalg.vector_norm(k_t_id, dim=(1, 2))).mean())
    err_cg_od = float((torch.linalg.vector_norm(k_cg_od.cpu() - k_t_od, dim=(1, 2)) / torch.linalg.vector_norm(k_t_od, dim=(1, 2))).mean())

    # Same-GPU 3D Jacobi-PCG timing at 32^3 and 64^3 (with CUDA warmup!)
    s3_od = make_3d_synthetic_pack(32, 2 if args.smoke else 6, 803, 200.0, 1000.0)
    r64_te = load_3d_ruhr64_pack(root / "rve_3d_ruhr64_compact.npz", "te64", device)
    bc3d = BCNO3D(steps=3, hidden=20).to(device).eval()
    # Warmup GPU FFT/Conv kernels before timing
    with torch.no_grad():
        for _ in range(3):
            bc3d(s3_od["kappa"][0:3].to(device), s3_od["lift"][0:3].to(device), s3_od["o1"][0:3].to(device), s3_od["o2"][0:3].to(device), s3_od["dist"][0:3].to(device), s3_od["labels"][0:3].to(device), s3_od["max_comp"])
            bc3d(r64_te["kappa"][0:3].to(device), r64_te["lift"][0:3].to(device), r64_te["o1"][0:3].to(device), r64_te["o2"][0:3].to(device), r64_te["dist"][0:3].to(device), r64_te["labels"][0:3].to(device), r64_te["max_comp"])
    _, _, _ = jacobi_pcg_3d(s3_od["kappa"][0:3].to(device), s3_od["lift"][0:3].to(device), tol=1e-6)
    _, it32, pcg32_ms = jacobi_pcg_3d(s3_od["kappa"][0:3].to(device), s3_od["lift"][0:3].to(device), tol=1e-6)
    _, it64, pcg64_ms = jacobi_pcg_3d(r64_te["kappa"][0:3].to(device), r64_te["lift"][0:3].to(device), tol=1e-6)
    with torch.no_grad():
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        bc3d(s3_od["kappa"][0:3].to(device), s3_od["lift"][0:3].to(device), s3_od["o1"][0:3].to(device), s3_od["o2"][0:3].to(device), s3_od["dist"][0:3].to(device), s3_od["labels"][0:3].to(device), s3_od["max_comp"])
        if device.type == "cuda": torch.cuda.synchronize()
        bc32_ms = (time.time() - t0) * 1000.0 / 3.0
        t0 = time.time()
        bc3d(r64_te["kappa"][0:3].to(device), r64_te["lift"][0:3].to(device), r64_te["o1"][0:3].to(device), r64_te["o2"][0:3].to(device), r64_te["dist"][0:3].to(device), r64_te["labels"][0:3].to(device), r64_te["max_comp"])
        if device.type == "cuda": torch.cuda.synchronize()
        bc64_ms = (time.time() - t0) * 1000.0 / 3.0

    out["classical_solvers"] = {
        "cgfft_willot_2d_pbc64": {
            "id_keff_err": round(err_cg_id, 4), "id_iters": it_id, "id_gpu_ms": round(ms_cg_id, 2),
            "ood_keff_err": round(err_cg_od, 4), "ood_iters": it_od, "ood_gpu_ms": round(ms_cg_od, 2),
        },
        "same_gpu_3d_pcg_vs_bcno3d": {
            "32_cubed_pcg_ms_3axes": round(pcg32_ms, 2), "32_cubed_pcg_iters": it32, "32_cubed_bcno_ms_per_axis": round(bc32_ms, 2),
            "64_cubed_pcg_ms_3axes": round(pcg64_ms, 2), "64_cubed_pcg_iters": it64, "64_cubed_bcno_ms_per_axis": round(bc64_ms, 2),
        },
    }
    print("  Willot CG-FFT PBC64:", out["classical_solvers"]["cgfft_willot_2d_pbc64"], flush=True)
    print("  Same-GPU 3D PCG vs BCNO:", out["classical_solvers"]["same_gpu_3d_pcg_vs_bcno3d"], flush=True)

    # ==== 3. Homogenization-Specific Learned Baseline: LS-FNO on N=128 fiber_columns & 2D PBC ====
    print("\n=== Part 3: LS-FNO Homogenization Baseline (N=128 fiber_columns + PBC64) ===", flush=True)
    f_tr = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "train")
    f_id = pack_from_pt(root / "rve_fiber_columns_N128_f64.pt", "test")
    f_od = pack_from_pt(root / "rve_fiber_columns_ood_N128_f64.pt", "test")
    ls_id, ls_od = [], []
    for sd in seeds:
        mdl = LSFNO2d(width=32, modes=16, steps=4, periodic=False).float()
        train_model(mdl, False, f_tr, epochs=ep, batch_size=8, lr=2e-3, seed=sd, device=device, use_keff_loss=False, use_sigma_loss=False)
        ls_id.append(evaluate_pack(mdl, f_id, False, device, use_dipole=True))
        ls_od.append(evaluate_pack(mdl, f_od, False, device, use_dipole=True))
    out["lsfno_fiber128"] = {"params": sum(p.numel() for p in LSFNO2d().parameters()), "id": agg_seeds(ls_id), "ood": agg_seeds(ls_od)}
    print(f"  [LS-FNO Fiber128] ID Dip={out['lsfno_fiber128']['id']['KeffRelErr']['mean']:.3f} Naive={out['lsfno_fiber128']['id']['KeffRelErr_Naive']['mean']:.3f} | OOD Dip={out['lsfno_fiber128']['ood']['KeffRelErr']['mean']:.3f} Naive={out['lsfno_fiber128']['ood']['KeffRelErr_Naive']['mean']:.3f} Fluct={out['lsfno_fiber128']['ood']['Fluctuation_L2']['mean']:.3f}", flush=True)

    # Also evaluate LSFNO2d(periodic=True) on 2D PBC N=64 (direct comparison to Nguyen & Schneider periodic homogenization)
    pbc_tr = build_periodic_2d_pack(64, 16 if args.smoke else 48, 101, 10.0, 80.0)
    ls_pbc_id, ls_pbc_od = [], []
    for sd in seeds:
        torch.manual_seed(sd)
        mdl_p = LSFNO2d(width=32, modes=16, steps=4, periodic=True).float().to(device)
        opt_p = torch.optim.AdamW(mdl_p.parameters(), lr=2e-3, weight_decay=1e-5)
        sched_p = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p, T_max=ep)
        B_p = pbc_tr["kappa"].shape[0]
        ax_p = torch.arange(64, dtype=torch.float32, device=device) / 64.0
        gx_p, gy_p = torch.meshgrid(ax_p, ax_p, indexing="ij")
        for _ in range(ep):
            mdl_p.train(); perm = torch.randperm(B_p)
            for s in range(0, B_p, 8):
                idx = perm[s : min(s + 8, B_p)]
                kap = pbc_tr["kappa"][idx].to(device); vt = pbc_tr["v_true"][idx].to(device)
                axb = pbc_tr["load_axis"][idx].to(device)
                lft = torch.where(axb[:, None, None] == 0, gx_p[None], gy_p[None])
                ort = torch.where(axb[:, None, None] == 0, gy_p[None], gx_p[None])
                dst = torch.zeros_like(kap)
                vp = mdl_p(kap, lft, ort, dst) - lft
                vp = vp - vp.mean((1, 2), keepdim=True)
                loss = (torch.sqrt(((vp - vt) ** 2).sum((1, 2)) + 1e-8) / torch.sqrt((vt ** 2).sum((1, 2)) + 1e-6).clamp_min(1e-2)).mean()
                opt_p.zero_grad(); loss.backward(); opt_p.step()
            sched_p.step()
        for lst, spk in [(ls_pbc_id, pbc_id), (ls_pbc_od, pbc_od)]:
            mdl_p.eval()
            vps = []
            with torch.no_grad():
                for s in range(0, spk["kappa"].shape[0], 16):
                    e = min(s + 16, spk["kappa"].shape[0])
                    kap = spk["kappa"][s:e].to(device); axb = spk["load_axis"][s:e].to(device)
                    lft = torch.where(axb[:, None, None] == 0, gx_p[None], gy_p[None])
                    ort = torch.where(axb[:, None, None] == 0, gy_p[None], gx_p[None])
                    vp = mdl_p(kap, lft, ort, torch.zeros_like(kap)) - lft
                    vps.append((vp - vp.mean((1, 2), keepdim=True)).cpu())
            vp_all = torch.cat(vps, 0)
            kd = periodic_dipole_keff(vp_all.double(), spk["kappa"].double(), spk["load_axis"], spk["nonperc"], spk["dx_wrap"].double(), spk["dy_wrap"].double()).float()
            kn = periodic_dipole_keff(vp_all.double(), spk["kappa"].double(), spk["load_axis"], torch.zeros_like(spk["nonperc"]), spk["dx_wrap"].double(), spk["dy_wrap"].double()).float()
            lst.append({
                "KeffDip": paired_tensor_error(kd, spk["k_cols"]),
                "KeffNaive": paired_tensor_error(kn, spk["k_cols"]),
                "FluctL2": float(relative_l2(vp_all, spk["v_true"]).mean()),
            })
    out["lsfno_pbc64"] = {"id": agg_seeds(ls_pbc_id), "ood": agg_seeds(ls_pbc_od)}
    print(f"  [LS-FNO PBC64   ] ID Dip={out['lsfno_pbc64']['id']['KeffDip']['mean']:.3f} Naive={out['lsfno_pbc64']['id']['KeffNaive']['mean']:.3f} | OOD Dip={out['lsfno_pbc64']['ood']['KeffDip']['mean']:.3f} Naive={out['lsfno_pbc64']['ood']['KeffNaive']['mean']:.3f} Fluct={out['lsfno_pbc64']['ood']['FluctL2']['mean']:.3f}", flush=True)

    # ==== 4. Fix Collapsed Baselines on 2D CT (Exp 6) & 3D 32^3 / 64^3 (Exp 8 & 12) ====
    print("\n=== Part 4: Fixed Non-Collapsed Baselines on 2D CT (Exp 6) ===", flush=True)
    ep_ct = 2 if args.smoke else 100
    exp6_fixed = {}
    for ct_name, pt_file, sp_test, norm_med, modes_f in [
        ("real_ct_32", "real_ct_32.pt", "locked_test", True, 10),
        ("ruhr_sandstone_N128_hc", "real_ct_ruhr_N128_hc.pt", "test", False, 16),
    ]:
        tr_p = pack_from_pt(root / pt_file, "train", normalize_median=norm_med)
        te_p = pack_from_pt(root / pt_file, sp_test, normalize_median=norm_med)
        ct_out = {}
        for name, ctor, lr_use in [
            ("MgNO2d", lambda: MgNO2dWrapper(5, width=48, levels=4), 3e-3),
            ("Transolver2d", lambda: Transolver2d(5, dim=64, depth=4, heads=4, n_slices=32), 2e-3),
            ("FFNO2d", lambda mf=modes_f: FFNO2d(5, width=48, modes=mf, depth=6), 3e-3),
            ("CNO2d", lambda: CNO2dWrapper(5, width=32), 3e-3),
            ("ResUNet", lambda: ResUNetSmall(5, width=32), 3e-3),
        ]:
            te_s = []
            for sd in seeds:
                mdl = train_2d_fixed(ctor().float(), tr_p, ep_ct, 8, lr_use, sd, device)
                te_s.append(evaluate_pack(mdl, te_p, False, device, use_dipole=not norm_med))
            ct_out[name] = agg_seeds(te_s)
            print(f"  [Fixed 2D CT {ct_name:22s} | {name:12s}] DiagDip={ct_out[name]['DiagRelErr']['mean']:.3f} DiagNaive={ct_out[name]['DiagRelErr_Naive']['mean']:.3f} TempL2={ct_out[name]['Temperature_L2']['mean']:.4f} FluctL2={ct_out[name]['Fluctuation_L2']['mean']:.3f}", flush=True)
        exp6_fixed[ct_name] = ct_out
    out["exp6_fixed_baselines"] = exp6_fixed

    print("\n=== Part 5: Fixed Non-Collapsed Baselines on 3D 32^3 & 64^3 (Exp 8 & Exp 12) ===", flush=True)
    s3_tr = make_3d_synthetic_pack(32, 4 if args.smoke else 24, 801, 10.0, 80.0)
    s3_id = make_3d_synthetic_pack(32, 2 if args.smoke else 8, 802, 10.0, 80.0)
    s3_od = make_3d_synthetic_pack(32, 2 if args.smoke else 8, 803, 200.0, 1000.0)
    r3_tr = load_3d_ruhr_pack(root / "rve_3d_ct_compact.npz", "tr32", solve_labels=True)
    r3_te = load_3d_ruhr_pack(root / "rve_3d_ct_compact.npz", "te32", solve_labels=True)

    ep_3d32 = 2 if args.smoke else 80
    exp8_fixed = {}
    for tag, tr_p, te_packs in [("Synth32", s3_tr, {"id": s3_id, "ood": s3_od}), ("RuhrCT32", r3_tr, {"test": r3_te})]:
        res = {}
        for name, ctor in [("ResUNet-3D", lambda: ResUNet3dSmall(6, width=20)), ("FNO-3D", lambda: FNO3dSmall(6, width=20, modes=8, depth=4))]:
            per_sp = {k: [] for k in te_packs}
            for sd in seeds:
                mdl = train_3d_fixed(ctor().float(), tr_p, ep_3d32, 6, 3e-3, sd, device)
                for sk, spk in te_packs.items():
                    per_sp[sk].append(eval_3d_dual(mdl, spk, False, device, bs=6))
            res[name] = {sk: agg_seeds(per_sp[sk]) for sk in te_packs}
            for sk in te_packs:
                print(f"  [Fixed 3D {tag:8s} | {sk:4s} | {name:10s}] Dip={res[name][sk]['DiagDip']['mean']:.3f} Naive={res[name][sk]['DiagNaive']['mean']:.3f} Fluct={res[name][sk]['FluctL2']['mean']:.3f}", flush=True)
        exp8_fixed[tag] = res
    out["exp8_fixed_baselines"] = exp8_fixed

    # 3D 64^3 Fixed Baselines (1 seed, 50 epochs)
    s64_tr = build_3d_64_synth_pack(64, 2 if args.smoke else 40, 2101, 10.0, 80.0, device)
    s64_id = build_3d_64_synth_pack(64, 1 if args.smoke else 10, 2102, 10.0, 80.0, device)
    s64_od = build_3d_64_synth_pack(64, 1 if args.smoke else 10, 2103, 200.0, 1000.0, device)
    r64_tr = load_3d_ruhr64_pack(root / "rve_3d_ruhr64_compact.npz", "tr64", device)
    ep_3d64 = 2 if args.smoke else 50
    exp12_fixed = {}
    for tag, tr_p, te_dict in [("Synth64", s64_tr, {"id": s64_id, "ood": s64_od}), ("Ruhr64", r64_tr, {"test": r64_te})]:
        res = {}
        for name, ctor in [("ResUNet-3D", lambda: ResUNet3dSmall(6, width=16)), ("FNO-3D", lambda: FNO3dSmall(6, width=16, modes=8, depth=4))]:
            mdl = train_3d_fixed(ctor().float(), tr_p, ep_3d64, 3, 3e-3, 0, device)
            res[name] = {sk: eval_3d_dual(mdl, spk, False, device, bs=3) for sk, spk in te_dict.items()}
            for sk, d in res[name].items():
                print(f"  [Fixed 3D64 {tag:7s} | {sk:4s} | {name:10s}] Dip={d['DiagDip']:.3f} Naive={d['DiagNaive']:.3f} Fluct={d['FluctL2']:.3f}", flush=True)
        exp12_fixed[tag] = res
    out["exp12_fixed_baselines"] = exp12_fixed

    # ==== 5. Unified Exp 11 (1000-train evaluated w/ Dipole+Naive on BOTH Exp2-OOD & Exp11-HardOOD) ====
    print("\n=== Part 6: Unified Exp 11 (1000-train Dipole+Naive on BOTH OOD splits) ===", flush=True)
    n_tr11 = 8 if args.smoke else 500
    n_te11 = 4 if args.smoke else 100
    big_tr = build_large_2d_pack(128, n_tr11, 1101, 10.0, 80.0, 0.30, 0.58, device)
    big_id = build_large_2d_pack(128, n_te11, 1102, 10.0, 80.0, 0.30, 0.58, device)
    big_od = build_large_2d_pack(128, n_te11, 1103, 100.0, 400.0, 0.45, 0.72, device)
    ep_big = 2 if args.smoke else 50
    exp11_u = {}
    for name, is_b, use_k, use_sig, ctor in [
        ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
        ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
        ("Transolver2d", False, False, False, lambda: Transolver2d(5, dim=64, depth=4, heads=4, n_slices=32)),
        ("FNO2d", False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
    ]:
        torch.manual_seed(0)
        mdl = ctor().float()
        train_model(mdl, is_b, big_tr, epochs=ep_big, batch_size=16, lr=(3e-3 if is_b else 2e-3), seed=0, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
        exp11_u[name] = {
            "id": evaluate_pack(mdl, big_id, is_b, device, use_dipole=True),
            "ood_exp2": evaluate_pack(mdl, f_od, is_b, device, use_dipole=True),
            "ood_hard": evaluate_pack(mdl, big_od, is_b, device, use_dipole=True),
        }
        e11 = exp11_u[name]
        print(f"  [Unified Exp11 {name:12s}] ID Dip={e11['id']['KeffRelErr']:.3f} | Exp2-OOD Dip={e11['ood_exp2']['KeffRelErr']:.3f} Naive={e11['ood_exp2']['KeffRelErr_Naive']:.3f} | Hard-OOD Dip={e11['ood_hard']['KeffRelErr']:.3f} Naive={e11['ood_hard']['KeffRelErr_Naive']:.3f}", flush=True)
    out["exp11_unified"] = exp11_u

    # ==== 6. Unified 3-seed Exp 10 (Thermal FE^2 Raw vs SPD-Projected, same run!) ====
    print("\n=== Part 7: Unified 3-Seed Exp 10 (Thermal FE^2 Raw vs SPD-Projected) ===", flush=True)
    M_mac = 6 if args.smoke else 16
    fe2_pk = build_fe2_micro_suite(M_macro=M_mac, N_micro=128, seed=2026)
    n_el = M_mac * M_mac
    K_true_mac = fe2_pk["k_cols"].view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_mac, M_mac, 2, 2)
    T_ref, flux_ref, _ = solve_macro_q1_plate(K_true_mac, M=M_mac)
    exp10_u = {"micro_solve_s": round(fe2_pk["micro_solve_s"], 2), "models": {}}
    for name, is_b, use_k, use_sig, ctor in [
        ("BC-NO", True, True, True, lambda: BCNO2D(steps=4, hidden=32)),
        ("MgNO2d", False, False, False, lambda: MgNO2dWrapper(5, width=48, levels=4)),
        ("FFNO2d", False, False, False, lambda: FFNO2d(5, width=48, modes=20, depth=6)),
        ("CNO2d", False, False, False, lambda: CNO2dWrapper(5, width=32)),
        ("ResUNet", False, False, False, lambda: ResUNetSmall(5, width=32)),
        ("FNO2d", False, False, False, lambda: FNO2dSmall(5, width=32, modes=16, depth=4)),
    ]:
        seed_rows = []
        for sd in seeds:
            torch.manual_seed(sd)
            mdl = ctor().float()
            train_model(mdl, is_b, f_tr, epochs=ep, batch_size=8, lr=(3e-3 if is_b else 2e-3), seed=sd, device=device, use_keff_loss=use_k, use_sigma_loss=use_sig)
            k_od_p, _, _, _ = predict_keff_cols(mdl, f_od, is_b, device)
            aud = constitutive_audit(k_od_p, f_od["kappa_raw"], f_od["load_axis"])
            k_fe2_p, k_fe2_dip, _, inf_s = predict_keff_cols(mdl, fe2_pk, is_b, device)

            # Evaluate BOTH raw/naive and SPD-projected (and also Dipole+SPD for baselines!)
            K_surr = k_fe2_p.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_mac, M_mac, 2, 2)
            K_sym = 0.5 * (K_surr + np.swapaxes(K_surr, -1, -2))
            eigs = np.linalg.eigvalsh(K_sym.reshape(-1, 2, 2))[:, 0]
            T_raw, f_raw, spd_all_raw = solve_macro_q1_plate(K_surr, M=M_mac)
            K_proj = spd_project_2x2(K_surr, min_eig=1.0)
            T_proj, f_proj, _ = solve_macro_q1_plate(K_proj, M=M_mac)

            # Also baseline w/ dipole + SPD projection for fair comparison
            K_dip_arr = k_fe2_dip.view(n_el, 2, 2).transpose(1, 2).numpy().reshape(M_mac, M_mac, 2, 2)
            K_dip_proj = spd_project_2x2(K_dip_arr, min_eig=1.0)
            T_dip_p, f_dip_p, _ = solve_macro_q1_plate(K_dip_proj, M=M_mac)

            seed_rows.append({
                "inf_s": inf_s, "speedup": fe2_pk["micro_solve_s"] / max(inf_s, 1e-4),
                "Sym": aud["OnsagerSymErr"], "SPD_ood": aud["SPDRate"], "VR_ood": aud["VoigtReussRate"],
                "rve_err": float((torch.linalg.vector_norm(k_fe2_p - fe2_pk["k_cols"], dim=1) / torch.linalg.vector_norm(fe2_pk["k_cols"], dim=1)).mean()),
                "spd_rate_plate": float(np.mean(eigs > 1e-6)),
                "macro_T_raw": float(np.linalg.norm(T_raw - T_ref) / np.linalg.norm(T_ref)) if np.isfinite(T_raw).all() else float("nan"),
                "macro_flux_raw": float(abs(f_raw[0] - flux_ref[0]) / abs(flux_ref[0])) if np.isfinite(f_raw).all() else float("nan"),
                "macro_T_spd": float(np.linalg.norm(T_proj - T_ref) / np.linalg.norm(T_ref)),
                "macro_flux_spd": float(abs(f_proj[0] - flux_ref[0]) / abs(flux_ref[0])),
                "macro_T_dip_spd": float(np.linalg.norm(T_dip_p - T_ref) / np.linalg.norm(T_ref)),
                "macro_flux_dip_spd": float(abs(f_dip_p[0] - flux_ref[0]) / abs(flux_ref[0])),
            })
        exp10_u["models"][name] = agg_seeds(seed_rows)
        m10 = exp10_u["models"][name]
        print(f"  [Unified Exp10 {name:8s}] Sym={m10['Sym']['mean']:.3f} | Raw T={m10['macro_T_raw']['mean']:.4f} Flux={m10['macro_flux_raw']['mean']:.4f} | SPD T={m10['macro_T_spd']['mean']:.4f} Flux={m10['macro_flux_spd']['mean']:.4f} | Dip+SPD Flux={m10['macro_flux_dip_spd']['mean']:.4f}", flush=True)
    out["exp10_unified"] = exp10_u

    out_path = root / "bcno_v6_audit_fixes_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
