"""Empirical mechanism figure from real discrete RVE solves (no cartoon text boxes).

Panels:
  (a) Controlled inclusion tilt error -> naive Keff error scales as Theta(kappa), dipole stays O(1)
  (b) Nodal sensitivity |g| map on the same U-bridge (where Theta(kappa) lives)
  (c) Exact equipotential clamp underprediction vs kappa on the same geometry

Outputs (flat /content or local cwd):
  fig_empirical_mechanism.pdf / .png
  empirical_mechanism_data.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bcno_model_2d import dipole_keff_columns
from colab_data_gen import fast_interior_solve
from rve_discrete_operator_2d import conservative_flux_columns


def make_u_bridge(n: int = 48, contrast: float = 1e3) -> np.ndarray:
    """Exact Exp.1 unseen U-bridge: u_kappa(48,c,8,8,30,4,18,6) as in v6 audit."""
    kap = np.ones((n, n), dtype=np.float64)
    kap[8:12, 8:38] = contrast
    kap[26:30, 8:38] = contrast
    kap[8:30, 8:14] = contrast
    return kap


def nodal_sensitivity_g(kappa: np.ndarray, load_axis: int = 0) -> np.ndarray:
    """Sparse readout weights g for conservative Keff (Proposition 1)."""
    n = kappa.shape[0]
    h = 1.0 / (n - 1)
    kap = torch.from_numpy(kappa)[None]
    kx = 2.0 * kap[:, :-1, :] * kap[:, 1:, :] / (kap[:, :-1, :] + kap[:, 1:, :]).clamp_min(1e-12)
    ky = 2.0 * kap[:, :, :-1] * kap[:, :, 1:] / (kap[:, :, :-1] + kap[:, :, 1:]).clamp_min(1e-12)
    g = np.zeros((n, n), dtype=np.float64)
    if load_axis == 0:
        # Keff_xx from interior x-faces; coeff of u_p at left/right of faces
        n_faces = (n - 1) * (n - 2)
        # face between (i,j) and (i+1,j) for j=1..n-2, contributes +/- k/(n_faces h)
        for i in range(n - 1):
            for j in range(1, n - 1):
                kface = float(kx[0, i, j])
                w = kface / (n_faces * h)
                g[i, j] -= w
                g[i + 1, j] += w
    else:
        n_faces = (n - 2) * (n - 1)
        for i in range(1, n - 1):
            for j in range(n - 1):
                kface = float(ky[0, i, j])
                w = kface / (n_faces * h)
                g[i, j] -= w
                g[i, j + 1] += w
    return g


def main(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 48
    contrasts = [10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 1e5, 1e6]
    eps = 0.02  # controlled tilt amplitude inside inclusion

    naive_err, dip_err, clamp_under, g1_norms, field_l2 = [], [], [], [], []
    g_map = None
    u_vis = None
    kap_vis = None

    axis = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    xg = torch.from_numpy(xx)[None]
    yg = torch.from_numpy(yy)[None]

    load_axis = 1  # matches Exp.1 / v6 bridge audit (transverse conduction through U-arms)
    for c in contrasts:
        kap = make_u_bridge(n, c)
        v, lift = fast_interior_solve(kap, load_axis)
        u = v + lift
        kap_t = torch.from_numpy(kap)[None]
        u_t = torch.from_numpy(u)[None]
        k_true = float(conservative_flux_columns(u_t, kap_t, kappa_channel=0)[0, load_axis])

        # Controlled tilt only inside inclusion (correlated with g sign pattern)
        inc = kap > 1.5
        e = np.zeros_like(u)
        coord = yy if load_axis == 1 else xx
        e[inc] = eps * (coord[inc] - coord[inc].mean())
        u_pert = u + e
        # keep Dirichlet BC
        u_pert[0, :] = lift[0, :]
        u_pert[-1, :] = lift[-1, :]
        u_pert[:, 0] = lift[:, 0]
        u_pert[:, -1] = lift[:, -1]
        up_t = torch.from_numpy(u_pert)[None]
        k_nv = float(conservative_flux_columns(up_t, kap_t, kappa_channel=0)[0, load_axis])
        k_dp = float(dipole_keff_columns(up_t, kap_t, xg, yg)[0, load_axis])
        fl2 = float(np.linalg.norm(e) / max(np.linalg.norm(u - lift), 1e-12))
        naive_err.append(abs(k_nv - k_true) / abs(k_true))
        dip_err.append(abs(k_dp - k_true) / abs(k_true))
        field_l2.append(fl2)

        # Equipotential clamp of exact solution
        u_cl = u.copy()
        u_cl[inc] = u[inc].mean()
        k_cl = float(conservative_flux_columns(torch.from_numpy(u_cl)[None], kap_t, kappa_channel=0)[0, load_axis])
        clamp_under.append((k_cl - k_true) / k_true)

        g = nodal_sensitivity_g(kap, load_axis)
        g1_norms.append(float(np.abs(g).sum()))
        if abs(c - 1e4) < 1e-9:
            g_map = g
            u_vis = u
            kap_vis = kap

    data = {
        "contrasts": contrasts,
        "tilt_eps": eps,
        "field_l2_of_tilt": field_l2,
        "naive_keff_rel_err": naive_err,
        "dipole_keff_rel_err": dip_err,
        "clamp_rel_underprediction": clamp_under,
        "g_l1_norm": g1_norms,
        "note": "U-bridge N=48; tilt error only inside inclusion; clamp of exact u",
    }
    (out_dir / "empirical_mechanism_data.json").write_text(json.dumps(data, indent=2))

    # --- Figure ---
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 180,
        "savefig.bbox": "tight",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.55))

    ax = axes[0]
    ax.loglog(contrasts, [max(v, 1e-4) for v in naive_err], "o-", color="#c44e52", lw=2.0, ms=5, label=r"Naive flux readout")
    ax.loglog(contrasts, [max(v, 1e-4) for v in dip_err], "s-", color="#4c72b0", lw=2.0, ms=5, label=r"Dipole readout")
    # reference slope Theta(kappa)
    ref = [naive_err[2] * (c / contrasts[2]) for c in contrasts]
    ax.loglog(contrasts, ref, ":", color="0.35", lw=1.5, label=r"$\propto\kappa$ reference")
    ax.axvspan(10, 1e3, color="#e8f4fa", alpha=0.55, zorder=0)
    ax.text(40, 2e-3, r"intermediate $\kappa$", fontsize=8, color="#1f4e79")
    ax.set_xlabel(r"Inclusion contrast $\kappa$")
    ax.set_ylabel(r"$K_{\mathrm{eff}}$ rel. error from fixed tilt $\varepsilon=0.02$")
    ax.set_title(r"(a) Measured $\Theta(\kappa)$ amplification on U-bridge")
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="upper left", frameon=True)

    ax = axes[1]
    assert g_map is not None and kap_vis is not None and u_vis is not None
    vmax = np.percentile(np.abs(g_map), 99.5)
    im = ax.imshow(g_map.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.contour((kap_vis > 1.5).astype(float).T, levels=[0.5], colors="k", linewidths=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(b) Readout weights $g$ at $\kappa=10^4$ ($\Vert g\Vert_1=\Theta(\kappa)$)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$g_p$")

    ax = axes[2]
    ax.semilogx(contrasts, [100.0 * v for v in clamp_under], "D-", color="#dd8452", lw=2.0, ms=5, label="Clamp + naive flux")
    ax.axhline(-54.0, color="0.3", ls="--", lw=1.2, label=r"Asymptote $\approx -54\%$")
    ax.axvspan(1e4, 1e6, color="#f7e8e8", alpha=0.5, zorder=0)
    ax.text(2e4, -20, r"perfect-conductor limit", fontsize=8, color="#7a2e2e")
    ax.set_xlabel(r"Inclusion contrast $\kappa$")
    ax.set_ylabel(r"Clamp underprediction of $K_{\mathrm{eff}}$ (%)")
    ax.set_title(r"(c) Exact clamp on same U-bridge (real solve)")
    ax.set_ylim(-70, 5)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_empirical_mechanism.{ext}")
    plt.close(fig)
    print("Saved:", out_dir / "fig_empirical_mechanism.pdf")
    print("g_L1 vs kappa:", list(zip(contrasts, [round(v, 1) for v in g1_norms])))
    print("naive err @1e6:", round(naive_err[-1], 4), "dipole:", round(dip_err[-1], 4), "clamp%:", round(100 * clamp_under[-1], 2))
    return data


if __name__ == "__main__":
    main(Path("."))
