"""Float64 harmonic-face RVE dataset generator for the BC-NO paper suite.

Generates:
  - wavy_laminate_N128.pt (non-elliptic curved layers, train=64, test_id=32, test_ood=32)
  - hard_bridge_N128.pt (U-bridges, L-corners, walls at N=128, contrast up to 1e6 OOD)
  - fiber_N64_eval.pt & fiber_N256_eval.pt (cross-resolution zero-shot evaluation)
  - real_ct_ruhr_N128_hc.pt (optional local extraction from Ruhr sandstone 3D CT TIFFs)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from rve_discrete_operator_2d import conservative_flux_columns, harmonic_mean


def fast_interior_solve(kappa: np.ndarray, load_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized scipy sparse assembly + float64 LU solve for harmonic-face PDE."""
    n = kappa.shape[0]
    m = n - 2
    h = 1.0 / (n - 1)
    scale = 1.0 / (h * h)

    ax = np.linspace(0.0, 1.0, n, dtype=np.float64)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    lift = x if load_axis == 0 else y

    kx = harmonic_mean(kappa[:-1, :], kappa[1:, :]) * scale
    ky = harmonic_mean(kappa[:, :-1], kappa[:, 1:]) * scale

    kxm = kx[:-1, 1:-1]
    kxp = kx[1:, 1:-1]
    kym = ky[1:-1, :-1]
    kyp = ky[1:-1, 1:]
    diag = (kxm + kxp + kym + kyp).reshape(-1)

    idx = np.arange(m * m, dtype=np.int64).reshape(m, m)
    rows = [idx.reshape(-1)]
    cols = [idx.reshape(-1)]
    vals = [diag]

    # x-neighbors (axis 0)
    rows.append(idx[1:, :].reshape(-1)); cols.append(idx[:-1, :].reshape(-1)); vals.append(-kxm[1:, :].reshape(-1))
    rows.append(idx[:-1, :].reshape(-1)); cols.append(idx[1:, :].reshape(-1)); vals.append(-kxp[:-1, :].reshape(-1))
    # y-neighbors (axis 1)
    rows.append(idx[:, 1:].reshape(-1)); cols.append(idx[:, :-1].reshape(-1)); vals.append(-kym[:, 1:].reshape(-1))
    rows.append(idx[:, :-1].reshape(-1)); cols.append(idx[:, 1:].reshape(-1)); vals.append(-kyp[:, :-1].reshape(-1))

    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m * m, m * m)).tocsr()

    # RHS = -A_full(lift) on interior nodes
    u0 = lift
    rhs_2d = -(
        (kxm + kxp + kym + kyp) * u0[1:-1, 1:-1]
        - kxm * u0[:-2, 1:-1]
        - kxp * u0[2:, 1:-1]
        - kym * u0[1:-1, :-2]
        - kyp * u0[1:-1, 2:]
    )
    v_in = spla.spsolve(A, rhs_2d.reshape(-1)).reshape(m, m)
    v = np.zeros((n, n), dtype=np.float64)
    v[1:-1, 1:-1] = v_in
    return v, lift


def pack_cases(kappas: list[np.ndarray], axes: list[int]) -> dict[str, torch.Tensor]:
    B = len(kappas)
    n = kappas[0].shape[0]
    a_arr = np.zeros((B, n, n, 4), dtype=np.float32)
    v_arr = np.zeros((B, n, n, 1), dtype=np.float32)
    l_arr = np.zeros((B, n, n, 1), dtype=np.float32)
    u_f64 = np.zeros((B, n, n, 1), dtype=np.float64)
    k_f64 = np.zeros((B, n, n, 1), dtype=np.float64)

    for i, (kap, ax) in enumerate(zip(kappas, axes)):
        v, lift = fast_interior_solve(kap, ax)
        a_arr[i, ..., 2] = kap.astype(np.float32)
        v_arr[i, ..., 0] = v.astype(np.float32)
        l_arr[i, ..., 0] = lift.astype(np.float32)
        u_f64[i, ..., 0] = lift + v
        k_f64[i, ..., 0] = kap

    k_cols = conservative_flux_columns(torch.from_numpy(u_f64), torch.from_numpy(k_f64), kappa_channel=0).float()
    return {
        "a": torch.from_numpy(a_arr),
        "u": torch.from_numpy(v_arr),
        "affine_lift": torch.from_numpy(l_arr),
        "load_axis": torch.tensor(axes, dtype=torch.long),
        "k_eff_columns": k_cols,
    }


def make_fiber_columns(n: int, seed: int, contrast: float, vf: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, n)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    n_cols = int(rng.integers(3, 7))
    n_rows = int(rng.integers(3, 7))
    radius = np.sqrt(vf / (np.pi * n_cols * n_rows)) * float(rng.uniform(0.85, 1.15))
    rx_s = float(rng.uniform(0.75, 1.45))
    ry_s = float(rng.uniform(0.75, 1.45))
    mask = np.zeros((n, n), dtype=bool)
    for c in range(n_cols):
        for r in range(n_rows):
            cx = (c + 0.5) / n_cols + rng.uniform(-0.08, 0.08) / n_cols
            cy = (r + 0.5) / n_rows + rng.uniform(-0.03, 0.03) / n_rows
            rx = radius * rx_s * float(rng.uniform(0.85, 1.15))
            ry = radius * ry_s * float(rng.uniform(0.85, 1.15))
            th = float(rng.uniform(-0.30, 0.30))
            x0, y0 = x - cx, y - cy
            xp = np.cos(th) * x0 + np.sin(th) * y0
            yp = -np.sin(th) * x0 + np.cos(th) * y0
            mask |= (xp / rx) ** 2 + (yp / ry) ** 2 <= 1.0
    return np.where(mask, contrast, 1.0)


def make_wavy_laminate(n: int, seed: int, contrast: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ax = np.linspace(0.0, 1.0, n)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    if rng.random() < 0.5:
        coord, perp = x, y
    else:
        coord, perp = y, x
    n_bands = int(rng.integers(2, 5))
    amp = float(rng.uniform(0.03, 0.09))
    freq = float(rng.integers(1, 4)) * 2.0 * np.pi
    phase = float(rng.uniform(0, 2 * np.pi))
    warped = coord + amp * np.sin(freq * perp + phase)
    width = float(rng.uniform(0.30, 0.55)) / n_bands
    mask = np.zeros((n, n), dtype=bool)
    for b in range(n_bands):
        center = (b + 0.5) / n_bands + float(rng.uniform(-0.04, 0.04))
        mask |= np.abs(warped - center) <= 0.5 * width
    return np.where(mask, contrast, 1.0)


def make_hard_topology(n: int, seed: int, contrast: float, kind: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kappa = np.ones((n, n), dtype=np.float64)
    if kind == "u_bridge":
        row0 = int(rng.integers(int(0.1 * n), int(0.25 * n)))
        col0 = int(rng.integers(int(0.1 * n), int(0.25 * n)))
        arm_h = int(rng.integers(int(0.45 * n), int(0.68 * n)))
        arm_w = int(rng.integers(max(3, int(0.06 * n)), int(0.14 * n)))
        gap = int(rng.integers(int(0.15 * n), int(0.42 * n)))
        bridge = int(rng.integers(max(2, int(0.04 * n)), int(0.22 * n)))
        top = min(n - 2, row0 + arm_h - bridge)
        r_end = min(n - 2, row0 + arm_h)
        right0 = min(n - 2 - arm_w, col0 + arm_w + gap)
        kappa[row0:top, col0 : col0 + arm_w] = contrast
        kappa[row0:top, right0 : right0 + arm_w] = contrast
        kappa[top:r_end, col0 : right0 + arm_w] = contrast
    elif kind == "l_shape":
        r0 = int(rng.integers(int(0.15 * n), int(0.35 * n)))
        c0 = int(rng.integers(int(0.15 * n), int(0.35 * n)))
        h1 = int(rng.integers(int(0.30 * n), int(0.55 * n)))
        w1 = int(rng.integers(int(0.08 * n), int(0.16 * n)))
        w2 = int(rng.integers(int(0.30 * n), int(0.55 * n)))
        h2 = int(rng.integers(int(0.08 * n), int(0.16 * n)))
        kappa[r0 : min(n - 2, r0 + h1), c0 : min(n - 2, c0 + w1)] = contrast
        kappa[min(n - 3, r0 + h1 - h2) : min(n - 2, r0 + h1), c0 : min(n - 2, c0 + w2)] = contrast
    else:  # near_wall_box
        h_box = int(rng.integers(int(0.25 * n), int(0.45 * n)))
        w_box = int(rng.integers(int(0.20 * n), int(0.38 * n)))
        r0 = int(rng.integers(int(0.15 * n), n - 2 - h_box))
        c0 = int(rng.integers(2, n - 2 - w_box))
        kappa[r0 : r0 + h_box, c0 : c0 + w_box] = contrast
    return kappa


def build_split_dict(train_k, train_ax, id_k, id_ax, ood_k, ood_ax) -> dict[str, torch.Tensor]:
    tr = pack_cases(train_k, train_ax)
    te = pack_cases(id_k, id_ax)
    ood = pack_cases(ood_k, ood_ax)
    return {
        "a_train": tr["a"], "u_train": tr["u"], "affine_lift_train": tr["affine_lift"],
        "load_axis_train": tr["load_axis"], "k_eff_columns_train": tr["k_eff_columns"],
        "a_test": te["a"], "u_test": te["u"], "affine_lift_test": te["affine_lift"],
        "load_axis_test": te["load_axis"], "k_eff_columns_test": te["k_eff_columns"],
        "a_ood": ood["a"], "u_ood": ood["u"], "affine_lift_ood": ood["affine_lift"],
        "load_axis_ood": ood["load_axis"], "k_eff_columns_ood": ood["k_eff_columns"],
    }


def unpack_compact_datasets(out_dir: Path) -> None:
    npz_path = out_dir / "rve_specs_compact.npz"
    if not npz_path.exists():
        return
    z = np.load(npz_path)

    def unpack_split(prefix: str, sp: str):
        shape = tuple(int(v) for v in z[f"{prefix}_{sp}_shape"])
        n_bits = int(np.prod(shape))
        masks = np.unpackbits(z[f"{prefix}_{sp}_bits"])[:n_bits].reshape(shape).astype(bool)
        c_vals = z[f"{prefix}_{sp}_c"]
        ax_vals = [int(a) for a in z[f"{prefix}_{sp}_ax"]]
        kappas = [np.where(masks[i], float(c_vals[i]), 1.0) for i in range(shape[0])]
        return pack_cases(kappas, ax_vals)

    fc_id_p = out_dir / "rve_fiber_columns_N128_f64.pt"
    if not fc_id_p.exists():
        t0 = time.time()
        tr = unpack_split("fcid", "train")
        te = unpack_split("fcid", "test")
        torch.save({
            "a_train": tr["a"], "u_train": tr["u"], "affine_lift_train": tr["affine_lift"],
            "load_axis_train": tr["load_axis"], "k_eff_columns_train": tr["k_eff_columns"],
            "a_test": te["a"], "u_test": te["u"], "affine_lift_test": te["affine_lift"],
            "load_axis_test": te["load_axis"], "k_eff_columns_test": te["k_eff_columns"],
        }, fc_id_p)
        print(f"Generated {fc_id_p.name} ({time.time()-t0:.1f}s)")

    fc_ood_p = out_dir / "rve_fiber_columns_ood_N128_f64.pt"
    if not fc_ood_p.exists():
        t0 = time.time()
        te = unpack_split("fcood", "test")
        torch.save({
            "a_test": te["a"], "u_test": te["u"], "affine_lift_test": te["affine_lift"],
            "load_axis_test": te["load_axis"], "k_eff_columns_test": te["k_eff_columns"],
        }, fc_ood_p)
        print(f"Generated {fc_ood_p.name} ({time.time()-t0:.1f}s)")

    ruhr_p = out_dir / "real_ct_ruhr_N128_hc.pt"
    if not ruhr_p.exists():
        t0 = time.time()
        tr = unpack_split("ruhr", "train")
        te = unpack_split("ruhr", "test")
        torch.save({
            "a_train": tr["a"], "u_train": tr["u"], "affine_lift_train": tr["affine_lift"],
            "load_axis_train": tr["load_axis"], "k_eff_columns_train": tr["k_eff_columns"],
            "a_test": te["a"], "u_test": te["u"], "affine_lift_test": te["affine_lift"],
            "load_axis_test": te["load_axis"], "k_eff_columns_test": te["k_eff_columns"],
        }, ruhr_p)
        print(f"Generated {ruhr_p.name} ({time.time()-t0:.1f}s)")

    ct32_p = out_dir / "real_ct_32.pt"
    if not ct32_p.exists():
        t0 = time.time()
        splits = {}
        for sp in ("train", "val", "locked_test"):
            kaps = z[f"ct32_{sp}_kappa"].astype(np.float64)
            axes = [int(a) for a in z[f"ct32_{sp}_ax"]]
            p = pack_cases([kaps[i] for i in range(kaps.shape[0])], axes)
            splits[sp] = {
                "kappa": p["a"][..., 2],
                "fluctuation": p["u"][..., 0],
                "affine_lift": p["affine_lift"][..., 0],
                "field": p["affine_lift"][..., 0] + p["u"][..., 0],
                "keff_column": p["k_eff_columns"],
                "load_axis": p["load_axis"],
            }
        torch.save({"splits": splits}, ct32_p)
        print(f"Generated {ct32_p.name} ({time.time()-t0:.1f}s)")


def generate_synthetic_benchmarks(out_dir: Path) -> None:
    unpack_compact_datasets(out_dir)
    t0 = time.time()
    rng = np.random.default_rng(2026)

    # 1. wavy_laminate_N128.pt
    wavy_p = out_dir / "wavy_laminate_N128.pt"
    if not wavy_p.exists():
        tr_k, tr_ax, id_k, id_ax, od_k, od_ax = [], [], [], [], [], []
        for s in range(32):
            c = float(np.exp(rng.uniform(np.log(10), np.log(80)))) if rng.random() < 0.7 else 1.0 / float(np.exp(rng.uniform(np.log(10), np.log(80))))
            kap = make_wavy_laminate(128, 1000 + s, c)
            for ax in (0, 1): tr_k.append(kap); tr_ax.append(ax)
        for s in range(16):
            c = float(np.exp(rng.uniform(np.log(10), np.log(80)))) if rng.random() < 0.7 else 1.0 / float(np.exp(rng.uniform(np.log(10), np.log(80))))
            kap = make_wavy_laminate(128, 2000 + s, c)
            for ax in (0, 1): id_k.append(kap); id_ax.append(ax)
        for s in range(16):
            c = float(np.exp(rng.uniform(np.log(150), np.log(500)))) if rng.random() < 0.75 else 1.0 / float(np.exp(rng.uniform(np.log(150), np.log(500))))
            kap = make_wavy_laminate(128, 3000 + s, c)
            for ax in (0, 1): od_k.append(kap); od_ax.append(ax)
        torch.save(build_split_dict(tr_k, tr_ax, id_k, id_ax, od_k, od_ax), wavy_p)
        print(f"Generated {wavy_p.name} ({time.time()-t0:.1f}s)")

    # 2. hard_bridge_N128.pt (U-bridge, L-shape, near-wall boxes; contrast 20..1e4 train/id -> 1e6 OOD!)
    hb_p = out_dir / "hard_bridge_N128.pt"
    if not hb_p.exists():
        t1 = time.time()
        tr_k, tr_ax, id_k, id_ax, od_k, od_ax = [], [], [], [], [], []
        kinds = ("near_wall_box", "u_bridge", "l_shape")
        for s in range(32):
            c = float(rng.choice([20.0, 100.0, 1000.0, 10000.0]))
            kap = make_hard_topology(128, 4000 + s, c, kinds[s % 3])
            for ax in (0, 1): tr_k.append(kap); tr_ax.append(ax)
        for s in range(12):
            c = float(rng.choice([100.0, 1000.0, 10000.0]))
            kap = make_hard_topology(128, 5000 + s, c, kinds[s % 3])
            for ax in (0, 1): id_k.append(kap); id_ax.append(ax)
        for s in range(12):
            kap = make_hard_topology(128, 6000 + s, 1.0e6, kinds[s % 3])
            for ax in (0, 1): od_k.append(kap); od_ax.append(ax)
        torch.save(build_split_dict(tr_k, tr_ax, id_k, id_ax, od_k, od_ax), hb_p)
        print(f"Generated {hb_p.name} ({time.time()-t1:.1f}s)")

    # 3. Cross-resolution evaluation sets: N=64 and N=256 (16 samples each)
    for res in (64, 256):
        rp = out_dir / f"fiber_N{res}_eval.pt"
        if not rp.exists():
            t2 = time.time()
            rk, rax = [], []
            for s in range(8):
                c = float(np.exp(rng.uniform(np.log(15), np.log(80)))) if s % 4 != 0 else 1.0 / float(np.exp(rng.uniform(np.log(15), np.log(80))))
                vf = float(rng.uniform(0.35, 0.55))
                kap = make_fiber_columns(res, 7000 + s, c, vf)
                for ax in (0, 1): rk.append(kap); rax.append(ax)
            d = pack_cases(rk, rax)
            torch.save({
                "a_test": d["a"], "u_test": d["u"], "affine_lift_test": d["affine_lift"],
                "load_axis_test": d["load_axis"], "k_eff_columns_test": d["k_eff_columns"],
            }, rp)
            print(f"Generated {rp.name} ({time.time()-t2:.1f}s)")


def generate_ruhr_ct_n128(tiff_dir: Path, out_path: Path) -> None:
    from PIL import Image
    files = sorted(tiff_dir.glob("*.tif"))
    assert len(files) >= 700
    t0 = time.time()
    # Train slabs z in [80, 320], Test slabs z in [450, 720] (strictly disjoint!)
    def extract_patch(z: int, y0: int, x0: int) -> np.ndarray:
        sl = np.asarray(Image.open(files[z]))[y0 : y0 + 128, x0 : x0 + 128]
        # Pore {1, 2} -> 1.0; Solid mineral grains {3..8} -> 1000.0 (high contrast rock matrix!)
        return np.where(sl <= 2, 1.0, 1000.0)

    rng = np.random.default_rng(99)
    tr_k, tr_ax, te_k, te_ax = [], [], [], []
    for _ in range(24):
        z = int(rng.integers(80, 320))
        y0 = int(rng.integers(50, 620))
        x0 = int(rng.integers(50, 620))
        kap = extract_patch(z, y0, x0)
        for ax in (0, 1): tr_k.append(kap); tr_ax.append(ax)
    for _ in range(12):
        z = int(rng.integers(450, 720))
        y0 = int(rng.integers(50, 620))
        x0 = int(rng.integers(50, 620))
        kap = extract_patch(z, y0, x0)
        for ax in (0, 1): te_k.append(kap); te_ax.append(ax)

    tr = pack_cases(tr_k, tr_ax)
    te = pack_cases(te_k, te_ax)
    torch.save({
        "a_train": tr["a"], "u_train": tr["u"], "affine_lift_train": tr["affine_lift"],
        "load_axis_train": tr["load_axis"], "k_eff_columns_train": tr["k_eff_columns"],
        "a_test": te["a"], "u_test": te["u"], "affine_lift_test": te["affine_lift"],
        "load_axis_test": te["load_axis"], "k_eff_columns_test": te["k_eff_columns"],
    }, out_path)
    print(f"Generated {out_path.name} ({time.time()-t0:.1f}s): train={tr['a'].shape[0]}, test={te['a'].shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--ruhr_tiff_dir", type=str, default="")
    parser.add_argument("--real_ct_only", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if args.ruhr_tiff_dir:
        generate_ruhr_ct_n128(Path(args.ruhr_tiff_dir), out_dir / "real_ct_ruhr_N128_hc.pt")
    if not args.real_ct_only:
        generate_synthetic_benchmarks(out_dir)


if __name__ == "__main__":
    main()
