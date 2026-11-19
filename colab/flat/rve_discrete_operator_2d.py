from __future__ import annotations

import numpy as np
import torch


def harmonic_mean(a, b, eps: float = 1e-12):
    """Harmonic face coefficient for NumPy arrays or torch tensors."""
    if isinstance(a, torch.Tensor):
        return 2.0 * a * b / (a + b).clamp_min(eps)
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    return 2.0 * a_np * b_np / np.maximum(a_np + b_np, eps)


def _as_scalar_field(field: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if field.ndim == 4 and field.shape[-1] == 1:
        return field[..., 0], True
    if field.ndim == 3:
        return field, False
    raise ValueError(f"field must be BHW or BHW1, got {tuple(field.shape)}")


def _as_kappa(kappa_or_input: torch.Tensor, kappa_channel: int) -> torch.Tensor:
    if kappa_or_input.ndim == 4:
        return kappa_or_input[..., kappa_channel].clamp_min(1e-6)
    if kappa_or_input.ndim == 3:
        return kappa_or_input.clamp_min(1e-6)
    raise ValueError(
        "kappa_or_input must be BHW or NHWC, "
        f"got {tuple(kappa_or_input.shape)}"
    )


def dirichlet_interior_mask_like(field: torch.Tensor) -> torch.Tensor:
    scalar, had_channel = _as_scalar_field(field)
    mask = torch.ones_like(scalar)
    mask[:, 0, :] = 0.0
    mask[:, -1, :] = 0.0
    mask[:, :, 0] = 0.0
    mask[:, :, -1] = 0.0
    return mask[..., None] if had_channel else mask


def face_conductivities(
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conductivities on all axis-0 and axis-1 grid faces."""
    kappa = _as_kappa(kappa_or_input, kappa_channel)
    kx = harmonic_mean(kappa[:, :-1, :], kappa[:, 1:, :])
    ky = harmonic_mean(kappa[:, :, :-1], kappa[:, :, 1:])
    return kx, ky


def apply_rve_operator(
    field: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 2,
) -> torch.Tensor:
    """Apply the SPD harmonic-face operator -div(kappa grad) with zero rows on the boundary."""
    u, had_channel = _as_scalar_field(field)
    kx, ky = face_conductivities(kappa_or_input, kappa_channel=kappa_channel)
    h = 1.0 / max(u.shape[-1] - 1, 1)
    out = torch.zeros_like(u)
    center = u[:, 1:-1, 1:-1]
    kxm = kx[:, :-1, 1:-1]
    kxp = kx[:, 1:, 1:-1]
    kym = ky[:, 1:-1, :-1]
    kyp = ky[:, 1:-1, 1:]
    out[:, 1:-1, 1:-1] = (
        (kxm + kxp + kym + kyp) * center
        - kxm * u[:, :-2, 1:-1]
        - kxp * u[:, 2:, 1:-1]
        - kym * u[:, 1:-1, :-2]
        - kyp * u[:, 1:-1, 2:]
    ) / (h * h)
    return out[..., None] if had_channel else out


def harmonic_face_submatrix(
    kappa: torch.Tensor,
    dof_rows: torch.Tensor,
    dof_cols: torch.Tensor,
) -> torch.Tensor:
    """Exact A[U,U] of the global harmonic-face operator with exterior Dirichlet.

    For any field supported on the listed interior cells, ``A_U @ u_U``
    equals the global ``apply_rve_operator`` restricted to those cells.
    Faces that leave U contribute only to the diagonal (neighbor frozen at 0).
    """
    if kappa.ndim != 2:
        raise ValueError(f"kappa must be HW, got {tuple(kappa.shape)}")
    if dof_rows.shape != dof_cols.shape or dof_rows.ndim != 1:
        raise ValueError("dof_rows and dof_cols must be 1-D and aligned")
    height, width = kappa.shape
    count = int(dof_rows.numel())
    if count == 0:
        return kappa.new_zeros((0, 0))
    location = torch.full(
        (height, width),
        -1,
        device=kappa.device,
        dtype=torch.long,
    )
    location[dof_rows, dof_cols] = torch.arange(
        count,
        device=kappa.device,
        dtype=torch.long,
    )
    kx, ky = face_conductivities(kappa[None])
    scale = float(max(width - 1, 1) ** 2)
    matrix = kappa.new_zeros((count, count))

    def _accumulate(
        conductance: torch.Tensor,
        row_a: torch.Tensor,
        col_a: torch.Tensor,
        row_b: torch.Tensor,
        col_b: torch.Tensor,
    ) -> None:
        left = location[row_a, col_a]
        right = location[row_b, col_b]
        weight = conductance * scale
        both = (left >= 0) & (right >= 0)
        only_left = (left >= 0) & (right < 0)
        only_right = (right >= 0) & (left < 0)
        if bool(both.any()):
            ii = left[both]
            jj = right[both]
            cc = weight[both]
            matrix.index_put_((ii, ii), cc, accumulate=True)
            matrix.index_put_((jj, jj), cc, accumulate=True)
            matrix.index_put_((ii, jj), -cc, accumulate=True)
            matrix.index_put_((jj, ii), -cc, accumulate=True)
        if bool(only_left.any()):
            matrix.index_put_(
                (left[only_left], left[only_left]),
                weight[only_left],
                accumulate=True,
            )
        if bool(only_right.any()):
            matrix.index_put_(
                (right[only_right], right[only_right]),
                weight[only_right],
                accumulate=True,
            )

    row_face, col_face = torch.meshgrid(
        torch.arange(height - 1, device=kappa.device),
        torch.arange(width, device=kappa.device),
        indexing="ij",
    )
    _accumulate(kx[0], row_face, col_face, row_face + 1, col_face)
    row_face, col_face = torch.meshgrid(
        torch.arange(height, device=kappa.device),
        torch.arange(width - 1, device=kappa.device),
        indexing="ij",
    )
    _accumulate(ky[0], row_face, col_face, row_face, col_face + 1)
    return matrix


def operator_diagonal(
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 2,
) -> torch.Tensor:
    kappa = _as_kappa(kappa_or_input, kappa_channel)
    kx, ky = face_conductivities(kappa, kappa_channel=kappa_channel)
    h = 1.0 / max(kappa.shape[-1] - 1, 1)
    diagonal = torch.ones_like(kappa)
    diagonal[:, 1:-1, 1:-1] = (
        kx[:, :-1, 1:-1]
        + kx[:, 1:, 1:-1]
        + ky[:, 1:-1, :-1]
        + ky[:, 1:-1, 1:]
    ) / (h * h)
    return diagonal[..., None]


def harmonic_face_fluxes(
    field: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q=-kappa grad(u) on all grid faces."""
    u, _ = _as_scalar_field(field)
    kx, ky = face_conductivities(kappa_or_input, kappa_channel=kappa_channel)
    h = 1.0 / max(u.shape[-1] - 1, 1)
    qx = -kx * (u[:, 1:, :] - u[:, :-1, :]) / h
    qy = -ky * (u[:, :, 1:] - u[:, :, :-1]) / h
    return qx, qy


def conservative_flux_columns(
    field: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 2,
) -> torch.Tensor:
    """Effective tensor columns from interior face averages."""
    qx, qy = harmonic_face_fluxes(
        field,
        kappa_or_input,
        kappa_channel=kappa_channel,
    )
    qx_inner = qx[:, :, 1:-1]
    qy_inner = qy[:, 1:-1, :]
    return torch.stack(
        [-qx_inner.mean(dim=(1, 2)), -qy_inner.mean(dim=(1, 2))],
        dim=-1,
    )


def energy_inner_product(
    left: torch.Tensor,
    right: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 2,
) -> torch.Tensor:
    """Per-sample discrete energy <left, A right> h^2."""
    left_scalar, _ = _as_scalar_field(left)
    applied, _ = _as_scalar_field(
        apply_rve_operator(
            right,
            kappa_or_input,
            kappa_channel=kappa_channel,
        )
    )
    h = 1.0 / max(left_scalar.shape[-1] - 1, 1)
    return (left_scalar * applied).sum(dim=(1, 2)) * (h * h)


def apply_rve_operator_numpy(field: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    field_t = torch.from_numpy(np.asarray(field, dtype=np.float64))[None]
    kappa_t = torch.from_numpy(np.asarray(kappa, dtype=np.float64))[None]
    return apply_rve_operator(field_t, kappa_t)[0].numpy()


def conservative_flux_columns_numpy(
    field: np.ndarray,
    kappa: np.ndarray,
) -> tuple[float, float]:
    field_t = torch.from_numpy(np.asarray(field, dtype=np.float64))[None]
    kappa_t = torch.from_numpy(np.asarray(kappa, dtype=np.float64))[None]
    columns = conservative_flux_columns(field_t, kappa_t)[0]
    # Return physical mean flux q, while conservative_flux_columns returns -mean(q).
    return -float(columns[0]), -float(columns[1])


__all__ = [
    "apply_rve_operator",
    "apply_rve_operator_numpy",
    "conservative_flux_columns",
    "conservative_flux_columns_numpy",
    "dirichlet_interior_mask_like",
    "energy_inner_product",
    "face_conductivities",
    "harmonic_face_fluxes",
    "harmonic_face_submatrix",
    "harmonic_mean",
    "operator_diagonal",
]
