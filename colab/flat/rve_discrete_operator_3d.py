from __future__ import annotations

import torch

from rve_discrete_operator_2d import harmonic_mean


def _as_scalar_field(field: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if field.ndim == 5 and field.shape[-1] == 1:
        return field[..., 0], True
    if field.ndim == 4:
        return field, False
    raise ValueError(f"field must be BDHW or BDHW1, got {tuple(field.shape)}")


def _as_kappa(
    kappa_or_input: torch.Tensor,
    kappa_channel: int,
) -> torch.Tensor:
    if kappa_or_input.ndim == 5:
        return kappa_or_input[..., kappa_channel].clamp_min(1e-6)
    if kappa_or_input.ndim == 4:
        return kappa_or_input.clamp_min(1e-6)
    raise ValueError(
        "kappa_or_input must be BDHW or BDHWC, "
        f"got {tuple(kappa_or_input.shape)}"
    )


def dirichlet_interior_mask_like_3d(field: torch.Tensor) -> torch.Tensor:
    scalar, had_channel = _as_scalar_field(field)
    mask = torch.ones_like(scalar)
    mask[:, 0] = 0.0
    mask[:, -1] = 0.0
    mask[:, :, 0] = 0.0
    mask[:, :, -1] = 0.0
    mask[:, :, :, 0] = 0.0
    mask[:, :, :, -1] = 0.0
    return mask[..., None] if had_channel else mask


def face_conductivities_3d(
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kappa = _as_kappa(kappa_or_input, kappa_channel)
    kx = harmonic_mean(kappa[:, :-1], kappa[:, 1:])
    ky = harmonic_mean(kappa[:, :, :-1], kappa[:, :, 1:])
    kz = harmonic_mean(kappa[:, :, :, :-1], kappa[:, :, :, 1:])
    return kx, ky, kz


def apply_rve_operator_3d(
    field: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 3,
) -> torch.Tensor:
    """Apply -div(kappa grad) with harmonic faces and zero boundary rows."""
    u, had_channel = _as_scalar_field(field)
    kx, ky, kz = face_conductivities_3d(
        kappa_or_input,
        kappa_channel=kappa_channel,
    )
    h = 1.0 / max(u.shape[-1] - 1, 1)
    out = torch.zeros_like(u)
    center = u[:, 1:-1, 1:-1, 1:-1]
    kxm = kx[:, :-1, 1:-1, 1:-1]
    kxp = kx[:, 1:, 1:-1, 1:-1]
    kym = ky[:, 1:-1, :-1, 1:-1]
    kyp = ky[:, 1:-1, 1:, 1:-1]
    kzm = kz[:, 1:-1, 1:-1, :-1]
    kzp = kz[:, 1:-1, 1:-1, 1:]
    out[:, 1:-1, 1:-1, 1:-1] = (
        (kxm + kxp + kym + kyp + kzm + kzp) * center
        - kxm * u[:, :-2, 1:-1, 1:-1]
        - kxp * u[:, 2:, 1:-1, 1:-1]
        - kym * u[:, 1:-1, :-2, 1:-1]
        - kyp * u[:, 1:-1, 2:, 1:-1]
        - kzm * u[:, 1:-1, 1:-1, :-2]
        - kzp * u[:, 1:-1, 1:-1, 2:]
    ) / (h * h)
    return out[..., None] if had_channel else out


def operator_diagonal_3d(
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 3,
) -> torch.Tensor:
    kappa = _as_kappa(kappa_or_input, kappa_channel)
    kx, ky, kz = face_conductivities_3d(kappa)
    h = 1.0 / max(kappa.shape[-1] - 1, 1)
    diagonal = torch.ones_like(kappa)
    diagonal[:, 1:-1, 1:-1, 1:-1] = (
        kx[:, :-1, 1:-1, 1:-1]
        + kx[:, 1:, 1:-1, 1:-1]
        + ky[:, 1:-1, :-1, 1:-1]
        + ky[:, 1:-1, 1:, 1:-1]
        + kz[:, 1:-1, 1:-1, :-1]
        + kz[:, 1:-1, 1:-1, 1:]
    ) / (h * h)
    return diagonal[..., None]


def harmonic_face_fluxes_3d(
    field: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, _ = _as_scalar_field(field)
    kx, ky, kz = face_conductivities_3d(
        kappa_or_input,
        kappa_channel=kappa_channel,
    )
    h = 1.0 / max(u.shape[-1] - 1, 1)
    qx = -kx * (u[:, 1:] - u[:, :-1]) / h
    qy = -ky * (u[:, :, 1:] - u[:, :, :-1]) / h
    qz = -kz * (u[:, :, :, 1:] - u[:, :, :, :-1]) / h
    return qx, qy, qz


def conservative_flux_columns_3d(
    field: torch.Tensor,
    kappa_or_input: torch.Tensor,
    *,
    kappa_channel: int = 3,
) -> torch.Tensor:
    qx, qy, qz = harmonic_face_fluxes_3d(
        field,
        kappa_or_input,
        kappa_channel=kappa_channel,
    )
    return torch.stack(
        [
            -qx[:, :, 1:-1, 1:-1].mean(dim=(1, 2, 3)),
            -qy[:, 1:-1, :, 1:-1].mean(dim=(1, 2, 3)),
            -qz[:, 1:-1, 1:-1, :].mean(dim=(1, 2, 3)),
        ],
        dim=-1,
    )


__all__ = [
    "apply_rve_operator_3d",
    "conservative_flux_columns_3d",
    "dirichlet_interior_mask_like_3d",
    "face_conductivities_3d",
    "harmonic_face_fluxes_3d",
    "operator_diagonal_3d",
]
