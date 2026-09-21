"""Coordinate Field Encoder (CFE).

Lifts scattered sensor histories into a continuous latent field via
kernel-weighted implicit representations with Fourier coordinate encodings.
Implements the coordinate lifting in Eq. (1) of the paper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """Random Fourier feature encoding rho(x) = [cos(2 pi B x), sin(2 pi B x)]."""

    def __init__(self, in_dim: int, num_features: int = 64, scale: float = 2.0):
        super().__init__()
        # B is fixed (not learned), drawn once from a Gaussian.
        B = torch.randn(num_features, in_dim) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        proj = 2.0 * math.pi * x @ self.B.t()  # (..., num_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class TemporalEncoder(nn.Module):
    """Encode each sensor's history into a shared latent representation."""

    def __init__(self, hist_len: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        return self.net(hist)


class CFE(nn.Module):
    """Coordinate Field Encoder.

    Reconstructs a continuous latent field z(x) on a regular grid from
    scattered sensor observations via Nadaraya-Watson kernel interpolation.
    """

    def __init__(
        self,
        hist_len: int,
        d_model: int = 192,
        num_fourier: int = 64,
        fourier_scale: float = 2.0,
        grid_size: int = 16,
        init_bandwidth: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.temporal = TemporalEncoder(hist_len, d_model)
        self.register_buffer("grid", self._build_grid(grid_size), persistent=False)
        self.coord = FourierFeatures(2, num_fourier, fourier_scale)
        post_in = d_model + self.coord.out_dim
        self.post = nn.Sequential(
            nn.Linear(post_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Learnable kernel bandwidth (sigma), kept positive via softplus.
        self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(init_bandwidth)))

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + 1e-4

    @staticmethod
    def _build_grid(grid_size: int) -> torch.Tensor:
        g = torch.linspace(0.0, 1.0, grid_size)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (G, 2)

    def make_grid(self, device) -> torch.Tensor:
        return self.grid.to(device=device)

    def _lift(self, z_i, coords, grid):
        # Nadaraya-Watson interpolation of encoded sensor codes.
        disp = grid.unsqueeze(1) - coords.unsqueeze(0)  # (G, N, 2)
        dist2 = (disp**2).sum(-1)  # (G, N)
        w = torch.softmax(-dist2 / (2.0 * self.sigma**2), dim=-1)
        z_grid = torch.einsum("gn,bnd->bgd", w, z_i)  # (B, G, d)

        gamma = self.coord(grid).unsqueeze(0).expand(z_i.shape[0], -1, -1)
        return self.post(torch.cat([gamma, z_grid], dim=-1))

    def forward(self, hist: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hist:   (B, N, T_h) sensor histories.
            coords: (N, 2) sensor coordinates in [0, 1]^2.
        Returns:
            z_grid: (B, G, d) continuous latent field on the regular grid,
                    where G = grid_size**2.
        """
        z_i = self.temporal(hist)  # (B, N, d)
        grid = self.make_grid(hist.device)  # (G, 2)
        return self._lift(z_i, coords, grid)
