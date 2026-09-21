"""Spectral Regime Encoder (SRE).

Computes the normalized spectral centroid alpha, clipped spectral slope beta,
and band-energy ratios eta from the latent field (Section 3.2, Eq. (3)).
"""

from contextlib import nullcontext

import torch
import torch.nn as nn


class SRE(nn.Module):
    def __init__(self, grid_size: int = 16, num_bands: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.num_bands = num_bands
        # conditioning vector c = [alpha, clipped beta, eta_1, ..., eta_B]
        self.cond_dim = 2 + num_bands

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_grid: (B, G, d) latent field, G = grid_size**2.
        Returns:
            c: (B, cond_dim) conditioning vector.
        """
        B, G, d = z_grid.shape
        H = W = self.grid_size
        out_dtype = z_grid.dtype
        ctx = (
            torch.cuda.amp.autocast(enabled=False) if z_grid.is_cuda else nullcontext()
        )
        with ctx:
            field = z_grid.float().mean(dim=-1).reshape(B, H, W)
            spec = torch.fft.rfft2(field, norm="ortho")
            power = spec.real**2 + spec.imag**2

        device = z_grid.device
        ky = torch.fft.fftfreq(H, device=device).abs()
        kx = torch.fft.rfftfreq(W, device=device).abs()
        kyg, kxg = torch.meshgrid(ky, kx, indexing="ij")
        kmag = torch.sqrt(kyg**2 + kxg**2) + 1e-6  # (H, Wf)

        # Radially flatten and fit power-law S ~ |k|^{-beta}.
        kflat = kmag.reshape(-1)
        mask = kflat > 1e-4
        k = kflat[mask]
        log_k = torch.log(kflat[mask])
        p = power.reshape(B, -1)[:, mask] + 1e-8  # (B, K)

        x = log_k - log_k.mean()
        y = torch.log(p) - torch.log(p).mean(dim=1, keepdim=True)
        beta = -((y * x.unsqueeze(0)).sum(dim=1) / ((x * x).sum() + 1e-8))

        # alpha: normalized spectral centroid, a scale indicator distinct
        # from beta instead of duplicating the same fitted slope.
        alpha = (p * k.unsqueeze(0)).sum(dim=1) / (p.sum(dim=1) * k.max() + 1e-8)

        edges = torch.linspace(
            log_k.min(), log_k.max(), self.num_bands + 1, device=device
        )
        ratios = []
        total = p.sum(dim=1) + 1e-8
        for i in range(self.num_bands):
            if i == self.num_bands - 1:
                sel = (log_k >= edges[i]) & (log_k <= edges[i + 1])
            else:
                sel = (log_k >= edges[i]) & (log_k < edges[i + 1])
            ratios.append(p[:, sel].sum(dim=1) / total)

        alpha = alpha.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        ratio = torch.stack(ratios, dim=1)
        c = torch.cat([alpha, beta, ratio], dim=-1)
        c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        c = torch.cat(
            [
                c[:, 0:1].clamp(0.0, 1.0),
                c[:, 1:2].clamp(-10.0, 10.0),
                c[:, 2:].clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        return c.to(out_dtype)
