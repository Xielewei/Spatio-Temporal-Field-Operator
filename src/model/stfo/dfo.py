"""Dual-Path Field Operator (DFO).

Learns the spatio-temporal evolution kernel, factorized into a
global spectral path (Eq. (6)) and a state-dependent interaction path
(Eq. (7)), conditioned on the spectral regime state c. Their residual
field update follows Eq. (8).
"""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBranch(nn.Module):
    """Global spectral path via retained 2D Fourier modes."""

    def __init__(self, d_model: int, modes: int = 8):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        scale = 1.0 / d_model
        self.weight = nn.Parameter(scale * torch.randn(modes, modes, d_model, 2))
        self.channel_mixer = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z_grid: torch.Tensor, grid_size: int) -> torch.Tensor:
        B, G, d = z_grid.shape
        H = W = grid_size
        out_dtype = z_grid.dtype
        ctx = (
            torch.cuda.amp.autocast(enabled=False) if z_grid.is_cuda else nullcontext()
        )
        with ctx:
            x = z_grid.float().reshape(B, H, W, d).permute(0, 3, 1, 2)
            xf = torch.fft.rfft2(x, norm="ortho")  # (B, d, H, Wf)

            m = min(self.modes, H, xf.shape[-1])
            out = torch.zeros_like(xf)
            sub = xf[:, :, :m, :m]  # (B, d, m, m)
            w = torch.view_as_complex(self.weight[:m, :m].contiguous())
            mixed = sub * w.permute(2, 0, 1).unsqueeze(0)
            out[:, :, :m, :m] = mixed
            y = torch.fft.irfft2(out, s=(H, W), norm="ortho")  # (B, d, H, W)
        y = y.permute(0, 2, 3, 1).reshape(B, G, d).to(out_dtype)
        return self.channel_mixer(y)


class LinearAttentionBranch(nn.Module):
    """State-dependent interaction path via normalized linear attention."""

    def __init__(self, d_model: int, attn_dim: int = None):
        super().__init__()
        attn_dim = attn_dim or min(64, d_model)
        self.q = nn.Linear(d_model, attn_dim)
        self.k = nn.Linear(d_model, attn_dim)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, z_grid: torch.Tensor) -> torch.Tensor:
        q = self._phi(self.q(z_grid))  # (B, G, d)
        k = self._phi(self.k(z_grid))
        v = self.v(z_grid)
        # linear attention: phi(Q) (phi(K)^T V) with a low-rank key/query space.
        kv = torch.einsum("bgr,bgd->brd", k, v)  # (B, r, d)
        z = torch.einsum("bgr,brd->bgd", q, kv)  # (B, G, d)
        denom = torch.einsum("bgr,br->bg", q, k.sum(1)).unsqueeze(-1) + 1e-6
        return self.proj(z / denom)


class DFO(nn.Module):
    """SRE-conditioned spectral and interaction paths with a residual update."""

    def __init__(
        self,
        d_model: int = 192,
        modes: int = 8,
        cond_dim: int = 6,
        grid_size: int = 16,
        attn_dim: int = None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spectral = SpectralBranch(d_model, modes)
        self.attn = LinearAttentionBranch(d_model, attn_dim)
        self.film = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.bias_field = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z_grid: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Update a (batch, grid points, channels) field using its SRE state."""
        c = torch.nan_to_num(c.float()).to(z_grid.dtype)
        gamma, delta = self.film(c).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        delta = 0.1 * torch.tanh(delta)
        zc = z_grid * (1 + gamma.unsqueeze(1)) + delta.unsqueeze(1)
        spectral_update = self.spectral(zc, self.grid_size)
        interaction_update = self.attn(zc)
        return (
            self.norm(z_grid + spectral_update + interaction_update) + self.bias_field
        )
