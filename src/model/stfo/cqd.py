"""Coordinate Query Decoder (CQD).

Evaluates the future field representation at arbitrary query coordinates,
including unseen sensors (Eq. (9)). Gathers the local field by the same
kernel interpolation used in CFE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cfe import _inv_softplus


class CQD(nn.Module):
    def __init__(
        self,
        d_model: int = 192,
        horizon: int = 12,
        grid_size: int = 16,
        init_bandwidth: float = 0.1,
        query_chunk_size: int = 256,
        checkpoint_chunks: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.horizon = horizon
        self.query_chunk_size = query_chunk_size
        self.checkpoint_chunks = checkpoint_chunks
        self.raw_sigma = nn.Parameter(torch.tensor(_inv_softplus(init_bandwidth)))
        self.register_buffer("grid", self._build_grid(grid_size), persistent=False)
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    @property
    def sigma(self):
        return F.softplus(self.raw_sigma) + 1e-4

    @staticmethod
    def _build_grid(grid_size):
        g = torch.linspace(0.0, 1.0, grid_size)
        gy, gx = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    def _grid(self, device):
        return self.grid.to(device=device)

    def _decode_chunk(self, h_grid: torch.Tensor, query_coords: torch.Tensor):
        device = h_grid.device
        grid = self._grid(device)  # (G, 2)
        disp = query_coords.unsqueeze(1) - grid.unsqueeze(0)  # (q, G, 2)
        dist2 = (disp**2).sum(-1)
        w = torch.softmax(-dist2 / (2 * self.sigma**2), dim=-1)  # (q, G)
        h_local = torch.einsum("qg,bgd->bqd", w, h_grid)  # (B, q, d)
        return self.readout(h_local)

    def forward(self, h_grid: torch.Tensor, query_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_grid:       (B, G, d) future field representation.
            query_coords: (Nq, 2) query coordinates (may be unseen sensors).
        Returns:
            pred: (B, Nq, horizon) predictions at the query coordinates.
        """
        n_query = query_coords.shape[0]
        use_checkpoint = (
            self.checkpoint_chunks and self.training and torch.is_grad_enabled()
        )
        chunks = []

        for start in range(0, n_query, self.query_chunk_size):
            q = query_coords[start : start + self.query_chunk_size]
            if use_checkpoint:
                pred = checkpoint(
                    self._decode_chunk,
                    h_grid,
                    q,
                    use_reentrant=False,
                )
            else:
                pred = self._decode_chunk(h_grid, q)
            chunks.append(pred)

        return torch.cat(chunks, dim=1)  # (B, Nq, horizon)
