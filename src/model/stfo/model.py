"""STFO: CFE -> SRE -> DFO -> CQD with a trainable local-history branch."""

import torch
import torch.nn as nn

from .cfe import CFE
from .cqd import CQD
from .dfo import DFO
from .sre import SRE


class STFOConfig:
    def __init__(self, args):
        model = getattr(args, "model", {}) or {}
        self.in_channel = model.get("in_channel", 12)
        self.out_channel = model.get("out_channel", getattr(args, "y_len", 12))
        self.hidden = model.get("hidden_channel", 128)
        self.grid_size = model.get("grid_size", 8)
        self.modes = model.get("modes", 4)
        self.num_fourier = model.get("num_fourier", 16)
        self.fourier_scale = model.get("fourier_scale", 2.0)
        self.num_bands = model.get("num_bands", 3)
        self.cfe_bandwidth = model.get("cfe_bandwidth", 0.1)
        self.cqd_bandwidth = model.get("cqd_bandwidth", 0.1)
        self.dfo_layers = model.get("dfo_layers", 1)
        self.attn_dim = model.get("attn_dim", min(64, self.hidden))
        self.query_chunk_size = model.get("query_chunk_size", 512)
        self.checkpoint_chunks = model.get("checkpoint_chunks", False)
        self.use_skip = model.get("use_skip", True)


class STFO(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cfg = STFOConfig(args)

        self.cfe = CFE(
            hist_len=self.cfg.in_channel,
            d_model=self.cfg.hidden,
            num_fourier=self.cfg.num_fourier,
            fourier_scale=self.cfg.fourier_scale,
            grid_size=self.cfg.grid_size,
            init_bandwidth=self.cfg.cfe_bandwidth,
        )
        self.sre = SRE(grid_size=self.cfg.grid_size, num_bands=self.cfg.num_bands)
        self.dfo_layers = nn.ModuleList(
            [
                DFO(
                    d_model=self.cfg.hidden,
                    modes=self.cfg.modes,
                    cond_dim=self.sre.cond_dim,
                    grid_size=self.cfg.grid_size,
                    attn_dim=self.cfg.attn_dim,
                )
                for _ in range(self.cfg.dfo_layers)
            ]
        )
        self.cqd = CQD(
            d_model=self.cfg.hidden,
            horizon=self.cfg.out_channel,
            grid_size=self.cfg.grid_size,
            init_bandwidth=self.cfg.cqd_bandwidth,
            query_chunk_size=self.cfg.query_chunk_size,
            checkpoint_chunks=self.cfg.checkpoint_chunks,
        )
        self.skip = (
            nn.Linear(self.cfg.in_channel, self.cfg.out_channel)
            if self.cfg.use_skip
            else None
        )
        self._initialize_skip()

    def _initialize_skip(self):
        if self.skip is None:
            return
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        last = min(self.cfg.in_channel - 1, self.skip.weight.shape[1] - 1)
        with torch.no_grad():
            for horizon in range(self.cfg.out_channel):
                self.skip.weight[horizon, last] = 1.0

    def _coords(self, adj, device):
        coords = getattr(self.args, "coords", None)
        expected_shape = (adj.shape[0], 2)
        if isinstance(coords, torch.Tensor) and tuple(coords.shape) == expected_shape:
            return coords.to(device=device, dtype=torch.float32)
        actual = None if coords is None else tuple(coords.shape)
        raise ValueError(
            "sensor coordinates are missing or misaligned: "
            f"expected {expected_shape}, received {actual}"
        )

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger = getattr(self.args, "logger", None)
        msg = f"STFO Total params: {total} | Trainable: {trainable}"
        (logger.info if logger else print)(msg)
        return total, trainable

    def forward_fields(self, hist, coords, query_coords):
        hist = torch.nan_to_num(hist.float(), nan=0.0, posinf=0.0, neginf=0.0)
        coords = coords.to(hist.device, dtype=torch.float32)
        query_coords = query_coords.to(hist.device, dtype=torch.float32)

        z_grid = self.cfe(hist, coords)
        sre_state = self.sre(z_grid)
        field = z_grid
        for dfo in self.dfo_layers:
            field = dfo(field, sre_state)
        pred = self.cqd(field, query_coords)
        return pred, {
            "z_grid": z_grid,
            "sre_state": sre_state,
        }

    def forward(self, data, adj):
        num_nodes = adj.shape[0]
        hist = data.x.reshape(-1, num_nodes, self.cfg.in_channel)
        hist = torch.nan_to_num(hist.float(), nan=0.0, posinf=0.0, neginf=0.0)
        coords = self._coords(adj, hist.device)

        pred, _ = self.forward_fields(hist, coords, coords)
        if self.skip is not None:
            pred = pred + self.skip(hist)
        return pred.reshape(-1, self.cfg.out_channel)
