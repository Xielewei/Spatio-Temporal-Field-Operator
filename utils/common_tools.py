"""Checkpoint loading for sequential STFO training."""

from pathlib import Path

import torch

from src.model.stfo import STFO


def mkdirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_best_model(args):
    """Load the previous period's checkpoint with the lowest recorded MAE."""
    folder = Path(args.path) / str(args.year - 1)
    candidates = []
    for path in folder.iterdir():
        if path.suffix != ".pkl":
            continue
        try:
            candidates.append((float(path.stem), path))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No validation checkpoint in {folder}")
    load_path = min(candidates, key=lambda item: item[0])[1]
    args.logger.info("Loading previous-period checkpoint: %s", load_path)
    payload = torch.load(load_path, map_location=args.device, weights_only=True)
    model = STFO(args)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(args.device), str(load_path)
