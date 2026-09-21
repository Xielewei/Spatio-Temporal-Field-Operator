"""Validate and prepare the STFO time-series cache."""

import os.path as osp
import zipfile
from pathlib import Path

import numpy as np

from utils.data_convert import generate_samples


def _npz_array_header(path, key):
    """Read an array's shape and dtype without materializing its NPZ payload."""
    with zipfile.ZipFile(path, "r") as archive:
        member = key + ".npy"
        if member not in archive.namelist():
            raise KeyError(f"missing array {key}")
        with archive.open(member, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY version {version} for {key}")
    return tuple(shape), np.dtype(dtype)


def _npz_scalar(path, key):
    """Read one scalar member directly instead of opening the full NPZ mapping."""
    with zipfile.ZipFile(path, "r") as archive:
        member = key + ".npy"
        if member not in archive.namelist():
            raise KeyError(f"missing array {key}")
        with archive.open(member, "r") as stream:
            value = np.lib.format.read_array(stream, allow_pickle=False)
    if value.shape != ():
        raise ValueError(f"{key} must be a scalar, got shape {value.shape}")
    return value.item()


def _processed_cache_status(cache_path, raw_path, graph_path, args, num_nodes):
    """Validate a processed cache cheaply using NPZ headers and scalar metadata."""
    if not osp.isfile(cache_path):
        return False, "cache file does not exist"
    newest_source = max(osp.getmtime(raw_path), osp.getmtime(graph_path))
    if osp.getmtime(cache_path) < newest_source:
        return False, "cache is older than RawData or graph"

    expected_shapes = {
        "train_x": (args.x_len, num_nodes),
        "val_x": (args.x_len, num_nodes),
        "test_x": (args.x_len, num_nodes),
        "train_y": (args.y_len, num_nodes),
        "val_y": (args.y_len, num_nodes),
        "test_y": (args.y_len, num_nodes),
    }
    try:
        for key, trailing_shape in expected_shapes.items():
            shape, dtype = _npz_array_header(cache_path, key)
            if len(shape) != 3 or shape[1:] != trailing_shape or shape[0] <= 0:
                return False, f"{key} has incompatible shape {shape}"
            if dtype.kind not in {"f", "i", "u"}:
                return False, f"{key} has unsupported dtype {dtype}"
        for key in ("target_mean", "target_std", "normalize_y"):
            shape, _ = _npz_array_header(cache_path, key)
            if shape != ():
                return False, f"{key} must be a scalar, got shape {shape}"
        cached_normalize_y = bool(_npz_scalar(cache_path, "normalize_y"))
        if cached_normalize_y != bool(getattr(args, "normalize_y", False)):
            return False, "normalize_y does not match the current config"
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        return False, str(exc)
    return True, "validated shape, metadata, and source freshness"


def load_inputs(args, year, graph):
    root = Path(args.data_root) / args.dataset
    raw_path = root / "RawData" / f"{year}.npz"
    graph_path = root / "graph" / f"{year}_adj.npz"
    cache_path = Path(args.cache_dir) / args.dataset / f"{year}.npz"
    valid, reason = _processed_cache_status(
        cache_path, raw_path, graph_path, args, graph.number_of_nodes()
    )
    if valid:
        args.logger.info("Period %s: using processed cache", year)
        with np.load(cache_path, allow_pickle=False) as archive:
            return dict(archive)
    args.logger.info("Period %s: preparing data (%s)", year, reason)
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = archive["x"]
    if raw.ndim != 2 or raw.shape[1] != graph.number_of_nodes():
        raise ValueError("Raw data must have shape (time steps, graph nodes)")
    effective_steps = (
        min(raw.shape[0], 31 * 288) if args.dataset == "PEMS" else raw.shape[0]
    )
    boundaries = [
        0,
        int(effective_steps * 0.6),
        int(effective_steps * 0.8),
        effective_steps,
    ]
    if min(np.diff(boundaries)) <= args.x_len + args.y_len:
        raise ValueError("Each 60/20/20 split must contain more than 24 time steps")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return generate_samples(31, str(cache_path), raw, graph, dataset=args.dataset)
