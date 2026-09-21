"""Run the STFO main experiments on a sequence of expanding sensor networks."""

import argparse
import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from src.data.cache import load_inputs
from torch_geometric.loader import DataLoader

from src.data.stfo_coordinates import STFOCoordinateSystem
from src.model.stfo import STFO
from src.trainer.checkpoint_soup_trainer import train_with_checkpoint_soup
from src.trainer.default_trainer import (
    build_test_dataset,
    prepare_target_scaler,
    test_model,
)
from utils.artifacts import portable_metadata
from utils.initialize import init_log, seed_anything

ROOT = Path(__file__).resolve().parent
WIDTHS = {"S": 32, "M": 64, "L": 128}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=Path, default=ROOT / "conf/STFO_PEMS.json")
    parser.add_argument(
        "--size", choices=WIDTHS, help="Override the configured hidden width"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed (default: config seed or 42)"
    )
    parser.add_argument(
        "--gpuid", type=int, default=0, help="CUDA device index; -1 selects CPU"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Directory containing PEMS, CA, and AIR",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/FastData",
        help="Processed window cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Run directory; training requires a new or empty directory",
    )
    parser.add_argument(
        "--epochs", type=int, help="Override the configured training epochs"
    )
    parser.add_argument(
        "--workers", type=int, help="Override the data loader worker count"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the selected checkpoints in --output",
    )
    cli = parser.parse_args(argv)
    with cli.conf.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("dataset") not in {"PEMS", "CA", "AIR"}:
        parser.error("Configuration dataset must be PEMS, CA, or AIR")
    if cli.size:
        config["model"]["hidden_channel"] = WIDTHS[cli.size]
    if cli.epochs is not None:
        config["epoch"] = cli.epochs
    if cli.workers is not None:
        config["num_workers"] = cli.workers
    if config["epoch"] < 1 or config["batch_size"] < 1 or config["num_workers"] < 0:
        parser.error(
            "Epochs and batch size must be positive; workers must be nonnegative"
        )
    if not config.get("normalize_y") or config["x_len"] != 12 or config["y_len"] != 12:
        parser.error(
            "Main experiments require normalized targets and 12-step inputs/targets"
        )
    if config["model"]["in_channel"] != 12 or config["model"]["out_channel"] != 12:
        parser.error("Model input and output channels must both be 12")
    if config["begin_year"] > config["end_year"]:
        parser.error("The beginning period must not exceed the ending period")
    seed = cli.seed if cli.seed is not None else int(config.get("seed", 42))
    width = config["model"]["hidden_channel"]
    size = next((key for key, value in WIDTHS.items() if value == width), str(width))
    output = (
        cli.output or ROOT / "outputs" / f"{config['dataset']}_STFO-{size}_seed{seed}"
    )
    if cli.evaluate and not output.is_dir():
        parser.error("Evaluation requires an existing run directory")
    if not cli.evaluate and output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; select a new --output directory")
    args = argparse.Namespace(**config)
    args.seed = seed
    args.data_root = str(cli.data_root.resolve())
    args.cache_dir = str(cli.cache_dir.resolve())
    args.coord_path = str(cli.data_root.resolve() / config["coord_path"])
    args.path = str(output.resolve())
    args.evaluate = cli.evaluate
    if torch.cuda.is_available() and cli.gpuid >= 0:
        torch.cuda.set_device(cli.gpuid)
        args.device = torch.device(f"cuda:{cli.gpuid}")
    else:
        args.device = torch.device("cpu")
    args.run_config = {**config, "seed": seed}
    return args


def save_results(args):
    """Write period-level results and their unweighted period means."""
    rows = []
    for period in range(args.begin_year, args.end_year + 1):
        for horizon in ("3", "6", "12", "Avg"):
            if period not in args.result[horizon][" MAE"]:
                continue
            rows.append(
                {
                    "period": period,
                    "horizon": horizon,
                    "MAE": float(args.result[horizon][" MAE"][period]),
                    "RMSE": float(args.result[horizon]["RMSE"][period]),
                    "MAPE": float(args.result[horizon]["MAPE"][period]),
                }
            )
    name = "evaluation" if args.evaluate else "metrics"
    with (Path(args.path) / f"{name}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["period", "horizon", "MAE", "RMSE", "MAPE"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for horizon in ("3", "6", "12", "Avg"):
        selected = [row for row in rows if row["horizon"] == horizon]
        if selected:
            summary[horizon] = {
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in ("MAE", "RMSE", "MAPE")
            }
    with (Path(args.path) / f"{name}.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {"periods": rows, "period_mean": summary}, stream, indent=2, allow_nan=False
        )
        stream.write("\n")


def main(args):
    seed_anything(args.seed)
    # Validate coordinate files before creating an output directory.
    coordinates = STFOCoordinateSystem(args)
    for period in range(args.begin_year, args.end_year + 1):
        root = Path(args.data_root) / args.dataset
        for path in (
            root / "RawData" / f"{period}.npz",
            root / "graph" / f"{period}_adj.npz",
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing data file: {path}")
    Path(args.path).mkdir(parents=True, exist_ok=True)
    init_log(args)
    if not args.evaluate:
        with (Path(args.path) / "config.json").open("w", encoding="utf-8") as stream:
            json.dump(portable_metadata(args.run_config), stream, indent=2)
            stream.write("\n")
    args.logger.info(
        "STFO dataset=%s width=%s seed=%s device=%s",
        args.dataset,
        args.model["hidden_channel"],
        args.seed,
        args.device,
    )
    args.result = {
        horizon: {metric: {} for metric in (" MAE", "RMSE", "MAPE")}
        for horizon in ("3", "6", "12", "Avg")
    }
    for period in range(args.begin_year, args.end_year + 1):
        args.year = period
        graph_path = Path(args.data_root) / args.dataset / "graph" / f"{period}_adj.npz"
        with np.load(graph_path, allow_pickle=False) as archive:
            raw_adj = archive["x"]
        if raw_adj.ndim != 2 or raw_adj.shape[0] != raw_adj.shape[1]:
            raise ValueError("Adjacency must be a square matrix")
        graph = nx.from_numpy_array(raw_adj)
        inputs = load_inputs(args, period, graph)
        prepare_target_scaler(inputs, args)
        coords, info = coordinates.coordinates_for_period(
            period, expected_nodes=raw_adj.shape[0]
        )
        args.logger.info("Period %s coordinates: %s", period, info)
        args.coords = torch.from_numpy(coords).float().to(args.device)
        adj = raw_adj / (np.sum(raw_adj, 1, keepdims=True) + 1e-6)
        args.adj = torch.from_numpy(adj).float().to(args.device)
        if args.evaluate:
            folder = Path(args.path) / str(period)
            with (folder / "checkpoint_soup.json").open(encoding="utf-8") as stream:
                selection = json.load(stream)
            checkpoint = folder / Path(selection["selected_checkpoint"]).name
            model = STFO(args).to(args.device)
            model.load_state_dict(
                torch.load(checkpoint, map_location=args.device, weights_only=True)[
                    "model_state_dict"
                ]
            )
            loader = DataLoader(
                build_test_dataset(inputs, args),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            test_model(model, args, loader, True)
        else:
            train_with_checkpoint_soup(inputs, args)
        save_results(args)
    args.logger.info("Completed all periods; results saved in the run directory")
    return args.result


if __name__ == "__main__":
    main(parse_args())
