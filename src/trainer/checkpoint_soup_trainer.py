"""Validation-selected checkpoint averaging for STFO training."""

import copy
import json
import os
import os.path as osp
from time import perf_counter

import torch
from torch_geometric.loader import DataLoader

import src.trainer.default_trainer as default_trainer
from src.data.dataset import SpatioTemporalDataset
from src.model.stfo import STFO
from utils.artifacts import portable_metadata, portable_text
from utils.metric import masked_mae_np


def _checkpoint_candidates(path, limit=3):
    candidates = []
    for filename in os.listdir(path):
        if not filename.endswith(".pkl"):
            continue
        try:
            validation_loss = float(filename[:-4])
        except ValueError:
            continue
        candidates.append((validation_loss, osp.join(path, filename)))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise RuntimeError(f"no numeric checkpoint found in {path}")
    return candidates[: max(1, int(limit))]


def _load_state_dict(path, device):
    payload = torch.load(path, map_location=device)
    if "model_state_dict" not in payload:
        raise KeyError(f"checkpoint {path} has no model_state_dict")
    return payload["model_state_dict"]


def average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("at least one state dict is required")
    reference_keys = tuple(state_dicts[0].keys())
    for state_dict in state_dicts[1:]:
        if tuple(state_dict.keys()) != reference_keys:
            raise ValueError("checkpoint state dict keys do not match")

    averaged = {}
    for key in reference_keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            accumulator = tensors[0].detach().to(torch.float64)
            for tensor in tensors[1:]:
                accumulator = accumulator + tensor.detach().to(torch.float64)
            averaged[key] = (accumulator / len(tensors)).to(tensors[0].dtype)
        else:
            averaged[key] = tensors[0].detach().clone()
    return averaged


def _build_model(args, state_dict):
    model = STFO(args).to(args.device)
    model.load_state_dict(state_dict, strict=True)
    return model


def _validation_mae(model, args, val_loader):
    model.eval()
    total = 0.0
    batches = 0
    null_val = default_trainer._null_val(args)
    with torch.no_grad():
        for data in val_loader:
            data = default_trainer._sanitize_data(
                data.to(args.device, non_blocking=True)
            )
            prediction = model(data, args.sub_adj)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(
                    "checkpoint soup produced non-finite validation output"
                )
            prediction = default_trainer._inverse_target_tensor(prediction, args)
            truth = default_trainer._inverse_target_tensor(data.y, args)
            total += float(
                masked_mae_np(
                    truth.cpu().numpy(),
                    prediction.cpu().numpy(),
                    null_val,
                )
            )
            batches += 1
    if batches == 0:
        raise RuntimeError("validation loader is empty")
    return total / batches


def _select_checkpoint_soup(args, val_loader, checkpoint_dir):
    top_k = int(getattr(args, "checkpoint_soup_top_k", 3))
    epsilon = float(getattr(args, "checkpoint_soup_epsilon", 1e-4))
    candidates = _checkpoint_candidates(checkpoint_dir, top_k)
    state_dicts = [
        _load_state_dict(checkpoint_path, args.device)
        for _, checkpoint_path in candidates
    ]

    evaluated = []
    for prefix_size in range(1, len(state_dicts) + 1):
        if prefix_size == 1:
            state_dict = copy.deepcopy(state_dicts[0])
        else:
            state_dict = average_state_dicts(state_dicts[:prefix_size])
        model = _build_model(args, state_dict)
        validation_mae = _validation_mae(model, args, val_loader)
        evaluated.append(
            {
                "prefix_size": prefix_size,
                "validation_mae": validation_mae,
                "state_dict": state_dict,
            }
        )
        del model

    best_single = evaluated[0]
    best_candidate = min(evaluated, key=lambda item: item["validation_mae"])
    if (
        best_candidate["prefix_size"] > 1
        and best_candidate["validation_mae"] < best_single["validation_mae"] - epsilon
    ):
        selected = best_candidate
    else:
        selected = best_single

    selected_validation_mae = float(selected["validation_mae"])
    selected_path = osp.join(checkpoint_dir, f"{selected_validation_mae:.8f}.pkl")
    torch.save(
        {
            "model_state_dict": selected["state_dict"],
            "checkpoint_soup": {
                "prefix_size": int(selected["prefix_size"]),
                "validation_mae": selected_validation_mae,
                "epsilon": epsilon,
                "source_checkpoints": [portable_text(path) for _, path in candidates],
            },
        },
        selected_path,
    )

    manifest = {
        "year": int(args.year),
        "epsilon": epsilon,
        "candidate_checkpoints": [portable_text(path) for _, path in candidates],
        "evaluated": [
            {
                "prefix_size": int(item["prefix_size"]),
                "validation_mae": float(item["validation_mae"]),
            }
            for item in evaluated
        ],
        "selected_prefix_size": int(selected["prefix_size"]),
        "selected_validation_mae": selected_validation_mae,
        "selected_checkpoint": selected_path,
    }
    manifest_path = osp.join(checkpoint_dir, "checkpoint_soup.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(portable_metadata(manifest), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return _build_model(args, selected["state_dict"]), manifest


def train_with_checkpoint_soup(
    inputs,
    args,
    *,
    initial_model=None,
    checkpoint_dir=None,
    evaluate_after_training=True,
):
    default_trainer.train(
        inputs,
        args,
        initial_model=initial_model,
        checkpoint_dir=checkpoint_dir,
        evaluate_after_training=False,
    )

    num_workers = int(getattr(args, "num_workers", 16))
    persistent_workers = (
        bool(getattr(args, "persistent_workers", False)) and num_workers > 0
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
    }
    val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"), **loader_kwargs)
    test_loader = DataLoader(
        default_trainer.build_test_dataset(inputs, args),
        **loader_kwargs,
    )
    vars(args)["sub_adj"] = vars(args)["adj"]

    selection_start = perf_counter()
    checkpoint_dir = checkpoint_dir or osp.join(args.path, str(args.year))
    selected_model, manifest = _select_checkpoint_soup(args, val_loader, checkpoint_dir)
    selection_seconds = perf_counter() - selection_start
    args.logger.info(
        "Checkpoint soup selection: year=%s candidates=%s selected_top_k=%s "
        "validation_mae=%.8f selection_seconds=%.4f",
        args.year,
        [
            {
                "top_k": item["prefix_size"],
                "validation_mae": round(item["validation_mae"], 8),
            }
            for item in manifest["evaluated"]
        ],
        manifest["selected_prefix_size"],
        manifest["selected_validation_mae"],
        selection_seconds,
    )
    args.result[args.year]["soup_selection_time"] = selection_seconds
    args.result[args.year]["soup_top_k"] = manifest["selected_prefix_size"]
    args.result[args.year]["soup_validation_mae"] = manifest["selected_validation_mae"]
    if evaluate_after_training:
        default_trainer.test_model(selected_model, args, test_loader, True)
    return selected_model, manifest["selected_checkpoint"]
