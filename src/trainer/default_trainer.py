"""Sequential STFO training with masked MAE and validation checkpointing."""

import os.path as osp
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

from src.data.dataset import SpatioTemporalDataset
from src.model.stfo import STFO
from utils.common_tools import load_best_model, mkdirs
from utils.metric import MAE_torch, cal_metric, masked_mae_np


def _optimization_loss(prediction, target, args):
    return MAE_torch(prediction, target, _null_val(args))


def build_test_dataset(inputs, args):
    return SpatioTemporalDataset(inputs, "test")


def _null_val(args):
    value = getattr(args, "null_val", 0.0)
    if isinstance(value, str) and value.lower() in ("nan", "none", "null"):
        return np.nan
    return value


def _sanitize_data(data):
    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=0.0, neginf=0.0)
    data.y = torch.nan_to_num(data.y, nan=0.0, posinf=0.0, neginf=0.0)
    return data


def _input_value(inputs, key, default):
    try:
        if key in inputs:
            value = inputs[key]
        else:
            return default
    except TypeError:
        return default
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return value


def prepare_target_scaler(inputs, args):
    normalize_y = bool(
        _input_value(inputs, "normalize_y", getattr(args, "normalize_y", False))
    )
    target_mean = float(_input_value(inputs, "target_mean", 0.0))
    target_std = float(_input_value(inputs, "target_std", 1.0))
    input_mean = float(
        _input_value(inputs, "input_mean", getattr(args, "input_mean", float("nan")))
    )
    input_std = float(
        _input_value(inputs, "input_std", getattr(args, "input_std", float("nan")))
    )
    if not np.isfinite(target_mean):
        target_mean = 0.0
    if not np.isfinite(target_std) or target_std < 1e-8:
        target_std = 1.0
    vars(args)["normalize_y"] = normalize_y
    vars(args)["target_mean"] = target_mean
    vars(args)["target_std"] = target_std
    vars(args)["input_mean"] = input_mean
    vars(args)["input_std"] = input_std


def _inverse_target_tensor(value, args):
    if not bool(getattr(args, "normalize_y", False)):
        return value
    return value.float() * float(getattr(args, "target_std", 1.0)) + float(
        getattr(args, "target_mean", 0.0)
    )


def train(
    inputs,
    args,
    *,
    initial_model=None,
    checkpoint_dir=None,
    evaluate_after_training=True,
):
    path = checkpoint_dir or osp.join(args.path, str(args.year))
    mkdirs(path)
    prepare_target_scaler(inputs, args)

    num_workers = int(getattr(args, "num_workers", 16))
    persistent_workers = (
        bool(getattr(args, "persistent_workers", False)) and num_workers > 0
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
    }
    train_loader = DataLoader(
        SpatioTemporalDataset(inputs, "train"), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        SpatioTemporalDataset(inputs, "val"), shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        build_test_dataset(inputs, args),
        shuffle=False,
        **loader_kwargs,
    )
    vars(args)["sub_adj"] = vars(args)["adj"]

    args.logger.info("[*] Year " + str(args.year) + " Dataset load!")

    if initial_model is not None:
        model = initial_model.to(args.device)
    elif args.year > args.begin_year:
        model, _ = load_best_model(args)
    else:
        model = STFO(args).to(args.device)

    model.count_parameters()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    null_val = _null_val(args)
    grad_clip = float(getattr(args, "grad_clip", 5.0) or 0.0)

    args.logger.info("[*] Year " + str(args.year) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = int(getattr(args, "patience", 5))
    model.train()
    use_time = []
    for epoch in range(args.epoch):
        start_time = datetime.now()

        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = _sanitize_data(data.to(args.device, non_blocking=True))
            optimizer.zero_grad()
            pred = model(data, args.sub_adj)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(
                    f"model produced non-finite output at epoch {epoch}, batch {batch_idx}"
                )

            loss = _optimization_loss(pred, data.y, args)
            if not torch.isfinite(loss):
                args.logger.warning(
                    "skip non-finite loss at epoch %s batch %s", epoch, batch_idx
                )
                continue

            training_loss += float(loss.detach()) * float(args.target_std)
            cn += 1

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        if cn == 0:
            raise RuntimeError("all training batches produced non-finite losses")
        training_loss = training_loss / cn

        validation_loss = 0.0
        cn = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = _sanitize_data(data.to(args.device, non_blocking=True))
                pred = model(data, args.sub_adj)
                if not torch.isfinite(pred).all():
                    raise FloatingPointError(
                        f"model produced non-finite validation output at epoch {epoch}"
                    )
                pred_eval = _inverse_target_tensor(pred, args)
                truth_eval = _inverse_target_tensor(data.y, args)
                loss = masked_mae_np(
                    truth_eval.cpu().data.numpy(),
                    pred_eval.cpu().data.numpy(),
                    null_val,
                )
                validation_loss += float(loss)
                cn += 1
        model.train()
        validation_loss = float(validation_loss / cn)

        args.logger.info(
            f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}"
        )

        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save(
                {"model_state_dict": model.state_dict()},
                osp.join(path, str(round(validation_loss, 4)) + ".pkl"),
            )
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss) + ".pkl")
    best_model = model

    best_model.load_state_dict(
        torch.load(best_model_path, args.device)["model_state_dict"]
    )
    best_model = best_model.to(args.device)

    if evaluate_after_training:
        test_model(best_model, args, test_loader, True)
    args.result[args.year] = {
        "total_time": total_time,
        "average_time": sum(use_time) / len(use_time),
        "epoch_num": epoch + 1,
    }
    args.logger.info(
        "Finished optimization, total time:{:.2f} s, best model:{}".format(
            total_time, best_model_path
        )
    )
    return best_model, best_model_path


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = _sanitize_data(data.to(args.device, non_blocking=pin_memory))
            pred = model(data, args.adj)
            if not torch.isfinite(pred).all():
                raise FloatingPointError("model produced non-finite test output")
            pred = _inverse_target_tensor(pred, args)
            truth = _inverse_target_tensor(data.y, args)
            loss += MAE_torch(pred, truth, _null_val(args))
            pred, _ = to_dense_batch(pred, batch=data.batch)
            truth, _ = to_dense_batch(truth, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(truth.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        cal_metric(truth_, pred_, args)
