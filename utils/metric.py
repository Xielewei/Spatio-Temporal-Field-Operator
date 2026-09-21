import numpy as np
import torch


def _normalize_null_val(null_val):
    if null_val is None:
        return np.nan
    if isinstance(null_val, str):
        if null_val.lower() in ("nan", "none", "null"):
            return np.nan
        return float(null_val)
    return null_val


def MAE_torch(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    null_val = _normalize_null_val(null_val)
    finite = torch.isfinite(target)
    if np.isnan(null_val):
        mask = finite
    else:
        eps = 5e-5
        null = torch.as_tensor(null_val, device=target.device, dtype=target.dtype)
        mask = finite & ~torch.isclose(
            null.expand_as(target), target, atol=eps, rtol=0.0
        )
    if not torch.any(mask):
        mask = finite
    mask = mask.float()
    mask /= torch.mean(mask) + 1e-8
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.nan_to_num(prediction) - torch.nan_to_num(target))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def mask_np(array, null_val):
    null_val = _normalize_null_val(null_val)
    finite = np.isfinite(array)
    if np.isnan(null_val):
        mask = finite
    else:
        mask = finite & np.not_equal(array, null_val)
    if not np.any(mask):
        mask = finite
    return mask.astype("float32")


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = mask_np(y_true, null_val) * (np.abs(y_true) > 1e-5)
        mask /= mask.mean() + 1e-8
        mape = np.abs((np.nan_to_num(y_pred) - np.nan_to_num(y_true)) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean() + 1e-8
    mse = (np.nan_to_num(y_true) - np.nan_to_num(y_pred)) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    return masked_mse_np(y_true, y_pred, null_val) ** 0.5


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean() + 1e-8
    mae = np.abs(np.nan_to_num(y_true) - np.nan_to_num(y_pred))
    return np.mean(np.nan_to_num(mask * mae))


def cal_metric(ground_truth, prediction, args):
    """Calculate metrics for each time step."""
    args.logger.info(f"[*] year {args.year}, testing")

    mae_list, rmse_list, mape_list = [], [], []

    # ground_truth and prediction have shape [batch_size, num_nodes, 12].
    # Compute metrics for every horizon.
    num_time_steps = ground_truth.shape[2]  # Expected to be 12

    for t in range(1, num_time_steps + 1):
        # Select the current horizon.
        # Evaluate horizon t (one-based indexing).
        gt_t = ground_truth[:, :, t - 1 : t]  # Ground truth at horizon t
        pred_t = prediction[:, :, t - 1 : t]  # Prediction at horizon t

        null_val = getattr(args, "null_val", 0)
        mae = masked_mae_np(gt_t, pred_t, null_val)
        rmse = masked_rmse_np(gt_t, pred_t, null_val)
        mape = masked_mape_np(gt_t, pred_t, null_val)

        mae_list.append(mae)
        rmse_list.append(rmse)
        mape_list.append(mape)

        # Report horizons 3, 6, and 12.
        if t in [3, 6, 12]:
            args.logger.info(
                f"T:{t}\tMAE\t{mae:.4f}\tRMSE\t{rmse:.4f}\tMAPE\t{mape:.4f}"
            )
            args.result[str(t)][" MAE"][args.year] = mae
            args.result[str(t)]["MAPE"][args.year] = mape
            args.result[str(t)]["RMSE"][args.year] = rmse

    # Average metrics over all horizons.
    avg_mae = np.mean(mae_list)
    avg_rmse = np.mean(rmse_list)
    avg_mape = np.mean(mape_list)

    args.result["Avg"][" MAE"][args.year] = avg_mae
    args.result["Avg"]["RMSE"][args.year] = avg_rmse
    args.result["Avg"]["MAPE"][args.year] = avg_mape

    args.logger.info(
        f"T:Avg\tMAE\t{avg_mae:.4f}\tRMSE\t{avg_rmse:.4f}\tMAPE\t{avg_mape:.4f}"
    )

    return mae_list, rmse_list, mape_list
