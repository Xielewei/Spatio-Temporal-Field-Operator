import numpy as np
import tqdm


def clean_raw_data(data):
    data = np.asarray(data, dtype=np.float32)
    if np.isfinite(data).all():
        return data
    col_mean = np.nanmean(np.where(np.isfinite(data), data, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0).astype(np.float32)
    rows, cols = np.where(~np.isfinite(data))
    data = data.copy()
    data[rows, cols] = col_mean[cols]
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def generate_dataset(data, idx, x_len=12, y_len=12):
    """Build input/target windows in reverse chronological order."""
    res = data[idx]
    node_size = data.shape[1]
    t = len(idx) - 1
    x_index, y_index = [], []

    for i in tqdm.tqdm(range(t, 0, -1)):
        if i - x_len - y_len >= 0:
            x_index.extend(list(range(i - x_len - y_len, i - y_len)))
            y_index.extend(list(range(i - y_len, i)))

    x_index = np.asarray(x_index, dtype=np.int64)
    y_index = np.asarray(y_index, dtype=np.int64)
    x = res[x_index].reshape((-1, x_len, node_size))
    y = res[y_index].reshape((-1, y_len, node_size))

    return x, y


def _finite_mean_std(data):
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 1.0
    mean = float(data[finite].mean())
    std = float(data[finite].std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0
    return mean, std


def _standardize(data, mean, std):
    data = np.asarray(data, dtype=np.float32)
    return np.nan_to_num((data - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )


def generate_samples(
    days, savepath, data, graph, dataset, train_rate=0.6, val_rate=0.2
):
    """
    Generate training, validation and test datasets and save them as .npz files
    """
    data = clean_raw_data(data)
    edge_index = np.array(list(graph.edges)).T
    del graph

    if dataset == "PEMS":
        data = data[0 : days * 288, :]

    t = data.shape[0]

    train_idx = [i for i in range(int(t * train_rate))]
    val_idx = [i for i in range(int(t * train_rate), int(t * (train_rate + val_rate)))]
    test_idx = [i for i in range(int(t * (train_rate + val_rate)), t)]

    train_x, train_y = generate_dataset(data, train_idx)
    val_x, val_y = generate_dataset(data, val_idx)
    test_x, test_y = generate_dataset(data, test_idx)

    x_mean, x_std = _finite_mean_std(train_x)
    y_mean, y_std = _finite_mean_std(train_y)
    train_x = _standardize(train_x, x_mean, x_std)
    val_x = _standardize(val_x, x_mean, x_std)
    test_x = _standardize(test_x, x_mean, x_std)
    train_y = _standardize(train_y, y_mean, y_std)
    val_y = _standardize(val_y, y_mean, y_std)
    test_y = _standardize(test_y, y_mean, y_std)

    payload = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "test_x": test_x,
        "test_y": test_y,
        "edge_index": edge_index,
        "target_mean": np.asarray(y_mean, dtype=np.float32),
        "target_std": np.asarray(y_std, dtype=np.float32),
        "normalize_y": np.asarray(True),
    }
    payload["input_mean"] = np.asarray(x_mean, dtype=np.float32)
    payload["input_std"] = np.asarray(x_std, dtype=np.float32)
    np.savez(savepath, **payload)
    return payload
