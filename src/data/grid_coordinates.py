"""Deterministic spatial grid assignment for STFO sensor coordinates."""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def _continuous_targets(longitude_latitude: np.ndarray, side: int) -> np.ndarray:
    """Map raw longitude/latitude ranges to continuous lattice coordinates."""
    lower = longitude_latitude.min(axis=0)
    span = longitude_latitude.max(axis=0) - lower
    targets = np.empty_like(longitude_latitude, dtype=np.float64)
    for axis in range(2):
        if span[axis] <= 1e-12:
            targets[:, axis] = 0.5 * (side - 1)
        else:
            targets[:, axis] = (
                (longitude_latitude[:, axis] - lower[axis]) / span[axis] * (side - 1)
            )
    return targets


def _all_lattice_cells(side: int) -> np.ndarray:
    grid_y, grid_x = np.meshgrid(
        np.arange(side, dtype=np.int64),
        np.arange(side, dtype=np.int64),
        indexing="ij",
    )
    return np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)


def _connected_cells(
    cells: np.ndarray,
    available: np.ndarray,
    target: np.ndarray,
    count: int,
) -> list[int]:
    """Reserve the closest available four-connected cell cluster."""
    side = int(round(math.sqrt(cells.shape[0])))
    free_indices = {int(index) for index in np.flatnonzero(available)}

    def neighbors(index: int) -> tuple[int, ...]:
        x, y = cells[index]
        result = []
        if x > 0:
            result.append(index - 1)
        if x + 1 < side:
            result.append(index + 1)
        if y > 0:
            result.append(index - side)
        if y + 1 < side:
            result.append(index + side)
        return tuple(result)

    clusters = {(index,) for index in free_indices}
    for _ in range(1, count):
        expanded: set[tuple[int, ...]] = set()
        for cluster in clusters:
            candidates = {
                candidate
                for index in cluster
                for candidate in neighbors(index)
                if candidate in free_indices and candidate not in cluster
            }
            for candidate in candidates:
                expanded.add(tuple(sorted((*cluster, candidate))))
        clusters = expanded
        if not clusters:
            raise RuntimeError(
                f"cannot reserve a connected block of {count} pseudo-grid cells"
            )

    distance = ((cells.astype(np.float64) - target) ** 2).sum(axis=1)
    anchor = min(
        free_indices,
        key=lambda index: (
            float(distance[index]),
            int(cells[index, 1]),
            int(cells[index, 0]),
        ),
    )
    direction_rank = np.full(cells.shape[0], cells.shape[0], dtype=np.int64)
    queue = deque([anchor])
    direction_rank[anchor] = 0
    next_rank = 1
    while queue:
        index = queue.popleft()
        x, y = cells[index]
        directional_neighbors = []
        if x + 1 < side:  # right
            directional_neighbors.append(index + 1)
        if y + 1 < side:  # up
            directional_neighbors.append(index + side)
        if x > 0:  # left
            directional_neighbors.append(index - 1)
        if y > 0:  # down
            directional_neighbors.append(index - side)
        for candidate in directional_neighbors:
            if direction_rank[candidate] == cells.shape[0]:
                direction_rank[candidate] = next_rank
                next_rank += 1
                queue.append(candidate)

    chosen_cluster = min(
        clusters,
        key=lambda cluster: (
            float(distance[np.asarray(cluster)].sum()),
            tuple(sorted(int(direction_rank[index]) for index in cluster)),
            cluster,
        ),
    )

    # Give the stable sensor-ID order a stable walk through the connected
    # block, starting at the cell closest to the shared lon/lat target.
    remaining = set(chosen_cluster)
    first = min(remaining, key=lambda index: (float(distance[index]), int(index)))
    ordered = [int(first)]
    remaining.remove(first)
    while remaining:
        adjacent = [
            index
            for index in remaining
            if any(
                np.abs(cells[index] - cells[chosen]).sum() == 1 for chosen in ordered
            )
        ]
        if not adjacent:
            raise AssertionError("selected pseudo-grid block is not connected")
        next_index = min(
            adjacent,
            key=lambda index: (float(distance[index]), int(index)),
        )
        ordered.append(int(next_index))
        remaining.remove(next_index)

    available[np.asarray(ordered)] = False
    return ordered


def spatial_grid_coordinates(
    positions_2d: np.ndarray,
    sensor_ids: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Assign sensors one-to-one to a grid using a continuous 2D layout.

    Exact co-locations are treated as groups and are reserved a deterministic
    four-connected block of cells.  Remaining sensors are assigned globally
    to remaining cells by minimum total squared displacement.
    """
    positions_2d = np.asarray(positions_2d, dtype=np.float64)
    if positions_2d.ndim != 2 or positions_2d.shape[1] != 2:
        raise ValueError(
            f"positions_2d must have shape (num_nodes, 2), got {positions_2d.shape}"
        )
    if not np.isfinite(positions_2d).all():
        raise ValueError("positions_2d contains non-finite values")
    num_nodes = int(positions_2d.shape[0])
    if num_nodes <= 0:
        raise ValueError("positions_2d must contain at least one node")
    if sensor_ids is None:
        sensor_ids = [str(index) for index in range(num_nodes)]
    if len(sensor_ids) != num_nodes:
        raise ValueError("sensor_ids and longitude_latitude are misaligned")

    side = int(math.ceil(math.sqrt(num_nodes)))
    targets = _continuous_targets(positions_2d, side)
    cells = _all_lattice_cells(side)
    available = np.ones(cells.shape[0], dtype=bool)
    assigned_cells = np.full(num_nodes, -1, dtype=np.int64)

    groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, point in enumerate(positions_2d):
        groups[(float(point[0]), float(point[1]))].append(index)

    tied_groups = [indices for indices in groups.values() if len(indices) > 1]
    tied_groups.sort(
        key=lambda indices: (
            -len(indices),
            float(targets[indices[0], 1]),
            float(targets[indices[0], 0]),
            tuple(sorted(str(sensor_ids[index]) for index in indices)),
        )
    )

    # Reserve duplicate groups before the global singleton assignment so
    # sensors at exactly the same location remain visibly contiguous.
    for indices in tied_groups:
        member_order = sorted(indices, key=lambda index: str(sensor_ids[index]))
        selected = _connected_cells(
            cells,
            available,
            targets[indices[0]],
            len(indices),
        )
        for node_index, cell_index in zip(member_order, selected):
            assigned_cells[node_index] = cell_index

    remaining_nodes = np.flatnonzero(assigned_cells < 0)
    remaining_cells = np.flatnonzero(available)
    if remaining_nodes.size:
        node_targets = targets[remaining_nodes]
        free_cells = cells[remaining_cells].astype(np.float64)
        cost = ((node_targets[:, None, :] - free_cells[None, :, :]) ** 2).sum(axis=2)
        row_indices, column_indices = linear_sum_assignment(cost)
        assigned_cells[remaining_nodes[row_indices]] = remaining_cells[column_indices]

    if np.any(assigned_cells < 0):
        raise AssertionError("spatial grid assignment left an unassigned sensor")
    if np.unique(assigned_cells).size != num_nodes:
        raise AssertionError("spatial grid assignment is not one-to-one")

    assigned_integer = cells[assigned_cells]
    denominator = max(side - 1, 1)
    coordinates = (assigned_integer / denominator).astype(np.float32)
    displacement = np.linalg.norm(assigned_integer.astype(np.float64) - targets, axis=1)
    tied_node_count = sum(len(indices) for indices in tied_groups)
    info: dict[str, float | int] = {
        "side": side,
        "unused_cells": int(side * side - num_nodes),
        "tied_groups": len(tied_groups),
        "tied_nodes": tied_node_count,
        "mean_grid_displacement": float(displacement.mean()),
        "max_grid_displacement": float(displacement.max()),
    }
    return coordinates, info
