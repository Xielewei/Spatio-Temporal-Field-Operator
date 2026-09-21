"""Load sensor identities and canonical geographic coordinates."""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def _canonical_sensor_id(value: str) -> str:
    """Normalize numeric CSV IDs while preserving alphanumeric station codes."""
    value = str(value).strip()
    if not value:
        raise ValueError("sensor identifier is empty")
    try:
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _great_circle_distance_km(a: np.ndarray, b: np.ndarray) -> float:
    """Haversine distance between [longitude, latitude] points in degrees."""
    lon1, lat1 = np.deg2rad(a)
    lon2, lat2 = np.deg2rad(b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(hav)))


@dataclass(frozen=True)
class CoordinateRecord:
    node_index: int
    sensor_id: str
    longitude: float
    latitude: float

    @property
    def lon_lat(self) -> np.ndarray:
        return np.asarray([self.longitude, self.latitude], dtype=np.float64)


def _read_coordinate_file(path: Path, id_column: str) -> tuple[CoordinateRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"physical coordinate file not found: {path}")

    required = {"node_index", id_column, "longitude", "latitude"}
    records = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            try:
                node_index = int(row["node_index"])
                sensor_id = _canonical_sensor_id(row[id_column])
                longitude = float(row["longitude"])
                latitude = float(row["latitude"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid coordinate record at {path}:{line_number}"
                ) from exc

            if not math.isfinite(longitude) or not math.isfinite(latitude):
                raise ValueError(f"non-finite coordinate at {path}:{line_number}")
            if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
                raise ValueError(
                    f"coordinate outside longitude/latitude bounds at {path}:{line_number}"
                )
            records.append(CoordinateRecord(node_index, sensor_id, longitude, latitude))

    if not records:
        raise ValueError(f"physical coordinate file is empty: {path}")

    records.sort(key=lambda record: record.node_index)
    indices = [record.node_index for record in records]
    if indices != list(range(len(records))):
        raise ValueError(
            f"{path} must contain each node_index exactly once from 0 to {len(records) - 1}"
        )

    sensor_ids = [record.sensor_id for record in records]
    if len(sensor_ids) != len(set(sensor_ids)):
        raise ValueError(f"{path} contains duplicate values in {id_column}")
    return tuple(records)


class LongitudeLatitudeCoordinateSystem:
    """Canonical longitude/latitude rows for STFO grid assignment."""

    def __init__(
        self,
        begin_period: int,
        end_period: int,
        path_template: str,
        id_column: str,
        require_nested: bool = True,
    ):
        if begin_period > end_period:
            raise ValueError("begin_period must not exceed end_period")
        if not path_template or "{year}" not in path_template:
            raise ValueError(
                "coord_path must be a non-empty template containing {year}"
            )
        if not id_column:
            raise ValueError("coord_id_column must be set for spatial coordinates")

        self.periods = tuple(range(begin_period, end_period + 1))
        self.path_template = path_template
        self.id_column = id_column
        self.require_nested = bool(require_nested)
        self.source_paths = {
            period: Path(path_template.format(year=period)) for period in self.periods
        }
        self.records_by_period = {
            period: _read_coordinate_file(self.source_paths[period], id_column)
            for period in self.periods
        }
        self.source_sha256 = {
            period: _file_sha256(self.source_paths[period]) for period in self.periods
        }
        self._validate_expansion()
        self.canonical_lon_lat, self.max_metadata_shift_km = (
            self._canonicalize_coordinates()
        )

    def _validate_expansion(self) -> None:
        if not self.require_nested:
            return
        previous_ids = None
        for period in self.periods:
            current_ids = {
                record.sensor_id for record in self.records_by_period[period]
            }
            if previous_ids is not None:
                missing = previous_ids.difference(current_ids)
                if missing:
                    preview = sorted(missing)[:5]
                    raise ValueError(
                        f"period {period} is not a nested sensor expansion; "
                        f"{len(missing)} prior sensor IDs are missing, e.g. {preview}"
                    )
            previous_ids = current_ids

    def _canonicalize_coordinates(self) -> tuple[dict[str, np.ndarray], float]:
        canonical = {}
        max_shift_km = 0.0
        for period in self.periods:
            for record in self.records_by_period[period]:
                point = record.lon_lat
                if record.sensor_id not in canonical:
                    canonical[record.sensor_id] = point
                else:
                    max_shift_km = max(
                        max_shift_km,
                        _great_circle_distance_km(canonical[record.sensor_id], point),
                    )
        return canonical, max_shift_km

    def longitude_latitude_for_period(
        self,
        period: int,
        expected_nodes: int | None = None,
    ) -> np.ndarray:
        if period not in self.records_by_period:
            raise KeyError(f"period {period} is outside the coordinate stream")
        records = self.records_by_period[period]
        if expected_nodes is not None and len(records) != int(expected_nodes):
            raise ValueError(
                f"period {period} has {len(records)} longitude/latitude rows, "
                f"but the graph/data contains {expected_nodes} nodes"
            )
        return np.stack(
            [self.canonical_lon_lat[record.sensor_id] for record in records],
            axis=0,
        )

    def summary(self) -> dict:
        return {
            "periods": list(self.periods),
            "sensor_counts": {
                period: len(self.records_by_period[period]) for period in self.periods
            },
            "canonical_sensor_count": len(self.canonical_lon_lat),
            "max_metadata_shift_km": self.max_metadata_shift_km,
            "source_sha256": dict(self.source_sha256),
            "projection": None,
            "normalization": "per-period lon/lat range to discrete lattice only",
        }
