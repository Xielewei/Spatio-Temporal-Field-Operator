"""Frozen 2D node embeddings from static metadata for grid construction."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .physical_coordinates import _canonical_sensor_id

_MISSING = "__MISSING__"
_UNKNOWN = "__UNKNOWN__"


def _clean_category(value: str | None) -> str:
    value = "" if value is None else str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return _MISSING
    return value


def _parse_numeric(value: str | None) -> float:
    value = "" if value is None else str(value).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        result = float(value)
    except ValueError:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


class StaticFeatureCoordinateSystem:
    """Blend geography with a first-period semantic PCA in a shared 2D range.

    The PCA basis, numeric normalization, and categorical vocabulary are fit
    only on the first configured period.  Later periods are transformed with
    those frozen statistics.  Each sensor's metadata is canonicalized at its
    first appearance so shared-node embeddings do not move because a later
    CSV revises a field.
    """

    def __init__(
        self,
        begin_period: int,
        end_period: int,
        path_template: str,
        id_column: str,
        numeric_columns: list[str] | tuple[str, ...],
        categorical_columns: list[str] | tuple[str, ...],
        require_nested: bool = True,
        blend_alpha: float = 0.10,
    ):
        if begin_period > end_period:
            raise ValueError("begin_period must not exceed end_period")
        if not path_template or "{year}" not in path_template:
            raise ValueError(
                "coord_path must be a non-empty template containing {year}"
            )
        if not id_column:
            raise ValueError("coord_id_column must be set for static features")

        self.periods = tuple(range(begin_period, end_period + 1))
        self.path_template = path_template
        self.id_column = id_column
        self.numeric_columns = tuple(str(column) for column in numeric_columns)
        self.categorical_columns = tuple(str(column) for column in categorical_columns)
        if not self.numeric_columns and not self.categorical_columns:
            raise ValueError(
                "at least one static numeric or categorical column is required"
            )
        all_columns = self.numeric_columns + self.categorical_columns
        if len(all_columns) != len(set(all_columns)):
            raise ValueError("static numeric and categorical columns must be unique")
        forbidden = {"node_index", id_column, "station_id", "station_code"}
        leakage_columns = forbidden.intersection(all_columns)
        if leakage_columns:
            raise ValueError(
                "node identity/index columns cannot be static features: "
                f"{sorted(leakage_columns)}"
            )

        self.require_nested = bool(require_nested)
        self.blend_alpha = float(blend_alpha)
        if not math.isfinite(self.blend_alpha) or not 0.0 <= self.blend_alpha <= 1.0:
            raise ValueError("static_blend_alpha must be within [0, 1]")
        self.rows_by_period = {
            period: self._read_period(period) for period in self.periods
        }
        self._validate_expansion()
        self.canonical_rows = self._canonicalize_rows()
        self._fit_encoder()

    def _read_period(self, period: int) -> tuple[dict[str, str], ...]:
        path = Path(self.path_template.format(year=period))
        if not path.is_file():
            raise FileNotFoundError(f"static feature file not found: {path}")
        required = {
            "node_index",
            self.id_column,
            *self.numeric_columns,
            *self.categorical_columns,
        }
        rows = []
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing static columns: {sorted(missing)}")
            for line_number, source in enumerate(reader, start=2):
                try:
                    node_index = int(source["node_index"])
                    sensor_id = _canonical_sensor_id(source[self.id_column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid static feature record at {path}:{line_number}"
                    ) from exc
                row = {column: source.get(column, "") for column in required}
                row["node_index"] = str(node_index)
                row[self.id_column] = sensor_id
                rows.append(row)

        rows.sort(key=lambda row: int(row["node_index"]))
        indices = [int(row["node_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(
                f"{path} must contain node_index 0 through {len(rows) - 1}"
            )
        sensor_ids = [row[self.id_column] for row in rows]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError(f"{path} contains duplicate values in {self.id_column}")
        return tuple(rows)

    def _validate_expansion(self) -> None:
        if not self.require_nested:
            return
        previous_ids = None
        for period in self.periods:
            current_ids = {row[self.id_column] for row in self.rows_by_period[period]}
            if previous_ids is not None:
                missing = previous_ids.difference(current_ids)
                if missing:
                    raise ValueError(
                        f"period {period} drops {len(missing)} prior static-feature nodes"
                    )
            previous_ids = current_ids

    def _canonicalize_rows(self) -> dict[str, dict[str, str]]:
        canonical = {}
        for period in self.periods:
            for row in self.rows_by_period[period]:
                sensor_id = row[self.id_column]
                if sensor_id not in canonical:
                    canonical[sensor_id] = dict(row)
        return canonical

    def _canonical_period_rows(self, period: int) -> list[dict[str, str]]:
        if period not in self.rows_by_period:
            raise KeyError(f"period {period} is outside the static-feature stream")
        return [
            self.canonical_rows[row[self.id_column]]
            for row in self.rows_by_period[period]
        ]

    def _fit_encoder(self) -> None:
        first_rows = self._canonical_period_rows(self.periods[0])
        self.numeric_median = {}
        self.numeric_mean = {}
        self.numeric_scale = {}
        for column in self.numeric_columns:
            values = np.asarray([_parse_numeric(row[column]) for row in first_rows])
            finite = np.isfinite(values)
            if not np.any(finite):
                raise ValueError(
                    f"static numeric column {column!r} has no finite first-period values"
                )
            median = float(np.median(values[finite]))
            filled = np.where(finite, values, median)
            mean = float(filled.mean())
            scale = float(filled.std())
            self.numeric_median[column] = median
            self.numeric_mean[column] = mean
            self.numeric_scale[column] = scale if scale > 1e-8 else 1.0

        self.category_values = {}
        for column in self.categorical_columns:
            observed = sorted({_clean_category(row[column]) for row in first_rows})
            self.category_values[column] = tuple(observed + [_UNKNOWN])

        first_features = self._encode_rows(first_rows)
        self._build_feature_slices()
        self._fit_geo_semantic_blend(first_features)

    def _build_feature_slices(self) -> None:
        self.feature_slices = {}
        cursor = 0
        for column in self.numeric_columns:
            self.feature_slices[column] = slice(cursor, cursor + 2)
            cursor += 2
        for column in self.categorical_columns:
            width = len(self.category_values[column])
            self.feature_slices[column] = slice(cursor, cursor + width)
            cursor += width

    @staticmethod
    def _deterministic_component_signs(components: np.ndarray) -> np.ndarray:
        components = components.copy()
        for axis in range(components.shape[0]):
            pivot = int(np.argmax(np.abs(components[axis])))
            if components[axis, pivot] < 0:
                components[axis] *= -1.0
        return components

    def _fit_geo_semantic_blend(self, first_features: np.ndarray) -> None:
        required = {"longitude", "latitude"}
        if not required.issubset(self.numeric_columns):
            raise ValueError("geo-semantic blend requires longitude and latitude")
        self.geo_indices = np.asarray(
            [
                self.feature_slices["longitude"].start,
                self.feature_slices["latitude"].start,
            ],
            dtype=np.int64,
        )
        excluded = set(
            range(*self.feature_slices["longitude"].indices(first_features.shape[1]))
        )
        excluded.update(
            range(*self.feature_slices["latitude"].indices(first_features.shape[1]))
        )
        self.semantic_indices = np.asarray(
            [
                index
                for index in range(first_features.shape[1])
                if index not in excluded
            ],
            dtype=np.int64,
        )
        if self.semantic_indices.size < 2:
            raise ValueError("geo-semantic blend requires non-geographic metadata")
        semantic = first_features[:, self.semantic_indices]
        self.semantic_mean = semantic.mean(axis=0)
        centered = semantic - self.semantic_mean
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        if right_vectors.shape[0] < 2:
            raise ValueError("semantic feature matrix does not support a 2D PCA")
        self.semantic_components = self._deterministic_component_signs(
            right_vectors[:2]
        )
        total_variance = float((singular_values**2).sum())
        if total_variance <= 1e-12:
            raise ValueError("static first-period semantic features have zero variance")
        self.explained_variance_ratio = (
            singular_values[:2] ** 2 / total_variance
        ).astype(np.float64)
        self.feature_mean = first_features.mean(axis=0)

    @staticmethod
    def _normalize_to_shared_range(values: np.ndarray) -> np.ndarray:
        """Map each 2D branch to the same per-period [-1, 1] square."""
        minimum = values.min(axis=0, keepdims=True)
        span = values.max(axis=0, keepdims=True) - minimum
        normalized = np.divide(
            values - minimum,
            span,
            out=np.full_like(values, 0.5, dtype=np.float64),
            where=span > 1e-12,
        )
        return np.clip(2.0 * normalized - 1.0, -1.0, 1.0)

    def _encode_rows(self, rows: list[dict[str, str]]) -> np.ndarray:
        blocks = []
        for column in self.numeric_columns:
            values = np.asarray([_parse_numeric(row[column]) for row in rows])
            missing = ~np.isfinite(values)
            filled = values.copy()
            filled[missing] = self.numeric_median[column]
            standardized = (filled - self.numeric_mean[column]) / self.numeric_scale[
                column
            ]
            blocks.append(standardized[:, None])
            blocks.append(missing.astype(np.float64)[:, None])

        for column in self.categorical_columns:
            vocabulary = self.category_values[column]
            lookup = {value: index for index, value in enumerate(vocabulary)}
            unknown_index = lookup[_UNKNOWN]
            encoded = np.zeros((len(rows), len(vocabulary)), dtype=np.float64)
            for row_index, row in enumerate(rows):
                value = _clean_category(row[column])
                encoded[row_index, lookup.get(value, unknown_index)] = 1.0
            blocks.append(encoded)
        return np.concatenate(blocks, axis=1)

    def embedding_for_period(
        self,
        period: int,
        expected_nodes: int | None = None,
    ) -> np.ndarray:
        rows = self._canonical_period_rows(period)
        if expected_nodes is not None and len(rows) != int(expected_nodes):
            raise ValueError(
                f"period {period} has {len(rows)} static rows, "
                f"but the graph/data contains {expected_nodes} nodes"
            )
        features = self._encode_rows(rows)
        geo = self._normalize_to_shared_range(features[:, self.geo_indices])
        semantic = features[:, self.semantic_indices]
        semantic_embedding = self._normalize_to_shared_range(
            (semantic - self.semantic_mean) @ self.semantic_components.T
        )
        embedding = (
            1.0 - self.blend_alpha
        ) * geo + self.blend_alpha * semantic_embedding
        if not np.isfinite(embedding).all():
            raise ValueError(f"period {period} produced non-finite static coordinates")
        return embedding.astype(np.float32)

    def sensor_ids_for_period(self, period: int) -> list[str]:
        if period not in self.rows_by_period:
            raise KeyError(f"period {period} is outside the static-feature stream")
        return [row[self.id_column] for row in self.rows_by_period[period]]

    def summary(self) -> dict:
        feature_dimension = int(self.feature_mean.shape[0])
        return {
            "reducer": "fixed_first_period_geo_semantic_blend",
            "fit_period": self.periods[0],
            "periods": list(self.periods),
            "sensor_counts": {
                period: len(self.rows_by_period[period]) for period in self.periods
            },
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "categorical_vocabulary_sizes": {
                column: len(values) for column, values in self.category_values.items()
            },
            "feature_dimension": feature_dimension,
            "blend_alpha": self.blend_alpha,
            "branch_range": [-1.0, 1.0],
            "range_scope": "per_period",
            "blend_formula": "(1-alpha)*geography + alpha*semantics",
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "explained_variance_ratio_sum": float(self.explained_variance_ratio.sum()),
        }
