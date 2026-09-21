"""STFO sensor coordinates for sequential forecasting."""

from .grid_coordinates import spatial_grid_coordinates
from .physical_coordinates import LongitudeLatitudeCoordinateSystem
from .static_feature_coordinates import StaticFeatureCoordinateSystem


class STFOCoordinateSystem:
    """Assign sensors to a grid using geography and frozen static features."""

    def __init__(self, args):
        if getattr(args, "coord_chart_scope", None) != "per_period":
            raise ValueError("STFO requires coord_chart_scope 'per_period'")
        numeric = tuple(getattr(args, "static_numeric_columns", ()))
        categorical = tuple(getattr(args, "static_categorical_columns", ()))
        self.geography_only = not (
            set(numeric) - {"longitude", "latitude"} or categorical
        )
        common = dict(
            begin_period=args.begin_year,
            end_period=args.end_year,
            path_template=args.coord_path,
            id_column=args.coord_id_column,
            require_nested=bool(getattr(args, "coord_require_nested", True)),
        )
        if self.geography_only:
            self.source = LongitudeLatitudeCoordinateSystem(**common)
        else:
            self.source = StaticFeatureCoordinateSystem(
                **common,
                numeric_columns=numeric,
                categorical_columns=categorical,
                blend_alpha=float(getattr(args, "static_blend_alpha", 0.10)),
            )

    def summary(self):
        return self.source.summary()

    def coordinates_for_period(self, year, expected_nodes):
        if self.geography_only:
            embedding = self.source.longitude_latitude_for_period(
                year, expected_nodes=expected_nodes
            )
            sensor_ids = [
                record.sensor_id for record in self.source.records_by_period[year]
            ]
        else:
            embedding = self.source.embedding_for_period(
                year, expected_nodes=expected_nodes
            )
            sensor_ids = self.source.sensor_ids_for_period(year)
        coords, info = spatial_grid_coordinates(embedding, sensor_ids)
        info.update(
            embedding_x_min=float(embedding[:, 0].min()),
            embedding_x_max=float(embedding[:, 0].max()),
            embedding_y_min=float(embedding[:, 1].min()),
            embedding_y_max=float(embedding[:, 1].max()),
        )
        return coords, info
