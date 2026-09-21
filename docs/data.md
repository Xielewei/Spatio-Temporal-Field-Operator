# Data format and preprocessing

Download and extract the experiment archive as described in the [README](../README.md#data). The archive already follows this layout; no manual sensor matching or raw-source conversion is needed.

## Directory layout


```text
data/
  PEMS/
    RawData/{year}.npz
    graph/{year}_adj.npz
    coordinates/{year}.csv
  CA/
    RawData/{year}.npz
    graph/{year}_adj.npz
    coordinates/{year}.csv
  AIR/
    RawData/{year}.npz
    graph/{year}_adj.npz
    coordinates/stfo_air_{year}_node_coordinates_reconciled.csv
```

Each raw-data NPZ must contain `x` with shape `(time_steps, sensors)`; each graph NPZ must contain `x` with shape `(sensors, sensors)`. Raw-data columns, graph rows/columns, and CSV `node_index` values must refer to the same sensor order. CSV indices must cover `0` through `sensors - 1` exactly once. Sensor identifiers must remain stable across periods, and each period must retain the previous period's sensors.

| Dataset | Periods | ID column | Numeric columns | Categorical columns |
| --- | --- | --- | --- | --- |
| PEMS | 2011–2017 | `station_id` | `longitude`, `latitude`, `abs_pm`, `lanes` | `county`, `freeway`, `direction`, `station_type` |
| CA | 0–3 | `station_id` | `longitude`, `latitude` | `district`, `county`, `freeway`, `direction` |
| AIR | 2016–2019 | `station_code` | `longitude`, `latitude` | None |

Every coordinate CSV also requires `node_index`. Geographic coordinates use longitude/latitude in degrees. PEMS and CA combine geography with static features whose encoder is fitted on the first period; AIR uses geography. Sensors are assigned to the STFO spatial grid independently for each period.

Preprocessing uses chronological 60%/20%/20% train/validation/test splits, 12 input steps, and 12 forecast steps. PEMS uses the first `31 × 288` time steps of each period. Input and target normalization statistics are fitted on training windows and reused for validation/test. Processed windows are generated automatically under `data/FastData/` and can be redirected with `--cache-dir`. Use `--data-root` to point to a directory containing the dataset folders.


## Data sources

We thank [STBP](https://github.com/Aoyu-Liu/STBP) for the continual forecasting benchmark and data pipeline, and [EAC](https://github.com/Onedean/EAC) for making the PEMS-Stream and AIR-Stream datasets available to the research community. The STBP repository also provides the upstream CA-Stream data link.

The release bundles the period-level time series, graphs, and aligned sensor metadata used by STFO. Please acknowledge the original dataset providers and benchmark authors when using these datasets.
