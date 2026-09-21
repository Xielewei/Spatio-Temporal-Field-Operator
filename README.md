# STFO

Code for the STFO main experiments on PEMS, CA, and AIR.

## Installation

Use Python 3.11 and install the dependencies in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is needed for GPU training. Use `--gpuid -1` for CPU execution.

## Repository

```text
main.py                 Training and evaluation entry point
conf/                   PEMS, CA, and AIR experiment configurations
src/model/stfo/         STFO: CFE, SRE, DFO, and CQD
src/data/               Window datasets, preprocessing cache, and sensor coordinates
src/trainer/            Sequential training and validation-selected checkpoint averaging
utils/                  Masked losses, metrics, preprocessing, seeds, and checkpoint loading
```

## Data preparation

Download [STFO_data.zip](https://github.com/Xielewei/STFO/releases/download/v1.0.0/STFO_data.zip) from the [data release](https://github.com/Xielewei/STFO/releases/tag/v1.0.0). The archive contains the exact period-level inputs used by the main experiments: time series, adjacency matrices, and aligned sensor coordinate files for all three datasets. It contains 45 files covering all 15 periods. Processed window caches and trained weights are not included.

Download and extract the data from inside the repository directory:

```bash
curl -L --fail -o STFO_data.zip https://github.com/Xielewei/STFO/releases/download/v1.0.0/STFO_data.zip
unzip STFO_data.zip -d .
python main.py --conf conf/STFO_PEMS.json --size L --seed 42 --gpuid 0
```

The extraction creates `data/PEMS/`, `data/CA/`, and `data/AIR/` with the layout below. No manual sensor matching or raw-source conversion is required. The first run generates the processed window cache automatically. Alternatively, extract the archive elsewhere and pass the resulting `data/` directory with `--data-root`.

The input layout is:

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

## Main experiments

Run from the repository directory:

```bash
python main.py --conf conf/STFO_PEMS.json --size L --seed 42 --gpuid 0
python main.py --conf conf/STFO_CA.json --size L --seed 42 --gpuid 0
python main.py --conf conf/STFO_AIR.json --size S --seed 42 --gpuid 0
```

`--size S`, `M`, and `L` select hidden widths 32, 64, and 128. Repeat each dataset/size configuration with seeds **42, 43, and 44** for the main-table runs. Without `--size`, the dataset defaults are PEMS-L, CA-L, and AIR-S. All other dataset-specific model and optimization settings are supplied by the configurations.

Each run trains the first period and then fine-tunes sequentially on subsequent periods. Training uses AdamW, masked MAE, gradient clipping, and validation-based early stopping. The checkpoint procedure compares the best single validation checkpoint with averages of up to three leading checkpoints and accepts an average only when validation MAE improves by more than `0.0001`. Test data is used only for final evaluation.

Outputs are written to `outputs/{dataset}_STFO-{size}_seed{seed}/`. Each period has training checkpoints and `checkpoint_soup.json`, which records the validation selection. `metrics.csv` contains period-level MAE, RMSE, and MAPE for horizons 3, 6, 12, and the average over all 12 horizons; `metrics.json` also contains unweighted period means. MAPE is expressed as a percentage. The run directory also stores its configuration and runtime log.

Use `--output` to select another run directory. Training refuses to overwrite a nonempty directory. `--workers` and `--epochs` override the corresponding configuration values when needed.

## Evaluation

Re-evaluate the validation-selected checkpoints from a completed run:

```bash
python main.py --conf outputs/PEMS_STFO-L_seed42/config.json \
  --output outputs/PEMS_STFO-L_seed42 --evaluate --gpuid 0
```

Pass the same `--data-root` if the data is stored externally. Evaluation writes `evaluation.csv` and `evaluation.json` in the run directory.
