# LSTM baseline for `final_train_dataset_19to25_master.csv`

This is a deliberately simple first LSTM run. It does not create derived weather
features, interpolate records, or impute feature values.

## Inputs

- Sequence key: rows ordered by `Date` within each `STN_ID`
- Sequence length: 7 observations
- Model features: the existing 16 GK-2A channel columns plus `LAT`, `LON`, `ALT`
- Targets: `TA`, `HM`
- `Date`, `Time`, and `STN_ID` are used only for ordering, grouping, and output keys.
- ASOS `TA`/`HM` never enter the input sequence.

The only numerical transformation is train-set mean/std normalization inside the
model pipeline. Its statistics are saved in `lstm_baseline_model.npz`, and all
reported predictions are converted back to degrees Celsius and percent humidity.

## Time split

- Train: 2019-2023
- Validation: 2024
- Test: 2025

Rows whose current `TA` or `HM` label is missing are excluded from training and
evaluation. No missing feature value is filled; the script stops if one is found.

## Colab

Open `notebooks/Colab_LSTM_Baseline.ipynb` and run the cells from top to bottom.
The two paths to check are:

```python
DATA_CSV = "/content/drive/MyDrive/SME_DATA/processed_station_features/final_train_dataset_19to25_master.csv"
OUTPUT_DIR = "/content/drive/MyDrive/SME_DATA/processed_station_features/lstm_baseline_19to25"
```

## Command line

```bash
python scripts/train_lstm_baseline.py \
  --input-csv final_train_dataset_19to25_master.csv \
  --output-dir lstm_baseline_19to25 \
  --sequence-length 7 \
  --hidden-size 32 \
  --epochs 30 \
  --batch-size 512 \
  --learning-rate 0.002 \
  --patience 6
```

Outputs:

- `lstm_baseline_model.npz`: weights and normalization statistics
- `lstm_metrics.json`: validation/test metrics and run configuration
- `lstm_training_history.csv`: epoch history
- `lstm_test_predictions_2025.csv`: 2025 actual/predicted values

## First verified run

Using seed 42, hidden size 32, and a 7-observation sequence:

| Split | Target | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| 2024 validation | TA | 2.1084 | 2.6915 | 0.4517 |
| 2024 validation | HM | 8.8430 | 11.4154 | 0.5528 |
| 2025 test | TA | 1.8297 | 2.3619 | 0.6161 |
| 2025 test | HM | 8.8967 | 11.5809 | 0.4856 |

This is a baseline, not a tuned competition model. In particular, the sequence
uses observation order and does not add a time-gap feature between summer seasons.

