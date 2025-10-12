# XGBoost Cross Layers Experiment

This directory contains an implementation for predicting the performance (GFLOPS) of different configuration layers using XGBoost regression.

## Main Script

[`agents_kilo_predict_xgboost_20251012.py`](experiments/xgb_cross_layers/agents_kilo_predict_xgboost_20251012.py:1) trains an XGBoost regressor to predict GFLOPS values for layer configurations.

### Workflow

1. **Data Loading**
   - Training data is loaded from `layer_0` and `layer_1`.
   - Validation data is loaded from `layer_10` and `layer_11`.
   - Each layer provides:
     - `features.csv`: Feature vectors for each configuration.
     - `parsed_trials.csv`: Actual GFLOPS values.

2. **Model Training**
   - Uses `XGBRegressor` with 200 estimators, max depth 6, and learning rate 0.05.
   - Trained on concatenated features and GFLOPS from training layers.

3. **Prediction & Evaluation**
   - Predicts GFLOPS for validation layers.
   - Calculates Pearson correlation coefficient between predicted and actual GFLOPS.
   - Saves results:
     - `prediction_vs_real.csv`: Contains predicted and actual GFLOPS for each configuration.
     - `pearson.txt`: Pearson correlation value.

## Requirements

- Python 3
- pandas
- xgboost
- scipy

Install dependencies with:
```bash
pip install pandas xgboost scipy
```

## Outputs

- `prediction_vs_real.csv`: Predicted vs. real GFLOPS for validation layers.
- `pearson.txt`: Pearson correlation coefficient.

## Usage

Run the script:
```bash
python agents_kilo_predict_xgboost_20251012.py
```
