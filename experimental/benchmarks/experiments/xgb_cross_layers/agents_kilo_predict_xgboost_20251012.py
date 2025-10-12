import os
import pandas as pd
try:
    from xgboost import XGBRegressor
except ImportError as e:
    print("Error: scikit-learn (sklearn) is required for XGBRegressor. Install it with:")
    print("    pip install scikit-learn")
    raise e
from scipy.stats import pearsonr

def load_train_data(layers):
    X_list = []
    y_list = []
    for layer in layers:
        X_path = os.path.join(layer, "features.csv")
        y_path = os.path.join(layer, "parsed_trials.csv")
        X_data = pd.read_csv(X_path, header=None)
        # Fix: set column names to strings for XGBoost compatibility
        X_data.columns = [f"f{c}" for c in range(X_data.shape[1])]
        y_data = pd.read_csv(y_path)["gflops"].str.rstrip('.').replace(r"[^\d\.eE\-]", "", regex=True)
        y_data = pd.to_numeric(y_data, errors="coerce")
        y_data = y_data.fillna(0)
        X_list.append(X_data)
        y_list.append(y_data)
    X_train = pd.concat(X_list, ignore_index=True)
    y_train = pd.concat(y_list, ignore_index=True)
    return X_train, y_train

def load_val_data(layer):
    X_path = os.path.join(layer, "features.csv")
    y_path = os.path.join(layer, "parsed_trials.csv")
    X_val = pd.read_csv(X_path, header=None)
    # Fix: set column names to strings for XGBoost compatibility
    X_val.columns = [f"f{c}" for c in range(X_val.shape[1])]
    y_val = pd.read_csv(y_path)["gflops"].str.rstrip('.').replace(r"[^\d\.eE\-]", "", regex=True)
    y_val = pd.to_numeric(y_val, errors="coerce")
    y_val = y_val.fillna(0)
    config_ids = list(range(len(y_val)))
    return X_val, y_val, config_ids

def main():
    train_layers = ["layer_0", "layer_1"]
    val_layers = ["layer_10", "layer_11"]
    
    # Load training data
    X_train, y_train = load_train_data(train_layers)
    
    # Train XGBoost model
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    
    # Collect predictions and evaluation
    all_y = []
    all_preds = []
    results = []
    for layer in val_layers:
        X_val, y_val, config_ids = load_val_data(layer)
        preds = model.predict(X_val)
        for cid, pred, actual in zip(config_ids, preds, y_val):
            results.append({
                "layer": layer,
                "config_id": cid,
                "predicted_gflops": pred,
                "real_gflops": actual
            })
        all_y.extend(list(y_val))
        all_preds.extend(list(preds))
    
    # Calculate Pearson correlation coefficient
    pearson_corr, _ = pearsonr(all_preds, all_y)
    print(f"Pearson Correlation: {pearson_corr:.4f}")
    
    # Save prediction vs real CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("prediction_vs_real.csv", index=False)
    
    # Save Pearson correlation in a text file
    with open("pearson.txt", "w") as f:
        f.write(str(pearson_corr))

if __name__ == "__main__":
    main()