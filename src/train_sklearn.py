import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

# Optional high-performance gradient boosting libraries
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def load_gold_data(train_path="data/gold/train.parquet", test_path="data/gold/test.parquet", sample_frac=None):
    print("\n" + "="*50)
    print("SINGLE-NODE ML: LOADING GOLD DATASET")
    print("="*50)
    start_time = time.time()
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    if sample_frac and sample_frac < 1.0:
        print(f"Subsampling {sample_frac*100:.1f}% for fast iteration...")
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        test_df = test_df.sample(frac=sample_frac, random_state=42)
        
    duration = time.time() - start_time
    print(f"Loaded Train: {len(train_df):,} rows | Test: {len(test_df):,} rows in {duration:.2f}s")
    return train_df, test_df

def prepare_features(train_df, test_df, target_col="units_sold"):
    start_time = time.time()
    
    # Feature definition
    categorical_cols = ["country", "city", "channel", "category", "subcategory", "brand", "season"]
    numeric_cols = [
        "temperature", "rain_mm", "latitude", "longitude", "list_price",
        "discount_pct", "promo_flag", "stock_on_hand", "stock_out_flag",
        "lead_time_days", "purchase_cost", "margin_pct", "quarter",
        "weekend_holiday", "discount_amount", "effective_price",
        "year", "month", "day", "weekday"
    ]
    
    # Ensure all selected features exist
    available_cat = [c for c in categorical_cols if c in train_df.columns]
    available_num = [c for c in numeric_cols if c in train_df.columns]
    feature_cols = available_cat + available_num
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].values
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].values
    
    # Ordinal encode categorical features
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[available_cat] = encoder.fit_transform(X_train[available_cat].astype(str))
    X_test[available_cat] = encoder.transform(X_test[available_cat].astype(str))
    
    duration = time.time() - start_time
    print(f"Feature Preprocessing completed in {duration:.2f}s | {len(feature_cols)} features used")
    return X_train, y_train, X_test, y_test, feature_cols, duration

def append_result(csv_path, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame([row_dict])
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Avoid duplicate model records for same framework/model/scale
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        dedup_keys = ["Framework", "Model", "Data Scale"] if "Data Scale" in df_combined.columns else ["Framework", "Model", "Data Rows"]
        df_combined.drop_duplicates(subset=dedup_keys, keep="last", inplace=True)
        df_combined.to_csv(csv_path, index=False)
    else:
        df_new.to_csv(csv_path, index=False)
        
    print(f"Logged result for [{row_dict['Framework']} - {row_dict['Model']} ({row_dict.get('Data Scale', 'N/A')})] to {csv_path}")

def train_and_evaluate(train_path=None, 
                       test_path=None,
                       scale="5M",
                       results_csv="results/experiment_results.csv",
                       sample_frac=None):
    if train_path is None or test_path is None:
        if scale == "1M" and os.path.exists("data/gold_1M/train.parquet"):
            train_path = "data/gold_1M/train.parquet"
            test_path = "data/gold_1M/test.parquet"
        elif scale == "3M" and os.path.exists("data/gold_3M/train.parquet"):
            train_path = "data/gold_3M/train.parquet"
            test_path = "data/gold_3M/test.parquet"
        elif os.path.exists("data/gold_5M/train.parquet"):
            train_path = "data/gold_5M/train.parquet"
            test_path = "data/gold_5M/test.parquet"
        else:
            train_path = "data/gold/train.parquet"
            test_path = "data/gold/test.parquet"

    print(f"\nRunning Single-Node ML benchmark on scale [{scale}] with dataset: {train_path}")
    train_df, test_df = load_gold_data(train_path, test_path, sample_frac)
    total_rows = len(train_df) + len(test_df)
    
    X_train, y_train, X_test, y_test, feature_cols, prep_time = prepare_features(train_df, test_df)
    
    models = [
        ("Single-Node", "Random Forest", RandomForestRegressor(n_estimators=50, max_depth=12, n_jobs=-1, random_state=42))
    ]
    
    if HAS_LIGHTGBM:
        models.append(("Single-Node", "LightGBM", LGBMRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)))
        
    if HAS_XGBOOST:
        models.append(("Single-Node", "XGBoost", XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, n_jobs=-1, random_state=42, tree_method="hist")))
        
    if HAS_CATBOOST:
        models.append(("Single-Node", "CatBoost", CatBoostRegressor(iterations=100, depth=8, learning_rate=0.1, random_seed=42, verbose=0, thread_count=-1)))
    
    for framework, model_name, model in models:
        print("\n" + "-"*50)
        print(f"TRAINING [{framework} - {model_name}] [{scale}]")
        print("-"*50)
        
        # Train
        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        print(f"Training Time: {train_time:.2f}s")
        
        # Predict
        pred_start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - pred_start
        print(f"Inference Time: {pred_time:.2f}s")
        
        # Metrics
        rmse = float(root_mean_squared_error(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        total_time = prep_time + train_time + pred_time
        
        print(f"Results for [{framework} - {model_name}] [{scale}]:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  Total Runtime: {total_time:.2f}s")
        
        append_result(results_csv, {
            "Framework": framework,
            "Model": model_name,
            "Data Scale": scale,
            "Data Rows": total_rows,
            "Train Rows": len(train_df),
            "Test Rows": len(test_df),
            "Features": len(feature_cols),
            "Preprocessing Time (s)": round(prep_time, 2),
            "Training Time (s)": round(train_time, 2),
            "Prediction Time (s)": round(pred_time, 2),
            "Total Time (s)": round(total_time, 2),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "Cluster Nodes": 1,
            "Partitions": 1
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scikit-Learn ML Training")
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--test-path", default=None)
    parser.add_argument("--scale", default="5M", help="Scale to run (e.g. 1M, 3M, 5M, or all)")
    parser.add_argument("--results-csv", default="results/experiment_results.csv")
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional downsampling for quick testing")
    
    args = parser.parse_args()
    
    if args.scale.lower() == "all":
        for s in ["1M", "3M", "5M"]:
            train_and_evaluate(scale=s, results_csv=args.results_csv, sample_frac=args.sample_frac)
    else:
        train_and_evaluate(args.train_path, args.test_path, scale=args.scale.upper(), results_csv=args.results_csv, sample_frac=args.sample_frac)
