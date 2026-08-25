"""
Unified Multi-Scale Comparison & Benchmarking Engine
Evaluates Scikit-Learn vs PySpark MLlib performance on FMCG (1M, 3M, 5M) Sales Prediction
"""

import os
import sys
import json
import argparse
import pandas as pd

def print_banner(text):
    print("\n" + "=" * 85)
    print(f" {text}")
    print("=" * 85)

def compare_results(results_csv="results/experiment_results.csv", metrics_json="results/pipeline_metrics.json"):
    print_banner("FMCG SALES PREDICTION: MULTI-SCALE (1M, 3M, 5M) BENCHMARK")
    
    if not os.path.exists(results_csv):
        print(f"[Error] Results file '{results_csv}' not found. Please run train_sklearn.py and train_pyspark.py first.")
        return
        
    df = pd.read_csv(results_csv)
    
    if os.path.exists(metrics_json):
        with open(metrics_json, "r") as f:
            pipeline_data = json.load(f)
            print(f"Medallion Pipeline Duration: {pipeline_data.get('overall_duration_sec', 'N/A')}s")
            print(f"Bronze Records Ingested: {pipeline_data.get('bronze', {}).get('records', 'N/A'):,}")
            print(f"Silver Cleaned Records:  {pipeline_data.get('silver', {}).get('final_records', 'N/A'):,}")
            if "gold_scales" in pipeline_data:
                for scale_name, g_info in pipeline_data["gold_scales"].items():
                    print(f"Gold [{scale_name}] Train/Test: {g_info.get('train_records', 'N/A'):,} / {g_info.get('test_records', 'N/A'):,}")
            print("-" * 85)
            
    print("\n[ MULTI-SCALE EXPERIMENT RESULTS TABLE ]")
    display_cols = ["Framework", "Model", "Data Scale", "Data Rows", "Training Time (s)", "Prediction Time (s)", "Total Time (s)", "RMSE", "MAE", "R2"]
    available_cols = [c for c in display_cols if c in df.columns]
    
    # Sort nicely by Scale (1M, 3M, 5M), Framework, Model
    scale_order = {"1M": 1, "3M": 2, "5M": 3}
    if "Data Scale" in df.columns:
        df["_scale_rank"] = df["Data Scale"].map(lambda s: scale_order.get(str(s).upper(), 99))
        df_sorted = df.sort_values(by=["_scale_rank", "Framework", "Model"]).drop(columns=["_scale_rank"])
    else:
        df_sorted = df.sort_values(by=["Data Rows", "Framework", "Model"])
        
    print(df_sorted[available_cols].to_string(index=False))
    
    # Scaling Analysis across 1M, 3M, 5M if multiple scales exist
    if "Data Scale" in df.columns and len(df["Data Scale"].dropna().unique()) > 1:
        print("\n" + "-" * 85)
        print("[ SCALABILITY & SPEEDUP ANALYSIS ACROSS 1M -> 3M -> 5M ]")
        print("-" * 85)
        
        pivot_train = df.pivot_table(index=["Framework", "Model"], columns="Data Scale", values="Training Time (s)")
        print("\nTraining Time by Dataset Scale (seconds):")
        print(pivot_train.to_string())
        
        # Model Speedups (Single-Node Time / Distributed Time) across all 4 models
        print("\nDistributed vs Single-Node Speedups (Single-Node Time / Distributed Time):")
        for m in ["Random Forest", "XGBoost", "CatBoost", "LightGBM"]:
            print(f"\n ► {m}:")
            for s in ["1M", "3M", "5M"]:
                sn_row = df[(df["Model"] == m) & (df["Framework"] == "Single-Node") & (df["Data Scale"] == s)]
                dist_row = df[(df["Model"] == m) & (df["Framework"] == "Distributed") & (df["Data Scale"] == s)]
                if not sn_row.empty and not dist_row.empty:
                    sn_time = sn_row.iloc[0]["Training Time (s)"]
                    dist_time = dist_row.iloc[0]["Training Time (s)"]
                    speedup = sn_time / dist_time if dist_time > 0 else 0
                    if speedup >= 1.0:
                        verdict = f"Distributed is {speedup:.2f}x FASTER"
                    else:
                        verdict = f"Single-Node is {1.0/speedup:.2f}x FASTER"
                    print(f"   • Scale {s:>2}: Single-Node = {sn_time:>7.2f}s | Distributed = {dist_time:>6.2f}s  ==>  {verdict}")

    print("\n" + "-" * 85)
    print("[ KEY RESEARCH FINDINGS & COMPARATIVE ANALYSIS ]")
    print("-" * 85)
    
    # Best overall models
    best_rmse_row = df.loc[df["RMSE"].idxmin()]
    best_r2_row = df.loc[df["R2"].idxmax()]
    fastest_train_row = df.loc[df["Training Time (s)"].idxmin()]
    
    print(f"• Lowest Error (Best RMSE):  {best_rmse_row['Framework']} - {best_rmse_row['Model']} ({best_rmse_row.get('Data Scale', 'N/A')}) (RMSE: {best_rmse_row['RMSE']})")
    print(f"• Highest Explained Variance: {best_r2_row['Framework']} - {best_r2_row['Model']} ({best_r2_row.get('Data Scale', 'N/A')}) (R²: {best_r2_row['R2']})")
    print(f"• Fastest Model Training:     {fastest_train_row['Framework']} - {fastest_train_row['Model']} ({fastest_train_row.get('Data Scale', 'N/A')}) ({fastest_train_row['Training Time (s)']}s)")
    
    print("\n[ ARCHITECTURAL TAKEAWAYS ]")
    print("1. Medallion Standardization: PySpark Bronze->Silver->Gold guarantees consistent, leak-free splits across 1M, 3M, and 5M datasets.")
    print("2. Distributed Scalability: As data scales from 1M to 5M rows, Scikit-learn CPU & single-node memory wall becomes pronounced, whereas PySpark MLlib distributes across executor nodes.")
    print("3. Production Readiness: Spark MLlib models integrate natively into large-scale distributed enterprise batch/streaming pipelines without requiring pandas downsampling.")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Comparison Engine")
    parser.add_argument("--results-csv", default="results/experiment_results.csv")
    parser.add_argument("--metrics-json", default="results/pipeline_metrics.json")
    args = parser.parse_args()
    compare_results(args.results_csv, args.metrics_json)

