"""
Medallion Pipeline for Distributed FMCG Sales Dataset (PySpark)
Bronze -> Silver -> Gold Architecture
"""

import os
import sys
import time
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType
)
from pyspark.sql.functions import (
    col, current_timestamp, lit, when, ceil, to_date, count, expr
)

def get_spark_session(app_name="FMCG_Medallion_Pipeline", master=None):
    java_options = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    )
    builder = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.extraJavaOptions", java_options) \
        .config("spark.executor.extraJavaOptions", java_options)
    
    if master:
        builder = builder.master(master)
    else:
        builder = builder.master("local[*]")
        
    return builder.getOrCreate()

def get_raw_schema():
    return StructType([
        StructField("date", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("month", IntegerType(), True),
        StructField("day", IntegerType(), True),
        StructField("weekofyear", IntegerType(), True),
        StructField("weekday", IntegerType(), True),
        StructField("is_weekend", IntegerType(), True),
        StructField("is_holiday", IntegerType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("rain_mm", DoubleType(), True),
        StructField("store_id", StringType(), True),
        StructField("country", StringType(), True),
        StructField("city", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("sku_id", StringType(), True),
        StructField("sku_name", StringType(), True),
        StructField("category", StringType(), True),
        StructField("subcategory", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("units_sold", IntegerType(), True),
        StructField("list_price", DoubleType(), True),
        StructField("discount_pct", DoubleType(), True),
        StructField("promo_flag", IntegerType(), True),
        StructField("gross_sales", DoubleType(), True),
        StructField("net_sales", DoubleType(), True),
        StructField("stock_on_hand", IntegerType(), True),
        StructField("stock_out_flag", IntegerType(), True),
        StructField("lead_time_days", IntegerType(), True),
        StructField("supplier_id", StringType(), True),
        StructField("purchase_cost", DoubleType(), True),
        StructField("margin_pct", DoubleType(), True),
    ])

def create_bronze(spark, raw_csv_path, bronze_path):
    print("\n" + "="*50)
    print("STAGE 1: INGESTION TO BRONZE LAYER")
    print("="*50)
    start_time = time.time()
    
    schema = get_raw_schema()
    df_raw = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(raw_csv_path)
    
    # Add metadata
    df_bronze = df_raw \
        .withColumn("ingestion_timestamp", current_timestamp()) \
        .withColumn("source_file", lit(os.path.basename(raw_csv_path)))
    
    # Save to Parquet
    df_bronze.write \
        .mode("overwrite") \
        .parquet(bronze_path)
    
    duration = time.time() - start_time
    total_records = df_bronze.count()
    col_count = len(df_bronze.columns)
    print(f"Bronze Layer completed in {duration:.2f}s")
    print(f"Total Bronze Records: {total_records:,} | Columns: {col_count}")
    
    return {
        "records": total_records,
        "columns": col_count,
        "duration_sec": round(duration, 2),
        "path": bronze_path
    }

def clean_to_silver(spark, bronze_path, silver_path):
    print("\n" + "="*50)
    print("STAGE 2: DATA QUALITY & CLEANING TO SILVER LAYER")
    print("="*50)
    start_time = time.time()
    
    df_bronze = spark.read.parquet(bronze_path)
    initial_count = df_bronze.count()
    
    # 1. Grain Deduplication: [date, store_id, sku_id]
    df_dedup = df_bronze.dropDuplicates(subset=["date", "store_id", "sku_id"])
    after_dedup = df_dedup.count()
    dedup_removed = initial_count - after_dedup
    
    # 2. Filtering & Boundary Validation
    df_cleaned = df_dedup.filter(
        (col("units_sold") > 0) &
        (col("list_price") > 0) &
        (col("discount_pct") >= 0.0) & (col("discount_pct") <= 100.0) &
        (col("temperature") >= -30.0) & (col("temperature") <= 60.0) &
        (col("rain_mm") >= 0.0) &
        (col("stock_on_hand") >= 0) &
        (col("lead_time_days") >= 0) &
        (col("purchase_cost") > 0) &
        (col("margin_pct") >= -1.0) & (col("margin_pct") <= 1.0) &
        col("country").isNotNull() & (col("country") != "") &
        col("city").isNotNull() & (col("city") != "") &
        col("channel").isNotNull() & (col("channel") != "") &
        col("category").isNotNull() & (col("category") != "") &
        col("brand").isNotNull() & (col("brand") != "")
    )
    
    # Write to Silver
    df_cleaned.write \
        .mode("overwrite") \
        .parquet(silver_path)
    
    duration = time.time() - start_time
    final_count = df_cleaned.count()
    invalid_removed = initial_count - final_count
    col_count = len(df_cleaned.columns)
    
    print(f"Silver Layer completed in {duration:.2f}s")
    print(f"Initial: {initial_count:,} | Final Silver: {final_count:,} | Removed Invalid/Dupes: {invalid_removed:,}")
    
    return {
        "initial_records": initial_count,
        "final_records": final_count,
        "duplicates_removed": dedup_removed,
        "invalid_removed": invalid_removed,
        "columns": col_count,
        "duration_sec": round(duration, 2),
        "path": silver_path
    }

def create_gold(spark, silver_path, gold_dir, train_ratio=0.8, seed=42, sample_fraction=None, scale_name="5M"):
    print("\n" + "="*50)
    print(f"STAGE 3: FEATURE ENGINEERING & GOLD ML SPLIT [{scale_name}]")
    print("="*50)
    start_time = time.time()
    
    df_silver = spark.read.parquet(silver_path)
    
    if sample_fraction is not None and sample_fraction < 1.0:
        print(f"Sampling {sample_fraction*100:.1f}% rows for scale {scale_name}...")
        df_silver = df_silver.sample(fraction=sample_fraction, seed=seed)
    
    # Feature Engineering
    df_features = df_silver \
        .withColumn("quarter", ceil(col("month") / 3).cast(IntegerType())) \
        .withColumn(
            "season",
            when(col("month").isin(12, 1, 2), "Winter")
            .when(col("month").isin(3, 4, 5), "Spring")
            .when(col("month").isin(6, 7, 8), "Summer")
            .otherwise("Autumn")
        ) \
        .withColumn(
            "weekend_holiday",
            when((col("is_weekend") == 1) | (col("is_holiday") == 1), 1).otherwise(0)
        ) \
        .withColumn(
            "discount_amount",
            (col("list_price") * (col("discount_pct") / 100.0)).cast(DoubleType())
        ) \
        .withColumn(
            "effective_price",
            (col("list_price") - col("discount_amount")).cast(DoubleType())
        )
    
    # Explicitly DROP mathematical target leakage columns
    leakage_cols = ["gross_sales", "net_sales", "ingestion_timestamp", "source_file"]
    for c in leakage_cols:
        if c in df_features.columns:
            df_features = df_features.drop(c)
            
    # Train / Test split
    train_df, test_df = df_features.randomSplit([train_ratio, 1.0 - train_ratio], seed=seed)
    
    train_path = os.path.join(gold_dir, "train.parquet")
    test_path = os.path.join(gold_dir, "test.parquet")
    
    train_df.write.mode("overwrite").parquet(train_path)
    test_df.write.mode("overwrite").parquet(test_path)
    
    duration = time.time() - start_time
    train_count = train_df.count()
    test_count = test_df.count()
    total_gold = train_count + test_count
    col_count = len(df_features.columns)
    
    print(f"Gold Layer [{scale_name}] completed in {duration:.2f}s")
    print(f"Total Gold: {total_gold:,} | Train: {train_count:,} (80%) | Test: {test_count:,} (20%)")
    print(f"Engineered Features: quarter, season, weekend_holiday, discount_amount, effective_price")
    print(f"Target Leakage Columns Excluded: {leakage_cols}")
    
    return {
        "scale": scale_name,
        "total_records": total_gold,
        "train_records": train_count,
        "test_records": test_count,
        "features_count": col_count,
        "feature_list": df_features.columns,
        "target": "units_sold",
        "duration_sec": round(duration, 2),
        "train_path": train_path,
        "test_path": test_path
    }

def run_pipeline(raw_csv_path, base_data_dir="data", results_dir="results", master=None, scales=["1M", "3M", "5M"]):
    os.makedirs(base_data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    bronze_path = os.path.join(base_data_dir, "bronze")
    silver_path = os.path.join(base_data_dir, "silver")
    
    print(f"Starting Medallion Pipeline for {raw_csv_path} with scales: {scales}")
    overall_start = time.time()
    
    spark = get_spark_session("FMCG_Medallion_Pipeline", master=master)
    
    try:
        bronze_metrics = create_bronze(spark, raw_csv_path, bronze_path)
        silver_metrics = clean_to_silver(spark, bronze_path, silver_path)
        
        gold_metrics_map = {}
        for scale in scales:
            scale_upper = scale.strip().upper()
            if scale_upper == "1M":
                frac = 1.0 / 5.0
                g_dir = os.path.join(base_data_dir, "gold_1M")
            elif scale_upper == "3M":
                frac = 3.0 / 5.0
                g_dir = os.path.join(base_data_dir, "gold_3M")
            elif scale_upper == "5M":
                frac = None
                g_dir = os.path.join(base_data_dir, "gold_5M")
            else:
                frac = None
                g_dir = os.path.join(base_data_dir, f"gold_{scale}")
                
            scale_res = create_gold(spark, silver_path, g_dir, sample_fraction=frac, scale_name=scale_upper)
            gold_metrics_map[scale_upper] = scale_res
            
            # Keep default data/gold pointing to 5M (or primary scale)
            if scale_upper == "5M":
                primary_gold_dir = os.path.join(base_data_dir, "gold")
                create_gold(spark, silver_path, primary_gold_dir, sample_fraction=None, scale_name="Primary (5M)")
        
        overall_duration = round(time.time() - overall_start, 2)
        
        metrics = {
            "pipeline_name": "FMCG Medallion Architecture",
            "source_file": raw_csv_path,
            "overall_duration_sec": overall_duration,
            "bronze": bronze_metrics,
            "silver": silver_metrics,
            "gold_scales": gold_metrics_map,
            "spark_version": spark.version
        }
        
        metrics_file = os.path.join(results_dir, "pipeline_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print("\n" + "="*50)
        print(f"MEDALLION PIPELINE COMPLETED SUCCESSFULLY IN {overall_duration:.2f}s")
        print(f"Metrics saved to {metrics_file}")
        print("="*50 + "\n")
        
        return metrics
        
    finally:
        spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark FMCG Medallion Pipeline")
    parser.add_argument("--input", default="data/raw/fmcg_sales_5M_rows.csv", help="Path to raw CSV file")
    parser.add_argument("--data-dir", default="data", help="Root data directory")
    parser.add_argument("--results-dir", default="results", help="Directory for pipeline metrics")
    parser.add_argument("--master", default=None, help="Spark master URL (e.g. spark://localhost:7077)")
    parser.add_argument("--scales", default="1M,3M,5M", help="Comma-separated list of scales to build (e.g. 1M,3M,5M)")
    
    args = parser.parse_args()
    
    # Fallback to local file if not found in data/raw
    input_file = args.input
    if not os.path.exists(input_file):
        if os.path.exists("fmcg_sales_5M_rows.csv"):
            input_file = "fmcg_sales_5M_rows.csv"
        elif os.path.exists("fmcg_sales_3years_1M_rows.csv"):
            input_file = "fmcg_sales_3years_1M_rows.csv"
            
    scales_list = [s.strip() for s in args.scales.split(",") if s.strip()]
    run_pipeline(input_file, args.data_dir, args.results_dir, master=args.master, scales=scales_list)
