#!/usr/bin/env bash
# ==============================================================================
# FMCG PySpark vs Scikit-Learn Multi-Scale Benchmark (1M, 3M, 5M Rows)
# All-In-One Automated Execution Script
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_EXEC="python3"
if [ -d "venv" ] && [ -f "venv/bin/python" ]; then
    PYTHON_EXEC="./venv/bin/python"
fi

echo "======================================================================"
echo " FMCG DISTRIBUTED SPARK & SCIKIT-LEARN MULTI-SCALE BENCHMARK"
echo " Comparing Performance & Scalability across 1M, 3M, and 5M Rows"
echo "======================================================================"

# Step 1: Ensure Raw Dataset Exists
if [ ! -f "fmcg_sales_5M_rows.csv" ] && [ ! -f "data/raw/fmcg_sales_5M_rows.csv" ]; then
    echo -e "\n[Step 1/6] Generating 5 Million Row Synthetic FMCG Dataset..."
    $PYTHON_EXEC generate_synthetic_5m.py
else
    echo -e "\n[Step 1/6] Raw 5M Dataset already present."
fi

# Step 2: Run Medallion Pipeline (Bronze -> Silver -> Gold 1M, 3M, 5M)
echo -e "\n[Step 2/6] Running Medallion PySpark Pipeline for Scales: 1M, 3M, 5M..."
$PYTHON_EXEC src/pipeline.py --input fmcg_sales_5M_rows.csv --scales 1M,3M,5M

# Step 3: Start Spark Docker Cluster
echo -e "\n[Step 3/6] Starting Spark Docker Cluster (1 Master + 3 Workers)..."
docker compose up -d

echo "Waiting for Spark Master & Workers to register..."
sleep 5

# Ensure results directory has full write permissions for Docker containers
chmod -R 777 results data 2>/dev/null || true

# Step 4: Run Single-Node ML Benchmark (Scikit-Learn, LightGBM, XGBoost, CatBoost) across 1M, 3M, 5M
echo -e "\n[Step 4/6] Running Single-Node ML Training (Scikit-Learn, LightGBM, XGBoost, CatBoost) across 1M, 3M, 5M scales..."
for scale in 1M 3M 5M; do
    echo "--------------------------------------------------"
    echo ">> Training Single-Node Models on scale: ${scale}"
    echo "--------------------------------------------------"
    $PYTHON_EXEC src/train_sklearn.py --scale "${scale}"
done

# Step 5: Run Distributed PySpark MLlib Benchmark across 1M, 3M, 5M on Cluster
echo -e "\n[Step 5/6] Running Distributed PySpark MLlib on Spark Cluster across 1M, 3M, 5M..."
for scale in 1M 3M 5M; do
    echo "--------------------------------------------------"
    echo ">> Submitting PySpark MLlib job on scale: ${scale} to spark-master"
    echo "--------------------------------------------------"
    docker exec -t spark-master /opt/spark/bin/spark-submit \
        --master spark://spark-master:7077 \
        /opt/spark-apps/src/train_pyspark.py --scale "${scale}"
done

# Step 6: Output Comparative Benchmark Report
echo -e "\n[Step 6/6] Generating Consolidated Multi-Scale Comparison Benchmark..."
$PYTHON_EXEC src/compare.py

echo "======================================================================"
echo " BENCHMARK COMPLETE!"
echo " Results written to results/experiment_results.csv"
echo " "
echo " To launch the interactive visual dashboard, run:"
echo "   $PYTHON_EXEC -m streamlit run dashboard/app.py"
echo " (Open http://localhost:8501 in your browser)"
echo "======================================================================"
