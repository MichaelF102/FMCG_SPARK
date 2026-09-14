"""
FMCG Big Data ML Benchmark: Single-Node vs Distributed (PySpark)
Main Entry Point: Executive Project Overview & Architecture Suite
"""

import os
import sys

dash_dir = os.path.dirname(os.path.abspath(__file__))
if dash_dir not in sys.path:
    sys.path.insert(0, dash_dir)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    inject_custom_css, render_sidebar, render_kpi_card, 
    render_interpretation_box, render_viva_insight,
    render_pipeline_stepper, render_mermaid_diagram,
    apply_plot_layout, load_metrics_json, load_experiment_results,
    load_aws_experiment_results, FRAMEWORK_COLORS, MODEL_COLORS
)

st.set_page_config(
    page_title="FMCG ML Benchmark: Overview & Architecture",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar()

pipeline_metrics = load_metrics_json()
results_df = load_experiment_results()
aws_df = load_aws_experiment_results()

# --- HEADER SECTION ---
st.markdown('<div class="main-header">⚡ FMCG Sales Prediction: Single-Node vs Distributed Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Empirical benchmark of single-node and distributed machine learning across 1M, 3M, 5M, and 10M retail transactions.</div>', unsafe_allow_html=True)

# Top Horizontal Pipeline Stepper
render_pipeline_stepper()

# Dynamic KPI derivation from results_df and aws_df
max_sp_str = "1.95x"
max_sp_sub = "Random Forest (5M) • PySpark Cluster"
best_acc_str = "0.8761"
best_acc_sub = "R² Score (XGBoost 10M • RMSE 13.13)"

if not results_df.empty:
    speedups = []
    for s in ["1M", "3M", "5M"]:
        for m in ["Random Forest", "Linear Regression", "XGBoost"]:
            sn = results_df[(results_df["Framework"] == "Single-Node") & (results_df["Model"] == m) & (results_df["Data Scale"] == s)]
            dist = results_df[(results_df["Framework"] == "Distributed") & (results_df["Model"] == m) & (results_df["Data Scale"] == s)]
            if not sn.empty and not dist.empty:
                sn_t = float(sn.iloc[0]["Training Time (s)"])
                dist_t = float(dist.iloc[0]["Training Time (s)"])
                sp = sn_t / dist_t if dist_t > 0 else 1.0
                speedups.append((sp, m, s, sn_t, dist_t))
    if speedups:
        best_sp = max(speedups, key=lambda x: x[0])
        max_sp_str = f"{best_sp[0]:.2f}x"
        max_sp_sub = f"{best_sp[1]} ({best_sp[2]}) • {best_sp[4]:.1f}s vs {best_sp[3]:.1f}s"
    
    if "R2" in results_df.columns:
        max_r2_row = results_df.loc[results_df["R2"].astype(float).idxmax()]
        best_acc_str = f"{float(max_r2_row['R2']):.4f}"
        best_acc_sub = f"R² ({max_r2_row['Model']} {max_r2_row['Data Scale']} • RMSE {float(max_r2_row['RMSE']):.2f})"

# --- 6 HIGH-QUALITY COMPACT KPI CARDS ---
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
with kpi1:
    render_kpi_card("Dataset Scale", "1M to 10M", "36 Months Retail POS Transactions", "#38BDF8")
with kpi2:
    render_kpi_card("Models Tested", "3 Models", "Random Forest, Linear Reg, XGBoost", "#818CF8")
with kpi3:
    render_kpi_card("Frameworks", "2 Paradigms", "Single-Node (CPU) vs PySpark Cluster", "#F59E0B")
with kpi4:
    render_kpi_card("Max Speedup", max_sp_str, max_sp_sub, "#10B981")
with kpi5:
    render_kpi_card("Best Accuracy", best_acc_str, best_acc_sub, "#38BDF8")
with kpi6:
    render_kpi_card("Spark Cluster", "1 Master + 3 Workers", "Distributed JVM Memory & Cores", "#0284C7")

st.markdown("<br>", unsafe_allow_html=True)

# --- GROUP INFORMATION ---
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid #334155; border-left: 4px solid #818CF8; border-radius: 8px; padding: 14px 20px; margin-bottom: 18px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                👥 Project Information & Contributors
            </div>
            <div style="font-size: 0.92rem; color: #E2E8F0; font-weight: 500; line-height: 1.8;">
                <div>📌 <b>Topic</b>: Machine Learning Project using PySpark - FMCG Sales Prediction</div>
                <div>👤 <span style="color: #38BDF8; font-weight: 600;">Michael Fernandes</span> (2509006) &nbsp;|&nbsp; 👤 <span style="color: #38BDF8; font-weight: 600;">Manav Williams</span> (2509032) &nbsp;|&nbsp; 👤 <span style="color: #38BDF8; font-weight: 600;">Anshul Shashidhar</span> (2509012)</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- RESEARCH QUESTION & EXPERIMENTAL CONTROLS ---
col_rq, col_scope = st.columns([1, 1])
with col_rq:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #334155; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 16px 20px; height: 100%;">
        <div style="font-size: 0.78rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
            Central Research Question
        </div>
        <div style="font-size: 1.02rem; font-weight: 600; color: #F8FAFC; line-height: 1.45;">
            "How does distributed PySpark machine learning compare with single-node machine learning in terms of execution time, resource utilization, predictive accuracy, and scalability as FMCG dataset size increases from 1M to 3M, 5M, and 10M records?"
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_scope:
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #334155; border-left: 4px solid #10B981; border-radius: 8px; padding: 16px 20px; height: 100%;">
        <div style="font-size: 0.78rem; font-weight: 700; color: #10B981; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
            Key Experimental Controls
        </div>
        <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.55;">
            • <b>Target Variable</b>: Continuous regression on daily <code>units_sold</code>.<br/>
            • <b>Leakage Elimination</b>: <code>gross_sales</code> & <code>net_sales</code> strictly excluded.<br/>
            • <b>Fair Ground Truth</b>: Exact same 80/20 train/test Parquet partitions.<br/>
            • <b>Multi-Scale Benchmark</b>: 1M, 3M, 5M (Local) and 1M, 3M, 5M, 10M (AWS Cloud).
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- HOW TO NAVIGATE ---
st.subheader("🧭 Comprehensive Benchmark Suite Navigation")

nav_c1, nav_c2 = st.columns(2)
with nav_c1:
    st.markdown("""
    1. **01 📊 Dataset Exploration**: Inspect distributions, geographic channels, and statistical correlations across retail sales data.
    2. **02 🏅 Medallion Pipeline**: Understand PySpark's Bronze (raw), Silver (cleansed), and Gold (engineered) transformations and leakage elimination.
    3. **03 🌐 Spark Cluster Topology**: Explore the standalone Dockerized Spark Master and 3-Worker container topology with DAG flow.
    4. **04 ⚡ Single-Node vs Distributed Benchmark**: Access detailed head-to-head performance metrics, speedup heatmaps, and CPU/RAM telemetry.
    """)
with nav_c2:
    st.markdown("""
    5. **05 🔮 Live Sales Prediction**: Interactive simulation engine featuring real-time inference, model consensus, and discount elasticity.
    6. **06 🏆 Final Findings & Conclusions**: Review the comprehensive academic scorecard, hardware ceiling findings, and architecture decision matrix.
    7. **07 ☁️ AWS Results**: Inspect cloud-scale benchmarks extending up to **10,000,000 records** on AWS EC2 with full telemetry and comparison plots.
    """)

render_viva_insight(
    "Evaluation Rigor & Evaluation Integrity",
    "This benchmark rigorously evaluates whether distributed big data frameworks are computationally justified for modern tabular tree models. By enforcing shared Gold datasets, strict 80/20 train/test splits, and excluding target leakage variables, the comparison isolates purely computational scalability and memory headroom from data artifacts."
)

st.markdown("---")

# --- FULL END-TO-END DISTRIBUTED DATA FLOW ARCHITECTURE ---
st.markdown("### 🗺️ End-to-End Distributed Lakehouse & ML Benchmark Flow")
overview_mermaid = """
graph TD
    classDef raw fill:#78350F,stroke:#F59E0B,stroke-width:2.5px,color:#FEF3C7;
    classDef bronze fill:#C2410C,stroke:#FB923C,stroke-width:2.5px,color:#FFF7ED;
    classDef silver fill:#1E3A8A,stroke:#3B82F6,stroke-width:2.5px,color:#EFF6FF;
    classDef gold fill:#581C87,stroke:#A855F7,stroke-width:2.5px,color:#FAF5FF;
    classDef split fill:#831843,stroke:#EC4899,stroke-width:2.5px,color:#FDF2F8;
    classDef model fill:#064E3B,stroke:#10B981,stroke-width:2.5px,color:#ECFDF5;
    classDef cloud fill:#0C4A6E,stroke:#38BDF8,stroke-width:2.5px,color:#E0F2FE;

    RAW["📁 <b>Raw FMCG Sales CSV</b><br/><b>1M to 10M Records</b><br/>Multi-Store Point-of-Sale Data"]:::raw
    BRONZE["🥉 <b>Bronze Parquet Ingestion</b><br/><code>data/bronze/</code> • <b>14 Partitions</b><br/>Schema Validation & Snappy Compression"]:::bronze
    SILVER["🥈 <b>Silver Cleansed Lakehouse</b><br/><code>data/silver/</code> • <b>4.98M+ Retained</b><br/>Deduplicated & Boundary Filtered"]:::silver
    GOLD["🥇 <b>Gold Feature Store</b><br/><code>data/gold/</code> • <b>27 ML Features</b><br/>Target Leakage Strictly Dropped"]:::gold
    
    subgraph SPLITS ["📊 Multi-Scale 80/20 Train / Test Partitions"]
        TRAIN["<b>Train Partitions (80%)</b><br/>1M • 3M • 5M • 10M"]:::split
        TEST["<b>Test Partitions (20%)</b><br/>20% Unseen Holdout Evaluation"]:::split
    end
    
    subgraph ENGINES ["🏆 Benchmark Machine Learning Engines"]
        M_RF["🌲 <b>Random Forest</b><br/>PySpark MLlib vs Scikit-Learn"]:::model
        M_LR["📈 <b>Linear Regression</b><br/>PySpark MLlib vs Scikit-Learn"]:::model
        M_XGB["🚀 <b>XGBoost</b><br/>Distributed vs Single-Node"]:::model
    end

    subgraph CLOUD ["☁️ Multi-Environment Deployment"]
        LOCAL["💻 <b>Local Host</b><br/>Single-Node Multiprocessing"]:::cloud
        AWS["🌐 <b>AWS EC2 Cluster</b><br/>Scalable Cloud Infrastructure (1M–10M)"]:::cloud
    end

    RAW -->|"src/pipeline.py (ingest)"| BRONZE
    BRONZE -->|"src/pipeline.py (clean & validate)"| SILVER
    SILVER -->|"src/pipeline.py (feature engineering)"| GOLD
    GOLD -->|"split train"| TRAIN
    GOLD -->|"split test"| TEST
    
    TRAIN --> M_RF
    TRAIN --> M_LR
    TRAIN --> M_XGB
    
    M_RF --> LOCAL
    M_RF --> AWS
    M_LR --> LOCAL
    M_LR --> AWS
    M_XGB --> LOCAL
    M_XGB --> AWS
"""
render_mermaid_diagram(overview_mermaid, height=980)

st.markdown("---")
st.caption("FMCG Big Data Machine Learning Benchmark Suite • Developed by Michael Fernandes, Manav Williams, and Anshul Shashidhar")
