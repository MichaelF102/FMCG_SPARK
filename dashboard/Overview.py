"""
FMCG Big Data ML Benchmark: Single-Node vs Distributed (PySpark)
Main Entry Point: Overview & Executive Research Summary
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
    FRAMEWORK_COLORS, MODEL_COLORS
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

# --- HEADER SECTION ---
st.markdown('<div class="main-header">FMCG Sales Prediction: Single-Node vs Distributed Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Empirical benchmark of single-node and distributed machine learning across 1M, 3M and 5M FMCG transactions.</div>', unsafe_allow_html=True)

# Top Horizontal Pipeline Stepper (Screenshot Format)
render_pipeline_stepper()

# --- 6 HIGH-QUALITY COMPACT KPI CARDS ---
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
with kpi1:
    render_kpi_card("Dataset Scale", "5M Rows", "3 Years FMCG Retail Transactions", "#38BDF8")
with kpi2:
    render_kpi_card("Models Tested", "4 Models", "RF, LightGBM, XGBoost, CatBoost", "#818CF8")
with kpi3:
    render_kpi_card("Frameworks", "2 Paradigms", "Single-Node (CPU) vs PySpark Cluster", "#F59E0B")
with kpi4:
    render_kpi_card("Max Speedup", "10.32x", "XGBoost (1M) | 6.50x on RF (5M)", "#10B981")
with kpi5:
    render_kpi_card("Best Accuracy", "0.8760", "R² Score (XGBoost 5M • RMSE 13.14)", "#38BDF8")
with kpi6:
    render_kpi_card("Spark Cluster", "3 Workers", "6 Executor Cores • 6 GB Cluster RAM", "#0284C7")

st.markdown("<br>", unsafe_allow_html=True)

# --- PROJECT INFORMATION & GROUP MEMBERS ---
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid #334155; border-left: 4px solid #818CF8; border-radius: 8px; padding: 14px 20px; margin-bottom: 18px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #818CF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px;">
                📌 Topic Name
            </div>
            <div style="font-size: 1.02rem; font-weight: 700; color: #F8FAFC;">
                Machine Learning Project using PySpark - FMCG Sales Prediction
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid #334155; border-left: 4px solid #818CF8; border-radius: 8px; padding: 14px 20px; margin-bottom: 18px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-size: 0.72rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">
                👥 Group Members
            </div>
            <div style="font-size: 0.9rem; color: #E2E8F0; font-weight: 500; line-height: 1.8;">
                <div>
                    <span style="color: #38BDF8; font-weight: 600;">Michael Fernandes</span> (2509006)
                </div>
                <div>
                    <span style="color: #38BDF8; font-weight: 600;">Manav Williams</span> (2509032)
                </div>
                <div>
                    <span style="color: #38BDF8; font-weight: 600;">Anshul Shashidhar</span> (2509012)
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# --- RESEARCH QUESTION SECTION ---
st.markdown("""
<div style="background: rgba(15, 23, 42, 0.85); border: 1px solid #334155; border-left: 4px solid #38BDF8; border-radius: 8px; padding: 16px 20px; margin-bottom: 15px;">
    <div style="font-size: 0.78rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;">
        Central Research Question
    </div>
    <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC; line-height: 1.45;">
        "How does distributed PySpark machine learning compare with single-node machine learning in terms of execution time, resource utilization, predictive accuracy, and scalability as FMCG dataset size increases?"
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
#### 🎯 Research Scope & Experimental Standards
- **Multi-Scale Scaling**: Benchmarked at **1,000,000**, **3,000,000**, and **5,000,000** records to measure memory ceilings and parallel speedup.
- **Standardized Partitions**: Strict 80/20 Train/Test splits generated by a shared PySpark Medallion Lakehouse pipeline.
- **Target Leakage Prohibition**: Mathematically derived variables ($gross\\_sales = units\\_sold \\times price$ and $net\\_sales$) are strictly dropped to preserve predictive validity.
- **Algorithmic Parity**: Identical tree hyperparameters (depth, estimators, feature binning) evaluated on the exact same host compute environment.
""")

st.markdown("---")

# --- HOW TO NAVIGATE ---
st.subheader("🧭 How to Navigate this Benchmark Suite")

nav_c1, nav_c2 = st.columns(2)
with nav_c1:
    st.markdown("""
    1. **01 Dataset Exploration**: Inspect distributions, geographic channels, and summary statistics of the 5M retail dataset.
    2. **02 Medallion Pipeline**: Understand PySpark's Bronze, Silver, and Gold transformations and target leakage prevention.
    3. **03 Spark Cluster Topology**: Explore the Dockerized Spark Master and 3-Worker architecture with core and memory allocations.
    """)
with nav_c2:
    st.markdown("""
    4. **04 Benchmark & Telemetry**: Access the head-to-head performance comparisons, speedup heatmaps, and CPU/RAM telemetry.
    5. **05 Sales Prediction**: Run interactive  simulations with model consensus, financial impacts, and elasticity curves.
    6. **06 Findings & Conclusions**: Review the comprehensive academic scorecard and architectural decision matrix.
    """)

render_viva_insight(
    "Research Problem & Evaluation Integrity",
    "This benchmark rigorously addresses whether distributed big data frameworks are computationally justified for modern tabular tree models. By enforcing shared Gold datasets, strict 80/20 train/test splits, and excluding target leakage variables (gross_sales, net_sales), the comparison isolates purely computational scalability from data artifacts."
)


# --- FULL END-TO-END DISTRIBUTED DATA FLOW ARCHITECTURE (SCREENSHOT FORMAT) ---
st.markdown("### 🗺️ End-to-End Distributed Data Flow Architecture")
overview_mermaid = """
graph TD
    classDef raw fill:#78350F,stroke:#F59E0B,stroke-width:2.5px,color:#FEF3C7;
    classDef bronze fill:#C2410C,stroke:#FB923C,stroke-width:2.5px,color:#FFF7ED;
    classDef silver fill:#1E3A8A,stroke:#3B82F6,stroke-width:2.5px,color:#EFF6FF;
    classDef gold fill:#581C87,stroke:#A855F7,stroke-width:2.5px,color:#FAF5FF;
    classDef split fill:#831843,stroke:#EC4899,stroke-width:2.5px,color:#FDF2F8;
    classDef model fill:#064E3B,stroke:#10B981,stroke-width:2.5px,color:#ECFDF5;

    RAW["📁 <b>Raw FMCG Sales CSV</b><br/><b>5,000,000 Records • 962 MB</b><br/>36 Months Multi-Store Point-of-Sale Data"]:::raw
    BRONZE["🥉 <b>Bronze Parquet Ingestion</b><br/><code>data/bronze/</code> • <b>14 Partitions</b><br/>Schema Validation & Snappy Compression"]:::bronze
    SILVER["🥈 <b>Silver Cleansed Lakehouse</b><br/><code>data/silver/</code> • <b>4,989,807 Retained</b><br/>Deduplicated & Out-of-Bounds Stripped"]:::silver
    GOLD["🥇 <b>Gold Feature Vectors</b><br/><code>data/gold/</code> • <b>27 ML Features</b><br/>Target Leakage Excluded"]:::gold
    
    subgraph SPLITS ["📊 Multi-Scale 80/20 Train / Test Partitions"]
        TRAIN["<b>Train Partition (80%)</b><br/>798K (1M) • 2.39M (3M) • 3.99M (5M)"]:::split
        TEST["<b>Test Partition (20%)</b><br/>199K (1M) • 598K (3M) • 996K (5M)"]:::split
    end
    
    subgraph ENGINES ["🏆 Benchmark Machine Learning Engines (Single-Node vs Distributed)"]
        M_RF["🌲 <b>Random Forest</b><br/>Spark MLlib vs Scikit-Learn"]:::model
        M_XGB["🚀 <b>XGBoost</b><br/>Distributed vs Single-Node"]:::model
        M_LGB["⚡ <b>LightGBM</b><br/>Partition vs Single-Node"]:::model
        M_CAT["🐱 <b>CatBoost</b><br/>Ordered vs Single-Node"]:::model
    end

    RAW -->|"src/pipeline.py (ingest)"| BRONZE
    BRONZE -->|"src/pipeline.py (clean & validate)"| SILVER
    SILVER -->|"src/pipeline.py (feature engineering)"| GOLD
    GOLD -->|"split train"| TRAIN
    GOLD -->|"split test"| TEST
    
    TRAIN --> M_RF
    TRAIN --> M_XGB
    TRAIN --> M_LGB
    TRAIN --> M_CAT
"""
render_mermaid_diagram(overview_mermaid, height=960)

st.markdown("---")