"""
Page 6: Final Findings, Conclusions & Architecture Decision Matrix
Academic Presentation Standard
"""

import os
import sys

dash_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dash_dir not in sys.path:
    sys.path.insert(0, dash_dir)

import streamlit as st
import pandas as pd
from utils import (
    inject_custom_css, render_sidebar, render_kpi_card,
    render_interpretation_box, render_viva_insight
)

st.set_page_config(
    page_title="FMCG ML Benchmark: Final Findings",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar()

st.markdown('<div class="main-header">Final Findings & Strategic Architecture Conclusions</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Synthesis and decision framework comparing Single-Node vs Distributed PySpark execution across 4 ML models and 3 dataset scales</div>', unsafe_allow_html=True)

# --- PRESENTATION-READY SCORECARD ---
st.markdown("""
<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155; border-radius: 12px; padding: 22px; margin-bottom: 24px; color: #F8FAFC;">
    <h3 style="color: #38BDF8; margin-top: 0; font-size: 1.25rem;">🏆 Benchmark Scorecard: Single-Node vs Distributed (PySpark)</h3>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: 600; color: #94A3B8; width: 28%;">Maximum Distributed Speedup:</td>
            <td style="padding: 10px; color: #10B981; font-weight: bold;">XGBoost (1M) & Random Forest (5M)</td>
            <td style="padding: 10px;"><span class="speedup-badge-win">10.32x Speedup (1M) / 6.50x Speedup (5M)</span> (70.2s vs 456.1s)</td>
        </tr>
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: 600; color: #94A3B8;">Fastest Model Execution:</td>
            <td style="padding: 10px; color: #38BDF8; font-weight: bold;">CatBoost & XGBoost</td>
            <td style="padding: 10px;">4.33s (1M Single-Node) • 4.81s (1M Distributed) • 15.64s (5M Distributed)</td>
        </tr>
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: 600; color: #94A3B8;">Lowest Predictive Error (Best RMSE):</td>
            <td style="padding: 10px; color: #818CF8; font-weight: bold;">XGBoost (5M Rows)</td>
            <td style="padding: 10px;"><b>RMSE = 13.14</b> | <b>MAE = 9.26</b> | <b>R² = 0.8760</b></td>
        </tr>
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: 600; color: #94A3B8;">Peak Memory Headroom:</td>
            <td style="padding: 10px; color: #38BDF8; font-weight: bold;">Distributed PySpark</td>
            <td style="padding: 10px;">Balanced at 73% cluster RAM vs Single-Node hitting <b>90.0% memory ceiling</b></td>
        </tr>
        <tr>
            <td style="padding: 10px; font-weight: 600; color: #94A3B8;">Scalability Crossover Point:</td>
            <td style="padding: 10px; color: #F59E0B; font-weight: bold;">~1,000,000 – 3,000,000 Rows</td>
            <td style="padding: 10px;">Distributed PySpark eliminates single-machine CPU/memory bottlenecks beyond 1M–3M rows</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# --- 6 STRUCTURED SYNTHESIS SECTIONS ---
st.subheader("💡 Key Benchmark Findings & Analytical Synthesis")

c_f1, c_f2 = st.columns(2)

with c_f1:
    st.markdown("""
    ##### 1. Computational Performance Findings
    - **Random Forest**: Demonstrates the clearest distributed advantage. Single-node Scikit-Learn scales quadratically (66.7s at 1M $\\to$ 456.1s at 5M), whereas PySpark MLlib scales near-linearly to 70.2s (**6.50x faster**).
    - **Gradient Boosting (LightGBM/CatBoost)**: Highly optimized single-node C++ libraries excel for small-to-medium volumes (<1M–2M rows) due to zero IPC overhead, while distributed ensembling reaches parity as volume grows.
    
    ##### 2. Predictive Accuracy Findings
    - **Algorithmic Invariance**: Predictive accuracy is statistically invariant across single-node and distributed frameworks ($R^2 \\approx 0.84 - 0.88$).
    - **Implication**: The primary benefit of distributed PySpark ML is **computational scalability and memory headroom**, not higher raw predictive accuracy.
    """)

with c_f2:
    st.markdown("""
    ##### 3. Resource Telemetry Findings
    - **Memory Wall**: Single-node execution on 5M rows pushes host RAM to 90.0% utilization, creating acute out-of-memory (OOM) risks during tree building.
    - **Cluster Headroom**: PySpark distributes partitions across 3 worker nodes, maintaining memory utilization around ~73% with steady garbage collection.
    
    ##### 4. Scalability & Load Balancing
    - **Executor Symmetry**: Worker CPU load distributions remain tightly balanced (70%–91% across Worker 1, 2, and 3), proving uniform partition allocation without stragglers.
    """)

st.markdown("---")

# --- ARCHITECTURAL DECISION MATRIX ---
st.subheader("🏛️ Strategic Architecture Decision Matrix: When to Use What?")

dm_col1, dm_col2 = st.columns(2)

with dm_col1:
    st.markdown("""
    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid #F59E0B; border-radius: 8px; padding: 18px; height: 100%;">
        <h4 style="color: #F59E0B; margin-top: 0;">🔹 Prefer Single-Node ML When:</h4>
        <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
            <li><b>Dataset Size</b>: Volume is under $< 1,000,000 – 2,000,000$ rows and fits comfortably within host RAM.</li>
            <li><b>Algorithm Choice</b>: Utilizing C++ optimized libraries (<code>LightGBM</code>, <code>CatBoost</code>) where multi-threading provides sub-10s training.</li>
            <li><b>Infrastructure Simplicity</b>: Minimal DevOps overhead is desired; zero cluster networking, JVM, or serialization complexity.</li>
            <li><b>Rapid Experimentation</b>: Data scientists requiring instant interactive hyperparameter tuning on local development workstations.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with dm_col2:
    st.markdown("""
    <div style="background: rgba(2, 132, 199, 0.08); border: 1px solid #0284C7; border-radius: 8px; padding: 18px; height: 100%;">
        <h4 style="color: #38BDF8; margin-top: 0;">🔹 Prefer Distributed PySpark ML When:</h4>
        <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.6;">
            <li><b>Dataset Scale</b>: Data volume exceeds $> 3,000,000 – 5,000,000+$ records or cannot fit in single-machine memory.</li>
            <li><b>Heavy Ensemble Models</b>: Training compute-intensive algorithms like <code>Random Forest</code> where parallel tree construction yields <b>>6x speedup</b>.</li>
            <li><b>Lakehouse Integration</b>: The data already resides in partitioned Parquet/Delta Lake tables, eliminating downsampling or export steps.</li>
            <li><b>Batch / Streaming Pipelines</b>: Unified end-to-end processing where Medallion ETL (Bronze/Silver/Gold) feeds directly into ML training.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- VIVA DEFENSE CHEAT SHEET ---
st.subheader("🎓 Master Viva Defense: Top 5 Anticipated Questions & Answers")

v1, v2 = st.columns(2)
with v1:
    st.markdown(r"""
    **Q1: Why is PySpark Random Forest faster than Scikit-Learn at 5M, but LightGBM is faster on Single-Node at 1M?**  
    *Answer:* Random Forest is embarrassingly parallel across trees; Spark distributes tree subsets across workers. Conversely, LightGBM uses highly optimized OpenMP C++ histogram binning on contiguous memory with zero network/JVM socket overhead, making it faster at smaller scales.
    
    **Q2: Why did you eliminate `gross_sales` and `net_sales` from feature engineering?**  
    *Answer:* Both variables are mathematically derived from `units_sold` ($gross\_sales = units \times price$). Keeping them would cause 100% target leakage, invalidating the model.
    """)
with v2:
    st.markdown(r"""
    **Q3: Does distributed ML increase predictive accuracy?**  
    *Answer:* No. The benchmark proves predictive accuracy ($R^2 \approx 0.87$) is invariant across frameworks. The value of PySpark ML is computational scalability and overcoming memory barriers, not improving RMSE.
    
    **Q4: What is the Medallion Architecture's role in this project?**  
    *Answer:* It establishes clean, reproducible data contracts (Bronze raw ingestion $\to$ Silver data cleaning/deduplication $\to$ Gold leak-free ML features), ensuring identical inputs for all benchmark runs.
    """)
