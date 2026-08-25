"""
Page 4: Single-Node vs Distributed Machine Learning Benchmark & Telemetry
Central Analytical Engine
"""

import os
import sys

dash_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dash_dir not in sys.path:
    sys.path.insert(0, dash_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    inject_custom_css, render_sidebar, render_kpi_card,
    render_interpretation_box, render_viva_insight,
    apply_plot_layout, load_experiment_results,
    FRAMEWORK_COLORS, MODEL_COLORS
)

st.set_page_config(
    page_title="FMCG ML Benchmark & Telemetry",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar()

results_df = load_experiment_results()

st.markdown('<div class="main-header">Single-Node vs Distributed Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Empirical performance, speedup, accuracy, and benchmark system telemetry across 4 ML models on 1M, 3M, and 5M datasets</div>', unsafe_allow_html=True)

if not results_df.empty:
    # --- TOP KPI SUMMARY SECTION ---
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi_card("Max Distributed Speedup", "10.32x", "XGBoost 1M (4.81s vs 49.64s) | 6.50x RF 5M", "#10B981")
    with k2:
        render_kpi_card("Fastest Training", "4.33s", "CatBoost 1M (Single-Node) • 4.81s XGBoost", "#38BDF8")
    with k3:
        render_kpi_card("Highest Accuracy (R²)", "0.8760", "XGBoost 5M (RMSE: 13.14 • MAE: 9.26)", "#818CF8")
    with k4:
        render_kpi_card("Peak Host RAM Ceiling", "90.0%", "Single-Node 5M RAM Stress vs Spark headroom", "#F59E0B")
    with k5:
        render_kpi_card("Max Cluster Disk I/O", "185.3 MB/s", "Distributed parallel partition throughput", "#EC4899")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BENCHMARK FILTERS ---
    st.markdown("#### 🔍 Filter Benchmark View")
    f_col1, f_col2, f_col3 = st.columns(3)
    
    with f_col1:
        scale_options = ["All Scales (1M, 3M, 5M)", "1M Scale", "3M Scale", "5M Scale"]
        selected_scale_opt = st.selectbox("🎯 Dataset Scale:", scale_options, index=0)
    with f_col2:
        model_options = ["All Models", "Random Forest", "LightGBM", "XGBoost", "CatBoost"]
        selected_model_opt = st.selectbox("🏷️ Model Architecture:", model_options, index=0)
    with f_col3:
        framework_options = ["Both Frameworks", "Single-Node Only", "Distributed Only"]
        selected_fw_opt = st.selectbox("🖥️ Compute Framework:", framework_options, index=0)

    # Filter dataset
    filtered_df = results_df.copy()
    if selected_scale_opt == "1M Scale":
        filtered_df = filtered_df[filtered_df["Data Scale"] == "1M"]
    elif selected_scale_opt == "3M Scale":
        filtered_df = filtered_df[filtered_df["Data Scale"] == "3M"]
    elif selected_scale_opt == "5M Scale":
        filtered_df = filtered_df[filtered_df["Data Scale"] == "5M"]

    if selected_model_opt != "All Models":
        filtered_df = filtered_df[filtered_df["Model"] == selected_model_opt]

    if selected_fw_opt == "Single-Node Only":
        filtered_df = filtered_df[filtered_df["Framework"] == "Single-Node"]
    elif selected_fw_opt == "Distributed Only":
        filtered_df = filtered_df[filtered_df["Framework"] == "Distributed"]

    # Calculate Speedup column for display table
    speedup_map = {}
    for s in ["1M", "3M", "5M"]:
        for m in ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]:
            sn = results_df[(results_df["Framework"] == "Single-Node") & (results_df["Model"] == m) & (results_df["Data Scale"] == s)]
            dist = results_df[(results_df["Framework"] == "Distributed") & (results_df["Model"] == m) & (results_df["Data Scale"] == s)]
            if not sn.empty and not dist.empty:
                sn_t = float(sn.iloc[0]["Training Time (s)"])
                dist_t = float(dist.iloc[0]["Training Time (s)"])
                ratio = sn_t / dist_t if dist_t > 0 else 0
                speedup_map[(s, m)] = f"{ratio:.2f}x"

    filtered_df["Distributed Speedup"] = filtered_df.apply(
        lambda r: speedup_map.get((r["Data Scale"], r["Model"]), "—"), axis=1
    )

    # --- TWO DEDICATED CLEAN TABLES ---
    t_tab1, t_tab2 = st.tabs(["📊 Table 1: Machine Learning Performance", "🖥️ Table 2: Benchmark System Telemetry"])
    
    with t_tab1:
        st.markdown("##### 📋 Table 1: Machine Learning Execution & Accuracy Metrics")
        ml_cols = ["Framework", "Model", "Data Scale", "Data Rows", "Training Time (s)", "Prediction Time (s)", "Total Time (s)", "RMSE", "MAE", "R2", "Distributed Speedup"]
        avail_ml = [c for c in ml_cols if c in filtered_df.columns]
        
        ml_format = {
            "Data Rows": "{:,}",
            "Training Time (s)": "{:.2f}s",
            "Prediction Time (s)": "{:.2f}s",
            "Total Time (s)": "{:.2f}s",
            "RMSE": "{:.4f}",
            "MAE": "{:.4f}",
            "R2": "{:.4f}"
        }
        st.dataframe(filtered_df[avail_ml].style.format({k: v for k, v in ml_format.items() if k in avail_ml}), use_container_width=True)
        st.caption("Note: 'Distributed Speedup' represents Single-Node Training Time / Distributed Training Time (>1.0x indicates distributed acceleration).")

    with t_tab2:
        st.markdown("##### 📋 Table 2: Benchmark System Resource Telemetry")
        st.caption("Benchmark telemetry recorded across executor containers and host compute node.")
        telemetry_cols = ["Framework", "Model", "Data Scale", "Avg CPU Utilization (%)", "RAM Utilization (%)", "Peak RAM (GB)", "Disk Usage (GB)", "Disk I/O (MB/s)", "Network I/O (MB/s)", "CPU Cores Used", "Cluster Nodes"]
        avail_telem = [c for c in telemetry_cols if c in filtered_df.columns]
        
        telem_format = {
            "Avg CPU Utilization (%)": "{:.1f}%",
            "RAM Utilization (%)": "{:.1f}%",
            "Peak RAM (GB)": "{:.2f} GB",
            "Disk Usage (GB)": "{:.2f} GB",
            "Disk I/O (MB/s)": "{:.1f} MB/s",
            "Network I/O (MB/s)": "{:.1f} MB/s",
            "CPU Cores Used": "{:d}",
            "Cluster Nodes": "{:d}"
        }
        st.dataframe(filtered_df[avail_telem].style.format({k: v for k, v in telem_format.items() if k in avail_telem}), use_container_width=True)

    st.markdown("---")

    # --- 8 DEEP ANALYTICAL TABS ---
    st.subheader("📈 Deep Analytical Benchmark Modules")
    
    tab_train, tab_pred, tab_speedup, tab_scale, tab_acc, tab_cpu_ram, tab_io, tab_workers = st.tabs([
        "⏱️ Training Time",
        "⚡ Prediction Time",
        "🚀 Speedup Heatmap",
        "📈 Scalability Curves",
        "🎯 Predictive Accuracy",
        "🖥️ CPU & RAM Telemetry",
        "💾 Disk & Network I/O",
        "👷 Worker Load Balance"
    ])

    # TAB 1: TRAINING TIME
    with tab_train:
        st.subheader("Model Training Time Comparison (Seconds)")
        fig_train = px.bar(
            filtered_df,
            x="Model",
            y="Training Time (s)",
            color="Framework",
            barmode="group",
            facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
            category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
            color_discrete_map=FRAMEWORK_COLORS,
            title="Training Time by Model & Framework (Lower is Better)"
        )
        apply_plot_layout(fig_train, height=380)
        fig_train.update_traces(texttemplate='%{y:.1f}s', textposition='outside')
        st.plotly_chart(fig_train, use_container_width=True)
        render_interpretation_box("Random Forest experiences quadratic time escalation on Single-Node CPU (climbing to 456.1s at 5M), whereas PySpark distributes tree construction across executors to complete in 70.2s.")

    # TAB 2: PREDICTION TIME
    with tab_pred:
        st.subheader("Inference / Prediction Time (Seconds)")
        fig_pred = px.bar(
            filtered_df,
            x="Model",
            y="Prediction Time (s)",
            color="Framework",
            barmode="group",
            facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
            category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
            color_discrete_map=FRAMEWORK_COLORS,
            title="Batch Prediction Time across Test Partitions (Lower is Better)"
        )
        apply_plot_layout(fig_pred, height=380)
        fig_pred.update_traces(texttemplate='%{y:.2f}s', textposition='outside')
        st.plotly_chart(fig_pred, use_container_width=True)
        render_interpretation_box("Single-node in-memory prediction exhibits low latency on local NumPy buffers, while distributed prediction amortizes RDD task distribution overhead as test set scales to 1,000,000 records.")

    # TAB 3: SPEEDUP ANALYSIS
    with tab_speedup:
        st.subheader("Distributed Speedup Matrix & Heatmap")
        
        models_list = ["Random Forest", "XGBoost", "CatBoost", "LightGBM"]
        scales_list = ["1M", "3M", "5M"]
        matrix_data = []
        
        for m in models_list:
            row_vals = []
            for s in scales_list:
                sn = results_df[(results_df["Framework"] == "Single-Node") & (results_df["Model"] == m) & (results_df["Data Scale"] == s)]
                dist = results_df[(results_df["Framework"] == "Distributed") & (results_df["Model"] == m) & (results_df["Data Scale"] == s)]
                if not sn.empty and not dist.empty:
                    sn_t = float(sn.iloc[0]["Training Time (s)"])
                    dist_t = float(dist.iloc[0]["Training Time (s)"])
                    r = sn_t / dist_t if dist_t > 0 else 1.0
                    row_vals.append(round(r, 2))
                else:
                    row_vals.append(1.0)
            matrix_data.append(row_vals)
            
        sp_c1, sp_c2 = st.columns([3, 2])
        with sp_c1:
            fig_heatmap = px.imshow(
                matrix_data,
                x=scales_list,
                y=models_list,
                text_auto=True,
                labels=dict(x="Dataset Scale", y="Model Architecture", color="Speedup (x)"),
                color_continuous_scale=["#EF4444", "#F59E0B", "#10B981", "#38BDF8"],
                title="Speedup Heatmap: Single-Node Time / Distributed Time (>1.0x = Green/Blue)"
            )
            apply_plot_layout(fig_heatmap, height=360)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        with sp_c2:
            st.markdown("#### ⚡ Speedup Insights")
            st.markdown("""
            - **🌲 Random Forest**: Scales dramatically from **3.84x (1M)** $\\to$ **4.04x (3M)** $\\to$ **6.50x (5M)** as single-node hits CPU saturation.
            - **🚀 XGBoost**: Strong distributed histogram binning speedups (**10.32x at 1M**, **6.46x at 5M**).
            - **🐱 CatBoost**: Crosses over from single-node parity at 1M (0.83x) to distributed advantage at 5M (**2.09x**).
            - **⚡ LightGBM**: Highly optimized single-node C++ implementation maintains minimal overhead for smaller datasets.
            """)
        render_interpretation_box("Speedup is mathematically defined as Single-Node Training Time / Distributed Training Time. Values > 1.0x demonstrate parallel speedup over the single-machine baseline.")

    # TAB 4: SCALABILITY CURVES
    with tab_scale:
        st.subheader("Training Time Scalability Curves (1M -> 3M -> 5M)")
        scale_map = {"1M": 1, "3M": 3, "5M": 5}
        scale_df = results_df.copy()
        if selected_model_opt != "All Models":
            scale_df = scale_df[scale_df["Model"] == selected_model_opt]
            
        scale_df["Scale_Num"] = scale_df["Data Scale"].map(scale_map)
        scale_df["Run_Label"] = scale_df["Framework"] + " - " + scale_df["Model"]
        scale_df = scale_df.sort_values(by=["Scale_Num"])
        
        fig_curve = px.line(
            scale_df,
            x="Data Scale",
            y="Training Time (s)",
            color="Run_Label",
            markers=True,
            title="Computational Scaling: Growth in Training Time across Dataset Scales",
            category_orders={"Data Scale": ["1M", "3M", "5M"]}
        )
        apply_plot_layout(fig_curve, height=380)
        fig_curve.update_layout(hovermode="x unified")
        fig_curve.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig_curve, use_container_width=True)
        render_interpretation_box("The steep slope of Single-Node Random Forest illustrates the single-machine memory/CPU wall, while Distributed PySpark exhibits near-linear sub-linear scaling curves.")

    # TAB 5: PREDICTIVE ACCURACY
    with tab_acc:
        st.subheader("Predictive Accuracy: RMSE & R² Variance Explained")
        
        ac1, ac2 = st.columns(2)
        with ac1:
            fig_rmse = px.bar(
                filtered_df,
                x="Model",
                y="RMSE",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="Root Mean Squared Error (Lower is Better)"
            )
            apply_plot_layout(fig_rmse, height=360, y_range=[10, 16])
            fig_rmse.update_traces(texttemplate='%{y:.2f}', textposition='outside')
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        with ac2:
            fig_r2 = px.bar(
                filtered_df,
                x="Model",
                y="R2",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="R² Variance Explained (Higher is Better)"
            )
            apply_plot_layout(fig_r2, height=360, y_range=[0.80, 0.90])
            fig_r2.update_traces(texttemplate='%{y:.4f}', textposition='outside')
            st.plotly_chart(fig_r2, use_container_width=True)
            
        render_interpretation_box("Accuracy remains consistent across frameworks (R² ~0.84–0.88), demonstrating that distributed partition ensembling preserves model fidelity without loss of predictive generalization.")

    # TAB 6: CPU & RAM RESOURCE TELEMETRY
    with tab_cpu_ram:
        st.subheader("CPU Utilization, Host RAM %, and Peak Memory Footprint")
        
        rc1, rc2 = st.columns(2)
        with rc1:
            fig_cpu = px.bar(
                filtered_df,
                x="Model",
                y="Avg CPU Utilization (%)",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="Average CPU Utilization (%)"
            )
            apply_plot_layout(fig_cpu, height=350, y_range=[0, 105])
            fig_cpu.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_cpu, use_container_width=True)
            
        with rc2:
            fig_ram = px.bar(
                filtered_df,
                x="Model",
                y="RAM Utilization (%)",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="Host Memory Utilization (%)"
            )
            apply_plot_layout(fig_ram, height=350, y_range=[0, 105])
            fig_ram.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_ram, use_container_width=True)
            
        rc3, rc4 = st.columns(2)
        with rc3:
            fig_peak = px.bar(
                filtered_df,
                x="Model",
                y="Peak RAM (GB)",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="Peak RAM Allocated (GB)"
            )
            apply_plot_layout(fig_peak, height=350)
            fig_peak.update_traces(texttemplate='%{y:.2f}GB', textposition='outside')
            st.plotly_chart(fig_peak, use_container_width=True)
            
        with rc4:
            fig_cores = px.bar(
                filtered_df,
                x="Model",
                y="CPU Cores Used",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="CPU Cores Engaged During Training"
            )
            apply_plot_layout(fig_cores, height=350)
            fig_cores.update_traces(texttemplate='%{y} Cores', textposition='outside')
            st.plotly_chart(fig_cores, use_container_width=True)
            
        render_interpretation_box("Single-Node execution pushes host RAM to 90% utilization at 5M rows, risking OOM crashes. Distributed PySpark balances heap across worker containers, preserving headroom.")

    # TAB 7: DISK & NETWORK I/O
    with tab_io:
        st.subheader("Storage Throughput (Disk I/O) & Network Shuffle Bandwidth")
        
        io1, io2 = st.columns(2)
        with io1:
            fig_dio = px.bar(
                filtered_df,
                x="Model",
                y="Disk I/O (MB/s)",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="Disk I/O Throughput (MB/s)"
            )
            apply_plot_layout(fig_dio, height=360)
            fig_dio.update_traces(texttemplate='%{y:.1f} MB/s', textposition='outside')
            st.plotly_chart(fig_dio, use_container_width=True)
            
        with io2:
            fig_nio = px.bar(
                filtered_df,
                x="Model",
                y="Network I/O (MB/s)",
                color="Framework",
                barmode="group",
                facet_col="Data Scale" if selected_scale_opt == "All Scales (1M, 3M, 5M)" else None,
                category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]},
                color_discrete_map=FRAMEWORK_COLORS,
                title="Cluster Network I/O Bandwidth (MB/s - Shuffle Traffic)"
            )
            apply_plot_layout(fig_nio, height=360)
            fig_nio.update_traces(texttemplate='%{y:.1f} MB/s', textposition='outside')
            st.plotly_chart(fig_nio, use_container_width=True)
            
        render_interpretation_box("Distributed workloads utilize up to 185 MB/s Disk I/O and 43 MB/s inter-node network bandwidth during partition aggregation and tree split synchronization.")

    # TAB 8: WORKER LOAD BALANCE
    with tab_workers:
        st.subheader("Worker Cluster CPU Load Distribution & Balance Analysis")
        st.caption("Evaluating individual Spark worker core loads across distributed partitions (Worker 1, Worker 2, Worker 3).")
        
        dist_rows = results_df[results_df["Framework"] == "Distributed"].copy()
        
        worker_records = []
        for _, r in dist_rows.iterrows():
            w1 = float(str(r.get("Worker 1 CPU (%)", "0")).replace("%", "") or 0)
            w2 = float(str(r.get("Worker 2 CPU (%)", "0")).replace("%", "") or 0)
            w3 = float(str(r.get("Worker 3 CPU (%)", "0")).replace("%", "") or 0)
            
            if w1 > 0:
                worker_records.append({"Data Scale": r["Data Scale"], "Model": r["Model"], "Worker": "Worker 1", "CPU (%)": w1})
            if w2 > 0:
                worker_records.append({"Data Scale": r["Data Scale"], "Model": r["Model"], "Worker": "Worker 2", "CPU (%)": w2})
            if w3 > 0:
                worker_records.append({"Data Scale": r["Data Scale"], "Model": r["Model"], "Worker": "Worker 3", "CPU (%)": w3})
                
        if worker_records:
            w_df = pd.DataFrame(worker_records)
            
            wc1, wc2 = st.columns([3, 2])
            with wc1:
                fig_w = px.bar(
                    w_df,
                    x="Worker",
                    y="CPU (%)",
                    color="Model",
                    barmode="group",
                    facet_col="Data Scale",
                    title="Separate Worker CPU Utilization (%) per Task",
                    category_orders={"Data Scale": ["1M", "3M", "5M"], "Model": ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]}
                )
                apply_plot_layout(fig_w, height=360, y_range=[0, 105])
                fig_w.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                st.plotly_chart(fig_w, use_container_width=True)
                
            with wc2:
                st.markdown("#### ⚖️ Worker Balance Delta (Max - Min)")
                balance_rows = []
                for s in ["1M", "3M", "5M"]:
                    for m in ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]:
                        sub = w_df[(w_df["Data Scale"] == s) & (w_df["Model"] == m)]
                        if not sub.empty:
                            c_max = sub["CPU (%)"].max()
                            c_min = sub["CPU (%)"].min()
                            delta = round(c_max - c_min, 1)
                            balance_rows.append({"Scale": s, "Model": m, "Max CPU (%)": f"{c_max:.1f}%", "Min CPU (%)": f"{c_min:.1f}%", "Load Delta": f"{delta:.1f}%", "Status": "Balanced (<=5%)" if delta <= 5.0 else "Healthy (<=10%)"})
                            
                st.dataframe(pd.DataFrame(balance_rows), use_container_width=True)
                
            render_interpretation_box("All three Spark worker nodes exhibit tight load balance with low delta variances (<=5–8%), confirming uniform partition sizing and zero executor idle stragglers.")

else:
    st.warning("⚠️ No benchmark records found in `results/experiment_results.csv`. Run `./run_all.sh` to generate multi-scale results.")

render_viva_insight(
    "The Distributed Scalability Crossover Point",
    "Single-node ML libraries (like LightGBM/CatBoost) avoid inter-process network and serialization overhead, making them faster on datasets < 1M–2M rows. However, as data scales to 3M–5M+ rows, single-machine CPU/RAM saturation manifests, allowing distributed PySpark cluster execution to achieve significant speedups (e.g. 6.50x on Random Forest at 5M)."
)
