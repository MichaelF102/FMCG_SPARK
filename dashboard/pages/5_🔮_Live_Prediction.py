"""
Page 5: Sales Prediction & What-If Simulator
Interactive Retail Demand Simulation using Benchmark-Trained Models
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
from utils import (
    inject_custom_css, render_sidebar, render_kpi_card,
    render_interpretation_box, render_viva_insight,
    apply_plot_layout, get_prediction_models,
    CATEGORICAL_COLS, NUMERIC_COLS
)

st.set_page_config(
    page_title="FMCG Sales Prediction & Simulator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
render_sidebar()

st.markdown('<div class="main-header">Sales Prediction </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive sales prediction and sensitivity simulation powered by pre-trained benchmark ML models</div>', unsafe_allow_html=True)

bundle = get_prediction_models()

if bundle is None:
    st.error("⚠️ Prediction models bundle could not be loaded. Ensure benchmark training data is available in `data/`.")
else:
    models = bundle["models"]
    encoder = bundle["encoder"]
    cat_cols = bundle["cat_cols"]
    feat_cols = bundle["feat_cols"]
    
    tab_single, tab_batch = st.tabs(["🎯 Single Item Scenario Simulator", "📁 Batch CSV Scoring"])
    
    with tab_single:
        st.markdown("#### 🎛️ Configure Scenario & Market Parameters")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("##### 🏪 Product & Store Context")
            category = st.selectbox("Product Category", ["Beverages", "Snacks", "Dairy & Eggs", "Personal Care", "Household Supplies", "Bakery"], index=0)
            subcategory = st.selectbox("Subcategory", ["Carbonated Drinks", "Juices & RTD", "Chips & Crisps", "Chocolates", "Milk & Yogurt", "Hair Care", "Laundry Detergent", "Biscuits"], index=0)
            brand = st.selectbox("Brand Name", ["Coca-Cola", "Pepsi", "Lays", "Doritos", "Nestle", "Danone", "Unilever", "P&G"], index=0)
            channel = st.selectbox("Retail Channel", ["Hypermarket", "Supermarket", "Convenience Store", "E-Commerce"], index=1)
            country = st.selectbox("Country Market", ["United States", "United Kingdom", "Germany", "India", "Brazil"], index=0)
            city = st.selectbox("City Tier", ["New York", "London", "Berlin", "Mumbai", "Sao Paulo", "Regional Hub"], index=0)
            
        with col_p2:
            st.markdown("##### 🏷️ Pricing & Inventory")
            list_price = st.number_input("List Price ($)", min_value=0.5, max_value=500.0, value=12.50, step=0.5)
            discount_pct = st.slider("Promotional Discount (%)", min_value=0, max_value=60, value=15, step=1)
            promo_flag = 1 if discount_pct > 0 else 0
            purchase_cost = st.number_input("Unit Purchase Cost ($)", min_value=0.1, max_value=300.0, value=round(list_price * 0.55, 2), step=0.5)
            stock_on_hand = st.number_input("Store Stock on Hand (Units)", min_value=0, max_value=10000, value=300, step=10)
            lead_time_days = st.slider("Supplier Lead Time (Days)", min_value=1, max_value=30, value=3)
            
        with col_p3:
            st.markdown("##### 🌦️ Environmental & Calendar Signals")
            season = st.selectbox("Seasonal Period", ["Summer", "Winter", "Spring", "Autumn"], index=0)
            temperature = st.slider("Ambient Temperature (°C)", min_value=-15, max_value=45, value=26)
            rain_mm = st.slider("Precipitation (mm)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
            weekend_holiday = st.selectbox("Day Type", ["Weekday (Regular)", "Weekend / Holiday"], index=0)
            month = st.slider("Month of Year", min_value=1, max_value=12, value=7)
            weekday = st.slider("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=3)

        # Derived calculations
        discount_amount = round(list_price * (discount_pct / 100.0), 2)
        effective_price = round(list_price - discount_amount, 2)
        margin_pct = round(((effective_price - purchase_cost) / effective_price * 100.0) if effective_price > 0 else 0.0, 2)
        stock_out_flag = 1 if stock_on_hand <= 0 else 0
        quarter = (month - 1) // 3 + 1
        is_wknd = 1 if "Weekend" in weekend_holiday else 0
        
        # Build inference input dataframe
        input_dict = {
            "country": country,
            "city": city,
            "channel": channel,
            "category": category,
            "subcategory": subcategory,
            "brand": brand,
            "season": season,
            "temperature": temperature,
            "rain_mm": rain_mm,
            "latitude": 40.71,
            "longitude": -74.00,
            "list_price": list_price,
            "discount_pct": discount_pct,
            "promo_flag": promo_flag,
            "stock_on_hand": stock_on_hand,
            "stock_out_flag": stock_out_flag,
            "lead_time_days": lead_time_days,
            "purchase_cost": purchase_cost,
            "margin_pct": margin_pct,
            "quarter": quarter,
            "weekend_holiday": is_wknd,
            "discount_amount": discount_amount,
            "effective_price": effective_price,
            "year": 2024,
            "month": month,
            "day": 15,
            "weekday": weekday
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess features
        X_live = input_df[[c for c in feat_cols if c in input_df.columns]].copy()
        X_live[cat_cols] = encoder.transform(X_live[cat_cols].astype(str))
        
        # Run inference across models
        predictions = {}
        for name, m in models.items():
            pred = float(m.predict(X_live)[0])
            predictions[name] = max(0.0, round(pred, 1))
            
        pred_vals = list(predictions.values())
        avg_units = round(np.mean(pred_vals), 1)
        min_units = round(np.min(pred_vals), 1)
        max_units = round(np.max(pred_vals), 1)
        spread_units = round(max_units - min_units, 1)
        
        est_revenue = round(avg_units * effective_price, 2)
        est_profit = round(avg_units * (effective_price - purchase_cost), 2)
        days_stock = round(stock_on_hand / max(avg_units, 0.1), 1)
        
        st.markdown("---")
        st.subheader("⚡ Scenario Inference Results & Model Consensus")
        
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        with pc1:
            render_kpi_card("Consensus Demand", f"{avg_units:,.1f}", "Mean Units Sold / Day", "#38BDF8")
        with pc2:
            render_kpi_card("XGBoost Estimate", f"{predictions.get('XGBoost', 0):,.1f}", "Units / Day (R² 0.8760)", "#F43F5E")
        with pc3:
            render_kpi_card("LightGBM Estimate", f"{predictions.get('LightGBM', 0):,.1f}", "Units / Day (Fastest)", "#8B5CF6")
        with pc4:
            render_kpi_card("CatBoost Estimate", f"{predictions.get('CatBoost', 0):,.1f}", "Units / Day (Categorical)", "#F59E0B")
        with pc5:
            render_kpi_card("Random Forest", f"{predictions.get('Random Forest', 0):,.1f}", "Units / Day (Ensemble)", "#10B981")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Agreement & Financial Impact Panels
        sc_col1, sc_col2 = st.columns([2, 3])
        
        with sc_col1:
            st.markdown("#### 🤝 Model Agreement & Metrics")
            st.markdown(f"""
            - **Minimum Prediction**: **`{min_units:,.1f}` units**
            - **Maximum Prediction**: **`{max_units:,.1f}` units**
            - **Prediction Spread (Range)**: **`{spread_units:,.1f}` units** (Tight spread indicates high model consensus)
            
            ---
            #### 💰 Commercial Scenario Projection
            - **Effective Unit Price**: **`${effective_price:.2f}`** (Base: `${list_price:.2f}` with `{discount_pct}%` discount)
            - **Projected Daily Revenue**: **`${est_revenue:,.2f}`**
            - **Projected Gross Profit**: **`${est_profit:,.2f}`** (Margin: `{margin_pct:.1f}%`)
            - **Inventory Depletion Horizon**: **`{days_stock}` days** of remaining stock
            """)
            
        with sc_col2:
            st.markdown("#### 📈 Promotional Discount Elasticity Curve")
            disc_range = list(range(0, 65, 5))
            curve_records = []
            for d in disc_range:
                sim_df = input_df.copy()
                sim_df["discount_pct"] = d
                sim_df["promo_flag"] = 1 if d > 0 else 0
                sim_df["discount_amount"] = list_price * (d / 100.0)
                sim_df["effective_price"] = list_price - sim_df["discount_amount"]
                sim_X = sim_df[[c for c in feat_cols if c in sim_df.columns]].copy()
                sim_X[cat_cols] = encoder.transform(sim_X[cat_cols].astype(str))
                pred_v = float(models["XGBoost"].predict(sim_X)[0])
                rev_v = pred_v * float(sim_df["effective_price"].iloc[0])
                curve_records.append({"Discount (%)": d, "Predicted Units": max(0.0, pred_v), "Estimated Revenue ($)": max(0.0, rev_v)})
                
            fig_sens = px.line(
                pd.DataFrame(curve_records),
                x="Discount (%)",
                y="Predicted Units",
                markers=True,
                title="Predicted Daily Units Sold vs Promotional Discount Level (%)",
                color_discrete_sequence=["#10B981"]
            )
            apply_plot_layout(fig_sens, height=300)
            st.plotly_chart(fig_sens, use_container_width=True)
            render_interpretation_box("As promotional discount scales from 0% to 50%, predicted unit demand increases non-linearly, enabling category managers to pinpoint optimal revenue vs margin thresholds.")

    with tab_batch:
        st.subheader("📁 Batch File Prediction & Scoring")
        st.markdown("Upload a CSV with new transactions or promotion scenarios to generate batch predictions across all 4 models.")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df):,} rows from uploaded file.")
            
            if st.button("🚀 Score Batch Dataset"):
                try:
                    X_batch = batch_df[[c for c in feat_cols if c in batch_df.columns]].copy()
                    X_batch[cat_cols] = encoder.transform(X_batch[cat_cols].astype(str))
                    for m_name, m_inst in models.items():
                        batch_df[f"pred_{m_name.lower().replace(' ', '_')}"] = m_inst.predict(X_batch).round(1)
                    batch_df["consensus_prediction"] = batch_df[[f"pred_{m_name.lower().replace(' ', '_')}" for m_name in models]].mean(axis=1).round(1)
                    st.success("Batch scoring complete!")
                    st.dataframe(batch_df.head(100), use_container_width=True)
                    csv_down = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Scored CSV", csv_down, "fmcg_scored_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error during batch scoring: {e}")

render_viva_insight(
    "Scenario Simulation vs Real-Time Streaming",
    "This simulator applies the pre-fitted benchmark models to user-defined scenario vectors. In enterprise FMCG systems, these models serve batch demand forecasting engines and scenario optimizers, where feature pipelines are pre-materialized in Lakehouse Gold tables."
)
