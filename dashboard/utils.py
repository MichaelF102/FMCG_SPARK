"""
Unified Dashboard Utilities, Dark Theme Styling, Data Loaders & Mermaid Architecture Engine
Academic & Research Standard for FMCG PySpark vs Single-Node Benchmark
"""

import os
import json
import joblib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- COLOR PALETTE (CONSISTENT VISUAL IDENTITY) ---
FRAMEWORK_COLORS = {
    "Single-Node": "#F59E0B",   # Warm Amber / Orange
    "Distributed": "#0284C7"    # Spark Sky Blue
}

MODEL_COLORS = {
    "Random Forest": "#10B981", # Emerald Green
    "LightGBM": "#8B5CF6",      # Purple
    "XGBoost": "#F43F5E",       # Rose Red
    "CatBoost": "#F59E0B"       # Amber
}

def apply_plot_layout(fig, height=360, y_range=None, title=None, show_legend=True):
    """
    Applies consistent dark analytical styling to Plotly figures without keyword collisions.
    """
    layout_update = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, sans-serif", color="#F1F5F9", size=12),
        margin=dict(l=25, r=25, t=45 if title else 25, b=25),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.35)",
            zerolinecolor="rgba(51, 65, 85, 0.5)"
        ),
        yaxis=dict(
            gridcolor="rgba(51, 65, 85, 0.35)",
            zerolinecolor="rgba(51, 65, 85, 0.5)"
        )
    )
    if title:
        layout_update["title"] = title
    if y_range:
        layout_update["yaxis"]["range"] = y_range
        
    fig.update_layout(**layout_update)
    return fig

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #F8FAFC;
        }
        
        .stApp {
            background-color: #0A0F1D;
        }
        
        /* Headers */
        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.15rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #94A3B8;
            margin-bottom: 1.25rem;
            line-height: 1.4;
        }
        
        /* Compact Metric Hero Cards */
        .metric-hero {
            background: linear-gradient(145deg, #0F172A 0%, #1E293B 100%);
            border-radius: 10px;
            border: 1px solid #334155;
            border-left: 4px solid #38BDF8;
            padding: 14px 18px;
            color: #F8FAFC;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            height: 100%;
        }
        .metric-hero-title {
            font-size: 0.72rem;
            font-weight: 700;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 2px;
        }
        .metric-hero-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #38BDF8;
            line-height: 1.2;
            margin: 2px 0;
        }
        .metric-hero-subtitle {
            font-size: 0.75rem;
            color: #64748B;
            line-height: 1.2;
        }
        
        /* Interpretation Box */
        .interp-box {
            background: rgba(15, 23, 42, 0.75);
            border-left: 3px solid #0284C7;
            border-radius: 6px;
            padding: 10px 14px;
            margin-top: 8px;
            margin-bottom: 16px;
            font-size: 0.85rem;
            color: #CBD5E1;
            line-height: 1.45;
        }
        
        /* Viva Insight Card */
        .viva-box {
            background: linear-gradient(145deg, #1E1B4B 0%, #0F172A 100%);
            border: 1px solid #4338CA;
            border-left: 4px solid #818CF8;
            border-radius: 8px;
            padding: 14px 18px;
            margin-top: 20px;
            margin-bottom: 20px;
            color: #E2E8F0;
        }
        .viva-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #A5B4FC;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .viva-content {
            font-size: 0.88rem;
            color: #CBD5E1;
            line-height: 1.5;
        }
        
        /* Speedup Badges */
        .speedup-badge-win {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid #059669;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.85rem;
        }
        
        /* Custom Table Styling */
        div[data-testid="stDataFrame"] {
            border: 1px solid #334155;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Tab Navigation Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
            border-bottom: 1px solid #334155;
            padding-bottom: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            border-radius: 6px 6px 0 0;
            padding: 8px 16px;
            color: #94A3B8;
            font-weight: 600;
            font-size: 0.85rem;
            background-color: transparent;
        }
        .stTabs [aria-selected="true"] {
            color: #38BDF8 !important;
            border-bottom: 2px solid #38BDF8 !important;
            background-color: rgba(56, 189, 248, 0.08) !important;
        }

        /* Horizontal Pipeline Stepper */
        .pipeline-stepper {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin: 15px 0 25px 0;
            overflow-x: auto;
        }
        .step-card {
            flex: 1;
            min-width: 140px;
            background: #0F172A;
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            border: 1px solid #334155;
            position: relative;
        }
        .step-card.bronze { border-color: #D97706; background: linear-gradient(145deg, #1E1B18 0%, #0F172A 100%); }
        .step-card.silver { border-color: #3B82F6; background: linear-gradient(145deg, #182234 0%, #0F172A 100%); }
        .step-card.gold { border-color: #A855F7; background: linear-gradient(145deg, #241A34 0%, #0F172A 100%); }
        .step-card.split { border-color: #EC4899; background: linear-gradient(145deg, #2D1A29 0%, #0F172A 100%); }
        .step-card.benchmark { border-color: #10B981; background: linear-gradient(145deg, #162B24 0%, #0F172A 100%); }
        
        .step-num { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 2px; }
        .step-title { font-size: 0.95rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px; }
        .step-desc { font-size: 0.72rem; color: #94A3B8; }
        .step-arrow { color: #64748B; font-size: 1.2rem; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

def render_pipeline_stepper():
    """Renders the top horizontal glowing pipeline stage cards from the screenshot format."""
    st.markdown("""
    <div class="pipeline-stepper">
        <div class="step-card bronze">
            <div class="step-num" style="color: #F59E0B;">🥉 1. BRONZE</div>
            <div class="step-title">Raw Ingestion</div>
            <div class="step-desc">Raw CSV → Parquet (5M)</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-card silver">
            <div class="step-num" style="color: #60A5FA;">🥈 2. SILVER</div>
            <div class="step-title">Cleansing</div>
            <div class="step-desc">Cleanse & Dedup (4.98M)</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-card gold">
            <div class="step-num" style="color: #C084FC;">🥇 3. GOLD</div>
            <div class="step-title">Feature Store</div>
            <div class="step-desc">27 ML Feature Vectors</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-card split">
            <div class="step-num" style="color: #F472B6;">📊 4. SPLIT</div>
            <div class="step-title">Train / Test</div>
            <div class="step-desc">80% Train / 20% Test (1M/3M/5M)</div>
        </div>
        <div class="step-arrow">→</div>
        <div class="step-card benchmark">
            <div class="step-num" style="color: #34D399;">🏆 5. BENCHMARK</div>
            <div class="step-title">Model Engines</div>
            <div class="step-desc">Single-Node vs Distributed</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_mermaid_diagram(code, height=980):
    """
    Renders large, high-definition dark-themed Mermaid flowcharts with clear readable typography and zero clipping.
    """
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
            
            html, body {{
                background-color: transparent;
                margin: 0;
                padding: 10px 0;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                font-family: 'Inter', -apple-system, sans-serif;
                overflow: visible;
            }}
            .mermaid {{
                width: 100%;
                display: flex;
                justify-content: center;
            }}
            .mermaid svg {{
                width: 95% !important;
                max-width: 1100px !important;
                height: auto !important;
                filter: drop-shadow(0 8px 24px rgba(0, 0, 0, 0.4));
            }}
            .node rect, .node circle, .node polygon {{
                rx: 10px !important;
                ry: 10px !important;
                stroke-width: 2.2px !important;
            }}
            .node .label {{
                font-family: 'Inter', sans-serif !important;
                font-size: 15px !important;
                font-weight: 600 !important;
                line-height: 1.45 !important;
            }}
            .node b {{
                font-weight: 800 !important;
                letter-spacing: 0.02em;
            }}
            .node code {{
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 13px !important;
                background: rgba(0,0,0,0.35);
                padding: 2px 5px;
                border-radius: 4px;
            }}
            .edgeLabel {{
                font-family: 'Inter', sans-serif !important;
                font-size: 13px !important;
                font-weight: 600 !important;
                color: #CBD5E1 !important;
                background-color: #0F172A !important;
                padding: 4px 8px !important;
                border-radius: 6px !important;
                border: 1px solid #334155 !important;
            }}
            .cluster rect {{
                rx: 12px !important;
                ry: 12px !important;
                stroke-width: 1.5px !important;
                stroke: #475569 !important;
                fill: rgba(15, 23, 42, 0.6) !important;
            }}
            .cluster span.nodeLabel {{
                font-size: 14px !important;
                font-weight: 700 !important;
                color: #94A3B8 !important;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
        {code}
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    darkMode: true,
                    background: '#0B0F19',
                    primaryColor: '#1E293B',
                    primaryTextColor: '#F8FAFC',
                    primaryBorderColor: '#38BDF8',
                    lineColor: '#64748B',
                    secondaryColor: '#0F172A',
                    tertiaryColor: '#1E1B4B',
                    fontSize: '15px'
                }},
                flowchart: {{
                    useMaxWidth: false,
                    htmlLabels: true,
                    curve: 'basis',
                    padding: 20,
                    nodeSpacing: 40,
                    rankSpacing: 45
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)

def render_sidebar():
    """Renders academic standard compact sidebar navigation and system status."""
    
    
    # Compact Benchmark Specs
    st.sidebar.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
        Benchmark Metadata
    </div>
    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid #1E293B; border-radius: 8px; padding: 10px; font-size: 0.8rem; line-height: 1.6;">
        <div><span style="color:#94A3B8;">Dataset:</span> <b>5M Rows (3 Years)</b></div>
        <div><span style="color:#94A3B8;">Scales:</span> <b>1M | 3M | 5M Rows</b></div>
        <div><span style="color:#94A3B8;">Models (4):</span> <b>RF, LGBM, XGB, CatB</b></div>
        <div><span style="color:#94A3B8;">Frameworks:</span> <b>Single-Node vs Spark</b></div>
        <div><span style="color:#94A3B8;">Target:</span> <code>units_sold</code></div>
        <div><span style="color:#94A3B8;">Cluster:</span> <b>1 Master + 3 Workers</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="margin-top: 15px; font-size: 0.7rem; color: #475569; text-align: center;">
        Created By Michael Fernandes
    </div>
    """, unsafe_allow_html=True)

def render_kpi_card(title, value, subtitle="", border_color="#38BDF8"):
    st.markdown(f"""
    <div class="metric-hero" style="border-left-color: {border_color};">
        <div class="metric-hero-title">{title}</div>
        <div class="metric-hero-value">{value}</div>
        <div class="metric-hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_interpretation_box(text):
    st.markdown(f"""
    <div class="interp-box">
        💡 <b>Interpretation:</b> {text}
    </div>
    """, unsafe_allow_html=True)

def render_viva_insight(title, content):
    st.markdown(f"""
    <div class="viva-box">
        <div class="viva-title">🎓 Viva & Academic Defense Insight: {title}</div>
        <div class="viva-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# --- CACHED DATA LOADERS ---
@st.cache_data
def load_metrics_json():
    metrics_path = "results/pipeline_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_experiment_results():
    results_path = "results/experiment_results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        if "Data Scale" in df.columns:
            df = df.dropna(subset=["Data Scale"])
            df = df[df["Data Scale"].astype(str).str.strip() != ""]
        return df
    return pd.DataFrame()

@st.cache_data
def load_sample_dataset(n_rows=5000):
    paths = [
        "data/gold_5M/train.parquet",
        "data/gold/train.parquet",
        "data/silver",
        "fmcg_sales_5M_rows.csv",
        "fmcg_sales_3years_1M_rows.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            if p.endswith(".parquet"):
                try:
                    return pd.read_parquet(p).head(n_rows)
                except Exception:
                    continue
            elif p.endswith(".csv"):
                try:
                    return pd.read_csv(p, nrows=n_rows)
                except Exception:
                    continue
    return pd.DataFrame()

# Categorical and Numerical Feature Definitions for Inference
CATEGORICAL_COLS = ["country", "city", "channel", "category", "subcategory", "brand", "season"]
NUMERIC_COLS = [
    "temperature", "rain_mm", "latitude", "longitude", "list_price",
    "discount_pct", "promo_flag", "stock_on_hand", "stock_out_flag",
    "lead_time_days", "purchase_cost", "margin_pct", "quarter",
    "weekend_holiday", "discount_amount", "effective_price",
    "year", "month", "day", "weekday"
]

@st.cache_resource
def get_prediction_models():
    """
    Loads or trains fast pre-fitted inference models on sample gold data for Sales prediction.
    """
    os.makedirs("models", exist_ok=True)
    model_file = "models/live_inference_bundle.joblib"
    
    if os.path.exists(model_file):
        try:
            return joblib.load(model_file)
        except Exception:
            pass

    from sklearn.preprocessing import OrdinalEncoder
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    sample_df = load_sample_dataset(50000)
    if sample_df.empty:
        return None
        
    feat_cols = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c in sample_df.columns]
    X = sample_df[feat_cols].copy()
    y = sample_df["units_sold"].values
    
    cat_present = [c for c in CATEGORICAL_COLS if c in X.columns]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_present] = encoder.fit_transform(X[cat_present].astype(str))
    
    models = {
        "XGBoost": XGBRegressor(n_estimators=60, max_depth=6, learning_rate=0.1, random_state=42).fit(X, y),
        "LightGBM": LGBMRegressor(n_estimators=60, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1).fit(X, y),
        "CatBoost": CatBoostRegressor(iterations=60, depth=6, learning_rate=0.1, random_seed=42, verbose=0).fit(X, y),
        "Random Forest": RandomForestRegressor(n_estimators=40, max_depth=8, n_jobs=-1, random_state=42).fit(X, y),
    }
    
    bundle = {
        "models": models,
        "encoder": encoder,
        "cat_cols": cat_present,
        "feat_cols": feat_cols
    }
    joblib.dump(bundle, model_file)
    return bundle
