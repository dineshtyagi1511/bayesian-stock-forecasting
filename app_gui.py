import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# --- CONFIGURATION ---
API_URL = "http://localhost:8000"
st.set_page_config(
    page_title="BAYES-TRADER | Quantum Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Futuristic UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #00ffc8; }
    .stMetric { background-color: #1a1c24; border: 1px solid #00ffc8; padding: 15px; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #00ffc8; font-family: 'Courier New', monospace; }
    .stButton>button { width: 100%; border-radius: 5px; background-color: #00ffc8; color: black; font-weight: bold; }
    .stButton>button:hover { background-color: #00d4a8; border: 1px solid #fff; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("⚡ QUANTUM BAYESIAN COMMAND")
st.markdown(f"**API Status:** `CONNECTED` | **Version:** `1.0.0` | **Environment:** `PRODUCTION` ")
st.divider()

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("📡 SENSOR CONTROLS")
    ticker = st.selectbox("SELECT ASSET", ["NIFTY50", "RELIANCE", "TCS"], index=0)
    model = st.selectbox("INFERENCE ENGINE", ["ensemble", "xgboost", "sarima", "bayesian"])
    horizon = st.slider("HORIZON (DAYS)", 1, 30, 5)
    
    if st.button("EXECUTE FORECAST"):
        with st.spinner("Decrypting Market Signals..."):
            try:
                payload = {"ticker": ticker, "model": model, "horizon_days": horizon}
                response = requests.post(f"{API_URL}/forecast", json=payload)
                data = response.json()
                st.session_state['last_forecast'] = data
            except Exception as e:
                st.error(f"Connection Failed: {e}")

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1, 1, 1])

if 'last_forecast' in st.session_state:
    f = st.session_state['last_forecast']
    
    # 1. Metrics Grid
    with col1:
        st.metric("PREDICTED RETURN", f"{f['predicted_return_pct']}%", 
                  delta=f['direction'], delta_color="normal")
    with col2:
        st.metric("CONFIDENCE SCORE", f"{f['confidence']*100:.1f}%")
    with col3:
        st.metric("MLFLOW RUN ID", f.get('mlflow_run_id', 'N/A'))

    st.divider()

    # 2. Visualizing Uncertainty (Bayesian logic)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("🔮 Probability Distribution")
        
        # Simulated distribution based on API response
        mu = f['predicted_return_pct']
        # If CI exists, use it; otherwise use a default spread
        std = (f['upper_95_pct'] - f['lower_95_pct'])/4 if f['lower_95_pct'] else 0.5
        
        x = pd.Series(np.random.normal(mu, std, 1000))
        fig = px.histogram(x, nbins=50, title="Posterior Return Distribution",
                           color_discrete_sequence=['#00ffc8'], opacity=0.7)
        fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("📊 Engine Specs")
        st.info(f"**Model:** {f['model_used'].upper()}")
        if f['lower_95_pct']:
            st.write(f"**95% Credible Interval:**")
            st.code(f"[{f['lower_95_pct']}% , {f['upper_95_pct']}%]")
            st.progress(f['confidence'])
        else:
            st.warning("This model does not provide uncertainty quantification (Point Forecast Only).")

# --- COMPARISON SECTION ---
st.divider()
st.subheader("⚔️ Strategy A/B Battle")
ab_col1, ab_col2 = st.columns(2)

with ab_col1:
    strat_a = st.selectbox("Strategy A", ["sarima", "xgboost"], index=0)
    strat_b = st.selectbox("Strategy B", ["xgboost", "bayesian"], index=0)
    
    if st.button("RUN BAYESIAN COMPARISON"):
        res = requests.post(f"{API_URL}/ab-test?ticker={ticker}&strategy_a={strat_a}&strategy_b={strat_b}")
        ab_data = res.json()
        st.session_state['ab_data'] = ab_data

if 'ab_data' in st.session_state:
    abd = st.session_state['ab_data']
    with ab_col2:
        st.success(f"**Verdict:** {abd['verdict']}")
        
        # Gauge Chart for Probability
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = abd['prob_b_beats_a'] * 100,
            title = {'text': f"P({strat_b} > {strat_a})"},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "#00ffc8"},
                'bar': {'color': "#00ffc8"},
                'steps': [
                    {'range': [0, 50], 'color': '#333'},
                    {'range': [90, 100], 'color': '#004433'}
                ],
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# --- SYSTEM LOGS ---
with st.expander("🛠️ SYSTEM METRICS & LOGS"):
    metrics_res = requests.get(f"{API_URL}/metrics/{ticker}")
    if metrics_res.status_code == 200:
        st.table(pd.DataFrame(metrics_res.json()['models']).T)