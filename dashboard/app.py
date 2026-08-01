import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import shap
from pathlib import Path


with st.sidebar:
    st.markdown("### About AgriPulse AI")
    st.markdown(
        "A satellite-based early warning system for agricultural "
        "vegetation stress, built using Google Earth Engine, "
        "machine learning, and explainable AI."
    )
    st.markdown("---")
    st.markdown("**Pilot districts:**")
    st.markdown("- Pune (Baramati)\n- Beed\n- Nagpur\n- Kolhapur")
    st.markdown("---")
    st.markdown("**Data sources:**")
    st.markdown("- Sentinel-2 (NDVI, NDMI)\n- CHIRPS (rainfall)\n- MODIS (temperature)")
    st.markdown("---")
    st.caption("Built by Srushti Shinde")

# ------------------------------------------
# Page setup - must be the FIRST Streamlit command
# ------------------------------------------
st.set_page_config(
    page_title="AgriPulse AI - Vegetation Stress Monitor",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

# ------------------------------------------
# Minor custom styling for spacing and card appearance
# ------------------------------------------
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# Load data and model (cached so they don't reload on every interaction)
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(processed_folder / "all_districts_with_labels.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

data = load_data()

@st.cache_resource
def load_model():
    model_path = BASE_DIR / "models" / "stress_forecast_rf_model.pkl"
    loaded = joblib.load(model_path)
    return loaded['model'], loaded['features']

model, FEATURES = load_model()

@st.cache_data
def load_features_data():
    df = pd.read_csv(processed_folder / "all_districts_with_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

features_data = load_features_data()

# ------------------------------------------
# Header
# ------------------------------------------
st.title("🌾 AgriPulse AI")
st.markdown("**Satellite-based vegetation stress monitoring — Maharashtra pilot districts**")
st.markdown("---")

# ------------------------------------------
# STEP 1: Risk Forecast — must come first, this is where
# latest_features gets defined, before anything else uses it
# ------------------------------------------
st.subheader("🔮 Risk Forecast — Next Month")
st.caption("Predicted stress risk based on the most recent available satellite data")

latest_features = features_data.sort_values('date').groupby('district').tail(1)

cols = st.columns(len(latest_features))

for col, (_, row) in zip(cols, latest_features.iterrows()):
    with col:
        feature_row = row[FEATURES]
        if feature_row.isna().any():
            st.metric(label=row['district'], value="Insufficient data")
            continue

        X_input = pd.DataFrame([feature_row.values], columns=FEATURES)
        proba = model.predict_proba(X_input)[0]
        stressed_idx = list(model.classes_).index('Stressed')
        risk_pct = proba[stressed_idx] * 100

        if risk_pct >= 50:
            icon = "🔴"
            risk_label = "High Risk"
        elif risk_pct >= 25:
            icon = "🟡"
            risk_label = "Moderate Risk"
        else:
            icon = "🟢"
            risk_label = "Low Risk"

        st.metric(
            label=f"{icon} {row['district']}",
            value=risk_label,
            delta=f"{risk_pct:.0f}% stress probability"
        )
        st.caption(f"Based on data through {row['date'].strftime('%B %Y')}")

st.markdown("---")

# ------------------------------------------
# STEP 2: SHAP explanation - comes AFTER latest_features exists
# ------------------------------------------
st.subheader("💡 Why This Prediction?")

risk_scores = {}
for _, row in latest_features.iterrows():
    feature_row = row[FEATURES]
    if feature_row.isna().any():
        continue
    X_input = pd.DataFrame([feature_row.values], columns=FEATURES)
    proba = model.predict_proba(X_input)[0]
    stressed_idx = list(model.classes_).index('Stressed')
    risk_scores[row['district']] = proba[stressed_idx]

if risk_scores:
    highest_risk_district = max(risk_scores, key=risk_scores.get)
    highest_risk_pct = risk_scores[highest_risk_district] * 100

    st.markdown(f"**Highest current risk: {highest_risk_district} ({highest_risk_pct:.0f}%)**")

    explain_row = latest_features[latest_features['district'] == highest_risk_district].iloc[0]
    X_explain = pd.DataFrame([explain_row[FEATURES].values], columns=FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_explain)

    stressed_idx = list(model.classes_).index('Stressed')
    if isinstance(shap_vals, list):
        contributions = shap_vals[stressed_idx][0]
    else:
        contributions = shap_vals[0, :, stressed_idx]

    contrib_series = pd.Series(contributions, index=FEATURES).sort_values(key=abs, ascending=False)

    readable_names = {
        'zscore_lag1': "Last month's vegetation anomaly (Z-score)",
        'NDVI_lag1': "Last month's vegetation index (NDVI)",
        'rainfall_lag1': "Last month's rainfall",
        'NDVI_rolling3_mean': "3-month average vegetation index",
        'NDVI_change_lag1': "Recent month-over-month vegetation change",
        'NDMI_lag1': "Last month's vegetation moisture (NDMI)",
        'LST_lag1': "Last month's land surface temperature",
        'NDVI_lag2': "Vegetation index, 2 months ago",
        'rainfall_rolling3_sum': "Total rainfall over the past 3 months"
    }

    st.markdown("**Top factors influencing this forecast:**")
    for feat, val in contrib_series.head(3).items():
        direction = "⬆️ increased" if val > 0 else "⬇️ decreased"
        name = readable_names.get(feat, feat)
        st.markdown(f"- {name}: **{direction}** predicted risk")

    st.caption("Explanation generated using SHAP (SHapley Additive exPlanations) on the trained Random Forest model.")

st.markdown("---")

# ------------------------------------------
# STEP 3: Current Status (observed, historical - for context
# alongside the forecast above)
# ------------------------------------------
st.subheader("Current Status (Most Recent Available Month)")

latest_per_district = data.sort_values('date').groupby('district').tail(1)

cols2 = st.columns(len(latest_per_district))
status_colors = {"Healthy": "🟢", "Emerging Stress": "🟡", "Persistent Stress": "🔴", "Recovery": "🔵"}
for col, (_, row) in zip(cols2, latest_per_district.iterrows()):
    with col:
        icon = status_colors.get(row['stress_state'], "⚪")
        st.metric(label=f"{icon} {row['district']}", value=row['stress_state'], delta=f"NDVI: {row['NDVI']:.3f}")
        st.caption(f"As of {row['date'].strftime('%B %Y')}")

st.markdown("---")

# ------------------------------------------
# STEP 4: District explorer - trend chart + stats + raw data
# ------------------------------------------
st.subheader("Explore a District")

selected_district = st.selectbox(
    "Select a district to view its full history:",
    options=sorted(data['district'].unique())
)

district_data = data[data['district'] == selected_district].sort_values('date')

col1, col2, col3, col4 = st.columns(4)

with col1:
    healthy_pct = (district_data['stress_state'] == 'Healthy').mean() * 100
    st.metric("Healthy Months", f"{healthy_pct:.0f}%")

with col2:
    stress_months = district_data['stress_state'].isin(['Emerging Stress', 'Persistent Stress']).sum()
    st.metric("Total Stress Events", int(stress_months))

with col3:
    avg_ndvi = district_data['NDVI'].mean()
    st.metric("Average NDVI", f"{avg_ndvi:.3f}")

with col4:
    latest_state = district_data.iloc[-1]['stress_state']
    st.metric("Latest State", latest_state)

st.markdown(f"#### NDVI Trend — {selected_district}")

state_colors = {
    "Healthy": "#2ecc71",
    "Emerging Stress": "#f39c12",
    "Persistent Stress": "#e74c3c",
    "Recovery": "#3498db"
}

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=district_data['date'],
    y=district_data['NDVI'],
    mode='lines',
    line=dict(color='lightgray', width=1),
    showlegend=False,
    hoverinfo='skip'
))

for state, color in state_colors.items():
    subset = district_data[district_data['stress_state'] == state]
    fig.add_trace(go.Scatter(
        x=subset['date'],
        y=subset['NDVI'],
        mode='markers',
        name=state,
        marker=dict(color=color, size=8),
        hovertemplate='%{x|%b %Y}<br>NDVI: %{y:.3f}<extra></extra>'
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="NDVI",
    height=400,
    hovermode='closest',
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("View raw monthly data"):
    display_cols = ['date', 'NDVI', 'NDVI_zscore', 'rainfall_mm', 'LST_celsius', 'stress_state']
    st.dataframe(
        district_data[display_cols].sort_values('date', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption(
    "Stress states are derived from NDVI Z-score anomalies relative to each district's "
    "own historical (2019-2024) monthly baseline. Data source: Sentinel-2 (NDVI/NDMI), "
    "CHIRPS (rainfall), MODIS (land surface temperature), via Google Earth Engine."
)