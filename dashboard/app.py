import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------
# Page setup - this must be the FIRST Streamlit command in the script
# ------------------------------------------
st.set_page_config(
    page_title="AgriPulse AI - Vegetation Stress Monitor",
    page_icon="🌾",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

# ------------------------------------------
# Load data - cached so it doesn't reload on every interaction
# (Streamlit reruns the ENTIRE script on every click/interaction -
# caching prevents re-reading the CSV every single time)
# ------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(processed_folder / "all_districts_with_labels.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

data = load_data()

# ------------------------------------------
# Header
# ------------------------------------------
st.title("🌾 AgriPulse AI")
st.markdown("**Satellite-based vegetation stress monitoring — Maharashtra pilot districts**")
st.markdown("---")

# ------------------------------------------
# Get the MOST RECENT month's data per district - this is the
# "current status" view, the single most important thing a user
# wants to see first
# ------------------------------------------
latest_per_district = data.sort_values('date').groupby('district').tail(1)

st.subheader("Current Status (Most Recent Available Month)")

# ------------------------------------------
# Show current status as colored cards - one per district
# This uses Streamlit's column layout to place 4 districts side by side
# ------------------------------------------
cols = st.columns(len(latest_per_district))

status_colors = {
    "Healthy": "🟢",
    "Emerging Stress": "🟡",
    "Persistent Stress": "🔴",
    "Recovery": "🔵"
}

for col, (_, row) in zip(cols, latest_per_district.iterrows()):
    with col:
        icon = status_colors.get(row['stress_state'], "⚪")
        st.metric(
            label=f"{icon} {row['district']}",
            value=row['stress_state'],
            delta=f"NDVI: {row['NDVI']:.3f}"
        )
        st.caption(f"As of {row['date'].strftime('%B %Y')}")

st.markdown("---")

# ------------------------------------------
# District selector - lets the user pick which district to explore
# in detail below
# ------------------------------------------
st.subheader("Explore a District")

selected_district = st.selectbox(
    "Select a district to view its full history:",
    options=sorted(data['district'].unique())
)

district_data = data[data['district'] == selected_district].sort_values('date')

# ------------------------------------------
# Summary stats for the selected district
# ------------------------------------------
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

# ------------------------------------------
# NDVI trend chart, color-coded by stress state
# ------------------------------------------
st.markdown(f"#### NDVI Trend — {selected_district}")

state_colors = {
    "Healthy": "#2ecc71",
    "Emerging Stress": "#f39c12",
    "Persistent Stress": "#e74c3c",
    "Recovery": "#3498db"
}

fig = go.Figure()

# Main NDVI line
fig.add_trace(go.Scatter(
    x=district_data['date'],
    y=district_data['NDVI'],
    mode='lines',
    line=dict(color='lightgray', width=1),
    showlegend=False,
    hoverinfo='skip'
))

# Colored dots per stress state, one trace per state so the legend works cleanly
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

# ------------------------------------------
# Raw data table - now secondary, collapsed by default
# ------------------------------------------
with st.expander("View raw monthly data"):
    display_cols = ['date', 'NDVI', 'NDVI_zscore', 'rainfall_mm', 'LST_celsius', 'stress_state']
    st.dataframe(
        district_data[display_cols].sort_values('date', ascending=False),
        use_container_width=True,
        hide_index=True
    )
# ------------------------------------------
# Footer / methodology note - honesty about what this shows
# ------------------------------------------
st.markdown("---")
st.caption(
    "Stress states are derived from NDVI Z-score anomalies relative to each district's "
    "own historical (2019-2024) monthly baseline. Data source: Sentinel-2 (NDVI/NDMI), "
    "CHIRPS (rainfall), MODIS (land surface temperature), via Google Earth Engine."
)