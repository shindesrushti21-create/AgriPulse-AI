from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import shap
from pathlib import Path

# ------------------------------------------
# Setup
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"
models_folder = BASE_DIR / "models"

app = FastAPI(title="AgriPulse AI API")

# ------------------------------------------
# CORS - allows a frontend running on a different address
# (e.g., a Lovable app, or localhost:3000) to call this API.
# Without this, browsers block the request by default for security.
# ------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development; restrict this later for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# Load model and data ONCE when the server starts
# (not on every request - that would be slow)
# ------------------------------------------
loaded = joblib.load(models_folder / "stress_forecast_rf_model.pkl")
model = loaded['model']
FEATURES = loaded['features']

features_data = pd.read_csv(processed_folder / "all_districts_with_features.csv")
labels_data = pd.read_csv(processed_folder / "all_districts_with_labels.csv")

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

# ------------------------------------------
# Endpoint 1: List all districts
# ------------------------------------------
@app.get("/districts")
def get_districts():
    return {"districts": sorted(features_data['district'].unique().tolist())}

# ------------------------------------------
# Endpoint 2: Get risk forecast for ALL districts
# ------------------------------------------
@app.get("/risk")
def get_all_risk():
    latest = features_data.sort_values('date').groupby('district').tail(1)
    results = []

    for _, row in latest.iterrows():
        feature_row = row[FEATURES]
        if feature_row.isna().any():
            results.append({"district": row['district'], "risk_pct": None, "status": "Insufficient data"})
            continue

        X_input = pd.DataFrame([feature_row.values], columns=FEATURES)
        proba = model.predict_proba(X_input)[0]
        stressed_idx = list(model.classes_).index('Stressed')
        risk_pct = round(proba[stressed_idx] * 100, 1)

        results.append({
            "district": row['district'],
            "risk_pct": risk_pct,
            "date": row['date'],
        })

    return {"forecasts": results}

# ------------------------------------------
# Endpoint 3: Get SHAP explanation for a specific district
# ------------------------------------------
@app.get("/explain/{district}")
def explain_prediction(district: str):
    row_match = features_data[features_data['district'] == district].sort_values('date').tail(1)
    if row_match.empty:
        raise HTTPException(status_code=404, detail="District not found")

    row = row_match.iloc[0]
    feature_row = row[FEATURES]
    if feature_row.isna().any():
        raise HTTPException(status_code=400, detail="Insufficient data for this district")

    X_input = pd.DataFrame([feature_row.values], columns=FEATURES)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_input)

    stressed_idx = list(model.classes_).index('Stressed')
    if isinstance(shap_vals, list):
        contributions = shap_vals[stressed_idx][0]
    else:
        contributions = shap_vals[0, :, stressed_idx]

    contrib_series = pd.Series(contributions, index=FEATURES).sort_values(key=abs, ascending=False)

    top_factors = []
    for feat, val in contrib_series.head(3).items():
        top_factors.append({
            "feature": readable_names.get(feat, feat),
            "direction": "increased" if val > 0 else "decreased",
            "impact": round(float(val), 4)
        })

    return {"district": district, "top_factors": top_factors}

# ------------------------------------------
# Endpoint 4: Get historical NDVI trend for a district
# ------------------------------------------
@app.get("/history/{district}")
def get_history(district: str):
    district_data = labels_data[labels_data['district'] == district].sort_values('date')
    if district_data.empty:
        raise HTTPException(status_code=404, detail="District not found")

    history = district_data[['date', 'NDVI', 'stress_state']].to_dict(orient='records')
    return {"district": district, "history": history}