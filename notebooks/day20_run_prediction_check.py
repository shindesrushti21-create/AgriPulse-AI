import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"
models_folder = BASE_DIR / "models"

# Load model
loaded = joblib.load(models_folder / "stress_forecast_rf_model.pkl")
model = loaded['model']
FEATURES = loaded['features']

# Load latest features
data = pd.read_csv(processed_folder / "all_districts_with_features.csv")
latest = data.sort_values('date').groupby('district').tail(1)

log_lines = [f"AgriPulse Risk Check - {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
log_lines.append("=" * 50)

for _, row in latest.iterrows():
    feature_row = row[FEATURES]
    if feature_row.isna().any():
        log_lines.append(f"{row['district']}: Insufficient data")
        continue
    X_input = pd.DataFrame([feature_row.values], columns=FEATURES)
    proba = model.predict_proba(X_input)[0]
    stressed_idx = list(model.classes_).index('Stressed')
    risk_pct = proba[stressed_idx] * 100
    log_lines.append(f"{row['district']}: {risk_pct:.0f}% stress risk")

output = "\n".join(log_lines)
print(output)

# Save a log file
log_path = BASE_DIR / "automation" / "last_risk_check.txt"
with open(log_path, "w") as f:
    f.write(output)

print(f"\nSaved log to: {log_path}")