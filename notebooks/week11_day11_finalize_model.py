import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"
models_folder = BASE_DIR / "models"
models_folder.mkdir(exist_ok=True)

input_path = processed_folder / "all_districts_model_ready.csv"
data = pd.read_csv(input_path)

# ------------------------------------------
# Final chosen feature set - the original 9 features.
# (consecutive_stress_months tested on Day 10, did not improve
# results, excluded here)
# ------------------------------------------
FINAL_FEATURES = [
    'NDVI_lag1', 'NDMI_lag1', 'rainfall_lag1', 'LST_lag1', 'zscore_lag1',
    'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1'
]

def to_binary(state):
    return 'Stressed' if state in ['Emerging Stress', 'Persistent Stress'] else 'Not Stressed'

data['target_binary'] = data['target_next_state'].apply(to_binary)

train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

X_train = train_data[FINAL_FEATURES]
y_train = train_data['target_binary']
X_test = test_data[FINAL_FEATURES]
y_test = test_data['target_binary']

# ------------------------------------------
# Train the final model (Day 6/9 configuration - confirmed best
# across all Week 4 experiments)
# ------------------------------------------
final_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("=== FINAL MODEL - Full Evaluation ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------
# Save the trained model + the exact feature list it expects
# (saving the feature list alongside the model avoids future
# mistakes about which columns/order it needs)
# ------------------------------------------
model_path = models_folder / "stress_forecast_rf_model.pkl"
joblib.dump({'model': final_model, 'features': FINAL_FEATURES}, model_path)
print(f"\nSaved final model to: {model_path}")

# ------------------------------------------
# Verify the saved model loads and predicts identically
# ------------------------------------------
loaded = joblib.load(model_path)
loaded_model = loaded['model']
loaded_features = loaded['features']

verify_pred = loaded_model.predict(X_test[loaded_features])
matches = (verify_pred == y_pred).all()
print(f"Verification - loaded model produces identical predictions: {matches}")

# ------------------------------------------
# Week 4 summary table - every attempt, side by side
# (numbers taken from Days 6, 8, 9, 10 - hardcoded here since
# each was run in a separate script; this is a documentation
# summary, not a re-computation)
# ------------------------------------------
summary = pd.DataFrame([
    {"Attempt": "Day 6: Random Forest (9 features, default threshold)", "Precision": 0.32, "Recall": 0.64, "F1": 0.42},
    {"Attempt": "Day 8: XGBoost (same 9 features)",                     "Precision": 0.15, "Recall": 0.18, "F1": 0.17},
    {"Attempt": "Day 9: Random Forest, tuned threshold (0.4)",          "Precision": 0.32, "Recall": 0.64, "F1": 0.42},
    {"Attempt": "Day 10: Random Forest + consecutive-streak feature",   "Precision": 0.29, "Recall": 0.55, "F1": 0.375},
])
print("\n=== Week 4 Summary - Stressed Class Metrics ===")
print(summary.to_string(index=False))
print("\nFinal choice: Day 6 configuration (Random Forest, 9 features, default threshold)")
print("Rationale: Best F1 among all attempts; simpler model preferred when tied on performance.")