import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

input_path = processed_folder / "all_districts_model_ready.csv"
data = pd.read_csv(input_path)

feature_cols = [
    'NDVI_lag1', 'NDMI_lag1', 'rainfall_lag1', 'LST_lag1', 'zscore_lag1',
    'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1'
]
target_col = 'target_next_state'

def to_binary(state):
    return 'Stressed' if state in ['Emerging Stress', 'Persistent Stress'] else 'Not Stressed'

data['target_binary'] = data[target_col].apply(to_binary)

train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

X_train = train_data[feature_cols]
X_test = test_data[feature_cols]
y_train = train_data['target_binary']
y_test = test_data['target_binary']

# ------------------------------------------
# XGBoost needs numeric labels, not text - encode them
# ------------------------------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
print("Label encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# ------------------------------------------
# Re-train Random Forest (Day 6 baseline) for direct side-by-side comparison
# ------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n=== RANDOM FOREST (Day 6 baseline, re-run here for comparison) ===")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(classification_report(y_test, rf_pred, zero_division=0))

# ------------------------------------------
# Train XGBoost
# scale_pos_weight compensates for class imbalance, XGBoost's
# equivalent to class_weight='balanced'
# ------------------------------------------
n_not_stressed = (y_train == 'Not Stressed').sum()
n_stressed = (y_train == 'Stressed').sum()
scale_pos_weight = n_not_stressed / n_stressed

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train_encoded)
xgb_pred_encoded = xgb_model.predict(X_test)
xgb_pred = le.inverse_transform(xgb_pred_encoded)

print("\n=== XGBOOST ===")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.3f}")
print(classification_report(y_test, xgb_pred, zero_division=0))

# ------------------------------------------
# Side-by-side comparison summary
# ------------------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n=== SIDE-BY-SIDE COMPARISON (Stressed class only) ===")
print(f"{'Metric':<12}{'RandomForest':<15}{'XGBoost':<15}")
print(f"{'Precision':<12}{precision_score(y_test, rf_pred, pos_label='Stressed'):<15.3f}{precision_score(y_test, xgb_pred, pos_label='Stressed'):<15.3f}")
print(f"{'Recall':<12}{recall_score(y_test, rf_pred, pos_label='Stressed'):<15.3f}{recall_score(y_test, xgb_pred, pos_label='Stressed'):<15.3f}")
print(f"{'F1':<12}{f1_score(y_test, rf_pred, pos_label='Stressed'):<15.3f}{f1_score(y_test, xgb_pred, pos_label='Stressed'):<15.3f}")