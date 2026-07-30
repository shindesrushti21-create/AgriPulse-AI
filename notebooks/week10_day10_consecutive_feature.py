import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

input_path = processed_folder / "all_districts_model_ready.csv"
data = pd.read_csv(input_path)
data = data.sort_values(['district', 'year', 'month']).reset_index(drop=True)

# ------------------------------------------
# Build "consecutive stress months" feature
# Counts how many consecutive PRIOR months had a negative NDVI Z-score
# (i.e., below that district's historical average for that month).
# Uses shift(1) so the streak is "known as of last month" - consistent
# with all other features in this project, which only use lagged,
# not current-month, values.
# ------------------------------------------
def build_consecutive_streak(group_df):
    group_df = group_df.copy()
    z_shifted = group_df['NDVI_zscore'].shift(1)

    streak = []
    current_streak = 0
    for z in z_shifted:
        if pd.isna(z):
            current_streak = 0
        elif z < 0:
            current_streak += 1
        else:
            current_streak = 0
        streak.append(current_streak)

    group_df['consecutive_stress_months'] = streak
    return group_df

pieces = []
for district_name in data['district'].unique():
    subset = data[data['district'] == district_name]
    pieces.append(build_consecutive_streak(subset))

data = pd.concat(pieces, ignore_index=True)

# ------------------------------------------
# Sanity check - verify the streak counts make sense
# ------------------------------------------
print("--- Consecutive stress streak check (Beed, first 10 rows) ---")
check = data[data['district'] == 'Beed'][['date', 'NDVI_zscore', 'consecutive_stress_months']].head(10)
print(check.to_string(index=False))

# ------------------------------------------
# Rebuild binary target (same as Day 6/8/9)
# ------------------------------------------
def to_binary(state):
    return 'Stressed' if state in ['Emerging Stress', 'Persistent Stress'] else 'Not Stressed'

data['target_binary'] = data['target_next_state'].apply(to_binary)

# ------------------------------------------
# Drop rows with missing target (some rows may have NaN target
# from Day 4's dropna - reload from all_districts_model_ready.csv
# already handles this, but double-check)
# ------------------------------------------
data = data.dropna(subset=['target_binary'])

# ------------------------------------------
# NEW feature set - original 9 features + the new one
# ------------------------------------------
original_features = [
    'NDVI_lag1', 'NDMI_lag1', 'rainfall_lag1', 'LST_lag1', 'zscore_lag1',
    'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1'
]
new_features = original_features + ['consecutive_stress_months']

train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

y_train = train_data['target_binary']
y_test = test_data['target_binary']

# ------------------------------------------
# Model WITHOUT new feature (Day 6 baseline, re-run for direct comparison)
# ------------------------------------------
X_train_orig = train_data[original_features]
X_test_orig = test_data[original_features]

model_orig = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model_orig.fit(X_train_orig, y_train)
pred_orig = model_orig.predict(X_test_orig)

print("\n=== WITHOUT consecutive_stress_months (Day 6 baseline) ===")
print(classification_report(y_test, pred_orig, zero_division=0))
f1_orig = f1_score(y_test, pred_orig, pos_label='Stressed')

# ------------------------------------------
# Model WITH new feature
# ------------------------------------------
X_train_new = train_data[new_features]
X_test_new = test_data[new_features]

model_new = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight='balanced')
model_new.fit(X_train_new, y_train)
pred_new = model_new.predict(X_test_new)

print("\n=== WITH consecutive_stress_months ===")
print(classification_report(y_test, pred_new, zero_division=0))
f1_new = f1_score(y_test, pred_new, pos_label='Stressed')

# ------------------------------------------
# Direct comparison
# ------------------------------------------
print(f"\n=== COMPARISON ===")
print(f"Stressed-class F1 WITHOUT new feature: {f1_orig:.3f}")
print(f"Stressed-class F1 WITH new feature:    {f1_new:.3f}")

# ------------------------------------------
# Feature importance for the new model - where does the new
# feature rank?
# ------------------------------------------
importances = pd.Series(model_new.feature_importances_, index=new_features).sort_values(ascending=False)
print("\nFeature importance (with new feature):")
print(importances)

# ------------------------------------------
# Save updated dataset with the new feature included
# ------------------------------------------
output_path = processed_folder / "all_districts_model_ready_v2.csv"
data.to_csv(output_path, index=False)
print("\nSaved:", output_path)