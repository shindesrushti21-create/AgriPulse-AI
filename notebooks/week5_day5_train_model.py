import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

input_path = processed_folder / "all_districts_model_ready.csv"

# ------------------------------------------
# Load Day 4 output
# ------------------------------------------
data = pd.read_csv(input_path)
print("Loaded data shape:", data.shape)
print("Split counts:\n", data['split'].value_counts())

# ------------------------------------------
# Define which columns are FEATURES (inputs) vs TARGET (what we predict)
# Only lagged/rolling features - never the current month's own NDVI/
# rainfall/etc, since those wouldn't be "known" before the fact in
# a real forecasting scenario.
# ------------------------------------------
feature_cols = [
    'NDVI_lag1', 'NDMI_lag1', 'rainfall_lag1', 'LST_lag1', 'zscore_lag1',
    'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1'
]
target_col = 'target_next_state'

# ------------------------------------------
# Split into train / test using the 'split' column from Day 4
# ------------------------------------------
train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

X_train = train_data[feature_cols]
y_train = train_data[target_col]

X_test = test_data[feature_cols]
y_test = test_data[target_col]

print(f"\nTraining on {len(X_train)} rows, testing on {len(X_test)} rows")

# ------------------------------------------
# Train the Random Forest
# ------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,      # number of trees in the forest
    max_depth=6,           # limits tree complexity - helps avoid overfitting on a small dataset
    random_state=42,       # ensures reproducible results every time you rerun this
    class_weight='balanced'  # compensates for "Healthy" being the majority class
)

model.fit(X_train, y_train)

# ------------------------------------------
# Predict on the test set
# ------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------
# Evaluate - report honestly, don't chase a high number
# ------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall accuracy on test set: {accuracy:.3f}")

print("\nFull classification report (per-class precision/recall/F1):")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------
# Feature importance - which inputs mattered most to the model
# ------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importance:")
print(importances)