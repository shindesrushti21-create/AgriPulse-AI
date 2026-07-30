import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"
docs_folder = BASE_DIR / "docs"

input_path = processed_folder / "all_districts_model_ready.csv"
data = pd.read_csv(input_path)

feature_cols = [
    'NDVI_lag1', 'NDMI_lag1', 'rainfall_lag1', 'LST_lag1', 'zscore_lag1',
    'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1'
]
target_col = 'target_next_state'

# ------------------------------------------
# Rebuild the binary target (same logic as Day 6)
# ------------------------------------------
def to_binary(state):
    if state in ['Emerging Stress', 'Persistent Stress']:
        return 'Stressed'
    else:
        return 'Not Stressed'

data['target_binary'] = data[target_col].apply(to_binary)

train_data = data[data['split'] == 'train']
X_train = train_data[feature_cols]
y_train = train_data['target_binary']

# ------------------------------------------
# Train the binary model (same settings as Day 6)
# ------------------------------------------
model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)

# ------------------------------------------
# Feature importance
# ------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("Feature importance (binary Stressed vs Not Stressed model):")
print(importances)

# ------------------------------------------
# Plot it
# ------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
importances.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance - Binary Stress Forecasting Model')
plt.tight_layout()
plt.savefig(docs_folder / "feature_importance_binary.png")
plt.close()
print("\nSaved plot to docs/feature_importance_binary.png")