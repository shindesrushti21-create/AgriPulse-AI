import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, classification_report
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
# Train the same Random Forest as Day 6/8
# ------------------------------------------
model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)

# ------------------------------------------
# Get predicted PROBABILITIES, not just the final label
# model.classes_ tells us which column corresponds to "Stressed"
# ------------------------------------------
print("Model classes order:", model.classes_)
stressed_index = list(model.classes_).index('Stressed')

y_proba = model.predict_proba(X_test)[:, stressed_index]

# Convert y_test to binary 1/0 for metric functions (1 = Stressed)
y_test_binary = (y_test == 'Stressed').astype(int)

# ------------------------------------------
# Try a range of thresholds manually, print precision/recall/F1 for each
# ------------------------------------------
print("\nThreshold sweep:")
print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")

thresholds_to_test = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
results = []

for t in thresholds_to_test:
    y_pred_at_t = (y_proba >= t).astype(int)
    p = precision_score(y_test_binary, y_pred_at_t, zero_division=0)
    r = recall_score(y_test_binary, y_pred_at_t, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_at_t, zero_division=0)
    results.append({'threshold': t, 'precision': p, 'recall': r, 'f1': f1})
    print(f"{t:<12}{p:<12.3f}{r:<12.3f}{f1:<12.3f}")

results_df = pd.DataFrame(results)

# ------------------------------------------
# Find the threshold with the best F1 (best balance)
# ------------------------------------------
best_row = results_df.loc[results_df['f1'].idxmax()]
print(f"\nBest threshold by F1 score: {best_row['threshold']} "
      f"(Precision={best_row['precision']:.3f}, Recall={best_row['recall']:.3f}, F1={best_row['f1']:.3f})")

# ------------------------------------------
# Full precision-recall curve (finer-grained than the manual sweep)
# ------------------------------------------
precisions, recalls, pr_thresholds = precision_recall_curve(y_test_binary, y_proba)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recalls, precisions, marker='.', color='steelblue')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve - Stressed Class (Random Forest)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(docs_folder / "precision_recall_curve.png")
plt.close()
print("\nSaved plot to docs/precision_recall_curve.png")

# ------------------------------------------
# Show full classification report at the BEST threshold found
# ------------------------------------------
best_threshold = best_row['threshold']
y_pred_best = (y_proba >= best_threshold).astype(int)
y_pred_best_labels = np.where(y_pred_best == 1, 'Stressed', 'Not Stressed')

print(f"\nFull classification report at best threshold ({best_threshold}):")
print(classification_report(y_test, y_pred_best_labels, zero_division=0))