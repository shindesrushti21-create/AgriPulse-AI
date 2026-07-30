import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

train_data = data[data['split'] == 'train']
test_data = data[data['split'] == 'test']

X_train = train_data[feature_cols]
y_train = train_data[target_col]
X_test = test_data[feature_cols]
y_test = test_data[target_col]

# ------------------------------------------
# Retrain the same baseline model from Day 5 (for the confusion matrix)
# ------------------------------------------
model = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------------------
# Confusion matrix - shows exactly what gets confused with what
# ------------------------------------------
labels = ['Healthy', 'Emerging Stress', 'Persistent Stress', 'Recovery']
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("Confusion Matrix (rows = actual, columns = predicted):")
cm_df = pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])
print(cm_df)

# Save a visual version
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix - 4-Class Baseline Model')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig(docs_folder / "confusion_matrix_4class.png")
plt.close()
print("\nSaved confusion matrix plot to docs/confusion_matrix_4class.png")

# ==========================================
# EXPERIMENT: Simplified binary target
# "Stressed" = Emerging Stress OR Persistent Stress
# "Not Stressed" = Healthy OR Recovery
# ==========================================
def to_binary(state):
    if state in ['Emerging Stress', 'Persistent Stress']:
        return 'Stressed'
    else:
        return 'Not Stressed'

data['target_binary'] = data[target_col].apply(to_binary)

train_data_bin = data[data['split'] == 'train']
test_data_bin = data[data['split'] == 'test']

X_train_bin = train_data_bin[feature_cols]
y_train_bin = train_data_bin['target_binary']
X_test_bin = test_data_bin[feature_cols]
y_test_bin = test_data_bin['target_binary']

print("\n\n=== BINARY MODEL (Stressed vs Not Stressed) ===")
print("Train target distribution:\n", y_train_bin.value_counts())
print("\nTest target distribution:\n", y_test_bin.value_counts())

model_bin = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42, class_weight='balanced'
)
model_bin.fit(X_train_bin, y_train_bin)
y_pred_bin = model_bin.predict(X_test_bin)

accuracy_bin = accuracy_score(y_test_bin, y_pred_bin)
print(f"\nBinary model accuracy: {accuracy_bin:.3f}")
print("\nBinary model classification report:")
print(classification_report(y_test_bin, y_pred_bin, zero_division=0))

cm_bin = confusion_matrix(y_test_bin, y_pred_bin, labels=['Not Stressed', 'Stressed'])
print("\nBinary confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm_bin,
                    index=['Actual: Not Stressed', 'Actual: Stressed'],
                    columns=['Pred: Not Stressed', 'Pred: Stressed']))