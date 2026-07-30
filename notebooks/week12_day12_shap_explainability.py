import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"
models_folder = BASE_DIR / "models"
docs_folder = BASE_DIR / "docs"

# ------------------------------------------
# Load the saved model + its feature list (from Day 11)
# ------------------------------------------
model_path = models_folder / "stress_forecast_rf_model.pkl"
loaded = joblib.load(model_path)
model = loaded['model']
FEATURES = loaded['features']

print("Loaded model. Expected features:", FEATURES)

# ------------------------------------------
# Load data and rebuild the test set
# ------------------------------------------
data = pd.read_csv(processed_folder / "all_districts_model_ready.csv")

def to_binary(state):
    return 'Stressed' if state in ['Emerging Stress', 'Persistent Stress'] else 'Not Stressed'

data['target_binary'] = data['target_next_state'].apply(to_binary)
test_data = data[data['split'] == 'test'].reset_index(drop=True)

X_test = test_data[FEATURES]

# ------------------------------------------
# Build the SHAP explainer for this Random Forest
# TreeExplainer is specifically optimized for tree-based models
# (Random Forest, XGBoost, etc.) - fast and exact, not approximate
# ------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ------------------------------------------
# shap_values structure check - for binary classification, this
# gives values for each class. We want the "Stressed" class index.
# ------------------------------------------
print("\nModel classes:", model.classes_)
stressed_idx = list(model.classes_).index('Stressed')

# Handle both possible SHAP output formats (list of arrays, or 3D array)
if isinstance(shap_values, list):
    shap_values_stressed = shap_values[stressed_idx]
else:
    shap_values_stressed = shap_values[:, :, stressed_idx]

# ------------------------------------------
# Global summary plot: which features matter most, across ALL
# test predictions (a SHAP-based alternative to Day 7's feature
# importance, but shows direction too - not just magnitude)
# ------------------------------------------
plt.figure()
shap.summary_plot(shap_values_stressed, X_test, show=False)
plt.tight_layout()
plt.savefig(docs_folder / "shap_summary_plot.png", bbox_inches='tight')
plt.close()
print("\nSaved global summary plot to docs/shap_summary_plot.png")

# ------------------------------------------
# Individual explanation: pick one real predicted-Stressed case
# and explain exactly why the model flagged it
# ------------------------------------------
predictions = model.predict(X_test)
stressed_predictions_idx = [i for i, p in enumerate(predictions) if p == 'Stressed']

if len(stressed_predictions_idx) > 0:
    example_idx = stressed_predictions_idx[0]
    example_row = test_data.iloc[example_idx]

    print(f"\n=== Individual Explanation Example ===")
    print(f"District: {example_row['district']}, Date: {example_row['date']}")
    print(f"Actual next-month state: {example_row['target_next_state']}")
    print(f"Model predicted: {predictions[example_idx]}")

    print("\nFeature values for this example:")
    print(X_test.iloc[example_idx])

    print("\nSHAP contribution per feature (positive = pushes toward Stressed):")
    contributions = pd.Series(shap_values_stressed[example_idx], index=FEATURES).sort_values(key=abs, ascending=False)
    print(contributions)

# Force plot for this single example (saved as an image)
    print("\nexplainer.expected_value type:", type(explainer.expected_value))
    print("explainer.expected_value:", explainer.expected_value)

    # Robustly extract a single scalar base value, regardless of
    # whether expected_value is a list, tuple, or numpy array
    ev = explainer.expected_value
    if hasattr(ev, '__len__') and len(ev) > 1:
        base_value = ev[stressed_idx]
    else:
        base_value = ev

    shap.plots.force(
        base_value,
        shap_values_stressed[example_idx],
        X_test.iloc[example_idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig(docs_folder / "shap_individual_example.png", bbox_inches='tight')
    plt.close()
    print("\nSaved individual explanation plot to docs/shap_individual_example.png")