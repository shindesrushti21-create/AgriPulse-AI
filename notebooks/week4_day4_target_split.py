import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

input_path = processed_folder / "all_districts_with_features.csv"
output_path = processed_folder / "all_districts_model_ready.csv"

# ------------------------------------------
# Load Day 3 output
# ------------------------------------------
data = pd.read_csv(input_path)
data = data.sort_values(['district', 'year', 'month']).reset_index(drop=True)

print("Loaded data shape:", data.shape)

# ------------------------------------------
# Step 1: Create the forecasting target
# ------------------------------------------
def build_target(group_df):
    group_df = group_df.copy()
    group_df['target_next_state'] = group_df['stress_state'].shift(-1)
    return group_df

target_pieces = []
for district_name in data['district'].unique():
    subset = data[data['district'] == district_name]
    target_pieces.append(build_target(subset))

data = pd.concat(target_pieces, ignore_index=True)

print("\n--- TARGET CHECK (Beed, first 8 rows) ---")
check = data[data['district'] == 'Beed'][['date', 'stress_state', 'target_next_state']].head(8)
print(check.to_string(index=False))

# ------------------------------------------
# Step 2: Drop rows with missing features or missing target
# ------------------------------------------
feature_cols = [
    'NDVI_lag1', 'NDMI_lag1', 'rainfall_lag1', 'LST_lag1', 'zscore_lag1',
    'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1'
]

before_drop = data.shape[0]
data_clean = data.dropna(subset=feature_cols + ['target_next_state']).copy()
after_drop = data_clean.shape[0]

print(f"\nRows before dropping incomplete rows: {before_drop}")
print(f"Rows after dropping incomplete rows: {after_drop}")
print(f"Rows dropped: {before_drop - after_drop}")

# ------------------------------------------
# Step 3: Time-based split, repositioned based on evidence
# Stress events cluster in 2022-2023 (see yearly breakdown below);
# a 2024-cutoff split left the test set with zero stress examples.
# Train: 2019-2022 | Test: 2023-2024 | Recent holdout: 2025-2026
# ------------------------------------------
train_data = data_clean[data_clean['year'] <= 2022].copy()
test_data = data_clean[(data_clean['year'] >= 2023) & (data_clean['year'] <= 2024)].copy()
holdout_recent = data_clean[data_clean['year'] >= 2025].copy()

print(f"\nTrain set (2019-2022): {train_data.shape[0]} rows")
print(f"Test set (2023-2024): {test_data.shape[0]} rows")
print(f"Recent holdout (2025-2026): {holdout_recent.shape[0]} rows")

print("\nTrain set target distribution:")
print(train_data['target_next_state'].value_counts())

print("\nTest set target distribution:")
print(test_data['target_next_state'].value_counts())

print("\nRecent holdout target distribution:")
print(holdout_recent['target_next_state'].value_counts())

# ------------------------------------------
# Yearly stress distribution (evidence for the split choice above)
# ------------------------------------------
print("\n--- Stress state distribution by year, all districts ---")
yearly_check = data_clean.groupby(['year', 'stress_state']).size().unstack(fill_value=0)
print(yearly_check)

# ------------------------------------------
# Step 4: Assign the 'split' column (train / test / holdout_recent)
# This goes on data_clean itself, so Day 5 can load one file and
# filter by this column directly.
# ------------------------------------------
def assign_split(y):
    if y <= 2022:
        return 'train'
    elif y <= 2024:
        return 'test'
    else:
        return 'holdout_recent'

data_clean['split'] = data_clean['year'].apply(assign_split)

print("\nFinal split counts:")
print(data_clean['split'].value_counts())

# ------------------------------------------
# Save
# ------------------------------------------
data_clean.to_csv(output_path, index=False)
print("\nSaved:", output_path)