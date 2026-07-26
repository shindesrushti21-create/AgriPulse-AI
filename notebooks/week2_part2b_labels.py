import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

input_path = processed_folder / "all_districts_with_zscore.csv"
output_path = processed_folder / "all_districts_with_labels.csv"

data = pd.read_csv(input_path)

# ------------------------------------------
# Fix naming: FAO GAUL dataset used "Bid" internally for Beed district
# Rename here so all outputs use the correct, human-readable name
# ------------------------------------------
data['district'] = data['district'].replace({'Bid': 'Beed'})

data = data.sort_values(['district', 'year', 'month']).reset_index(drop=True)

print("Loaded data shape:", data.shape)
print("Districts:", data['district'].unique())

# ------------------------------------------
# Data-driven thresholds, based on observed Z-score percentiles
# (fixed -2 threshold was never reached in this dataset - see
# percentile check: min Z was -1.99)
# ------------------------------------------
emerging_threshold = data['NDVI_zscore'].quantile(0.10)   # ~ -1.29
persistent_threshold = data['NDVI_zscore'].quantile(0.05)  # ~ -1.58

print(f"\nUsing data-driven thresholds:")
print(f"Emerging Stress cutoff (10th percentile): {emerging_threshold:.3f}")
print(f"Persistent Stress cutoff (5th percentile): {persistent_threshold:.3f}")

def assign_states(group_df):
    states = []
    prev_state = "Healthy"

    for z in group_df['NDVI_zscore']:
        if pd.isna(z):
            states.append("Unknown")
            prev_state = "Unknown"
            continue

        if z > emerging_threshold:
            if prev_state in ["Emerging Stress", "Persistent Stress"]:
                current_state = "Recovery"
            else:
                current_state = "Healthy"
        elif persistent_threshold < z <= emerging_threshold:
            current_state = "Emerging Stress"
        else:  # z <= persistent_threshold
            if prev_state in ["Emerging Stress", "Persistent Stress"]:
                current_state = "Persistent Stress"
            else:
                current_state = "Emerging Stress"

        states.append(current_state)
        prev_state = current_state

    group_df = group_df.copy()
    group_df['stress_state'] = states
    return group_df

labeled_pieces = []
for district_name in data['district'].unique():
    subset = data[data['district'] == district_name]
    labeled_subset = assign_states(subset)
    labeled_pieces.append(labeled_subset)

data = pd.concat(labeled_pieces, ignore_index=True)

print("\nColumns after labeling:", data.columns.tolist())

print("\nStress state counts per district:")
print(data.groupby(['district', 'stress_state']).size().unstack(fill_value=0))

print("\nOverall stress state distribution:")
print(data['stress_state'].value_counts())

print("\nSample sequence check (Beed):")
sample = data[data['district'] == 'Beed'][['date', 'NDVI', 'NDVI_zscore', 'stress_state']]
print(sample.to_string(index=False))

# ------------------------------------------
# Check: which months ever hit severe anomaly threshold (Z <= -2)?
# Helps verify whether "Persistent Stress" should logically appear
# ------------------------------------------
print("\nMonths with Z <= -2 (severe anomaly candidates):")
severe = data[data['NDVI_zscore'] <= -2][['district', 'date', 'NDVI_zscore', 'stress_state']]
print(severe.to_string(index=False))

# ------------------------------------------
# Check the actual distribution of Z-scores to calibrate
# a data-driven "Persistent Stress" threshold instead of
# assuming -2 in advance
# ------------------------------------------
print("\nZ-score percentiles (all districts combined):")
print(data['NDVI_zscore'].describe(percentiles=[0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95]))

data.to_csv(output_path, index=False)
print("\nSaved:", output_path)