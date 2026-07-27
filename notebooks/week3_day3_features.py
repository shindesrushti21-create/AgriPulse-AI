import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
processed_folder = BASE_DIR / "data" / "processed"

input_path = processed_folder / "all_districts_with_labels.csv"
output_path = processed_folder / "all_districts_with_features.csv"

# ------------------------------------------
# Load Week 2 output
# ------------------------------------------
data = pd.read_csv(input_path)
data = data.sort_values(['district', 'year', 'month']).reset_index(drop=True)

print("Loaded data shape:", data.shape)

# ------------------------------------------
# Build lag/rolling features PER DISTRICT
# (must be done separately per district so Beed's Dec doesn't
# leak into Nagpur's Jan, etc.)
# ------------------------------------------
def build_features(group_df):
    group_df = group_df.copy()

    # Previous month's raw values (1-month lag)
    group_df['NDVI_lag1'] = group_df['NDVI'].shift(1)
    group_df['NDMI_lag1'] = group_df['NDMI'].shift(1)
    group_df['rainfall_lag1'] = group_df['rainfall_mm'].shift(1)
    group_df['LST_lag1'] = group_df['LST_celsius'].shift(1)
    group_df['zscore_lag1'] = group_df['NDVI_zscore'].shift(1)

    # 2-month lag (gives the model a sense of trend/direction)
    group_df['NDVI_lag2'] = group_df['NDVI'].shift(2)

    # 3-month rolling average of NDVI, using only past data
    # shift(1) first so the current month is excluded from its own average
    group_df['NDVI_rolling3_mean'] = group_df['NDVI'].shift(1).rolling(window=3).mean()

    # Rainfall accumulated over the past 3 months
    group_df['rainfall_rolling3_sum'] = group_df['rainfall_mm'].shift(1).rolling(window=3).sum()

    # Month-over-month NDVI change (is it declining or improving?)
    group_df['NDVI_change_lag1'] = group_df['NDVI'].shift(1) - group_df['NDVI'].shift(2)

    return group_df

feature_pieces = []
for district_name in data['district'].unique():
    subset = data[data['district'] == district_name]
    feature_pieces.append(build_features(subset))

data = pd.concat(feature_pieces, ignore_index=True)

# ------------------------------------------
# LEAKAGE CHECK - critical, do not skip
# Confirm every new feature column only uses PAST information,
# by manually checking one row against raw values
# ------------------------------------------
print("\n--- LEAKAGE CHECK (Beed, first 10 rows) ---")
check = data[data['district'] == 'Beed'][
    ['date', 'NDVI', 'NDVI_lag1', 'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_mm', 'rainfall_rolling3_sum']
].head(10)
print(check.to_string(index=False))

print("\nManually verify: for any row, 'NDVI_lag1' should equal the PREVIOUS")
print("row's 'NDVI' value - not the current row's value. Check this by eye above.")

# ------------------------------------------
# Check how many rows have missing lag values (expected at the
# start of each district's time series, since there's no "previous
# month" for the very first month)
# ------------------------------------------
print("\nMissing values per feature column:")
feature_cols = ['NDVI_lag1', 'NDVI_lag2', 'NDVI_rolling3_mean', 'rainfall_rolling3_sum', 'NDVI_change_lag1']
print(data[feature_cols].isna().sum())

print(f"\nExpected missing rows: {data['district'].nunique()} districts x up to 3 months each (start of series)")

# ------------------------------------------
# Save
# ------------------------------------------
data.to_csv(output_path, index=False)
print("\nSaved:", output_path)