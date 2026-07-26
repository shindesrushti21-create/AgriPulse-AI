import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------
# Step 0: Robust paths - works regardless of where script is run from
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_folder = BASE_DIR / "data" / "raw"
processed_folder = BASE_DIR / "data" / "processed"
docs_folder = BASE_DIR / "docs"

print("Base directory:", BASE_DIR)
print("Looking for data in:", data_folder)

# ------------------------------------------
# Step 1: Load all 4 district CSVs
# ------------------------------------------
files = {
    "Pune_Baramati": "pune_baramati_monthly_features_2019_2026.csv",
    "Beed": "beed_monthly_features_2019_2026.csv",
    "Nagpur": "nagpur_monthly_features_2019_2026.csv",
    "Kolhapur": "kolhapur_monthly_features_2019_2026.csv",
}

all_dfs = []
for district_name, filename in files.items():
    filepath = data_folder / filename
    df = pd.read_csv(filepath)
    all_dfs.append(df)

data = pd.concat(all_dfs, ignore_index=True)

print("\nCombined data shape:", data.shape)
print(data.head())
print("\nDistricts found:", data['district'].unique())

# ------------------------------------------
# Step 2: Historical mean & std of NDVI, per district+month (2019-2024 baseline)
# ------------------------------------------
baseline = data[(data['year'] >= 2019) & (data['year'] <= 2024)]

monthly_stats = baseline.groupby(['district', 'month'])['NDVI'].agg(['mean', 'std']).reset_index()
monthly_stats.columns = ['district', 'month', 'NDVI_hist_mean', 'NDVI_hist_std']

print("\nHistorical monthly baseline (sample):")
print(monthly_stats.head(12))

# ------------------------------------------
# Step 3: Merge baseline stats back into main data
# ------------------------------------------
data = data.merge(monthly_stats, on=['district', 'month'], how='left')

# ------------------------------------------
# Step 4: Compute Z-score
# ------------------------------------------
data['NDVI_zscore'] = (data['NDVI'] - data['NDVI_hist_mean']) / data['NDVI_hist_std']

print("\nSample rows with Z-score:")
print(data[['district', 'date', 'NDVI', 'NDVI_hist_mean', 'NDVI_hist_std', 'NDVI_zscore']].head(15))

# ------------------------------------------
# Step 5: Sanity check plot
# ------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
districts = data['district'].unique()

for ax, district in zip(axes.flatten(), districts):
    subset = data[data['district'] == district]
    ax.hist(subset['NDVI_zscore'].dropna(), bins=20, edgecolor='black')
    ax.set_title(f"NDVI Z-score distribution - {district}")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(docs_folder / "zscore_distribution_check.png")
plt.show()

# ------------------------------------------
# Step 6: Save enriched dataset
# ------------------------------------------
processed_folder.mkdir(parents=True, exist_ok=True)
output_path = processed_folder / "all_districts_with_zscore.csv"
data.to_csv(output_path, index=False)
print("\nSaved:", output_path)