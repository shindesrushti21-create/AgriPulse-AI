# Project Log

## Week 1 — Data Pipeline (Pune/Baramati, Beed, Nagpur, Kolhapur)

### What was built
- A reusable Google Earth Engine (GEE) pipeline that extracts monthly
  vegetation and climate features per district:
  - NDVI (vegetation health) and NDMI (moisture) from Sentinel-2, with
    QA60 pixel-level cloud masking
  - Monthly rainfall totals from CHIRPS
  - Monthly mean land surface temperature (LST) from MODIS, converted to Celsius
- District boundaries sourced from FAO GAUL 2015 (official admin boundaries,
  no manual polygon drawing needed)
- Time range: January 2019 to the last fully completed month (dynamically
  calculated, so the script stays correct without manual updates)
- Applied to 4 pilot districts chosen for contrasting agro-climatic conditions:
  Pune/Baramati (mixed irrigated/rain-fed), Beed (drought-prone, Marathwada),
  Nagpur (Vidarbha, different rainfall/crop pattern), Kolhapur (wetter,
  used as a "healthy" comparison region)

### Bugs encountered and fixed
- **Cloud join fragility**: initially joined Sentinel-2 SR with the Cloud
  Probability dataset by system:index; some dates didn't match and threw
  errors. Switched to Sentinel-2's built-in QA60 band for cloud masking —
  slightly less precise, but far more robust (no join required).
- **Interactive console memory/aggregation limits**: printing large tables
  and charts directly in the console hit "User memory limit exceeded" and
  "Too many concurrent aggregations" errors. Fixed by combining all 4
  variables (NDVI, NDMI, rainfall, LST) into a single multi-band image per
  month, reduced in one `reduceRegion` call instead of four, and moving
  straight to Export rather than forcing interactive evaluation.
- **Empty image collections**: MODIS LST had at least one month with zero
  available images across the full 2019–2026 range, causing a
  "0 bands vs 1 band" multiply error. Fixed by checking image count before
  computing stats and substituting a safe placeholder image when a
  collection is empty.
- **Boolean filter bug**: `is_complete_month` was stored as a GEE Number
  (1/0), but the filter compared it against the literal boolean `true`,
  silently filtering out every row and producing an empty (2-byte) export.
  Fixed by filtering against `1` instead of `true`.
- **District name mismatch**: "Beed" returned zero results when filtering
  FAO GAUL by ADM2_NAME. Diagnosed by printing all Maharashtra district
  names from the dataset — discovered it's stored as "Bid". Fixed the
  filter, while keeping "Beed" as the human-readable label in the
  exported `district` column.

### Data quality observations
- All four districts show clear, realistic seasonal NDVI cycles that
  track the monsoon (rainfall rising June–September, NDVI peaking
  Oct–Nov).
- Beed shows a notably lower dry-season NDVI floor (~0.10–0.17,
  Mar–Jun) compared to Pune/Baramati (~0.24–0.30) — consistent with
  Beed's known drought-prone reputation, and a good sign the pipeline
  captures real regional differences rather than noise.
- One anomaly noted: in an earlier 2023–2024 test run, monsoon months
  (Jul–Aug) showed unexpectedly low NDVI (~0.02–0.06), likely residual
  cloud/shadow contamination not fully caught by QA60 during heavy
  monsoon cover. Flagged as a known limitation; Cloud Score+ or the
  Cloud Probability dataset identified as a future improvement if this
  recurs in the full dataset.

### Next (Week 2)
- Compute historical per-month mean and standard deviation of NDVI for
  each district (2019–2024 baseline).
- Convert monthly NDVI into a Z-score anomaly relative to that baseline.
- Define stress states (Healthy / Emerging Stress / Persistent Stress /
  Recovery) using Z-score thresholds, requiring persistence across 2+
  consecutive months to avoid one-off noise being mislabeled as stress.
- Validate that label distribution looks reasonable per district before
  moving to model training.


## Week 2 — NDVI Anomaly Detection & Stress Labeling

### What was built
- Combined all 4 district CSVs (Pune/Baramati, Beed, Nagpur, Kolhapur)
  from Week 1 into a single dataset (360 monthly records).
- Computed a historical baseline for each district, for each calendar
  month, using 2019-2024 data: the mean and standard deviation of NDVI.
- Calculated a Z-score for every month:
  Z = (NDVI - historical monthly mean) / historical monthly std dev
  This measures how unusual a given month's vegetation health is,
  relative to that same district's own typical pattern for that month
  of the year (comparing July to July, not July to January).
- Verified the Z-score distribution visually (histograms per district)
  before proceeding - confirmed no broken/extreme values.

### Threshold recalibration (data-driven, not assumed)
- The originally planned fixed threshold (Z <= -2 for "severe stress")
  was never reached anywhere in the dataset - the most extreme month
  observed across all 4 districts was Z = -1.99.
- Rather than force an unreached threshold, thresholds were recalibrated
  using the actual observed Z-score percentiles:
  - Emerging Stress: Z below the 10th percentile (~-1.29)
  - Persistent Stress: Z below the 5th percentile (~-1.58), required
    to persist for 2+ consecutive months (to avoid a single noisy
    month being mislabeled as a sustained stress event)
  - Recovery: assigned when a district was previously in a stress
    state and NDVI climbs back above the Emerging Stress cutoff
- This produced a believable label distribution: 303 Healthy, 31
  Emerging Stress, 21 Recovery, 5 Persistent Stress months (out of 360
  total). No single district dominates the stress categories; Nagpur
  had zero Persistent Stress months, consistent with its different
  agro-climatic profile.

### Verification
- Manually traced Beed's full month-by-month sequence (2019-2026)
  against the raw NDVI and Z-score values to confirm the state
  transition logic behaves correctly - e.g., two consecutive severe
  months correctly escalate to "Persistent Stress," and recovery is
  only assigned after a prior stress period, not randomly.

### Known limitation (documented, not hidden)
- The historical baseline period (2019-2024) is not fully independent
  from the data being labeled, since it includes the same years being
  evaluated. With only 6 years of data per calendar-month baseline, a
  single unusual year can influence what counts as "normal." This is
  a standard limitation in short-record remote sensing studies, but
  worth being explicit about rather than presenting the thresholds as
  more validated than they are. A longer historical baseline (10+
  years) would make this more robust in a future version.

### Next (Week 3)
- Feature engineering: build predictive features from data available
  *before* the month being predicted (e.g., prior month's NDVI, prior
  month's rainfall, rolling averages) - critical to avoid data leakage
  when forecasting next month's stress state.
- Time-based train/test split (train on 2019-2024, test on 2025-2026)
  rather than random shuffling, since this is a forecasting problem.
- Train a baseline Random Forest classifier to predict next month's
  stress state transition.