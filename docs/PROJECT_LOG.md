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