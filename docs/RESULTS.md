# 📊 Results — Western Maharashtra Soil Erosion Analysis

> All values are computed at runtime from GEE. The numbers below are representative estimates based on test runs; exact values are printed in the notebook's **Section 22 Final Summary**.

---

## 1. Study Area Overview

| Parameter | Value |
|:---|:---|
| Total study area | ~35,000 km² |
| Analysis period | 2020 – 2023 |
| Spatial resolution | 30m (native SRTM) |
| Coordinate system | EPSG:32643 (UTM Zone 43N) |
| GEE sample points | 8,000 |

---

## 2. Soil Loss Summary (Study Area)

| Statistic | Approximate Value |
|:---|:---|
| Mean soil loss | 15–25 t/ha/yr |
| Median soil loss | 8–14 t/ha/yr |
| Range | 0 – >100 t/ha/yr |
| Hotspot area (>40 t/ha/yr) | 8–15% of study area |

---

## 3. District-Wise Results

| District | Key Characteristic | Expected Erosion Level |
|:---|:---|:---:|
| **Satara** | Western Ghats escarpment — steep slopes, very high rainfall | High |
| **Kolhapur** | Krishna River basin — moderate slopes, high rainfall | Moderate–High |
| **Sangli** | Flat to undulating — irrigated crops, moderate rainfall | Low–Moderate |
| **Solapur** | Semi-arid Deccan plateau — low rainfall, sparse cover | Low–Moderate |

> Exact mean/median/P75 soil loss values, hotspot areas (ha), vulnerability scores, and erosion class areas are computed per district in **Section 13** and saved to `district_statistics.csv`.

---

## 4. Erosion Class Distribution

Approximate area distribution across the study region:

| Class | t/ha/yr | Estimated % of Area |
|:---:|:---:|:---:|
| Very Low | 0–5 | ~35–45% |
| Low | 5–10 | ~20–25% |
| Moderate | 10–20 | ~15–20% |
| High | 20–40 | ~8–12% |
| Very High | 40–80 | ~4–7% |
| Severe | >80 | ~1–3% |

---

## 5. Temporal Trend (2020–2023)

Year-by-year analysis in **Section 14** computes annual mean/median soil loss and applies a Mann-Kendall trend test. Results depend on inter-annual GPM rainfall variability.

Key metrics printed at runtime:
- Mann-Kendall τ (tau)
- p-value
- Trend direction: Increasing / Decreasing / No Significant Trend

---

## 6. Machine Learning Performance

| Model | R² (approx.) | Rank |
|:---|:---:|:---:|
| Random Forest | 0.93–0.97 | 🥇 |
| Extra Trees | 0.91–0.95 | 🥈 |
| Stacking Ensemble | 0.90–0.94 | 🥉 |
| Gradient Boosting | 0.87–0.92 | 4th |
| Neural Net (MLP) | 0.82–0.88 | 5th |
| Ridge | 0.70–0.78 | 6th |
| Elastic Net | 0.68–0.76 | 7th |

> Exact R², MAE, RMSE are saved to `ml_model_performance.csv`.

---

## 7. Top Erosion Drivers (Permutation Importance)

Typical ranking (varies by run):

1. **LS_Factor** — topographic slope-length steepness (dominant)
2. **Annual_Rainfall / R_Factor** — high Western Ghats monsoon intensity
3. **Slope** — direct driver of LS; strong spatial gradient in study area
4. **NDVI** — vegetation cover quality and density
5. **C_Factor** — land-cover type (cropland vs. forest)

---

## 8. Spatial Clusters (K-Means, k=5)

Five spatially coherent erosion zones are identified by k-means clustering on {Soil_Loss, LS_Factor, Rainfall, NDVI, TRI, Vulnerability}. Typical cluster profiles:

| Cluster | Profile |
|:---:|:---|
| **0** | Low erosion — dense forest, gentle slope, moderate rainfall |
| **1** | Moderate erosion — rainfed cropland, rolling terrain |
| **2** | High erosion — steep Ghats escarpment, high rainfall |
| **3** | Very high erosion — degraded bare slopes, concentrated runoff |
| **4** | Moderate risk — semi-arid Deccan, low rainfall but sparse cover |

---

## 9. SWC Priority Zones

Indicative areas by zone (varies by run, saved to `SWC_PriorityZones_30m.tif`):

| Zone | Treatment | Notes |
|:---:|:---|:---|
| 1 | Bunding & Terracing | Concentrated on Ghats foothills |
| 2 | Check Dams | Valley bottoms and gully networks |
| 3 | Vegetative Barriers | Cropland on moderate slopes |
| 4 | Afforestation | Bare hillslopes in Satara/Kolhapur |
| 5 | Agroforestry | Sloped cropland across all districts |

---

## 10. Key Outputs Produced

| File | Contents |
|:---|:---|
| `SoilLoss_30m.tif` | Soil loss in t/ha/yr (Float32 GeoTIFF) |
| `ErosionClass_30m.tif` | 1–6 severity class (Int8 GeoTIFF) |
| `VulnerabilityIndex_30m.tif` | 0–100 composite risk |
| `SWC_PriorityZones_30m.tif` | 0–5 SWC zone codes |
| `sample_points_30m.csv` | 8,000 labelled sample points |
| `district_statistics.csv` | Complete per-district summary |
| `ml_model_performance.csv` | R², MAE, RMSE for all models |
| `temporal_trend.csv` | Year-by-year statistics |
| `feature_importance.csv` | SHAP/permutation importance |
| `western_maharashtra_interactive_map.html` | Folium map with all GEE layers |
| `plot_*.html` (×25) | Interactive Plotly charts |

---

*For full methodology behind these results, see [METHODOLOGY.md](METHODOLOGY.md).*
