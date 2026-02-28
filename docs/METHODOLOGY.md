# 🔬 Methodology — Western Maharashtra Soil Erosion Analysis

## Overview

This analysis implements the **Revised Universal Soil Loss Equation (RUSLE)** entirely within Google Earth Engine at **30m native resolution**, combined with advanced machine learning for feature importance analysis, spatial clustering, and erosion class prediction.

---

## 1. RUSLE Framework

The RUSLE equation estimates long-term average annual soil loss:

```
A = R × K × LS × C × P
```

| Symbol | Factor | Unit | Source |
|:---:|:---|:---|:---|
| **A** | Annual soil loss | t/ha/yr | Computed |
| **R** | Rainfall erosivity | MJ·mm/ha/h/yr | GPM IMERG |
| **K** | Soil erodibility | t·h/MJ·mm | SoilGrids |
| **LS** | Slope length-steepness | Dimensionless | SRTM 30m |
| **C** | Cover-management | 0–1 | WorldCover + S2 |
| **P** | Support practices | 0–1 | Slope-based |

---

## 2. R-Factor (Rainfall Erosivity)

**Data:** NASA/JAXA GPM IMERG Final V06 — `precipitationCal` band (mm/hr)

**Method:** Wischmeier & Smith (1978) regression using the Modified Fournier Index (MFI):

```
R = 0.5 × MFI + 0.363 × P_annual + 79   [MJ·mm/ha/h/yr]

MFI = Σ(pₘ² / P_annual)    summed over 12 months
```

Where `pₘ` is mean monthly precipitation (mm) and `P_annual` is mean annual precipitation (mm). GPM half-hourly data at 0.1° (~10km) is bilinearly resampled to 30m.

---

## 3. K-Factor (Soil Erodibility)

**Data:** ISRIC SoilGrids v2 — clay, sand, silt (g/kg ÷ 10 → %), SOC (dg/kg ÷ 10 → g/100g) at 0–5cm depth

**Method:** EPIC equation (Williams et al., 1990):

```python
f_csand = exp(-0.01 × sand%) × 0.3 + 0.2           # coarse sand factor
f_clsi  = (silt% / (clay% + silt%))^0.3              # clay-silt interaction
f_orgC  = (SOC% × 0.1 + 1)^(-0.5)                   # organic carbon factor

K = f_csand × f_clsi × f_orgC × 0.1317   [t·h/MJ·mm]
```

SoilGrids at 250m is bilinearly resampled to 30m.

---

## 4. LS-Factor (Slope Length-Steepness)

**Data:** USGS/NASA SRTMGL1_003 — elevation at **native 30m resolution**

**Method:** McCool et al. (1989) slope steepness factor with fixed 30m slope length:

```python
# Slope steepness (S)
S = sin(θ) × 10.8 + 0.03    if slope < 9°
S = sin(θ) × 16.8 - 0.50    if slope ≥ 9°

# Slope length (L), fixed cell size = 30m
L = (30 / 22.13)^0.5 ≈ 1.164

LS = L × S
```

No resolution resampling is applied — SRTM 30m is used at its native resolution, preserving maximum topographic detail.

---

## 5. C-Factor (Cover Management)

**Data:**
- ESA WorldCover v200 at 10m → mode-aggregated to 30m
- Sentinel-2 SR Harmonized 2023 median NDVI at 10m → mean-aggregated to 30m

**C-value lookup:**

| WorldCover Code | Class | C-Value |
|:---:|:---|:---:|
| 10 | Tree cover | 0.001 |
| 20 | Shrubland | 0.050 |
| 30 | Grassland | 0.010 |
| 40 | Cropland | 0.200 |
| 50 | Built-up | 0.000 |
| 60 | Bare/sparse | 0.450 |
| 70 | Snow/ice | 0.000 |
| 80 | Permanent water | 0.000 |
| 90 | Herbaceous wetland | 0.001 |
| 95 | Mangroves | 0.000 |
| 100 | Moss/lichen | 0.000 |
| (default) | Unclassified | 0.350 |

---

## 6. P-Factor (Support Practices)

Based on USDA slope-class guidelines:

| Slope (°) | P-Value |
|:---:|:---:|
| < 2 | 0.60 |
| 2–5 | 0.50 |
| 5–8 | 0.60 |
| 8–12 | 0.70 |
| 12–16 | 0.80 |
| 16–20 | 0.90 |
| > 20 | 1.00 |

Water bodies and built-up areas receive P = 0.

---

## 7. SCS-CN Annual Runoff

**Method:** NRCS Curve Number method for estimating runoff depth from annual rainfall:

```
S  = 25400 / CN − 254           (potential maximum retention, mm)
Ia = 0.2 × S                    (initial abstraction ≈ 20% of S)

Q  = (P − Ia)² / (P − Ia + S)  when P > Ia
Q  = 0                          when P ≤ Ia
```

CN values assigned from WorldCover mode class at 30m:
- Tree cover → CN = 70
- Grassland → CN = 80
- Cropland → CN = 85
- Bare/sparse → CN = 90
- Default → CN = 75

---

## 8. Vulnerability Index

A multi-criteria, slope-weighted composite (0–100):

```
VI = 35% × Slope_norm + 30% × (1−NDVI)_norm + 20% × Rainfall_norm + 15% × TRI_norm
```

Normalization:
- Slope → ÷ 30 (30° ≈ maximum practical threshold)
- NDVI → inverted: (1 − NDVI), so low vegetation = high vulnerability
- Rainfall → ÷ 3000 (high monsoon benchmark)
- TRI → ÷ 50 (50 = very rugged terrain)

All values clamped to [0, 1] before weighting, then multiplied by 100.

---

## 9. Machine Learning Framework

### Purpose
ML is used for:
1. **Regression** — predict soil loss from the feature stack (independent validation)
2. **Feature importance** — quantify which RUSLE factors most drive erosion
3. **Multi-class classification** — predict erosion severity class
4. **Spatial clustering** — identify spatially coherent erosion zones
5. **Dimensionality reduction** — visualise high-dimensional feature space

### Feature Engineering
15 input features:
```
R_Factor, K_Factor, LS_Factor, C_Factor, P_Factor,
Slope, Elevation, TRI, Annual_Rainfall, Monsoon_Rainfall,
NDVI, Clay, Sand, Silt, SOC
```

### Model Training Protocol
- **Preprocessing:** RobustScaler (IQR-based, robust to outliers)
- **Train/Test split:** 80/20 stratified
- **Cross-validation:** 5-fold for tree models
- **Outlier handling:** Target clipped at 99th percentile before training

---

## 10. Temporal Analysis

Annual soil loss is computed for each year (2020–2023) by deriving a year-specific R-factor from annual GPM data and combining with the time-invariant K, LS, C, P factors.

**Mann-Kendall Trend Test:**
```
τ = Kendall rank correlation (year vs. mean soil loss)
p-value < 0.05 → statistically significant trend
```

---

## 11. SWC Priority Zones

Zones are defined by GEE conditional logic combining soil loss, slope, NDVI, land cover, and runoff:

| Zone | Algorithm |
|:---:|:---|
| 1 — Bunding | SL ≥ 20 AND slope 5–15° |
| 2 — Check Dams | SL ≥ 10 AND runoff ≥ 100mm AND slope < 5° |
| 3 — Vegetative | SL ≥ 5 AND NDVI < 0.35 AND cropland |
| 4 — Afforestation | SL ≥ 10 AND bare/sparse land |
| 5 — Agroforestry | SL ≥ 10 AND cropland AND slope ≥ 5° |

---

## References

1. Wischmeier, W.H. & Smith, D.D. (1978). Predicting Rainfall Erosion Losses. USDA Agriculture Handbook No. 537.
2. Renard, K.G. et al. (1997). RUSLE: Guide to Conservation Planning. USDA-ARS.
3. McCool, D.K. et al. (1989). Revised slope length factor for USLE. Transactions ASAE, 32(5), 1571–1576.
4. Williams, J.R. et al. (1990). EPIC — Erosion/Productivity Impact Calculator: Model Documentation. USDA.
5. Huffman, G.J. et al. (2019). NASA Global Precipitation Measurement (GPM) IMERG. Algorithm Theoretical Basis Document.
6. Poggio, L. et al. (2021). SoilGrids 2.0. SOIL, 7(1), 217–240.
7. Zanaga, D. et al. (2022). ESA WorldCover 10m 2021 v200. Zenodo.
8. Gorelick, N. et al. (2017). Google Earth Engine: Planetary-scale geospatial analysis. Remote Sensing of Environment, 202, 18–27.
