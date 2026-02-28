# 📦 Data Sources

All datasets are accessed via the **Google Earth Engine** public data catalog. No manual downloads are required.

---

## Primary Datasets

### 1. GPM IMERG Final V06 — Rainfall

| Field | Value |
|:---|:---|
| **GEE Asset** | `NASA/GPM_L3/IMERG_V06` |
| **Provider** | NASA / JAXA |
| **Band used** | `precipitationCal` (mm/hr) |
| **Native resolution** | 0.1° (~10 km) |
| **Temporal** | Half-hourly, Jan 2000–present |
| **Study period** | 2020-01-01 → 2023-12-31 |
| **Processing** | × 24 → mm/day; sum per month/year; bilinear resample to 30m |
| **Citation** | Huffman, G.J., et al. (2019). doi:10.5067/GPM/IMERG/3B-HH/06 |

### 2. SRTMGL1 v003 — Elevation

| Field | Value |
|:---|:---|
| **GEE Asset** | `USGS/SRTMGL1_003` |
| **Provider** | NASA / USGS |
| **Band used** | `elevation` (m) |
| **Native resolution** | 1 arc-second (~30m) |
| **Used at** | **Native 30m** — no resampling for LS-factor |
| **Derived products** | Slope (°), TRI (Terrain Ruggedness Index) |
| **Citation** | Farr, T.G., et al. (2007). doi:10.1029/2005RG000183 |

### 3. SoilGrids v2 — Soil Properties

| Field | Value |
|:---|:---|
| **GEE Assets** | `projects/soilgrids-isric/clay_mean` `sand_mean` `silt_mean` `soc_mean` |
| **Provider** | ISRIC — World Soil Information |
| **Depth** | 0–5cm (surface layer) |
| **Bands used** | `clay_0-5cm_mean`, `sand_0-5cm_mean`, `silt_0-5cm_mean`, `soc_0-5cm_mean` |
| **Units** | g/kg (texture) · dg/kg (SOC) — divided by 10 in code |
| **Native resolution** | ~250m |
| **Processing** | Bilinear resample to 30m |
| **Citation** | Poggio, L., et al. (2021). SOIL, 7(1), 217–240. doi:10.5194/soil-7-217-2021 |

### 4. ESA WorldCover v200 — Land Cover

| Field | Value |
|:---|:---|
| **GEE Asset** | `ESA/WorldCover/v200` |
| **Provider** | ESA / VITO |
| **Year** | 2021 |
| **Band used** | `Map` (integer class code) |
| **Native resolution** | 10m |
| **Processing** | Mode aggregation to 30m (majority class per 30m cell) |
| **Classes used** | 10 (Trees), 20 (Shrub), 30 (Grass), 40 (Crop), 50 (Built-up), 60 (Bare), 80 (Water) |
| **Citation** | Zanaga, D., et al. (2022). doi:10.5281/zenodo.7254221 |

### 5. Sentinel-2 SR Harmonized — NDVI

| Field | Value |
|:---|:---|
| **GEE Asset** | `COPERNICUS/S2_SR_HARMONIZED` |
| **Provider** | ESA / Google |
| **Bands used** | `B8` (NIR), `B4` (Red) → NDVI = (B8−B4)/(B8+B4) |
| **Period** | 2023-01-01 → 2023-12-31 |
| **Cloud filter** | `CLOUDY_PIXEL_PERCENTAGE` < 20% |
| **Composite** | Annual median |
| **Native resolution** | 10m |
| **Processing** | Mean aggregation to 30m |
| **Citation** | ESA Sentinel-2 Mission. https://sentinel.esa.int/web/sentinel/missions/sentinel-2 |

### 6. FAO GAUL 2015 Level-2 — Administrative Boundaries

| Field | Value |
|:---|:---|
| **GEE Asset** | `FAO/GAUL/2015/level2` |
| **Provider** | Food and Agriculture Organization |
| **Fields used** | `ADM1_NAME` (state), `ADM2_NAME` (district) |
| **Districts** | Satara, Sangli, Kolhapur, Solapur |
| **Note** | Used for study area definition and district-wise statistics only |

---

## Derived Products

| Product | Source | GEE Operation |
|:---|:---|:---|
| **Daily Rainfall** | GPM IMERG | × 24 (mm/hr → mm/day) |
| **Annual Rainfall** | GPM Daily | `.sum()` over study years ÷ n_years |
| **Monsoon Rainfall** | GPM Daily | Filter months 6–9, `.sum()` ÷ n_years |
| **MFI** | GPM Monthly | Σ(pₘ² / P_annual) |
| **Slope** | SRTM | `ee.Terrain.slope()` |
| **TRI** | SRTM | `reduceNeighborhood(stdDev, kernel(3))` |
| **Annual Runoff** | GPM + WorldCover | SCS-CN formula |

---

## Data Access via GEE Python API

```python
import ee
ee.Initialize(project='your-project-id')

# Example: load GPM IMERG
gpm = (ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
       .filterDate('2020-01-01', '2023-12-31')
       .select('precipitationCal'))

# Example: load SRTM at native 30m
dem = ee.Image("USGS/SRTMGL1_003").select('elevation')

# Example: load SoilGrids clay
clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
```

---

## Licences & Terms of Use

| Dataset | Licence |
|:---|:---|
| GPM IMERG | NASA Open Data |
| SRTM | Public Domain (USGS) |
| SoilGrids | CC BY 4.0 |
| ESA WorldCover | CC BY 4.0 |
| Sentinel-2 | Free & Open (Copernicus) |
| FAO GAUL | CC BY-NC 3.0 IGO |

All data is accessed through the Google Earth Engine Terms of Service: https://earthengine.google.com/terms/
