# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Sentinel-1 SAR soil moisture integration
- MODIS annual land cover → temporal C-factor
- XGBoost / LightGBM models
- Uncertainty quantification (Monte Carlo)
- Full Maharashtra state coverage (35+ districts)

---

## [1.0.0] — 2024-12-01

### Added
- Full RUSLE analysis at **30m native SRTM resolution** for Western Maharashtra
- **22 numbered sections** in chronological execution order
- GPM IMERG V06 rainfall erosivity (R-factor) replacing CHIRPS
- SoilGrids v2 K-factor via EPIC equation (clay/sand/silt/SOC)
- SRTM LS-factor at native 30m — McCool (1989) slope steepness
- ESA WorldCover 10m → 30m C-factor + Sentinel-2 NDVI
- Slope-class P-factor (support practices)
- SCS-CN annual runoff calculation (prerequisite for SWC zones)
- 6-class erosion severity map (Very Low → Severe)
- Vulnerability Index (weighted composite, 0–100)
- Erosion hotspot mask (> 40 t/ha/yr)
- District-wise statistics for Satara, Sangli, Kolhapur, Solapur
- Temporal year-by-year RUSLE (2020–2023) + Mann-Kendall trend test
- Statistical tests: Shapiro-Wilk, Kruskal-Wallis, Spearman, VIF, Partial Correlations
- 8 ML models: RF, Extra Trees, Gradient Boosting, Ridge, Elastic Net, MLP, Stacking
- SHAP feature importance (falls back to permutation importance)
- K-Means, DBSCAN, Agglomerative spatial clustering
- PCA, t-SNE, UMAP dimensionality reduction
- 5-zone SWC Priority Map with actionable recommendations
- 15 core Plotly charts (Section 18)
- 7 advanced visualisations including 3D scatter, t-SNE, radar charts (Section 19)
- Dual interactive maps: geemap (inline) + Folium (inline + HTML export)
- One-click export: 17 GeoTIFFs @ 30m to Google Drive, 6 CSVs, 25+ HTML charts, ZIP bundle
- GitHub Actions CI: lint, notebook validation, offline unit tests
- Full repository structure with docs, tests, config, scripts

### Changed
- Resolution changed from 100m → **30m** (native SRTM, no degradation)
- SCS-CN Runoff moved to Section 9 (before RUSLE) — required by SWC Priority Zones
- Section ordering completely restructured for strict chronological dependency

### Removed
- CHIRPS rainfall (replaced by GPM IMERG)
- 100m resolution configuration

---

## [0.9.0] — 2024-10-15 *(Pre-release)*

### Added
- Initial RUSLE notebook at 100m resolution
- CHIRPS-based R-factor
- Basic district statistics
- Initial Plotly visualisations

---

[Unreleased]: https://github.com/YOUR_GITHUB_USERNAME/western-maharashtra-erosion/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/YOUR_GITHUB_USERNAME/western-maharashtra-erosion/releases/tag/v1.0.0
[0.9.0]: https://github.com/YOUR_GITHUB_USERNAME/western-maharashtra-erosion/releases/tag/v0.9.0
