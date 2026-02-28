# 🤝 Contributing to Western Maharashtra Soil Erosion Analysis

Thank you for your interest in contributing! This is an open-science project — every contribution, no matter how small, makes a real difference for land-use planners and researchers studying soil degradation.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Notebook Conventions](#notebook-conventions)
- [Testing](#testing)
- [Documentation](#documentation)
- [Commit Message Format](#commit-message-format)

---

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing. We are committed to a welcoming environment for everyone.

---

## Ways to Contribute

| Type | Description |
|:---|:---|
| 🐛 **Bug fixes** | Fix incorrect RUSLE calculations, broken GEE calls, or visualisation errors |
| 🌍 **New regions** | Extend the analysis to other districts / states |
| 📦 **New data sources** | Integrate Sentinel-1, MODIS LULC, ERA5 PET, etc. |
| 🤖 **New ML models** | Add XGBoost, LightGBM, CatBoost, or deep learning models |
| 📊 **Visualisations** | New Plotly chart types or improved styling |
| 📖 **Documentation** | Improve methodology docs, add tutorials |
| 🧪 **Tests** | Expand the offline test suite |
| 🌐 **Translations** | Translate README / docs to Marathi, Hindi |

---

## Development Workflow

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/western-maharashtra-erosion.git
cd western-maharashtra-erosion

# 3. Create your feature branch (from main)
git checkout -b feature/add-xgboost-model

# 4. Set up your environment
conda env create -f environment.yml
conda activate erosion-analysis

# 5. Make your changes
# → For notebook changes: test full top-to-bottom run in Colab first
# → For script changes: add/update tests in tests/

# 6. Run tests
pytest tests/ -v

# 7. Commit (see commit format below)
git add .
git commit -m "feat(ml): add XGBoost regressor with early stopping"

# 8. Push and open a PR
git push origin feature/add-xgboost-model
# → Open Pull Request on GitHub using the PR template
```

---

## Code Style

- **Python:** Follow PEP 8. Max line length 110 characters.
- **Notebooks:** See [Notebook Conventions](#notebook-conventions).
- **Run:** `flake8 scripts/ tests/ --max-line-length=110`
- **Docstrings:** Google-style for all public functions.

```python
def compute_rusle(r, k, ls, c, p):
    """Compute RUSLE annual soil loss.

    Args:
        r: ee.Image — Rainfall erosivity factor (MJ·mm/ha/h/yr)
        k: ee.Image — Soil erodibility factor (t·h/MJ·mm)
        ls: ee.Image — Slope length-steepness factor (dimensionless)
        c: ee.Image — Cover management factor (0–1)
        p: ee.Image — Support practice factor (0–1)

    Returns:
        ee.Image: Soil loss in t/ha/yr
    """
    return r.multiply(k).multiply(ls).multiply(c).multiply(p).rename('Soil_Loss')
```

---

## Notebook Conventions

1. **Section numbering:** Always use `## SECTION N — Title` headings for notebook sections
2. **CONFIG dict:** Never hardcode parameters inside cells — add them to `CONFIG`
3. **Print statements:** Every code cell should print a ✓ confirmation when complete
4. **Resolution:** Always pass `scale=SCALE` and `setDefaultProjection(crs=TARGET_CRS, scale=SCALE)` to GEE calls
5. **tileScale:** Set `tileScale=8` (or higher) for all `reduceRegion` calls at 30m
6. **Comments:** GEE operations must have a comment explaining the transformation
7. **No credentials:** Never commit GEE project IDs — they should be replaced with `'your-project-id'`

---

## Testing

Tests live in `tests/`. They are **offline** (no GEE connection required) and use the sample CSV in `data/sample/`.

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_rusle_factors.py -v

# With coverage
pytest tests/ --cov=scripts --cov-report=html
```

**Adding a test:** Create `tests/test_your_feature.py` using the `unittest` or `pytest` style:

```python
import pytest
import numpy as np

def test_k_factor_range():
    """K-factor values should be in physical range 0.01–0.08."""
    from scripts.rusle_utils import compute_k_factor_epic
    k = compute_k_factor_epic(clay=25, sand=40, silt=35, soc=12)
    assert 0.01 <= k <= 0.08, f"K={k} out of physical range"
```

---

## Documentation

- Update `docs/METHODOLOGY.md` if you change any RUSLE equation
- Update `docs/DATA_SOURCES.md` if you add a new dataset
- Update `docs/RESULTS.md` with new district summaries if applicable
- Update `CHANGELOG.md` under `[Unreleased]`

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

| Type | Use for |
|:---:|:---|
| `feat` | New feature or dataset |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructure (no logic change) |
| `test` | Adding or updating tests |
| `chore` | Build, CI, dependency updates |
| `perf` | Performance improvement |

**Examples:**
```
feat(ml): add LightGBM regressor with hyperparameter search
fix(rusle): correct K-factor EPIC equation divisor for SOC
docs(methodology): clarify MFI calculation for GPM IMERG
test(rusle): add unit tests for P-factor slope class assignment
```

---

*Questions? Open a [Discussion](https://github.com/YOUR_GITHUB_USERNAME/western-maharashtra-erosion/discussions) — we'd love to help.*
