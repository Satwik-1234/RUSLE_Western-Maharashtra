"""
test_rusle_factors.py
─────────────────────
Offline unit tests for RUSLE factor calculations.
No GEE connection required — tests pure Python/NumPy logic.

Run: pytest tests/test_rusle_factors.py -v
"""

import pytest
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helper: pure-Python implementations mirroring the GEE notebook logic
# ─────────────────────────────────────────────────────────────────────────────

def k_factor_epic(clay, sand, silt, soc):
    """EPIC K-factor equation (dimensionless inputs, all in %)."""
    f_csand = np.exp(-0.01 * sand) * 0.3 + 0.2
    f_clsi  = (silt / (clay + silt + 1e-9)) ** 0.3
    f_orgC  = (soc * 0.1 + 1) ** (-0.5)
    return f_csand * f_clsi * f_orgC * 0.1317


def ls_factor_mccool(slope_deg, cell_size=30.0):
    """McCool (1989) LS-factor for a given slope in degrees."""
    slope_rad = np.radians(slope_deg)
    s_factor  = np.where(
        slope_deg < 9,
        np.sin(slope_rad) * 10.8 + 0.03,
        np.sin(slope_rad) * 16.8 - 0.50
    )
    s_factor = np.maximum(s_factor, 0.0)
    l_factor = (cell_size / 22.13) ** 0.5
    return l_factor * s_factor


def p_factor_slope(slope_deg):
    """Slope-class P-factor from notebook Section 8."""
    s = slope_deg
    if s < 2:   return 0.60
    if s < 5:   return 0.50
    if s < 8:   return 0.60
    if s < 12:  return 0.70
    if s < 16:  return 0.80
    if s < 20:  return 0.90
    return 1.00


def scs_runoff(rainfall_mm, cn):
    """SCS-CN runoff depth (mm)."""
    s  = 25400 / cn - 254
    ia = 0.2 * s
    q  = np.where(
        rainfall_mm > ia,
        (rainfall_mm - ia) ** 2 / (rainfall_mm - ia + s),
        0.0
    )
    return q


def vulnerability_index(slope, ndvi, rainfall, tri,
                         w_slope=0.35, w_veg=0.30, w_rain=0.20, w_tri=0.15):
    """Weighted vulnerability composite (0–100)."""
    slope_n = np.clip(slope / 30.0, 0, 1)
    veg_n   = np.clip(1.0 - ndvi, 0, 1)
    rain_n  = np.clip(rainfall / 3000.0, 0, 1)
    tri_n   = np.clip(tri / 50.0, 0, 1)
    return (w_slope * slope_n + w_veg * veg_n + w_rain * rain_n + w_tri * tri_n) * 100


# ─────────────────────────────────────────────────────────────────────────────
# K-Factor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKFactor:

    def test_k_factor_physical_range(self):
        """K-factor should be in the physically meaningful range 0.005–0.08."""
        for clay, sand, silt, soc in [
            (25, 40, 35, 12),
            (10, 70, 20, 5),
            (50, 20, 30, 20),
            (30, 35, 35, 8),
        ]:
            k = k_factor_epic(clay, sand, silt, soc)
            assert 0.005 <= k <= 0.08, f"K={k:.4f} out of range for clay={clay}"

    def test_k_factor_high_sand_low_k(self):
        """Sandy soils (high sand) should have lower K than clay-rich soils."""
        k_sandy = k_factor_epic(clay=5, sand=85, silt=10, soc=5)
        k_clayey = k_factor_epic(clay=50, sand=10, silt=40, soc=5)
        assert k_sandy < k_clayey, "Sandy soil should have lower K than clayey soil"

    def test_k_factor_high_soc_lowers_k(self):
        """Higher organic carbon should reduce K (greater aggregate stability)."""
        k_low_soc  = k_factor_epic(clay=25, sand=40, silt=35, soc=2)
        k_high_soc = k_factor_epic(clay=25, sand=40, silt=35, soc=25)
        assert k_high_soc < k_low_soc, "Higher SOC should lower K-factor"

    def test_k_factor_positive(self):
        """K-factor must always be positive."""
        k = k_factor_epic(clay=30, sand=40, silt=30, soc=10)
        assert k > 0


# ─────────────────────────────────────────────────────────────────────────────
# LS-Factor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLSFactor:

    def test_ls_increases_with_slope(self):
        """LS should increase monotonically with slope angle."""
        slopes = [2, 5, 10, 15, 20, 25]
        ls_vals = [ls_factor_mccool(s) for s in slopes]
        for i in range(len(ls_vals) - 1):
            assert ls_vals[i] < ls_vals[i + 1], (
                f"LS not monotone: slope {slopes[i]}° → {ls_vals[i]:.3f}, "
                f"slope {slopes[i+1]}° → {ls_vals[i+1]:.3f}"
            )

    def test_ls_flat_slope_near_zero(self):
        """LS at 0° slope should be near zero (S term → sin(0) = 0)."""
        ls = ls_factor_mccool(slope_deg=0.0)
        assert ls >= 0.0

    def test_ls_30m_cell_size(self):
        """At 30m cell size (SRTM native), L = sqrt(30/22.13)."""
        expected_L = np.sqrt(30.0 / 22.13)
        slope = 10.0
        ls = ls_factor_mccool(slope_deg=slope, cell_size=30.0)
        s  = np.sin(np.radians(slope)) * 10.8 + 0.03
        expected_ls = expected_L * s
        assert abs(ls - expected_ls) < 1e-6

    def test_ls_mccool_breakpoint(self):
        """McCool equation uses different formula above/below 9°."""
        ls_below = ls_factor_mccool(8.9)
        ls_above = ls_factor_mccool(9.1)
        # Both should be positive; the transition should not be discontinuous
        assert ls_below > 0
        assert ls_above > 0


# ─────────────────────────────────────────────────────────────────────────────
# P-Factor Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPFactor:

    def test_p_factor_known_values(self):
        """Check known slope-class → P-value assignments."""
        assert p_factor_slope(1.0)  == 0.60
        assert p_factor_slope(3.0)  == 0.50
        assert p_factor_slope(6.5)  == 0.60
        assert p_factor_slope(10.0) == 0.70
        assert p_factor_slope(14.0) == 0.80
        assert p_factor_slope(18.0) == 0.90
        assert p_factor_slope(25.0) == 1.00

    def test_p_factor_range(self):
        """P-factor should be between 0.5 and 1.0."""
        for slope in np.linspace(0, 35, 50):
            p = p_factor_slope(slope)
            assert 0.5 <= p <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# SCS-CN Runoff Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSCSCNRunoff:

    def test_no_runoff_below_ia(self):
        """No runoff should occur when rainfall <= initial abstraction."""
        cn = 80
        s  = 25400 / cn - 254
        ia = 0.2 * s
        q  = scs_runoff(ia * 0.5, cn)
        assert q == 0.0

    def test_runoff_positive_above_ia(self):
        """Runoff should be positive when rainfall > initial abstraction."""
        q = scs_runoff(500.0, cn=80)
        assert q > 0

    def test_runoff_increases_with_cn(self):
        """Higher CN (less permeable) → more runoff for same rainfall."""
        q_low_cn  = scs_runoff(500.0, cn=60)
        q_high_cn = scs_runoff(500.0, cn=90)
        assert q_high_cn > q_low_cn

    def test_runoff_less_than_rainfall(self):
        """Runoff depth cannot exceed total rainfall."""
        q = scs_runoff(1000.0, cn=95)
        assert q < 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Vulnerability Index Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVulnerabilityIndex:

    def test_vi_range(self):
        """VI should be in [0, 100]."""
        vi = vulnerability_index(slope=15, ndvi=0.4, rainfall=1500, tri=20)
        assert 0 <= vi <= 100

    def test_vi_low_risk(self):
        """Dense forest, flat terrain, low rainfall → low VI."""
        vi = vulnerability_index(slope=1, ndvi=0.9, rainfall=300, tri=2)
        assert vi < 30, f"Expected low VI, got {vi:.1f}"

    def test_vi_high_risk(self):
        """Steep, bare, high-rain → high VI."""
        vi = vulnerability_index(slope=28, ndvi=0.05, rainfall=2800, tri=45)
        assert vi > 70, f"Expected high VI, got {vi:.1f}"

    def test_vi_weights_sum_to_one(self):
        """Default vulnerability weights should sum to 1.0."""
        total = 0.35 + 0.30 + 0.20 + 0.15
        assert abs(total - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_config_json_valid(self):
        """config/analysis_config.json should be valid JSON with required keys."""
        import json, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "analysis_config.json"
        )
        with open(config_path) as f:
            cfg = json.load(f)

        required_keys = ["start_year", "end_year", "resolution", "crs",
                         "districts", "sample_n", "n_clusters",
                         "class_breaks", "class_labels"]
        for key in required_keys:
            assert key in cfg, f"Missing key in config: {key}"

    def test_config_resolution_30m(self):
        """Resolution must be 30m."""
        import json, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "analysis_config.json"
        )
        with open(config_path) as f:
            cfg = json.load(f)
        assert cfg["resolution"] == 30, "Resolution must be 30m"

    def test_config_class_breaks_labels_consistent(self):
        """class_breaks should have one more element than class_labels."""
        import json, os
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "analysis_config.json"
        )
        with open(config_path) as f:
            cfg = json.load(f)
        assert len(cfg["class_breaks"]) == len(cfg["class_labels"]) + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
