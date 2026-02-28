#!/usr/bin/env python3
"""
validate_gee_assets.py
─────────────────────
Pre-flight check: verifies that all Google Earth Engine datasets used by the
Western Maharashtra Erosion notebook are accessible before a full run.

Usage:
    python scripts/validate_gee_assets.py --project your-gee-project-id
"""

import argparse
import sys

REQUIRED_ASSETS = {
    "GPM IMERG V06":          "NASA/GPM_L3/IMERG_V06",
    "SRTM 30m":               "USGS/SRTMGL1_003",
    "SoilGrids Clay":         "projects/soilgrids-isric/clay_mean",
    "SoilGrids Sand":         "projects/soilgrids-isric/sand_mean",
    "SoilGrids Silt":         "projects/soilgrids-isric/silt_mean",
    "SoilGrids SOC":          "projects/soilgrids-isric/soc_mean",
    "ESA WorldCover v200":    "ESA/WorldCover/v200",
    "Sentinel-2 SR Harm.":    "COPERNICUS/S2_SR_HARMONIZED",
    "FAO GAUL Level-2":       "FAO/GAUL/2015/level2",
}


def check_assets(project_id: str) -> bool:
    """Check all required GEE assets and print a status table."""
    try:
        import ee
    except ImportError:
        print("ERROR: earthengine-api not installed. Run: pip install earthengine-api")
        return False

    print("Authenticating with GEE...")
    try:
        ee.Initialize(project=project_id)
        print(f"  ✓ Authenticated — project: {project_id}\n")
    except Exception as e:
        print(f"  ✗ GEE authentication failed: {e}")
        print("  Run: earthengine authenticate")
        return False

    all_ok = True
    col_w = max(len(k) for k in REQUIRED_ASSETS.keys()) + 2

    print(f"{'Asset':<{col_w}} {'Status':<10} {'Notes'}")
    print("─" * 72)

    for name, asset_id in REQUIRED_ASSETS.items():
        try:
            # Try to get metadata — fails if no access
            if "ImageCollection" in asset_id or any(
                x in asset_id for x in ["GPM", "S2", "WorldCover"]
            ):
                obj = ee.ImageCollection(asset_id).limit(1).getInfo()
                count = len(obj.get("features", [obj]))
                status = "✓ OK"
                note = f"ImageCollection ({count} feature sampled)"
            elif "GAUL" in asset_id:
                obj = ee.FeatureCollection(asset_id).limit(1).getInfo()
                status = "✓ OK"
                note = "FeatureCollection"
            else:
                obj = ee.Image(asset_id).bandNames().getInfo()
                status = "✓ OK"
                note = f"Image — bands: {', '.join(obj[:3])}"

        except Exception as e:
            status = "✗ FAIL"
            note = str(e)[:60]
            all_ok = False

        color = "\033[92m" if "✓" in status else "\033[91m"
        reset = "\033[0m"
        print(f"{name:<{col_w}} {color}{status:<10}{reset} {note}")

    print("─" * 72)
    if all_ok:
        print("\n✅ All GEE assets accessible — safe to run the full notebook.\n")
    else:
        print("\n⚠️  Some assets failed. Check your GEE permissions and project quota.\n")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Validate GEE asset accessibility for the erosion analysis notebook."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Your Google Earth Engine project ID",
    )
    args = parser.parse_args()

    success = check_assets(args.project)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
