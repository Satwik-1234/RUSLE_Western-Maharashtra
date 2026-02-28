"""
test_ml_pipeline.py
───────────────────
Offline smoke tests for the ML pipeline.
Uses synthetic data — no GEE connection required.

Run: pytest tests/test_ml_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

ML_FEATURES = [
    'R_Factor', 'K_Factor', 'LS_Factor', 'C_Factor', 'P_Factor',
    'Slope', 'Elevation', 'TRI', 'Annual_Rainfall', 'Monsoon_Rainfall',
    'NDVI', 'Clay', 'Sand', 'Silt', 'SOC'
]

@pytest.fixture
def synthetic_dataframe():
    """Generate a synthetic DataFrame mirroring the GEE sample output."""
    rng = np.random.default_rng(42)
    n   = 500

    df = pd.DataFrame({
        'R_Factor'        : rng.uniform(200, 1500, n),
        'K_Factor'        : rng.uniform(0.01, 0.06, n),
        'LS_Factor'       : rng.uniform(0.1, 15.0, n),
        'C_Factor'        : rng.uniform(0.001, 0.45, n),
        'P_Factor'        : rng.uniform(0.5, 1.0, n),
        'Slope'           : rng.uniform(0, 30, n),
        'Elevation'       : rng.uniform(400, 1400, n),
        'TRI'             : rng.uniform(0, 50, n),
        'Annual_Rainfall' : rng.uniform(500, 3000, n),
        'Monsoon_Rainfall': rng.uniform(400, 2500, n),
        'NDVI'            : rng.uniform(-0.1, 0.85, n),
        'Clay'            : rng.uniform(10, 55, n),
        'Sand'            : rng.uniform(10, 75, n),
        'Silt'            : rng.uniform(5, 40, n),
        'SOC'             : rng.uniform(2, 30, n),
    })

    # Simulated soil loss: correlated with LS, R, and slope
    df['Soil_Loss'] = (
        df['R_Factor'] * df['K_Factor'] * df['LS_Factor'] *
        df['C_Factor'] * df['P_Factor']
    ).clip(0, 200)

    # Erosion classes
    bins   = [0, 5, 10, 20, 40, 80, 9999]
    labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Severe']
    df['Erosion_Class'] = pd.cut(df['Soil_Loss'], bins=bins, labels=labels, right=False)
    df['Class_Num']     = df['Erosion_Class'].cat.codes + 1

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPreparation:

    def test_feature_columns_present(self, synthetic_dataframe):
        """All expected ML feature columns must be present."""
        df = synthetic_dataframe
        for feat in ML_FEATURES:
            assert feat in df.columns, f"Missing feature: {feat}"

    def test_no_nulls_after_dropna(self, synthetic_dataframe):
        """No NaN values after dropna()."""
        df = synthetic_dataframe[ML_FEATURES + ['Soil_Loss']].dropna()
        assert df.isnull().sum().sum() == 0

    def test_soil_loss_non_negative(self, synthetic_dataframe):
        """Soil loss must be non-negative."""
        assert (synthetic_dataframe['Soil_Loss'] >= 0).all()

    def test_erosion_class_categories(self, synthetic_dataframe):
        """Erosion classes should be a subset of the 6 defined classes."""
        expected = {'Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Severe'}
        actual   = set(synthetic_dataframe['Erosion_Class'].dropna().unique())
        assert actual.issubset(expected), f"Unexpected classes: {actual - expected}"

    def test_cap_at_99th_percentile(self, synthetic_dataframe):
        """99th-percentile cap should reduce max without affecting median."""
        df = synthetic_dataframe
        cap = df['Soil_Loss'].quantile(0.99)
        df_capped = df['Soil_Loss'].clip(0, cap)
        assert df_capped.max() == cap
        assert abs(df_capped.median() - df['Soil_Loss'].median()) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Scaler Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScaler:

    def test_robust_scaler_no_nan(self, synthetic_dataframe):
        """RobustScaler should not produce NaN values."""
        df = synthetic_dataframe[ML_FEATURES].dropna()
        X  = df.values
        Xs = RobustScaler().fit_transform(X)
        assert not np.isnan(Xs).any()

    def test_scaler_fitted_only_on_train(self, synthetic_dataframe):
        """Scaler must be fit on train and only transform test."""
        df  = synthetic_dataframe[ML_FEATURES + ['Soil_Loss']].dropna()
        X   = df[ML_FEATURES].values
        y   = df['Soil_Loss'].values
        X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        sc  = RobustScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        # Median of train should be near zero (centering)
        assert np.abs(np.median(X_tr_s, axis=0)).mean() < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModels:

    @pytest.fixture
    def prepared_data(self, synthetic_dataframe):
        df  = synthetic_dataframe[ML_FEATURES + ['Soil_Loss']].dropna()
        cap = df['Soil_Loss'].quantile(0.99)
        y   = df['Soil_Loss'].clip(0, cap).values
        X   = df[ML_FEATURES].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sc  = RobustScaler()
        return sc.fit_transform(X_tr), sc.transform(X_te), y_tr, y_te

    def test_rf_fits_and_predicts(self, prepared_data):
        """Random Forest should fit without error and predict on test set."""
        X_tr, X_te, y_tr, y_te = prepared_data
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        assert len(y_pred) == len(y_te)
        assert not np.isnan(y_pred).any()

    def test_rf_r2_positive(self, prepared_data):
        """Random Forest R² on synthetic data should be > 0."""
        X_tr, X_te, y_tr, y_te = prepared_data
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(X_tr, y_tr)
        r2 = r2_score(y_te, rf.predict(X_te))
        assert r2 > 0.0, f"RF R²={r2:.3f} is not positive on synthetic data"

    def test_gb_r2_positive(self, prepared_data):
        """Gradient Boosting R² on synthetic data should be > 0."""
        X_tr, X_te, y_tr, y_te = prepared_data
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=3,
                                        learning_rate=0.1, random_state=42)
        gb.fit(X_tr, y_tr)
        r2 = r2_score(y_te, gb.predict(X_te))
        assert r2 > 0.0

    def test_feature_importances_sum_to_one(self, prepared_data):
        """RF feature importances must sum to 1.0."""
        X_tr, _, y_tr, _ = prepared_data
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_tr, y_tr)
        assert abs(rf.feature_importances_.sum() - 1.0) < 1e-6

    def test_predictions_nonnegative(self, prepared_data):
        """Soil loss predictions from RF should be non-negative (trained on clipped y)."""
        X_tr, X_te, y_tr, y_te = prepared_data
        rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
        rf.fit(X_tr, np.clip(y_tr, 0, None))
        y_pred = rf.predict(X_te)
        assert (y_pred >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Clustering Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestClustering:

    def test_kmeans_n_clusters(self, synthetic_dataframe):
        """K-Means should produce exactly n_clusters cluster labels."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler

        df    = synthetic_dataframe[['Soil_Loss', 'LS_Factor', 'Annual_Rainfall',
                                      'NDVI', 'TRI']].dropna()
        Xc    = MinMaxScaler().fit_transform(df)
        km    = KMeans(n_clusters=5, random_state=42, n_init=5)
        labels = km.fit_predict(Xc)
        assert set(labels) == {0, 1, 2, 3, 4}

    def test_kmeans_silhouette_positive(self, synthetic_dataframe):
        """Silhouette score should be > 0 for a reasonable k."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import silhouette_score

        df    = synthetic_dataframe[['Soil_Loss', 'LS_Factor', 'Annual_Rainfall',
                                      'NDVI', 'TRI']].dropna()
        Xc    = MinMaxScaler().fit_transform(df)
        km    = KMeans(n_clusters=5, random_state=42, n_init=5)
        labels = km.fit_predict(Xc)
        score  = silhouette_score(Xc, labels)
        assert score > 0, f"Silhouette={score:.3f} is not positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
