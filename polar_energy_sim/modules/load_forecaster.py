"""
Module 3: Load Forecasting Engine
===================================
Predicts next-24-hour station demand broken down by load tier using a
separate LightGBM gradient-boosted model.

This is intentionally distinct from the generation forecaster (Module 2):
  - Different feature set (consumption history, time patterns, occupancy proxy)
  - Different target variables (tier1_kw, tier2_kw, tier3_kw, total_kw)
  - The MPC controller (Module 4) needs BOTH forecasts independently

Design principles:
  - Offline, local CSV/parquet retraining only.
  - Three sub-models: one per tier. Keeps each model simple and interpretable.
  - Robust to sparse training data (regularised LightGBM, conservative lags).
  - Falls back to a deterministic schedule if the model fails.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build load-forecasting features from historical consumption data.

    Expected input columns:
        timestamp, tier1_load_kw, tier2_load_kw, tier3_load_kw, total_load_kw

    Engineered features:
        - Hour-of-day (cyclic), day-of-week, day-of-year (cyclic), month
        - Occupancy proxy: weekday vs weekend (Antarctica has some routine)
        - Lagged tier loads: t-1, t-2, t-3, t-6, t-12, t-24, t-48
        - Rolling means of each tier: 3h, 6h, 24h
        - Is-winter flag (DOY 90-270 → polar night)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek          # 0=Monday
    df["doy"] = ts.dt.day_of_year
    df["month"] = ts.dt.month

    # Cyclic encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Polar winter flag
    df["is_winter"] = ((df["doy"] >= 90) & (df["doy"] <= 270)).astype(float)

    # Lagged features for each tier
    for tier in ["tier1_load_kw", "tier2_load_kw", "tier3_load_kw", "total_load_kw"]:
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            df[f"{tier}_lag{lag}"] = df[tier].shift(lag)
        for window in [3, 6, 24]:
            df[f"{tier}_roll{window}"] = df[tier].rolling(window, min_periods=1).mean()

    df = df.ffill().bfill()
    return df


def _tier_features(tier: str) -> list[str]:
    """Return the feature list for a given tier model."""
    base = [
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "dow_sin", "dow_cos", "month", "is_winter",
    ]
    tier_lags = [f"{tier}_lag{l}" for l in [1, 2, 3, 6, 12, 24, 48]]
    tier_rolls = [f"{tier}_roll{w}" for w in [3, 6, 24]]
    # Cross-tier context (total load lags help predict each tier)
    total_lags = [f"total_load_kw_lag{l}" for l in [1, 6, 24]]
    return base + tier_lags + tier_rolls + total_lags


_TIERS = ["tier1_load_kw", "tier2_load_kw", "tier3_load_kw"]

# Deterministic fallback schedule (kW) by hour of day — used when model fails
# Shape: (24,) for each tier
_FALLBACK_TIER1 = np.full(24, 20.0)   # heating always on; flat safe estimate
_FALLBACK_TIER2 = np.where(
    np.arange(24) < 8, 2.0,
    np.where(np.arange(24) < 18, 8.0, 2.0)   # working hours
)
_FALLBACK_TIER3 = np.array([
    1.5, 1.0, 1.0, 1.0, 1.0, 1.5,    # midnight-05
    2.5, 4.0, 3.0, 2.5, 2.5, 2.5,    # 06-11
    4.5, 3.0, 2.5, 2.5, 2.5, 2.5,    # 12-17
    2.5, 4.5, 3.0, 2.5, 2.0, 1.5,    # 18-23
])

_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 10,
    "verbose": -1,
    "n_jobs": 1,
}
_LGB_NUM_ROUNDS = 400
_LGB_EARLY_STOP = 40


# ---------------------------------------------------------------------------
# LoadForecaster
# ---------------------------------------------------------------------------

class LoadForecaster:
    """
    Trains three independent LightGBM models (one per load tier) and produces
    24-hour-ahead per-tier demand forecasts.

    Usage
    -----
    >>> lf = LoadForecaster()
    >>> lf.train(historical_df)
    >>> t1, t2, t3 = lf.forecast_24h(context_df)
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._models: dict[str, lgb.Booster] = {}
        self._model_dir = Path(model_dir) if model_dir else None
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, save_models: bool = True) -> dict:
        """
        Train per-tier load models on a historical DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: timestamp, tier1_load_kw, tier2_load_kw,
                          tier3_load_kw, total_load_kw
        Returns
        -------
        dict mapping tier name → validation RMSE.
        """
        feat_df = _build_load_features(df)
        feat_df = feat_df.dropna()

        split_idx = int(len(feat_df) * 0.8)
        train_df = feat_df.iloc[:split_idx]
        val_df = feat_df.iloc[split_idx:]

        metrics = {}
        for tier in _TIERS:
            feats = _tier_features(tier)
            X_tr = train_df[feats]
            y_tr = train_df[tier]
            X_val = val_df[feats]
            y_val = val_df[tier]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                _LGB_PARAMS,
                dtrain,
                num_boost_round=_LGB_NUM_ROUNDS,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(_LGB_EARLY_STOP, verbose=False),
                            lgb.log_evaluation(-1)],
            )
            self._models[tier] = model

            preds = np.clip(model.predict(X_val), 0, None)
            rmse = float(np.sqrt(np.mean((preds - y_val.values) ** 2)))
            metrics[tier] = round(rmse, 3)
            print(f"[LoadForecaster] {tier}  val_RMSE={rmse:.3f} kW")

        self._trained = True

        if save_models and self._model_dir:
            self._save(self._model_dir)

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forecast_24h(
        self,
        context_df: pd.DataFrame,
        forecast_start: Optional[pd.Timestamp] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce a 24-hour-ahead forecast for each load tier (kW).

        Parameters
        ----------
        context_df : pd.DataFrame
            Recent history window (≥48 h recommended for lag features).
        forecast_start : pd.Timestamp, optional
            Defaults to the hour after the last row of context_df.

        Returns
        -------
        (tier1_24h, tier2_24h, tier3_24h) — each a length-24 numpy array (kW).

        Falls back to a deterministic hourly schedule if the model fails.
        """
        if not self._trained:
            return self._safe_fallback(context_df, forecast_start)

        try:
            return self._predict_24h(context_df, forecast_start)
        except Exception as exc:
            print(f"[LoadForecaster] WARNING: forecast failed ({exc}), using fallback.")
            return self._safe_fallback(context_df, forecast_start)

    def _predict_24h(
        self,
        context_df: pd.DataFrame,
        forecast_start: Optional[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Single-pass 24-step prediction.
        Build features once on context, then project forward using last-known
        lag values and stepped calendar features.
        """
        ctx = context_df.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
        ctx = ctx.sort_values("timestamp").reset_index(drop=True)

        if forecast_start is None:
            last_ts = ctx["timestamp"].iloc[-1]
            forecast_start = last_ts + pd.Timedelta(hours=1)

        feat_df = _build_load_features(ctx)
        last_row = feat_df.iloc[-1].copy()

        tier1_preds, tier2_preds, tier3_preds = [], [], []

        for step in range(24):
            future_ts = forecast_start + pd.Timedelta(hours=step)
            hour = future_ts.hour
            doy = future_ts.timetuple().tm_yday
            month = future_ts.month
            dow = future_ts.weekday()

            row = last_row.copy()
            row["hour"] = hour
            row["doy"] = doy
            row["month"] = month
            row["dow"] = dow
            row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            row["doy_sin"] = np.sin(2 * np.pi * doy / 365)
            row["doy_cos"] = np.cos(2 * np.pi * doy / 365)
            row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            row["is_winter"] = float(90 <= doy <= 270)

            # Feed back previous step prediction into lag_1
            if step > 0:
                row["tier1_load_kw_lag1"] = tier1_preds[-1]
                row["tier2_load_kw_lag1"] = tier2_preds[-1]
                row["tier3_load_kw_lag1"] = tier3_preds[-1]
                row["total_load_kw_lag1"] = tier1_preds[-1] + tier2_preds[-1] + tier3_preds[-1]

            preds = {}
            for tier in _TIERS:
                feats = _tier_features(tier)
                p = float(np.clip(
                    self._models[tier].predict(pd.DataFrame([row])[feats])[0],
                    0.0, None
                ))
                preds[tier] = p

            tier1_preds.append(preds["tier1_load_kw"])
            tier2_preds.append(preds["tier2_load_kw"])
            tier3_preds.append(preds["tier3_load_kw"])

        return np.array(tier1_preds), np.array(tier2_preds), np.array(tier3_preds)

    def _safe_fallback(
        self,
        context_df: pd.DataFrame,
        forecast_start: Optional[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic hourly schedule based on hour-of-day profile.
        Uses actual recent mean to scale the profile if context is available.
        """
        if forecast_start is None and context_df is not None and len(context_df):
            last_ts = pd.to_datetime(context_df["timestamp"].iloc[-1])
            start_hour = (last_ts + pd.Timedelta(hours=1)).hour
        else:
            start_hour = 0

        hours = np.arange(start_hour, start_hour + 24) % 24

        # Scale to recent actuals if available
        if context_df is not None and len(context_df) >= 24:
            t1_scale = context_df["tier1_load_kw"].tail(24).mean() / _FALLBACK_TIER1.mean()
            t2_scale = context_df["tier2_load_kw"].tail(24).mean() / max(_FALLBACK_TIER2.mean(), 1.0)
            t3_scale = context_df["tier3_load_kw"].tail(24).mean() / max(_FALLBACK_TIER3.mean(), 1.0)
        else:
            t1_scale = t2_scale = t3_scale = 1.0

        return (
            _FALLBACK_TIER1[hours] * t1_scale,
            _FALLBACK_TIER2[hours] * t2_scale,
            _FALLBACK_TIER3[hours] * t3_scale,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for tier, model in self._models.items():
            model.save_model(str(model_dir / f"load_{tier}.lgb"))
        print(f"[LoadForecaster] Models saved → {model_dir}")

    def load(self, model_dir: str) -> None:
        d = Path(model_dir)
        for tier in _TIERS:
            self._models[tier] = lgb.Booster(
                model_file=str(d / f"load_{tier}.lgb")
            )
        self._trained = True
        print(f"[LoadForecaster] Models loaded from {d}")

    @property
    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from polar_energy_sim.modules.data_generator import generate_historical_log, generate_year

    print("Training load forecaster on historical log...")
    hist = generate_historical_log(n_days=90)
    lf = LoadForecaster()
    metrics = lf.train(hist, save_models=False)
    print(f"Validation metrics: {metrics}")

    year_df = generate_year()
    context = year_df.iloc[:72]
    t1, t2, t3 = lf.forecast_24h(context)
    print(f"\nTier 1 forecast (kW): {t1.round(2)}")
    print(f"Tier 2 forecast (kW): {t2.round(2)}")
    print(f"Tier 3 forecast (kW): {t3.round(2)}")
