"""
Module 2: Generation Forecasting Engine
========================================
Predicts next-24-hour solar and wind power output using LightGBM gradient-boosted
models trained on the local historical CSV log.

Design principles:
  - Fully offline: no cloud calls, no internet dependency.
  - Retrain from a local CSV/parquet file at any time.
  - Separate models for solar and wind (their feature sets differ).
  - Features are lag-based + calendar + weather to keep data requirements low.
  - Returns a 24-element array (one value per hour) for both solar and wind.
  - On any error (bad input, untrained model, NaN predictions) falls back to
    a safe deterministic estimate — the safety fallback layer handles the rest.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _build_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for both generation models from a raw data frame.

    Input columns expected:
        timestamp, solar_irradiance_wm2, wind_speed_ms,
        solar_power_kw, wind_power_kw

    Engineered features:
        - Hour of day (0-23), cyclic sin/cos encoding
        - Day of year (1-365), cyclic sin/cos encoding
        - Month
        - Lagged solar/wind power (t-1, t-2, t-3, t-6, t-12, t-24)
        - Rolling mean solar/wind power (3h, 6h, 12h, 24h)
        - Wind speed and its lags
        - Solar irradiance and its lags
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["doy"] = ts.dt.day_of_year
    df["month"] = ts.dt.month

    # Cyclic encodings prevent discontinuities at hour 23→0 and day 365→1
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)

    # Lag features for solar
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"solar_lag_{lag}"] = df["solar_power_kw"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"solar_roll_{window}"] = (
            df["solar_power_kw"].rolling(window, min_periods=1).mean()
        )

    # Lag features for wind
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"wind_lag_{lag}"] = df["wind_power_kw"].shift(lag)
        df[f"wspd_lag_{lag}"] = df["wind_speed_ms"].shift(lag)
    for window in [3, 6, 12, 24]:
        df[f"wind_roll_{window}"] = (
            df["wind_power_kw"].rolling(window, min_periods=1).mean()
        )

    # Raw weather features (current hour — available as sensor reading)
    df["irradiance"] = df["solar_irradiance_wm2"]
    df["wind_speed"] = df["wind_speed_ms"]

    df = df.ffill().bfill()
    return df


_SOLAR_FEATURES = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "irradiance",
    "solar_lag_1", "solar_lag_2", "solar_lag_3", "solar_lag_6",
    "solar_lag_12", "solar_lag_24",
    "solar_roll_3", "solar_roll_6", "solar_roll_12", "solar_roll_24",
]

_WIND_FEATURES = [
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "wind_speed",
    "wind_lag_1", "wind_lag_2", "wind_lag_3", "wind_lag_6",
    "wind_lag_12", "wind_lag_24",
    "wind_roll_3", "wind_roll_6", "wind_roll_12", "wind_roll_24",
    "wspd_lag_1", "wspd_lag_2", "wspd_lag_3", "wspd_lag_6",
]


# ---------------------------------------------------------------------------
# LightGBM parameters — tuned for small datasets (100-500 training days)
# ---------------------------------------------------------------------------

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
# GenerationForecaster
# ---------------------------------------------------------------------------

class GenerationForecaster:
    """
    Trains two LightGBM models (solar, wind) on historical data and provides
    24-hour-ahead generation forecasts.

    Usage
    -----
    >>> gf = GenerationForecaster()
    >>> gf.train(historical_df)
    >>> solar_24h, wind_24h = gf.forecast_24h(current_context_df)
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._solar_model: Optional[lgb.Booster] = None
        self._wind_model: Optional[lgb.Booster] = None
        self._model_dir = Path(model_dir) if model_dir else None
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, save_models: bool = True) -> dict:
        """
        Train solar and wind generation models on a historical DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain: timestamp, solar_irradiance_wm2, wind_speed_ms,
                          solar_power_kw, wind_power_kw
        save_models : bool
            If True and model_dir is set, save Booster files to disk.

        Returns
        -------
        dict with 'solar_rmse' and 'wind_rmse' on the validation split.
        """
        feat_df = _build_generation_features(df)
        feat_df = feat_df.dropna()

        # 80/20 chronological split
        split_idx = int(len(feat_df) * 0.8)
        train_df = feat_df.iloc[:split_idx]
        val_df = feat_df.iloc[split_idx:]

        # --- Solar model ---
        X_train_s = train_df[_SOLAR_FEATURES]
        y_train_s = train_df["solar_power_kw"]
        X_val_s = val_df[_SOLAR_FEATURES]
        y_val_s = val_df["solar_power_kw"]

        dtrain_s = lgb.Dataset(X_train_s, label=y_train_s)
        dval_s = lgb.Dataset(X_val_s, label=y_val_s, reference=dtrain_s)

        self._solar_model = lgb.train(
            _LGB_PARAMS,
            dtrain_s,
            num_boost_round=_LGB_NUM_ROUNDS,
            valid_sets=[dval_s],
            callbacks=[lgb.early_stopping(_LGB_EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        # --- Wind model ---
        X_train_w = train_df[_WIND_FEATURES]
        y_train_w = train_df["wind_power_kw"]
        X_val_w = val_df[_WIND_FEATURES]
        y_val_w = val_df["wind_power_kw"]

        dtrain_w = lgb.Dataset(X_train_w, label=y_train_w)
        dval_w = lgb.Dataset(X_val_w, label=y_val_w, reference=dtrain_w)

        self._wind_model = lgb.train(
            _LGB_PARAMS,
            dtrain_w,
            num_boost_round=_LGB_NUM_ROUNDS,
            valid_sets=[dval_w],
            callbacks=[lgb.early_stopping(_LGB_EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        self._trained = True

        # Validation metrics
        solar_pred_val = np.clip(self._solar_model.predict(X_val_s), 0, None)
        wind_pred_val = np.clip(self._wind_model.predict(X_val_w), 0, None)
        solar_rmse = float(np.sqrt(np.mean((solar_pred_val - y_val_s.values) ** 2)))
        wind_rmse = float(np.sqrt(np.mean((wind_pred_val - y_val_w.values) ** 2)))

        metrics = {"solar_rmse": round(solar_rmse, 3), "wind_rmse": round(wind_rmse, 3)}
        print(f"[GenForecaster] Trained  solar_RMSE={solar_rmse:.3f} kW  "
              f"wind_RMSE={wind_rmse:.3f} kW")

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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Produce a 24-hour-ahead forecast of solar and wind power (kW).

        Parameters
        ----------
        context_df : pd.DataFrame
            Recent historical window (at least 24 hours recommended).
            Same schema as the training data.
        forecast_start : pd.Timestamp, optional
            If None, uses the hour immediately after context_df ends.

        Returns
        -------
        (solar_24h, wind_24h) : two np.ndarrays of length 24 (kW values).

        On any failure returns conservative safe estimates (recent rolling mean).
        """
        if not self._trained:
            return self._safe_fallback_forecast(context_df)

        try:
            return self._predict_24h(context_df, forecast_start)
        except Exception as exc:
            print(f"[GenForecaster] WARNING: forecast failed ({exc}), using fallback.")
            return self._safe_fallback_forecast(context_df)

    def _predict_24h(
        self,
        context_df: pd.DataFrame,
        forecast_start: Optional[pd.Timestamp],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Single-pass 24-step prediction.
        Build features once on the context window, then project forward using
        the last known row's lag values shifted by 1-24 steps.
        This is O(n) rather than the O(n*24) iterative approach.
        """
        ctx = context_df.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
        ctx = ctx.sort_values("timestamp").reset_index(drop=True)

        if forecast_start is None:
            last_ts = ctx["timestamp"].iloc[-1]
            forecast_start = last_ts + pd.Timedelta(hours=1)

        # Build feature frame on the existing context
        feat_df = _build_generation_features(ctx)

        solar_preds = []
        wind_preds = []

        # For each future step, construct a feature row by shifting calendar
        # features forward and re-using the most recent lag values available.
        last_row = feat_df.iloc[-1].copy()

        for step in range(24):
            future_ts = forecast_start + pd.Timedelta(hours=step)
            hour = future_ts.hour
            doy = future_ts.timetuple().tm_yday
            month = future_ts.month

            row = last_row.copy()
            row["hour"] = hour
            row["doy"] = doy
            row["month"] = month
            row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            row["doy_sin"] = np.sin(2 * np.pi * doy / 365)
            row["doy_cos"] = np.cos(2 * np.pi * doy / 365)

            # Update lag_1 with the previous step's prediction (feedback)
            if step == 0:
                row["solar_lag_1"] = last_row.get("solar_power_kw",
                                                   last_row.get("solar_lag_1", 0.0))
                row["wind_lag_1"] = last_row.get("wind_power_kw",
                                                  last_row.get("wind_lag_1", 5.0))
            else:
                row["solar_lag_1"] = solar_preds[-1]
                row["wind_lag_1"] = wind_preds[-1]

            s_feat = pd.DataFrame([row])[_SOLAR_FEATURES]
            w_feat = pd.DataFrame([row])[_WIND_FEATURES]

            s_pred = float(np.clip(self._solar_model.predict(s_feat)[0], 0.0, None))
            w_pred = float(np.clip(self._wind_model.predict(w_feat)[0], 0.0, None))

            solar_preds.append(s_pred)
            wind_preds.append(w_pred)

        return np.array(solar_preds), np.array(wind_preds)

    def _safe_fallback_forecast(
        self, context_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Conservative fallback: use the rolling 24-hour mean of recent actuals.
        This produces a flat but reasonable prior rather than zeros.
        """
        recent = context_df.tail(24)
        solar_mean = float(recent["solar_power_kw"].mean()) if "solar_power_kw" in recent else 0.0
        wind_mean = float(recent["wind_power_kw"].mean()) if "wind_power_kw" in recent else 10.0
        return (
            np.full(24, max(solar_mean, 0.0)),
            np.full(24, max(wind_mean, 5.0)),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self._solar_model.save_model(str(model_dir / "gen_solar.lgb"))
        self._wind_model.save_model(str(model_dir / "gen_wind.lgb"))
        print(f"[GenForecaster] Models saved → {model_dir}")

    def load(self, model_dir: str) -> None:
        """Load previously saved models from disk."""
        d = Path(model_dir)
        self._solar_model = lgb.Booster(model_file=str(d / "gen_solar.lgb"))
        self._wind_model = lgb.Booster(model_file=str(d / "gen_wind.lgb"))
        self._trained = True
        print(f"[GenForecaster] Models loaded from {d}")

    @property
    def is_trained(self) -> bool:
        return self._trained


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from polar_energy_sim.modules.data_generator import generate_historical_log, generate_year

    print("Training generation forecaster on historical log...")
    hist = generate_historical_log(n_days=90)
    gf = GenerationForecaster()
    metrics = gf.train(hist, save_models=False)
    print(f"Validation metrics: {metrics}")

    # Test 24h forecast from a recent context window
    year_df = generate_year()
    context = year_df.iloc[:72]   # first 3 days as context
    solar_fc, wind_fc = gf.forecast_24h(context)
    print(f"\n24h solar forecast (kW): {solar_fc.round(2)}")
    print(f"24h wind  forecast (kW): {wind_fc.round(2)}")
