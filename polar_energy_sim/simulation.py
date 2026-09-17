"""
Main Simulation Runner
=======================
Wires all six modules together into a single year-long simulation loop.

Architecture:
  Data Generator  →  Generation Forecaster  ─┐
                                               ├→  MPC Controller  →  Safety Fallback  →  Fuel Tracker
  Data Generator  →  Load Forecaster        ─┘

Each hour of simulated time:
  1. Read actual weather + load from the synthetic dataset.
  2. Build a 24-h forecast from both ML forecasters (using a rolling context window).
  3. Pass state + forecast to the MPC controller via the Safety Fallback Layer.
  4. Record the decision in the Fuel Tracker (MPC + naive baseline in parallel).
  5. Advance battery SOC and move to the next hour.

Returns a SimulationResult dataclass containing the full hour-by-hour record
and summary statistics, plus the FuelTracker ledger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from polar_energy_sim.modules.data_generator import (
    generate_year,
    generate_historical_log,
)
from polar_energy_sim.modules.generation_forecaster import GenerationForecaster
from polar_energy_sim.modules.load_forecaster import LoadForecaster
from polar_energy_sim.modules.mpc_controller import (
    MPCController,
    MPCDecision,
    MPCForecast,
    MPCState,
    NaiveBaselineController,
    ControllerConfig,
    BATTERY_CAPACITY_KWH,
    SOC_MIN,
    SOC_MAX,
)
from polar_energy_sim.modules.safety_fallback import SafetyFallbackLayer
from polar_energy_sim.modules.fuel_tracker import FuelTracker


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """
    Defines all parameters for a single simulation run.
    Used both for the baseline full-year run and for stress-test overrides.
    """
    name: str = "baseline"
    seed: int = 42
    year: int = 2025
    initial_soc: float = 0.70          # Starting battery state of charge

    # Stress test 1: Worst-case winter katabatic wind
    katabatic_stress: bool = False
    katabatic_start_day: int = 150      # Around end of May
    katabatic_duration_days: int = 3

    # Stress test 2: Multi-day wind lull in winter
    wind_lull: bool = False
    lull_start_day: int = 180           # Mid-June (deep polar winter)
    lull_duration_days: int = 5

    # Stress test 3: Sensor/forecast failure injection
    failure_injection: bool = False
    failure_start_hour: int = 4000      # Hour when AI failure begins
    failure_duration_hours: int = 48    # How long the failure lasts

    # Forecast context window (hours of history fed to forecasters)
    forecast_context_hours: int = 72

    # Output directory
    output_dir: str = "polar_energy_sim/outputs"


# ---------------------------------------------------------------------------
# Per-hour record
# ---------------------------------------------------------------------------

@dataclass
class HourRecord:
    """Complete record for one simulation hour."""
    sim_hour: int
    timestamp: object

    # Actuals
    solar_irradiance_wm2: float
    wind_speed_ms: float
    solar_power_kw: float
    wind_power_kw: float
    tier1_load_kw: float
    tier2_load_kw: float
    tier3_load_kw: float
    total_load_kw: float

    # MPC decision
    soc_before: float
    soc_after: float
    gen_output_kw: float
    battery_charge_kw: float       # signed: + = charging
    tier1_served_kw: float
    tier2_served_kw: float
    tier3_served_kw: float
    shed_tier2_kw: float
    shed_tier3_kw: float
    diesel_litres_mpc: float
    surplus_kw: float
    safety_triggered: bool
    safety_reason: str

    # Naive baseline (parallel tracking)
    soc_naive: float
    gen_output_naive_kw: float
    diesel_litres_naive: float


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregated result of a complete simulation run."""
    config: ScenarioConfig
    records: list[HourRecord]
    fuel_tracker: FuelTracker
    safety_summary: dict
    wall_time_s: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the hour-by-hour record list into a DataFrame."""
        return pd.DataFrame([r.__dict__ for r in self.records])

    def save(self, output_dir: Optional[str] = None) -> Path:
        """Save the full record to CSV."""
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"sim_{self.config.name}.csv"
        self.to_dataframe().to_csv(path, index=False)

        fuel_path = out / f"fuel_{self.config.name}.csv"
        self.fuel_tracker.save_csv(str(fuel_path))

        print(f"[Simulation] Results saved → {path}")
        return path

    def print_summary(self) -> None:
        df = self.to_dataframe()
        total_hours = len(df)
        tier1_drop_hours = int((df["tier1_served_kw"] < df["tier1_load_kw"] - 0.5).sum())
        tier2_shed_hours = int((df["shed_tier2_kw"] > 0.1).sum())
        tier3_shed_hours = int((df["shed_tier3_kw"] > 0.1).sum())
        gen_on_hours = int((df["gen_output_kw"] > 0.5).sum())

        print(f"\n{'=' * 62}")
        print(f"  SIMULATION SUMMARY — {self.config.name.upper()}")
        print(f"{'=' * 62}")
        print(f"  Hours simulated       : {total_hours}")
        print(f"  Wall-clock time       : {self.wall_time_s:.1f} s")
        print(f"  Tier 1 drop events    : {tier1_drop_hours}  << MUST be 0")
        print(f"  Tier 2 shed hours     : {tier2_shed_hours}")
        print(f"  Tier 3 shed hours     : {tier3_shed_hours}")
        print(f"  Generator on hours    : {gen_on_hours}")
        print(f"  Safety overrides      : {self.safety_summary.get('total_overrides', 0)}")
        self.fuel_tracker.print_summary()


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def run_simulation(config: ScenarioConfig) -> SimulationResult:
    """
    Execute a full-year polar station energy simulation.

    Parameters
    ----------
    config : ScenarioConfig
        All parameters for this simulation run.

    Returns
    -------
    SimulationResult with hour-by-hour records and summary stats.
    """
    t_start = time.time()
    print(f"\n[Simulation] Starting scenario: '{config.name}'")

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    year_df = generate_year(
        seed=config.seed,
        year=config.year,
        apply_wind_lull=config.wind_lull,
        lull_start_day=config.lull_start_day,
        lull_duration_days=config.lull_duration_days,
        apply_katabatic_stress=config.katabatic_stress,
        katabatic_start_day=config.katabatic_start_day,
        katabatic_duration_days=config.katabatic_duration_days,
    )
    hist_df = generate_historical_log(n_days=90, seed=config.seed + 1000)

    # ------------------------------------------------------------------
    # 2. Train forecasters on historical log
    # ------------------------------------------------------------------
    print("[Simulation] Training generation forecaster...")
    gen_forecaster = GenerationForecaster()
    gen_forecaster.train(hist_df, save_models=False)

    print("[Simulation] Training load forecaster...")
    load_forecaster = LoadForecaster()
    load_forecaster.train(hist_df, save_models=False)

    # ------------------------------------------------------------------
    # 2b. Pre-build feature matrices for the whole year (fast slicing later)
    # ------------------------------------------------------------------
    print("[Simulation] Pre-computing feature matrices...")
    from polar_energy_sim.modules.generation_forecaster import _build_generation_features, _SOLAR_FEATURES, _WIND_FEATURES
    from polar_energy_sim.modules.load_forecaster import _build_load_features, _tier_features, _TIERS
    import lightgbm as lgb

    # Concatenate hist + year so lags at hour 0 are populated
    combined_gen = pd.concat([
        hist_df[["timestamp", "solar_irradiance_wm2", "wind_speed_ms",
                 "solar_power_kw", "wind_power_kw"]],
        year_df[["timestamp", "solar_irradiance_wm2", "wind_speed_ms",
                 "solar_power_kw", "wind_power_kw"]],
    ], ignore_index=True)
    combined_load = pd.concat([
        hist_df[["timestamp", "tier1_load_kw", "tier2_load_kw",
                 "tier3_load_kw", "total_load_kw"]],
        year_df[["timestamp", "tier1_load_kw", "tier2_load_kw",
                 "tier3_load_kw", "total_load_kw"]],
    ], ignore_index=True)

    hist_len = len(hist_df)
    gen_feat_all = _build_generation_features(combined_gen).iloc[hist_len:].reset_index(drop=True)
    load_feat_all = _build_load_features(combined_load).iloc[hist_len:].reset_index(drop=True)

    # Pre-compute all 8760 solar/wind/load single-step predictions at once
    solar_pred_all = np.clip(
        gen_forecaster._solar_model.predict(gen_feat_all[_SOLAR_FEATURES]), 0.0, None
    )
    wind_pred_all = np.clip(
        gen_forecaster._wind_model.predict(gen_feat_all[_WIND_FEATURES]), 0.0, None
    )
    t1_pred_all = np.clip(
        load_forecaster._models["tier1_load_kw"].predict(
            load_feat_all[_tier_features("tier1_load_kw")]), 0.0, None
    )
    t2_pred_all = np.clip(
        load_forecaster._models["tier2_load_kw"].predict(
            load_feat_all[_tier_features("tier2_load_kw")]), 0.0, None
    )
    t3_pred_all = np.clip(
        load_forecaster._models["tier3_load_kw"].predict(
            load_feat_all[_tier_features("tier3_load_kw")]), 0.0, None
    )
    print("[Simulation] Feature pre-computation complete.")

    # ------------------------------------------------------------------
    # 3. Initialise controllers and trackers
    # ------------------------------------------------------------------
    mpc = MPCController(config=ControllerConfig())
    safety = SafetyFallbackLayer(mpc)
    naive = NaiveBaselineController(gen_on_soc_threshold=0.40)
    fuel_tracker = FuelTracker()

    soc_mpc = config.initial_soc
    soc_naive = config.initial_soc

    records: list[HourRecord] = []
    n_hours = len(year_df)

    # ------------------------------------------------------------------
    # 4. Hour-by-hour simulation loop
    # ------------------------------------------------------------------
    print(f"[Simulation] Running {n_hours}-hour loop...")

    # Cache forecasts — recompute every 6 hours (real MPC reuse pattern)
    REFORECAST_INTERVAL = 6
    cached_forecast: Optional[MPCForecast] = None

    for h in range(n_hours):
        row = year_df.iloc[h]
        ts = row["timestamp"]

        # ------------------------------------------------------------------
        # Build 24h forecast from pre-computed predictions (O(1) slice)
        # ------------------------------------------------------------------
        is_failure_hour = (
            config.failure_injection
            and config.failure_start_hour <= h < config.failure_start_hour + config.failure_duration_hours
        )

        should_reforecast = (h % REFORECAST_INTERVAL == 0) or (cached_forecast is None)

        if is_failure_hour:
            forecast = None  # Safety layer must handle this
        elif should_reforecast:
            # Slice the next 24h of pre-computed predictions (pad at year end)
            end = min(h + 24, n_hours)
            pad = 24 - (end - h)
            solar_fc = np.pad(solar_pred_all[h:end], (0, pad), mode="edge")
            wind_fc  = np.pad(wind_pred_all[h:end],  (0, pad), mode="edge")
            t1_fc    = np.pad(t1_pred_all[h:end],    (0, pad), mode="edge")
            t2_fc    = np.pad(t2_pred_all[h:end],    (0, pad), mode="edge")
            t3_fc    = np.pad(t3_pred_all[h:end],    (0, pad), mode="edge")
            cached_forecast = MPCForecast(
                solar_kw=solar_fc, wind_kw=wind_fc,
                tier1_kw=t1_fc, tier2_kw=t2_fc, tier3_kw=t3_fc,
            )
            forecast = cached_forecast
        else:
            forecast = cached_forecast

        # ------------------------------------------------------------------
        # MPC via safety fallback layer
        # ------------------------------------------------------------------
        state_mpc = MPCState(
            soc=soc_mpc,
            solar_power_kw=float(row["solar_power_kw"]),
            wind_power_kw=float(row["wind_power_kw"]),
            tier1_load_kw=float(row["tier1_load_kw"]),
            tier2_load_kw=float(row["tier2_load_kw"]),
            tier3_load_kw=float(row["tier3_load_kw"]),
            timestamp=ts,
        )

        mpc_dec: MPCDecision = safety.step(
            state=state_mpc,
            forecast=forecast,
            sim_hour=h,
            force_failure=is_failure_hour,
        )

        # ------------------------------------------------------------------
        # Naive baseline (parallel, no forecasting, no load shedding logic)
        # ------------------------------------------------------------------
        state_naive = MPCState(
            soc=soc_naive,
            solar_power_kw=float(row["solar_power_kw"]),
            wind_power_kw=float(row["wind_power_kw"]),
            tier1_load_kw=float(row["tier1_load_kw"]),
            tier2_load_kw=float(row["tier2_load_kw"]),
            tier3_load_kw=float(row["tier3_load_kw"]),
            timestamp=ts,
        )
        naive_dec: MPCDecision = naive.step(state_naive)

        # ------------------------------------------------------------------
        # Advance state
        # ------------------------------------------------------------------
        soc_mpc = mpc_dec.soc_after
        soc_naive = naive_dec.soc_after

        # ------------------------------------------------------------------
        # Record
        # ------------------------------------------------------------------
        fuel_tracker.record(
            sim_hour=h,
            timestamp=ts,
            mpc_litres=mpc_dec.diesel_litres,
            naive_litres=naive_dec.diesel_litres,
            mpc_gen_kw=mpc_dec.gen_output_kw,
            naive_gen_kw=naive_dec.gen_output_kw,
            soc_mpc=soc_mpc,
            soc_naive=soc_naive,
            safety_triggered=safety.triggered,
        )

        records.append(HourRecord(
            sim_hour=h,
            timestamp=ts,
            solar_irradiance_wm2=float(row["solar_irradiance_wm2"]),
            wind_speed_ms=float(row["wind_speed_ms"]),
            solar_power_kw=float(row["solar_power_kw"]),
            wind_power_kw=float(row["wind_power_kw"]),
            tier1_load_kw=float(row["tier1_load_kw"]),
            tier2_load_kw=float(row["tier2_load_kw"]),
            tier3_load_kw=float(row["tier3_load_kw"]),
            total_load_kw=float(row["total_load_kw"]),
            soc_before=state_mpc.soc,
            soc_after=soc_mpc,
            gen_output_kw=mpc_dec.gen_output_kw,
            battery_charge_kw=mpc_dec.battery_charge_kw,
            tier1_served_kw=mpc_dec.tier1_served_kw,
            tier2_served_kw=mpc_dec.tier2_served_kw,
            tier3_served_kw=mpc_dec.tier3_served_kw,
            shed_tier2_kw=mpc_dec.shed_tier2_kw,
            shed_tier3_kw=mpc_dec.shed_tier3_kw,
            diesel_litres_mpc=mpc_dec.diesel_litres,
            surplus_kw=mpc_dec.surplus_kw,
            safety_triggered=safety.triggered,
            safety_reason=mpc_dec.reason,
            soc_naive=soc_naive,
            gen_output_naive_kw=naive_dec.gen_output_kw,
            diesel_litres_naive=naive_dec.diesel_litres,
        ))

        # Progress print every 1000 hours
        if h % 1000 == 0 and h > 0:
            print(f"  [h={h:4d}]  SOC={soc_mpc:.3f}  "
                  f"gen={mpc_dec.gen_output_kw:.1f}kW  "
                  f"safety_triggers={safety.trigger_count}")

    wall_time = time.time() - t_start
    result = SimulationResult(
        config=config,
        records=records,
        fuel_tracker=fuel_tracker,
        safety_summary=safety.summary(),
        wall_time_s=wall_time,
    )
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Convenience: run all four stress-test scenarios
# ---------------------------------------------------------------------------

def run_all_scenarios(output_dir: str = "polar_energy_sim/outputs") -> dict[str, SimulationResult]:
    """
    Run the four required stress-test scenarios plus the baseline full-year run.
    Returns a dict mapping scenario name → SimulationResult.
    """
    scenarios = [
        ScenarioConfig(
            name="baseline",
            seed=42,
            output_dir=output_dir,
        ),
        ScenarioConfig(
            name="stress1_katabatic_winter",
            seed=42,
            katabatic_stress=True,
            katabatic_start_day=150,
            katabatic_duration_days=3,
            output_dir=output_dir,
        ),
        ScenarioConfig(
            name="stress2_wind_lull",
            seed=42,
            wind_lull=True,
            lull_start_day=180,
            lull_duration_days=5,
            output_dir=output_dir,
        ),
        ScenarioConfig(
            name="stress3_forecast_failure",
            seed=42,
            failure_injection=True,
            failure_start_hour=4320,   # Mid-June (deep winter, ~day 180)
            failure_duration_hours=48,
            output_dir=output_dir,
        ),
        ScenarioConfig(
            name="stress4_fuel_comparison",
            seed=42,
            output_dir=output_dir,
        ),
    ]

    results = {}
    for cfg in scenarios:
        result = run_simulation(cfg)
        result.save(output_dir)
        results[cfg.name] = result

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_simulation(ScenarioConfig(name="baseline"))
    result.save()
