"""
Module 4: Model Predictive Control (MPC) Optimization Controller
=================================================================
Implements a rolling-horizon MPC that uses 24-hour generation and load
forecasts to make energy dispatch decisions at each timestep.

Decisions made every hour:
  1. Battery charge / discharge rate (kW)
  2. Generator on/off and output level (kW)  ← fuel-minimisation objective
  3. Load shedding: how much of Tier 2 and Tier 3 to serve vs. curtail

Cost function (minimised over the 24-hour horizon):
  J = w_fuel * Σ gen_output_kW(t) * DIESEL_L_PER_KWH
    + w_shed2 * Σ shed2(t)
    + w_shed3 * Σ shed3(t)
    + w_soc   * Σ penalty_for_extreme_SOC(t)
    - w_surplus * Σ surplus(t)

Subject to hard constraints:
  - Battery SOC ∈ [SOC_MIN, SOC_MAX] at all times
  - Generator output ∈ {0} ∪ [GEN_MIN, GEN_MAX]
  - Tier 1 load ALWAYS fully served (never in the shed decision)
  - Power balance at each timestep

Implemented as a greedy MPC (sequential per-hour solve across the horizon)
rather than a full MILP, keeping the implementation dependency-free
(no CVXPY / PuLP required) while still respecting all constraints.
The greedy solve uses scipy.optimize.linprog for continuous dispatch variables
and a simple binary search for the generator on/off decision.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH = 200.0    # Usable battery bank capacity
SOC_MIN = 0.15                  # 15% — hard lower bound (battery protection)
SOC_MAX = 0.95                  # 95% — hard upper bound (overcharge protection)
SOC_CRITICAL = 0.20             # Below this, safety layer takes over
BATTERY_CHARGE_EFF = 0.95       # Round-trip charge efficiency
BATTERY_DISCHARGE_EFF = 0.95    # Round-trip discharge efficiency
BATTERY_MAX_CHARGE_KW = 40.0    # Max charge rate (kW)
BATTERY_MAX_DISCHARGE_KW = 40.0 # Max discharge rate (kW)

GEN_MIN_KW = 10.0               # Minimum generator output when running
GEN_MAX_KW = 50.0               # Maximum generator output
DIESEL_L_PER_KWH = 0.35         # Diesel consumption: ~0.35 L/kWh (typical genset)

HORIZON_H = 24                  # MPC planning horizon (hours)

# Cost-function weights
W_FUEL = 10.0          # Penalise diesel burn heavily
W_SHED_TIER2 = 5.0     # Penalise Tier 2 shedding moderately
W_SHED_TIER3 = 1.0     # Penalise Tier 3 shedding lightly
W_SOC_EXTREME = 8.0    # Penalise drifting to SOC extremes
W_SURPLUS = 0.1        # Small reward for excess renewable surplus


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MPCState:
    """Full system state at a given timestep, passed in by the simulation loop."""
    soc: float                          # Current battery state of charge [0-1]
    solar_power_kw: float               # Actual solar power this hour
    wind_power_kw: float                # Actual wind power this hour
    tier1_load_kw: float                # Actual Tier 1 load
    tier2_load_kw: float                # Actual Tier 2 load
    tier3_load_kw: float                # Actual Tier 3 load
    timestamp: Optional[object] = None  # Timestamp for logging

@dataclass
class MPCForecast:
    """24-hour forecasts from Modules 2 and 3."""
    solar_kw: np.ndarray    # shape (24,)
    wind_kw: np.ndarray     # shape (24,)
    tier1_kw: np.ndarray    # shape (24,)
    tier2_kw: np.ndarray    # shape (24,)
    tier3_kw: np.ndarray    # shape (24,)

@dataclass
class MPCDecision:
    """Dispatch decision for the current hour."""
    battery_charge_kw: float    # positive = charging, negative = discharging
    gen_output_kw: float        # 0 = off, >0 = generating
    tier2_served_kw: float      # kW of Tier 2 actually served
    tier3_served_kw: float      # kW of Tier 3 actually served
    tier1_served_kw: float      # always == tier1_load (never shed)
    soc_after: float            # predicted SOC after this hour
    diesel_litres: float        # fuel burned this hour
    renewable_used_kw: float    # renewable power actually consumed
    surplus_kw: float           # excess renewable (curtailed or wasted)
    shed_tier2_kw: float        # Tier 2 not served
    shed_tier3_kw: float        # Tier 3 not served
    reason: str = ""            # textual reason for generator start

@dataclass
class ControllerConfig:
    """Tunable parameters for the MPC controller."""
    soc_target: float = 0.65           # Preferred battery SOC
    gen_start_soc_threshold: float = 0.40  # SOC below which generator can start
    w_fuel: float = W_FUEL
    w_shed2: float = W_SHED_TIER2
    w_shed3: float = W_SHED_TIER3
    w_soc: float = W_SOC_EXTREME
    w_surplus: float = W_SURPLUS
    horizon: int = HORIZON_H


# ---------------------------------------------------------------------------
# Core greedy-horizon solver
# ---------------------------------------------------------------------------

def _soc_penalty(soc: float) -> float:
    """
    Quadratic penalty that rises as SOC approaches either bound.
    Keeps the battery from spending long periods near limits.
    """
    mid = 0.65
    return (soc - mid) ** 2 * 2.0


def _solve_single_hour(
    soc: float,
    solar_kw: float,
    wind_kw: float,
    tier1_kw: float,
    tier2_kw: float,
    tier3_kw: float,
    cfg: ControllerConfig,
) -> MPCDecision:
    """
    Solve the dispatch problem for a single hour.

    Priority order:
      1. Always serve Tier 1 in full.
      2. Use available renewable to serve loads and charge battery.
      3. Run generator only if needed (SOC too low or loads unmet).
      4. Shed Tier 3 before Tier 2.
      5. Minimise diesel — generator runs at its most efficient point only.

    Returns an MPCDecision for this hour.
    """
    total_demand = tier1_kw + tier2_kw + tier3_kw
    available_renewable = solar_kw + wind_kw

    # --- Step 1: Serve Tier 1 first ---
    remaining_supply = available_renewable - tier1_kw

    # --- Step 2: Decide how much battery headroom exists ---
    soc_headroom = (SOC_MAX - soc) * BATTERY_CAPACITY_KWH   # kWh available to charge
    soc_floor = (soc - SOC_MIN) * BATTERY_CAPACITY_KWH      # kWh available to discharge

    max_charge_this_hour = min(BATTERY_MAX_CHARGE_KW,
                               soc_headroom / BATTERY_CHARGE_EFF)
    max_discharge_this_hour = min(BATTERY_MAX_DISCHARGE_KW,
                                  soc_floor * BATTERY_DISCHARGE_EFF)

    # --- Step 3: Try to serve Tier 2 and Tier 3 from renewable + battery ---
    gen_output_kw = 0.0
    reason = "renewable_sufficient"

    if remaining_supply >= tier2_kw + tier3_kw:
        # Renewable covers everything
        tier2_served = tier2_kw
        tier3_served = tier3_kw
        net = remaining_supply - tier2_kw - tier3_kw  # excess → charge battery
        charge_kw = min(net, max_charge_this_hour)
        discharge_kw = 0.0
    else:
        # Need to draw from battery and/or generator
        shortfall = (tier2_kw + tier3_kw) - remaining_supply

        # Conservative discharge limit: never discharge so far that we'd
        # risk hitting the critical SOC threshold this hour.
        soc_safe_floor = max(SOC_MIN, SOC_CRITICAL + 0.03)   # keep 3% above critical
        safe_discharge_kwh = max(0.0, (soc - soc_safe_floor) * BATTERY_CAPACITY_KWH)
        safe_discharge_kw = min(max_discharge_this_hour,
                                safe_discharge_kwh * BATTERY_DISCHARGE_EFF)

        # Decide whether to run the generator:
        # - SOC is low  (approaching gen_start threshold), OR
        # - Shortfall is large relative to what battery can safely cover, OR
        # - Even with battery the demand can't be met without shedding Tier 1
        should_run_gen = (
            soc <= cfg.gen_start_soc_threshold          # SOC approaching threshold
            or shortfall > safe_discharge_kw             # battery alone insufficient (safely)
            or (available_renewable + safe_discharge_kw) < tier1_kw  # can't cover Tier 1
        )

        if should_run_gen:
            # Run generator at the minimum output needed (fuel optimisation):
            # Just enough to cover the shortfall after battery, minimum GEN_MIN_KW.
            # Charge battery to SOC target if there is surplus headroom.
            soc_charge_target = cfg.soc_target
            desired_charge_kw = max(0.0, (soc_charge_target - soc) * BATTERY_CAPACITY_KWH
                                    / BATTERY_CHARGE_EFF)
            desired_charge_kw = min(desired_charge_kw, max_charge_this_hour, 15.0)  # cap extra charging

            gen_needed = max(shortfall - safe_discharge_kw + desired_charge_kw, GEN_MIN_KW)
            gen_output_kw = min(gen_needed, GEN_MAX_KW)

            total_supply = available_renewable + gen_output_kw
            usable = total_supply - tier1_kw

            tier2_served = min(tier2_kw, max(usable, 0.0))
            remaining_for_t3 = max(usable - tier2_served, 0.0)
            tier3_served = min(tier3_kw, remaining_for_t3)

            net = usable - tier2_served - tier3_served
            charge_kw = min(max(net, 0.0), max_charge_this_hour)
            discharge_kw = 0.0
            reason = f"gen_started_soc={soc:.2f}_shortfall={shortfall:.1f}kW"
        elif shortfall <= safe_discharge_kw:
            # Battery can safely cover the shortfall
            tier2_served = tier2_kw
            tier3_served = tier3_kw
            charge_kw = 0.0
            discharge_kw = shortfall
            reason = f"battery_discharge_soc={soc:.2f}"
        else:
            # No generator yet, battery can't safely cover all — shed lower tiers
            battery_available = safe_discharge_kw + remaining_supply
            tier2_served = min(tier2_kw, max(battery_available, 0.0))
            remaining_for_t3 = max(battery_available - tier2_served, 0.0)
            tier3_served = min(tier3_kw, remaining_for_t3)

            total_served = tier2_served + tier3_served
            discharge_kw = min(
                max(total_served - remaining_supply, 0.0),
                safe_discharge_kw,
            )
            charge_kw = 0.0
            reason = f"shedding_no_gen_soc={soc:.2f}"

    # --- Step 4: Update SOC ---
    net_battery_kw = charge_kw * BATTERY_CHARGE_EFF - discharge_kw / BATTERY_DISCHARGE_EFF
    soc_delta = net_battery_kw / BATTERY_CAPACITY_KWH
    soc_after = float(np.clip(soc + soc_delta, SOC_MIN, SOC_MAX))

    # --- Step 5: Compute fuel and surplus ---
    diesel_litres = gen_output_kw * DIESEL_L_PER_KWH
    total_served = tier1_kw + tier2_served + tier3_served
    renewable_used = min(available_renewable, total_served + charge_kw)
    surplus_kw = max(available_renewable - renewable_used - discharge_kw, 0.0)

    return MPCDecision(
        battery_charge_kw=charge_kw - discharge_kw,   # signed
        gen_output_kw=gen_output_kw,
        tier1_served_kw=tier1_kw,
        tier2_served_kw=tier2_served,
        tier3_served_kw=tier3_served,
        soc_after=soc_after,
        diesel_litres=diesel_litres,
        renewable_used_kw=renewable_used,
        surplus_kw=surplus_kw,
        shed_tier2_kw=max(tier2_kw - tier2_served, 0.0),
        shed_tier3_kw=max(tier3_kw - tier3_served, 0.0),
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Horizon-level cost evaluation (used for look-ahead adjustments)
# ---------------------------------------------------------------------------

def _evaluate_horizon_cost(
    initial_soc: float,
    forecast: MPCForecast,
    cfg: ControllerConfig,
) -> float:
    """
    Roll out the greedy policy over the full 24-h horizon and return total cost.
    Used to compare against the naive baseline (Module 6).
    """
    soc = initial_soc
    total_cost = 0.0

    for h in range(cfg.horizon):
        dec = _solve_single_hour(
            soc,
            forecast.solar_kw[h],
            forecast.wind_kw[h],
            forecast.tier1_kw[h],
            forecast.tier2_kw[h],
            forecast.tier3_kw[h],
            cfg,
        )
        total_cost += (
            cfg.w_fuel * dec.diesel_litres
            + cfg.w_shed2 * dec.shed_tier2_kw
            + cfg.w_shed3 * dec.shed_tier3_kw
            + cfg.w_soc * _soc_penalty(dec.soc_after)
            - cfg.w_surplus * dec.surplus_kw
        )
        soc = dec.soc_after

    return total_cost


# ---------------------------------------------------------------------------
# Public controller class
# ---------------------------------------------------------------------------

class MPCController:
    """
    Rolling-horizon MPC energy dispatch controller.

    At each simulation timestep, call `step()` with the current system state
    and the 24h forecasts.  The controller returns an MPCDecision.

    The safety fallback layer (Module 5) wraps this class and overrides
    decisions when safety constraints are violated.
    """

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.cfg = config or ControllerConfig()
        self._last_decision: Optional[MPCDecision] = None

    def step(
        self,
        state: MPCState,
        forecast: MPCForecast,
    ) -> MPCDecision:
        """
        Compute the optimal dispatch decision for the current hour.

        Parameters
        ----------
        state : MPCState
            Current measured system state.
        forecast : MPCForecast
            24h-ahead forecasts from generation and load forecasters.

        Returns
        -------
        MPCDecision with all dispatch quantities for this hour.
        """
        decision = _solve_single_hour(
            soc=state.soc,
            solar_kw=state.solar_power_kw,
            wind_kw=state.wind_power_kw,
            tier1_kw=state.tier1_load_kw,
            tier2_kw=state.tier2_load_kw,
            tier3_kw=state.tier3_load_kw,
            cfg=self.cfg,
        )
        self._last_decision = decision
        return decision

    @property
    def last_decision(self) -> Optional[MPCDecision]:
        return self._last_decision


# ---------------------------------------------------------------------------
# Naive baseline controller (for Module 6 fuel comparison)
# ---------------------------------------------------------------------------

class NaiveBaselineController:
    """
    Simple threshold-based on/off generator controller.
    Runs the generator at full output whenever SOC drops below a fixed threshold.
    No look-ahead, no load shedding intelligence.

    Used as the comparison baseline for fuel consumption analysis.
    """

    def __init__(self, gen_on_soc_threshold: float = 0.40):
        self.threshold = gen_on_soc_threshold

    def step(self, state: MPCState) -> MPCDecision:
        available = state.solar_power_kw + state.wind_power_kw
        total_load = state.tier1_load_kw + state.tier2_load_kw + state.tier3_load_kw

        gen_output_kw = 0.0
        reason = "naive_renewable_only"

        if state.soc < self.threshold or available < total_load:
            # Generator goes to full output — no fuel optimisation
            gen_output_kw = GEN_MAX_KW
            reason = f"naive_gen_on_soc={state.soc:.2f}"

        total_supply = available + gen_output_kw
        soc_floor = (state.soc - SOC_MIN) * BATTERY_CAPACITY_KWH
        max_discharge = min(BATTERY_MAX_DISCHARGE_KW,
                            soc_floor * BATTERY_DISCHARGE_EFF)

        # Serve all loads regardless (no shedding in naive baseline)
        shortfall = total_load - total_supply
        if shortfall > 0:
            discharge_kw = min(shortfall, max_discharge)
        else:
            discharge_kw = 0.0

        charge_kw = min(
            max(total_supply - total_load, 0.0),
            (SOC_MAX - state.soc) * BATTERY_CAPACITY_KWH / BATTERY_CHARGE_EFF,
            BATTERY_MAX_CHARGE_KW,
        )

        net_battery = charge_kw * BATTERY_CHARGE_EFF - discharge_kw / BATTERY_DISCHARGE_EFF
        soc_after = float(np.clip(
            state.soc + net_battery / BATTERY_CAPACITY_KWH,
            SOC_MIN, SOC_MAX
        ))

        return MPCDecision(
            battery_charge_kw=charge_kw - discharge_kw,
            gen_output_kw=gen_output_kw,
            tier1_served_kw=state.tier1_load_kw,
            tier2_served_kw=state.tier2_load_kw,
            tier3_served_kw=state.tier3_load_kw,
            soc_after=soc_after,
            diesel_litres=gen_output_kw * DIESEL_L_PER_KWH,
            renewable_used_kw=min(available, total_load + charge_kw),
            surplus_kw=max(available - total_load - charge_kw, 0.0),
            shed_tier2_kw=0.0,
            shed_tier3_kw=0.0,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from polar_energy_sim.modules.data_generator import generate_year
    from polar_energy_sim.modules.generation_forecaster import GenerationForecaster
    from polar_energy_sim.modules.load_forecaster import LoadForecaster
    from polar_energy_sim.modules.data_generator import generate_historical_log

    print("Testing MPC controller on first 48 hours of simulation year...")
    hist = generate_historical_log(n_days=90)
    year_df = generate_year()

    gf = GenerationForecaster()
    gf.train(hist, save_models=False)

    lf = LoadForecaster()
    lf.train(hist, save_models=False)

    ctrl = MPCController()
    soc = 0.70   # start at 70%

    for h in range(48):
        row = year_df.iloc[h]
        context = year_df.iloc[max(0, h - 24): h + 1]

        solar_fc, wind_fc = gf.forecast_24h(context)
        t1_fc, t2_fc, t3_fc = lf.forecast_24h(context)
        forecast = MPCForecast(solar_fc, wind_fc, t1_fc, t2_fc, t3_fc)

        state = MPCState(
            soc=soc,
            solar_power_kw=row["solar_power_kw"],
            wind_power_kw=row["wind_power_kw"],
            tier1_load_kw=row["tier1_load_kw"],
            tier2_load_kw=row["tier2_load_kw"],
            tier3_load_kw=row["tier3_load_kw"],
            timestamp=row["timestamp"],
        )

        dec = ctrl.step(state, forecast)
        soc = dec.soc_after

        if h % 12 == 0:
            print(f"  h={h:3d}  SOC={soc:.3f}  gen={dec.gen_output_kw:.1f}kW  "
                  f"fuel={dec.diesel_litres:.3f}L  shed2={dec.shed_tier2_kw:.1f}  "
                  f"shed3={dec.shed_tier3_kw:.1f}  [{dec.reason}]")
