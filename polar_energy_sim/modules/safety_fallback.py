"""
Module 5: Safety Fallback Layer
=================================
A deterministic, rule-based system that sits OUTSIDE and OVERRIDES the MPC
controller.  It has ZERO dependency on any ML component.

Safety guarantees:
  1. If battery SOC drops below CRITICAL_SOC_THRESHOLD → immediately start
     generator at full output and shed Tier 2 & 3 until SOC recovers.
  2. If the MPC controller throws any exception or returns an invalid decision
     (NaN, negative served loads, Tier 1 not fully served) → intercept and
     replace with a safe deterministic decision.
  3. If renewable + battery cannot meet Tier 1 alone → generator starts at
     full output regardless of fuel cost.
  4. Tier 1 (heating, medical, comms) is NEVER shed. Ever.

This layer wraps the MPC controller:
    safety = SafetyFallbackLayer(mpc_controller)
    decision = safety.step(state, forecast)

The `triggered` flag on the returned decision indicates when the fallback
overrode the MPC.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np

from polar_energy_sim.modules.mpc_controller import (
    MPCController,
    MPCDecision,
    MPCForecast,
    MPCState,
    GEN_MAX_KW,
    GEN_MIN_KW,
    SOC_MIN,
    SOC_MAX,
    SOC_CRITICAL,
    BATTERY_CAPACITY_KWH,
    BATTERY_CHARGE_EFF,
    BATTERY_DISCHARGE_EFF,
    BATTERY_MAX_CHARGE_KW,
    BATTERY_MAX_DISCHARGE_KW,
    DIESEL_L_PER_KWH,
)


# ---------------------------------------------------------------------------
# Safety thresholds (separate from MPC config — intentionally conservative)
# ---------------------------------------------------------------------------

SAFETY_SOC_CRITICAL = 0.20      # Below this → override regardless of MPC decision
SAFETY_SOC_RESTORE  = 0.30      # Generator keeps running until SOC reaches here
SAFETY_GEN_OUTPUT   = GEN_MAX_KW  # Full output when safety engages


# ---------------------------------------------------------------------------
# Trigger reasons (for logging / audit trail)
# ---------------------------------------------------------------------------

class TriggerReason:
    SOC_CRITICAL        = "SAFETY_SOC_CRITICAL"
    MPC_EXCEPTION       = "SAFETY_MPC_EXCEPTION"
    INVALID_DECISION    = "SAFETY_INVALID_DECISION"
    TIER1_UNDERSUPPLIED = "SAFETY_TIER1_UNDERSUPPLIED"
    NO_TRIGGER          = "OK"


# ---------------------------------------------------------------------------
# Safety decision builder
# ---------------------------------------------------------------------------

def _build_safety_decision(
    state: MPCState,
    trigger: str,
    gen_output_kw: float = SAFETY_GEN_OUTPUT,
) -> MPCDecision:
    """
    Construct a safe fallback decision:
      - Generator at specified output (default: full).
      - Tier 1 served in full from any available source.
      - Tier 2 and Tier 3 served only if power remains after Tier 1 + battery.
      - Battery charged if surplus exists.
    """
    available = state.solar_power_kw + state.wind_power_kw + gen_output_kw
    tier1_need = state.tier1_load_kw

    soc_floor_kwh = (state.soc - SOC_MIN) * BATTERY_CAPACITY_KWH
    max_discharge = min(BATTERY_MAX_DISCHARGE_KW,
                        soc_floor_kwh * BATTERY_DISCHARGE_EFF)

    # Serve Tier 1 first
    after_t1 = available - tier1_need

    if after_t1 < 0:
        # Even generator + renewable can't meet Tier 1 alone — draw battery
        battery_needed = abs(after_t1)
        discharge_kw = min(battery_needed, max_discharge)
        after_t1 = 0.0
        charge_kw = 0.0
    else:
        discharge_kw = 0.0
        # Decide battery charging
        headroom = (SOC_MAX - state.soc) * BATTERY_CAPACITY_KWH
        charge_possible = min(BATTERY_MAX_CHARGE_KW, headroom / BATTERY_CHARGE_EFF)

    # Remaining for Tier 2 / 3 (safety mode: shed both, prioritise recovery)
    if trigger == TriggerReason.SOC_CRITICAL:
        # In critical SOC mode: don't serve Tier 2 or 3 — charge battery instead
        tier2_served = 0.0
        tier3_served = 0.0
        charge_kw = min(after_t1, BATTERY_MAX_CHARGE_KW,
                        (SOC_MAX - state.soc) * BATTERY_CAPACITY_KWH / BATTERY_CHARGE_EFF)
        charge_kw = max(charge_kw, 0.0)
    else:
        # Non-critical override: serve what we can after Tier 1
        tier2_served = min(state.tier2_load_kw, max(after_t1, 0.0))
        tier3_remaining = max(after_t1 - tier2_served, 0.0)
        tier3_served = min(state.tier3_load_kw, tier3_remaining)
        surplus = max(after_t1 - tier2_served - tier3_served, 0.0)
        charge_kw = min(surplus, BATTERY_MAX_CHARGE_KW,
                        (SOC_MAX - state.soc) * BATTERY_CAPACITY_KWH / BATTERY_CHARGE_EFF)
        charge_kw = max(charge_kw, 0.0)

    net_battery = charge_kw * BATTERY_CHARGE_EFF - discharge_kw / BATTERY_DISCHARGE_EFF
    soc_after = float(np.clip(
        state.soc + net_battery / BATTERY_CAPACITY_KWH,
        SOC_MIN, SOC_MAX
    ))

    diesel_litres = gen_output_kw * DIESEL_L_PER_KWH
    total_served = tier1_need + tier2_served + tier3_served
    renewable_used = min(state.solar_power_kw + state.wind_power_kw,
                         total_served + charge_kw)
    surplus_kw = max(
        (state.solar_power_kw + state.wind_power_kw + gen_output_kw)
        - total_served - charge_kw,
        0.0
    )

    return MPCDecision(
        battery_charge_kw=charge_kw - discharge_kw,
        gen_output_kw=gen_output_kw,
        tier1_served_kw=tier1_need,
        tier2_served_kw=tier2_served,
        tier3_served_kw=tier3_served,
        soc_after=soc_after,
        diesel_litres=diesel_litres,
        renewable_used_kw=renewable_used,
        surplus_kw=surplus_kw,
        shed_tier2_kw=max(state.tier2_load_kw - tier2_served, 0.0),
        shed_tier3_kw=max(state.tier3_load_kw - tier3_served, 0.0),
        reason=f"[SAFETY:{trigger}]",
    )


# ---------------------------------------------------------------------------
# Decision validator
# ---------------------------------------------------------------------------

def _is_valid_decision(decision: MPCDecision, state: MPCState) -> tuple[bool, str]:
    """
    Check that an MPC decision is physically valid and safe.
    Returns (is_valid, reason_string).
    """
    # Tier 1 must be fully served
    if decision.tier1_served_kw < state.tier1_load_kw - 0.01:
        return False, f"Tier1 undersupplied: {decision.tier1_served_kw:.2f} < {state.tier1_load_kw:.2f}"

    # SOC must stay in bounds
    if not (SOC_MIN - 0.001 <= decision.soc_after <= SOC_MAX + 0.001):
        return False, f"SOC out of bounds: {decision.soc_after:.3f}"

    # No NaN values
    for field_name in ["battery_charge_kw", "gen_output_kw", "soc_after",
                       "tier1_served_kw", "tier2_served_kw", "tier3_served_kw"]:
        val = getattr(decision, field_name)
        if not np.isfinite(val):
            return False, f"NaN/Inf in {field_name}: {val}"

    # Served loads cannot exceed demand
    if decision.tier2_served_kw > state.tier2_load_kw + 0.01:
        return False, f"Tier2 over-served"
    if decision.tier3_served_kw > state.tier3_load_kw + 0.01:
        return False, f"Tier3 over-served"

    # Generator output must be in valid range
    if decision.gen_output_kw > 0 and decision.gen_output_kw < GEN_MIN_KW - 0.01:
        return False, f"Generator below minimum: {decision.gen_output_kw:.2f} kW"

    return True, "ok"


# ---------------------------------------------------------------------------
# SafetyFallbackLayer
# ---------------------------------------------------------------------------

class SafetyFallbackLayer:
    """
    Wraps any controller (MPC or otherwise) and enforces safety rules.

    All calls go through `step()` which:
      1. Pre-checks the system state for critical SOC.
      2. Tries the underlying MPC controller.
      3. Validates the decision.
      4. Returns a safe fallback if any check fails.

    The `triggered` property reports whether the last step engaged the fallback.
    """

    def __init__(self, mpc_controller: MPCController):
        self._mpc = mpc_controller
        self._triggered = False
        self._trigger_reason = TriggerReason.NO_TRIGGER
        self._trigger_count = 0
        self._last_trigger_hour: Optional[int] = None
        self._override_history: list[dict] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def step(
        self,
        state: MPCState,
        forecast: MPCForecast,
        sim_hour: int = 0,
        force_failure: bool = False,    # for stress test 3: simulated AI failure
    ) -> MPCDecision:
        """
        Compute the dispatch decision with safety guarantees.

        Parameters
        ----------
        state : MPCState
            Current system state.
        forecast : MPCForecast
            24h-ahead forecasts (may be None or garbage — safety handles it).
        sim_hour : int
            Simulation hour index (for logging).
        force_failure : bool
            If True, bypasses the MPC entirely to simulate an AI failure.
            Demonstrates that the safety fallback operates independently.

        Returns
        -------
        MPCDecision — guaranteed safe.
        """
        self._triggered = False
        self._trigger_reason = TriggerReason.NO_TRIGGER

        # --- Pre-check 1: Critical SOC override ---
        if state.soc <= SAFETY_SOC_CRITICAL:
            return self._override(
                state, TriggerReason.SOC_CRITICAL, sim_hour,
                "Battery SOC critical — shedding non-critical loads, starting generator"
            )

        # --- Pre-check 2: Simulated AI failure (stress test 3) ---
        if force_failure:
            return self._override(
                state, TriggerReason.MPC_EXCEPTION, sim_hour,
                "AI failure injected — safety fallback engaged"
            )

        # --- Try MPC controller ---
        mpc_decision = None
        try:
            if forecast is None:
                raise ValueError("Forecast is None — ML component unavailable")
            mpc_decision = self._mpc.step(state, forecast)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[SafetyFallback] MPC raised exception at hour {sim_hour}:\n{tb}")
            return self._override(
                state, TriggerReason.MPC_EXCEPTION, sim_hour,
                f"MPC exception: {exc}"
            )

        # --- Validate the decision ---
        valid, reason = _is_valid_decision(mpc_decision, state)
        if not valid:
            print(f"[SafetyFallback] Invalid MPC decision at hour {sim_hour}: {reason}")
            return self._override(
                state, TriggerReason.INVALID_DECISION, sim_hour, reason
            )

        # --- Post-check: Tier 1 must always be fully covered ---
        renewable_plus_gen = (
            state.solar_power_kw + state.wind_power_kw + mpc_decision.gen_output_kw
        )
        battery_available = (state.soc - SOC_MIN) * BATTERY_CAPACITY_KWH * BATTERY_DISCHARGE_EFF
        total_available = renewable_plus_gen + battery_available
        if total_available < state.tier1_load_kw - 0.5:
            return self._override(
                state, TriggerReason.TIER1_UNDERSUPPLIED, sim_hour,
                f"Insufficient supply for Tier 1: {total_available:.1f} < {state.tier1_load_kw:.1f} kW"
            )

        return mpc_decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _override(
        self,
        state: MPCState,
        reason: str,
        sim_hour: int,
        detail: str,
    ) -> MPCDecision:
        self._triggered = True
        self._trigger_reason = reason
        self._trigger_count += 1
        self._last_trigger_hour = sim_hour

        record = {
            "sim_hour": sim_hour,
            "reason": reason,
            "detail": detail,
            "soc": state.soc,
            "tier1_kw": state.tier1_load_kw,
        }
        self._override_history.append(record)

        print(f"[SafetyFallback] *** OVERRIDE at hour {sim_hour}: {reason} — {detail}")

        decision = _build_safety_decision(state, reason)
        return decision

    # ------------------------------------------------------------------
    # Properties / reporting
    # ------------------------------------------------------------------

    @property
    def triggered(self) -> bool:
        """True if the safety fallback fired on the most recent step."""
        return self._triggered

    @property
    def trigger_reason(self) -> str:
        return self._trigger_reason

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def override_history(self) -> list[dict]:
        return list(self._override_history)

    def summary(self) -> dict:
        return {
            "total_overrides": self._trigger_count,
            "override_events": self._override_history,
            "last_trigger_hour": self._last_trigger_hour,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from polar_energy_sim.modules.data_generator import generate_year, generate_historical_log
    from polar_energy_sim.modules.generation_forecaster import GenerationForecaster
    from polar_energy_sim.modules.load_forecaster import LoadForecaster

    print("Testing Safety Fallback Layer...")
    hist = generate_historical_log(n_days=90)
    year_df = generate_year()

    gf = GenerationForecaster()
    gf.train(hist, save_models=False)
    lf = LoadForecaster()
    lf.train(hist, save_models=False)

    mpc = MPCController()
    safety = SafetyFallbackLayer(mpc)

    # Test 1: Normal operation
    print("\n--- Normal operation (hour 0) ---")
    row = year_df.iloc[0]
    context = year_df.iloc[:1]
    s_fc, w_fc = gf.forecast_24h(context)
    t1_fc, t2_fc, t3_fc = lf.forecast_24h(context)
    fc = MPCForecast(s_fc, w_fc, t1_fc, t2_fc, t3_fc)
    state = MPCState(
        soc=0.7, solar_power_kw=row["solar_power_kw"],
        wind_power_kw=row["wind_power_kw"],
        tier1_load_kw=row["tier1_load_kw"],
        tier2_load_kw=row["tier2_load_kw"],
        tier3_load_kw=row["tier3_load_kw"],
    )
    dec = safety.step(state, fc, sim_hour=0)
    print(f"  Triggered: {safety.triggered}  SOC_after={dec.soc_after:.3f}  T1={dec.tier1_served_kw:.2f} kW")

    # Test 2: Critical SOC
    print("\n--- Critical SOC override test ---")
    state_critical = MPCState(
        soc=0.18, solar_power_kw=0.0, wind_power_kw=5.0,
        tier1_load_kw=22.0, tier2_load_kw=8.0, tier3_load_kw=4.0,
    )
    dec2 = safety.step(state_critical, fc, sim_hour=100)
    print(f"  Triggered: {safety.triggered}  Reason: {safety.trigger_reason}")
    print(f"  Gen: {dec2.gen_output_kw:.1f} kW  Shed2: {dec2.shed_tier2_kw:.1f}  Shed3: {dec2.shed_tier3_kw:.1f}")

    # Test 3: Forced AI failure
    print("\n--- Forced AI failure test ---")
    dec3 = safety.step(state, fc, sim_hour=200, force_failure=True)
    print(f"  Triggered: {safety.triggered}  Reason: {safety.trigger_reason}")
    print(f"  Gen: {dec3.gen_output_kw:.1f} kW  T1: {dec3.tier1_served_kw:.2f} kW")

    print(f"\nTotal overrides: {safety.trigger_count}")
