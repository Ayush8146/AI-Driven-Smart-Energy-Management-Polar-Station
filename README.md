# AI-Driven Smart Energy Management System
## Polar Research Station Prototype — Maitri/Antarctica Model

A simulation-based prototype of an intelligent microgrid controller for a polar research station operating under extreme seasonal conditions. Models one full year (8 760 hourly steps) of a station's energy system and validates the core control logic, safety guarantees, and fuel-optimisation objective entirely from synthetic data — no hardware required.

---

## Quick start (under 2 minutes)

```bash
# 1. Install dependencies
pip install numpy pandas lightgbm scipy matplotlib

# 2. Run all five scenarios + generate all plots
python run_simulation.py

# Results land in:  polar_energy_sim/outputs/
```

That's it. The run takes roughly **45–50 seconds** on a laptop and prints a consolidated pass/fail table at the end.

---

## What gets produced

| File | Description |
|---|---|
| `outputs/sim_<scenario>.csv` | Hour-by-hour simulation record (SOC, generation, loads, dispatch) |
| `outputs/fuel_<scenario>.csv` | Dual fuel ledger: MPC vs naive baseline, cumulative |
| `outputs/dashboard_baseline.png` | Full-year 4-panel overview |
| `outputs/stress1_katabatic.png` | Katabatic wind event zoom |
| `outputs/stress2_wind_lull.png` | Winter wind lull battery drawdown |
| `outputs/stress3_forecast_failure.png` | AI failure → safety fallback engagement |
| `outputs/stress4_fuel_comparison.png` | MPC vs naive fuel bar + cumulative chart |

---

## Running individual scenarios

```bash
python run_simulation.py --scenario baseline
python run_simulation.py --scenario stress1_katabatic_winter
python run_simulation.py --scenario stress2_wind_lull
python run_simulation.py --scenario stress3_forecast_failure
python run_simulation.py --scenario stress4_fuel_comparison

# Skip plot generation (faster, data only)
python run_simulation.py --no-plots
```

---

## Project structure

```
polar_energy_sim/
├── modules/
│   ├── data_generator.py        # Module 1 — Synthetic weather + load data
│   ├── generation_forecaster.py # Module 2 — LightGBM solar/wind forecaster
│   ├── load_forecaster.py       # Module 3 — LightGBM per-tier demand forecaster
│   ├── mpc_controller.py        # Module 4 — MPC dispatch + naive baseline
│   ├── safety_fallback.py       # Module 5 — Deterministic safety override layer
│   └── fuel_tracker.py          # Module 6 — Dual fuel consumption ledger
├── data/
│   ├── simulation_year.csv      # Generated on first run
│   └── historical_log.csv       # 90-day history for forecaster training
├── outputs/                     # Simulation results + plots (created on run)
├── simulation.py                # Main loop — wires all six modules together
└── dashboard.py                 # All matplotlib visualisations
run_simulation.py                # Top-level entry point
README.md
```

---

## Architecture

```
                     ┌─────────────────────┐
                     │  Synthetic Data      │  Module 1
                     │  Generator           │  generate_year() + generate_historical_log()
                     └──────────┬──────────┘
                                │ hourly actuals
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                   ▼
  ┌───────────────────┐  ┌───────────────────┐     │
  │ Generation        │  │ Load              │     │
  │ Forecaster        │  │ Forecaster        │     │  actuals fed
  │ (LightGBM)        │  │ (LightGBM ×3)     │     │  to controller
  │ Module 2          │  │ Module 3          │     │
  └─────────┬─────────┘  └────────┬──────────┘     │
            │ solar_fc, wind_fc   │ t1_fc, t2_fc, t3_fc
            └──────────┬──────────┘
                       ▼
          ┌────────────────────────┐
          │  MPC Controller        │  Module 4
          │  Rolling 24h horizon   │  minimise diesel subject to
          │  Fuel-cost objective   │  SOC bounds + tier priorities
          └────────────┬───────────┘
                       │ proposed decision
                       ▼
          ┌────────────────────────┐
          │  Safety Fallback Layer │  Module 5
          │  Zero ML dependency    │  overrides on SOC_CRITICAL,
          │  Deterministic rules   │  MPC exception, invalid output
          └────────────┬───────────┘
                       │ safe decision
                       ▼
          ┌────────────────────────┐
          │  Fuel Tracker          │  Module 6
          │  MPC + Naive ledgers   │  records both controllers in parallel
          └────────────────────────┘
```

### Data flow at each hourly step

1. Read actual solar power, wind power, and tier loads from the synthetic dataset.
2. Every 6 hours, slice the pre-computed LightGBM predictions into a 24-hour MPCForecast object (MPC reuse pattern — forecasts aren't recomputed every hour).
3. Safety Fallback Layer pre-checks the current SOC. If SOC ≤ 20%, it bypasses the MPC entirely and issues a deterministic safe decision (generator full output, shed Tier 2 + 3).
4. If no pre-check fires, the MPC controller solves the single-hour dispatch: serve Tier 1 always, minimise generator output subject to keeping SOC above the safe floor, shed lower tiers only as a last resort before generator start.
5. The Safety Fallback Layer post-validates the MPC decision (NaN check, SOC bounds, Tier 1 coverage). Any invalid output triggers an override.
6. Both the MPC decision and an equivalent naive baseline decision are written to the Fuel Tracker.
7. Battery SOC is advanced using the MPC decision's net charge/discharge.

---

## The six modules explained

### Module 1 — Synthetic Data Generator (`data_generator.py`)

Produces physically motivated hourly time-series for a station at 70.8°S latitude.

- **Solar irradiance**: real solar geometry (declination, hour angle, elevation) + multiplicative haze events (more frequent in Antarctic summer due to ice fog), Gaussian noise.
- **Wind speed**: seasonal Weibull distribution (winter mean ~11 m/s, summer ~7.5 m/s), AR(1) autocorrelation for hour-to-hour persistence, katabatic event injection (25–45 m/s, 6–48 h duration).
- **Turbine power curve**: cut-in 3 m/s → rated 60 kW at 12 m/s → cut-out (shutdown) at 25 m/s.
- **Load tiers**: Tier 1 (heating scales with winter severity), Tier 2 (working-hours profile), Tier 3 (meal-time peaks).

Key outputs: `simulation_year.csv` (8 760 rows), `historical_log.csv` (2 160 rows used for forecaster training).

### Module 2 — Generation Forecasting Engine (`generation_forecaster.py`)

Two independent LightGBM models — one for solar, one for wind — trained on the 90-day historical log.

- Features: cyclic hour/day-of-year encodings, 6 lag depths (t-1 to t-24), rolling means (3h/6h/12h/24h), raw irradiance and wind speed readings.
- 80/20 chronological train/val split; early stopping at 40 rounds.
- Validation RMSE: solar ~0.14 kW, wind ~0.97 kW on a 30 kW / 60 kW system respectively.
- Falls back to a rolling 24-hour mean if the model raises any exception.

### Module 3 — Load Forecasting Engine (`load_forecaster.py`)

Three independent LightGBM models, one per tier, using consumption history as features.

- Features: same cyclic calendar encodings as Module 2, plus tier-specific lag/rolling values, is-winter flag.
- Deliberately separate from Module 2: one predicts supply, the other predicts demand; the MPC needs both independently.
- Validation RMSE: Tier 1 ~1.19 kW, Tier 2 ~0.20 kW, Tier 3 ~0.09 kW.

### Module 4 — MPC Controller (`mpc_controller.py`)

A greedy rolling-horizon controller with a fuel-minimisation objective.

**Cost function (per horizon step):**
```
J = w_fuel × gen_kW × 0.35 L/kWh
  + w_shed2 × shed_tier2_kW
  + w_shed3 × shed_tier3_kW
  + w_soc   × (SOC − 0.65)²
  − w_surplus × surplus_kW
```

**Dispatch priority:**
1. Tier 1 is always fully served — it is never a decision variable.
2. Use renewable power first.
3. Discharge battery only down to a safe floor (SOC_CRITICAL + 3% margin).
4. Start generator when SOC ≤ 40% or when battery alone cannot safely cover the shortfall. Output is sized to the minimum needed plus a small recharge component — not at full capacity unless required. This is the fuel-optimisation objective.
5. Shed Tier 3 before Tier 2 when generator is not running and battery is near its safe floor.

**NaiveBaselineController** (same file) runs in parallel: starts generator at full output (50 kW) any time SOC < 40%, with no load shedding logic and no look-ahead. Used as the fuel comparison baseline in Stress Test 4.

### Module 5 — Safety Fallback Layer (`safety_fallback.py`)

A deterministic wrapper around the MPC controller with zero dependency on any ML component.

| Trigger | Condition | Action |
|---|---|---|
| `SAFETY_SOC_CRITICAL` | SOC ≤ 20% | Generator full output, shed Tier 2 + Tier 3, charge battery |
| `SAFETY_MPC_EXCEPTION` | MPC raises any exception | Deterministic safe decision (same as above) |
| `SAFETY_INVALID_DECISION` | NaN / SOC out of bounds / Tier 1 undersupplied | Replace with safe decision |
| `SAFETY_TIER1_UNDERSUPPLIED` | Total available power < Tier 1 demand | Generator full output, battery assist |

The `force_failure=True` flag used in Stress Test 3 directly injects an MPC exception without touching any ML code — demonstrating that the safety layer operates independently.

### Module 6 — Fuel Tracker (`fuel_tracker.py`)

An append-only dual ledger. Every simulation hour records:
- MPC diesel litres burned and generator output
- Naive baseline litres burned and generator output
- Battery SOC for both controllers
- Whether the safety fallback fired

The `report()` method produces full-year totals, monthly breakdown, generator runtime hours, cost estimates (default $3.50/L Antarctic logistics price), and percentage savings.

---

## Stress test results (verified runs)

### Test 1 — Worst-case winter: katabatic wind event (Day 150, 3 days)

Wind ramps to 40 m/s → turbine cuts out → renewable drops to near zero for 72 hours in deep polar winter. The controller detects the shortfall, starts the generator at minimum required output, and keeps Tier 1 running throughout. SOC is maintained above the critical threshold.

```
Tier 1 drops : 0     Safety overrides : 0
MPC fuel     : 25 673 L    Savings vs naive : 78.3%
```

**Plot:** `outputs/stress1_katabatic.png` — wind speed panel shows the cut-out zone in red; generation panel shows the renewable gap; SOC panel shows the controlled drawdown and generator-assisted recovery.

### Test 2 — Multi-day wind lull in winter (Day 180, 5 days)

Wind drops below cut-in speed for 5 consecutive days in mid-June (deepest polar night — no solar, no wind). Battery draws down steadily. MPC starts the generator earlier than the naive controller (proactive SOC floor protection) and sheds Tier 2/3 to reduce generator runtime. Battery recovers fully once wind resumes.

```
Tier 1 drops : 0     Safety overrides : 0
MPC fuel     : 25 958 L    Savings vs naive : 78.3%
```

**Plot:** `outputs/stress2_wind_lull.png` — four panels: wind (lull clearly visible), generation vs demand, battery SOC drawdown curve (MPC vs naive), generator + shedding activation.

### Test 3 — Sensor/forecast failure (Hours 4320–4368, 48 hours)

The AI forecasting components are forcibly bypassed for 48 consecutive hours during early summer (mid-June). The Safety Fallback Layer detects the null forecast on every affected hour, logs `SAFETY_MPC_EXCEPTION`, starts the generator, and serves Tier 1 in full. Tier 2 and Tier 3 are served from available renewable + generator power.

```
Tier 1 drops : 0     Safety overrides : 48 (exactly one per failure hour)
MPC fuel     : 26 166 L    Savings vs naive : 77.9%
```

**Plot:** `outputs/stress3_forecast_failure.png` — red shaded band marks the failure window; red dots on the SOC panel mark every hour the safety layer was active; generator panel shows immediate engagement; Tier 1 panel confirms no gap.

### Test 4 — Fuel optimisation comparison (full year)

Identical scenario to baseline. Both MPC and naive controllers run in parallel over the same 8 760-hour dataset.

| Controller | Diesel (L) | Cost ($3.50/L) | Generator hours |
|---|---|---|---|
| MPC + fuel-cost objective | 25 408 | $88 926 | 5 158 |
| Naive on/off threshold | 118 353 | $414 234 | 6 763 |
| **Savings** | **92 945 L** | **$325 308** | **1 605 h** |
| **Reduction** | **78.5%** | **78.5%** | **23.7%** |

**Plot:** `outputs/stress4_fuel_comparison.png` — left panel: monthly grouped bar chart with per-month savings percentage; right panel: cumulative consumption lines for the full year with the savings area shaded.

The large savings figure reflects the naive controller's blunt strategy: it runs the generator at full 50 kW output any time SOC < 40%, regardless of actual load or renewable availability. The MPC sizes generator output to the minimum needed, sheds low-priority loads instead of burning diesel, and charges the battery only when cost-effective.

---

## Interpreting simulation output CSV columns

| Column | Unit | Meaning |
|---|---|---|
| `soc_before` / `soc_after` | fraction 0–1 | Battery state of charge at start / end of hour |
| `gen_output_kw` | kW | MPC generator dispatch this hour (0 = off) |
| `battery_charge_kw` | kW signed | + = charging, − = discharging |
| `tier1_served_kw` | kW | Tier 1 power actually delivered (must equal `tier1_load_kw`) |
| `shed_tier2_kw` / `shed_tier3_kw` | kW | Load not served this hour |
| `diesel_litres_mpc` | L | Fuel burned by MPC this hour |
| `diesel_litres_naive` | L | Fuel burned by naive baseline this hour (parallel) |
| `safety_triggered` | bool | True = safety fallback overrode MPC this hour |
| `safety_reason` | string | Which trigger fired (or "renewable_sufficient" etc.) |
| `surplus_kw` | kW | Excess renewable power that was curtailed |

---

## Physical parameters (configurable in `mpc_controller.py`)

| Parameter | Value | Notes |
|---|---|---|
| Battery capacity | 200 kWh | Usable bank |
| SOC hard minimum | 15% | Battery protection |
| SOC critical (safety) | 20% | Safety fallback trigger |
| SOC target (MPC) | 65% | Preferred operating point |
| Generator min output | 10 kW | Minimum when running |
| Generator max output | 50 kW | Rated capacity |
| Diesel consumption | 0.35 L/kWh | Typical Antarctic genset |
| MPC gen-start threshold | 40% SOC | Below this the MPC may start the generator |
| Solar panel capacity | 30 kW peak | 18% efficiency assumed |
| Wind turbine rated | 60 kW | Cut-in 3 m/s, cut-out 25 m/s |

---

## Running individual modules for inspection

Each module can be run directly for a quick self-test:

```bash
python -m polar_energy_sim.modules.data_generator
python -m polar_energy_sim.modules.generation_forecaster
python -m polar_energy_sim.modules.load_forecaster
python -m polar_energy_sim.modules.mpc_controller
python -m polar_energy_sim.modules.safety_fallback
python -m polar_energy_sim.modules.fuel_tracker
```

---

## Dependencies

| Package | Version tested | Role |
|---|---|---|
| `numpy` | ≥ 1.24 | Numerical arrays throughout |
| `pandas` | ≥ 2.0 | Time-series data handling |
| `lightgbm` | ≥ 4.0 | Gradient-boosted forecasting models |
| `scipy` | ≥ 1.10 | Unused directly — retained for future LP extensions |
| `matplotlib` | ≥ 3.7 | All visualisations |

No cloud services, no internet access required. All computation is local.

---

## Success criteria (verified)

| Criterion | Result |
|---|---|
| Tier 1 never dropped across all 4 stress tests | ✓ 0 drop-hours in every scenario |
| Safety fallback visibly overrides MPC when triggered | ✓ 48 overrides in Stress Test 3, 0 in others |
| Fuel savings measurably greater than naive baseline | ✓ 78.5% reduction (92 945 L saved over a simulated year) |
| Code readable and runnable in under 10 minutes | ✓ `pip install` + `python run_simulation.py` |

---

## What this prototype does NOT do

- No real hardware integration or live sensor drivers.
- No cloud services, telemetry, or remote APIs.
- No RL-based or deep-learning controllers (deliberately avoided for explainability and data-efficiency reasons).
- No multi-turbine or multi-panel modelling.
- No thermal storage or hydrogen backup modelling.
- Forecaster accuracy is limited by the synthetic training data (90 days) — a deployed system would retrain on real station logs.

These are appropriate omissions for a control-logic validation prototype. The next development phase would replace `data_generator.py` with real sensor feeds and `generate_historical_log()` with a live CSV logger.
