# PART 2 — Architecture, Technology Stack & Module Breakdown

---

## 3. COMPLETE PROJECT ARCHITECTURE

### Architecture Hierarchy

```
USER (Engineer at station / researcher reviewing results)
  │
  ▼
WEB DASHBOARD  (webapp/app.py — Dash/Plotly on localhost:8050)
  │  ← reads sim_*.csv and fuel_*.csv from outputs/
  │
  ▼
SIMULATION RUNNER  (run_simulation.py + simulation.py)
  │  ← orchestrates all 6 modules
  │
  ├──────────────────────────────────────────────┐
  ▼                                              ▼
MODULE 1                               MODULE 2 + MODULE 3
Synthetic Data Generator               ML Forecasting Layer
(data_generator.py)                    (generation_forecaster.py)
  │                                    (load_forecaster.py)
  │  year_df (8760h)                      │
  │  hist_df (2160h)  ──────────────────→ training input
  │                                       │
  │  actuals (per hour)    forecasts ←────┘ (pre-computed batch)
  │       │                    │
  ▼       ▼                    ▼
      MODULE 4:  MPC CONTROLLER  (mpc_controller.py)
          │  proposed MPCDecision
          ▼
      MODULE 5:  SAFETY FALLBACK LAYER  (safety_fallback.py)
          │  validated / overridden MPCDecision
          ▼
      MODULE 6:  FUEL TRACKER  (fuel_tracker.py)
          │  HourRecord + FuelRecord
          ▼
      CSV OUTPUTS  (polar_energy_sim/outputs/)
          │
          ▼
      DASHBOARD  (webapp/app.py)
```

---

### Component-by-Component Breakdown

| Component | Purpose | Input | Processing | Output | Connects to |
|---|---|---|---|---|---|
| **data_generator.py** | Synthetic time-series generation | Seed, year, scenario flags | Solar geometry, Weibull wind, AR(1), tier load profiles | 8760-row DataFrame + 2160-row historical log | generation_forecaster, load_forecaster, simulation loop |
| **generation_forecaster.py** | Predict 24h solar+wind power | Historical log CSV | LightGBM training + cyclic feature engineering | solar_pred_all[8760], wind_pred_all[8760] | simulation.py (batch) |
| **load_forecaster.py** | Predict 24h per-tier demand | Historical log CSV | 3 independent LightGBM models | t1/t2/t3_pred_all[8760] | simulation.py (batch) |
| **mpc_controller.py** | Fuel-minimising hourly dispatch | MPCState + MPCForecast | Greedy priority solver, SOC safe-floor guard, generator sizing | MPCDecision (7 fields) | safety_fallback.py |
| **safety_fallback.py** | Life-safety override | MPCState + MPCForecast | Pre-check SOC, try MPC, validate decision, override if needed | Guaranteed safe MPCDecision | fuel_tracker, simulation loop |
| **fuel_tracker.py** | Dual fuel ledger | sim_hour, mpc_litres, naive_litres per hour | Append-only accumulation, monthly groupby | FuelRecord list → DataFrame → CSV | simulation.py |
| **simulation.py** | Orchestration | ScenarioConfig | Feature pre-computation, per-hour loop, parallel naive | SimulationResult + CSVs | All modules |
| **webapp/app.py** | Interactive results browser | sim_*.csv + fuel_*.csv | Dash callbacks, Plotly charts | 6-page web dashboard | User (browser) |
| **dashboard.py** | Static matplotlib plots | SimulationResult | resample + plot | 7 PNG files | User (outputs/) |

---

### What this project does NOT have

| Component | Status | Reason |
|---|---|---|
| Cloud services | Not implemented | Deliberately offline-only (no internet at station) |
| REST APIs | Not implemented | Single-machine simulation, no network communication needed |
| Database (SQL/NoSQL) | Not used | CSV files serve as the data store for this prototype |
| User authentication | Not implemented | Single-user local dashboard; not a multi-user system |
| Real hardware drivers | Not implemented | Prototype only; Modbus/OPC-UA drivers would replace data_generator.py |
| Docker / cloud deployment | Not implemented | Runs as a local Python application |

---

## 4. TECHNOLOGY STACK

### Summary Table

| Layer | Technology | Version | Purpose | Why Used | Project Usage |
|---|---|---|---|---|---|
| Language | Python | 3.13 | All computation | Ecosystem richness, readable code, scientific computing standard | All modules |
| ML Framework | LightGBM | ≥ 4.0 | Gradient-boosted regression | Fast, works on small datasets, no deep learning, offline | Modules 2 & 3 |
| Data Manipulation | pandas | ≥ 2.0 | Time-series DataFrames | Standard for tabular time-series work | All modules |
| Numerical Computing | numpy | ≥ 1.24 | Array math, physics models | Vectorised computation for 8760-row arrays | All modules |
| Scientific Computing | scipy | ≥ 1.10 | Available for LP extensions | Retained for future optimisation work | Not actively used yet |
| Visualisation (static) | matplotlib | ≥ 3.7 | PNG plot generation | Offline, no server needed, fine for publication-quality plots | dashboard.py |
| Web Dashboard | Dash | 4.4.1 | Interactive browser UI | Python-native, no JS required, built on Flask | webapp/app.py |
| Dashboard Components | dash-bootstrap-components | 2.0.4 | Responsive layout grid | Professional layout without writing CSS manually | webapp/app.py |
| Interactive Charts | Plotly | 7.1.0 | Interactive web charts | Works natively with Dash; hover, zoom, export built in | webapp/app.py |
| Version Control | Git | 2.55 | Source control | Industry standard | .git/ |
| Repository Hosting | GitHub | — | Remote backup + sharing | Free, accessible, integrates with gh CLI | github.com/Ayush8146 |

---

### Technology Deep-Dives

---

#### LightGBM

**Category:** Machine Learning — Gradient-Boosted Decision Tree Framework

**Version tested:** ≥ 4.0

**Beginner:** LightGBM is like a team of thousands of simple decision trees where each tree learns from the mistakes of the previous one. The result is a model that can predict numbers (like "how much wind power will there be at 3 AM tomorrow?") very accurately even with limited training data.

**Intermediate:** LightGBM uses gradient boosting with leaf-wise tree growth (rather than level-wise like XGBoost). This makes it faster on small-to-medium datasets. It supports early stopping on a validation set, which prevents overfitting when training data is limited (90 days).

**Technical/Jury level:** Selected over deep learning models (LSTM, Transformer) because:
1. The training dataset is only 2,160 rows (90 days × 24h). Deep learning models typically need thousands to hundreds of thousands of samples to generalise.
2. LightGBM is interpretable — feature importance scores explain which lag features matter most.
3. It runs entirely offline without GPU requirements, critical for an isolated polar station.
4. The forecasting problem is tabular regression, not sequence-to-sequence, making tree ensembles the right tool class.

**Configuration used:**
```python
{"objective": "regression", "metric": "rmse", "num_leaves": 31,
 "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
 "bagging_freq": 5, "min_child_samples": 10, "num_boost_round": 400,
 "early_stopping_rounds": 40}
```

**Why not XGBoost?** LightGBM is faster on this dataset size and uses less memory. Feature parity for this use case.
**Why not sklearn GradientBoostingRegressor?** Significantly slower for the same model quality.

---

#### Dash + Plotly

**Category:** Web Framework (Python) + Interactive Charting

**Beginner:** Dash is a tool that lets you build websites in Python without needing to know web programming languages like JavaScript. Plotly makes charts that you can hover over, zoom into, and interact with.

**Intermediate:** Dash wraps Flask as an HTTP server and uses React.js for the frontend components. Callbacks are Python functions decorated with `@app.callback` that fire when the user interacts with a component (clicks a button, changes a dropdown).

**Technical/Jury:** The dashboard uses URL-based page routing via `dcc.Location` and a single router callback that renders different page functions based on the pathname. The Run Simulation page uses a background `threading.Thread` to run `run_simulation.py` as a subprocess, with a `dcc.Interval` polling callback to stream log output and progress to the browser without blocking the Dash server.

**Why Dash over Flask+Jinja2?** Dash provides reactive components (callbacks) that would require significant JavaScript to implement in plain Flask. For a data-heavy dashboard this saves hundreds of lines of code.

**Why Plotly over matplotlib for the web dashboard?** matplotlib produces static PNG files. Plotly charts are interactive (zoom, hover tooltips, pan, export) and embed natively in Dash without any extra configuration.

---

#### Model Predictive Control (MPC)

**Category:** Control Theory / Optimization

**Beginner:** MPC is like a chess player who plans several moves ahead. Every hour, the controller looks at the next 24 hours of predicted weather and demand, and finds the cheapest safe set of moves (charge the battery now? run the generator for 2 hours? shed the labs?).

**Intermediate:** The MPC uses a rolling 24-hour horizon. Every hour it re-optimises with fresh forecasts. This is "receding horizon" control — the horizon always stays 24 hours ahead.

**Technical/Jury:** The implementation uses a greedy priority-based solver rather than a full MILP (Mixed Integer Linear Program). This design decision was intentional:
- No external solver dependencies (no CVXPY, PuLP, Gurobi)
- The priority ordering (Tier 1 → renewable → battery → generator → shed) creates a deterministic, explainable solution at each step
- The quadratic SOC penalty term keeps the battery away from both extremes
- The generator sizing logic ensures minimum-needed output rather than binary on/off, which is where most of the 78.5% fuel savings come from

**Why MPC over Reinforcement Learning?**
- MPC produces explainable decisions — a station engineer can read the cost function and understand why the generator started
- RL requires months of simulation episodes to converge; with only 90 days of historical data, RL would underfit
- MPC respects hard safety constraints by design; RL safety guarantees require additional constrained RL formulations
- In a life-safety application, an explainable rule-based fallback (Module 5) must wrap the controller; this is natural with MPC and awkward with RL

---

#### Python Dataclasses

**Category:** Language Feature

**Purpose:** Type-safe, self-documenting parameter passing between modules. Used for `MPCState`, `MPCForecast`, `MPCDecision`, `ControllerConfig`, `ScenarioConfig`, `HourRecord`, `SimulationResult`, `FuelRecord`.

**Why:** Avoids passing large dictionaries or positional arguments. Each field is named and typed, making the code self-documenting and making errors (e.g. passing tier2 where tier1 is expected) immediately obvious.

---

## 5. MODULE BREAKDOWN (Detailed)

---

### Module 1 — Synthetic Data Generator (`data_generator.py`)

**Purpose:** Create physically motivated synthetic time-series that accurately represent one year of Antarctic conditions without requiring real sensor hardware.

**Input:** Seed integer, year, scenario override flags (wind lull, katabatic stress)

**Processing:**
1. Build hourly timestamp index (8,760 rows for the simulation year, 2,160 for historical log)
2. Compute solar elevation angle using real declination/hour-angle formulas for LATITUDE_DEG = -70.8°
3. Generate irradiance from clear-sky model × multiplicative haze events × Gaussian noise
4. Generate wind speed via seasonal Weibull draws → AR(1) smoothing → stochastic katabatic injection
5. Apply stress-test overrides (forced lull / katabatic ramp) if flags are set
6. Convert irradiance → kW using panel efficiency (18%) and back-calculated panel area
7. Convert wind speed → kW using the piecewise turbine power curve
8. Generate tier loads: Tier 1 with cosine winter scaling, Tier 2 with step working-hours profile, Tier 3 with Gaussian meal peaks

**Technologies:** numpy (all array math), pandas (DataFrame construction, DatetimeIndex)

**Output:**
- `year_df` — 8,760-row DataFrame: timestamp, solar_irradiance_wm2, wind_speed_ms, solar_power_kw, wind_power_kw, tier1/2/3_load_kw, total_load_kw, renewable_power_kw, net_power_kw
- `hist_df` — 2,160-row DataFrame (same schema, previous 90 days)

**Error handling:** No exception paths needed — all computation is deterministic given the seed. Numpy clips ensure all values stay in physical bounds.

**Connection to other modules:** year_df → simulation loop (actuals per hour); hist_df → GenerationForecaster.train() + LoadForecaster.train()

---

### Module 2 — Generation Forecasting Engine (`generation_forecaster.py`)

**Purpose:** Predict next-24-hour solar power and wind power output from historical patterns and weather features.

**Input:** hist_df (training); context window of recent actuals (inference)

**Processing:**
1. `_build_generation_features()` — create 16 solar features + 20 wind features from raw columns
2. `train()` — chronological 80/20 split, train two LightGBM models with early stopping
3. `_predict_24h()` — single-pass loop over 24 steps; update lag_1 feedback from previous step's prediction
4. `_safe_fallback_forecast()` — rolling 24h mean if model unavailable or exception raised

**Technologies:** LightGBM (gradient boosting), numpy (feature arrays), pandas (feature engineering)

**Output:** solar_pred_all[8760] and wind_pred_all[8760] (batch); or 24-element arrays per call

**Error handling:** Any exception in `_predict_24h()` → `_safe_fallback_forecast()` returns flat arrays from recent rolling mean. Model absence → same fallback.

**Validation RMSE (actual):** solar = 0.140 kW, wind = 0.971 kW

---

### Module 3 — Load Forecasting Engine (`load_forecaster.py`)

**Purpose:** Predict next-24-hour station electricity demand broken down by Tier 1, Tier 2, Tier 3.

**Input:** hist_df (training); context window (inference)

**Feature set (per tier):** 8 base calendar features + 7 tier-specific lags + 3 tier rolling means + 3 total_load cross-tier lags = 21 features per model

**Fallback schedule:** Deterministic hardcoded arrays scaled by recent actuals — Tier 1 flat 20 kW, Tier 2 step function (2/8/2 kW), Tier 3 meal-peak profile

**Output:** t1/t2/t3_pred_all[8760] (batch); or three 24-element arrays per call

**Validation RMSE (actual):** Tier 1 = 1.190 kW, Tier 2 = 0.195 kW, Tier 3 = 0.088 kW

---

### Module 4 — MPC Controller (`mpc_controller.py`)

**Purpose:** Compute the optimal energy dispatch decision for each hour, minimising diesel consumption while maintaining battery SOC within safe bounds and serving load tiers in priority order.

**Input:** MPCState (soc, solar_kw, wind_kw, tier1/2/3 loads), MPCForecast (24h arrays)

**Core logic (`_solve_single_hour`):**
```
Step 1: Serve Tier 1 from available renewable (non-negotiable)
Step 2: Compute safe discharge limit = min(max_discharge, (soc - 0.23) × 200 kWh × 0.95)
Step 3: Decide generator start condition:
        soc ≤ 0.40  OR  shortfall > safe_discharge  OR  renewable < Tier1
Step 4a (gen runs): size at min(shortfall - safe_discharge + recharge_cap_15kW, GEN_MIN_KW=10)
Step 4b (battery): discharge up to safe limit, serve Tier 2 then Tier 3
Step 4c (shed): shed Tier 3 first, then Tier 2, up to battery limit
Step 5: Update SOC; compute diesel_litres = gen_kW × 0.35
```

**Cost function weights:** W_FUEL=10, W_SHED2=5, W_SHED3=1, W_SOC=8, W_SURPLUS=0.1

**Output:** MPCDecision dataclass with 11 fields

**Also contains:** NaiveBaselineController — runs generator at full 50 kW whenever SOC < 0.40, no forecasting, no load shedding. Used as comparison baseline.

---

### Module 5 — Safety Fallback Layer (`safety_fallback.py`)

**Purpose:** Wrap the MPC controller with deterministic safety guarantees that do not depend on any ML component functioning correctly.

**Trigger hierarchy (evaluated in order):**

| Order | Check | Trigger | Action |
|---|---|---|---|
| 1 | soc ≤ 0.20 | SAFETY_SOC_CRITICAL | Bypass MPC entirely; generator full; shed T2+T3; charge battery |
| 2 | force_failure=True | SAFETY_MPC_EXCEPTION | Bypass MPC; deterministic safe decision |
| 3 | MPC raises exception | SAFETY_MPC_EXCEPTION | Replace with safe decision |
| 4 | Decision validation fails | SAFETY_INVALID_DECISION | Replace with safe decision |
| 5 | Post-check T1 coverage | SAFETY_TIER1_UNDERSUPPLIED | Generator full; battery assist |

**Validation checks in `_is_valid_decision()`:**
- tier1_served ≥ tier1_load - 0.01
- SOC_MIN - 0.001 ≤ soc_after ≤ SOC_MAX + 0.001
- No NaN/Inf in any field
- No tier served > demand
- Generator ∉ (0, GEN_MIN) when running

**Zero ML dependency:** The safety layer imports only numpy and the MPC dataclasses. It contains no model calls, no pandas, no CSV reads.

**Output:** Guaranteed-safe MPCDecision + audit trail in `_override_history`

---

### Module 6 — Fuel Tracker (`fuel_tracker.py`)

**Purpose:** Record diesel consumption for both MPC and naive controllers every hour, providing the quantitative proof that the fuel-optimization objective is working.

**Input:** Per-hour: sim_hour, mpc_litres, naive_litres, mpc_gen_kw, naive_gen_kw, soc_mpc, soc_naive, safety_triggered

**Processing:** Append FuelRecord to list; accumulate `_mpc_cumulative` and `_naive_cumulative` running totals

**report() output fields:** mpc_total_litres, naive_total_litres, savings_litres, savings_pct, mpc_cost_usd, naive_cost_usd, cost_savings_usd, mpc_generator_hours, naive_generator_hours, safety_override_count, monthly_breakdown, simulation_hours

**Actual verified results:**
| Metric | Value |
|---|---|
| MPC total | 25,408 L |
| Naive total | 118,353 L |
| Savings | 92,945 L (78.5%) |
| Cost saving | $325,308 |
| MPC gen hours | 5,158 h |
| Naive gen hours | 6,763 h |
