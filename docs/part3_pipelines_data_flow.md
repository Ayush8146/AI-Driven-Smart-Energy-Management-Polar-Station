# PART 3 — Implementation, Data Flow, Pipelines & Flowcharts

---

## 6. END-TO-END DATA FLOW

```
INPUT
  └─ ScenarioConfig (name, seed, stress flags)
  └─ No user data entry — all inputs are programmatic
     │
     ▼
GENERATION (Module 1 — data_generator.py)
  What enters:  seed, year, scenario flags
  What happens: Solar geometry calculated; Weibull wind drawn; loads synthesised
  Technology:   numpy (array math), pandas (DataFrame)
  What comes out: year_df (8760 × 11 cols), hist_df (2160 × 11 cols)
  Where next:   hist_df → Forecaster training; year_df → simulation loop
     │
     ▼
ML TRAINING (Modules 2 & 3)
  What enters:  hist_df (2160 rows)
  What happens: Feature engineering (cyclic encodings, lags, rolling means)
                LightGBM training with 80/20 chronological split
                Early stopping at 40 rounds of no val improvement
  Technology:   LightGBM, numpy, pandas
  What comes out: Trained GenerationForecaster (2 models) + LoadForecaster (3 models)
  Where next:   Used in batch pre-computation step
     │
     ▼
BATCH PRE-COMPUTATION (simulation.py)
  What enters:  Trained models + combined hist+year feature matrices
  What happens: Single batch predict() call for all 8760 hours at once
                Produces 5 numpy arrays: solar_pred_all, wind_pred_all,
                t1_pred_all, t2_pred_all, t3_pred_all
  Technology:   LightGBM batch inference, numpy
  What comes out: 5 × numpy arrays of shape (8760,)
  Where next:   Sliced every 6 hours inside simulation loop
     │
     ▼
SIMULATION LOOP — 8760 iterations (simulation.py)
  Each iteration (one simulated hour):
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  ACTUAL READINGS  ←  year_df.iloc[h]               │
  │    solar_power_kw, wind_power_kw,                   │
  │    tier1/2/3_load_kw, timestamp                     │
  │         │                                           │
  │         ▼                                           │
  │  FORECAST SLICE  ←  pre-computed arrays[h:h+24]    │
  │    (refreshed every 6 hours, O(1) array slice)      │
  │         │                                           │
  │         ▼                                           │
  │  MPCState construction                              │
  │    { soc, solar_kw, wind_kw, tier1/2/3_load_kw }   │
  │         │                                           │
  │         ▼                                           │
  │  SAFETY FALLBACK PRE-CHECK (Module 5)               │
  │    if soc ≤ 0.20 → OVERRIDE (skip MPC)             │
  │    if force_failure → OVERRIDE (skip MPC)           │
  │         │                                           │
  │         ▼ (if no pre-check trigger)                 │
  │  MPC CONTROLLER (Module 4)                          │
  │    _solve_single_hour() → MPCDecision               │
  │         │                                           │
  │         ▼                                           │
  │  SAFETY VALIDATION                                  │
  │    _is_valid_decision() → if invalid → OVERRIDE     │
  │         │                                           │
  │         ▼                                           │
  │  MPCDecision (guaranteed safe)                      │
  │    battery_charge_kw, gen_output_kw,                │
  │    tier_served, soc_after, diesel_litres            │
  │         │                                           │
  │         ├──────────────────────────────────────┐   │
  │         ▼                                       ▼   │
  │  MPC STATE UPDATE              NAIVE BASELINE RUN   │
  │    soc_mpc = dec.soc_after     (parallel, same h)   │
  │                                                     │
  │         │                       │                   │
  │         └──────────┬────────────┘                   │
  │                    ▼                                │
  │  FUEL TRACKER RECORD (Module 6)                     │
  │    mpc_litres, naive_litres, soc_mpc, soc_naive     │
  │                                                     │
  │  HOUR RECORD APPEND                                 │
  │    All 26 fields to HourRecord list                 │
  └─────────────────────────────────────────────────────┘
     │
     ▼
POST-SIMULATION STORAGE
  What enters:  SimulationResult (8760 HourRecords + FuelTracker)
  What happens: to_dataframe() converts list to pandas DataFrame
                .to_csv() writes sim_*.csv and fuel_*.csv
                dashboard.py generates 7 PNG plots via matplotlib
  Technology:   pandas, matplotlib, pathlib
  What comes out: sim_baseline.csv (8760 × 26), fuel_baseline.csv (8760 × 11), 7 PNGs
  Where next:   webapp/app.py reads CSVs for interactive display
     │
     ▼
WEB DASHBOARD OUTPUT
  What enters:  sim_*.csv and fuel_*.csv files
  What happens: Dash callbacks read CSVs on page navigation
                Plotly renders interactive charts
                Resampling (daily averages) for overview charts
  Technology:   Dash, Plotly, pandas, dash-bootstrap-components
  What comes out: 6-page interactive web application at localhost:8050
     │
     ▼
USER (reads KPI cards, explores stress test charts, interprets fuel savings)
```

---

## 7. PROJECT PIPELINES

### Pipeline 1 — Overall System Pipeline

```
User runs: python run_simulation.py
  ↓
ScenarioConfig parsed (name, seed, stress flags)
  ↓
Synthetic data generated (8,760 + 2,160 hourly rows)
  ↓
LightGBM models trained on 90-day historical log
  ↓
8,760-hour feature matrices pre-computed (batch inference)
  ↓
Per-hour simulation loop (8,760 iterations, ~8 seconds)
  ↓
CSV outputs written (sim_*.csv + fuel_*.csv)
  ↓
Matplotlib plots generated (7 PNG files)
  ↓
Console summary printed (PASS/FAIL per scenario)
  ↓
User opens webapp: python webapp/app.py → http://127.0.0.1:8050
  ↓
Interactive exploration of results
```

---

### Pipeline 2 — Technical Pipeline (per simulation hour)

```
year_df.iloc[h]  (actual weather + loads for hour h)
  ↓
MPCState object (soc, solar_kw, wind_kw, tier1/2/3)
  ↓
Pre-computed forecast array slice [h : h+24]
  ↓
MPCForecast object (5 arrays of length 24)
  ↓
SafetyFallbackLayer.step(state, forecast, sim_hour=h)
  ↓  [if safe]
MPCController.step(state, forecast)
  ↓  [always]
MPCDecision validated + returned
  ↓
soc_mpc ← decision.soc_after
  ↓
NaiveBaselineController.step(state_naive) [parallel]
  ↓
FuelTracker.record(h, ts, mpc_L, naive_L, ...)
  ↓
HourRecord appended to list
```

---

### Pipeline 3 — Data Pipeline

```
DATA SOURCE: Synthetic (data_generator.py)
  ↓
COLLECTION: generate_year() + generate_historical_log()
  Functions: _generate_solar_irradiance(), _generate_wind_speed(),
             _generate_station_load()
  Output: 8760 + 2160 hourly rows
  ↓
TRANSFORMATION: Power conversion
  wind_to_power_kw()  →  applies turbine power curve (cut-in/rated/cut-out)
  solar_to_power_kw() →  irradiance × efficiency × panel_area
  ↓
FEATURE ENGINEERING (Modules 2 & 3)
  _build_generation_features():
    - Cyclic encodings: hour_sin/cos, doy_sin/cos
    - Lags: solar_lag_{1,2,3,6,12,24}, wind_lag_{1..24}
    - Rolling: solar_roll_{3,6,12,24}, wind_roll_{3,6,12,24}
    - Raw: irradiance, wind_speed
  _build_load_features():
    - Cyclic: hour/doy/dow sin/cos
    - Polar winter flag: is_winter (DOY 90-270)
    - Tier lags: t-{1,2,3,6,12,24,48}
    - Rolling: 3h/6h/24h means
    - Cross-tier: total_load lags
  ↓
TRAINING (80/20 chronological split)
  LightGBM.train() × 5 models
  Early stopping: 40 rounds
  ↓
STORAGE: in-memory numpy arrays (8760-element batch predictions)
  ↓
ANALYSIS: per-hour dispatch decisions + SOC tracking
  ↓
OUTPUT: sim_*.csv (26 cols), fuel_*.csv (11 cols), 7 PNG plots
```

---

### Pipeline 4 — AI/ML Pipeline

```
DATASET
  Source: Synthetic — data_generator.py
  Training size: 2,160 rows (90 days × 24h)
  Simulation size: 8,760 rows (365 days × 24h)
  Columns: timestamp, solar_irradiance_wm2, wind_speed_ms,
           solar_power_kw, wind_power_kw,
           tier1/2/3_load_kw, total_load_kw
  ↓
DATA CLEANING
  - ffill().bfill() to handle any NaN from lagging operations
  - dropna() before training (removes initial lag rows)
  - np.clip() to enforce physical bounds on all outputs
  ↓
PREPROCESSING
  - Chronological sort (never shuffle time-series data)
  - 80/20 chronological split (train = first 1728 rows, val = last 432 rows)
  - NO normalisation needed for tree models
  ↓
FEATURE ENGINEERING (see Data Pipeline above)
  Generation models: 16 solar features, 20 wind features
  Load models: 21 features per tier (8 base + 7 lags + 3 rolls + 3 cross-tier)
  ↓
MODEL TRAINING (5 LightGBM regressors)
  Algorithm: Gradient-Boosted Decision Trees
  Objective: regression (RMSE minimisation)
  Hyperparameters:
    num_leaves: 31, learning_rate: 0.05
    feature_fraction: 0.8 (column subsampling)
    bagging_fraction: 0.8, bagging_freq: 5 (row subsampling)
    min_child_samples: 10 (regularisation)
    num_boost_round: 400, early_stopping_rounds: 40
  ↓
VALIDATION (held-out last 20% of chronological data)
  Solar model: val_RMSE = 0.140 kW  (on 30 kW system = 0.47% relative error)
  Wind model:  val_RMSE = 0.971 kW  (on 60 kW system = 1.62% relative error)
  Tier 1 model: val_RMSE = 1.190 kW (on ~20 kW load = 5.95% relative error)
  Tier 2 model: val_RMSE = 0.195 kW (on ~8 kW load  = 2.44% relative error)
  Tier 3 model: val_RMSE = 0.088 kW (on ~4 kW load  = 2.20% relative error)
  ↓
DEPLOYMENT (in-process, no serving infrastructure)
  Batch inference: model.predict(feature_matrix_8760rows)
  Result: 5 numpy arrays stored in memory
  Forecast refresh: every 6 simulation hours (O(1) array slice)
  Failure handling: fallback to rolling 24h mean (generation) or
                    deterministic profile (load) on any exception
  ↓
PREDICTION → MPC uses forecasts as the planning horizon input
```

---

### Pipeline 5 — Deployment Pipeline (Current State)

```
DEVELOPMENT
  Python 3.13 + pip virtual environment (or system Python)
  Code in: polar_energy_sim/ + webapp/ + run_simulation.py
  Version control: Git (local .git/)
  ↓
TESTING (executed as part of development)
  Each module has a smoke test in __main__ block
  Full integration: python run_simulation.py → all 5 scenarios pass
  Stress tests: 4 scenarios with defined pass criteria
  All pass: Tier 1 drops = 0 in every scenario
  ↓
BUILD
  No build step required — pure Python, no compilation
  Dependencies: pip install numpy pandas lightgbm scipy matplotlib dash
                dash-bootstrap-components plotly
  ↓
DEPLOYMENT (local)
  Simulation: python run_simulation.py
  Dashboard:  python webapp/app.py → http://127.0.0.1:8050
  ↓
HOSTING
  Local machine only (Windows 10/11, Python 3.13)
  No cloud, no server, fully offline
  ↓
MONITORING
  Console logging: [Module] prefix on all status messages
  Safety override log: printed to console during simulation
  Fuel summary: printed after each scenario
  ↓
REPOSITORY
  GitHub: github.com/Ayush8146/AI-Driven-Smart-Energy-Management-Polar-Station
  Branch: main
  Files: 22 tracked files (source code + sample plots)
```

---

## 8. FLOWCHARTS

### Flowchart 1 — Overall Project Workflow

```
START
  │
  ▼
Run: python run_simulation.py [--scenario X]
  │
  ▼
For each ScenarioConfig in list:
  │
  ├──→ Generate synthetic year_df + hist_df
  │         │
  │         ▼
  │    Train GenerationForecaster (2 LightGBM models)
  │         │
  │         ▼
  │    Train LoadForecaster (3 LightGBM models)
  │         │
  │         ▼
  │    Batch pre-compute feature matrices (8760 rows)
  │         │
  │         ▼
  │    Run 8760-hour simulation loop
  │         │
  │         ▼
  │    Save sim_*.csv + fuel_*.csv
  │         │
  │         ▼
  │    Generate 7 PNG plots
  │
  ├──→ [Next scenario or END]
  │
  ▼
Print FINAL CONSOLIDATED REPORT (PASS/FAIL per scenario)
  │
  ▼
User launches: python webapp/app.py
  │
  ▼
Browser: http://127.0.0.1:8050
  │
  ▼
User explores results on 6 dashboard pages
  │
  ▼
END
```

---

### Flowchart 2 — Per-Hour Simulation Decision

```
START HOUR h
  │
  ▼
Read actuals from year_df.iloc[h]
  │
  ▼
[h % 6 == 0?] ──YES──→ Slice forecast arrays [h:h+24]
  │                      Cache as MPCForecast
  │◄─────────────────────────────────────────
  NO (use cached forecast)
  │
  ▼
[is_failure_hour?]
  │
  ├──YES──→ forecast = None
  │
  NO
  │
  ▼
Construct MPCState(soc_mpc, actuals)
  │
  ▼
SafetyFallbackLayer.step()
  │
  ▼
[soc_mpc ≤ 0.20?]
  │
  ├──YES──→ SAFETY_SOC_CRITICAL override
  │           gen = 50kW, shed T2+T3
  │           │
  │           ▼
  │         [skip MPC entirely]
  │
  NO
  │
  ▼
[force_failure = True?]
  │
  ├──YES──→ SAFETY_MPC_EXCEPTION override
  │
  NO
  │
  ▼
Try: MPCController.step(state, forecast)
  │
  ├──EXCEPTION──→ SAFETY_MPC_EXCEPTION override
  │
  SUCCESS
  │
  ▼
_is_valid_decision(decision, state)?
  │
  ├──INVALID──→ SAFETY_INVALID_DECISION override
  │
  VALID
  │
  ▼
Post-check: can Tier 1 be fully served?
  │
  ├──NO──→ SAFETY_TIER1_UNDERSUPPLIED override
  │
  YES
  │
  ▼
Return MPCDecision (guaranteed safe)
  │
  ▼
[In parallel] NaiveBaselineController.step()
  │
  ▼
Update soc_mpc, soc_naive
  │
  ▼
FuelTracker.record() + HourRecord.append()
  │
  ▼
h = h + 1
  │
  ▼
[h < 8760?] ──YES──→ back to START HOUR
  │
  NO
  │
  ▼
Save results, generate plots
```

---

### Flowchart 3 — MPC Dispatch Logic (`_solve_single_hour`)

```
INPUTS: soc, solar_kw, wind_kw, tier1_kw, tier2_kw, tier3_kw
  │
  ▼
Step 1: remaining = (solar + wind) - tier1  [serve Tier 1 first, always]
  │
  ▼
Step 2: Compute safe_discharge_kw = min(max_discharge, (soc - 0.23) × 200 × 0.95)
  │
  ▼
[remaining ≥ tier2 + tier3?]
  │
  ├──YES──→ Renewable covers all loads
  │           charge_kw = min(surplus, max_charge)
  │           No generator
  │
  NO
  │
  ▼
shortfall = (tier2 + tier3) - remaining
  │
  ▼
[soc ≤ 0.40  OR  shortfall > safe_discharge  OR  renewable < tier1?]
  │
  ├──YES──→ START GENERATOR
  │           gen_needed = max(shortfall - safe_discharge + recharge(≤15kW), 10kW)
  │           gen_output = min(gen_needed, 50kW)
  │           Serve Tier 2, then Tier 3 from remaining supply
  │           Charge battery if surplus
  │
  NO
  │
  ▼
[shortfall ≤ safe_discharge?]
  │
  ├──YES──→ BATTERY DISCHARGE
  │           serve all of Tier 2 + Tier 3
  │           discharge_kw = shortfall
  │
  NO
  │
  ▼
SHED LOWER TIERS (no gen, battery at limit)
  │
  Serve as much of Tier 2 as possible, then Tier 3
  │
  ▼
Update SOC, compute diesel_litres, return MPCDecision
```

---

### Flowchart 4 — Safety Fallback Layer

```
call: safety.step(state, forecast, sim_hour, force_failure)
  │
  ▼
PRE-CHECK 1: state.soc ≤ 0.20?
  ├──YES──→ log SAFETY_SOC_CRITICAL
  │          gen=50kW, shed T2+T3, charge battery
  │          return safe_decision   [MPC never called]
  NO
  │
  ▼
PRE-CHECK 2: force_failure == True?
  ├──YES──→ log SAFETY_MPC_EXCEPTION
  │          return safe_decision   [MPC never called]
  NO
  │
  ▼
try: mpc_decision = mpc_controller.step(state, forecast)
  ├──EXCEPTION──→ log + traceback
  │               return SAFETY_MPC_EXCEPTION safe_decision
  │
  SUCCESS
  │
  ▼
_is_valid_decision(mpc_decision, state)
  ├──INVALID──→ log SAFETY_INVALID_DECISION
  │              return safe_decision
  │
  VALID
  │
  ▼
Post-check: total_available ≥ tier1_load - 0.5?
  ├──NO──→ log SAFETY_TIER1_UNDERSUPPLIED
  │          return safe_decision
  │
  YES
  │
  ▼
return mpc_decision   [unmodified — safety layer passes through]
```

---

### Flowchart 5 — LightGBM Forecasting (Training + Inference)

```
TRAINING
  Input: hist_df (2160 rows)
    │
    ▼
  _build_generation_features() / _build_load_features()
    Creates: cyclic encodings, lag features, rolling means
    │
    ▼
  dropna() → clean feature matrix
    │
    ▼
  Chronological split: train = first 80%, val = last 20%
    │
    ▼
  lgb.Dataset(X_train, y_train) + lgb.Dataset(X_val, y_val)
    │
    ▼
  lgb.train(params, dtrain, valid_sets=[dval],
            callbacks=[early_stopping(40), log_evaluation(-1)])
    │
    ▼
  Evaluate val_RMSE → print result
    │
    ▼
  Store trained model in self._solar_model / self._wind_model / self._models[tier]

INFERENCE (batch, simulation.py)
  Combined feature matrix (hist + year, 2160+8760 rows)
    │
    ▼
  model.predict(feature_matrix)  → 8760-element numpy array
    │
    ▼
  np.clip(..., 0.0, None)  → enforce non-negative power
    │
    ▼
  Stored as solar_pred_all, wind_pred_all, t1/t2/t3_pred_all
    │
    ▼
  Per-hour: forecast_slice = pred_all[h : h+24]  (O(1) array slice)
```

---

## 9. VISUAL DIAGRAM SPECIFICATIONS

### Diagram 1 — System Architecture Overview
**Why useful:** Shows jury the complete component map and data flows at a glance
**Should contain:**
- 6 coloured module boxes (one colour per module)
- Arrows: data_generator → (gen_forecaster AND load_forecaster) → simulation loop → safety_fallback → mpc_controller → fuel_tracker → CSV → webapp
- Label each arrow with what data flows: "hist_df (2160 rows)", "MPCForecast (5×24h arrays)", "MPCDecision", etc.
- Separate the safety layer visually (red border) to emphasise it wraps everything
- Add a "User → Browser → Dash App → CSV files" chain on the right side

### Diagram 2 — MPC Dispatch Priority Flow
**Why useful:** Explains the controller logic visually — the most technically complex part
**Should contain:** Decision diamond tree matching Flowchart 3 above

### Diagram 3 — Safety Layer Trigger Hierarchy
**Why useful:** Shows the jury the safety system is layered and deterministic
**Should contain:** 4-level hierarchy showing pre-check → MPC try → validate → post-check

### Diagram 4 — LightGBM Feature Engineering
**Why useful:** Explains how time-series features are built
**Should contain:** Raw timestamp column → feature extraction boxes → feature vectors → LightGBM model

### Diagram 5 — Seasonal Energy Mix (Concept)
**Why useful:** Explains the core problem visually
**Should contain:** 12-month calendar bar showing: summer = solar dominant, winter = wind dominant, zero solar in polar night

### Diagram 6 — Load Tier Priority Pyramid
**Why useful:** Instantly communicates the priority system
**Should contain:** Triangle with Tier 1 (red, never shed) at top, Tier 2 (amber) middle, Tier 3 (green, shed first) at bottom

---

## 10. AI IMAGE GENERATION PROMPTS

### Prompt 1 — System Architecture Diagram
**Title:** Polar EMS — System Architecture
**Style:** Clean technical flowchart, dark navy background, white boxes with coloured borders, thin arrows with labels
**Prompt:**
"Create a professional technical architecture diagram for a polar energy management system. Dark navy (#0b0f1a) background. Six rectangular module boxes arranged vertically and connected with directional arrows. Module 1 (blue) 'Synthetic Data Generator' at top, splits to Module 2 (cyan) 'Generation Forecaster' and Module 3 (green) 'Load Forecaster' in parallel, both merge into Module 4 (purple) 'MPC Controller', wrapped by Module 5 (red border) 'Safety Fallback Layer', connected to Module 6 (amber) 'Fuel Tracker', finally to a browser icon 'Web Dashboard'. Label each arrow with data type. Clean Inter font, no decorative elements, academic publication quality."

### Prompt 2 — Load Tier Priority Pyramid
**Title:** Station Load Priority Tiers
**Style:** Triangle pyramid infographic
**Prompt:**
"Create a clean infographic showing three priority tiers as a vertical pyramid. Top tier (smallest, red): 'Tier 1 — NEVER SHED: Heating • Medical • Communications • 18kW base'. Middle tier (amber): 'Tier 2 — SHED UNDER STRESS: Lab Equipment • Non-essential Lighting • 8kW base'. Bottom tier (largest, green): 'TIER 3 — SHED FIRST: Kitchen Amenities • Convenience Loads • 4kW base'. Dark background, white text, minimal clean style, no borders except the pyramid outline."

### Prompt 3 — Seasonal Energy Mix
**Title:** Antarctic Station — Seasonal Energy Sources
**Style:** Horizontal stacked bar chart concept
**Prompt:**
"Create a clean 12-month timeline diagram showing seasonal energy source availability at an Antarctic station. January-March (summer): tall yellow solar bar + medium cyan wind bar. April-September (polar winter): zero solar, tall cyan wind bar, small purple diesel bar. October-December (summer return): solar bar growing. Dark background, labeled months, legend. Caption: 'Solar provides summer power; wind dominates winter; diesel is backup only'."

### Prompt 4 — MPC vs Naive Fuel Comparison
**Title:** Fuel Optimisation Result
**Style:** Clean comparison bar with annotation
**Prompt:**
"Create a professional comparison infographic. Left side: large red block labelled 'Naive Controller: 118,353 L diesel / year'. Right side: small blue block labelled 'MPC Controller: 25,408 L / year'. Green arrow between them labelled '78.5% reduction = $325,308 saved'. Dark background, clean modern style, no chart axes needed."
