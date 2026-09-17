"""
Module 1: Synthetic Data Generator
===================================
Generates a full year of hourly time-series data for:
  - Solar irradiance  (near-zero in polar winter, high but haze-degraded in summer)
  - Wind speed        (primary winter source, including katabatic extreme events)
  - Station load      (Tier 1 / Tier 2 / Tier 3 breakdown)
  - Historical consumption log (used to train the load forecaster)

All values are physically motivated but synthetic — no real sensor hardware required.
Modelled on the Maitri station latitude (~70.8°S).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Station / physics constants
# ---------------------------------------------------------------------------

LATITUDE_DEG = -70.8          # Maitri / Antarctic station latitude
PANEL_CAPACITY_KW = 30.0      # Installed solar panel peak capacity (kW)
TURBINE_CAPACITY_KW = 60.0    # Installed wind turbine rated capacity (kW)
TURBINE_CUT_IN_MS = 3.0       # Wind turbine cut-in speed (m/s)
TURBINE_RATED_MS = 12.0       # Wind turbine rated speed (m/s)
TURBINE_CUT_OUT_MS = 25.0     # Wind turbine cut-out (storm protection) (m/s)

# Tier load baselines (kW) — approximate for a ~30-person polar station
TIER1_BASE_KW = 18.0          # Heating + medical + comms (always on)
TIER2_BASE_KW = 8.0           # Lab equipment + non-essential lighting
TIER3_BASE_KW = 4.0           # Kitchen amenities + convenience loads

HOURS_PER_YEAR = 8760


# ---------------------------------------------------------------------------
# Helper: day-of-year solar geometry for southern hemisphere station
# ---------------------------------------------------------------------------

def _solar_declination_deg(day_of_year: np.ndarray) -> np.ndarray:
    """Return solar declination in degrees for each day."""
    return 23.45 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 81)))


def _daylight_fraction(day_of_year: np.ndarray) -> np.ndarray:
    """
    Approximate fraction of daylight hours for the given latitude.
    Returns a value in [0, 1] representing proportion of 24 h that is daylit.
    Near the poles this is 0 (polar night) or 1 (midnight sun).
    """
    lat_rad = np.radians(LATITUDE_DEG)
    decl_rad = np.radians(_solar_declination_deg(day_of_year))
    # Hour-angle formula for sunrise/sunset
    cos_ha = -np.tan(lat_rad) * np.tan(decl_rad)
    cos_ha = np.clip(cos_ha, -1.0, 1.0)
    ha_rad = np.arccos(cos_ha)
    daylight_hours = 2.0 * np.degrees(ha_rad) / 15.0  # hours
    return np.clip(daylight_hours / 24.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Solar irradiance synthesis
# ---------------------------------------------------------------------------

def _generate_solar_irradiance(timestamps: pd.DatetimeIndex,
                                rng: np.random.Generator) -> np.ndarray:
    """
    Simulate hourly solar irradiance (W/m²).

    Physics:
      - Uses actual solar geometry for the station latitude.
      - Peak summer irradiance ~900 W/m² (clear sky).
      - Polar-night hours return 0.
      - Atmospheric haze modelled as a multiplicative random factor,
        more frequent and severe in summer (paradoxically — moisture, ice fog).
      - Gaussian noise added to simulate sensor / cloud variability.
    """
    n = len(timestamps)
    doy = timestamps.day_of_year.values.astype(float)
    hour = timestamps.hour.values.astype(float)

    # 1. Solar elevation angle for each hour
    decl_deg = _solar_declination_deg(doy)
    decl_rad = np.radians(decl_deg)
    lat_rad = np.radians(LATITUDE_DEG)
    hour_angle_rad = np.radians((hour - 12.0) * 15.0)
    sin_elev = (np.sin(lat_rad) * np.sin(decl_rad)
                + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad))
    sin_elev = np.clip(sin_elev, 0.0, 1.0)

    # 2. Clear-sky irradiance
    clear_sky_peak = 900.0  # W/m²
    irradiance = clear_sky_peak * sin_elev

    # 3. Haze/cloud attenuation — persistent multi-hour events
    #    More likely in Antarctic summer (Dec/Jan/Feb → southern summer ≈ DOY 335–90)
    haze_factor = np.ones(n)
    i = 0
    while i < n:
        # Southern summer = DOY 335-365 or 1-90 (approx)
        is_summer = (doy[i] > 300) or (doy[i] < 100)
        haze_prob = 0.25 if is_summer else 0.08   # base hourly haze probability
        if rng.random() < haze_prob:
            duration = int(rng.integers(2, 12))   # 2–12 hour haze event
            severity = rng.uniform(0.3, 0.85)     # 15%–70% attenuation
            haze_factor[i:i + duration] = severity
            i += duration
        else:
            i += 1

    irradiance *= haze_factor

    # 4. Small Gaussian noise (±5%)
    noise = rng.normal(1.0, 0.05, n)
    irradiance *= np.clip(noise, 0.5, 1.3)

    return np.clip(irradiance, 0.0, clear_sky_peak)


# ---------------------------------------------------------------------------
# Wind speed synthesis
# ---------------------------------------------------------------------------

def _generate_wind_speed(timestamps: pd.DatetimeIndex,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Simulate hourly wind speed (m/s) at hub height.

    Modelling:
      - Seasonal Weibull distribution: stronger, more variable in winter.
      - Katabatic events: sudden ramp-up to 25–45 m/s, lasting 6–48 h,
        more frequent April–September (polar winter).
      - Temporal autocorrelation via AR(1) process.
    """
    n = len(timestamps)
    doy = timestamps.day_of_year.values.astype(float)

    # Southern hemisphere winter = DOY ~90-270 (April–September)
    is_winter = (doy >= 90) & (doy <= 270)

    # Weibull scale (lambda) by season: winter ~10 m/s mean, summer ~7 m/s mean
    scale = np.where(is_winter, 11.0, 7.5)
    shape_k = 2.0   # Rayleigh-like (Weibull shape)

    # Base wind via Weibull draws, then AR(1) smoothing
    base_wind = scale * rng.weibull(shape_k, n)

    # AR(1) smoothing to introduce hour-to-hour persistence
    alpha = 0.7
    wind = np.zeros(n)
    wind[0] = base_wind[0]
    for t in range(1, n):
        wind[t] = alpha * wind[t - 1] + (1 - alpha) * base_wind[t]

    # Katabatic events (extreme gusts)
    i = 0
    while i < n:
        # Higher probability in winter
        event_prob = 0.003 if is_winter[i] else 0.0005
        if rng.random() < event_prob:
            duration = int(rng.integers(6, 49))
            peak_speed = rng.uniform(25.0, 45.0)
            # Ramp up and down within the event
            ramp = np.linspace(wind[i], peak_speed, duration // 2)
            decay = np.linspace(peak_speed, wind[min(i + duration, n - 1)], duration - duration // 2)
            event_profile = np.concatenate([ramp, decay])[:duration]
            end = min(i + duration, n)
            wind[i:end] = np.maximum(wind[i:end], event_profile[:end - i])
            i += duration
        else:
            i += 1

    return np.clip(wind, 0.0, 60.0)


# ---------------------------------------------------------------------------
# Wind power conversion
# ---------------------------------------------------------------------------

def wind_to_power_kw(wind_ms: np.ndarray) -> np.ndarray:
    """
    Convert wind speed (m/s) to turbine output power (kW) using a
    simplified power curve with cut-in, rated, and cut-out thresholds.
    """
    w = np.asarray(wind_ms, dtype=float)
    power = np.zeros_like(w)

    # Linear ramp between cut-in and rated speed
    ramp_mask = (w >= TURBINE_CUT_IN_MS) & (w < TURBINE_RATED_MS)
    power[ramp_mask] = TURBINE_CAPACITY_KW * (
        (w[ramp_mask] - TURBINE_CUT_IN_MS) / (TURBINE_RATED_MS - TURBINE_CUT_IN_MS)
    ) ** 3

    # Rated power region
    rated_mask = (w >= TURBINE_RATED_MS) & (w < TURBINE_CUT_OUT_MS)
    power[rated_mask] = TURBINE_CAPACITY_KW

    # Above cut-out: turbine shuts down (storm protection)
    power[w >= TURBINE_CUT_OUT_MS] = 0.0

    return power


# ---------------------------------------------------------------------------
# Solar power conversion
# ---------------------------------------------------------------------------

def solar_to_power_kw(irradiance_wm2: np.ndarray,
                       panel_efficiency: float = 0.18) -> np.ndarray:
    """
    Convert irradiance (W/m²) to solar panel output (kW).
    Panel area is back-calculated from PANEL_CAPACITY_KW at 1000 W/m².
    """
    panel_area_m2 = (PANEL_CAPACITY_KW * 1000.0) / (1000.0 * panel_efficiency)
    power_w = irradiance_wm2 * panel_efficiency * panel_area_m2
    return np.clip(power_w / 1000.0, 0.0, PANEL_CAPACITY_KW)


# ---------------------------------------------------------------------------
# Station load synthesis
# ---------------------------------------------------------------------------

def _generate_station_load(timestamps: pd.DatetimeIndex,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate hourly load per tier (kW).

    Tier 1: heating + medical + comms — nearly constant but with heating
            ramping up sharply in winter, slight hour-of-day variation.
    Tier 2: lab equipment — peaks during working hours (08–18), lower at night.
    Tier 3: kitchen + convenience — breakfast/lunch/dinner peaks.

    Returns (tier1_kw, tier2_kw, tier3_kw).
    """
    n = len(timestamps)
    doy = timestamps.day_of_year.values.astype(float)
    hour = timestamps.hour.values.astype(float)

    # --- Seasonal heating demand for Tier 1 ---
    # Winter (DOY 90-270) heating peaks at ~1.6x baseline; summer at ~1.0x
    winter_fraction = 0.5 * (1.0 - np.cos(np.radians(360.0 * (doy - 90.0) / 365.0)))
    # Peaks in mid-winter (DOY ~180), close to zero in summer
    heating_multiplier = 1.0 + 0.6 * np.clip(winter_fraction, 0.0, 1.0)

    tier1_base = TIER1_BASE_KW * heating_multiplier
    tier1_noise = rng.normal(0.0, 0.5, n)
    tier1 = np.clip(tier1_base + tier1_noise, TIER1_BASE_KW * 0.8, TIER1_BASE_KW * 2.0)

    # --- Tier 2: lab loads follow an 08–18 working-hours profile ---
    lab_profile = np.where((hour >= 8) & (hour < 18), 1.0, 0.25)
    tier2_base = TIER2_BASE_KW * lab_profile
    tier2_noise = rng.normal(0.0, 0.4, n)
    tier2 = np.clip(tier2_base + tier2_noise, 0.0, TIER2_BASE_KW * 1.5)

    # --- Tier 3: kitchen peaks at 07, 12, 19 (breakfast, lunch, dinner) ---
    breakfast = np.exp(-0.5 * ((hour - 7.0) / 0.8) ** 2)
    lunch = np.exp(-0.5 * ((hour - 12.0) / 0.8) ** 2)
    dinner = np.exp(-0.5 * ((hour - 19.0) / 0.8) ** 2)
    meal_profile = breakfast + lunch + dinner
    meal_profile = meal_profile / meal_profile.max()
    tier3_base = TIER3_BASE_KW * (0.4 + 0.6 * meal_profile)
    tier3_noise = rng.normal(0.0, 0.2, n)
    tier3 = np.clip(tier3_base + tier3_noise, 0.0, TIER3_BASE_KW * 1.5)

    return tier1, tier2, tier3


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_year(
    seed: int = 42,
    year: int = 2025,
    apply_wind_lull: bool = False,
    lull_start_day: int = 180,
    lull_duration_days: int = 5,
    apply_katabatic_stress: bool = False,
    katabatic_start_day: int = 150,
    katabatic_duration_days: int = 3,
) -> pd.DataFrame:
    """
    Generate one full year of hourly polar-station energy data.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    year : int
        Calendar year for the timestamp index.
    apply_wind_lull : bool
        If True, zero out wind during a specified multi-day window (stress test 2).
    lull_start_day : int
        Day-of-year when the wind lull begins.
    lull_duration_days : int
        Duration of the forced wind lull in days.
    apply_katabatic_stress : bool
        If True, inject a guaranteed severe katabatic event (stress test 1).
    katabatic_start_day : int
        Day-of-year when the katabatic event is injected.
    katabatic_duration_days : int
        Duration of the injected katabatic event in days.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, solar_irradiance_wm2, wind_speed_ms,
        solar_power_kw, wind_power_kw,
        tier1_load_kw, tier2_load_kw, tier3_load_kw, total_load_kw
    """
    rng = np.random.default_rng(seed)

    # Build hourly timestamp index for the full year
    start = pd.Timestamp(f"{year}-01-01 00:00")
    end = pd.Timestamp(f"{year}-12-31 23:00")
    timestamps = pd.date_range(start=start, end=end, freq="h")
    n = len(timestamps)

    # --- Weather ---
    irradiance = _generate_solar_irradiance(timestamps, rng)
    wind = _generate_wind_speed(timestamps, rng)

    # Stress-test overrides ---
    if apply_wind_lull:
        lull_start_h = (lull_start_day - 1) * 24
        lull_end_h = lull_start_h + lull_duration_days * 24
        wind[lull_start_h:lull_end_h] = rng.uniform(0.0, TURBINE_CUT_IN_MS * 0.5,
                                                      lull_end_h - lull_start_h)

    if apply_katabatic_stress:
        kat_start_h = (katabatic_start_day - 1) * 24
        kat_end_h = kat_start_h + katabatic_duration_days * 24
        kat_hours = kat_end_h - kat_start_h
        # Severe katabatic: ramp to 40 m/s then fall back
        peak = 40.0
        ramp = np.linspace(wind[kat_start_h], peak, kat_hours // 2)
        decay = np.linspace(peak, wind[min(kat_end_h, n - 1)], kat_hours - kat_hours // 2)
        wind[kat_start_h:kat_end_h] = np.concatenate([ramp, decay])[:kat_hours]

    # --- Power conversion ---
    solar_power = solar_to_power_kw(irradiance)
    wind_power = wind_to_power_kw(wind)

    # --- Loads ---
    tier1, tier2, tier3 = _generate_station_load(timestamps, rng)
    total_load = tier1 + tier2 + tier3

    df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_irradiance_wm2": irradiance,
        "wind_speed_ms": wind,
        "solar_power_kw": solar_power,
        "wind_power_kw": wind_power,
        "tier1_load_kw": tier1,
        "tier2_load_kw": tier2,
        "tier3_load_kw": tier3,
        "total_load_kw": total_load,
    })
    df["renewable_power_kw"] = df["solar_power_kw"] + df["wind_power_kw"]
    df["net_power_kw"] = df["renewable_power_kw"] - df["total_load_kw"]

    return df


def generate_historical_log(
    n_days: int = 90,
    seed: int = 99,
    year: int = 2024,
) -> pd.DataFrame:
    """
    Generate a historical consumption log (last n_days before the simulation year).
    Used to train the load forecaster.

    Returns the same schema as generate_year() but for a shorter historical window.
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp(f"{year}-10-03 00:00")   # ~90 days before Jan 1
    end = start + pd.Timedelta(hours=n_days * 24 - 1)
    timestamps = pd.date_range(start=start, end=end, freq="h")

    irradiance = _generate_solar_irradiance(timestamps, rng)
    wind = _generate_wind_speed(timestamps, rng)
    solar_power = solar_to_power_kw(irradiance)
    wind_power = wind_to_power_kw(wind)
    tier1, tier2, tier3 = _generate_station_load(timestamps, rng)
    total_load = tier1 + tier2 + tier3

    df = pd.DataFrame({
        "timestamp": timestamps,
        "solar_irradiance_wm2": irradiance,
        "wind_speed_ms": wind,
        "solar_power_kw": solar_power,
        "wind_power_kw": wind_power,
        "tier1_load_kw": tier1,
        "tier2_load_kw": tier2,
        "tier3_load_kw": tier3,
        "total_load_kw": total_load,
    })
    df["renewable_power_kw"] = df["solar_power_kw"] + df["wind_power_kw"]
    df["net_power_kw"] = df["renewable_power_kw"] - df["total_load_kw"]

    return df


def save_datasets(output_dir: str = "polar_energy_sim/data", **kwargs) -> dict[str, Path]:
    """
    Generate and save both the simulation-year dataset and the historical log to CSV.
    Returns a dict with paths to the saved files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    year_df = generate_year(**kwargs)
    hist_df = generate_historical_log()

    year_path = out / "simulation_year.csv"
    hist_path = out / "historical_log.csv"

    year_df.to_csv(year_path, index=False)
    hist_df.to_csv(hist_path, index=False)

    print(f"[DataGenerator] Saved simulation year  → {year_path}  ({len(year_df)} rows)")
    print(f"[DataGenerator] Saved historical log   → {hist_path}  ({len(hist_df)} rows)")

    return {"simulation_year": year_path, "historical_log": hist_path}


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    paths = save_datasets()
    df = pd.read_csv(paths["simulation_year"])
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nSeasonal averages (solar power kW):")
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    print(df.groupby("month")[["solar_power_kw", "wind_power_kw", "total_load_kw"]].mean().round(2))
