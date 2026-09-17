"""
Dashboard & Visualization
==========================
Produces all required plots from simulation results using matplotlib only.

Plots generated:
  1. Full-year overview dashboard (4 panels):
       - Battery SOC (MPC vs naive)
       - Load tiers served + shedding
       - Generator activation (on/off hours)
       - Cumulative fuel consumption (MPC vs naive)

  2. Stress test 1 — Katabatic winter window (zoomed 2-week view)
  3. Stress test 2 — Wind lull battery drawdown (zoomed 10-day view)
  4. Stress test 3 — Forecast failure + safety fallback engagement
  5. Stress test 4 — Fuel comparison bar chart + cumulative lines

All figures are saved as PNG to the outputs/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — works on headless systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from polar_energy_sim.simulation import SimulationResult


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4a",
    "axes.labelcolor":  "#c8cdd8",
    "xtick.color":      "#7a7f90",
    "ytick.color":      "#7a7f90",
    "text.color":       "#c8cdd8",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.6,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4a",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
}

COLORS = {
    "soc_mpc":      "#4fc3f7",
    "soc_naive":    "#ef9a9a",
    "tier1":        "#f44336",
    "tier2":        "#ff9800",
    "tier3":        "#4caf50",
    "shed2":        "#ff5722",
    "shed3":        "#ffc107",
    "gen_mpc":      "#ce93d8",
    "gen_naive":    "#ef9a9a",
    "fuel_mpc":     "#4fc3f7",
    "fuel_naive":   "#ef9a9a",
    "solar":        "#ffeb3b",
    "wind":         "#80cbc4",
    "safety":       "#ff1744",
    "soc_critical": "#ff1744",
    "soc_min":      "#ff6d00",
}


def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Dashboard] Saved → {path}")


def _ts_axis(ax, df: pd.DataFrame, col: str = "timestamp"):
    """Set timestamp x-axis formatting."""
    if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")


# ---------------------------------------------------------------------------
# Plot 1: Full-year overview dashboard
# ---------------------------------------------------------------------------

def plot_full_year_dashboard(
    result: SimulationResult,
    output_dir: str = "polar_energy_sim/outputs",
) -> Path:
    """4-panel full-year overview."""
    _apply_style()
    df = result.to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Downsample to daily means for readability over 8760 hours
    # Only average numeric columns to avoid issues with string fields
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    df_d = (
        df.set_index("timestamp")[numeric_cols]
        .resample("D").mean()
        .reset_index()
    )

    fig = plt.figure(figsize=(16, 12), facecolor=STYLE["figure.facecolor"])
    fig.suptitle(
        f"Polar Station Energy System — Full Year Overview\n"
        f"Scenario: {result.config.name}",
        fontsize=13, color="#e0e4f0", y=0.98
    )

    gs = GridSpec(4, 1, figure=fig, hspace=0.42)

    # ── Panel 1: Battery SOC ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_d["timestamp"], df_d["soc_after"],
             color=COLORS["soc_mpc"], lw=1.4, label="MPC Battery SOC")
    ax1.plot(df_d["timestamp"], df_d["soc_naive"],
             color=COLORS["soc_naive"], lw=1.0, alpha=0.7, label="Naive SOC")
    ax1.axhline(0.20, color=COLORS["soc_critical"], lw=1.0, ls="--", alpha=0.8, label="Critical SOC (20%)")
    ax1.axhline(0.15, color=COLORS["soc_min"], lw=0.8, ls=":", alpha=0.6, label="Hard minimum (15%)")
    ax1.fill_between(df_d["timestamp"], df_d["soc_after"], 0.20,
                     where=df_d["soc_after"] < 0.20,
                     color=COLORS["soc_critical"], alpha=0.25, label="_nolegend_")
    ax1.set_ylabel("State of Charge")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Battery State of Charge (MPC vs Naive Baseline)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(True)
    _ts_axis(ax1, df_d)

    # ── Panel 2: Load tiers served + shedding ────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(df_d["timestamp"], 0, df_d["tier1_served_kw"],
                     color=COLORS["tier1"], alpha=0.85, label="Tier 1 (heating/medical/comms)")
    ax2.fill_between(df_d["timestamp"], df_d["tier1_served_kw"],
                     df_d["tier1_served_kw"] + df_d["tier2_served_kw"],
                     color=COLORS["tier2"], alpha=0.75, label="Tier 2 (labs)")
    ax2.fill_between(df_d["timestamp"],
                     df_d["tier1_served_kw"] + df_d["tier2_served_kw"],
                     df_d["tier1_served_kw"] + df_d["tier2_served_kw"] + df_d["tier3_served_kw"],
                     color=COLORS["tier3"], alpha=0.65, label="Tier 3 (amenities)")
    # Shedding shown as red/orange bands at top
    shed_base = df_d["tier1_served_kw"] + df_d["tier2_served_kw"] + df_d["tier3_served_kw"]
    ax2.fill_between(df_d["timestamp"], shed_base, shed_base + df_d["shed_tier2_kw"],
                     color=COLORS["shed2"], alpha=0.7, label="Shed Tier 2")
    ax2.fill_between(df_d["timestamp"],
                     shed_base + df_d["shed_tier2_kw"],
                     shed_base + df_d["shed_tier2_kw"] + df_d["shed_tier3_kw"],
                     color=COLORS["shed3"], alpha=0.7, label="Shed Tier 3")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Load Tiers Served (stacked) — Shaded top = shedding")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.grid(True)
    _ts_axis(ax2, df_d)

    # ── Panel 3: Generator output ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(df_d["timestamp"], 0, df_d["gen_output_kw"],
                     color=COLORS["gen_mpc"], alpha=0.8, label="MPC Generator output")
    ax3.fill_between(df_d["timestamp"], 0, df_d["gen_output_naive_kw"],
                     color=COLORS["gen_naive"], alpha=0.35, label="Naive Generator output")
    ax3.set_ylabel("Generator output (kW)")
    ax3.set_title("Generator Activation — MPC (purple) vs Naive (pink overlay)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True)
    _ts_axis(ax3, df_d)

    # ── Panel 4: Cumulative fuel ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    fuel_df = result.fuel_tracker.to_dataframe()
    fuel_df["timestamp"] = pd.to_datetime(fuel_df["timestamp"])
    fuel_d = fuel_df.set_index("timestamp").resample("D").last().reset_index()
    ax4.plot(fuel_d["timestamp"], fuel_d["cum_mpc_litres"],
             color=COLORS["fuel_mpc"], lw=1.8, label="MPC cumulative fuel (L)")
    ax4.plot(fuel_d["timestamp"], fuel_d["cum_naive_litres"],
             color=COLORS["fuel_naive"], lw=1.4, alpha=0.8, label="Naive cumulative fuel (L)")
    ax4.fill_between(fuel_d["timestamp"],
                     fuel_d["cum_mpc_litres"], fuel_d["cum_naive_litres"],
                     color="#4fc3f7", alpha=0.15, label="Savings area")
    ax4.set_ylabel("Cumulative diesel (L)")
    ax4.set_title("Cumulative Fuel Consumption — MPC vs Naive Baseline")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.grid(True)
    _ts_axis(ax4, fuel_d)

    out = Path(output_dir) / f"dashboard_{result.config.name}.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 2: Stress Test 1 — Katabatic winter (zoomed)
# ---------------------------------------------------------------------------

def plot_stress1_katabatic(
    result: SimulationResult,
    output_dir: str = "polar_energy_sim/outputs",
    window_days: int = 14,
) -> Path:
    """Zoomed view around the katabatic event."""
    _apply_style()
    df = result.to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Centre the window on the katabatic start day
    kat_h = (result.config.katabatic_start_day - 1) * 24
    start_h = max(0, kat_h - 48)
    end_h = min(len(df), start_h + window_days * 24)
    dw = df.iloc[start_h:end_h].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              facecolor=STYLE["figure.facecolor"], sharex=True)
    fig.suptitle(
        "Stress Test 1 — Worst-Case Winter: Katabatic Wind Event\n"
        f"({window_days}-day window, starting day {result.config.katabatic_start_day})",
        fontsize=12, color="#e0e4f0"
    )

    # Wind speed + cut-out threshold
    axes[0].plot(dw["timestamp"], dw["wind_speed_ms"],
                 color=COLORS["wind"], lw=1.2, label="Wind speed (m/s)")
    axes[0].axhline(25.0, color="#ff6d00", lw=1.0, ls="--", label="Cut-out (25 m/s) — turbine shuts down")
    axes[0].fill_between(dw["timestamp"], 0, dw["wind_speed_ms"],
                          where=dw["wind_speed_ms"] >= 25.0,
                          color="#ff5722", alpha=0.35, label="Katabatic (cut-out zone)")
    axes[0].set_ylabel("Wind speed (m/s)")
    axes[0].set_title("Wind Speed")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # Wind + solar power output
    axes[1].fill_between(dw["timestamp"], 0, dw["wind_power_kw"],
                          color=COLORS["wind"], alpha=0.7, label="Wind power (kW)")
    axes[1].fill_between(dw["timestamp"], dw["wind_power_kw"],
                          dw["wind_power_kw"] + dw["solar_power_kw"],
                          color=COLORS["solar"], alpha=0.6, label="Solar power (kW)")
    axes[1].plot(dw["timestamp"], dw["total_load_kw"],
                 color="white", lw=1.0, ls="--", alpha=0.7, label="Total demand")
    axes[1].plot(dw["timestamp"], dw["tier1_load_kw"],
                 color=COLORS["tier1"], lw=1.0, ls=":", alpha=0.9, label="Tier 1 (critical)")
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_title("Generation vs Demand")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True)

    # SOC + generator
    ax3b = axes[2].twinx()
    axes[2].fill_between(dw["timestamp"], 0, dw["soc_after"],
                          color=COLORS["soc_mpc"], alpha=0.4)
    axes[2].plot(dw["timestamp"], dw["soc_after"],
                 color=COLORS["soc_mpc"], lw=1.5, label="Battery SOC")
    axes[2].axhline(0.20, color=COLORS["soc_critical"], lw=1.0, ls="--", alpha=0.8)
    axes[2].set_ylabel("Battery SOC", color=COLORS["soc_mpc"])
    axes[2].set_ylim(0, 1.0)
    ax3b.fill_between(dw["timestamp"], 0, dw["gen_output_kw"],
                       color=COLORS["gen_mpc"], alpha=0.55, label="Generator (kW)")
    ax3b.set_ylabel("Generator output (kW)", color=COLORS["gen_mpc"])

    # Safety trigger markers
    safety_rows = dw[dw["safety_triggered"]]
    if not safety_rows.empty:
        axes[2].scatter(safety_rows["timestamp"], [0.05] * len(safety_rows),
                        color=COLORS["safety"], zorder=5, s=40,
                        label="Safety fallback triggered", marker="v")

    axes[2].set_title("Battery SOC + Generator Activation (safety triggers = ▼)")
    lines_l, labels_l = axes[2].get_legend_handles_labels()
    lines_r, labels_r = ax3b.get_legend_handles_labels()
    axes[2].legend(lines_l + lines_r, labels_l + labels_r, fontsize=8)
    axes[2].grid(True)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    out = Path(output_dir) / "stress1_katabatic.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 3: Stress Test 2 — Wind lull battery drawdown
# ---------------------------------------------------------------------------

def plot_stress2_wind_lull(
    result: SimulationResult,
    output_dir: str = "polar_energy_sim/outputs",
    window_days: int = 12,
) -> Path:
    """Zoomed view around the wind lull + battery recovery."""
    _apply_style()
    df = result.to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    lull_h = (result.config.lull_start_day - 1) * 24
    start_h = max(0, lull_h - 24)
    end_h = min(len(df), start_h + window_days * 24)
    dw = df.iloc[start_h:end_h].copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                              facecolor=STYLE["figure.facecolor"], sharex=True)
    fig.suptitle(
        "Stress Test 2 — Multi-Day Winter Wind Lull: Battery Drawdown Test\n"
        f"(Lull starts day {result.config.lull_start_day}, "
        f"duration {result.config.lull_duration_days} days)",
        fontsize=12, color="#e0e4f0"
    )

    # Wind speed
    axes[0].fill_between(dw["timestamp"], 0, dw["wind_speed_ms"],
                          color=COLORS["wind"], alpha=0.6, label="Wind speed (m/s)")
    axes[0].axvline(
        dw["timestamp"].iloc[24] if len(dw) > 24 else dw["timestamp"].iloc[0],
        color="white", lw=0.8, ls="--", alpha=0.5, label="Lull begins"
    )
    axes[0].set_ylabel("Wind (m/s)")
    axes[0].set_title("Wind Speed — Lull clearly visible")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    # Renewable power vs demand
    axes[1].fill_between(dw["timestamp"], 0, dw["wind_power_kw"],
                          color=COLORS["wind"], alpha=0.65, label="Wind power")
    axes[1].fill_between(dw["timestamp"], dw["wind_power_kw"],
                          dw["wind_power_kw"] + dw["solar_power_kw"],
                          color=COLORS["solar"], alpha=0.55, label="Solar power")
    axes[1].plot(dw["timestamp"], dw["tier1_load_kw"],
                 color=COLORS["tier1"], lw=1.2, ls="--", label="Tier 1 (critical load)")
    axes[1].plot(dw["timestamp"], dw["total_load_kw"],
                 color="white", lw=0.9, ls=":", alpha=0.6, label="Total demand")
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_title("Renewable Generation vs Demand")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True)

    # Battery SOC — the core of this test
    axes[2].fill_between(dw["timestamp"], 0, dw["soc_after"],
                          color=COLORS["soc_mpc"], alpha=0.30)
    axes[2].plot(dw["timestamp"], dw["soc_after"],
                 color=COLORS["soc_mpc"], lw=2.0, label="Battery SOC (MPC)")
    axes[2].plot(dw["timestamp"], dw["soc_naive"],
                 color=COLORS["soc_naive"], lw=1.2, ls="--", alpha=0.7, label="Battery SOC (naive)")
    axes[2].axhline(0.20, color=COLORS["soc_critical"], lw=1.2, ls="--", label="Critical (20%)")
    axes[2].axhline(0.15, color=COLORS["soc_min"], lw=0.8, ls=":", label="Hard min (15%)")
    axes[2].fill_between(dw["timestamp"], 0.0, 0.20, color="#ff1744", alpha=0.08)
    axes[2].set_ylabel("SOC")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Battery Drawdown — SOC must not breach critical threshold")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].grid(True)

    # Generator + load shedding
    axes[3].fill_between(dw["timestamp"], 0, dw["gen_output_kw"],
                          color=COLORS["gen_mpc"], alpha=0.7, label="Generator (kW)")
    axes[3].fill_between(dw["timestamp"], 0, dw["shed_tier2_kw"] + dw["shed_tier3_kw"],
                          color=COLORS["shed2"], alpha=0.5, label="Load shed (T2+T3)")
    safety_rows = dw[dw["safety_triggered"]]
    if not safety_rows.empty:
        axes[3].scatter(safety_rows["timestamp"],
                        [2.0] * len(safety_rows),
                        color=COLORS["safety"], zorder=5, s=50,
                        label="Safety override", marker="^")
    axes[3].set_ylabel("kW")
    axes[3].set_title("Generator Activation & Load Shedding (safety overrides = ▲)")
    axes[3].legend(fontsize=8)
    axes[3].grid(True)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    out = Path(output_dir) / "stress2_wind_lull.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 4: Stress Test 3 — Forecast failure + safety fallback
# ---------------------------------------------------------------------------

def plot_stress3_forecast_failure(
    result: SimulationResult,
    output_dir: str = "polar_energy_sim/outputs",
    window_hours: int = 120,
) -> Path:
    """Shows the safety fallback engaging when the AI fails."""
    _apply_style()
    df = result.to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fail_h = result.config.failure_start_hour
    fail_end = fail_h + result.config.failure_duration_hours
    start_h = max(0, fail_h - 12)
    end_h = min(len(df), fail_h + window_hours)
    dw = df.iloc[start_h:end_h].copy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                              facecolor=STYLE["figure.facecolor"], sharex=True)
    fig.suptitle(
        "Stress Test 3 — Sensor/Forecast Failure: Safety Fallback Engagement\n"
        f"(Failure injected hours {fail_h}–{fail_end}, "
        f"duration {result.config.failure_duration_hours} h)",
        fontsize=12, color="#e0e4f0"
    )

    # Mark the failure window on all panels
    failure_start_ts = df["timestamp"].iloc[fail_h]
    failure_end_ts = df["timestamp"].iloc[min(fail_end, len(df) - 1)]

    def _shade_failure(ax):
        ax.axvspan(failure_start_ts, failure_end_ts,
                   color="#ff1744", alpha=0.12, label="AI failure window")

    # SOC
    _shade_failure(axes[0])
    axes[0].plot(dw["timestamp"], dw["soc_after"],
                 color=COLORS["soc_mpc"], lw=1.8, label="Battery SOC (MPC+Safety)")
    axes[0].axhline(0.20, color=COLORS["soc_critical"], lw=1.0, ls="--", label="Critical SOC")
    safety_rows = dw[dw["safety_triggered"]]
    if not safety_rows.empty:
        axes[0].scatter(safety_rows["timestamp"],
                        safety_rows["soc_after"],
                        color=COLORS["safety"], zorder=6, s=30,
                        label=f"Safety overrides ({len(safety_rows)} hours)", marker="o")
    axes[0].set_ylabel("SOC")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Battery SOC — Red dots = safety fallback active (AI bypassed)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True)

    # Generator output
    _shade_failure(axes[1])
    axes[1].fill_between(dw["timestamp"], 0, dw["gen_output_kw"],
                          color=COLORS["gen_mpc"], alpha=0.8, label="Generator output (kW)")
    axes[1].set_ylabel("Generator (kW)")
    axes[1].set_title("Generator Activation — Safety layer starts generator immediately on failure")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    # Tier 1 served — must never drop
    _shade_failure(axes[2])
    axes[2].fill_between(dw["timestamp"], 0, dw["tier1_served_kw"],
                          color=COLORS["tier1"], alpha=0.7, label="Tier 1 served (kW)")
    axes[2].plot(dw["timestamp"], dw["tier1_load_kw"],
                 color="white", lw=1.0, ls="--", alpha=0.6, label="Tier 1 demand")
    axes[2].fill_between(dw["timestamp"], 0, dw["shed_tier2_kw"] + dw["shed_tier3_kw"],
                          color=COLORS["shed2"], alpha=0.5, label="Shed T2+T3 (kW)")
    axes[2].set_ylabel("Power (kW)")
    axes[2].set_title("Tier 1 Always Served — Tier 2/3 shed during safety override")
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    out = Path(output_dir) / "stress3_forecast_failure.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Plot 5: Stress Test 4 — Fuel comparison
# ---------------------------------------------------------------------------

def plot_stress4_fuel_comparison(
    result: SimulationResult,
    output_dir: str = "polar_energy_sim/outputs",
) -> Path:
    """Bar + line chart comparing MPC vs naive fuel consumption."""
    _apply_style()
    fuel_df = result.fuel_tracker.to_dataframe()
    fuel_df["timestamp"] = pd.to_datetime(fuel_df["timestamp"])
    fuel_df["month"] = fuel_df["timestamp"].dt.month

    monthly = fuel_df.groupby("month")[["mpc_litres", "naive_litres"]].sum()
    monthly.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ][:len(monthly)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), facecolor=STYLE["figure.facecolor"]
    )
    fig.suptitle(
        "Stress Test 4 — Fuel Optimisation: MPC vs Naive Baseline",
        fontsize=13, color="#e0e4f0"
    )

    # ── Left: Monthly grouped bar chart ──────────────────────────────────
    x = np.arange(len(monthly))
    w = 0.38
    bars1 = ax1.bar(x - w / 2, monthly["naive_litres"], width=w,
                    color=COLORS["fuel_naive"], alpha=0.85, label="Naive baseline")
    bars2 = ax1.bar(x + w / 2, monthly["mpc_litres"], width=w,
                    color=COLORS["fuel_mpc"], alpha=0.85, label="MPC controller")

    # Annotate savings % per month
    for i, (naive_v, mpc_v) in enumerate(zip(monthly["naive_litres"], monthly["mpc_litres"])):
        if naive_v > 0:
            pct = 100 * (naive_v - mpc_v) / naive_v
            ax1.text(i, max(naive_v, mpc_v) + 20, f"{pct:.0f}%",
                     ha="center", va="bottom", fontsize=7.5,
                     color="#b0bec5")

    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly.index, fontsize=9)
    ax1.set_ylabel("Diesel consumed (litres)")
    ax1.set_title("Monthly Diesel Consumption")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="y")

    # Add totals text box
    r = result.fuel_tracker.report()
    textstr = (
        f"Full-year totals\n"
        f"Naive:  {r['naive_total_litres']:,.0f} L\n"
        f"MPC:    {r['mpc_total_litres']:,.0f} L\n"
        f"Saved:  {r['savings_litres']:,.0f} L  ({r['savings_pct']:.1f}%)\n"
        f"Cost Δ: ${r['cost_savings_usd']:,.0f}"
    )
    ax1.text(0.02, 0.97, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#2a2d3a", edgecolor="#4fc3f7", alpha=0.9),
             color="#e0e4f0")

    # ── Right: Cumulative lines ───────────────────────────────────────────
    fuel_d = fuel_df.set_index("timestamp").resample("D").last().reset_index()
    ax2.plot(fuel_d["timestamp"], fuel_d["cum_naive_litres"],
             color=COLORS["fuel_naive"], lw=2.0, label="Naive cumulative (L)")
    ax2.plot(fuel_d["timestamp"], fuel_d["cum_mpc_litres"],
             color=COLORS["fuel_mpc"], lw=2.0, label="MPC cumulative (L)")
    ax2.fill_between(fuel_d["timestamp"],
                     fuel_d["cum_mpc_litres"], fuel_d["cum_naive_litres"],
                     color="#4fc3f7", alpha=0.18, label="Cumulative savings")
    ax2.set_ylabel("Cumulative diesel (L)")
    ax2.set_title("Cumulative Fuel: MPC vs Naive (full year)")
    ax2.legend(fontsize=9)
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    # Annotate final saving at year-end
    if len(fuel_d):
        last = fuel_d.iloc[-1]
        mid_y = (last["cum_naive_litres"] + last["cum_mpc_litres"]) / 2
        ax2.annotate(
            f"  {r['savings_pct']:.1f}% saved\n  {r['savings_litres']:,.0f} L",
            xy=(last["timestamp"], mid_y),
            fontsize=9, color="#4fc3f7",
            arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=0.8),
            xytext=(-80, 20), textcoords="offset points",
        )

    out = Path(output_dir) / "stress4_fuel_comparison.png"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Generate all plots for a result set
# ---------------------------------------------------------------------------

def generate_all_plots(
    results: dict[str, "SimulationResult"],
    output_dir: str = "polar_energy_sim/outputs",
) -> list[Path]:
    """
    Generate all required plots from a dict of scenario results.
    Returns list of saved file paths.
    """
    saved = []

    if "baseline" in results:
        saved.append(plot_full_year_dashboard(results["baseline"], output_dir))

    if "stress1_katabatic_winter" in results:
        saved.append(plot_stress1_katabatic(results["stress1_katabatic_winter"], output_dir))
        saved.append(plot_full_year_dashboard(results["stress1_katabatic_winter"], output_dir))

    if "stress2_wind_lull" in results:
        saved.append(plot_stress2_wind_lull(results["stress2_wind_lull"], output_dir))

    if "stress3_forecast_failure" in results:
        saved.append(plot_stress3_forecast_failure(results["stress3_forecast_failure"], output_dir))

    if "stress4_fuel_comparison" in results:
        saved.append(plot_stress4_fuel_comparison(results["stress4_fuel_comparison"], output_dir))
    elif "baseline" in results:
        # Fuel comparison plot also works from baseline run
        saved.append(plot_stress4_fuel_comparison(results["baseline"], output_dir))

    return saved
