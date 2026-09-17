"""
run_simulation.py — Top-level entry point
==========================================
Run this file to execute all five scenarios (baseline + 4 stress tests)
and generate all plots.

Usage:
    python run_simulation.py                   # all scenarios + all plots
    python run_simulation.py --scenario baseline
    python run_simulation.py --scenario stress1_katabatic_winter
    python run_simulation.py --scenario stress2_wind_lull
    python run_simulation.py --scenario stress3_forecast_failure
    python run_simulation.py --scenario stress4_fuel_comparison
    python run_simulation.py --no-plots        # skip plot generation

Results are written to: polar_energy_sim/outputs/
"""

import argparse
import sys
from pathlib import Path

from polar_energy_sim.simulation import ScenarioConfig, run_simulation, run_all_scenarios
from polar_energy_sim.dashboard import (
    generate_all_plots,
    plot_full_year_dashboard,
    plot_stress1_katabatic,
    plot_stress2_wind_lull,
    plot_stress3_forecast_failure,
    plot_stress4_fuel_comparison,
)

OUTPUT_DIR = "polar_energy_sim/outputs"

SCENARIO_CONFIGS = {
    "baseline": ScenarioConfig(
        name="baseline",
        seed=42,
        output_dir=OUTPUT_DIR,
    ),
    "stress1_katabatic_winter": ScenarioConfig(
        name="stress1_katabatic_winter",
        seed=42,
        katabatic_stress=True,
        katabatic_start_day=150,
        katabatic_duration_days=3,
        output_dir=OUTPUT_DIR,
    ),
    "stress2_wind_lull": ScenarioConfig(
        name="stress2_wind_lull",
        seed=42,
        wind_lull=True,
        lull_start_day=180,
        lull_duration_days=5,
        output_dir=OUTPUT_DIR,
    ),
    "stress3_forecast_failure": ScenarioConfig(
        name="stress3_forecast_failure",
        seed=42,
        failure_injection=True,
        failure_start_hour=4320,
        failure_duration_hours=48,
        output_dir=OUTPUT_DIR,
    ),
    "stress4_fuel_comparison": ScenarioConfig(
        name="stress4_fuel_comparison",
        seed=42,
        output_dir=OUTPUT_DIR,
    ),
}


def run_single(name: str, plots: bool = True):
    if name not in SCENARIO_CONFIGS:
        print(f"Unknown scenario '{name}'. Choose from: {list(SCENARIO_CONFIGS.keys())}")
        sys.exit(1)

    cfg = SCENARIO_CONFIGS[name]
    result = run_simulation(cfg)
    result.save(OUTPUT_DIR)

    if plots:
        print(f"\n[Main] Generating plots for '{name}'...")
        if name == "baseline":
            plot_full_year_dashboard(result, OUTPUT_DIR)
            plot_stress4_fuel_comparison(result, OUTPUT_DIR)
        elif name == "stress1_katabatic_winter":
            plot_full_year_dashboard(result, OUTPUT_DIR)
            plot_stress1_katabatic(result, OUTPUT_DIR)
        elif name == "stress2_wind_lull":
            plot_full_year_dashboard(result, OUTPUT_DIR)
            plot_stress2_wind_lull(result, OUTPUT_DIR)
        elif name == "stress3_forecast_failure":
            plot_full_year_dashboard(result, OUTPUT_DIR)
            plot_stress3_forecast_failure(result, OUTPUT_DIR)
        elif name == "stress4_fuel_comparison":
            plot_full_year_dashboard(result, OUTPUT_DIR)
            plot_stress4_fuel_comparison(result, OUTPUT_DIR)

    return result


def run_all(plots: bool = True):
    print("=" * 64)
    print("  Polar Station Energy Management System — Full Run")
    print("=" * 64)

    results = {}
    for name, cfg in SCENARIO_CONFIGS.items():
        result = run_simulation(cfg)
        result.save(OUTPUT_DIR)
        results[name] = result

    if plots:
        print("\n[Main] Generating all plots...")
        saved_plots = generate_all_plots(results, OUTPUT_DIR)
        print(f"\n[Main] {len(saved_plots)} plots saved to {OUTPUT_DIR}/")

    # Final consolidated report
    print("\n" + "=" * 64)
    print("  FINAL CONSOLIDATED REPORT")
    print("=" * 64)
    for name, result in results.items():
        df = result.to_dataframe()
        tier1_drops = int((df["tier1_served_kw"] < df["tier1_load_kw"] - 0.5).sum())
        safety_count = result.safety_summary.get("total_overrides", 0)
        fuel = result.fuel_tracker.report()
        status = "✓ PASS" if tier1_drops == 0 else "✗ FAIL"
        print(
            f"  {status}  {name:<35}  "
            f"T1-drops={tier1_drops}  "
            f"safety={safety_count}  "
            f"MPC={fuel['mpc_total_litres']:>7.0f}L  "
            f"naive={fuel['naive_total_litres']:>7.0f}L  "
            f"saved={fuel['savings_pct']:.1f}%"
        )
    print("=" * 64)

    return results


def main():
    parser = argparse.ArgumentParser(description="Polar Energy Management System Simulator")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=list(SCENARIO_CONFIGS.keys()) + ["all"],
        help="Which scenario to run (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if args.scenario == "all":
        run_all(plots=not args.no_plots)
    else:
        run_single(args.scenario, plots=not args.no_plots)


if __name__ == "__main__":
    main()
