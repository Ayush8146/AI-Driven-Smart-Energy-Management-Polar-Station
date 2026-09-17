"""
Module 6: Fuel Tracking Module
================================
A simple, append-only ledger that records diesel consumption hour-by-hour
and provides comparison metrics against the naive on/off-threshold baseline.

Responsibilities:
  - Track cumulative diesel litres burned by the MPC controller.
  - Track cumulative diesel litres burned by the naive baseline (run in parallel).
  - Compute fuel savings, reduction percentage, and cost estimates.
  - Persist the ledger to CSV for post-run analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ledger entry
# ---------------------------------------------------------------------------

@dataclass
class FuelRecord:
    """One hour of fuel consumption data."""
    sim_hour: int
    timestamp: Optional[object]
    mpc_litres: float       # Diesel burned by MPC controller this hour
    naive_litres: float     # Diesel burned by naive baseline this hour
    mpc_gen_kw: float       # MPC generator output (kW)
    naive_gen_kw: float     # Naive generator output (kW)
    soc_mpc: float          # Battery SOC after MPC decision
    soc_naive: float        # Battery SOC after naive decision
    safety_triggered: bool  # Whether safety fallback fired


# ---------------------------------------------------------------------------
# FuelTracker
# ---------------------------------------------------------------------------

class FuelTracker:
    """
    Dual-ledger fuel consumption tracker.

    Receives one record per simulation hour for both the MPC and naive
    baseline controllers, then reports cumulative totals and savings.

    Usage
    -----
    >>> ft = FuelTracker()
    >>> ft.record(sim_hour=0, timestamp=ts, mpc_litres=0.0, naive_litres=17.5, ...)
    >>> report = ft.report()
    """

    def __init__(self):
        self._records: list[FuelRecord] = []
        self._mpc_cumulative = 0.0
        self._naive_cumulative = 0.0

    def record(
        self,
        sim_hour: int,
        timestamp,
        mpc_litres: float,
        naive_litres: float,
        mpc_gen_kw: float = 0.0,
        naive_gen_kw: float = 0.0,
        soc_mpc: float = 0.0,
        soc_naive: float = 0.0,
        safety_triggered: bool = False,
    ) -> None:
        """Append one hour of consumption data to the ledger."""
        self._mpc_cumulative += mpc_litres
        self._naive_cumulative += naive_litres

        self._records.append(FuelRecord(
            sim_hour=sim_hour,
            timestamp=timestamp,
            mpc_litres=mpc_litres,
            naive_litres=naive_litres,
            mpc_gen_kw=mpc_gen_kw,
            naive_gen_kw=naive_gen_kw,
            soc_mpc=soc_mpc,
            soc_naive=soc_naive,
            safety_triggered=safety_triggered,
        ))

    # ------------------------------------------------------------------
    # Cumulative tracking
    # ------------------------------------------------------------------

    @property
    def mpc_total_litres(self) -> float:
        return self._mpc_cumulative

    @property
    def naive_total_litres(self) -> float:
        return self._naive_cumulative

    @property
    def savings_litres(self) -> float:
        return self._naive_cumulative - self._mpc_cumulative

    @property
    def savings_pct(self) -> float:
        if self._naive_cumulative < 0.001:
            return 0.0
        return 100.0 * self.savings_litres / self._naive_cumulative

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self, diesel_price_per_litre: float = 3.50) -> dict:
        """
        Produce a summary report of fuel consumption and savings.

        Parameters
        ----------
        diesel_price_per_litre : float
            Approximate cost of diesel at a polar station (USD/litre).
            Default 3.50 reflects Antarctic logistics cost.

        Returns
        -------
        dict with full statistics.
        """
        if not self._records:
            return {"error": "No records logged yet."}

        df = self.to_dataframe()

        # Monthly breakdown
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            df["month"] = pd.to_datetime(df["timestamp"]).dt.month
            monthly = df.groupby("month")[["mpc_litres", "naive_litres"]].sum()
            monthly["savings_litres"] = monthly["naive_litres"] - monthly["mpc_litres"]
            monthly_dict = monthly.round(1).to_dict()
        else:
            monthly_dict = {}

        # Generator runtime hours
        mpc_gen_hours = int((df["mpc_gen_kw"] > 0.5).sum())
        naive_gen_hours = int((df["naive_gen_kw"] > 0.5).sum())

        return {
            "mpc_total_litres": round(self._mpc_cumulative, 1),
            "naive_total_litres": round(self._naive_cumulative, 1),
            "savings_litres": round(self.savings_litres, 1),
            "savings_pct": round(self.savings_pct, 1),
            "mpc_cost_usd": round(self._mpc_cumulative * diesel_price_per_litre, 2),
            "naive_cost_usd": round(self._naive_cumulative * diesel_price_per_litre, 2),
            "cost_savings_usd": round(self.savings_litres * diesel_price_per_litre, 2),
            "mpc_generator_hours": mpc_gen_hours,
            "naive_generator_hours": naive_gen_hours,
            "safety_override_count": int(df["safety_triggered"].sum()),
            "monthly_breakdown": monthly_dict,
            "simulation_hours": len(self._records),
        }

    # ------------------------------------------------------------------
    # DataFrame / CSV
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full ledger as a pandas DataFrame."""
        if not self._records:
            return pd.DataFrame()

        rows = []
        cum_mpc = 0.0
        cum_naive = 0.0
        for r in self._records:
            cum_mpc += r.mpc_litres
            cum_naive += r.naive_litres
            rows.append({
                "sim_hour": r.sim_hour,
                "timestamp": r.timestamp,
                "mpc_litres": r.mpc_litres,
                "naive_litres": r.naive_litres,
                "mpc_gen_kw": r.mpc_gen_kw,
                "naive_gen_kw": r.naive_gen_kw,
                "soc_mpc": r.soc_mpc,
                "soc_naive": r.soc_naive,
                "cum_mpc_litres": cum_mpc,
                "cum_naive_litres": cum_naive,
                "safety_triggered": r.safety_triggered,
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> Path:
        """Save the ledger to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(p, index=False)
        print(f"[FuelTracker] Ledger saved → {p}")
        return p

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_summary(self, diesel_price_per_litre: float = 3.50) -> None:
        r = self.report(diesel_price_per_litre)
        print("\n" + "=" * 60)
        print("  FUEL CONSUMPTION SUMMARY")
        print("=" * 60)
        print(f"  Simulation hours tracked : {r.get('simulation_hours', 0)}")
        print(f"  MPC controller           : {r['mpc_total_litres']:>10.1f} L  (${r['mpc_cost_usd']:,.2f})")
        print(f"  Naive baseline           : {r['naive_total_litres']:>10.1f} L  (${r['naive_cost_usd']:,.2f})")
        print(f"  Savings                  : {r['savings_litres']:>10.1f} L  ({r['savings_pct']:.1f}%)  ${r['cost_savings_usd']:,.2f}")
        print(f"  MPC generator hours      : {r['mpc_generator_hours']}")
        print(f"  Naive generator hours    : {r['naive_generator_hours']}")
        print(f"  Safety overrides         : {r['safety_override_count']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ft = FuelTracker()

    # Simulate 48 hours of data
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(7)
    ts = pd.date_range("2025-01-01", periods=48, freq="h")
    for h in range(48):
        mpc_l = rng.uniform(0, 5)
        naive_l = rng.uniform(5, 20)
        ft.record(
            sim_hour=h,
            timestamp=ts[h],
            mpc_litres=mpc_l,
            naive_litres=naive_l,
            mpc_gen_kw=mpc_l / 0.35 if mpc_l > 0 else 0,
            naive_gen_kw=naive_l / 0.35,
            soc_mpc=rng.uniform(0.3, 0.8),
            soc_naive=rng.uniform(0.3, 0.7),
            safety_triggered=(h == 20),
        )

    ft.print_summary()
    df = ft.to_dataframe()
    print(f"\nLedger shape: {df.shape}")
    print(df.tail(3).to_string(index=False))
