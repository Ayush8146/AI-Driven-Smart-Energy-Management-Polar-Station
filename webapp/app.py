"""
Polar Energy Management System — Web Dashboard
================================================
Professional Dash app with:
  - Sidebar navigation (5 pages)
  - User profile card
  - Overview: live KPI cards + full-year charts
  - Stress Tests: scenario selector + interactive multi-panel plots
  - Fuel Comparison: MPC vs naive monthly bar + cumulative lines
  - Forecast Viewer: 24-h generation & load forecast for any day
  - Safety Log: paginated event table with filters
  - Run Simulation: trigger any scenario from the browser with live log

Run:  python webapp/app.py
Then open http://127.0.0.1:8050
"""

import sys, os, json, subprocess, threading, time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Paths ────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
OUT    = ROOT / "polar_energy_sim" / "outputs"
SIM_PY = ROOT / "run_simulation.py"

SCENARIOS = {
    "baseline":                "Baseline (Full Year)",
    "stress1_katabatic_winter":"Stress 1 — Katabatic Winter",
    "stress2_wind_lull":       "Stress 2 — Wind Lull",
    "stress3_forecast_failure":"Stress 3 — Forecast Failure",
    "stress4_fuel_comparison": "Stress 4 — Fuel Comparison",
}

# ── Theme palette ────────────────────────────────────────────────────
T = dict(
    bg       = "#0b0f1a",
    surface  = "#111827",
    card     = "#1a2236",
    border   = "#243050",
    blue     = "#3b82f6",
    cyan     = "#22d3ee",
    green    = "#34d399",
    amber    = "#fbbf24",
    red      = "#f87171",
    purple   = "#a78bfa",
    text     = "#e2e8f8",
    muted    = "#506080",
    grid     = "#1e2a40",
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Inter, sans-serif", color=T["text"], size=12),
    margin        = dict(l=10, r=10, t=30, b=10),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis         = dict(gridcolor=T["grid"], linecolor=T["border"], tickfont=dict(size=10)),
    yaxis         = dict(gridcolor=T["grid"], linecolor=T["border"], tickfont=dict(size=10)),
    hoverlabel    = dict(bgcolor=T["card"], font=dict(size=12), bordercolor=T["border"]),
)

# ── Data helpers ─────────────────────────────────────────────────────

def _load(scenario: str, kind: str = "sim") -> pd.DataFrame:
    p = OUT / f"{kind}_{scenario}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["timestamp"])
    return df

def _daily(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns.tolist()
    return (df.set_index("timestamp")[num]
              .resample("D").mean()
              .reset_index())

def _monthly(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").columns.tolist()
    return (df.set_index("timestamp")[num]
              .resample("ME").sum()
              .reset_index())

# ── App init ─────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Polar EMS Dashboard",
)
server = app.server

# ── Sidebar ──────────────────────────────────────────────────────────

NAV = [
    ("overview",   "📊", "Overview",        "Mission Control"),
    ("stress",     "⚡", "Stress Tests",     "4 Scenarios"),
    ("fuel",       "🔥", "Fuel Comparison",  "MPC vs Naive"),
    ("forecast",   "🔭", "Forecast Viewer",  "24-h Ahead"),
    ("safety",     "🛡️", "Safety Log",       "Override Events"),
    ("run",        "▶️", "Run Simulation",   "Execute Scenarios"),
]

def sidebar():
    return html.Div(id="sidebar", children=[
        # Logo
        html.Div(className="sidebar-logo", children=[
            html.Div("🧊", className="logo-icon"),
            html.Div("Polar EMS", className="logo-title"),
            html.Div("Maitri Station · Antarctica", className="logo-sub"),
        ]),
        # Nav
        html.Div(className="sidebar-section-label", children="Navigation"),
        html.Div([
            dcc.Link(
                href=f"/{page}",
                className="nav-link-item",
                id=f"nav-{page}",
                children=[
                    html.Span(icon, className="nav-icon"),
                    html.Span(label),
                    html.Span(sub, style={"fontSize":"10px","color":"var(--text-muted)",
                                         "marginLeft":"auto","whiteSpace":"nowrap"})
                        if sub else None,
                ]
            )
            for page, icon, label, sub in NAV
        ]),
        # Footer / user card
        html.Div(className="sidebar-footer", children=[
            html.Div(className="user-card", children=[
                html.Div("SB", className="user-avatar"),
                html.Div([
                    html.Div("Sourabh B.", className="user-name"),
                    html.Div("Station Engineer", className="user-role"),
                ]),
                html.Div(className="user-status"),
            ]),
        ]),
    ])

# ── Top bar ──────────────────────────────────────────────────────────

def topbar(title, sub):
    return html.Div(id="topbar", children=[
        html.Div([
            html.Div(title, className="topbar-title"),
            html.Div(sub,   className="topbar-sub"),
        ]),
        html.Div(className="topbar-right", children=[
            html.Span("● System Online", className="topbar-badge"),
            html.Span(id="clock", className="topbar-time"),
            dcc.Interval(id="clock-interval", interval=1000),
        ]),
    ])

# ── KPI card helper ──────────────────────────────────────────────────

def kpi(icon, label, value, unit="", delta="", delta_class="neu", color="blue"):
    return html.Div(className=f"kpi-card {color}", children=[
        html.Div(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div([
            html.Span(value, className="kpi-value"),
            html.Span(unit,  className="kpi-unit") if unit else None,
        ]),
        html.Div(delta, className=f"kpi-delta {delta_class}") if delta else None,
    ])

# ── Plotly fig base ──────────────────────────────────────────────────

def base_fig(rows=1, cols=1, titles=None, shared_x=False, heights=None):
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles or [],
        shared_xaxes=shared_x,
        vertical_spacing=0.08,
        row_heights=heights,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    for ann in fig.layout.annotations:
        ann.font.color = T["text"]
        ann.font.size  = 12
    return fig

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════

def page_overview():
    df = _load("baseline")
    if df.empty:
        return html.Div("No simulation data found. Run the simulation first.", style={"color":T["muted"],"padding":"40px"})

    dd = _daily(df)
    last = dd.iloc[-1]
    fuel_df = _load("baseline", "fuel")
    total_mpc   = fuel_df["cum_mpc_litres"].iloc[-1]   if not fuel_df.empty else 0
    total_naive = fuel_df["cum_naive_litres"].iloc[-1] if not fuel_df.empty else 0
    savings_pct = 100*(total_naive-total_mpc)/total_naive if total_naive else 0

    gen_on  = int((df["gen_output_kw"] > 0.5).sum())
    shed2   = int((df["shed_tier2_kw"] > 0.1).sum())
    shed3   = int((df["shed_tier3_kw"] > 0.1).sum())
    safety  = int(df["safety_triggered"].sum())

    # ── SOC full-year chart
    fig_soc = base_fig()
    fig_soc.add_trace(go.Scatter(
        x=dd["timestamp"], y=dd["soc_after"],
        name="MPC Battery SOC", fill="tozeroy",
        fillcolor="rgba(59,130,246,.12)", line=dict(color=T["blue"], width=2),
    ))
    fig_soc.add_trace(go.Scatter(
        x=dd["timestamp"], y=dd["soc_naive"],
        name="Naive SOC", line=dict(color=T["red"], width=1.2, dash="dot"),
    ))
    fig_soc.add_hline(y=0.20, line_color=T["red"],  line_dash="dash", line_width=1,
                      annotation_text="Critical 20%", annotation_font_color=T["red"])
    fig_soc.update_yaxes(range=[0,1], tickformat=".0%", title="SOC")
    fig_soc.update_layout(title="Battery State of Charge — Full Year (MPC vs Naive)", height=280)

    # ── Generation stacked
    fig_gen = base_fig()
    fig_gen.add_trace(go.Scatter(
        x=dd["timestamp"], y=dd["wind_power_kw"],
        name="Wind", fill="tozeroy", stackgroup="g",
        fillcolor="rgba(34,211,238,.35)", line=dict(color=T["cyan"], width=1),
    ))
    fig_gen.add_trace(go.Scatter(
        x=dd["timestamp"], y=dd["solar_power_kw"],
        name="Solar", fill="tonexty", stackgroup="g",
        fillcolor="rgba(251,191,36,.25)", line=dict(color=T["amber"], width=1),
    ))
    fig_gen.add_trace(go.Scatter(
        x=dd["timestamp"], y=dd["gen_output_kw"],
        name="Diesel Generator", fill="tonexty", stackgroup="g",
        fillcolor="rgba(167,139,250,.25)", line=dict(color=T["purple"], width=1),
    ))
    fig_gen.add_trace(go.Scatter(
        x=dd["timestamp"], y=dd["total_load_kw"],
        name="Total Demand", line=dict(color="white", width=1.5, dash="dot"),
    ))
    fig_gen.update_yaxes(title="Power (kW)")
    fig_gen.update_layout(title="Energy Generation Mix vs Total Demand", height=280)

    # ── Load tiers
    fig_load = base_fig()
    fig_load.add_trace(go.Bar(x=dd["timestamp"], y=dd["tier1_served_kw"], name="Tier 1 (Critical)",
                              marker_color="rgba(248,113,113,.8)"))
    fig_load.add_trace(go.Bar(x=dd["timestamp"], y=dd["tier2_served_kw"], name="Tier 2 (Labs)",
                              marker_color="rgba(251,191,36,.7)"))
    fig_load.add_trace(go.Bar(x=dd["timestamp"], y=dd["tier3_served_kw"], name="Tier 3 (Amenities)",
                              marker_color="rgba(52,211,153,.6)"))
    fig_load.add_trace(go.Bar(x=dd["timestamp"], y=dd["shed_tier2_kw"]+dd["shed_tier3_kw"],
                              name="Shed (T2+T3)", marker_color="rgba(239,68,68,.5)"))
    fig_load.update_layout(barmode="stack", title="Daily Load Profile by Tier", height=260)
    fig_load.update_yaxes(title="Power (kW)")

    # ── Cumulative fuel
    if not fuel_df.empty:
        fd = fuel_df.set_index("timestamp").resample("D").last().reset_index()
        fig_fuel = base_fig()
        fig_fuel.add_trace(go.Scatter(
            x=fd["timestamp"], y=fd["cum_naive_litres"],
            name="Naive Baseline", fill="tozeroy",
            fillcolor="rgba(248,113,113,.1)", line=dict(color=T["red"], width=2),
        ))
        fig_fuel.add_trace(go.Scatter(
            x=fd["timestamp"], y=fd["cum_mpc_litres"],
            name="MPC Controller", fill="tozeroy",
            fillcolor="rgba(59,130,246,.15)", line=dict(color=T["blue"], width=2.5),
        ))
        fig_fuel.update_yaxes(title="Cumulative Diesel (L)")
        fig_fuel.update_layout(title="Cumulative Fuel Consumption", height=240)
    else:
        fig_fuel = go.Figure()

    return html.Div([
        html.Div(className="section-header", children=[
            html.Div("Mission Control Overview", className="section-title"),
            html.Div("Baseline simulation — full year · Maitri Station, 70.8°S", className="section-sub"),
        ]),
        # KPI row
        dbc.Row([
            dbc.Col(kpi("🔋","Battery SOC",f"{last['soc_after']*100:.1f}","%","Year-end daily mean","neu","blue"), md=2),
            dbc.Col(kpi("⚡","Fuel Saved",f"{savings_pct:.1f}","%",f"vs naive  {total_naive:,.0f}L burned","pos","green"), md=2),
            dbc.Col(kpi("🌬️","Generator Hours",f"{gen_on:,}","h",f"of 8 760 total hours","neu","purple"), md=2),
            dbc.Col(kpi("💧","MPC Diesel",f"{total_mpc:,.0f}","L",f"Naive: {total_naive:,.0f} L","pos","amber"), md=2),
            dbc.Col(kpi("🛡️","Safety Overrides",f"{safety}","",f"Tier 1 drops: 0","pos" if safety==0 else "neg","cyan"), md=2),
            dbc.Col(kpi("📉","Shed Hours (T2)",f"{shed2:,}","h",f"T3 shed: {shed3:,} h","neu","red"), md=2),
        ], className="g-3 mb-4"),
        # Charts
        dbc.Row([
            dbc.Col(html.Div(className="chart-card", children=[
                html.Div(className="chart-card-header", children=[
                    html.Div([html.Div("Battery SOC", className="chart-card-title"),
                              html.Div("Daily mean · MPC vs naive baseline", className="chart-card-sub")]),
                ]),
                dcc.Graph(figure=fig_soc, config={"displayModeBar":True,"displaylogo":False}),
            ]), md=12),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(html.Div(className="chart-card", children=[
                html.Div(className="chart-card-header", children=[
                    html.Div([html.Div("Generation Mix", className="chart-card-title"),
                              html.Div("Wind + Solar + Diesel vs total demand", className="chart-card-sub")]),
                ]),
                dcc.Graph(figure=fig_gen, config={"displayModeBar":True,"displaylogo":False}),
            ]), md=7),
            dbc.Col(html.Div(className="chart-card", children=[
                html.Div(className="chart-card-header", children=[
                    html.Div([html.Div("Cumulative Fuel", className="chart-card-title"),
                              html.Div("MPC savings vs naive baseline", className="chart-card-sub")]),
                ]),
                dcc.Graph(figure=fig_fuel, config={"displayModeBar":True,"displaylogo":False}),
            ]), md=5),
        ], className="g-3 mt-0"),
        dbc.Row([
            dbc.Col(html.Div(className="chart-card", children=[
                html.Div(className="chart-card-header", children=[
                    html.Div([html.Div("Load Tiers", className="chart-card-title"),
                              html.Div("Stacked daily — served vs shed", className="chart-card-sub")]),
                ]),
                dcc.Graph(figure=fig_load, config={"displayModeBar":True,"displaylogo":False}),
            ]), md=12),
        ], className="g-3 mt-0"),
    ])

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — STRESS TESTS
# ═══════════════════════════════════════════════════════════════════

STRESS_META = {
    "stress1_katabatic_winter": {
        "title": "Stress Test 1 — Katabatic Winter Wind Event",
        "desc":  "Wind ramps to 40 m/s → turbine cut-out for 72 h in deep polar winter (Day 150).",
        "window_days": 14, "center_day": 150,
    },
    "stress2_wind_lull": {
        "title": "Stress Test 2 — Multi-Day Wind Lull",
        "desc":  "5-day wind lull mid-June (Day 180). Battery drawdown test.",
        "window_days": 12, "center_day": 180,
    },
    "stress3_forecast_failure": {
        "title": "Stress Test 3 — AI/Forecast Failure",
        "desc":  "48-hour ML forecasting failure at hour 4320. Safety fallback must engage.",
        "window_days": 5, "center_day": 181,
    },
    "stress4_fuel_comparison": {
        "title": "Stress Test 4 — Fuel Optimisation",
        "desc":  "Full-year MPC vs naive baseline fuel consumption comparison.",
        "window_days": 365, "center_day": 182,
    },
}

def page_stress():
    return html.Div([
        html.Div(className="section-header", children=[
            html.Div("Stress Test Scenarios", className="section-title"),
            html.Div("Select a scenario to explore its detailed multi-panel results", className="section-sub"),
        ]),
        # Scenario pills
        html.Div([
            html.Span(STRESS_META[k]["title"].split("—")[0].strip(),
                      id=f"pill-{k}", className="scenario-pill",
                      **{"data-scenario": k})
            for k in STRESS_META
        ], style={"marginBottom":"24px"}),
        # Hidden store for selected scenario
        dcc.Store(id="selected-stress", data="stress1_katabatic_winter"),
        html.Div(id="stress-content"),
    ])

def build_stress_charts(scenario: str) -> html.Div:
    df = _load(scenario)
    if df.empty:
        return html.Div("Data not found — run this scenario first.", style={"color":T["muted"],"padding":"20px"})

    meta = STRESS_META[scenario]
    start_h = max(0, (meta["center_day"] - 2) * 24 - 48)
    if scenario == "stress4_fuel_comparison":
        dw = _daily(df)
        is_daily = True
    else:
        days = meta["window_days"]
        end_h = min(len(df), start_h + days * 24)
        dw = df.iloc[start_h:end_h].copy()
        is_daily = False

    # ── Panel A: Wind speed
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=["Wind Speed (m/s)",
                                        "Generation vs Demand (kW)",
                                        "Battery SOC",
                                        "Generator & Load Shedding"],
                        vertical_spacing=0.07)
    fig.update_layout(**PLOTLY_LAYOUT, height=780,
                      title=meta["title"])
    for ann in fig.layout.annotations:
        ann.font.color = T["text"]; ann.font.size = 11

    if not is_daily:
        # Wind
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["wind_speed_ms"],
                                 fill="tozeroy", fillcolor="rgba(34,211,238,.15)",
                                 line=dict(color=T["cyan"],width=1.5), name="Wind (m/s)"), row=1, col=1)
        fig.add_hline(y=25, line_color=T["red"], line_dash="dash", line_width=1,
                      annotation_text="Cut-out 25 m/s", annotation_font_color=T["red"], row=1, col=1)

        # Generation
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["wind_power_kw"],
                                 fill="tozeroy", fillcolor="rgba(34,211,238,.2)",
                                 line=dict(color=T["cyan"],width=1), name="Wind Power"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["solar_power_kw"]+dw["wind_power_kw"],
                                 fill="tonexty", fillcolor="rgba(251,191,36,.18)",
                                 line=dict(color=T["amber"],width=1), name="+ Solar"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["tier1_load_kw"],
                                 line=dict(color=T["red"],width=1.5,dash="dot"), name="Tier 1 (critical)"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["total_load_kw"],
                                 line=dict(color="white",width=1,dash="dot"), name="Total demand"), row=2, col=1)

        # SOC
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["soc_after"],
                                 fill="tozeroy", fillcolor="rgba(59,130,246,.15)",
                                 line=dict(color=T["blue"],width=2), name="MPC SOC"), row=3, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["soc_naive"],
                                 line=dict(color=T["red"],width=1,dash="dot"), name="Naive SOC"), row=3, col=1)
        fig.add_hline(y=0.20, line_color=T["red"], line_dash="dash", line_width=1, row=3, col=1)

        # Safety markers
        safety_rows = dw[dw["safety_triggered"]==True]
        if not safety_rows.empty:
            fig.add_trace(go.Scatter(x=safety_rows["timestamp"],
                                     y=[0.05]*len(safety_rows),
                                     mode="markers", marker=dict(color=T["red"],size=8,symbol="triangle-down"),
                                     name="Safety override"), row=3, col=1)

        # Gen + shedding
        fig.add_trace(go.Bar(x=dw["timestamp"], y=dw["gen_output_kw"],
                             marker_color="rgba(167,139,250,.7)", name="Generator (kW)"), row=4, col=1)
        fig.add_trace(go.Bar(x=dw["timestamp"], y=dw["shed_tier2_kw"]+dw["shed_tier3_kw"],
                             marker_color="rgba(251,191,36,.6)", name="Shed T2+T3 (kW)"), row=4, col=1)
        fig.update_layout(barmode="stack")
    else:
        # Fuel comparison daily view
        fuel_df = _load(scenario, "fuel")
        if not fuel_df.empty:
            fd = fuel_df.set_index("timestamp").resample("D").last().reset_index()
            fig.add_trace(go.Scatter(x=fd["timestamp"], y=fd["soc_mpc"],
                                     fill="tozeroy", fillcolor="rgba(59,130,246,.15)",
                                     line=dict(color=T["blue"],width=1.5), name="MPC SOC"), row=3, col=1)
            fig.add_trace(go.Scatter(x=fd["timestamp"], y=fd["soc_naive"],
                                     line=dict(color=T["red"],width=1,dash="dot"), name="Naive SOC"), row=3, col=1)
            fig.add_trace(go.Scatter(x=fd["timestamp"], y=fd["cum_mpc_litres"],
                                     fill="tozeroy", fillcolor="rgba(59,130,246,.12)",
                                     line=dict(color=T["blue"],width=2), name="MPC Cumulative (L)"), row=4, col=1)
            fig.add_trace(go.Scatter(x=fd["timestamp"], y=fd["cum_naive_litres"],
                                     fill="tozeroy", fillcolor="rgba(248,113,113,.1)",
                                     line=dict(color=T["red"],width=2), name="Naive Cumulative (L)"), row=4, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["wind_power_kw"],
                                 fill="tozeroy", fillcolor="rgba(34,211,238,.2)",
                                 line=dict(color=T["cyan"],width=1), name="Wind Power"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["solar_power_kw"],
                                 fill="tonexty", fillcolor="rgba(251,191,36,.15)",
                                 line=dict(color=T["amber"],width=1), name="Solar Power"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["gen_output_kw"],
                                 fill="tozeroy", fillcolor="rgba(167,139,250,.2)",
                                 line=dict(color=T["purple"],width=1), name="Generator"), row=2, col=1)
        fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["total_load_kw"],
                                 line=dict(color="white",width=1,dash="dot"), name="Total Demand"), row=2, col=1)

    fig.update_yaxes(row=3, col=1, range=[0,1], tickformat=".0%")

    # Metrics strip
    t1_drops = int((df["tier1_served_kw"] < df["tier1_load_kw"] - 0.5).sum())
    safety_n = int(df["safety_triggered"].sum())
    fuel_df2 = _load(scenario, "fuel")
    mpc_l  = fuel_df2["mpc_litres"].sum()   if not fuel_df2.empty else 0
    naive_l= fuel_df2["naive_litres"].sum() if not fuel_df2.empty else 0
    pct    = 100*(naive_l-mpc_l)/naive_l    if naive_l else 0

    metrics = dbc.Row([
        dbc.Col(html.Div([
            html.Div("Tier 1 Drops", className="kpi-label"),
            html.Div([html.Span(str(t1_drops), className="kpi-value",
                               style={"color":T["green"] if t1_drops==0 else T["red"],"fontSize":"22px"}),
                      html.Span(" hours", className="kpi-unit")]),
        ], className="kpi-card green"), md=3),
        dbc.Col(html.Div([
            html.Div("Safety Overrides", className="kpi-label"),
            html.Div([html.Span(str(safety_n), className="kpi-value",
                               style={"color":T["amber"],"fontSize":"22px"}),
                      html.Span(" triggers", className="kpi-unit")]),
        ], className="kpi-card amber"), md=3),
        dbc.Col(html.Div([
            html.Div("MPC Diesel", className="kpi-label"),
            html.Div([html.Span(f"{mpc_l:,.0f}", className="kpi-value",
                               style={"color":T["blue"],"fontSize":"22px"}),
                      html.Span(" L", className="kpi-unit")]),
        ], className="kpi-card blue"), md=3),
        dbc.Col(html.Div([
            html.Div("Fuel Saved", className="kpi-label"),
            html.Div([html.Span(f"{pct:.1f}", className="kpi-value",
                               style={"color":T["green"],"fontSize":"22px"}),
                      html.Span(" %", className="kpi-unit")]),
        ], className="kpi-card green"), md=3),
    ], className="g-3 mb-4")

    return html.Div([
        html.Div(className="chart-card", children=[
            html.Div(className="chart-card-header", children=[
                html.Div([
                    html.Div(meta["title"], className="chart-card-title"),
                    html.Div(meta["desc"],  className="chart-card-sub"),
                ]),
            ]),
            metrics,
            dcc.Graph(figure=fig, config={"displayModeBar":True,"displaylogo":False}),
        ]),
    ])

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — FUEL COMPARISON
# ═══════════════════════════════════════════════════════════════════

def page_fuel():
    fuel_df = _load("baseline", "fuel")
    if fuel_df.empty:
        return html.Div("No fuel data found.", style={"color":T["muted"],"padding":"40px"})

    fuel_df["month"] = fuel_df["timestamp"].dt.month
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = fuel_df.groupby("month")[["mpc_litres","naive_litres"]].sum().reset_index()
    monthly["month_name"]  = monthly["month"].apply(lambda m: MONTHS[m-1])
    monthly["savings"]     = monthly["naive_litres"] - monthly["mpc_litres"]
    monthly["savings_pct"] = 100 * monthly["savings"] / monthly["naive_litres"].replace(0,1)

    total_mpc   = fuel_df["cum_mpc_litres"].iloc[-1]
    total_naive = fuel_df["cum_naive_litres"].iloc[-1]
    savings_l   = total_naive - total_mpc
    savings_p   = 100 * savings_l / total_naive

    # Monthly bar
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=monthly["month_name"], y=monthly["naive_litres"],
                             name="Naive Baseline", marker_color="rgba(248,113,113,.8)",
                             text=[f"{v:,.0f}L" for v in monthly["naive_litres"]],
                             textposition="outside", textfont_size=9))
    fig_bar.add_trace(go.Bar(x=monthly["month_name"], y=monthly["mpc_litres"],
                             name="MPC Controller", marker_color="rgba(59,130,246,.85)",
                             text=[f"{v:,.0f}L" for v in monthly["mpc_litres"]],
                             textposition="outside", textfont_size=9))
    for i, row in monthly.iterrows():
        fig_bar.add_annotation(x=row["month_name"], y=max(row["naive_litres"], row["mpc_litres"])+200,
                               text=f"↓{row['savings_pct']:.0f}%", showarrow=False,
                               font=dict(color=T["green"], size=10))
    fig_bar.update_layout(**PLOTLY_LAYOUT, title="Monthly Diesel Consumption", barmode="group",
                          height=340, yaxis_title="Diesel (litres)")

    # Cumulative
    fd = fuel_df.set_index("timestamp").resample("D").last().reset_index()
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=fd["timestamp"], y=fd["cum_naive_litres"],
                                 name="Naive Baseline", fill="tozeroy",
                                 fillcolor="rgba(248,113,113,.08)",
                                 line=dict(color=T["red"],width=2.5)))
    fig_cum.add_trace(go.Scatter(x=fd["timestamp"], y=fd["cum_mpc_litres"],
                                 name="MPC Controller", fill="tozeroy",
                                 fillcolor="rgba(59,130,246,.12)",
                                 line=dict(color=T["blue"],width=2.5)))
    fig_cum.add_trace(go.Scatter(x=fd["timestamp"],
                                 y=fd["cum_naive_litres"]-fd["cum_mpc_litres"],
                                 name="Running Savings", fill="tozeroy",
                                 fillcolor="rgba(52,211,153,.15)",
                                 line=dict(color=T["green"],width=1.5,dash="dot")))
    fig_cum.update_layout(**PLOTLY_LAYOUT, title="Cumulative Fuel — Full Year",
                          height=300, yaxis_title="Litres")

    # Generator runtime pie
    mpc_off  = 8760 - int((fuel_df["mpc_gen_kw"]>0.5).sum())
    mpc_on   = 8760 - mpc_off
    naive_off= 8760 - int((fuel_df["naive_gen_kw"]>0.5).sum())
    naive_on = 8760 - naive_off
    fig_pie = make_subplots(rows=1, cols=2, specs=[[{"type":"pie"},{"type":"pie"}]],
                            subplot_titles=["MPC Generator Runtime","Naive Generator Runtime"])
    fig_pie.add_trace(go.Pie(labels=["Running","Off"],
                              values=[mpc_on, mpc_off],
                              marker_colors=[T["purple"],"rgba(36,48,80,.6)"],
                              hole=0.55, textfont_size=11), row=1, col=1)
    fig_pie.add_trace(go.Pie(labels=["Running","Off"],
                              values=[naive_on, naive_off],
                              marker_colors=[T["red"],"rgba(36,48,80,.6)"],
                              hole=0.55, textfont_size=11), row=1, col=2)
    fig_pie.update_layout(**PLOTLY_LAYOUT, title="Generator Runtime Distribution (hours / 8 760)", height=280)
    for ann in fig_pie.layout.annotations:
        ann.font.color = T["text"]; ann.font.size = 11

    return html.Div([
        html.Div(className="section-header", children=[
            html.Div("Fuel Optimisation Comparison", className="section-title"),
            html.Div("MPC controller vs naive on/off threshold baseline — full year", className="section-sub"),
        ]),
        dbc.Row([
            dbc.Col(html.Div(className="kpi-card green", children=[
                html.Div("🔥", className="kpi-icon"),
                html.Div("Total Savings", className="kpi-label"),
                html.Div([html.Span(f"{savings_l:,.0f}", className="kpi-value"), html.Span(" L", className="kpi-unit")]),
                html.Div(f"↓ {savings_p:.1f}% reduction vs naive", className="kpi-delta pos"),
            ]), md=3),
            dbc.Col(html.Div(className="kpi-card blue", children=[
                html.Div("💧", className="kpi-icon"),
                html.Div("MPC Total", className="kpi-label"),
                html.Div([html.Span(f"{total_mpc:,.0f}", className="kpi-value"), html.Span(" L", className="kpi-unit")]),
                html.Div(f"${total_mpc*3.5:,.0f} at $3.50/L", className="kpi-delta neu"),
            ]), md=3),
            dbc.Col(html.Div(className="kpi-card red", children=[
                html.Div("⚠️", className="kpi-icon"),
                html.Div("Naive Total", className="kpi-label"),
                html.Div([html.Span(f"{total_naive:,.0f}", className="kpi-value"), html.Span(" L", className="kpi-unit")]),
                html.Div(f"${total_naive*3.5:,.0f} at $3.50/L", className="kpi-delta neg"),
            ]), md=3),
            dbc.Col(html.Div(className="kpi-card amber", children=[
                html.Div("💰", className="kpi-icon"),
                html.Div("Cost Savings", className="kpi-label"),
                html.Div([html.Span(f"${savings_l*3.5:,.0f}", className="kpi-value")]),
                html.Div("Antarctic logistics pricing", className="kpi-delta neu"),
            ]), md=3),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(html.Div(className="chart-card",
                             children=[dcc.Graph(figure=fig_bar, config={"displaylogo":False})]), md=8),
            dbc.Col(html.Div(className="chart-card",
                             children=[dcc.Graph(figure=fig_pie, config={"displaylogo":False})]), md=4),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(html.Div(className="chart-card",
                             children=[dcc.Graph(figure=fig_cum, config={"displaylogo":False})]), md=12),
        ], className="g-3 mt-0"),
    ])

# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — FORECAST VIEWER
# ═══════════════════════════════════════════════════════════════════

def page_forecast():
    df = _load("baseline")
    if df.empty:
        return html.Div("No simulation data.", style={"color":T["muted"],"padding":"40px"})

    min_d = df["timestamp"].dt.date.min().isoformat()
    max_d = df["timestamp"].dt.date.max().isoformat()

    return html.Div([
        html.Div(className="section-header", children=[
            html.Div("24-Hour Forecast Viewer", className="section-title"),
            html.Div("Inspect modelled generation and load forecasts for any day in the simulation", className="section-sub"),
        ]),
        html.Div(className="chart-card", style={"marginBottom":"20px"}, children=[
            dbc.Row([
                dbc.Col([
                    html.Div("Select Date", className="kpi-label", style={"marginBottom":"8px"}),
                    dcc.DatePickerSingle(
                        id="forecast-date",
                        min_date_allowed=min_d,
                        max_date_allowed=max_d,
                        initial_visible_month=min_d,
                        date=min_d,
                        display_format="MMM DD, YYYY",
                        style={"colorScheme":"dark"},
                    ),
                ], md=4),
                dbc.Col([
                    html.Div("Scenario", className="kpi-label", style={"marginBottom":"8px"}),
                    dcc.Dropdown(
                        id="forecast-scenario",
                        options=[{"label":v,"value":k} for k,v in SCENARIOS.items()],
                        value="baseline",
                        clearable=False,
                        style={"background":T["card"],"color":T["text"],"border":f"1px solid {T['border']}"},
                    ),
                ], md=4),
                dbc.Col([
                    html.Div("Window", className="kpi-label", style={"marginBottom":"8px"}),
                    dcc.Slider(id="forecast-hours", min=24, max=72, step=24, value=24,
                               marks={24:"24h",48:"48h",72:"72h"},
                               tooltip={"placement":"bottom"}),
                ], md=4),
            ], className="g-3"),
        ]),
        html.Div(id="forecast-charts"),
    ])

def build_forecast_charts(date_str, scenario, hours):
    df = _load(scenario)
    if df.empty:
        return html.Div("No data.", style={"color":T["muted"]})

    try:
        target = pd.Timestamp(date_str)
    except Exception:
        return html.Div("Invalid date.", style={"color":T["muted"]})

    mask = df["timestamp"].dt.date == target.date()
    start_h = df[mask].index.min() if mask.any() else 0
    if pd.isna(start_h):
        start_h = 0
    dw = df.iloc[int(start_h): int(start_h) + int(hours)].copy()
    if dw.empty:
        return html.Div("No data for selected date.", style={"color":T["muted"]})

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Solar Power (kW)", "Wind Power (kW)",
                                        "Load by Tier (kW)", "Battery SOC"],
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    for ann in fig.layout.annotations:
        ann.font.color = T["text"]; ann.font.size = 11

    # Solar
    fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["solar_power_kw"],
                             fill="tozeroy", fillcolor="rgba(251,191,36,.2)",
                             line=dict(color=T["amber"],width=2), name="Solar"), row=1, col=1)
    # Wind
    fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["wind_power_kw"],
                             fill="tozeroy", fillcolor="rgba(34,211,238,.2)",
                             line=dict(color=T["cyan"],width=2), name="Wind"), row=1, col=2)
    # Loads stacked
    fig.add_trace(go.Bar(x=dw["timestamp"], y=dw["tier1_served_kw"],
                         name="Tier 1", marker_color="rgba(248,113,113,.8)"), row=2, col=1)
    fig.add_trace(go.Bar(x=dw["timestamp"], y=dw["tier2_served_kw"],
                         name="Tier 2", marker_color="rgba(251,191,36,.7)"), row=2, col=1)
    fig.add_trace(go.Bar(x=dw["timestamp"], y=dw["tier3_served_kw"],
                         name="Tier 3", marker_color="rgba(52,211,153,.6)"), row=2, col=1)
    fig.update_layout(barmode="stack")
    # SOC
    fig.add_trace(go.Scatter(x=dw["timestamp"], y=dw["soc_after"],
                             fill="tozeroy", fillcolor="rgba(59,130,246,.18)",
                             line=dict(color=T["blue"],width=2), name="SOC"), row=2, col=2)
    fig.add_hline(y=0.20, line_color=T["red"], line_dash="dash", line_width=1, row=2, col=2)
    fig.update_yaxes(row=2, col=2, range=[0,1], tickformat=".0%")

    # Stat cards for this window
    solar_mean = dw["solar_power_kw"].mean()
    wind_mean  = dw["wind_power_kw"].mean()
    gen_mean   = dw["gen_output_kw"].mean()
    soc_min    = dw["soc_after"].min()

    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(className="kpi-card amber", children=[
                html.Div("☀️ Solar Avg", className="kpi-label"),
                html.Div([html.Span(f"{solar_mean:.1f}", className="kpi-value",style={"fontSize":"20px"}),
                          html.Span(" kW",className="kpi-unit")]),
            ]), md=3),
            dbc.Col(html.Div(className="kpi-card cyan", children=[
                html.Div("🌬️ Wind Avg", className="kpi-label"),
                html.Div([html.Span(f"{wind_mean:.1f}", className="kpi-value",style={"fontSize":"20px"}),
                          html.Span(" kW",className="kpi-unit")]),
            ]), md=3),
            dbc.Col(html.Div(className="kpi-card purple", children=[
                html.Div("⚙️ Gen Avg", className="kpi-label"),
                html.Div([html.Span(f"{gen_mean:.1f}", className="kpi-value",style={"fontSize":"20px"}),
                          html.Span(" kW",className="kpi-unit")]),
            ]), md=3),
            dbc.Col(html.Div(className=f"kpi-card {'red' if soc_min<0.25 else 'green'}", children=[
                html.Div("🔋 Min SOC", className="kpi-label"),
                html.Div([html.Span(f"{soc_min*100:.1f}", className="kpi-value",style={"fontSize":"20px"}),
                          html.Span(" %",className="kpi-unit")]),
            ]), md=3),
        ], className="g-3 mb-3"),
        html.Div(className="chart-card",
                 children=[dcc.Graph(figure=fig, config={"displayModeBar":True,"displaylogo":False})]),
    ])

# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — SAFETY LOG
# ═══════════════════════════════════════════════════════════════════

def page_safety():
    rows = []
    for scen, label in SCENARIOS.items():
        df = _load(scen)
        if df.empty:
            continue
        safety_df = df[df["safety_triggered"]==True].copy()
        for _, r in safety_df.iterrows():
            rows.append({
                "Scenario": label,
                "Hour":     int(r["sim_hour"]),
                "Timestamp":str(r["timestamp"])[:16],
                "Reason":   str(r["safety_reason"]).replace("[SAFETY:","").replace("]",""),
                "SOC":      f"{r['soc_before']*100:.1f}%",
                "Generator":f"{r['gen_output_kw']:.0f} kW",
                "T1 Served":f"{r['tier1_served_kw']:.1f} kW",
                "T1 Demand":f"{r['tier1_load_kw']:.1f} kW",
            })
    if not rows:
        return html.Div([
            html.Div(className="section-header", children=[
                html.Div("Safety Override Log", className="section-title"),
            ]),
            html.Div(className="chart-card", style={"padding":"40px","textAlign":"center"}, children=[
                html.Div("✅", style={"fontSize":"48px","marginBottom":"12px"}),
                html.Div("No safety overrides recorded.", style={"color":T["green"],"fontSize":"15px","fontWeight":"600"}),
                html.Div("All scenarios ran within safe operating bounds.", style={"color":T["muted"],"marginTop":"6px"}),
            ]),
        ])

    log_df = pd.DataFrame(rows)
    total  = len(log_df)
    by_reason = log_df["Reason"].value_counts()

    # Timeline scatter
    df_all = pd.concat([_load(s) for s in SCENARIOS if not _load(s).empty])
    if not df_all.empty:
        safety_all = df_all[df_all["safety_triggered"]==True]
        fig_tl = go.Figure()
        for reason in safety_all["safety_reason"].unique():
            sub = safety_all[safety_all["safety_reason"]==reason]
            fig_tl.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["soc_before"],
                mode="markers",
                marker=dict(size=7, color=T["red"] if "SOC" in str(reason) else T["amber"],
                            opacity=0.8, symbol="circle"),
                name=str(reason)[:40],
                hovertemplate="Hour %{text}<br>SOC: %{y:.1%}<extra></extra>",
                text=sub["sim_hour"].astype(str),
            ))
        fig_tl.update_layout(**PLOTLY_LAYOUT,
                             title="Safety Override Events — SOC at Time of Trigger",
                             height=280, yaxis_title="SOC at trigger",
                             yaxis_tickformat=".0%")
    else:
        fig_tl = go.Figure()

    return html.Div([
        html.Div(className="section-header", children=[
            html.Div("Safety Override Log", className="section-title"),
            html.Div("All SAFETY_FALLBACK triggers across every scenario", className="section-sub"),
        ]),
        dbc.Row([
            dbc.Col(html.Div(className="kpi-card red", children=[
                html.Div("🛡️", className="kpi-icon"),
                html.Div("Total Overrides", className="kpi-label"),
                html.Div(html.Span(str(total), className="kpi-value")),
            ]), md=3),
        ] + [
            dbc.Col(html.Div(className="kpi-card amber", children=[
                html.Div(reason[:30], className="kpi-label"),
                html.Div(html.Span(str(count), className="kpi-value")),
            ]), md=3)
            for reason, count in by_reason.items()
        ], className="g-3 mb-4"),
        html.Div(className="chart-card mb-4",
                 children=[dcc.Graph(figure=fig_tl, config={"displaylogo":False})]),
        html.Div(className="chart-card", children=[
            html.Div(className="chart-card-header", children=[
                html.Div("Event Detail Table", className="chart-card-title"),
            ]),
            html.Div(style={"overflowX":"auto","maxHeight":"420px","overflowY":"auto"}, children=[
                html.Table(
                    [html.Thead(html.Tr([html.Th(c) for c in log_df.columns]))] +
                    [html.Tbody([
                        html.Tr([html.Td(str(row[c])) for c in log_df.columns])
                        for _, row in log_df.iterrows()
                    ])],
                    className="data-table",
                )
            ]),
        ]),
    ])

# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — RUN SIMULATION
# ═══════════════════════════════════════════════════════════════════

_run_log   = []
_run_state = {"running": False, "done": False, "progress": 0}

def page_run():
    return html.Div([
        html.Div(className="section-header", children=[
            html.Div("Run Simulation", className="section-title"),
            html.Div("Execute scenarios directly from the browser — results update all pages automatically", className="section-sub"),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(className="run-panel", children=[
                    html.Div("Select Scenario", className="kpi-label", style={"marginBottom":"10px"}),
                    dcc.Dropdown(
                        id="run-scenario-select",
                        options=[{"label":v,"value":k} for k,v in SCENARIOS.items()]
                                + [{"label":"▶  All Scenarios","value":"all"}],
                        value="all",
                        clearable=False,
                        style={"marginBottom":"18px","background":T["card"],
                               "color":T["text"],"border":f"1px solid {T['border']}"},
                    ),
                    html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap"}, children=[
                        html.Button("▶  Run", id="btn-run", n_clicks=0, className="btn-primary"),
                        html.Button("↺  Reset Log", id="btn-clear-log", n_clicks=0, className="btn-secondary"),
                    ]),
                    html.Div(className="progress-bar-wrap", children=[
                        html.Div(id="progress-fill", className="progress-bar-fill", style={"width":"0%"}),
                    ]),
                    html.Div(id="progress-label", children="Ready",
                             style={"fontSize":"11px","color":T["muted"],"marginBottom":"8px"}),
                    html.Div(id="run-log-box", className="log-box",
                             children="Waiting for run…"),
                    dcc.Interval(id="run-interval", interval=800, n_intervals=0, disabled=True),
                    dcc.Store(id="run-store", data={"running":False,"done":False,"progress":0,"log":[]}),
                ]),
            ], md=8),
            dbc.Col([
                html.Div(className="chart-card", children=[
                    html.Div("Scenario Guide", className="chart-card-title", style={"marginBottom":"14px"}),
                    html.Div([
                        html.Div(style={"marginBottom":"14px","paddingBottom":"14px",
                                        "borderBottom":f"1px solid {T['border']}"}, children=[
                            html.Div(v, style={"color":T["cyan"],"fontWeight":"600","fontSize":"12.5px"}),
                            html.Div(desc, style={"color":T["muted"],"fontSize":"11.5px","marginTop":"3px"}),
                        ])
                        for k, v in SCENARIOS.items()
                        for desc in [{
                            "baseline":                "Full-year baseline. No stress events.",
                            "stress1_katabatic_winter":"3-day katabatic wind (40 m/s) injected at day 150.",
                            "stress2_wind_lull":       "5-day wind lull injected at day 180 (deep winter).",
                            "stress3_forecast_failure":"48-hour AI/ML failure at hour 4320.",
                            "stress4_fuel_comparison": "Full-year MPC vs naive fuel tracking.",
                        }[k]]
                    ]),
                ]),
            ], md=4),
        ], className="g-3"),
    ])

# ═══════════════════════════════════════════════════════════════════
# ROOT LAYOUT
# ═══════════════════════════════════════════════════════════════════

app.layout = html.Div(id="app-shell", children=[
    dcc.Location(id="url", refresh=False),
    sidebar(),
    html.Div(id="main-content", children=[
        html.Div(id="topbar-container"),
        html.Div(id="page-content"),
    ]),
])

# ═══════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════

# ── Clock ────────────────────────────────────────────────────────────
@app.callback(Output("clock","children"), Input("clock-interval","n_intervals"))
def update_clock(_):
    return datetime.now().strftime("UTC+0  %H:%M:%S")

# ── Page routing ──────────────────────────────────────────────────────
PAGE_MAP = {
    "/":         ("overview",  "Overview",       "Mission Control · Full-Year Baseline"),
    "/overview": ("overview",  "Overview",       "Mission Control · Full-Year Baseline"),
    "/stress":   ("stress",    "Stress Tests",   "4 Extreme Scenarios · Interactive Explorer"),
    "/fuel":     ("fuel",      "Fuel Comparison","MPC vs Naive Baseline · Full Year"),
    "/forecast": ("forecast",  "Forecast Viewer","24-h Generation & Load · Any Day"),
    "/safety":   ("safety",    "Safety Log",     "Override Events · All Scenarios"),
    "/run":      ("run",       "Run Simulation", "Execute Scenarios · Live Progress"),
}

@app.callback(
    Output("topbar-container", "children"),
    Output("page-content",     "children"),
    Input("url", "pathname"),
)
def route(pathname):
    key = pathname or "/"
    page_id, title, sub = PAGE_MAP.get(key, PAGE_MAP["/"])
    tb = topbar(title, sub)
    pages = {
        "overview": page_overview,
        "stress":   page_stress,
        "fuel":     page_fuel,
        "forecast": page_forecast,
        "safety":   page_safety,
        "run":      page_run,
    }
    content = pages[page_id]()
    return tb, content

# ── Active nav highlight ──────────────────────────────────────────────
@app.callback(
    [Output(f"nav-{p[0]}","className") for p in NAV],
    Input("url","pathname"),
)
def highlight_nav(pathname):
    active = (pathname or "/").lstrip("/") or "overview"
    return [
        "nav-link-item active" if p[0] == active else "nav-link-item"
        for p in NAV
    ]

# ── Stress scenario pills ────────────────────────────────────────────
@app.callback(
    Output("selected-stress","data"),
    Output("stress-content","children"),
    [Input(f"pill-{k}","n_clicks") for k in STRESS_META],
    State("selected-stress","data"),
    prevent_initial_call=False,
)
def select_stress(*args):
    store = args[-1] or "stress1_katabatic_winter"
    triggered = ctx.triggered_id
    if triggered and triggered.startswith("pill-"):
        store = triggered[5:]
    return store, build_stress_charts(store)

# ── Forecast date/scenario change ────────────────────────────────────
@app.callback(
    Output("forecast-charts","children"),
    Input("forecast-date","date"),
    Input("forecast-scenario","value"),
    Input("forecast-hours","value"),
)
def update_forecast(date, scenario, hours):
    return build_forecast_charts(date, scenario or "baseline", hours or 24)

# ── Run simulation ────────────────────────────────────────────────────
@app.callback(
    Output("run-interval","disabled"),
    Output("run-store","data"),
    Input("btn-run","n_clicks"),
    State("run-scenario-select","value"),
    State("run-store","data"),
    prevent_initial_call=True,
)
def start_run(n_clicks, scenario, store):
    if not n_clicks or store.get("running"):
        return True, store

    store = {"running":True,"done":False,"progress":0,"log":["▶ Starting simulation…"]}

    def _run():
        try:
            arg = [] if scenario == "all" else ["--scenario", scenario]
            cmd = [sys.executable, str(SIM_PY)] + arg
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONIOENCODING":"utf-8"},
            )
            lines = []
            progress = 5
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                lines.append(line)
                if "Running" in line:       progress = 20
                if "h=1000" in line:        progress = 40
                if "h=4000" in line:        progress = 65
                if "h=7000" in line:        progress = 85
                if "PASS" in line or "Saved" in line: progress = 95
                if "plots saved" in line:   progress = 98
                _run_state.update({"running":True,"done":False,
                                   "progress":progress,"log":lines[-30:]})
            proc.wait()
            lines.append("✅ Simulation complete — refresh pages to see updated data.")
            _run_state.update({"running":False,"done":True,
                               "progress":100,"log":lines[-30:]})
        except Exception as e:
            _run_state.update({"running":False,"done":True,"progress":0,
                               "log":[f"❌ Error: {e}"]})

    threading.Thread(target=_run, daemon=True).start()
    return False, store

@app.callback(
    Output("run-log-box",    "children"),
    Output("progress-fill",  "style"),
    Output("progress-label", "children"),
    Output("run-store",      "data", allow_duplicate=True),
    Output("run-interval",   "disabled", allow_duplicate=True),
    Input("run-interval",    "n_intervals"),
    State("run-store",       "data"),
    prevent_initial_call=True,
)
def poll_run(_, store):
    state = _run_state
    lines = state.get("log", [])
    pct   = state.get("progress", 0)
    done  = state.get("done", False)
    running = state.get("running", False)

    log_children = [html.Div(l) for l in lines] if lines else [html.Div("Waiting…")]
    bar_style = {"width": f"{pct}%"}
    label = f"Running… {pct}%" if running else ("Complete ✅" if done else "Ready")
    new_store = {**store, "running": running, "done": done, "progress": pct}
    disable_interval = not running

    return log_children, bar_style, label, new_store, disable_interval

@app.callback(
    Output("run-log-box",  "children", allow_duplicate=True),
    Output("progress-fill","style",    allow_duplicate=True),
    Output("progress-label","children",allow_duplicate=True),
    Input("btn-clear-log", "n_clicks"),
    prevent_initial_call=True,
)
def clear_log(_):
    _run_state.update({"running":False,"done":False,"progress":0,"log":[]})
    return [html.Div("Log cleared.")], {"width":"0%"}, "Ready"

# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*56)
    print("  Polar EMS Dashboard")
    print("  http://127.0.0.1:8050")
    print("="*56 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
