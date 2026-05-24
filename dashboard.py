from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_CONFIG, FUEL_TARGET_COLUMNS, SHOP_TARGET_COLUMNS, TARGET_COLUMNS
from data_preparation import load_source_data, prepare_model_frame, validate_source_data
from predict import make_demo_forecast, model_artifacts_exist, predict_with_saved_model
from recommendations import build_recommendations


PAGE_TITLE = "TFT-анализ спроса АЗС"
# Main defense view is focused on forecast, comparative analysis, and recommendations.
SHOW_OVERVIEW_TAB = False
SHOW_FACTOR_TABS = False
SHOW_TRAINING_TAB = True
SHOW_PREFLIGHT_SUMMARY = False
PROJECT_AUTHOR = os.getenv("TFT_PROJECT_AUTHOR", "Журавлев Данил Артемович")
PROJECT_COPYRIGHT = f"Авторская проектная работа: {PROJECT_AUTHOR}. Использование без указания автора запрещено."
ACCENT = "#20E3B2"
BLUE = "#5B8CFF"
AMBER = "#FFB020"
PINK = "#FF5C7A"
VIOLET = "#9B8CFF"
TEXT = "#E6EDF7"
MUTED = "#8A96A8"
COMPETITOR_PRICE_PAIRS = {
    "AI92": ("price_AI92", "competitor_price_AI92"),
    "AI95": ("price_AI95", "competitor_price_AI95"),
    "ДТ EURO": ("price_DT_EURO", "competitor_price_DT"),
    "ДТ TANEKO": ("price_DT_TANEKO", "competitor_price_DT"),
    "ДТ SUMMER": ("price_DT_SUMMER", "competitor_price_DT"),
    "ДТ WINTER": ("price_DT_WINTER", "competitor_price_DT"),
}
TRAFFIC_LABELS = {
    "traffic_Passengers_cars": "Легковые",
    "traffic_Truck_short": "Короткие грузовики",
    "traffic_Truck": "Грузовики",
    "traffic_Truck_long": "Длинные грузовики",
    "traffic_Transporter": "Транспортеры",
    "traffic_Undefined": "Не определено",
    "total_traffic": "Весь трафик",
}
WEEKDAY_LABELS = {
    0: "Пн",
    1: "Вт",
    2: "Ср",
    3: "Чт",
    4: "Пт",
    5: "Сб",
    6: "Вс",
}
BINARY_FILTER_OPTIONS = ["Любое", "Да", "Нет"]
TRAINING_RUNS_DIR = DEFAULT_CONFIG.artifacts_dir / "training_runs"
TRAINING_UPLOADS_DIR = DEFAULT_CONFIG.artifacts_dir / "training_uploads"


st.set_page_config(page_title=PAGE_TITLE, page_icon="⛽", layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080B12;
            --surface: #0E1420;
            --elevated: #131B2A;
            --border: rgba(255,255,255,.08);
            --text: #E6EDF7;
            --muted: #8A96A8;
            --accent: #20E3B2;
            --blue: #5B8CFF;
            --amber: #FFB020;
            --pink: #FF5C7A;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 18% 0%, rgba(32, 227, 178, .10), transparent 26rem),
                linear-gradient(135deg, #080B12 0%, #0B1020 45%, #080B12 100%) !important;
            color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(10,14,24,.98) 0%, rgba(8,11,18,.98) 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--muted) !important;
        }
        [data-testid="block-container"] {
            padding: 1.15rem 2rem 2.5rem;
            max-width: 1600px;
        }
        h1, h2, h3, p, label, span, div { letter-spacing: 0 !important; }
        h1 { font-size: 2.35rem !important; font-weight: 780 !important; margin-bottom: .2rem !important; }
        h2 { font-size: 1.35rem !important; font-weight: 720 !important; }
        h3 { font-size: 1rem !important; font-weight: 680 !important; color: var(--text) !important; }
        .sidebar-brand {
            border: 1px solid var(--border);
            border-radius: .85rem;
            padding: 1rem;
            margin: .2rem 0 1rem;
            background:
                linear-gradient(135deg, rgba(32,227,178,.13), transparent 45%),
                linear-gradient(180deg, rgba(19,27,42,.94), rgba(10,14,24,.94));
            box-shadow: 0 20px 52px rgba(0,0,0,.30);
        }
        .brand-mark {
            width: 2.1rem;
            height: 2.1rem;
            display: grid;
            place-items: center;
            border-radius: .62rem;
            background: linear-gradient(135deg, var(--accent), var(--blue));
            color: #061018;
            font-weight: 900;
            margin-bottom: .75rem;
        }
        .brand-name { font-size: 1rem; font-weight: 790; color: var(--text); }
        .brand-caption { color: var(--muted); font-size: .78rem; margin-top: .24rem; line-height: 1.35; }
        .author-mark {
            margin-top: .75rem;
            padding-top: .7rem;
            border-top: 1px solid rgba(255,255,255,.08);
            color: var(--accent);
            font-size: .78rem;
            font-weight: 720;
            line-height: 1.35;
        }
        .top-shell {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: stretch;
            margin-bottom: 1rem;
        }
        .hero-panel {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,.10);
            border-radius: .9rem;
            padding: 1.15rem 1.25rem;
            background:
                linear-gradient(90deg, rgba(19,27,42,.96), rgba(14,20,32,.76)),
                repeating-linear-gradient(90deg, rgba(255,255,255,.035) 0 1px, transparent 1px 92px);
            box-shadow: 0 24px 70px rgba(0,0,0,.34);
        }
        .hero-panel:after {
            content: "";
            position: absolute;
            inset: auto 0 0 0;
            height: 2px;
            background: linear-gradient(90deg, var(--accent), var(--blue), var(--amber), transparent);
            opacity: .9;
        }
        .hero-kicker {
            color: var(--accent);
            font-size: .72rem;
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: .38rem;
        }
        .command-stack {
            display: grid;
            gap: .55rem;
            min-width: 18rem;
        }
        .command-chip {
            border: 1px solid var(--border);
            border-radius: .65rem;
            padding: .7rem .8rem;
            background: rgba(19,27,42,.78);
            box-shadow: 0 18px 50px rgba(0,0,0,.23);
        }
        .chip-label { color: var(--muted); font-size: .72rem; text-transform: uppercase; font-weight: 720; }
        .chip-value { color: var(--text); font-size: .96rem; font-weight: 760; margin-top: .14rem; }
        .accent-line {
            height: 1px;
            margin: .9rem 0 1rem;
            background: linear-gradient(90deg, transparent, rgba(32,227,178,.65), rgba(91,140,255,.55), transparent);
        }
        .subtitle { color: var(--muted); font-size: .95rem; margin-top: .25rem; }
        .ownership-note {
            margin-top: .85rem;
            color: var(--accent);
            font-size: .82rem;
            font-weight: 720;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            padding: .56rem .72rem;
            border: 1px solid var(--border);
            border-radius: .65rem;
            background: rgba(19, 27, 42, .88);
            box-shadow: 0 18px 50px rgba(0,0,0,.25);
            color: var(--text);
            font-size: .84rem;
            white-space: nowrap;
        }
        .dot { width: .55rem; height: .55rem; border-radius: 999px; background: var(--accent); box-shadow: 0 0 18px var(--accent); }
        .dot.off { background: var(--amber); box-shadow: 0 0 18px rgba(255,176,32,.55); }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .85rem;
            margin: .4rem 0 1rem;
        }
        .kpi-card, .glass-card, .recommendation-card {
            border: 1px solid var(--border);
            border-radius: .78rem;
            background:
                linear-gradient(180deg, rgba(19, 27, 42, .95), rgba(14, 20, 32, .84));
            box-shadow: 0 18px 48px rgba(0,0,0,.24);
            backdrop-filter: blur(16px);
        }
        .kpi-card {
            position: relative;
            overflow: hidden;
            padding: 1rem 1.05rem;
            min-height: 6.6rem;
        }
        .kpi-card:before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: linear-gradient(180deg, var(--accent), var(--blue));
        }
        .kpi-card:after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 6rem;
            height: 6rem;
            background: linear-gradient(135deg, rgba(32,227,178,.18), transparent 62%);
            pointer-events: none;
        }
        .kpi-label { color: var(--muted); font-size: .76rem; text-transform: uppercase; font-weight: 650; }
        .kpi-value { color: var(--text); font-size: 1.72rem; font-weight: 820; margin-top: .4rem; }
        .kpi-delta { color: var(--accent); font-size: .82rem; margin-top: .28rem; }
        .glass-card { padding: 1rem; margin-bottom: .85rem; }
        .recommendation-card { padding: .9rem 1rem; margin-bottom: .75rem; }
        .rec-title { font-weight: 720; color: var(--text); }
        .rec-body { color: var(--muted); font-size: .9rem; margin-top: .25rem; }
        .rec-metric { float: right; color: var(--accent); font-weight: 760; }
        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--border);
            border-radius: .85rem;
            background:
                linear-gradient(180deg, rgba(19,27,42,.78), rgba(10,14,24,.70));
            box-shadow: 0 20px 52px rgba(0,0,0,.20);
            padding: .55rem .55rem .2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: .32rem;
            background: rgba(14, 20, 32, .88);
            border: 1px solid var(--border);
            border-radius: .85rem;
            padding: .34rem;
            margin-bottom: .85rem;
            box-shadow: 0 18px 48px rgba(0,0,0,.18);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: .62rem;
            color: var(--muted);
            padding: .65rem .95rem;
            font-weight: 720;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(32, 227, 178, .18), rgba(91,140,255,.12));
            color: var(--text);
            box-shadow: inset 0 0 0 1px rgba(32,227,178,.20);
        }
        [data-testid="stMetric"], [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: .78rem;
            background: rgba(14, 20, 32, .72);
            padding: .75rem;
        }
        div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label, div[data-testid="stSlider"] label {
            color: var(--muted) !important;
            font-weight: 650;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-baseweb="popover"] {
            background: rgba(19,27,42,.92) !important;
            border-color: rgba(255,255,255,.10) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] .stButton button,
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            border-radius: .65rem !important;
            border: 1px solid var(--border) !important;
            background: rgba(19,27,42,.82) !important;
            color: var(--text) !important;
        }
        .stDataFrame div { color: var(--text); }
        .footer-author {
            margin-top: 1.25rem;
            padding: .8rem 1rem;
            border: 1px solid var(--border);
            border-radius: .65rem;
            background: rgba(14, 20, 32, .60);
            color: var(--muted);
            font-size: .82rem;
        }
        @media (max-width: 1100px) {
            .top-shell { grid-template-columns: 1fr; }
            .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 640px) {
            [data-testid="block-container"] { padding: 1rem; }
            .kpi-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_template() -> dict:
    return {
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"color": TEXT, "family": "Inter, Segoe UI, sans-serif"},
            "colorway": [ACCENT, BLUE, AMBER, PINK, VIOLET],
            "margin": {"l": 20, "r": 20, "t": 44, "b": 24},
            "hovermode": "x unified",
            "xaxis": {"gridcolor": "rgba(255,255,255,.06)", "zerolinecolor": "rgba(255,255,255,.08)"},
            "yaxis": {"gridcolor": "rgba(255,255,255,.06)", "zerolinecolor": "rgba(255,255,255,.08)"},
            "legend": {"orientation": "h", "y": 1.08, "x": 0},
        }
    }


@st.cache_data(show_spinner="Загружаем данные сети АЗС...")
def load_dashboard_data(data_path: str, metadata_path: str) -> tuple[pd.DataFrame, dict, dict]:
    data, _metadata = load_source_data(data_path, metadata_path)
    frame, manifest = prepare_model_frame(data)
    report = validate_source_data(frame)
    return frame, manifest, report


def forecast_artifact_token() -> tuple[int, ...]:
    artifact_paths = [
        DEFAULT_CONFIG.artifacts_dir / "tft_station_model.pkl",
        DEFAULT_CONFIG.artifacts_dir / "preprocessors.pkl",
        DEFAULT_CONFIG.artifacts_dir / "manifest.json",
    ]
    return tuple(path.stat().st_mtime_ns for path in artifact_paths)


@st.cache_data(show_spinner="Building TFT forecast...")
def load_tft_forecast(station_id: int | None, horizon: int, artifact_token: tuple[int, ...]) -> pd.DataFrame:
    del artifact_token
    return predict_with_saved_model(
        data_path=DEFAULT_CONFIG.data_path,
        metadata_path=DEFAULT_CONFIG.metadata_path,
        artifacts_dir=DEFAULT_CONFIG.artifacts_dir,
        station_id=station_id,
        horizon=horizon,
    )


def make_dashboard_forecast(
    data: pd.DataFrame,
    station_id: int | None,
    horizon: int,
    model_ready: bool,
) -> tuple[pd.DataFrame, str, str | None]:
    if not model_ready:
        return make_demo_forecast(data, station_id, horizon), "Baseline forecast", None

    try:
        forecast = load_tft_forecast(station_id, horizon, forecast_artifact_token())
        if station_id is None and not forecast.empty:
            target_cols = [col for col in TARGET_COLUMNS if col in forecast.columns]
            forecast = forecast.groupby("timestamp", as_index=False)[target_cols].sum()
        return forecast, "TFT forecast", None
    except Exception as exc:
        fallback = make_demo_forecast(data, station_id, horizon)
        return fallback, "Baseline fallback", str(exc)


def format_number(value: float, suffix: str = "") -> str:
    if abs(value) >= 1_000_000:
        result = f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        result = f"{value / 1_000:.1f}K"
    else:
        result = f"{value:.0f}"
    return f"{result}{suffix}"


def kpi_card(label: str, value: str, delta: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-delta">{delta}</div>'
        f"</div>"
    )


def section_header(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="glass-card" style="padding:.85rem 1rem;margin:.15rem 0 .85rem;">
            <div class="kpi-label">{caption}</div>
            <h3 style="font-size:1.08rem!important;margin:.18rem 0 0!important;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="glass-card"><h3>{title}</h3><div class="subtitle">{body}</div></div>', unsafe_allow_html=True)


def aggregate_by_period(df: pd.DataFrame, period: str, metric_cols: list[str]) -> pd.DataFrame:
    if period == "Час":
        return df.groupby("timestamp", as_index=False)[metric_cols].sum()
    freq = {"День": "D", "Неделя": "W"}[period]
    out = df.set_index("timestamp").groupby(pd.Grouper(freq=freq))[metric_cols].sum().reset_index()
    return out


def line_chart(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.area(df, x="timestamp", y=metric, title=title)
    fig.update_traces(line_color=ACCENT, fillcolor="rgba(32, 227, 178, .18)")
    fig.update_layout(template=plotly_template(), height=420)
    return fig


def fuel_mix_chart(df: pd.DataFrame) -> go.Figure:
    cols = [col for col in FUEL_TARGET_COLUMNS if col in df.columns]
    totals = df[cols].sum().sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=totals.values, y=[c.replace("sales_", "") for c in totals.index], orientation="h"))
    fig.update_traces(marker_color=[ACCENT, BLUE, AMBER, PINK, VIOLET, "#35C2FF", "#7EE787"][: len(totals)])
    fig.update_layout(template=plotly_template(), title="Структура продаж топлива", height=360)
    return fig


def heatmap_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    pivot = df.pivot_table(values=metric, index="day_of_week", columns="hour", aggfunc="mean").fillna(0)
    fig = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#101827"], [0.45, BLUE], [1, ACCENT]],
        labels={"x": "Час", "y": "День недели", "color": metric},
        title="Тепловая карта спроса",
    )
    fig.update_layout(template=plotly_template(), height=360)
    return fig


def station_ranking(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["station_id", "station_name"], as_index=False)
        .agg(
            value=(metric, "sum"),
            traffic=("total_traffic", "sum"),
            shop_revenue=("shop_total_revenue", "sum"),
        )
        .sort_values("value", ascending=False)
    )


def forecast_breakdown(
    forecast: pd.DataFrame,
    columns: list[str],
    name_prefix: str,
    value_column: str,
) -> pd.DataFrame:
    forecast_cols = [col for col in columns if col in forecast.columns]
    if forecast.empty or not forecast_cols:
        return pd.DataFrame()

    totals = forecast[forecast_cols].clip(lower=0).sum().sort_values(ascending=False)
    total_value = totals.sum()
    breakdown = totals.rename(value_column).reset_index()
    breakdown = breakdown.rename(columns={breakdown.columns[0]: "metric"})
    breakdown["name"] = breakdown["metric"].str.replace(name_prefix, "", regex=False)
    breakdown["share"] = breakdown[value_column] / total_value if total_value else 0
    breakdown["Доля"] = breakdown["share"].map(lambda value: f"{value:.1%}")
    return breakdown


def history_with_forecast_chart(
    history: pd.DataFrame,
    future: pd.DataFrame,
    columns: list[str],
    title: str,
    value_label: str,
    colors: list[str],
) -> go.Figure:
    fig = go.Figure()
    for index, col in enumerate(columns):
        color = colors[index % len(colors)]
        label = col.replace("sales_", "").replace("shop_", "")
        fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history[col],
                mode="lines",
                name=label,
                legendgroup=col,
                line={"color": color, "width": 2},
            )
        )
        if not future.empty and col in future.columns:
            fig.add_trace(
                go.Scatter(
                    x=future["timestamp"],
                    y=future[col],
                    mode="lines",
                    name=f"{label} · прогноз",
                    legendgroup=col,
                    showlegend=False,
                    line={"color": color, "width": 2, "dash": "dash"},
                )
            )

    if not future.empty:
        fig.add_vline(
            x=future["timestamp"].min(),
            line_color="rgba(255,255,255,.35)",
            line_dash="dot",
        )
    fig.update_layout(
        template=plotly_template(),
        title=title,
        height=500,
        xaxis_title="время",
        yaxis_title=value_label,
        legend_title_text="ряд",
    )
    return fig


def factor_mean_frame(df: pd.DataFrame, metric: str, factor: str, label: str) -> pd.DataFrame:
    if factor not in df.columns or metric not in df.columns:
        return pd.DataFrame()

    frame = df[[factor, metric]].copy()
    frame[factor] = frame[factor].fillna("unknown").astype(str)
    frame = frame.groupby(factor, as_index=False).agg(
        **{
            "Среднее за час": (metric, "mean"),
            "Наблюдений, ч": (metric, "size"),
        }
    )
    return frame.rename(columns={factor: label})


def restore_factor_labels(df: pd.DataFrame, label: str, category_mapping: dict[str, int] | None) -> pd.DataFrame:
    if df.empty or not category_mapping:
        return df

    code_to_name = {str(code): name for name, code in category_mapping.items()}
    out = df.copy()
    out[label] = out[label].map(lambda value: code_to_name.get(str(value), value))
    return out


def holiday_mean_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns or "is_holiday" not in df.columns:
        return pd.DataFrame()

    frame = df[[metric, "is_holiday"]].copy()
    if "holiday_name" in df.columns:
        holiday_name = df["holiday_name"].fillna("").astype(str).str.strip()
    else:
        holiday_name = pd.Series("", index=df.index)
    frame["День"] = "Обычный день"
    frame.loc[frame["is_holiday"].eq(1), "День"] = holiday_name[frame["is_holiday"].eq(1)]
    frame.loc[frame["День"].eq(""), "День"] = "Праздник"
    return frame.groupby("День", as_index=False).agg(
        **{
            "Среднее за час": (metric, "mean"),
            "Наблюдений, ч": (metric, "size"),
        }
    )


def marketing_mean_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {"promotion_fuel_active", "ad_active", metric}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    frame = df[[metric, "promotion_fuel_active", "ad_active"]].copy()
    promo = frame["promotion_fuel_active"].eq(1)
    ad = frame["ad_active"].eq(1)
    frame["Сценарий"] = "Без промо и рекламы"
    frame.loc[promo & ~ad, "Сценарий"] = "Только промо"
    frame.loc[~promo & ad, "Сценарий"] = "Только реклама"
    frame.loc[promo & ad, "Сценарий"] = "Промо + реклама"
    return frame.groupby("Сценарий", as_index=False).agg(
        **{
            "Среднее за час": (metric, "mean"),
            "Наблюдений, ч": (metric, "size"),
        }
    )


def factor_mean_chart(df: pd.DataFrame, label: str, title: str, color: str) -> go.Figure:
    ordered = df.sort_values("Среднее за час", ascending=True)
    fig = px.bar(
        ordered,
        x="Среднее за час",
        y=label,
        orientation="h",
        title=title,
    )
    fig.update_traces(marker_color=color)
    fig.update_layout(template=plotly_template(), height=max(315, min(650, 120 + 28 * len(ordered))), yaxis_title=None)
    return fig


def competitor_position_frame(df: pd.DataFrame, fuel: str) -> pd.DataFrame:
    own_col, competitor_col = COMPETITOR_PRICE_PAIRS[fuel]
    required = {"station_id", "station_name", own_col, competitor_col}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    frame = df[["station_id", "station_name", own_col, competitor_col]].dropna().copy()
    if frame.empty:
        return pd.DataFrame()

    position = frame.groupby(["station_id", "station_name"], as_index=False).agg(
        **{
            "Своя цена, руб": (own_col, "mean"),
            "Цена конкурентов, руб": (competitor_col, "mean"),
            "Наблюдений, ч": (own_col, "size"),
        }
    )
    position["Разница, руб"] = position["Своя цена, руб"] - position["Цена конкурентов, руб"]
    position["Разница, %"] = position["Разница, руб"] / position["Цена конкурентов, руб"] * 100
    position["Оценка"] = "На уровне конкурентов"
    position.loc[position["Разница, руб"] <= -0.25, "Оценка"] = "Дешевле конкурентов"
    position.loc[position["Разница, руб"] >= 0.25, "Оценка"] = "Дороже конкурентов"
    return position.sort_values("Разница, руб")


def competitor_position_chart(position: pd.DataFrame, fuel: str) -> go.Figure:
    fig = px.bar(
        position,
        x="Разница, руб",
        y="station_name",
        orientation="h",
        color="Разница, руб",
        color_continuous_scale=[[0, ACCENT], [0.5, BLUE], [1, PINK]],
        title=f"Ценовая позиция АЗС по {fuel}: своя цена - конкуренты",
    )
    fig.add_vline(x=0, line_color="rgba(255,255,255,.50)", line_dash="dot")
    fig.update_layout(
        template=plotly_template(),
        height=max(340, min(620, 118 + 34 * len(position))),
        yaxis_title=None,
        coloraxis_showscale=False,
    )
    return fig


def competitor_price_trend(df: pd.DataFrame, station_id: int, fuel: str) -> go.Figure:
    own_col, competitor_col = COMPETITOR_PRICE_PAIRS[fuel]
    station = df[df["station_id"].eq(station_id)]
    daily = (
        station.set_index("timestamp")
        .groupby(pd.Grouper(freq="D"))[[own_col, competitor_col]]
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    plot = daily.rename(
        columns={
            own_col: f"Своя цена {fuel}",
            competitor_col: "Цена конкурентов",
        }
    )
    fig = px.line(
        plot,
        x="timestamp",
        y=[f"Своя цена {fuel}", "Цена конкурентов"],
        title=f"Динамика цены выбранной АЗС по {fuel}",
    )
    fig.update_layout(template=plotly_template(), height=360, yaxis_title="руб.")
    return fig


def decode_factor_values(values: pd.Series, category_mapping: dict[str, int] | None) -> pd.Series:
    if not category_mapping:
        return values.fillna("unknown").astype(str)

    code_to_name = {str(code): name for name, code in category_mapping.items()}

    def decode(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        key = str(value)
        if isinstance(value, float) and value.is_integer():
            key = str(int(value))
        return code_to_name.get(key, key)

    return values.map(decode)


def interval_band(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(numeric, bins=bins, labels=labels, include_lowest=True).astype("string").fillna("unknown")


def qcut_band(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        return pd.qcut(numeric, q=4, duplicates="drop").astype("string").fillna("unknown")
    except ValueError:
        return pd.Series("Все значения", index=values.index, dtype="string")


def binary_factor(values: pd.Series) -> pd.Series:
    return values.fillna(0).eq(1).map({True: "Да", False: "Нет"})


def prepare_analysis_frame(df: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    out = df.copy()
    encoders = manifest.get("categorical_encoders", {})
    for col in ["road_type", "direction", "settlement_size"]:
        if col in out.columns:
            out[f"{col}_label"] = decode_factor_values(out[col], encoders.get(col))

    if "day_of_week" in out.columns:
        out["weekday_label"] = out["day_of_week"].map(WEEKDAY_LABELS).fillna("unknown")
    if "hour" in out.columns:
        out["hour_label"] = out["hour"].map(lambda value: f"{int(value):02d}:00" if pd.notna(value) else "unknown")
    if "month" in out.columns:
        out["month_label"] = out["month"].map(lambda value: f"{int(value):02d}" if pd.notna(value) else "unknown")

    for source, label in {
        "is_weekend": "is_weekend_label",
        "is_holiday": "is_holiday_label",
        "is_rush_hour": "is_rush_hour_label",
        "is_night": "is_night_label",
        "ad_active": "ad_active_label",
        "promotion_fuel_active": "promotion_fuel_label",
        "promotion_shop_active": "promotion_shop_label",
        "promotion_cafe_active": "promotion_cafe_label",
        "has_car_wash": "has_car_wash_label",
        "has_cafe": "has_cafe_label",
        "has_shop": "has_shop_label",
    }.items():
        if source in out.columns:
            out[label] = binary_factor(out[source])

    if "distance_to_city_km" in out.columns:
        out["distance_band"] = interval_band(
            out["distance_to_city_km"],
            [-float("inf"), 5, 15, 30, float("inf")],
            ["до 5 км", "5-15 км", "15-30 км", "30+ км"],
        )
    if "temperature" in out.columns:
        out["temperature_band"] = interval_band(
            out["temperature"],
            [-float("inf"), -10, 0, 10, 20, float("inf")],
            ["ниже -10", "-10..0", "0..10", "10..20", "20+"],
        )
    if "precipitation_mm" in out.columns:
        out["precipitation_band"] = interval_band(
            out["precipitation_mm"],
            [-float("inf"), 0, 1, 3, float("inf")],
            ["нет", "до 1 мм", "1-3 мм", "3+ мм"],
        )
    if "competitors_within_5km" in out.columns:
        out["competitor_count_band"] = interval_band(
            out["competitors_within_5km"],
            [-float("inf"), 0, 2, 4, float("inf")],
            ["0", "1-2", "3-4", "5+"],
        )
    if "total_traffic" in out.columns:
        out["traffic_band"] = qcut_band(out["total_traffic"])

    traffic_cols = [col for col in TRAFFIC_LABELS if col.startswith("traffic_") and col in out.columns]
    if traffic_cols:
        dominant = out[traffic_cols].idxmax(axis=1)
        out["dominant_traffic"] = dominant.map(TRAFFIC_LABELS).fillna("unknown")
    return out


def filter_categories(df: pd.DataFrame, column: str, selected: list[str], all_values: list[str]) -> pd.DataFrame:
    if column not in df.columns or not selected or set(selected) == set(all_values):
        return df
    return df[df[column].astype(str).isin(selected)]


def filter_binary(df: pd.DataFrame, column: str, state: str) -> pd.DataFrame:
    if column not in df.columns or state == "Любое":
        return df
    expected = 1 if state == "Да" else 0
    return df[df[column].fillna(0).eq(expected)]


def filter_interval(df: pd.DataFrame, column: str, bounds: tuple[float, float] | None) -> pd.DataFrame:
    if column not in df.columns or bounds is None:
        return df
    values = pd.to_numeric(df[column], errors="coerce")
    return df[values.between(bounds[0], bounds[1], inclusive="both")]


def sorted_options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def range_filter_widget(
    df: pd.DataFrame,
    column: str,
    label: str,
    key: str,
    step: float,
) -> tuple[float, float] | None:
    if column not in df.columns:
        return None

    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    if numeric.empty:
        return None

    low, high = float(numeric.min()), float(numeric.max())
    if low == high:
        st.caption(f"{label}: {low:g}")
        return low, high
    return st.slider(label, min_value=low, max_value=high, value=(low, high), step=step, key=key)


def scenario_trend_chart(base: pd.DataFrame, scenario: pd.DataFrame, metric: str, title: str) -> go.Figure:
    whole = base.set_index("timestamp").groupby(pd.Grouper(freq="D"))[metric].mean().rename("Весь срез")
    selected = scenario.set_index("timestamp").groupby(pd.Grouper(freq="D"))[metric].mean().rename("Сценарий")
    trend = pd.concat([whole, selected], axis=1).reset_index()
    fig = px.line(trend, x="timestamp", y=["Весь срез", "Сценарий"], title=title)
    fig.update_layout(template=plotly_template(), height=390, yaxis_title="среднее за час")
    return fig


def component_mean_chart(df: pd.DataFrame, columns: list[str], title: str, color: str) -> go.Figure:
    present = [col for col in columns if col in df.columns]
    values = df[present].mean().sort_values(ascending=True)
    names = [col.replace("sales_", "").replace("shop_", "") for col in values.index]
    fig = go.Figure(go.Bar(x=values.values, y=names, orientation="h", marker_color=color))
    fig.update_layout(template=plotly_template(), title=title, height=max(320, 120 + 30 * len(values)), xaxis_title="среднее за час")
    return fig


def traffic_mix_comparison_chart(base: pd.DataFrame, scenario: pd.DataFrame) -> go.Figure:
    traffic_cols = [col for col in TRAFFIC_LABELS if col.startswith("traffic_") and col in base.columns]
    rows = []
    for col in traffic_cols:
        rows.append({"Транспорт": TRAFFIC_LABELS[col], "Срез": "Весь срез", "Среднее за час": base[col].mean()})
        rows.append({"Транспорт": TRAFFIC_LABELS[col], "Срез": "Сценарий", "Среднее за час": scenario[col].mean()})
    chart_data = pd.DataFrame(rows)
    fig = px.bar(
        chart_data,
        x="Транспорт",
        y="Среднее за час",
        color="Срез",
        barmode="group",
        title="Состав трафика в выбранном сценарии",
    )
    fig.update_layout(template=plotly_template(), height=360)
    return fig


def station_scenario_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    ranking = (
        df.groupby(["station_id", "station_name"], as_index=False)
        .agg(**{"Среднее за час": (metric, "mean"), "Наблюдений, ч": (metric, "size")})
        .sort_values("Среднее за час", ascending=False)
        .head(15)
        .sort_values("Среднее за час", ascending=True)
    )
    fig = px.bar(
        ranking,
        x="Среднее за час",
        y="station_name",
        orientation="h",
        hover_data=["Наблюдений, ч"],
        title="АЗС с максимальным средним спросом в сценарии",
    )
    fig.update_traces(marker_color=BLUE)
    fig.update_layout(template=plotly_template(), height=max(340, 120 + 28 * len(ranking)), yaxis_title=None)
    return fig


def driver_scatter_chart(df: pd.DataFrame, metric: str, driver: str, driver_label: str) -> go.Figure:
    sample = df.sample(min(9000, len(df)), random_state=31) if len(df) > 9000 else df
    hover_cols = [col for col in ["station_name", "weather_condition", "total_traffic"] if col in sample.columns]
    color = "dominant_traffic" if "dominant_traffic" in sample.columns else None
    fig = px.scatter(
        sample,
        x=driver,
        y=metric,
        color=color,
        hover_data=hover_cols,
        opacity=0.5,
        title=f"{driver_label} и выбранная метрика",
    )
    fig.update_traces(marker={"size": 5})
    fig.update_layout(template=plotly_template(), height=390, showlegend=color is not None)
    return fig


def analysis_factor_specs(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    specs = {
        "АЗС": ("station_name", "АЗС"),
        "Погода": ("weather_condition", "Погода"),
        "Температурный диапазон": ("temperature_band", "Температура"),
        "Осадки": ("precipitation_band", "Осадки"),
        "Сезон": ("season", "Сезон"),
        "Месяц": ("month_label", "Месяц"),
        "День недели": ("weekday_label", "День"),
        "Час суток": ("hour_label", "Час"),
        "Праздник": ("is_holiday_label", "Праздник"),
        "Выходной": ("is_weekend_label", "Выходной"),
        "Час пик": ("is_rush_hour_label", "Час пик"),
        "Ночь": ("is_night_label", "Ночь"),
        "Реклама": ("ad_active_label", "Реклама"),
        "Канал рекламы": ("ad_channel", "Канал"),
        "Промо топлива": ("promotion_fuel_label", "Промо"),
        "Промо магазина": ("promotion_shop_label", "Промо"),
        "Промо кафе": ("promotion_cafe_label", "Промо"),
        "Доминирующий транспорт": ("dominant_traffic", "Транспорт"),
        "Интенсивность трафика": ("traffic_band", "Трафик"),
        "Тип дороги": ("road_type_label", "Тип дороги"),
        "Направление": ("direction_label", "Направление"),
        "Размер населенного пункта": ("settlement_size_label", "Населенный пункт"),
        "Удаленность от города": ("distance_band", "Удаленность"),
        "Конкуренты рядом": ("competitor_count_band", "Конкуренты"),
        "Мойка": ("has_car_wash_label", "Мойка"),
        "Кафе": ("has_cafe_label", "Кафе"),
        "Магазин на АЗС": ("has_shop_label", "Магазин"),
    }
    return {name: spec for name, spec in specs.items() if spec[0] in df.columns}


def numeric_driver_specs(df: pd.DataFrame) -> dict[str, str]:
    specs = {
        "Температура": "temperature",
        "Осадки": "precipitation_mm",
        "Видимость": "visibility_km",
        "Ветер": "wind_speed_ms",
        "Весь трафик": "total_traffic",
        "Легковые": "traffic_Passengers_cars",
        "Короткие грузовики": "traffic_Truck_short",
        "Грузовики": "traffic_Truck",
        "Длинные грузовики": "traffic_Truck_long",
        "Транспортеры": "traffic_Transporter",
        "Удаленность от города": "distance_to_city_km",
        "Конкуренты в 5 км": "competitors_within_5km",
        "Число колонок": "total_pumps",
        "Площадь магазина": "shop_area_m2",
        "Лояльность клиентов": "customer_loyalty_score",
        "Вовлеченность персонала": "staff_engagement_score",
        "Доля корпоративных клиентов": "corporate_customer_ratio",
        "Цена AI92": "price_AI92",
        "Цена AI95": "price_AI95",
        "Цена DT EURO": "price_DT_EURO",
        "Цена конкурентов AI92": "competitor_price_AI92",
        "Цена конкурентов AI95": "competitor_price_AI95",
        "Цена конкурентов DT": "competitor_price_DT",
    }
    return {name: col for name, col in specs.items() if col in df.columns}


def render_scenario_filters(base: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    weather_options = sorted_options(base, "weather_condition")
    season_options = sorted_options(base, "season")
    channel_options = sorted_options(base, "ad_channel")
    road_options = sorted_options(base, "road_type_label")
    direction_options = sorted_options(base, "direction_label")
    settlement_options = sorted_options(base, "settlement_size_label")
    transport_options = sorted_options(base, "dominant_traffic")
    traffic_range_columns = list(dict.fromkeys(col for col in ["total_traffic", *TRAFFIC_LABELS] if col in base.columns))

    with st.expander("Сценарий: условия часа и параметры АЗС", expanded=True):
        weather_col, calendar_col, marketing_col, static_col = st.columns(4)
        with weather_col:
            weather_selected = st.multiselect("Погода", weather_options, default=weather_options, key=f"{key_prefix}_weather")
            temp_range = range_filter_widget(base, "temperature", "Температура", f"{key_prefix}_temperature", 0.1)
            precipitation_range = range_filter_widget(base, "precipitation_mm", "Осадки, мм", f"{key_prefix}_precipitation", 0.1)
            visibility_range = range_filter_widget(base, "visibility_km", "Видимость, км", f"{key_prefix}_visibility", 0.1)
            wind_range = range_filter_widget(base, "wind_speed_ms", "Ветер, м/с", f"{key_prefix}_wind", 0.1)
            snow_state = st.selectbox("Снег", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_snow")
            rain_state = st.selectbox("Дождь", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_rain")
            fog_state = st.selectbox("Туман", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_fog")
        with calendar_col:
            season_selected = st.multiselect("Сезон", season_options, default=season_options, key=f"{key_prefix}_season")
            hour_range = range_filter_widget(base, "hour", "Час суток", f"{key_prefix}_hour", 1.0)
            weekend_state = st.selectbox("Выходной", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_weekend")
            holiday_state = st.selectbox("Праздник", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_holiday")
            rush_state = st.selectbox("Час пик", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_rush")
            night_state = st.selectbox("Ночь", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_night")
        with marketing_col:
            ad_state = st.selectbox("Реклама", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_ad")
            channel_selected = st.multiselect("Канал рекламы", channel_options, default=channel_options, key=f"{key_prefix}_channel")
            fuel_promo_state = st.selectbox("Промо топлива", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_fuel_promo")
            shop_promo_state = st.selectbox("Промо магазина", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_shop_promo")
            cafe_promo_state = st.selectbox("Промо кафе", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_cafe_promo")
            dominant_selected = st.multiselect(
                "Доминирующий транспорт",
                transport_options,
                default=transport_options,
                key=f"{key_prefix}_dominant_transport",
            )
            traffic_range_col = st.selectbox(
                "Диапазон по трафику",
                traffic_range_columns,
                format_func=lambda col: TRAFFIC_LABELS.get(col, col),
                key=f"{key_prefix}_traffic_range_column",
            )
            traffic_range = range_filter_widget(
                base,
                traffic_range_col,
                TRAFFIC_LABELS.get(traffic_range_col, traffic_range_col),
                f"{key_prefix}_traffic_range_{traffic_range_col}",
                1.0,
            )
        with static_col:
            road_selected = st.multiselect("Тип дороги", road_options, default=road_options, key=f"{key_prefix}_road")
            direction_selected = st.multiselect("Направление", direction_options, default=direction_options, key=f"{key_prefix}_direction")
            settlement_selected = st.multiselect(
                "Размер населенного пункта",
                settlement_options,
                default=settlement_options,
                key=f"{key_prefix}_settlement",
            )
            distance_range = range_filter_widget(base, "distance_to_city_km", "Удаленность от города, км", f"{key_prefix}_distance", 0.1)
            competitor_range = range_filter_widget(base, "competitors_within_5km", "Конкуренты в 5 км", f"{key_prefix}_competitors", 1.0)
            pumps_range = range_filter_widget(base, "total_pumps", "Колонки", f"{key_prefix}_pumps", 1.0)
            shop_area_range = range_filter_widget(base, "shop_area_m2", "Площадь магазина, м2", f"{key_prefix}_shop_area", 1.0)
            car_wash_state = st.selectbox("Мойка", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_wash")
            cafe_state = st.selectbox("Кафе", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_cafe")
            shop_state = st.selectbox("Магазин", BINARY_FILTER_OPTIONS, key=f"{key_prefix}_shop")

    scenario = base.copy()
    for col, selected, options in [
        ("weather_condition", weather_selected, weather_options),
        ("season", season_selected, season_options),
        ("ad_channel", channel_selected, channel_options),
        ("dominant_traffic", dominant_selected, transport_options),
        ("road_type_label", road_selected, road_options),
        ("direction_label", direction_selected, direction_options),
        ("settlement_size_label", settlement_selected, settlement_options),
    ]:
        scenario = filter_categories(scenario, col, selected, options)

    for col, state in {
        "is_snow": snow_state,
        "is_rain": rain_state,
        "is_fog": fog_state,
        "is_weekend": weekend_state,
        "is_holiday": holiday_state,
        "is_rush_hour": rush_state,
        "is_night": night_state,
        "ad_active": ad_state,
        "promotion_fuel_active": fuel_promo_state,
        "promotion_shop_active": shop_promo_state,
        "promotion_cafe_active": cafe_promo_state,
        "has_car_wash": car_wash_state,
        "has_cafe": cafe_state,
        "has_shop": shop_state,
    }.items():
        scenario = filter_binary(scenario, col, state)

    for col, bounds in {
        "temperature": temp_range,
        "precipitation_mm": precipitation_range,
        "visibility_km": visibility_range,
        "wind_speed_ms": wind_range,
        "hour": hour_range,
        traffic_range_col: traffic_range,
        "distance_to_city_km": distance_range,
        "competitors_within_5km": competitor_range,
        "total_pumps": pumps_range,
        "shop_area_m2": shop_area_range,
    }.items():
        scenario = filter_interval(scenario, col, bounds)
    return scenario


def render_demand_explorer(
    base: pd.DataFrame,
    metric_options: list[str],
    component_columns: list[str],
    title: str,
    unit: str,
    color: str,
    key_prefix: str,
) -> None:
    section_header(title, "Интерактивный исторический анализ условий спроса")
    st.caption(
        "Фильтры ниже отбирают реальные часы из текущего периода и выбранных АЗС. "
        "Сравнения показывают наблюдаемую связь признаков со спросом, а не контрфактический прогноз TFT."
    )
    metric = st.selectbox("Метрика анализа", metric_options, key=f"{key_prefix}_metric")
    scenario = render_scenario_filters(base, key_prefix)
    if scenario.empty:
        st.warning("По этому сочетанию условий в текущем срезе нет исторических наблюдений.")
        return

    baseline_mean = base[metric].mean()
    scenario_mean = scenario[metric].mean()
    mean_delta = (scenario_mean / baseline_mean - 1) if baseline_mean else 0
    coverage = len(scenario) / len(base) if len(base) else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Часов в сценарии", f"{len(scenario):,}", f"{coverage:.1%} среза")
    k2.metric("АЗС в сценарии", str(scenario["station_id"].nunique()))
    k3.metric("Среднее за час", format_number(scenario_mean, unit), f"{mean_delta:+.1%} к срезу")
    k4.metric("Сумма в сценарии", format_number(scenario[metric].sum(), unit))

    trend_col, mix_col = st.columns([1.28, 1])
    with trend_col:
        st.plotly_chart(
            scenario_trend_chart(base, scenario, metric, "Динамика метрики: сценарий против всего среза"),
            use_container_width=True,
            key=f"{key_prefix}_trend_chart",
        )
    with mix_col:
        st.plotly_chart(
            component_mean_chart(scenario, component_columns, "Средний состав продаж в сценарии", color),
            use_container_width=True,
            key=f"{key_prefix}_component_chart",
        )

    factor_specs = analysis_factor_specs(scenario)
    driver_specs = numeric_driver_specs(scenario)
    compare_col, scatter_col = st.columns([1.05, 1])
    with compare_col:
        factor_name = st.selectbox("Разрез для сравнения", list(factor_specs), key=f"{key_prefix}_factor")
        factor_col, factor_label = factor_specs[factor_name]
        factor_frame = factor_mean_frame(scenario, metric, factor_col, factor_label)
        st.plotly_chart(
            factor_mean_chart(factor_frame, factor_label, f"{metric}: {factor_name.lower()}", color),
            use_container_width=True,
            key=f"{key_prefix}_factor_chart",
        )
        factor_table = factor_frame.sort_values("Среднее за час", ascending=False).copy()
        factor_table["Среднее за час"] = factor_table["Среднее за час"].round(2)
        st.dataframe(factor_table, hide_index=True, use_container_width=True, key=f"{key_prefix}_factor_table")
    with scatter_col:
        driver_name = st.selectbox("Числовой признак", list(driver_specs), key=f"{key_prefix}_driver")
        driver_col = driver_specs[driver_name]
        st.plotly_chart(
            driver_scatter_chart(scenario, metric, driver_col, driver_name),
            use_container_width=True,
            key=f"{key_prefix}_driver_chart",
        )
        st.plotly_chart(
            traffic_mix_comparison_chart(base, scenario),
            use_container_width=True,
            key=f"{key_prefix}_traffic_mix_chart",
        )

    st.plotly_chart(station_scenario_chart(scenario, metric), use_container_width=True, key=f"{key_prefix}_station_chart")


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name)
    return cleaned.strip("._") or "uploaded.csv"


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / safe_filename(uploaded_file.name)
    output.write_bytes(uploaded_file.getvalue())
    return output


def preview_csv(source, rows: int = 5) -> tuple[pd.DataFrame | None, str | None]:
    try:
        if hasattr(source, "seek"):
            source.seek(0)
        preview = pd.read_csv(source, nrows=rows)
        if hasattr(source, "seek"):
            source.seek(0)
        return preview, None
    except Exception as exc:
        return None, str(exc)


def validate_training_preview(preview: pd.DataFrame | None) -> dict:
    if preview is None:
        return {"ok": False, "missing_required": [], "available_targets": []}
    required = ["timestamp", "station_id", "station_name"]
    available_targets = [col for col in TARGET_COLUMNS if col in preview.columns]
    missing_required = [col for col in required if col not in preview.columns]
    return {
        "ok": not missing_required and bool(available_targets),
        "missing_required": missing_required,
        "available_targets": available_targets,
    }


def is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=flags,
                timeout=5,
            )
            return str(pid) in result.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_log_tail(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return "Лог пока не создан."
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


def latest_training_runs(limit: int = 5) -> list[Path]:
    if not TRAINING_RUNS_DIR.exists():
        return []
    return sorted(
        [path for path in TRAINING_RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]


def read_training_log(run_dir: Path, max_chars: int = 800_000) -> str:
    log_path = run_dir / "train.log"
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


def parse_training_log_metrics(log_text: str) -> pd.DataFrame:
    rows = []
    normalized = log_text.replace("\r", "\n")
    pattern = re.compile(
        r"Epoch\s+(\d+):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s+\[[^\]]*?([0-9.]+)it/s.*?train_loss=([0-9.eE+-]+)"
    )
    for idx, match in enumerate(pattern.finditer(normalized)):
        epoch, percent, step, total, speed, loss = match.groups()
        rows.append(
            {
                "step": idx,
                "epoch": int(epoch),
                "epoch_percent": int(percent),
                "batch": int(step),
                "batches_total": int(total),
                "speed_it_s": float(speed),
                "metric": "train_loss",
                "loss": float(loss),
            }
        )
    return pd.DataFrame(rows)


def read_tensorboard_metrics(run_dir: Path) -> pd.DataFrame:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return pd.DataFrame()

    rows = []
    event_dirs = sorted({path.parent for path in run_dir.rglob("events.out.tfevents.*")})
    for event_dir in event_dirs:
        try:
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
        except Exception:
            continue
        for tag in ("train_loss", "val_loss"):
            if tag not in scalar_tags:
                continue
            try:
                for event in accumulator.Scalars(tag):
                    rows.append(
                        {
                            "step": int(event.step),
                            "epoch": None,
                            "epoch_percent": None,
                            "batch": None,
                            "batches_total": None,
                            "speed_it_s": None,
                            "metric": tag,
                            "loss": float(event.value),
                        }
                    )
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["step", "metric"], keep="last").sort_values("step")


def training_metrics(run_dir: Path) -> pd.DataFrame:
    tb_metrics = read_tensorboard_metrics(run_dir)
    log_metrics = parse_training_log_metrics(read_training_log(run_dir))
    if tb_metrics.empty:
        return log_metrics
    if log_metrics.empty:
        return tb_metrics
    val_metrics = tb_metrics[tb_metrics["metric"] == "val_loss"]
    if val_metrics.empty:
        return log_metrics
    return pd.concat([log_metrics, val_metrics], ignore_index=True).sort_values(["metric", "step"])


def training_status_info(run_dir: Path) -> dict:
    status = read_json_file(run_dir / "status.json")
    log_text = read_training_log(run_dir, max_chars=20000)
    pid = status.get("pid")
    mode = status.get("mode", "train")
    if '"status": "trained"' in log_text:
        return {"code": "trained", "title": "Завершено", "body": "Модель обучена, артефакты сохранены."}
    if '"status": "dry_run_ok"' in log_text:
        return {"code": "dry_run_ok", "title": "Проверка OK", "body": "Данные подготовлены, manifest записан."}
    if "Traceback" in log_text or "RuntimeError" in log_text or "Error" in log_text:
        return {"code": "error", "title": "Ошибка", "body": "Процесс завершился с ошибкой, смотрите лог внизу."}
    if is_pid_running(pid):
        title = "Идёт проверка" if mode == "dry_run" else "Идёт обучение"
        return {"code": "running", "title": title, "body": f"Процесс PID {pid} активен."}
    if pid:
        return {
            "code": "stopped",
            "title": "Процесс не активен",
            "body": "PID уже не найден. Если нет статуса trained/dry_run_ok, проверьте лог.",
        }
    return {"code": "none", "title": "Нет запуска", "body": "Запуск ещё не выполнялся."}


def detect_training_status(run_dir: Path) -> tuple[str, str]:
    status = training_status_info(run_dir)
    return status["title"], status["body"]


def training_progress_state(run_dir: Path, status: dict, metrics: pd.DataFrame) -> dict:
    status_json = read_json_file(run_dir / "status.json")
    params = status_json.get("params", {})
    total_epochs = int(params.get("epochs") or 0)
    mode = status_json.get("mode", "train")
    if status["code"] in {"trained", "dry_run_ok"}:
        return {"percent": 100.0, "epoch": total_epochs, "epochs": total_epochs, "batch": None, "batches_total": None}
    if mode == "dry_run":
        return {"percent": 35.0 if status["code"] == "running" else 0.0, "epoch": 0, "epochs": 0, "batch": None, "batches_total": None}
    del metrics
    train_metrics = parse_training_log_metrics(read_training_log(run_dir))
    if train_metrics.empty:
        return {"percent": 0, "epoch": 0, "epochs": total_epochs, "batch": None, "batches_total": None}
    row = train_metrics.tail(1).iloc[0]
    epoch = int(row.get("epoch") or 0)
    batch = int(row.get("batch") or 0) if pd.notna(row.get("batch")) else None
    batches_total = int(row.get("batches_total") or 0) if pd.notna(row.get("batches_total")) else None
    within_epoch = (batch / batches_total) if batch and batches_total else 0
    if total_epochs > 0:
        percent = min(99.0, round(((epoch + within_epoch) / total_epochs) * 100, 1))
    else:
        percent = float(row.get("epoch_percent") or 0)
    return {
        "percent": max(0, percent),
        "epoch": epoch + 1,
        "epochs": total_epochs,
        "batch": batch,
        "batches_total": batches_total,
    }


def training_loss_chart(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if metrics.empty:
        fig.update_layout(template=plotly_template(), title="Loss пока не записан", height=360)
        return fig
    labels = {"train_loss": "Train loss", "val_loss": "Validation loss"}
    for metric_name, group in metrics.groupby("metric"):
        fig.add_trace(
            go.Scatter(
                x=group["step"],
                y=group["loss"],
                mode="lines",
                name=labels.get(metric_name, metric_name),
            )
        )
    fig.update_layout(template=plotly_template(), title="Динамика loss", height=360)
    fig.update_xaxes(title="Шаг обучения")
    fig.update_yaxes(title="Loss")
    return fig


def build_train_command(
    data_path: Path,
    metadata_path: Path,
    output_dir: Path,
    params: dict,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "train.py"),
        "--data",
        str(data_path),
        "--metadata",
        str(metadata_path),
        "--output",
        str(output_dir),
        "--epochs",
        str(params["epochs"]),
        "--input-chunk-length",
        str(params["input_chunk_length"]),
        "--output-chunk-length",
        str(params["output_chunk_length"]),
        "--hidden-size",
        str(params["hidden_size"]),
        "--lstm-layers",
        str(params["lstm_layers"]),
        "--attention-heads",
        str(params["attention_heads"]),
        "--dropout",
        str(params["dropout"]),
        "--batch-size",
        str(params["batch_size"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--precision",
        params["precision"],
        "--random-state",
        str(params["random_state"]),
    ]
    if params.get("station_limit"):
        command += ["--station-limit", str(params["station_limit"])]
    if dry_run:
        command.append("--dry-run")
    return command


def start_training_process(command: list[str], run_dir: Path) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (run_dir / "train.log").open("ab", buffering=0)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    log_handle.close()
    return int(process.pid)


def launch_training_run(
    mode: str,
    data_file,
    metadata_file,
    replace_artifacts: bool,
    params: dict,
) -> tuple[int, Path]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = TRAINING_RUNS_DIR / run_id
    output_dir = DEFAULT_CONFIG.artifacts_dir if replace_artifacts else run_dir / "artifacts"
    data_path = DEFAULT_CONFIG.data_path
    metadata_path = DEFAULT_CONFIG.metadata_path
    if data_file is not None:
        data_path = save_uploaded_file(data_file, TRAINING_UPLOADS_DIR / run_id)
    if metadata_file is not None:
        metadata_path = save_uploaded_file(metadata_file, TRAINING_UPLOADS_DIR / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    dry_run = mode == "dry_run"
    command = build_train_command(data_path, metadata_path, output_dir, params, dry_run=dry_run)
    pid = start_training_process(command, run_dir)
    status_payload = {
        "pid": pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "data_path": str(data_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "params": params,
        "command": command,
    }
    (run_dir / "status.json").write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return pid, run_dir


def render_training_monitor(selected_run: Path) -> None:
    status = training_status_info(selected_run)
    metrics = training_metrics(selected_run)
    progress = training_progress_state(selected_run, status, metrics)
    status_json = read_json_file(selected_run / "status.json")
    mode_label = "Проверка данных" if status_json.get("mode") == "dry_run" else "Обучение модели"
    train_metrics = metrics[metrics["metric"] == "train_loss"] if not metrics.empty else pd.DataFrame()
    val_metrics = metrics[metrics["metric"] == "val_loss"] if not metrics.empty else pd.DataFrame()
    last_train = train_metrics["loss"].iloc[-1] if not train_metrics.empty else None
    last_val = val_metrics["loss"].iloc[-1] if not val_metrics.empty else None

    st.caption(status["body"])
    st.progress(progress["percent"] / 100, text=f"{progress['percent']}%")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Режим", mode_label)
    if progress["epochs"]:
        m2.metric("Эпоха", f"{progress['epoch']} / {progress['epochs']}")
    else:
        m2.metric("Эпоха", "-")
    if progress["batch"] and progress["batches_total"]:
        m3.metric("Batch", f"{progress['batch']} / {progress['batches_total']}")
    else:
        m3.metric("Batch", "-")
    m4.metric("Train loss", f"{last_train:.4f}" if last_train is not None else "-")

    if last_val is not None:
        st.metric("Validation loss", f"{last_val:.4f}")

    st.plotly_chart(training_loss_chart(metrics), use_container_width=True, key=f"training_loss_{selected_run.name}")
    with st.expander("Параметры запуска", expanded=False):
        st.json(status_json)
    with st.expander("Сырой лог", expanded=False):
        st.code(read_log_tail(selected_run / "train.log"), language="text")


@st.fragment(run_every="5s")
def render_training_monitor_fragment(selected_run_name: str) -> None:
    render_training_monitor(TRAINING_RUNS_DIR / selected_run_name)


def render_training_tab() -> None:
    section_header("Обучение TFT-модели", "Загрузка данных, гиперпараметры и запуск train.py")
    st.caption(
        "Вкладка запускает реальный процесс обучения через текущий Python интерпретатор Streamlit. "
        "Для быстрой проверки используйте dry-run или ограничение числа АЗС."
    )

    upload_col, config_col = st.columns([1, 1])
    with upload_col:
        data_file = st.file_uploader("Новые детальные данные CSV", type=["csv"], key="train_data_upload")
        metadata_file = st.file_uploader("Метаданные АЗС CSV", type=["csv"], key="train_metadata_upload")
        data_source = data_file if data_file is not None else DEFAULT_CONFIG.data_path
        metadata_source = metadata_file if metadata_file is not None else DEFAULT_CONFIG.metadata_path
        data_preview, data_error = preview_csv(data_source)
        metadata_preview, metadata_error = preview_csv(metadata_source, rows=3)
        validation = validate_training_preview(data_preview)
        if data_error:
            st.error(f"Не удалось прочитать данные: {data_error}")
        elif validation["ok"]:
            st.success(f"Данные выглядят пригодными: найдено целей {len(validation['available_targets'])}.")
            st.dataframe(data_preview, use_container_width=True, key="training_data_preview")
        else:
            st.warning(
                "Проверьте CSV: нужны timestamp, station_id, station_name и хотя бы одна целевая колонка из config.py."
            )
            if validation["missing_required"]:
                st.caption(f"Не хватает: {', '.join(validation['missing_required'])}")
        if metadata_error:
            st.warning(f"Метаданные не прочитались: {metadata_error}")
        elif metadata_preview is not None:
            with st.expander("Предпросмотр metadata", expanded=False):
                st.dataframe(metadata_preview, use_container_width=True, key="training_metadata_preview")

    with config_col:
        st.markdown("##### Гиперпараметры")
        p1, p2, p3 = st.columns(3)
        epochs = p1.number_input("Epochs", min_value=1, max_value=200, value=15, step=1, key="train_epochs")
        station_limit = p2.number_input("Лимит АЗС", min_value=0, max_value=25, value=0, step=1, key="train_station_limit")
        batch_size = p3.number_input("Batch size", min_value=1, max_value=512, value=64, step=1, key="train_batch_size")
        c1, c2, c3 = st.columns(3)
        input_chunk = c1.number_input("Input chunk", min_value=24, max_value=1440, value=DEFAULT_CONFIG.input_chunk_length, step=24, key="train_input_chunk")
        output_chunk = c2.number_input("Output chunk", min_value=24, max_value=336, value=DEFAULT_CONFIG.output_chunk_length, step=24, key="train_output_chunk")
        hidden_size = c3.number_input("Hidden size", min_value=8, max_value=512, value=64, step=8, key="train_hidden_size")
        c4, c5, c6 = st.columns(3)
        lstm_layers = c4.number_input("LSTM layers", min_value=1, max_value=4, value=1, step=1, key="train_lstm_layers")
        attention_heads = c5.number_input("Attention heads", min_value=1, max_value=16, value=4, step=1, key="train_attention_heads")
        dropout = c6.slider("Dropout", min_value=0.0, max_value=0.6, value=0.1, step=0.05, key="train_dropout")
        c7, c8, c9 = st.columns(3)
        learning_rate = c7.number_input("Learning rate", min_value=1e-6, max_value=1e-2, value=1e-4, step=1e-5, format="%.6f", key="train_lr")
        weight_decay = c8.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=1e-4, step=1e-5, format="%.6f", key="train_weight_decay")
        random_state = c9.number_input("Seed", min_value=0, max_value=100000, value=42, step=1, key="train_seed")
        precision = st.selectbox("Precision", ["auto", "32-true", "16-mixed", "bf16-mixed"], key="train_precision")
        replace_artifacts = st.checkbox("Сохранять результат в рабочую папку artifacts", value=False, key="train_replace_artifacts")
        if not replace_artifacts:
            st.caption("Если галочка выключена, модель сохранится в отдельную папку run-а и не заменит текущий прогноз.")

    params = {
        "epochs": int(epochs),
        "station_limit": int(station_limit) if int(station_limit) > 0 else None,
        "batch_size": int(batch_size),
        "input_chunk_length": int(input_chunk),
        "output_chunk_length": int(output_chunk),
        "hidden_size": int(hidden_size),
        "lstm_layers": int(lstm_layers),
        "attention_heads": int(attention_heads),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "precision": precision,
        "random_state": int(random_state),
    }

    preview_run_id = "RUN_ID"
    preview_run_dir = TRAINING_RUNS_DIR / preview_run_id
    preview_output_dir = DEFAULT_CONFIG.artifacts_dir if replace_artifacts else preview_run_dir / "artifacts"
    preview_data_path = DEFAULT_CONFIG.data_path if data_file is None else TRAINING_UPLOADS_DIR / preview_run_id / safe_filename(data_file.name)
    preview_metadata_path = DEFAULT_CONFIG.metadata_path if metadata_file is None else TRAINING_UPLOADS_DIR / preview_run_id / safe_filename(metadata_file.name)
    train_command_preview = build_train_command(preview_data_path, preview_metadata_path, preview_output_dir, params, dry_run=False)
    dry_command_preview = build_train_command(preview_data_path, preview_metadata_path, preview_output_dir, params, dry_run=True)
    with st.expander("Команды запуска", expanded=False):
        st.markdown("**Проверка подготовки данных**")
        st.code(" ".join(f'"{part}"' if " " in part else part for part in dry_command_preview), language="powershell")
        st.markdown("**Полное обучение**")
        st.code(" ".join(f'"{part}"' if " " in part else part for part in train_command_preview), language="powershell")

    latest_runs = latest_training_runs()
    active_run = next((path for path in latest_runs if training_status_info(path)["code"] == "running"), None)
    can_start = validation["ok"] and active_run is None
    b1, b2 = st.columns([1, 1])
    dry_clicked = b1.button("Проверить подготовку данных", disabled=not can_start, key="train_dry_run_button")
    train_clicked = b2.button("Запустить обучение", disabled=not can_start, type="primary", key="train_start_button")

    if active_run is not None:
        st.info(f"Уже идёт процесс: {active_run.name}. Монитор ниже обновляется автоматически каждые 5 секунд.")

    if dry_clicked:
        pid, run_dir = launch_training_run("dry_run", data_file, metadata_file, replace_artifacts, params)
        st.success(f"Проверка подготовки данных запущена: PID {pid}. Монитор обновится автоматически.")
        st.session_state["training_run_select"] = run_dir.name

    if train_clicked:
        pid, run_dir = launch_training_run("train", data_file, metadata_file, replace_artifacts, params)
        st.success(f"Обучение запущено: PID {pid}. Монитор обновится автоматически.")
        st.session_state["training_run_select"] = run_dir.name

    st.markdown("##### Последние запуски")
    runs = latest_training_runs()
    if not runs:
        st.caption("Запусков обучения пока нет.")
        return
    selected_run_name = st.selectbox("Run", [path.name for path in runs], key="training_run_select")
    selected_run = TRAINING_RUNS_DIR / selected_run_name
    status_title, _status_body = detect_training_status(selected_run)
    st.metric("Статус", status_title)
    render_training_monitor_fragment(selected_run_name)


def correlation_chart(df: pd.DataFrame) -> go.Figure:
    candidates = [
        "total_fuel_sales",
        "shop_total_revenue",
        "total_traffic",
        "temperature",
        "precipitation_mm",
        "promotion_fuel_active",
        "promotion_shop_active",
        "ad_active",
        "competitor_price_AI92",
        "competitor_price_AI95",
        "competitor_price_DT",
    ]
    cols = [col for col in candidates if col in df.columns]
    corr = df[cols].corr(numeric_only=True).fillna(0)
    fig = px.imshow(corr, color_continuous_scale=[[0, PINK], [0.5, "#111827"], [1, ACCENT]], zmin=-1, zmax=1)
    fig.update_layout(template=plotly_template(), title="Корреляции факторов спроса", height=520)
    return fig


def render_recommendations(data: pd.DataFrame, forecast: pd.DataFrame | None, station_id: int | None) -> None:
    for item in build_recommendations(data, forecast, station_id):
        st.markdown(
            f"""
            <div class="recommendation-card">
                <span class="rec-metric">{item['metric']}</span>
                <div class="rec-title">{item['title']}</div>
                <div class="rec-body">{item['body']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    inject_css()
    data_path = Path(DEFAULT_CONFIG.data_path)
    metadata_path = Path(DEFAULT_CONFIG.metadata_path)
    data, manifest, report = load_dashboard_data(str(data_path), str(metadata_path))

    model_ready = model_artifacts_exist(DEFAULT_CONFIG.artifacts_dir)
    station_labels = (
        data[["station_id", "station_name"]]
        .drop_duplicates()
        .sort_values("station_id")
        .assign(label=lambda x: x["station_id"].astype(str) + " · " + x["station_name"])
    )

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="brand-mark">T</div>
                <div class="brand-name">TFT Intelligence</div>
                <div class="brand-caption">Прогноз, сравнительный анализ и рекомендации для сети АЗС.</div>
                <div class="author-mark">Автор: {PROJECT_AUTHOR}<br/>© 2026. Учебная проектная работа.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        station_label = st.selectbox("АЗС", ["Вся сеть"] + station_labels["label"].tolist())
        metric = st.selectbox("Целевая метрика", [col for col in TARGET_COLUMNS if col in data.columns])
        aggregation = st.segmented_control("Агрегация", ["Час", "День", "Неделя"], default="День")
        horizon = st.select_slider("Горизонт прогноза", options=[24, 72, 168], value=168)
        min_date, max_date = data["timestamp"].dt.date.min(), data["timestamp"].dt.date.max()
        date_range = st.slider(
            "Период",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD.MM.YYYY",
        )

    filtered = data.copy()
    selected_station_id = None
    if station_label != "Вся сеть":
        selected_station_id = int(station_label.split(" · ")[0])
        filtered = filtered[filtered["station_id"] == selected_station_id]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered = filtered[(filtered["timestamp"] >= start) & (filtered["timestamp"] < end)]

    status_text = "Артефакты TFT доступны" if model_ready else "TFT не загружена: показан baseline"
    dot_class = "dot" if model_ready else "dot off"
    st.markdown(
        f"""
        <div class="top-shell">
            <div class="hero-panel">
                <div class="hero-kicker">Задание: TFT-анализ сети АЗС · Автор: {PROJECT_AUTHOR}</div>
                <h1>{PAGE_TITLE}</h1>
                <div class="subtitle">Факт продаж топлива и товаров, факторы спроса, прогнозы и рекомендации.</div>
                <div class="accent-line"></div>
                <div class="subtitle">Данные: {report['stations']} АЗС · {report['rows']:,} почасовых строк · {len(manifest['targets'])} целевых метрик · {len(manifest['covariates'])} факторов спроса</div>
                <div class="ownership-note">{PROJECT_COPYRIGHT}</div>
            </div>
            <div class="command-stack">
                <div class="status-pill"><span class="{dot_class}"></span>{status_text}</div>
                <div class="command-chip"><div class="chip-label">Период данных</div><div class="chip-value">{report['start'][:10]} → {report['end'][:10]}</div></div>
                <div class="command-chip"><div class="chip-label">Горизонт</div><div class="chip-value">{horizon} часов</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        card("Нет данных после фильтрации", "Ослабьте фильтры в левой панели.")
        return

    analysis_filtered = prepare_analysis_frame(filtered, manifest)
    current_end = filtered["timestamp"].max()
    current = filtered[filtered["timestamp"] > current_end - pd.Timedelta(days=14)]
    kpis = [
        ("Топливо", format_number(current["total_fuel_sales"].sum(), " л"), "последние 14 дней выборки"),
        ("Магазин", format_number(current["shop_total_revenue"].sum(), " ₽"), "выручка сопутствующих товаров"),
        ("Трафик", format_number(current["total_traffic"].sum()), "все типы транспорта"),
        ("АЗС", str(current["station_id"].nunique()), "станций в текущем срезе"),
        ("Промо + реклама", f"{current[['promotion_fuel_active','ad_active']].max(axis=1).mean()*100:.1f}%", "активные часы в срезе"),
    ]
    st.caption("Карточки ниже - быстрые суммы по фактическим данным последних 14 дней выбранной выборки.")
    st.markdown('<div class="kpi-grid">' + "".join(kpi_card(*item) for item in kpis) + "</div>", unsafe_allow_html=True)

    tab_names = []
    if SHOW_OVERVIEW_TAB:
        tab_names.append("Обзор данных")
    tab_names += ["Прогноз", "Сравнение: топливо", "Сравнение: магазин"]
    if SHOW_TRAINING_TAB:
        tab_names.append("Обучение")
    if SHOW_FACTOR_TABS:
        tab_names += ["Акции и реклама", "Трафик", "Цены"]
    tab_names.append("Рекомендации")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    template_metrics = [metric, "total_fuel_sales", "shop_total_revenue", "total_traffic"]
    metrics = list(dict.fromkeys([col for col in template_metrics if col in filtered.columns]))
    aggregated = aggregate_by_period(filtered, aggregation, metrics)
    forecast, forecast_label, forecast_error = make_dashboard_forecast(data, selected_station_id, horizon, model_ready)
    forecast_scope = f"АЗС {selected_station_id}" if selected_station_id is not None else "вся сеть"

    if SHOW_OVERVIEW_TAB:
        with tabs["Обзор данных"]:
            section_header("Исторический обзор спроса", "Факт по выбранным фильтрам")
            c1, c2 = st.columns([1.35, 1])
            with c1:
                st.plotly_chart(line_chart(aggregated, metric, f"{metric}: динамика"), use_container_width=True)
            with c2:
                st.plotly_chart(heatmap_chart(filtered, metric), use_container_width=True)
            c3, c4 = st.columns([1, 1])
            with c3:
                ranking = station_ranking(filtered, metric)
                fig = px.bar(ranking.head(12), x="value", y="station_name", orientation="h", title="Рейтинг АЗС")
                fig.update_traces(marker_color=BLUE)
                fig.update_layout(template=plotly_template(), height=390, yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            with c4:
                st.plotly_chart(fuel_mix_chart(filtered), use_container_width=True)

    with tabs["Прогноз"]:
        forecast_caption = "Сохраненная TFT-модель" if forecast_label == "TFT forecast" else "Baseline по часовому профилю"
        section_header("Прогноз продаж", forecast_caption)
        if selected_station_id is None:
            station_history = data.groupby("timestamp", as_index=False)[[metric]].sum().tail(24 * 21)
        else:
            station_history = data[data["station_id"] == selected_station_id].tail(24 * 21)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=station_history["timestamp"], y=station_history[metric], mode="lines", name="Факт", line={"color": ACCENT}))
        if not forecast.empty and metric in forecast.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast["timestamp"],
                    y=forecast[metric],
                    mode="lines",
                    name=forecast_label,
                    line={"color": AMBER, "dash": "dash"},
                )
            )
        fig.update_layout(template=plotly_template(), title=f"Прогноз {horizon} ч · {forecast_scope}", height=460)
        st.plotly_chart(fig, use_container_width=True)
        if forecast_label == "TFT forecast":
            card("TFT-прогноз построен", "График и рекомендации используют сохранённые веса и препроцессоры из artifacts.")
        elif forecast_error:
            st.warning("TFT-артефакты найдены, но inference не завершился. Показан baseline.")
            st.caption(f"TFT inference error: {forecast_error}")
        else:
            card(
                "TFT-модель пока не обучена",
                "Показан baseline по часовому профилю последних данных. Для настоящего TFT-прогноза запустите train.py отдельно.",
            )
        section_header("Контекст спроса и конкуренты", "Срезы данных для объяснения спроса и ценовой позиции АЗС")
        st.caption(
            "Контекст спроса сравнивает фактические средние значения за час, а режим конкурентов - цены АЗС "
            "с ценами ближайшего конкурентного окружения. Это полезные срезы данных, но не доказательство причинного эффекта."
        )
        analysis_mode = st.segmented_control(
            "Режим анализа",
            ["Контекст спроса", "Конкуренты АЗС"],
            default="Контекст спроса",
            key="forecast_analysis_mode",
        )
        if analysis_mode == "Контекст спроса":
            holiday_means = holiday_mean_frame(filtered, metric)
            weather_means = factor_mean_frame(filtered, metric, "weather_condition", "Погода")
            marketing_means = marketing_mean_frame(filtered, metric)
            road_means = factor_mean_frame(filtered, metric, "road_type", "Тип дороги")
            road_means = restore_factor_labels(
                road_means,
                "Тип дороги",
                manifest.get("categorical_encoders", {}).get("road_type"),
            )
            factor_views = {
                "Праздники": (holiday_means, "День", "Праздники и обычные дни", AMBER),
                "Погода": (weather_means, "Погода", "Погодные условия", BLUE),
                "Промо и реклама": (marketing_means, "Сценарий", "Промо и реклама", ACCENT),
                "Тип дороги": (road_means, "Тип дороги", "Тип дороги АЗС", VIOLET),
            }
            factor_name = st.selectbox("Что сравнить", list(factor_views), key="forecast_factor_view")
            factor_frame, factor_label, factor_title, factor_color = factor_views[factor_name]
            if factor_frame.empty:
                st.info("Для выбранного среза нет данных.")
            else:
                c1, c2 = st.columns([1.28, 1])
                with c1:
                    st.plotly_chart(
                        factor_mean_chart(factor_frame, factor_label, factor_title, factor_color),
                        use_container_width=True,
                    )
                with c2:
                    factor_table = factor_frame.sort_values("Среднее за час", ascending=False).copy()
                    factor_table["Среднее за час"] = factor_table["Среднее за час"].round(2)
                    st.dataframe(factor_table, hide_index=True, use_container_width=True)
                    st.caption("Таблица рядом показывает, на каком объёме часов рассчитано каждое среднее.")
        else:
            fuel = st.selectbox("Топливо для ценового сравнения", list(COMPETITOR_PRICE_PAIRS), key="competitor_fuel")
            position = competitor_position_frame(filtered, fuel)
            st.caption(
                "Оценка считает среднюю разницу цены выбранной АЗС и цены конкурентов за выбранный период. "
                "Отрицательная разница означает, что станция дешевле конкурентов."
            )
            if fuel.startswith("ДТ "):
                st.caption("Для дизеля выбранная собственная цена сравнивается с общим полем competitor_price_DT.")
            if position.empty:
                st.info("В выбранном срезе нет цен для сравнения с конкурентами.")
            else:
                c1, c2 = st.columns([1.18, 1])
                with c1:
                    st.plotly_chart(competitor_position_chart(position, fuel), use_container_width=True)
                with c2:
                    competitor_table = position[
                        [
                            "station_name",
                            "Своя цена, руб",
                            "Цена конкурентов, руб",
                            "Разница, руб",
                            "Разница, %",
                            "Оценка",
                            "Наблюдений, ч",
                        ]
                    ].copy()
                    for col in ["Своя цена, руб", "Цена конкурентов, руб", "Разница, руб", "Разница, %"]:
                        competitor_table[col] = competitor_table[col].round(2)
                    st.dataframe(
                        competitor_table.rename(columns={"station_name": "АЗС"}),
                        hide_index=True,
                        use_container_width=True,
                    )

                station_options = (
                    position[["station_id", "station_name"]]
                    .sort_values("station_id")
                    .assign(label=lambda x: x["station_id"].astype(str) + " · " + x["station_name"])
                )
                competitor_station_id = selected_station_id
                if competitor_station_id is None:
                    competitor_station_label = st.selectbox(
                        "АЗС для динамики цен",
                        station_options["label"].tolist(),
                        key="competitor_station_detail",
                    )
                    competitor_station_id = int(competitor_station_label.split(" · ")[0])
                station_position = position[position["station_id"].eq(competitor_station_id)]
                if not station_position.empty:
                    station_row = station_position.iloc[0]
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Своя средняя цена", f"{station_row['Своя цена, руб']:.2f} руб.")
                    s2.metric("Конкуренты", f"{station_row['Цена конкурентов, руб']:.2f} руб.")
                    s3.metric("Разница", f"{station_row['Разница, руб']:+.2f} руб.", station_row["Оценка"])
                    st.plotly_chart(
                        competitor_price_trend(filtered, int(competitor_station_id), fuel),
                        use_container_width=True,
                    )

    with tabs["Сравнение: топливо"]:
        section_header("Сравнительный анализ топлива", "Продажи по видам топлива, прогнозный состав и сценарии факторов")
        fuel_cols = [col for col in FUEL_TARGET_COLUMNS if col in filtered.columns]
        fuel_agg = aggregate_by_period(filtered, aggregation, fuel_cols)
        fuel_future = aggregate_by_period(forecast, aggregation, fuel_cols) if not forecast.empty else pd.DataFrame()
        fig = history_with_forecast_chart(
            fuel_agg,
            fuel_future,
            fuel_cols,
            "Продажи по видам топлива: факт и прогноз",
            "литры",
            [BLUE, ACCENT, PINK, "#FF3B3B", "#7EE787", "#35C2FF", AMBER],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtered[fuel_cols].sum().sort_values(ascending=False).to_frame("liters"), use_container_width=True)
        fuel_forecast = forecast_breakdown(forecast, fuel_cols, "sales_", "Прогноз, л")
        if not fuel_forecast.empty:
            st.caption(f"Прогнозный состав спроса на горизонт {horizon} ч · {forecast_scope}.")
            st.dataframe(
                fuel_forecast[["name", "Прогноз, л", "Доля"]].rename(columns={"name": "Вид топлива"}),
                hide_index=True,
                use_container_width=True,
            )
        render_demand_explorer(
            analysis_filtered,
            ["total_fuel_sales"] + fuel_cols,
            fuel_cols,
            "Сценарный анализ топлива",
            " л",
            BLUE,
            "fuel_explorer",
        )

    with tabs["Сравнение: магазин"]:
        section_header("Сравнительный анализ магазина", "Выручка, категории, прогнозный состав и сценарии факторов")
        shop_cols = [col for col in SHOP_TARGET_COLUMNS if col in filtered.columns]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            shop_agg = aggregate_by_period(filtered, aggregation, ["shop_total_revenue"] + shop_cols)
            shop_future = (
                aggregate_by_period(forecast, aggregation, ["shop_total_revenue"])
                if not forecast.empty and "shop_total_revenue" in forecast.columns
                else pd.DataFrame()
            )
            fig = history_with_forecast_chart(
                shop_agg,
                shop_future,
                ["shop_total_revenue"],
                "Выручка магазина: факт и прогноз",
                "рубли",
                [VIOLET],
            )
            fig.update_layout(height=430, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(values=filtered[shop_cols].sum().values, names=[c.replace("shop_", "") for c in shop_cols], hole=.58)
            fig.update_layout(template=plotly_template(), title="Категории магазина", height=430)
            st.plotly_chart(fig, use_container_width=True)
        shop_forecast = forecast_breakdown(forecast, shop_cols, "shop_", "Прогноз, ₽")
        if not shop_forecast.empty:
            st.caption(f"Прогноз выручки категорий на горизонт {horizon} ч · {forecast_scope}.")
            st.dataframe(
                shop_forecast[["name", "Прогноз, ₽", "Доля"]].rename(columns={"name": "Категория"}),
                hide_index=True,
                use_container_width=True,
            )
        render_demand_explorer(
            analysis_filtered,
            ["shop_total_revenue"] + shop_cols,
            shop_cols,
            "Сценарный анализ магазина",
            " ₽",
            VIOLET,
            "shop_explorer",
        )

    if SHOW_FACTOR_TABS:
        with tabs["Акции и реклама"]:
            section_header("Акции и рекламная активность", "Факторы спроса из данных")
            promo_cols = [col for col in ["promotion_fuel_active", "promotion_shop_active", "promotion_cafe_active", "ad_active"] if col in filtered]
            promo = filtered.groupby("timestamp", as_index=False)[promo_cols + ["total_fuel_sales"]].mean()
            fig = px.line(promo, x="timestamp", y=promo_cols, title="Активность промо и рекламы")
            fig.update_layout(template=plotly_template(), height=380)
            st.plotly_chart(fig, use_container_width=True)
            impact = filtered.groupby("ad_active", as_index=False).agg(total_fuel_sales=("total_fuel_sales", "mean"), shop_total_revenue=("shop_total_revenue", "mean"))
            st.dataframe(impact, use_container_width=True)

        with tabs["Трафик"]:
            section_header("Трафик по типам транспорта", "Фактор спроса из данных")
            traffic_cols = [col for col in filtered.columns if col.startswith("traffic_") and col != "traffic_Undefined"]
            c1, c2 = st.columns([1.15, 1])
            with c1:
                traffic_agg = aggregate_by_period(filtered, aggregation, traffic_cols)
                fig = px.area(traffic_agg, x="timestamp", y=traffic_cols, title="Трафик по типам транспорта")
                fig.update_layout(template=plotly_template(), height=430)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.scatter(
                    filtered.sample(min(7000, len(filtered)), random_state=42),
                    x="total_traffic",
                    y="total_fuel_sales",
                    color="station_name",
                    title="Трафик → продажи топлива",
                )
                fig.update_layout(template=plotly_template(), height=430, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tabs["Цены"]:
            section_header("Цены и конкурентная среда", "Факторы спроса из данных")
            price_cols = [col for col in filtered.columns if col.startswith("price_") or col.startswith("competitor_price_")]
            price_agg = filtered.set_index("timestamp").groupby(pd.Grouper(freq="D"))[price_cols].mean().reset_index()
            fig = px.line(price_agg, x="timestamp", y=price_cols[:8], title="Собственные цены и цены конкурентов")
            fig.update_layout(template=plotly_template(), height=440)
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(correlation_chart(filtered.sample(min(20000, len(filtered)), random_state=7)), use_container_width=True)

    if SHOW_TRAINING_TAB:
        with tabs["Обучение"]:
            render_training_tab()

    with tabs["Рекомендации"]:
        section_header("Рекомендации", "Правила по факту и прогнозу")
        st.columns([1, 1])
        render_recommendations(filtered, forecast, selected_station_id)
        if SHOW_PREFLIGHT_SUMMARY:
            with st.expander("Preflight summary", expanded=False):
                st.json({"data_report": report, "targets": manifest["targets"], "covariates": manifest["covariates"][:40]})
    st.markdown(f'<div class="footer-author">{PROJECT_COPYRIGHT}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
