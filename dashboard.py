from __future__ import annotations

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
# These factor tabs cover assignment inputs. Keep them on for the full task view.
SHOW_FACTOR_TABS = False
ACCENT = "#20E3B2"
BLUE = "#5B8CFF"
AMBER = "#FFB020"
PINK = "#FF5C7A"
VIOLET = "#9B8CFF"
TEXT = "#E6EDF7"
MUTED = "#8A96A8"


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
            """
            <div class="sidebar-brand">
                <div class="brand-mark">T</div>
                <div class="brand-name">TFT Intelligence</div>
                <div class="brand-caption">Фильтры для анализа продаж, факторов спроса, прогноза и рекомендаций.</div>
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
                <div class="hero-kicker">Задание: TFT-анализ сети АЗС</div>
                <h1>{PAGE_TITLE}</h1>
                <div class="subtitle">Факт продаж топлива и товаров, факторы спроса, прогнозы и рекомендации.</div>
                <div class="accent-line"></div>
                <div class="subtitle">Данные: {report['stations']} АЗС · {report['rows']:,} почасовых строк · {len(manifest['targets'])} целевых метрик · {len(manifest['covariates'])} факторов спроса</div>
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

    current_end = filtered["timestamp"].max()
    current = filtered[filtered["timestamp"] > current_end - pd.Timedelta(days=14)]
    kpis = [
        ("Топливо", format_number(current["total_fuel_sales"].sum(), " л"), "последние 14 дней выборки"),
        ("Магазин", format_number(current["shop_total_revenue"].sum(), " ₽"), "выручка сопутствующих товаров"),
        ("Трафик", format_number(current["total_traffic"].sum()), "все типы транспорта"),
        ("АЗС", str(current["station_id"].nunique()), "станций в текущем срезе"),
        ("Промо + реклама", f"{current[['promotion_fuel_active','ad_active']].max(axis=1).mean()*100:.1f}%", "активные часы в срезе"),
    ]
    st.caption("Карточки ниже - быстрые суммы по фактическим данным последних 14 дней выбранной выборки. Это не прогноз TFT.")
    st.markdown('<div class="kpi-grid">' + "".join(kpi_card(*item) for item in kpis) + "</div>", unsafe_allow_html=True)

    tab_names = ["Обзор данных", "Прогноз", "Топливо", "Магазин"]
    if SHOW_FACTOR_TABS:
        tab_names += ["Акции и реклама", "Трафик", "Цены"]
    tab_names.append("Рекомендации")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))
    template_metrics = [metric, "total_fuel_sales", "shop_total_revenue", "total_traffic"]
    metrics = list(dict.fromkeys([col for col in template_metrics if col in filtered.columns]))
    aggregated = aggregate_by_period(filtered, aggregation, metrics)
    forecast, forecast_label, forecast_error = make_dashboard_forecast(data, selected_station_id, horizon, model_ready)
    forecast_scope = f"АЗС {selected_station_id}" if selected_station_id is not None else "вся сеть"

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

    with tabs["Топливо"]:
        section_header("Продажи по видам топлива", "Целевая часть задания")
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

    with tabs["Магазин"]:
        section_header("Сопутствующие товары", "История выручки и категорий")
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

    with tabs["Рекомендации"]:
        section_header("Рекомендации", "Правила по факту и прогнозу")
        st.columns([1, 1])
        render_recommendations(filtered, forecast, selected_station_id)
        with st.expander("Preflight summary", expanded=False):
            st.json({"data_report": report, "targets": manifest["targets"], "covariates": manifest["covariates"][:40]})


if __name__ == "__main__":
    main()
