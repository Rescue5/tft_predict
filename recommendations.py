from __future__ import annotations

import pandas as pd

from config import FUEL_TARGET_COLUMNS, SHOP_TARGET_COLUMNS


def build_recommendations(
    data: pd.DataFrame,
    forecast: pd.DataFrame | None = None,
    station_id: int | None = None,
) -> list[dict[str, str]]:
    frame = data if station_id is None else data[data["station_id"] == station_id]
    if frame.empty:
        return []

    latest = frame.sort_values("timestamp").tail(24 * 14)
    previous = frame.sort_values("timestamp").iloc[-24 * 28 : -24 * 14]
    recommendations: list[dict[str, str]] = []

    total_now = latest["total_fuel_sales"].sum()
    total_prev = previous["total_fuel_sales"].sum() if not previous.empty else total_now
    traffic_now = latest["total_traffic"].sum() if "total_traffic" in latest else 0
    traffic_prev = previous["total_traffic"].sum() if not previous.empty and "total_traffic" in previous else traffic_now

    if total_prev and total_now / total_prev < 0.92:
        recommendations.append(
            {
                "level": "critical",
                "title": "Проверить просадку продаж топлива",
                "metric": f"{(total_now / total_prev - 1) * 100:.1f}%",
                "body": "Последние 14 дней ниже предыдущего периода. Проверьте цены конкурентов, рекламу и часы с низким трафиком.",
            }
        )

    if traffic_prev and traffic_now / traffic_prev > 1.08 and total_prev and total_now / total_prev < 1.03:
        recommendations.append(
            {
                "level": "warning",
                "title": "Усилить конверсию трафика в продажи",
                "metric": f"+{(traffic_now / traffic_prev - 1) * 100:.1f}% трафика",
                "body": "Трафик растёт быстрее продаж. Имеет смысл проверить выкладку, персонал и наличие популярных видов топлива.",
            }
        )

    if forecast is not None and not forecast.empty:
        fuel_cols = [col for col in FUEL_TARGET_COLUMNS if col in forecast.columns]
        if fuel_cols:
            top_fuel = forecast[fuel_cols].sum().sort_values(ascending=False).index[0]
            recommendations.append(
                {
                    "level": "success",
                    "title": f"Подготовить запас {top_fuel.replace('sales_', '')}",
                    "metric": f"{forecast[top_fuel].sum():,.0f} л".replace(",", " "),
                    "body": "На прогнозном горизонте этот вид топлива даёт максимальный вклад в спрос.",
                }
            )

    shop_cols = [col for col in SHOP_TARGET_COLUMNS if col in latest.columns]
    if shop_cols and latest[shop_cols].sum().sum() > 0:
        top_shop = latest[shop_cols].sum().sort_values(ascending=False).index[0]
        recommendations.append(
            {
                "level": "info",
                "title": f"Развить категорию {top_shop.replace('shop_', '')}",
                "metric": f"{latest[top_shop].sum():,.0f} ₽".replace(",", " "),
                "body": "Категория лидирует по выручке за последние две недели. Проверьте акционные связки с топливом.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "level": "info",
                "title": "Сеть работает стабильно",
                "metric": "OK",
                "body": "Критичных отклонений в последних периодах не обнаружено. Используйте прогноз после обучения TFT.",
            }
        )
    return recommendations[:5]
