from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


TARGET_COLUMNS = [
    "total_fuel_sales",
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
    "shop_total_revenue",
    "shop_напитки",
    "shop_закуски",
    "shop_автотовары",
    "shop_кофе",
    "shop_табак",
]

STATIC_COLUMNS = [
    "road_type",
    "direction",
    "settlement_size",
    "distance_to_city_km",
    "total_pumps",
    "shop_area_m2",
    "has_car_wash",
    "has_cafe",
    "has_shop",
    "competitors_within_5km",
    "corporate_customer_ratio",
    "staff_engagement_score",
    "customer_loyalty_score",
]

KNOWN_DYNAMIC_COLUMNS = [
    "hour",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "is_holiday",
    "is_rush_hour",
    "is_night",
]

OBSERVED_DYNAMIC_COLUMNS = [
    "temperature",
    "precipitation_mm",
    "visibility_km",
    "wind_speed_ms",
    "is_snow",
    "is_rain",
    "is_fog",
    "traffic_Passengers_cars",
    "traffic_Truck_short",
    "traffic_Truck",
    "traffic_Truck_long",
    "traffic_Transporter",
    "traffic_Undefined",
    "total_traffic",
    "promotion_fuel_active",
    "promotion_shop_active",
    "promotion_cafe_active",
    "ad_active",
    "competitor_price_AI92",
    "competitor_price_AI95",
    "competitor_price_DT",
    "price_AI92",
    "price_AI95",
    "price_AI98",
    "price_DT_EURO",
    "price_DT_TANEKO",
    "price_DT_SUMMER",
    "price_DT_WINTER",
]

CATEGORICAL_COLUMNS = [
    "road_type",
    "direction",
    "settlement_size",
    "weather_condition",
    "ad_channel",
    "season",
    "holiday_name",
]

FUEL_TARGET_COLUMNS = [
    "sales_AI92",
    "sales_AI95",
    "sales_AI98",
    "sales_DT_EURO",
    "sales_DT_TANEKO",
    "sales_DT_SUMMER",
    "sales_DT_WINTER",
]

SHOP_TARGET_COLUMNS = [
    "shop_напитки",
    "shop_закуски",
    "shop_автотовары",
    "shop_кофе",
    "shop_табак",
]


@dataclass(frozen=True)
class ProjectConfig:
    data_path: Path = ROOT_DIR / "detailed_data.csv"
    metadata_path: Path = ROOT_DIR / "stations_metadata.csv"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    timestamp_col: str = "timestamp"
    station_id_col: str = "station_id"
    station_name_col: str = "station_name"
    frequency: str = "h"
    input_chunk_length: int = 336
    output_chunk_length: int = 168
    forecast_horizon: int = 168
    targets: list[str] = field(default_factory=lambda: TARGET_COLUMNS.copy())
    static_columns: list[str] = field(default_factory=lambda: STATIC_COLUMNS.copy())
    known_dynamic_columns: list[str] = field(default_factory=lambda: KNOWN_DYNAMIC_COLUMNS.copy())
    observed_dynamic_columns: list[str] = field(default_factory=lambda: OBSERVED_DYNAMIC_COLUMNS.copy())
    categorical_columns: list[str] = field(default_factory=lambda: CATEGORICAL_COLUMNS.copy())


DEFAULT_CONFIG = ProjectConfig()
