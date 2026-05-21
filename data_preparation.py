from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_CONFIG, ProjectConfig


def load_source_data(
    data_path: str | Path = DEFAULT_CONFIG.data_path,
    metadata_path: str | Path = DEFAULT_CONFIG.metadata_path,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(data_path, parse_dates=[config.timestamp_col])
    metadata = pd.read_csv(metadata_path)
    return data, metadata


def validate_source_data(data: pd.DataFrame, config: ProjectConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    required = [config.timestamp_col, config.station_id_col, config.station_name_col]
    missing_required = [col for col in required if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    available_targets = [col for col in config.targets if col in data.columns]
    missing_targets = [col for col in config.targets if col not in data.columns]
    rows_per_station = data.groupby(config.station_id_col).size()
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "stations": int(data[config.station_id_col].nunique()),
        "start": str(data[config.timestamp_col].min()),
        "end": str(data[config.timestamp_col].max()),
        "rows_per_station_min": int(rows_per_station.min()),
        "rows_per_station_max": int(rows_per_station.max()),
        "available_targets": available_targets,
        "missing_targets": missing_targets,
    }


def enrich_time_features(df: pd.DataFrame, config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[config.timestamp_col])
    defaults = {
        "hour": ts.dt.hour,
        "day_of_week": ts.dt.dayofweek,
        "week_of_year": ts.dt.isocalendar().week.astype(int),
        "month": ts.dt.month,
        "quarter": ts.dt.quarter,
        "is_weekend": (ts.dt.dayofweek >= 5).astype(int),
        "is_rush_hour": ts.dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int),
        "is_night": ts.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(int),
    }
    for col, values in defaults.items():
        if col not in out.columns:
            out[col] = values
    if "is_holiday" not in out.columns:
        out["is_holiday"] = 0
    return out


def encode_categoricals(
    df: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    out = df.copy()
    encoders: dict[str, dict[str, int]] = {}
    for col in config.categorical_columns:
        if col not in out.columns:
            continue
        values = out[col].fillna("unknown").astype(str)
        categories = sorted(values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(categories)}
        out[f"{col}_encoded"] = values.map(mapping).astype("int32")
        encoders[col] = mapping
    return out, encoders


def prepare_model_frame(
    data: pd.DataFrame,
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = enrich_time_features(data, config)
    df, encoders = encode_categoricals(df, config)
    df = df.sort_values([config.station_id_col, config.timestamp_col]).drop_duplicates(
        [config.station_id_col, config.timestamp_col]
    )

    available_targets = [col for col in config.targets if col in df.columns]
    if not available_targets:
        raise ValueError("No configured target columns were found in the data.")

    for col in available_targets:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    numeric_candidates = (
        config.known_dynamic_columns
        + config.observed_dynamic_columns
        + [f"{col}_encoded" for col in config.categorical_columns]
    )
    covariates = [col for col in numeric_candidates if col in df.columns and col not in available_targets]
    for col in covariates:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    static_columns = [col for col in config.static_columns if col in df.columns]
    for col in static_columns:
        if df[col].dtype == "object":
            encoded_col = f"{col}_encoded"
            if encoded_col in df.columns:
                df[col] = df[encoded_col]
            else:
                df[col] = pd.factorize(df[col].fillna("unknown").astype(str))[0]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    manifest = {
        "config": _jsonable_config(config),
        "targets": available_targets,
        "covariates": covariates,
        "static_columns": static_columns,
        "categorical_encoders": encoders,
        "validation": validate_source_data(df, config),
    }
    return df, manifest


def build_darts_series(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    config: ProjectConfig = DEFAULT_CONFIG,
    station_limit: int | None = None,
):
    try:
        from darts import TimeSeries
    except ImportError as exc:
        raise RuntimeError("Darts is not installed. Install dependencies from requirements.txt first.") from exc

    targets = manifest["targets"]
    covariates = manifest["covariates"]
    static_columns = manifest["static_columns"]
    station_ids = sorted(df[config.station_id_col].unique().tolist())
    if station_limit:
        station_ids = station_ids[:station_limit]

    series_list = []
    covariate_list = []
    for station_id in station_ids:
        station_df = df[df[config.station_id_col] == station_id].sort_values(config.timestamp_col)
        target_ts = TimeSeries.from_dataframe(
            station_df,
            time_col=config.timestamp_col,
            value_cols=targets,
            fill_missing_dates=True,
            freq=config.frequency,
        )
        cov_ts = TimeSeries.from_dataframe(
            station_df,
            time_col=config.timestamp_col,
            value_cols=covariates,
            fill_missing_dates=True,
            freq=config.frequency,
        )
        if static_columns:
            static_df = station_df[static_columns].iloc[[0]].reset_index(drop=True)
            target_ts = target_ts.with_static_covariates(static_df)
        series_list.append(target_ts)
        covariate_list.append(cov_ts)

    return series_list, covariate_list, station_ids


def save_manifest(manifest: dict[str, Any], artifacts_dir: str | Path) -> Path:
    path = Path(artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    output = path / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_manifest(artifacts_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(artifacts_dir) / "manifest.json").read_text(encoding="utf-8"))


def _jsonable_config(config: ProjectConfig) -> dict[str, Any]:
    raw = asdict(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}
