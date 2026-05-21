from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from config import DEFAULT_CONFIG, ProjectConfig
from data_preparation import build_darts_series, load_manifest, load_source_data, prepare_model_frame


def model_artifacts_exist(artifacts_dir: str | Path = DEFAULT_CONFIG.artifacts_dir) -> bool:
    root = Path(artifacts_dir)
    required_files = ["tft_station_model.pkl", "preprocessors.pkl", "manifest.json"]
    return all((root / filename).exists() for filename in required_files)


def make_demo_forecast(data: pd.DataFrame, station_id: int, horizon: int = DEFAULT_CONFIG.forecast_horizon) -> pd.DataFrame:
    station = data[data["station_id"] == station_id].sort_values("timestamp")
    numeric_targets = [col for col in DEFAULT_CONFIG.targets if col in station.columns]
    if station.empty or not numeric_targets:
        return pd.DataFrame()

    tail = station.tail(max(24 * 14, horizon)).copy()
    profile = tail.groupby(tail["timestamp"].dt.hour)[numeric_targets].mean()
    start = station["timestamp"].max() + pd.Timedelta(hours=1)
    future_index = pd.date_range(start=start, periods=horizon, freq="h")
    rows = []
    for ts in future_index:
        values = profile.loc[ts.hour].to_dict() if ts.hour in profile.index else tail[numeric_targets].mean().to_dict()
        rows.append({"timestamp": ts, **values})
    return pd.DataFrame(rows)


def predict_with_saved_model(
    data_path: str | Path = DEFAULT_CONFIG.data_path,
    metadata_path: str | Path = DEFAULT_CONFIG.metadata_path,
    artifacts_dir: str | Path = DEFAULT_CONFIG.artifacts_dir,
    station_id: int | None = None,
    horizon: int = DEFAULT_CONFIG.forecast_horizon,
) -> pd.DataFrame:
    artifacts = Path(artifacts_dir)
    if not model_artifacts_exist(artifacts):
        raise FileNotFoundError(f"Saved TFT artifacts were not found in {artifacts}")

    try:
        from darts.models import TFTModel
    except ImportError as exc:
        raise RuntimeError("Darts is not installed. Install dependencies from requirements.txt first.") from exc

    config = ProjectConfig(data_path=Path(data_path), metadata_path=Path(metadata_path), artifacts_dir=artifacts)
    data, _metadata = load_source_data(config.data_path, config.metadata_path, config)
    frame, _manifest = prepare_model_frame(data, config)
    manifest = load_manifest(artifacts)
    max_direct_horizon = int(manifest.get("config", {}).get("output_chunk_length", config.output_chunk_length))
    if horizon > max_direct_horizon:
        raise ValueError(
            f"Saved TFT inference supports up to {max_direct_horizon} hours without future past covariates; "
            f"got horizon={horizon}."
        )
    if station_id is not None:
        frame = frame[frame[config.station_id_col] == station_id]
    series_list, covariate_list, station_ids = build_darts_series(frame, manifest, config)
    if not series_list:
        raise ValueError(f"No forecast input series were built for station_id={station_id!r}.")

    model = TFTModel.load(str(artifacts / "tft_station_model.pkl"))
    with (artifacts / "preprocessors.pkl").open("rb") as handle:
        preprocessors = pickle.load(handle)

    target_scaler = preprocessors["target_scaler"]
    covariate_scaler = preprocessors["covariate_scaler"]
    static_transformer = preprocessors["static_transformer"]
    trained_station_ids = [int(sid) for sid in manifest.get("station_ids", [])]
    station_indices = None
    if trained_station_ids:
        index_by_station = {sid: idx for idx, sid in enumerate(trained_station_ids)}
        unknown_station_ids = [int(sid) for sid in station_ids if int(sid) not in index_by_station]
        if unknown_station_ids:
            raise ValueError(f"Saved preprocessors do not include station_ids={unknown_station_ids}.")
        station_indices = [index_by_station[int(sid)] for sid in station_ids]

    encoded_series = static_transformer.transform(series_list, series_idx=station_indices)
    scaled_series = [
        ts.astype("float32")
        for ts in target_scaler.transform(encoded_series, series_idx=station_indices)
    ]
    scaled_covariates = [
        ts.astype("float32")
        for ts in covariate_scaler.transform(covariate_list, series_idx=station_indices)
    ]
    prediction = model.predict(n=horizon, series=scaled_series, past_covariates=scaled_covariates)
    forecast = target_scaler.inverse_transform(prediction, series_idx=station_indices)

    frames = []
    for sid, ts in zip(station_ids, forecast):
        if hasattr(ts, "to_dataframe"):
            pdf = ts.to_dataframe().reset_index()
        else:
            pdf = ts.pd_dataframe().reset_index()
        pdf = pdf.rename(columns={"time": "timestamp"})
        if "timestamp" not in pdf.columns:
            pdf = pdf.rename(columns={pdf.columns[0]: "timestamp"})
        pdf["station_id"] = sid
        frames.append(pdf)
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate forecasts using saved TFT artifacts.")
    parser.add_argument("--data", default=str(DEFAULT_CONFIG.data_path))
    parser.add_argument("--metadata", default=str(DEFAULT_CONFIG.metadata_path))
    parser.add_argument("--artifacts", default=str(DEFAULT_CONFIG.artifacts_dir))
    parser.add_argument("--station-id", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.forecast_horizon)
    parser.add_argument("--output", default="forecast.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecast = predict_with_saved_model(args.data, args.metadata, args.artifacts, args.station_id, args.horizon)
    forecast.to_csv(args.output, index=False)
    print(forecast.head().to_string(index=False))


if __name__ == "__main__":
    main()
