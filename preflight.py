from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DEFAULT_CONFIG, ProjectConfig
from data_preparation import load_source_data, prepare_model_frame, save_manifest, validate_source_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate data and prepare TFT manifest without training.")
    parser.add_argument("--data", default=str(DEFAULT_CONFIG.data_path))
    parser.add_argument("--metadata", default=str(DEFAULT_CONFIG.metadata_path))
    parser.add_argument("--output", default=str(DEFAULT_CONFIG.artifacts_dir))
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(data_path=Path(args.data), metadata_path=Path(args.metadata), artifacts_dir=Path(args.output))
    data, _metadata = load_source_data(config.data_path, config.metadata_path, config)
    frame, manifest = prepare_model_frame(data, config)
    report = validate_source_data(frame, config)
    duplicates = int(frame.duplicated([config.station_id_col, config.timestamp_col]).sum())
    missing_numeric = int(frame.select_dtypes("number").isna().sum().sum())
    output = {
        "status": "ok",
        "report": report,
        "duplicates_station_timestamp": duplicates,
        "missing_numeric_values": missing_numeric,
        "targets": manifest["targets"],
        "covariates_count": len(manifest["covariates"]),
        "static_columns": manifest["static_columns"],
    }
    if args.write_manifest:
        output["manifest_path"] = str(save_manifest(manifest, config.artifacts_dir))
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
