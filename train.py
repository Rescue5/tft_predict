from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

from config import DEFAULT_CONFIG, ProjectConfig
from data_preparation import build_darts_series, load_source_data, prepare_model_frame, save_manifest

import shutil
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Darts TFT model for Tatneft gas station analytics.")
    parser.add_argument("--data", default=str(DEFAULT_CONFIG.data_path), help="Path to data/detailed_data.csv")
    parser.add_argument("--metadata", default=str(DEFAULT_CONFIG.metadata_path), help="Path to data/stations_metadata.csv")
    parser.add_argument("--output", default=str(DEFAULT_CONFIG.artifacts_dir), help="Artifacts directory")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs. Keep small for demo runs.")
    parser.add_argument("--station-limit", type=int, default=None, help="Optional station limit for smoke training.")
    parser.add_argument("--input-chunk-length", type=int, default=DEFAULT_CONFIG.input_chunk_length)
    parser.add_argument("--output-chunk-length", type=int, default=DEFAULT_CONFIG.output_chunk_length)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "32-true", "16-mixed", "bf16-mixed"],
        help="Lightning precision. auto uses bf16-mixed on supported CUDA GPUs and 32-true otherwise.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Prepare series and write manifest without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_author = os.getenv("TFT_PROJECT_AUTHOR", "Журавлев Данил Артемович")
    config = ProjectConfig(
        data_path=Path(args.data),
        metadata_path=Path(args.metadata),
        artifacts_dir=Path(args.output),
        input_chunk_length=args.input_chunk_length,
        output_chunk_length=args.output_chunk_length,
        forecast_horizon=args.output_chunk_length,
    )
    data, _metadata = load_source_data(config.data_path, config.metadata_path, config)
    frame, manifest = prepare_model_frame(data, config)
    manifest["training_params"] = {
        "input_chunk_length": args.input_chunk_length,
        "output_chunk_length": args.output_chunk_length,
        "hidden_size": args.hidden_size,
        "lstm_layers": args.lstm_layers,
        "attention_heads": args.attention_heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "random_state": args.random_state,
    }
    manifest["author"] = project_author
    manifest["copyright"] = f"Авторская проектная работа: {project_author}. Использование без указания автора запрещено."
    manifest_path = save_manifest(manifest, config.artifacts_dir)

        # перед tb_logger = TensorBoardLogger(...)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    tb_log_dir = config.artifacts_dir / "tb_logs" / "tft_station_model"

    # если там файл или битая директория — удалить
    if tb_log_dir.exists() and not tb_log_dir.is_dir():
        tb_log_dir.unlink()  # удалить файл
    # на всякий случай можно подчистить старое дерево логов
    os.makedirs(tb_log_dir, exist_ok=True)

    series_list, covariate_list, station_ids = build_darts_series(
        frame, manifest, config, station_limit=args.station_limit
    )
    manifest["station_ids"] = station_ids
    save_manifest(manifest, config.artifacts_dir)

    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
        return

    try:
        import torch
        from darts.dataprocessing.transformers import Scaler, StaticCovariatesTransformer
        from darts.models import TFTModel
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger

        tb_logger = TensorBoardLogger(
            save_dir=str(config.artifacts_dir / "tb_logs"),
            name="tft_station_model",
        )
    except ImportError as exc:
        raise RuntimeError("Install TFT dependencies first: pip install -r requirements.txt") from exc

    static_transformer = StaticCovariatesTransformer()
    encoded_series = static_transformer.fit_transform(series_list)

    target_scaler = Scaler()
    covariate_scaler = Scaler()
    scaled_targets = [ts.astype("float32") for ts in target_scaler.fit_transform(encoded_series)]
    scaled_covariates = [ts.astype("float32") for ts in covariate_scaler.fit_transform(covariate_list)]

    train_len = int(0.8 * min(len(ts) for ts in scaled_targets))
    train_targets = [ts[:train_len] for ts in scaled_targets]
    val_targets = [ts[train_len:] for ts in scaled_targets]
    train_covariates = [ts[:train_len] for ts in scaled_covariates]
    val_covariates = [ts[train_len:] for ts in scaled_covariates]

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    precision = args.precision
    if precision == "auto" and accelerator == "cuda":
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "32-true"
    if precision == "auto":
        precision = "32-true"
    if accelerator == "cuda":
        torch.set_float32_matmul_precision("medium")

    tb_logger = TensorBoardLogger(
        save_dir=str(config.artifacts_dir / "tb_logs"),
        name="tft_station_model",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.artifacts_dir / "checkpoints"),
        filename="tft-epoch={epoch:02d}-val_loss={val_loss:.4f}",
        save_top_k=-1,      # сохранять все чекпоинты
        every_n_epochs=1,
        monitor="val_loss",
    )

    model = TFTModel(
        input_chunk_length=config.input_chunk_length,
        output_chunk_length=config.output_chunk_length,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        num_attention_heads=args.attention_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        optimizer_kwargs={"lr": args.learning_rate, "weight_decay": args.weight_decay},
        add_relative_index=True,
        add_encoders={"cyclic": {"future": ["hour", "dayofweek", "month"]}},
        use_static_covariates=True,
        random_state=args.random_state,
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
            "enable_progress_bar": True,
            "enable_checkpointing": True,
            "callbacks": [checkpoint_callback],
            "logger": tb_logger,
        },
    )
    model.fit(
        series=train_targets,
        past_covariates=train_covariates,
        val_series=val_targets,
        val_past_covariates=val_covariates,
        verbose=True,
    )

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(config.artifacts_dir / "tft_station_model.pkl"))
    with (config.artifacts_dir / "preprocessors.pkl").open("wb") as handle:
        pickle.dump(
            {
                "target_scaler": target_scaler,
                "covariate_scaler": covariate_scaler,
                "static_transformer": static_transformer,
            },
            handle,
        )
    print(json.dumps({"status": "trained", "artifacts": str(config.artifacts_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":

    main()
