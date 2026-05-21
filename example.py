import warnings
warnings.filterwarnings('ignore')

# Установка необходимых библиотек
import subprocess
import sys

def install_packages():
    packages = ['darts', 'pytorch-lightning', 'pandas', 'numpy', 'scikit-learn']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import torch
from datetime import datetime, timedelta
import os

# Параметры
USE_FULL_DATA = False  # True - использовать все 25 АЗС, False - только 5
TARGET_VARIABLES = ['total_fuel_sales', 'sales_AI92', 'sales_AI95', 'sales_DT_EURO', 'shop_total_revenue', 'sales_DT_TANEKO',
                    'sales_DT_SUMMER', 'sales_DT_WINTER', 'shop_напитки', 'shop_закуски',	'shop_автотовары', 'shop_кофе', 'shop_табак'
]
FORECAST_HORIZON = 168  # 7 дней * 24 часа
INPUT_LENGTH = 336  # 14 дней * 24 часа для контекста

# Загрузка данных
def load_and_prepare_data():
    """Загрузка всех CSV файлов и подготовка данных"""

    # Загрузка метаданных
    print("Загрузка метаданных...")
    try:
        metadata_5 = pd.read_csv('5 stations metadata.csv')
        metadata_25 = pd.read_csv('tatneft stations metadata.csv')
    except:
        metadata_5 = pd.read_csv('5_stations_metadata.csv')
        metadata_25 = pd.read_csv('tatneft_stations_metadata.csv')

    # Определение столбца идентификатора
    id_col = 'station_id' if 'station_id' in metadata_5.columns else 'station_name'

    # Загрузка детальных данных
    print("Загрузка детальных данных...")
    try:
        if USE_FULL_DATA:
            detailed_data = pd.read_csv('tatneft detailed data.csv', parse_dates=['timestamp'])
            metadata = metadata_25
        else:
            detailed_data = pd.read_csv('5 stations data.csv', parse_dates=['timestamp'])
            metadata = metadata_5
    except:
        if USE_FULL_DATA:
            detailed_data = pd.read_csv('tatneft_detailed_data.csv', parse_dates=['timestamp'])
            metadata = metadata_25
        else:
            detailed_data = pd.read_csv('5_stations_data.csv', parse_dates=['timestamp'])
            metadata = metadata_5

    print(f"Загружено {len(detailed_data.columns)} записей детальных данных")
    print(f"Загружено {len(metadata.columns)} записей метаданных")

    # Объединение данных
    if id_col in detailed_data.columns:
        data = detailed_data.merge(metadata, on=id_col, how='left', suffixes=('', '_meta'))
        data = data.loc[:, ~data.columns.str.endswith('_meta')]
    else:
        # Если столбец называется по-другому
        name_col = [col for col in detailed_data.columns if 'station' in col.lower() and 'name' in col.lower()]
        if name_col:
            data = detailed_data.merge(metadata, left_on=name_col[0], right_on=id_col, how='left')
        else:
            # Используем station_id если есть в detailed_data
            id_col_detailed = [col for col in detailed_data.columns if 'station' in col.lower()]
            if id_col_detailed:
                data = detailed_data.merge(metadata, left_on=id_col_detailed[0], right_on=id_col, how='left')
            else:
                print("Не удалось найти столбец для объединения. Использую данные без метаданных.")
                data = detailed_data.copy()

    print(f"После объединения: {len(data)} записей, {len(data.columns)} колонок")
    return data, metadata, id_col

# Создание временных признаков
def create_time_features(df):
    """Создание временных признаков из timestamp"""
    print("Создание временных признаков...")

    # Проверяем наличие timestamp
    if 'timestamp' not in df.columns:
        print("Ошибка: столбец 'timestamp' не найден")
        return df

    # Извлекаем признаки
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                          (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_holiday'] = 0  # Можно расширить для конкретных праздников

    # Месяц и день месяца
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day

    return df

# Кодирование категориальных переменных
def encode_categorical_features(df):
    """Кодирование категориальных признаков"""
    print("Кодирование категориальных признаков...")

    categorical_columns = ['road_type', 'direction', 'settlement_size', 'weather_condition', 'ad_channel', 'season']
    label_encoders = {}

    for col in categorical_columns:
        if col in df.columns:
            # Заполняем пропуски
            df[col] = df[col].fillna('unknown')

            # Создаем и применяем LabelEncoder
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Колонка '{col}' закодирована, уникальных значений: {len(le.classes_)}")
        else:
            print(f"  Колонка '{col}' не найдена, пропускаем")

    return df, label_encoders

# Проверка и подготовка целевых переменных
def prepare_target_variables(df):
    """Проверка наличия целевых переменных и их подготовка"""
    print("Проверка целевых переменных...")

    available_targets = []
    for target in TARGET_VARIABLES:
        if target in df.columns:
            # Заполняем пропуски
            df[target] = df[target].fillna(0)
            # Убеждаемся, что нет отрицательных значений
            df[target] = df[target].clip(lower=0)
            available_targets.append(target)
            print(f"  Целевая переменная '{target}' найдена")
        else:
            print(f"  Целевая переменная '{target}' НЕ найдена")

    if not available_targets:
        print("Ошибка: не найдено ни одной целевой переменной")
        # Пробуем найти похожие колонки
        fuel_cols = [col for col in df.columns if 'sales' in col.lower() or 'fuel' in col.lower()]
        if fuel_cols:
            available_targets = fuel_cols[:5]
            print(f"  Используем альтернативные колонки: {available_targets}")

    return df, available_targets

# Подготовка данных для darts
def prepare_darts_data_per_station(df, target_cols, station_id_col='station_id', static_encoding=True):
    """Подготовка данных: список временных рядов по станциям с embedded static covariates"""
    print("Подготовка данных для обучения по станциям...")

    # Сортировка и удаление дубликатов
    df = df.sort_values([station_id_col, 'timestamp']).reset_index(drop=True)
    df = df.drop_duplicates(subset=[station_id_col, 'timestamp'])

    # ---- Статические колонки (проверяем, какие реально существуют) ----
    static_candidates = [
        'road_type', 'direction', 'settlement_size', 'distance_to_city_km',
        'total_pumps', 'shop_area_m2', 'has_car_wash', 'has_cafe', 'has_shop',
        'competitors_within_5km', 'corporate_customer_ratio',
        'staff_engagement_score', 'customer_loyalty_score'
    ]
    # Оставляем только те, что есть в df
    static_cols = [col for col in static_candidates if col in df.columns]
    print(f"  Найдено статических колонок: {len(static_cols)}")

    # ---- Динамические признаки (ковариаты) ----
    base_features = ['hour', 'day_of_week', 'is_weekend', 'is_holiday', 'is_peak_hour', 'month', 'day']
    cat_encoded = [c for c in df.columns if c.endswith('_encoded')]
    feature_cols = base_features + cat_encoded

    # Другие числовые колонки (исключаем целевые, признаки, статику, идентификаторы, время)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(target_cols + feature_cols + static_cols + [station_id_col, 'timestamp'])
    other_num = [c for c in numeric_cols if c not in exclude]
    other_num = other_num[:75]
    feature_cols.extend(other_num)
    print(f"  Всего динамических признаков (ковариат): {len(feature_cols)}")

    # ---- Кодирование статических категориальных колонок ----
    label_encoders = {}
    if static_encoding:
        for col in static_cols:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                print(f"  Закодирована статическая колонка '{col}'")

    # ---- Создание списков рядов ----
    series_list = []
    cov_list = []
    station_ids = df[station_id_col].unique()

    for sid in station_ids:
        station_data = df[df[station_id_col] == sid].copy().sort_values('timestamp')

        # Целевые переменные (многомерный ряд)
        target_ts = TimeSeries.from_dataframe(
            station_data, 'timestamp', target_cols, fill_missing_dates=True, freq='h'
        )

        # Ковариаты (динамические признаки)
        cov_ts = TimeSeries.from_dataframe(
            station_data, 'timestamp', feature_cols, fill_missing_dates=True, freq='h'
        )

        # Добавляем статические признаки в целевой ряд
        static_vals = station_data[static_cols].iloc[0].to_dict()
        static_df = pd.DataFrame([static_vals])
        target_ts = target_ts.with_static_covariates(static_df)

        series_list.append(target_ts)
        cov_list.append(cov_ts)

    print(f"  Создано рядов для станций: {len(series_list)}")
    return series_list, cov_list, feature_cols, station_ids

from darts.dataprocessing.transformers import StaticCovariatesTransformer
# Обучение модели
def train_tft_model_on_stations(series_list, cov_list, target_cols):
    """Обучение TFT на нескольких станциях со встроенными статическими ковариатами"""
    print("\n=== ОБУЧЕНИЕ TFT (МНОГОСТАНЦИОННОЕ) ===")

    # ---------- ПРИМЕНЯЕМ STATIC COVARIATES TRANSFORMER ----------
    # Преобразуем все статические ковариаты в числовой формат
    print("Преобразование статических ковариат в числовой формат...")
    static_transformer = StaticCovariatesTransformer()
    series_list_encoded = static_transformer.fit_transform(series_list)
    print(f"  Статические ковариаты обработаны")

    # ---------- МАСШТАБИРОВАНИЕ ----------
    print("Масштабирование данных...")
    target_scaler = Scaler()
    scaled_targets = target_scaler.fit_transform(series_list_encoded)

    cov_scaler = Scaler()
    scaled_covs = cov_scaler.fit_transform(cov_list)
    print(f"  Масштабирование завершено")

    # ---------- РАЗДЕЛЕНИЕ НА TRAIN/VAL ----------
    # Определяем длину тренировочной выборки (80% от длины первого ряда)
    train_len = int(0.8 * len(scaled_targets[0]))
    print(f"  Train: {train_len} точек, Val: {len(scaled_targets[0]) - train_len} точек")

    train_targets = [ts[:train_len] for ts in scaled_targets]
    val_targets   = [ts[train_len:] for ts in scaled_targets]
    train_covs    = [cv[:train_len] for cv in scaled_covs]
    val_covs      = [cv[train_len:] for cv in scaled_covs]

    # ---------- СОЗДАНИЕ МОДЕЛИ ----------
    print("Создание модели TFT...")

    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Используем устройство: {device}")

    model = TFTModel(
        input_chunk_length=168,
        output_chunk_length=168,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=64,
        n_epochs=3,
        optimizer_kwargs={'lr': 1e-4, 'weight_decay': 1e-4},
        add_relative_index=True,
        add_encoders={'cyclic': {'future': ['hour', 'dayofweek', 'month']}},
        use_static_covariates=True,   # Обязательно True для использования статики
        random_state=42,
        pl_trainer_kwargs={
            'accelerator': 'cuda' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'enable_progress_bar': True
        }
    )

    # ---------- ОБУЧЕНИЕ ----------
    print("Обучение модели...")
    model.fit(
        series=train_targets,
        past_covariates=train_covs,
        val_series=val_targets,
        val_past_covariates=val_covs,
        verbose=True
    )

    # ---------- СОХРАНЕНИЕ ----------
    print("Сохранение модели...")
    model.save(f'{save_path}tft_station_model.pkl')

    # Сохраняем скейлеры
    import pickle
    with open(f'{save_path}station_scalers.pkl', 'wb') as f:
        pickle.dump({'target_scaler': target_scaler, 'cov_scaler': cov_scaler}, f)
        return model, target_scaler, cov_scaler

# Прогнозирование
def make_forecast(model, scaler, cov_scaler, last_data, forecast_horizon):
    """Создание прогноза на заданный горизонт"""
    print(f"\n=== ПРОГНОЗ НА {forecast_horizon} ЧАСОВ ===")

    # Прогнозирование
    forecast_scaled = model.predict(n=forecast_horizon)

    # Обратное масштабирование
    forecast = scaler.inverse_transform(forecast_scaled)

    # Преобразование в DataFrame
    forecast_df = forecast.pd_dataframe()
    forecast_df = forecast_df.reset_index()
    forecast_df.columns = ['timestamp'] + TARGET_VARIABLES[:len(forecast_df.columns)-1]

    return forecast_df

# Основная функция
def main():
    print("=" * 60)
    print("ЗАПУСК АНАЛИЗА ДАННЫХ АЗС С ИСПОЛЬЗОВАНИЕМ TFT")
    print("=" * 60)

    # Загрузка данных
    data, metadata, id_col = load_and_prepare_data()

    # Создание временных признаков
    data = create_time_features(data)

    # Кодирование категориальных признаков
    data, label_encoders = encode_categorical_features(data)

    # Подготовка целевых переменных
    data, available_targets = prepare_target_variables(data)

    if not available_targets:
        print("Ошибка: нет целевых переменных для прогнозирования")
        return

    # НОВАЯ ПОДГОТОВКА ДАННЫХ (вместо старой prepare_darts_data)
    # Определяем колонку-идентификатор станции
    station_id_col = 'station_id' if 'station_id' in data.columns else 'station_name'

    series_list, cov_list, feature_cols, station_ids = prepare_darts_data_per_station(
        data, available_targets, station_id_col, static_encoding=True
    )

    # НОВОЕ ОБУЧЕНИЕ МОДЕЛИ
    model, target_scaler, cov_scaler = train_tft_model_on_stations(
        series_list, cov_list, available_targets
    )

    # Если обучение прошло успешно, выводим сообщение
    print("\n✅ Модель успешно обучена и сохранена.")
    print(f"Обучено станций: {len(station_ids)}")
    print(f"Целевые переменные: {available_targets}")


if __name__ == "__main__":
    main()