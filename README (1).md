# TFT Fuel Demand Intelligence

Подготовленный проект для TFT-анализа сети АЗС на данных `detailed_data.csv` и `stations_metadata.csv`.

## Что внутри

- `data_preparation.py` - загрузка, проверка данных, признаки, manifest, сборка Darts `TimeSeries`.
- `train.py` - обучение Darts `TFTModel` и сохранение артефактов. Обучение не запускается автоматически.
- `predict.py` - загрузка сохранённой модели и прогноз, плюс лёгкий baseline для dashboard без модели.
- `recommendations.py` - правила рекомендаций на основе факта и прогноза.
- `dashboard.py` - ultra-modern Streamlit dashboard в тёмной premium SaaS-теме.
- `preflight.py` - проверка данных без обучения и без запуска TFT.

## Установка

```powershell
python -m pip install -r requirements.txt
```

Если системного Python нет, используйте Python из Codex runtime или установленный локально Python 3.10+.

## Проверка без обучения

```powershell
python preflight.py --write-manifest
```

Ожидаемо для полного набора:

- `219000` строк;
- `25` АЗС;
- период `2023-01-01 00:00:00` - `2023-12-31 23:00:00`;
- `8760` строк на каждую АЗС.

## Dashboard

```powershell
streamlit run dashboard.py
```

Dashboard не запускает обучение. Если артефактов TFT нет в `artifacts/`, приложение работает в exploratory + baseline mode и явно показывает статус модели.

## Обучение TFT

Команда подготовлена, но в рамках текущей задачи обучение не запускалось:

```powershell
python train.py --data detailed_data.csv --metadata stations_metadata.csv --output artifacts
```

По умолчанию `train.py` использует безопасный `--precision auto`: `bf16-mixed` на CUDA GPU с поддержкой BF16, иначе `32-true`. Для Darts TFT не используйте `16-mixed`: внутри attention mask есть значение `-1e9`, которое переполняет `float16`.

Для короткой проверки подготовки данных без обучения:

```powershell
python train.py --dry-run --station-limit 2
```

`--dry-run` требует установленный Darts, потому что проверяет построение `TimeSeries`.
