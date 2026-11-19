# Лабораторная работа 2. Детекция бытовых предметов на YOLO26 в Android (LiteRT)

Обучена собственная модель YOLO26 для обнаружения предметов гигиены, модель экспортирована в LiteRT (`.tflite`) и запущена в Android-приложении с замером времени вызова модели.

| Класс | Описание |
|---|---|
| `toothbrush` | зубная щётка |
| `toothpaste` | тюбик зубной пасты |
| `soap` | твёрдое мыло |
| `liquid_soap` | ёмкость с жидким мылом (дополнительный, четвёртый класс) |

Обучены и сравнены две модели: `v1_3cls` (три обязательных класса) и `v2_4cls` (с жидким мылом).

## Ссылки

- Размеченный набор данных: [prepared_4cls.zip (GitHub Release)](https://github.com/dareart03/lab2_NN/releases/tag/dataset-v2)
- Ноутбук с обучением: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dareart03/lab2_NN/blob/main/notebooks/Lab2_YOLO26_variant1.ipynb)
- Источники кода и данных, лицензии: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md)

## Структура репозитория

```
.
├── Dockerfile                  # образ для обучения
├── scripts/
│   ├── merge_datasets.py       # сбор датасета из нескольких наборов Roboflow
│   ├── dataset_report.py       # статистика по набору и preview.jpg
│   ├── dataset_summary.py      # получение сводки по датасету
│   └── train.py                # обучение (точка входа контейнера)
├── external/ultralytics/       # submodule: ultralytics
├── notebooks/                  # Lab2_YOLO26_variant1.ipynb (обучение, сравнение, экспорт)
├── results/
│   ├── v1_3cls/                # веса, кривые, results.csv (3 класса)
│   └── v2_4cls/                # то же для 4 классов, best.tflite
├── YOLOv8-Object-Detector-Android-Tflite/   # Android-приложение
└── docs/
    ├── ATTRIBUTION.md
    └── profiling/              # скриншоты профилирования
```

## Набор данных

Набор собран скриптом `scripts/merge_datasets.py` из пяти наборов Roboflow Universe (список, ссылки и лицензии — в [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md)).

Что делает скрипт:
- берёт только нужные классы и приводит их к единым названиям;
- переводит полигоны сегментации в рамки (min/max по точкам);
- отбрасывает слишком мелкие рамки и кадры, где рамка занимает почти весь кадр (упаковки и коробки);
- у наборов с аугментацией берёт один кадр на исходное изображение;
- удаляет дубликаты между наборами (dHash 16×16);
- делит train/val **по исходному кадру**, чтобы копии одного кадра не попадали в оба сплита.

Формат: YOLO (`images/{train,val}`, `labels/{train,val}`, `data.yaml`).

| Выборка | Картинок | Рамок: toothbrush | toothpaste | soap | liquid_soap |
|---|---|---|---|---|---|
| train | 1329 | 477 | 773 | 1203 | 438 |
| val | 234 | 64 | 149 | 138 | 102 |

Всего 1563 картинки и 3344 рамки. Сводку можно получить командой `python scripts/dataset_summary.py dataset/prepared_4cls`. В val рамок класса `toothbrush` немного (64), поэтому метрика по нему шумная.

Воспроизведение:
```
python scripts/merge_datasets.py --src-root dataset/raw_sources --out dataset/prepared_4cls --with-liquid
```
Без `--with-liquid` получится набор из трёх классов.

## Обучение

Базовая модель: `yolo26n.pt`, `imgsz=640`, `epochs=100`, `patience=25`, `seed=42`.

### Docker

Скрипт `scripts/train.py` принимает два параметра: директорию набора данных и директорию результатов.

```
git clone --recurse-submodules https://github.com/dareart03/lab2_NN.git
cd lab2_NN
docker build -t lab2-yolo .

# с GPU
docker run --rm --gpus all -v <путь_к_датасету>:/data -v <путь_к_результатам>:/results lab2-yolo --epochs 100

# без GPU (медленно, для проверки)
docker run --rm -v <путь_к_датасету>:/data -v <путь_к_результатам>:/results lab2-yolo --epochs 1 --imgsz 64 --batch 4 --device cpu --name smoke
```

Параметры `train.py`: `--data-dir` (по умолчанию `/data`), `--results-dir` (по умолчанию `/results`), `--epochs`, `--imgsz`, `--batch`, `--device`, `--name`, `--export litert`. В результатах появляются `weights/best.pt`, `metrics.json` (метрики на val по классам), графики Ultralytics и данные `results.csv`.

Образ собирался и проверялся запуском на CPU (1 эпоха), а полное обучение выполнялось в Colab.

### Colab 

Подробно (обучение, сравнение моделей, экспорт) — в ноутбуке `notebooks/Lab2_YOLO26_variant1.ipynb`.

## Результаты

Метрики на валидационной выборке (`yolo val`):

| Модель | Классы | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| v1_3cls | 3 | 0.846 | 0.646 |
| v2_4cls | 4 | 0.876 | 0.673 |

Значения «all» усреднены по разному числу классов, поэтому напрямую сравнивать их нельзя. Среднее mAP@0.5 по трём общим классам: 0.846 у v1_3cls и 0.861 у v2_4cls.

По классам (mAP@0.5):

| Модель | toothbrush | toothpaste | soap | liquid_soap |
|---|---|---|---|---|
| v1_3cls | 0.847 | 0.893 | 0.799 | — |
| v2_4cls | 0.793 | 0.926 | 0.864 | 0.919 |

По классам (mAP@0.5:0.95):

| Модель | toothbrush | toothpaste | soap | liquid_soap |
|---|---|---|---|---|
| v1_3cls | 0.616 | 0.627 | 0.696 | — |
| v2_4cls | 0.568 | 0.662 | 0.795 | 0.668 |

Precision / Recall (все классы): v1_3cls — 0.820 / 0.815; v2_4cls — 0.858 / 0.792.

Выбранная для приложения модель: **v2_4cls**. Общий mAP@0.5 выше (0.876 против 0.846), у `toothpaste` и `soap` метрики лучше, а `liquid_soap` добавляет четвёртый класс. Недостаток: у `toothbrush` mAP@0.5 ниже (0.793 против 0.847), возможно, из-за сходства флакона с носиком и щётки (это предположение, оно не проверялось).

## Экспорт в LiteRT

```python
from ultralytics import YOLO
YOLO("results/v2_4cls/weights/best.pt").export(format="litert", imgsz=640)
```

Результат: `best.tflite`.
- вход `[1, 3, 640, 640]` (NCHW), float32, значения 0..1;
- выход `[1, 4 + nc, 8400]`: координаты рамок (нормированные) и оценки классов; NMS в модель не включён и выполняется в приложении.

## Android-приложение

Основа — приложение [surendramaran/YOLO](https://github.com/surendramaran/YOLO) (`YOLOv8-Object-Detector-Android-Tflite`, CC BY 4.0)

Изменения относительно оригинала:
- заменены модель (`app/src/main/assets/best.tflite`) и `labels.txt`;
- в `Detector.kt` добавлена поддержка входа `[1, 3, H, W]` (NCHW): исходное приложение подаёт данные в порядке NHWC;
- безопасное получение подписи класса при неожиданном индексе;
- GPU-делегат отключён (`USE_GPU = false`) ради стабильности на тестовом устройстве;
- добавлено профилирование (см. ниже).

Сборка: открыть папку `YOLOv8-Object-Detector-Android-Tflite` в Android Studio, собрать и запустить на устройстве.

## Профилирование

Замеряется только вызов модели `interpreter.run(...)` (без предобработки и NMS) через `SystemClock.elapsedRealtimeNanos()`. Значение выводится на экране приложения.

| Параметр | Значение |
|---|---|
| Устройство | HUAWEI Pura 90s Pro (MLN-LX9), EMUI 16.0.0 |
| Режим | CPU, 4 потока |
| Размер входа | 640×640 |
| Среднее время `interpreter.run`, мс | 153.6 (медиана 149) |
| Минимум / максимум, мс | 145 / 202 |

Скриншоты: [docs/profiling/PROFILING.md](docs/profiling/PROFILING.md)

## Лицензии и источники

Код приложения — CC BY 4.0 (surendramaran/YOLO), обучение и экспорт — Ultralytics (AGPL-3.0), наборы данных — Roboflow Universe (лицензии указаны для каждого набора). Полный список — в [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).
