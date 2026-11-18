# Источники и лицензии

## Код

| Компонент | Источник | Лицензия |
|---|---|---|
| Android-приложение (основа) | [surendramaran/YOLO](https://github.com/surendramaran/YOLO), папка `YOLOv8-Object-Detector-Android-Tflite` (в лекции 3 — слайд 22) | CC BY 4.0 (интеграция в мобильное приложение) |
| Обучение и экспорт моделей | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), подключён как submodule `external/ultralytics` | AGPL-3.0 |
| Предобученные веса `yolo26n.pt` | релизы Ultralytics | AGPL-3.0 |

Изменения в приложении: заменены модель и `labels.txt`, в `Detector.kt` добавлена поддержка входа `[1,3,H,W]` (NCHW), безопасное получение подписей классов и замер времени только вызова модели (`interpreter.run`). Файлы `LICENSE` и `NOTICE` исходного репозитория сохранены.

## Наборы данных (Roboflow Universe)

Обучающая выборка собрана скриптом `scripts/merge_datasets.py` из пяти наборов. Каждый использован в соответствии со своей лицензией (указана на странице набора и в его `README.roboflow.txt`); авторство сохраняется.

| Класс | Набор | Ссылка | Лицензия |
|---|---|---|---|
| toothbrush, toothpaste (часть) | Toothbrushes v1 (xushi) | https://universe.roboflow.com/xushi/toothbrushes | CC BY 4.0 |
| toothpaste | toothpaste v2 (test-puzab) | https://universe.roboflow.com/test-puzab/toothpaste-wwvx1-7gdqs | CC BY 4.0 |
| soap | soap seg v4 (jameshrih) | https://universe.roboflow.com/jameshrih/soap-seg-pwk0o | CC BY 4.0 |
| soap | soap v4 | https://universe.roboflow.com/ss-vizyy/soap-nn1c1 | CC BY 4.0 |
| liquid_soap | liquid soap v2 | https://universe.roboflow.com/vision-wajsm/liquid-soap | CC BY 4.0 |

Состав и преобразования: из каждого набора берутся только нужные классы, полигоны переводятся в рамки, повторяющиеся кадры удаляются, а разбиение train/val делается по исходному кадру (подробности — в `scripts/merge_datasets.py`).