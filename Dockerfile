# Образ для обучения YOLO26 (лабораторная 2, вариант 1).
# Сборка (из корня репозитория, submodule должен быть инициализирован):
#   git submodule update --init --recursive
#   docker build -t lab2-yolo .
# Запуск (директория датасета -> /data, директория результатов -> /results):
#   docker run --rm --gpus all -v <датасет>:/data -v <результаты>:/results lab2-yolo --epochs 100
# Без GPU уберите --gpus all (обучение пойдёт на CPU, очень медленно).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=False

# системные библиотеки для OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Ultralytics ставится из submodule (зафиксированная версия кода)
COPY external/ultralytics /workspace/external/ultralytics
RUN pip install --no-cache-dir /workspace/external/ultralytics

# веса yolo26n.pt скачиваются при сборке, чтобы запуск не зависел от сети
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

COPY scripts/train.py /workspace/scripts/train.py

VOLUME ["/data", "/results"]
ENTRYPOINT ["python", "/workspace/scripts/train.py"]
CMD ["--data-dir", "/data", "--results-dir", "/results"]
