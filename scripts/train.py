"""Обучение YOLO26 для лабораторной 2 (вариант 1: зубная щётка, тюбик пасты, мыло).

Скрипт рассчитан на запуск в Docker-контейнере, но работает и локально.
Параметры контейнера:
    --data-dir     директория с размеченным набором данных (YOLO-формат + data.yaml с names)
    --results-dir  директория для результатов обучения и логов эксперимента

Поддерживаются две раскладки набора данных:
    images/{train,val}  + labels/{train,val}    (то, что делает scripts/merge_datasets.py)
    {train,valid|val}/images + {train,valid|val}/labels   (экспорт Roboflow)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def find_split_dirs(data_dir: Path):
    """Возвращает (train_images, val_images) относительно data_dir."""
    candidates = [
        ('images/train', 'images/val'),
        ('images/train', 'images/valid'),
        ('train/images', 'val/images'),
        ('train/images', 'valid/images'),
    ]
    for train, val in candidates:
        if (data_dir / train).is_dir() and (data_dir / val).is_dir():
            return train, val
    sys.exit(f'Не найдены папки train/val в {data_dir}. Ожидается images/{{train,val}} '
             f'или {{train,valid}}/images.')


def build_data_yaml(data_dir: Path, results_dir: Path) -> Path:
    """Пишет рабочий data.yaml с путями внутри контейнера (в исходном могут быть пути с чужого компьютера)."""
    src = data_dir / 'data.yaml'
    if not src.exists():
        sys.exit(f'В {data_dir} нет data.yaml (нужен список имён классов names).')
    names = yaml.safe_load(src.read_text(encoding='utf-8'))['names']
    train, val = find_split_dirs(data_dir)
    cfg = {'path': str(data_dir.resolve()), 'train': train, 'val': val, 'names': names}
    out = results_dir / 'data_runtime.yaml'
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print('Классы:', names)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=Path('/data'))
    ap.add_argument('--results-dir', type=Path, default=Path('/results'))
    ap.add_argument('--model', default='yolo26n.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--device', default='auto', help='auto | cpu | 0 | 0,1')
    ap.add_argument('--name', default='train')
    ap.add_argument('--export', choices=['none', 'litert'], default='none',
                    help='после обучения экспортировать best.pt (litert -> best.tflite)')
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO

    a.results_dir = a.results_dir.resolve()  # относительный путь Ultralytics дополнил бы префиксом runs/detect/
    a.results_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = build_data_yaml(a.data_dir, a.results_dir)

    device = a.device
    if device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Устройство: {device}; CUDA доступна: {torch.cuda.is_available()}')

    model = YOLO(a.model)
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                patience=a.patience, seed=a.seed, workers=a.workers, device=device,
                project=str(a.results_dir), name=a.name, exist_ok=True)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / 'weights' / 'best.pt'

    # итоговые метрики на валидации по классам -> metrics.json
    r = YOLO(str(best)).val(data=str(data_yaml), imgsz=a.imgsz, device=device, plots=False, verbose=False)
    per_class = {}
    for k, ci in enumerate(r.box.ap_class_index):
        p, rc, ap50, ap = r.box.class_result(k)
        per_class[r.names[int(ci)]] = {'P': float(p), 'R': float(rc), 'mAP50': float(ap50), 'mAP50-95': float(ap)}
    metrics = {'all': {'P': float(r.box.mp), 'R': float(r.box.mr),
                       'mAP50': float(r.box.map50), 'mAP50-95': float(r.box.map)},
               'per_class': per_class}
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if a.export == 'litert':
        try:
            out = YOLO(str(best)).export(format='litert', imgsz=a.imgsz, nms=False, device='cpu')
            shutil.copy2(out, run_dir / 'weights' / 'best.tflite')
            print('Экспорт выполнен:', run_dir / 'weights' / 'best.tflite')
        except Exception as e:  # экспорт не должен ронять уже завершённое обучение
            print('Экспорт LiteRT не удался:', e)

    print('Готово. Результаты:', run_dir)


if __name__ == '__main__':
    main()
