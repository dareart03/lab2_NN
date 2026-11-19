"""Сводка по подготовленному набору (раскладка images/{train,val}, labels/{train,val}).
Запуск: python scripts/dataset_summary.py dataset/prepared_4cls
Печатает число картинок и рамок по классам для train и val (для таблицы в README).
"""
import sys
from collections import Counter
from pathlib import Path

import yaml

root = Path(sys.argv[1])
names = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))['names']
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print('Классы:', names)
print(f"{'выборка':8}{'картинок':>10}{'рамок':>8}  " + '  '.join(f'{n:>12}' for n in names))
for split in ('train', 'val'):
    labs = list((root / 'labels' / split).glob('*.txt'))
    boxes, empty = Counter(), 0
    for f in labs:
        rows = [l.split() for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
        empty += not rows
        for r in rows:
            boxes[names[int(r[0])]] += 1
    print(f"{split:8}{len(labs):>10}{sum(boxes.values()):>8}  " + '  '.join(f'{boxes[n]:>12}' for n in names)
          + (f'   (пустых: {empty})' if empty else ''))
