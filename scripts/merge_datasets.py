"""Слияние Roboflow-датасетов варианта 1 в один YOLO-датасет.

Классы: 0=toothbrush, 1=toothpaste, 2=soap, (3=liquid_soap при --with-liquid)

Запуск (из папки репозитория):
    python scripts/merge_datasets.py --src-root "E:\\Загрузки" --out dataset/prepared
    python scripts/merge_datasets.py --src-root "E:\\Загрузки" --out dataset/prepared --with-liquid

Что делает:
  * полигоны -> рамки (min/max по точкам), нужные классы переименовываются, остальные выбрасываются;
  * картинки, у которых после фильтра не осталось рамок, пропускаются;
  * у источников с аугментацией берётся одна картинка на исходный кадр (имя до ".rf.");
  * убираются дубли между датасетами (dHash);
  * train/val делится по исходному кадру (копии одного кадра не попадают в оба сплита);
  * пишет data.yaml и docker_data.yaml и печатает статистику.
Нужно: pip install pillow pyyaml
"""
import argparse, hashlib, random, re, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

# ---- настройки источников (папка в --src-root, правила переименования классов) ----------
# per_group=1 -> брать одну картинку на исходный кадр; cap -> максимум картинок из источника;
# max_area -> выкинуть картинку, если рамка занимает больше этой доли кадра (упаковки/коробки).
SOURCES = [
    dict(folder='Toothbrushes.v1i.yolo26', tag='tb',
         map={'Toothbrush': 'toothbrush', 'Toothpaste': 'toothpaste', 'Soap': 'soap'},
         per_group=None, cap=None, max_area=None),
    dict(folder='toothpaste.v2i.yolo26', tag='tp',
         map={'toothpaste': 'toothpaste'},
         per_group=None, cap=500, max_area=None),
    dict(folder='soap.v4i.yolo26', tag='sp',
         map={'Soap': 'soap', 'Dove Soap': 'soap', '1 Dove Soap': 'soap',
              '2 Dove Soap': 'soap', '4 Dove Soap': 'soap'},
         per_group=1, cap=400, max_area=0.6),
    dict(folder='soap seg.v4i.yolo26', tag='ss',
         map={'soap': 'soap'},
         per_group=1, cap=150, max_area=None),
    dict(folder='liquid soap.v2i.yolo26', tag='ls',
         map={'liquid-soap': 'liquid_soap'},
         per_group=1, cap=300, max_area=0.6, only_with_liquid=True),
]
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MIN_SIDE = 0.01  # доля кадра; более мелкие рамки считаем шумом
DUP_DIST = 6     # расстояние Хэмминга (из 256 бит) для дублей


def dhash(path: Path) -> int:
    """dHash 16x16 (256 бит): на белых каталожных фонах 8x8 даёт ложные совпадения."""
    px = Image.open(path).convert('L').resize((17, 16)).tobytes()
    bits = 0
    for r in range(16):
        for c in range(16):
            bits = (bits << 1) | (px[r * 17 + c] > px[r * 17 + c + 1])
    return bits


def group_id(stem: str) -> str:
    return re.split(r'\.rf\.', stem)[0]


def parse_label(path: Path, names, mapping, max_area):
    """Возвращает список (target_name, cx, cy, w, h) или None, если картинку надо пропустить."""
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        src_name = names[int(p[0])]
        if src_name not in mapping:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, w, h = v
        else:  # полигон x1 y1 x2 y2 ...
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        if w < MIN_SIDE or h < MIN_SIDE:
            continue
        if max_area is not None and w * h > max_area:
            return None  # упаковка/коробка -> вся картинка не подходит
        out.append((mapping[src_name], cx, cy, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('dataset/prepared'))
    ap.add_argument('--with-liquid', action='store_true', help='добавить 4-й класс liquid_soap')
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    classes = ['toothbrush', 'toothpaste', 'soap'] + (['liquid_soap'] if a.with_liquid else [])
    cid = {n: i for i, n in enumerate(classes)}

    items = []  # dict(src, gid, img, boxes)
    for s in SOURCES:
        if s.get('only_with_liquid') and not a.with_liquid:
            continue
        root = a.src_root / s['folder']
        if not (root / 'data.yaml').exists():
            print(f'[!] нет датасета {root}, пропускаю', file=sys.stderr)
            continue
        names = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))['names']
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        groups = defaultdict(list)
        skipped = Counter()
        for split in ('train', 'valid', 'val', 'test'):
            img_dir, lab_dir = root / split / 'images', root / split / 'labels'
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                lab = lab_dir / (img.stem + '.txt')
                if img.suffix.lower() not in IMG_EXT or not lab.exists():
                    continue
                boxes = parse_label(lab, names, s['map'], s['max_area'])
                if boxes is None:
                    skipped['большая рамка (упаковка)'] += 1
                elif not boxes:
                    skipped['нет нужных рамок'] += 1
                else:
                    groups[group_id(img.stem)].append(dict(src=s['tag'], gid=s['tag'] + ':' + group_id(img.stem),
                                                           img=img, boxes=boxes))
        gids = sorted(groups)
        rnd.shuffle(gids)
        chosen = []
        for g in gids:
            members = groups[g]
            if s['per_group']:
                members = rnd.sample(members, min(s['per_group'], len(members)))
            chosen.extend(members)
        if s['cap'] and len(chosen) > s['cap']:
            chosen = rnd.sample(chosen, s['cap'])
        print(f"{s['folder']}: взято {len(chosen)} картинок из {sum(len(v) for v in groups.values())}; "
              f"пропущено: {dict(skipped)}")
        items.extend(chosen)

    # дубли между датасетами
    hashes, kept, dups = [], [], 0
    for it in items:
        h = dhash(it['img'])
        if any(bin(h ^ x).count('1') <= DUP_DIST for x in hashes):
            dups += 1
            continue
        hashes.append(h); kept.append(it)
    print(f'дублей удалено: {dups}')
    items = kept

    # сплит по исходному кадру
    gids = sorted({it['gid'] for it in items})
    rnd.shuffle(gids)
    val_g = set(gids[:max(1, int(len(gids) * a.val_frac))])
    seen_val = set()
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    stats = {'train': Counter(), 'val': Counter()}
    imgs = Counter()
    for it in items:
        split = 'val' if it['gid'] in val_g else 'train'
        if split == 'val':  # в val одна картинка на кадр
            if it['gid'] in seen_val:
                continue
            seen_val.add(it['gid'])
        name = f"{it['src']}_{hashlib.md5(str(it['img']).encode()).hexdigest()[:10]}"
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(it['img'], out / 'images' / split / (name + it['img'].suffix.lower()))
        lines = []
        for n, cx, cy, w, h in it['boxes']:
            if n not in cid:
                continue
            lines.append(f'{cid[n]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
            stats[split][n] += 1
        (out / 'labels' / split / (name + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        imgs[split, it['src']] += 1

    names_block = {i: n for i, n in enumerate(classes)}
    for fname, path in (('data.yaml', out.resolve().as_posix()), ('docker_data.yaml', '/workspace/dataset/prepared')):
        (out / fname).write_text(yaml.safe_dump(
            dict(path=path, train='images/train', val='images/val', names=names_block),
            allow_unicode=True, sort_keys=False), encoding='utf-8')

    print('\n=== ИТОГ ===')
    for sp in ('train', 'val'):
        print(sp, 'картинок:', sum(v for (s, _), v in imgs.items() if s == sp),
              '| рамок по классам:', {c: stats[sp][c] for c in classes})
    print('по источникам:', {f'{s}/{t}': v for (s, t), v in sorted(imgs.items())})


if __name__ == '__main__':
    main()
