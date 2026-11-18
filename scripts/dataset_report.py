"""Отчёт по YOLO-датасету из Roboflow.
Запуск: python dataset_report.py <папка_с_data.yaml> [сколько_картинок_в_превью]
Выводит: число изображений/рамок по классам и сплитам, формат разметки,
и сохраняет <папка>/preview.jpg (случайные 20 картинок с рамками).
Нужно: pip install pillow pyyaml
"""
import glob, collections, random, sys, os
import yaml
from PIL import Image, ImageDraw

root = sys.argv[1]
n_prev = int(sys.argv[2]) if len(sys.argv) > 2 else 20
names = yaml.safe_load(open(os.path.join(root, 'data.yaml'), encoding='utf-8'))['names']
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print('classes:', names)

for s in ('train', 'valid', 'val', 'test'):
    lab_dir = os.path.join(root, s, 'labels')
    if not os.path.isdir(lab_dir):
        continue
    boxes = collections.Counter(); imgs = collections.Counter(); cols = collections.Counter(); empty = 0
    files = glob.glob(os.path.join(lab_dir, '*.txt'))
    for f in files:
        rows = [l.split() for l in open(f, encoding='utf-8') if l.strip()]
        empty += not rows
        seen = set()
        for r in rows:
            cols[len(r)] += 1; boxes[names[int(r[0])]] += 1; seen.add(names[int(r[0])])
        for c in seen:
            imgs[c] += 1
    print(f'\n[{s}] images={len(files)} empty_labels={empty}')
    print('  boxes per class :', dict(boxes))
    print('  images per class:', dict(imgs))
    print('  columns per row :', dict(cols), '(5 = рамки; >5 = полигоны/сегментация)')

pics = sorted(glob.glob(os.path.join(root, 'train', 'images', '*')))
random.seed(1)
pics = random.sample(pics, min(n_prev, len(pics)))
cols_n = 5; rows_n = (len(pics) + cols_n - 1) // cols_n
sheet = Image.new('RGB', (cols_n * 256, rows_n * 256))
for i, f in enumerate(pics):
    im = Image.open(f).convert('RGB'); W, H = im.size; dr = ImageDraw.Draw(im)
    lf = os.path.join(root, 'train', 'labels', os.path.splitext(os.path.basename(f))[0] + '.txt')
    if os.path.exists(lf):
        for l in open(lf, encoding='utf-8'):
            p = l.split()
            if len(p) < 5: continue
            c = int(p[0])
            if len(p) == 5:
                cx, cy, w, h = map(float, p[1:5]); x1, y1, x2, y2 = (cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H
            else:
                xs = [float(v)*W for v in p[1::2]]; ys = [float(v)*H for v in p[2::2]]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
            dr.text((x1+4, y1+4), names[c], fill=(255, 255, 0))
    sheet.paste(im.resize((256, 256)), ((i % cols_n) * 256, (i // cols_n) * 256))
out = os.path.join(root, 'preview.jpg')
sheet.save(out); print('\npreview saved:', out)