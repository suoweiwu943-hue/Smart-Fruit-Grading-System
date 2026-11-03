"""把随包新增数据合入本机v3；不修改原数据集。"""
from pathlib import Path
import collections
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import yaml

HERE = Path(__file__).resolve().parent
OLD = Path(r'D:\fruit_ai\detect_dataset_v3')
OUTPUT = Path(r'D:\fruit_ai\detect_dataset_v4')
NAMES = ['apple', 'banana', 'cherry_tomato', 'kiwi', 'mandarin', 'xiang_pear']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_text(path):
    if not path.is_file():
        raise RuntimeError('缺少标签，不能自动当作背景：' + str(path))
    text = path.read_text(encoding='utf-8-sig')
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f'标签格式无效：{path}:{line_number}')
        cls = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if not (0 <= cls < 6 and all(math.isfinite(v) and 0 <= v <= 1 for v in (x,y,w,h)) and w > 0 and h > 0):
            raise RuntimeError(f'标签数值无效：{path}:{line_number}')
    return text


def prepare(old=OLD, output=OUTPUT, here=HERE):
    old, output, here = Path(old), Path(output), Path(here)
    if output.exists():
        raise RuntimeError('输出目录已经存在，未覆盖：' + str(output))
    yaml_path = old / 'data.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError('找不到旧数据集：' + str(yaml_path))
    config = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
    names = config.get('names')
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(6)]
    if names != NAMES:
        raise RuntimeError('旧数据集类别顺序不一致，停止合并')

    rows = []
    known = {}
    for split in ('train', 'val'):
        folder = old / 'images' / split
        if not folder.is_dir():
            raise FileNotFoundError(str(folder))
        count = 0
        for image in sorted(folder.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            count += 1
            relative = image.relative_to(folder)
            label = (old / 'labels' / split / relative).with_suffix('.txt')
            text = label_text(label)
            sha = digest(image)
            if sha in known:
                previous = known[sha]
                if previous['split'] != split:
                    raise RuntimeError('旧数据集训练/验证存在重复图片，请先处理：' + str(image))
                if previous['text'].strip() != text.strip():
                    raise RuntimeError('相同图片存在不同标签：' + str(image))
                continue
            row = dict(image=image, text=text, split=split, sha=sha, source='old')
            rows.append(row)
            known[sha] = row
        if not count:
            raise RuntimeError('旧数据集为空：' + str(folder))

    manifest = json.loads((here / 'manifest.json').read_text(encoding='utf-8'))
    new_groups = collections.defaultdict(list)
    for item in manifest:
        image = here / 'new_data' / 'images' / item['split'] / item['filename']
        label = (here / 'new_data' / 'labels' / item['split'] / item['filename']).with_suffix('.txt')
        if not image.is_file() or digest(image) != item['sha256']:
            raise RuntimeError('新增图片缺失或已改变：' + str(image))
        text = label_text(label)
        if item['kind'] == 'background' and text.strip():
            raise RuntimeError('背景标签应为空：' + str(label))
        if item['kind'] == 'banana' and (not text.strip() or any(line.split()[0] != '1' for line in text.splitlines() if line.strip())):
            raise RuntimeError('香蕉标签必须非空且类别为1：' + str(label))
        new_groups[item['group']].append((item, image, text))

    skipped = 0
    for group, members in sorted(new_groups.items()):
        old_splits = {known[item['sha256']]['split'] for item, _, _ in members if item['sha256'] in known}
        if len(old_splits) > 1:
            raise RuntimeError('同一新增采集组已出现在旧数据集的训练和验证两侧：' + group)
        # 如本组部分图片已存在旧数据中，新增图片跟随旧分组，避免泄漏。
        split = next(iter(old_splits)) if old_splits else members[0][0]['split']
        for item, image, text in members:
            sha = item['sha256']
            if sha in known:
                if known[sha]['text'].strip() != text.strip():
                    raise RuntimeError('新旧重复图片标签不一致：' + str(image))
                skipped += 1
                continue
            row = dict(image=image, text=text, split=split, sha=sha, source=item['kind'])
            rows.append(row)
            known[sha] = row

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '_building_', dir=output.parent))
    stats = {s: dict(images=0, background=0, boxes=[0]*6) for s in ('train','val')}
    try:
        for split in ('train', 'val'):
            (stage / 'images' / split).mkdir(parents=True)
            (stage / 'labels' / split).mkdir(parents=True)
        report = []
        for index, row in enumerate(rows):
            # 编号+摘要保证文件及标签名称唯一。
            name = f"{row['source']}_{index:05d}_{row['sha'][:12]}"
            destination = stage / 'images' / row['split'] / (name + row['image'].suffix.lower())
            shutil.copy2(row['image'], destination)
            (stage / 'labels' / row['split'] / (name + '.txt')).write_text(row['text'], encoding='utf-8')
            stat = stats[row['split']]
            stat['images'] += 1
            lines = [line for line in row['text'].splitlines() if line.strip()]
            if not lines:
                stat['background'] += 1
            for line in lines:
                stat['boxes'][int(line.split()[0])] += 1
            report.append(dict(source=str(row['image']), split=row['split'], filename=destination.name, sha256=row['sha']))
        data = dict(path=output.resolve().as_posix(), train='images/train', val='images/val', names=dict(enumerate(NAMES)))
        (stage / 'data.yaml').write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        (stage / 'merge_report.json').write_text(json.dumps(dict(stats=stats, duplicate_new_images_skipped=skipped, files=report), ensure_ascii=False, indent=2), encoding='utf-8')
        # 重命名前再检查，避免覆盖已有数据目录。
        if output.exists():
            raise RuntimeError('输出目录已被创建，停止写入')
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print('\n========== v4合并完成 ==========')
    print('位置：', output)
    print('新增重复图片跳过：', skipped)
    for split, stat in stats.items():
        print(split, '图片',stat['images'],'无目标',stat['background'])
        for name, count in zip(NAMES, stat['boxes']):
            print(' ',name,count)
    print('原v3保持不变；尚未开始训练。')
    return output


if __name__ == '__main__':
    prepare()
