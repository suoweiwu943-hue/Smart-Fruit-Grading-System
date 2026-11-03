"""合并补充样本后，从v4最佳权重训练。原数据和模型保持不变。"""
from pathlib import Path
import hashlib
import shutil
import yaml

BASE = Path(__file__).resolve().parent
OLD_YAML = Path(r'D:\fruit_ai\ir_damage_dataset_v4\data.yaml')
NEW_ROOT = Path(r'D:\fruit_ai\ir_damage_dataset_v5')
WEIGHTS = Path(r'D:\fruit_ai\runs\ir_damage_v4\weights\best.pt')
EXPECTED = ['fruit', 'invalid', 'damage']
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def resolve_split(value, root):
    values = value if isinstance(value, list) else [value]
    result = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise RuntimeError('data.yaml 中缺少 train 或 val 路径')
        p = Path(item)
        if not p.is_absolute():
            p = root / p
        result.append(p.resolve())
    return result


def image_list(paths):
    result = []
    for p in paths:
        if p.is_dir():
            result.extend(x.resolve() for x in p.rglob('*') if x.suffix.lower() in EXT)
        elif p.suffix.lower() == '.txt' and p.is_file():
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                f = Path(line.strip())
                if not f.is_absolute():
                    f = p.parent / f
                result.append(f.resolve())
        else:
            raise RuntimeError('未找到数据路径：' + str(p))
    if not result or any(not p.is_file() for p in result):
        raise RuntimeError('数据列表为空或部分图片不存在，请检查原 data.yaml')
    return sorted(set(result))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    if not WEIGHTS.is_file():
        raise RuntimeError('未找到模型：' + str(WEIGHTS))
    data = yaml.safe_load(OLD_YAML.read_text(encoding='utf-8-sig'))
    names = data.get('names', [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k: int(k))]
    if list(names) != EXPECTED:
        raise RuntimeError('原数据类别顺序不匹配，预期 0 fruit、1 invalid、2 damage；实际：' + str(names))
    raw_root = Path(data.get('path') or OLD_YAML.parent)
    if not raw_root.is_absolute():
        raise RuntimeError('原 data.yaml 的 path 是相对路径，请先改成原数据集的绝对路径')
    root = raw_root.resolve()
    train = image_list(resolve_split(data.get('train'), root))
    val = image_list(resolve_split(data.get('val'), root))
    val_hashes = {digest(p) for p in val}
    train_hashes = {digest(p) for p in train}
    if train_hashes & val_hashes:
        raise RuntimeError('原训练集与验证集存在完全重复图片，请先清理数据划分')
    additions = sorted((BASE/'images/train').glob('*.jpg'))
    if len(additions) != 7:
        raise RuntimeError('补充包应有7张图片，请完整解压后运行')
    if any(digest(p) in val_hashes for p in additions):
        raise RuntimeError('补充照片已存在于验证集，不能直接加入训练集')
    for p in additions:
        if not (BASE/'labels/train'/p.with_suffix('.txt').name).is_file():
            raise RuntimeError('缺少标注：' + p.name)
    added = []
    for p in additions:
        if digest(p) in train_hashes:
            print('原训练集已有相同图片，跳过：', p.name)
            continue
        image = NEW_ROOT/'images/train'/p.name
        label = NEW_ROOT/'labels/train'/p.with_suffix('.txt').name
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        # 同名但内容不同则停止，避免覆盖其他补充批次。
        source_label = BASE/'labels/train'/label.name
        for source, target in [(p, image), (source_label, label)]:
            if target.exists() and source.read_bytes() != target.read_bytes():
                raise RuntimeError('目标文件已有不同内容：' + str(target))
            shutil.copy2(source, target)
        added.append(image.resolve())
    NEW_ROOT.mkdir(parents=True, exist_ok=True)
    (NEW_ROOT/'train.txt').write_text('\n'.join(p.as_posix() for p in train+added)+'\n', encoding='utf-8')
    (NEW_ROOT/'val.txt').write_text('\n'.join(p.as_posix() for p in val)+'\n', encoding='utf-8')
    config = {'path': NEW_ROOT.as_posix(), 'train': 'train.txt', 'val': 'val.txt', 'names': dict(enumerate(EXPECTED))}
    target_yaml = NEW_ROOT/'data.yaml'
    target_yaml.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print('原训练图片：',len(train),'新增：',len(added),'验证图片：',len(val),flush=True)
    print('配置：',target_yaml,flush=True)
    return target_yaml


def main():
    from ultralytics import YOLO
    config = prepare()
    model = YOLO(str(WEIGHTS))
    if model.task != 'segment':
        raise RuntimeError('当前权重不是分割模型')
    # fruit、invalid、damage 存在包含关系，各类别保留独立掩膜。
    model.train(data=str(config), epochs=200, batch=2, imgsz=640,
                device=0, workers=0, overlap_mask=False,
                project=r'D:\fruit_ai\runs', name='ir_damage_v5', exist_ok=False)


if __name__ == '__main__':
    main()
