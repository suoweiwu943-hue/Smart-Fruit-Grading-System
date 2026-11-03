"""先运行prepare_v4.py完成合并，再运行本文件训练。"""
from pathlib import Path
import yaml
from ultralytics import YOLO


def main():
    weights = Path(r'D:\fruit_ai\runs\fruit_detect_v3\weights\best.pt')
    data = Path(r'D:\fruit_ai\detect_dataset_v4\data.yaml')
    if not weights.is_file():
        raise FileNotFoundError('找不到v3模型：' + str(weights))
    if not data.is_file():
        raise FileNotFoundError('请先运行prepare_v4.py：' + str(data))
    config = yaml.safe_load(data.read_text(encoding='utf-8-sig'))
    names = config.get('names')
    if isinstance(names, dict):
        names = [names.get(i,names.get(str(i))) for i in range(6)]
    expected = ['apple','banana','cherry_tomato','kiwi','mandarin','xiang_pear']
    if names != expected:
        raise RuntimeError('类别顺序不一致，停止训练')
    dataset_root = data.parent
    for split in ('train','val'):
        present = set()
        for label in (dataset_root/'labels'/split).glob('*.txt'):
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():present.add(int(line.split()[0]))
        if present != set(range(6)):
            raise RuntimeError(split+'未包含完整六类标注，停止训练')
    YOLO(str(weights)).train(
        data=str(data), epochs=80, patience=20,
        imgsz=640, batch=8, device=0, workers=0,
        project=r'D:\fruit_ai\runs', name='fruit_detect_v4',
        exist_ok=False, resume=False
    )


if __name__ == '__main__':
    main()
