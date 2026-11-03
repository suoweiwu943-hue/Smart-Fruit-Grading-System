from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    dataset = root / "dataset"

    if not (dataset / "train").is_dir():
        raise RuntimeError("请先运行split_data.py划分数据")

    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用，请先检查PyTorch环境")

    print("GPU:", torch.cuda.get_device_name(0))

    # 首次运行会下载预训练模型
    model = YOLO("yolo11n-cls.pt")

    model.train(
        data=str(dataset),
        epochs=40,
        imgsz=224,
        batch=16,
        device=0,
        workers=0,
        cache=False,
        patience=10,
        seed=42,
        project=str(root / "runs"),
        name="fruit_cls",
        plots=True,
    )


if __name__ == "__main__":
    main()