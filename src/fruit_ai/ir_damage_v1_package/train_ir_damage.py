from pathlib import Path
from ultralytics import YOLO


def main():
    dataset = Path(__file__).resolve().parent
    weights = Path(r"D:\fruit_ai\runs\ir_mask_demo_v5\weights\best.pt")
    if not weights.is_file():
        raise FileNotFoundError(f"找不到初始化模型：{weights}")
    # 使用绝对数据路径，避免从其他目录启动时找不到图片。
    yaml_path = dataset / "data.yaml"
    yaml_path.write_text(
        f'path: "{dataset.as_posix()}"\n'
        'train: images/train\nval: images/train\n'
        'names:\n  0: fruit\n  1: invalid\n  2: damage\n',
        encoding="utf-8",
    )
    for path in dataset.rglob("*.cache"):
        path.unlink()
    print("三类统一训练：0=fruit，1=invalid，2=damage")
    print("本次只有8张有损伤标注图片，验证复用训练集，仅检查拟合。")
    model = YOLO(str(weights))
    model.train(
        data=str(yaml_path), epochs=200, patience=0,
        imgsz=640, batch=2, nbs=2, device=0, workers=0,
        optimizer="AdamW", lr0=0.001, lrf=0.1,
        warmup_epochs=0, overlap_mask=False, mask_ratio=2,
        mosaic=0, mixup=0, copy_paste=0,
        hsv_h=0, hsv_s=0, hsv_v=0,
        degrees=0, translate=0, scale=0, fliplr=0, flipud=0,
        val=True, project=r"D:\fruit_ai\runs",
        name="ir_damage_v1", exist_ok=False,
    )
    print("训练完成，本次模型：", Path(model.trainer.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
