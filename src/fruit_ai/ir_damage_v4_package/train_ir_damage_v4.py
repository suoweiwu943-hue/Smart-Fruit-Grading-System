from pathlib import Path
import hashlib
import json
import math
import shutil
import uuid

from PIL import Image
import yaml

ROOT = Path(r"D:\fruit_ai")
OLD_DATA = ROOT / "ir_damage_dataset_v3"
NEW_SAMPLES = Path(__file__).resolve().parent
MERGED_DATA = ROOT / "ir_damage_dataset_v4"
WEIGHTS = ROOT / "runs" / "ir_damage_v3" / "weights" / "best.pt"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {0: "fruit", 1: "invalid", 2: "damage"}


def labels_checked(path, allowed):
    text = path.read_text(encoding="utf-8-sig")
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        try:
            category = int(values[0])
            coordinates = [float(v) for v in values[1:]]
        except (ValueError, IndexError):
            raise ValueError(f"标注格式错误：{path} 第{number}行")
        if (category not in allowed or len(coordinates) < 6
                or len(coordinates) % 2
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates)):
            raise ValueError(f"标注坐标或类别错误：{path} 第{number}行")
    return text


def collect(folder, prefix, expected_names):
    config = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8-sig"))
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or {int(k): v for k, v in names.items()} != expected_names:
        raise ValueError(f"类别顺序不符合预期：{folder}")
    files = sorted(p for p in (folder / "images" / "train").iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"没有找到训练图片：{folder}")
    result = []
    for index, source in enumerate(files, 1):
        label = folder / "labels" / "train" / (source.stem + ".txt")
        if not label.is_file():
            raise FileNotFoundError(f"图片缺少标注：{label}")
        text = labels_checked(label, set(expected_names))
        with Image.open(source) as im:
            rgb = im.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        result.append({
            "source": source, "label": label, "text": text,
            "key": f"{prefix}_{index:03d}", "image_hash": digest,
            "file_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    return result


def build_dataset():
    old = collect(OLD_DATA, "old_v3", NAMES)
    new = collect(NEW_SAMPLES, "correction_v4", NAMES)
    items = old + new
    seen = {}
    for item in items:
        if item["image_hash"] in seen:
            raise ValueError(f"发现重复原图，避免重复或冲突标注，已停止：\n{seen[item['image_hash']]}\n{item['source']}")
        seen[item["image_hash"]] = str(item["source"])
    signature_data = [(str(i["source"]), i["file_hash"], i["text"]) for i in items]
    signature = hashlib.sha256(json.dumps(signature_data, ensure_ascii=False).encode()).hexdigest()
    marker = MERGED_DATA / "merge_report.json"
    if MERGED_DATA.exists():
        if not marker.is_file():
            raise RuntimeError(f"{MERGED_DATA} 已存在但缺少完成记录，请先改名备份。")
        report = json.loads(marker.read_text(encoding="utf-8"))
        if report.get("signature") != signature:
            raise RuntimeError("源数据已变化，未覆盖旧合并数据；请先将 ir_damage_dataset_v4 改名备份。")
        for item in items:
            image = MERGED_DATA / "images/train" / (item["key"] + item["source"].suffix.lower())
            label = MERGED_DATA / "labels/train" / (item["key"] + ".txt")
            if (not image.is_file() or not label.is_file()
                    or hashlib.sha256(image.read_bytes()).hexdigest() != item["file_hash"]
                    or label.read_text(encoding="utf-8") != item["text"]):
                raise RuntimeError("合并文件不完整或已被修改，请先将 ir_damage_dataset_v4 改名备份。")
        print("合并数据已存在且核对一致，直接训练。")
    else:
        temporary = ROOT / ("ir_damage_merge_" + uuid.uuid4().hex[:10])
        try:
            (temporary / "images/train").mkdir(parents=True)
            (temporary / "labels/train").mkdir(parents=True)
            for item in items:
                shutil.copyfile(item["source"], temporary / "images/train" /
                                (item["key"] + item["source"].suffix.lower()))
                (temporary / "labels/train" / (item["key"] + ".txt")).write_text(
                    item["text"], encoding="utf-8")
            report = {
                "old_images": len(old), "new_images": len(new), "total_images": len(items),
                "signature": signature, "old_damage_labels": "preserved three-class labels",
                "validation": "training images reused; not independent validation",
                "mapping": [{"image": str(i["source"]), "new_name": i["key"]} for i in items],
            }
            (temporary / "merge_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.rename(MERGED_DATA)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    config_path = MERGED_DATA / "data.yaml"
    config_path.write_text(yaml.safe_dump({
        "path": MERGED_DATA.as_posix(), "train": "images/train", "val": "images/train",
        "names": NAMES,
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for path in MERGED_DATA.rglob("*.cache"):
        path.unlink()
    print(f"旧图 {len(old)} 张 + 新纠错图 {len(new)} 张 = {len(items)} 张")
    return config_path


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"找不到三类 v3 模型：{WEIGHTS}")
    config = build_dataset()
    from ultralytics import YOLO
    print("开始训练三类统一模型 v4；验证复用训练集，分数仅检查拟合。")
    model = YOLO(str(WEIGHTS))
    model.train(
        data=str(config), epochs=200, patience=0,
        imgsz=640, batch=2, nbs=2, device=0, workers=0,
        optimizer="AdamW", lr0=0.0003, lrf=0.1, warmup_epochs=0,
        overlap_mask=False, mask_ratio=2,
        mosaic=0, mixup=0, copy_paste=0,
        hsv_h=0, hsv_s=0, hsv_v=0,
        degrees=0, translate=0.05, scale=0.15, fliplr=0.5, flipud=0,
        val=True, project=str(ROOT / "runs"), name="ir_damage_v4", exist_ok=False,
    )
    print("训练完成，本次模型：", Path(model.trainer.save_dir) / "weights/best.pt")


if __name__ == "__main__":
    main()
