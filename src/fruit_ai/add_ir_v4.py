from pathlib import Path
import json
import shutil
import uuid

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(r"D:\fruit_ai")
OLD_DATA = ROOT / "ir_mask_dataset_demo_v3"
NEW_DATA = ROOT / "ir_mask_dataset_demo_v4"
SAMPLES = ROOT / "ir_mask_correction_v4"

OLD_MODEL = ROOT / "runs" / "ir_mask_demo_v3" / "weights" / "best.pt"
RUNS = ROOT / "runs"

MARKER = "v4_merge_complete.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path):
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"无法读取图片：{path}")
    return image


def polygon_mask(annotations, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    for annotation in annotations:
        segments = annotation.get("segmentation", [])
        if not isinstance(segments, list):
            raise ValueError("需要 COCO 多边形标注，不支持 RLE。")

        for segment in segments:
            points = np.asarray(segment, dtype=np.float64)

            if (
                points.ndim != 1
                or points.size < 6
                or points.size % 2
                or not np.isfinite(points).all()
            ):
                raise ValueError("多边形坐标格式错误。")

            points = points.reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)

            cv2.fillPoly(
                mask,
                [np.rint(points).astype(np.int32)],
                255,
            )

    return mask


def mask_to_labels(mask, class_id):
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue

        points = cv2.approxPolyDP(
            contour, 0.8, True
        ).reshape(-1, 2)

        if len(points) < 3:
            continue

        points = points.astype(np.float64)
        points[:, 0] /= width
        points[:, 1] /= height

        coordinates = " ".join(
            f"{value:.7f}" for value in points.flatten()
        )
        lines.append(f"{class_id} {coordinates}")

    return lines


def build_dataset():
    if NEW_DATA.exists():
        marker = NEW_DATA / MARKER
        if not marker.is_file():
            raise RuntimeError(
                f"{NEW_DATA} 已存在，但没有完成标记。\n"
                "请先把该目录改名备份，再重新运行。"
            )

        print("v4 数据集已经生成，直接继续训练。")
        return NEW_DATA / "data.yaml"

    old_images = OLD_DATA / "images" / "train"
    old_labels = OLD_DATA / "labels" / "train"

    if not old_images.is_dir() or not old_labels.is_dir():
        raise FileNotFoundError(f"找不到 v3 数据集：{OLD_DATA}")

    old_yaml = OLD_DATA / "data.yaml"
    configuration = yaml.safe_load(
        old_yaml.read_text(encoding="utf-8-sig")
    )
    names = configuration.get("names")

    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError("v3 data.yaml 缺少类别 names。")

    names = {int(key): value for key, value in names.items()}
    if names != {0: "fruit", 1: "invalid"}:
        raise ValueError(f"v3 类别顺序不符：{names}")

    json_files = sorted(SAMPLES.glob("*.json"))
    if len(json_files) != 1:
        raise ValueError(
            f"{SAMPLES} 中应当只有一份标注 JSON，"
            f"目前找到 {len(json_files)} 份。"
        )

    annotation_file = json_files[0]
    data = json.loads(
        annotation_file.read_text(encoding="utf-8-sig")
    )

    categories = {
        item["id"]: item["name"]
        for item in data["categories"]
    }
    if set(categories.values()) != {"fruit", "invalid"}:
        raise ValueError(f"标注类别不符：{categories}")

    if len(data["images"]) != 1:
        raise ValueError("这次 v4 补充应当只有一张图片。")

    record = data["images"][0]
    filename = record["file_name"]

    if Path(filename).name != filename or "\\" in filename:
        raise ValueError("标注中的图片名应当是不含目录的文件名。")

    source = SAMPLES / filename
    if not source.is_file():
        raise FileNotFoundError(f"找不到对应原图：{source}")

    image = read_image(source)
    height, width = image.shape[:2]

    if (width, height) != (record["width"], record["height"]):
        raise ValueError("原图尺寸和 JSON 标注尺寸不一致。")

    grouped = {"fruit": [], "invalid": []}
    for annotation in data["annotations"]:
        if annotation["image_id"] != record["id"]:
            raise ValueError("标注引用了未知图片。")

        category_id = annotation["category_id"]
        if category_id not in categories:
            raise ValueError("标注引用了未知类别。")

        grouped[categories[category_id]].append(annotation)

    fruit = polygon_mask(grouped["fruit"], width, height)
    invalid_raw = polygon_mask(grouped["invalid"], width, height)

    # 过曝区域只保留位于水果内部的部分。
    invalid = cv2.bitwise_and(fruit, invalid_raw)

    fruit_lines = mask_to_labels(fruit, 0)
    invalid_lines = mask_to_labels(invalid, 1)

    if not fruit_lines or not invalid_lines:
        raise ValueError("没有生成有效的 fruit 或 invalid 多边形。")

    original_images = [
        path for path in old_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    for path in original_images:
        label = old_labels / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"旧图片缺少标注：{label}")

        old_image = read_image(path)
        if (
            old_image.shape == image.shape
            and np.array_equal(old_image, image)
        ):
            raise ValueError(
                f"这张原图已经在 v3 中：{path.name}\n"
                "需要更新已有标注，不能重复追加。"
            )

    key = "correction_v4_001"
    if any(path.stem == key for path in original_images):
        raise ValueError(f"旧数据集中已经有同名图片：{key}")

    temporary = ROOT / f"ir_mask_v4_build_{uuid.uuid4().hex[:8]}"

    try:
        shutil.copytree(OLD_DATA, temporary)

        for cache in temporary.rglob("*.cache"):
            cache.unlink()

        destination = (
            temporary / "images" / "train"
            / f"{key}{source.suffix.lower()}"
        )
        shutil.copy2(source, destination)

        label_path = temporary / "labels" / "train" / f"{key}.txt"
        label_path.write_text(
            "\n".join(fruit_lines + invalid_lines) + "\n",
            encoding="utf-8",
        )

        # 继续使用训练集自身检查拟合情况。
        # 这里的验证结果不能当成独立测试准确率。
        dataset_yaml = {
            "path": NEW_DATA.as_posix(),
            "train": "images/train",
            "val": "images/train",
            "names": {0: "fruit", 1: "invalid"},
        }

        (temporary / "data.yaml").write_text(
            yaml.safe_dump(
                dataset_yaml,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        shutil.copy2(
            annotation_file,
            temporary / "correction_v4_original.json",
        )

        report = {
            "source_image": filename,
            "old_images": len(original_images),
            "added_images": 1,
            "total_images": len(original_images) + 1,
            "fruit_pixels": int(np.count_nonzero(fruit)),
            "invalid_pixels": int(np.count_nonzero(invalid)),
            "validation": "training images; not independent",
        }

        (temporary / MARKER).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        temporary.rename(NEW_DATA)

    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"数据集生成成功：{NEW_DATA}")
    print(f"原有 {len(original_images)} 张，新增 1 张。")
    print(f"合计 {len(original_images) + 1} 张。")

    return NEW_DATA / "data.yaml"


def main():
    if not OLD_MODEL.is_file():
        raise FileNotFoundError(f"找不到 v3 模型：{OLD_MODEL}")

    yaml_path = build_dataset()

    print("\n开始使用 v3 模型继续训练 v4……")
    print("注意：当前验证使用训练图片，只用于检查拟合情况。\n")

    model = YOLO(str(OLD_MODEL))
    model.train(
        data=str(yaml_path),
        epochs=150,
        patience=0,
        imgsz=640,
        batch=2,
        nbs=2,
        device=0,
        workers=0,
        optimizer="AdamW",
        lr0=0.0003,
        lrf=0.1,
        warmup_epochs=0,
        overlap_mask=False,
        mask_ratio=2,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        degrees=0,
        translate=0,
        scale=0,
        fliplr=0,
        flipud=0,
        val=True,
        project=str(RUNS),
        name="ir_mask_demo_v4",
        exist_ok=False,
    )

    output = Path(model.trainer.save_dir)
    print("\n训练完成！")
    print(f"本次模型：{output / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()