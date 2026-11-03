from pathlib import Path
import hashlib
import json
import shutil
import uuid

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(r"D:\fruit_ai")

OLD_DATA = ROOT / "ir_mask_dataset_demo_v4"
NEW_DATA = ROOT / "ir_mask_dataset_demo_v5"

SAMPLES = ROOT / "ir_renamed_batch"
ANNOTATIONS = SAMPLES / "labels_renamed.json"

OLD_MODEL = (
    ROOT
    / "runs"
    / "ir_mask_demo_v4"
    / "weights"
    / "best.pt"
)

MARKER = "v5_build_complete.json"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def read_image(path):
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(f"无法读取图片：{path}")

    return image


def image_digest(image):
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def polygon_mask(annotations, width, height):
    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for annotation in annotations:
        segments = annotation.get("segmentation")

        if not isinstance(segments, list):
            raise ValueError("需要 COCO 多边形标注")

        for segment in segments:
            points = np.asarray(
                segment,
                dtype=np.float64,
            )

            if (
                points.ndim != 1
                or points.size < 6
                or points.size % 2 != 0
                or not np.isfinite(points).all()
            ):
                raise ValueError("多边形坐标格式错误")

            points = points.reshape(-1, 2)

            points[:, 0] = np.clip(
                points[:, 0], 0, width - 1
            )
            points[:, 1] = np.clip(
                points[:, 1], 0, height - 1
            )

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
            contour,
            0.8,
            True,
        ).reshape(-1, 2)

        if len(points) < 3:
            continue

        points = points.astype(np.float64)
        points[:, 0] /= width
        points[:, 1] /= height

        coordinates = " ".join(
            f"{value:.7f}"
            for value in points.flatten()
        )

        lines.append(
            f"{class_id} {coordinates}"
        )

    return lines


def build_dataset():
    if NEW_DATA.exists():
        marker_path = NEW_DATA / MARKER

        if not marker_path.is_file():
            raise RuntimeError(
                f"{NEW_DATA} 已存在，但没有完成标记。\n"
                "请先将这个 v5 文件夹改名备份，再运行。"
            )

        print("v5 数据集已生成，直接进入训练。")
        return NEW_DATA / "data.yaml"

    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(
            f"找不到标注文件：{ANNOTATIONS}\n"
            "请检查是否多套了一层 ir_renamed_batch 文件夹。"
        )

    old_images = OLD_DATA / "images" / "train"
    old_labels = OLD_DATA / "labels" / "train"

    if not old_images.is_dir() or not old_labels.is_dir():
        raise FileNotFoundError(
            f"找不到 v4 数据集：{OLD_DATA}"
        )

    configuration = yaml.safe_load(
        (OLD_DATA / "data.yaml").read_text(
            encoding="utf-8-sig"
        )
    )

    names = configuration.get("names")

    if isinstance(names, list):
        names = dict(enumerate(names))

    if not isinstance(names, dict):
        raise ValueError("旧数据集缺少类别 names")

    names = {
        int(key): value
        for key, value in names.items()
    }

    if names != {0: "fruit", 1: "invalid"}:
        raise ValueError(
            f"旧数据集类别顺序不正确：{names}"
        )

    old_files = sorted(
        path
        for path in old_images.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    seen_images = set()

    for path in old_files:
        label = old_labels / f"{path.stem}.txt"

        if not label.is_file():
            raise FileNotFoundError(
                f"旧图片缺少标注：{label}"
            )

        seen_images.add(
            image_digest(read_image(path))
        )

    data = json.loads(
        ANNOTATIONS.read_text(
            encoding="utf-8-sig"
        )
    )

    categories = {
        item["id"]: item["name"]
        for item in data["categories"]
    }

    if (
        len(categories) != 2
        or set(categories.values())
        != {"fruit", "invalid"}
    ):
        raise ValueError(
            f"新标注类别不正确：{categories}"
        )

    images = data["images"]
    image_ids = {
        item["id"] for item in images
    }

    if len(image_ids) != len(images):
        raise ValueError("JSON 存在重复图片编号")

    filenames = [
        item["file_name"] for item in images
    ]

    if len(set(filenames)) != len(filenames):
        raise ValueError("JSON 存在重复图片名")

    grouped = {
        image_id: {
            "fruit": [],
            "invalid": [],
        }
        for image_id in image_ids
    }

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id not in grouped:
            raise ValueError("标注引用了未知图片")

        if category_id not in categories:
            raise ValueError("标注引用了未知类别")

        grouped[image_id][
            categories[category_id]
        ].append(annotation)

    additions = []

    for item in images:
        filename = item["file_name"]

        if (
            Path(filename).name != filename
            or "\\" in filename
        ):
            raise ValueError("图片名不能包含目录")

        source = SAMPLES / filename
        image = read_image(source)

        height, width = image.shape[:2]

        if (width, height) != (
            item["width"],
            item["height"],
        ):
            raise ValueError(
                f"原图尺寸与标注不一致：{filename}"
            )

        digest = image_digest(image)

        if digest in seen_images:
            raise ValueError(
                f"发现重复图片：{filename}\n"
                "已停止，避免重复加入训练。"
            )

        seen_images.add(digest)

        annotations = grouped[item["id"]]

        fruit = polygon_mask(
            annotations["fruit"],
            width,
            height,
        )

        invalid = polygon_mask(
            annotations["invalid"],
            width,
            height,
        )

        # 只保留水果内部的过曝区域。
        invalid = cv2.bitwise_and(
            fruit,
            invalid,
        )

        fruit_lines = mask_to_labels(fruit, 0)
        invalid_lines = mask_to_labels(invalid, 1)

        if not fruit_lines:
            raise ValueError(
                f"没有有效水果标注：{filename}"
            )

        # 没有 invalid 的图片允许只有 fruit。
        key = f"batch_v5_{item['id']:03d}"

        if (old_labels / f"{key}.txt").exists():
            raise ValueError(
                f"与旧数据名称冲突：{key}"
            )

        additions.append({
            "source": source,
            "key": key,
            "lines": fruit_lines + invalid_lines,
        })

    temporary = ROOT / (
        "ir_v5_build_" + uuid.uuid4().hex[:8]
    )

    try:
        shutil.copytree(
            OLD_DATA,
            temporary,
        )

        for cache in temporary.rglob("*.cache"):
            cache.unlink()

        for addition in additions:
            source = addition["source"]
            key = addition["key"]

            destination = (
                temporary
                / "images"
                / "train"
                / f"{key}{source.suffix.lower()}"
            )

            shutil.copy2(
                source,
                destination,
            )

            label_path = (
                temporary
                / "labels"
                / "train"
                / f"{key}.txt"
            )

            label_path.write_text(
                "\n".join(addition["lines"]) + "\n",
                encoding="utf-8",
            )

        dataset_yaml = {
            "path": NEW_DATA.as_posix(),
            "train": "images/train",
            "val": "images/train",
            "names": {
                0: "fruit",
                1: "invalid",
            },
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
            ANNOTATIONS,
            temporary / "v5_original_annotations.json",
        )

        report = {
            "old_images": len(old_files),
            "added_images": len(additions),
            "total_images": (
                len(old_files) + len(additions)
            ),
            "validation": (
                "Training images are reused for validation; "
                "not an independent accuracy evaluation."
            ),
        }

        (temporary / MARKER).write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        temporary.rename(NEW_DATA)

    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"原有图片：{len(old_files)} 张")
    print(f"新增图片：{len(additions)} 张")
    print(
        f"合计图片："
        f"{len(old_files) + len(additions)} 张"
    )
    print(f"数据集位置：{NEW_DATA}")

    return NEW_DATA / "data.yaml"


def main():
    if not OLD_MODEL.is_file():
        raise FileNotFoundError(
            f"找不到 v4 模型：{OLD_MODEL}"
        )

    yaml_path = build_dataset()

    print("\n开始训练红外分割模型 v5。")
    print(
        "当前验证使用训练图片自身，"
        "只检查拟合情况，不代表独立测试准确率。\n"
    )

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
        project=str(ROOT / "runs"),
        name="ir_mask_demo_v5",
        exist_ok=False,
    )

    save_dir = Path(
        model.trainer.save_dir
    )

    print("\n训练完成！")
    print(
        "本次模型位置：",
        save_dir / "weights" / "best.pt",
    )


if __name__ == "__main__":
    main()