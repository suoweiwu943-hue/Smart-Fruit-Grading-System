from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import cv2
import numpy as np
import yaml


# ==================== 配置 ====================

OLD = Path(
    r"D:\fruit_ai\ir_mask_dataset_demo_v2"
)

NEW = Path(
    r"D:\fruit_ai\ir_mask_dataset_demo_v3"
)

SAMPLES = Path(
    r"D:\fruit_ai\ir_mask_new_samples"
)

JSON_PATH = SAMPLES / (
    "labels_my-project-name_2026-10-31-10-51-22.json"
)

CLASS_NAMES = {
    0: "fruit",
    1: "invalid",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


# ==================== 基础函数 ====================

def read_image(path, flags):
    """支持中文路径。"""
    data = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(data, flags)

    if image is None:
        raise RuntimeError(
            "图片读取失败：" + str(path)
        )

    return image


def image_digest(image):
    """
    按解码后的像素检查重复图片。
    尺寸也参与摘要。
    """
    digest = hashlib.sha256()
    digest.update(
        str(image.shape).encode("ascii")
    )
    digest.update(image.tobytes())
    return digest.hexdigest()


def normalized_names(names):
    if isinstance(names, dict):
        return [
            names.get(i, names.get(str(i)))
            for i in range(2)
        ]

    return names


def validate_label(path):
    """检查旧YOLO分割标签。"""
    if not path.is_file():
        raise FileNotFoundError(
            "旧图片缺少标签：" + str(path)
        )

    text = path.read_text(
        encoding="utf-8-sig"
    )

    for number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        fields = line.split()

        if (
            len(fields) < 7
            or (len(fields) - 1) % 2
        ):
            raise ValueError(
                f"分割标签格式错误：{path}:{number}"
            )

        if fields[0] not in ("0", "1"):
            raise ValueError(
                f"类别编号错误：{path}:{number}"
            )

        coordinates = np.asarray(
            fields[1:],
            dtype=np.float64,
        )

        if (
            not np.isfinite(coordinates).all()
            or (coordinates < 0).any()
            or (coordinates > 1).any()
        ):
            raise ValueError(
                f"标签坐标错误：{path}:{number}"
            )

    return text


# ==================== 多边形转换 ====================

def make_mask(
    width,
    height,
    annotations,
    category_id,
):
    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for annotation in annotations:
        if annotation["category_id"] != category_id:
            continue

        segments = annotation.get("segmentation")

        if not isinstance(segments, list):
            raise ValueError(
                "只支持COCO多边形标注，不支持RLE"
            )

        for polygon in segments:
            if (
                not isinstance(polygon, list)
                or len(polygon) < 6
                or len(polygon) % 2
            ):
                raise ValueError(
                    "多边形格式错误"
                )

            points = np.asarray(
                polygon,
                dtype=np.float64,
            ).reshape(-1, 2)

            if not np.isfinite(points).all():
                raise ValueError(
                    "多边形包含无效坐标"
                )

            if (
                (points[:, 0] < 0).any()
                or (points[:, 0] > width).any()
                or (points[:, 1] < 0).any()
                or (points[:, 1] > height).any()
            ):
                raise ValueError(
                    "多边形坐标超出图片"
                )

            points = np.rint(
                points
            ).astype(np.int32)

            points[:, 0] = np.clip(
                points[:, 0],
                0,
                width - 1,
            )

            points[:, 1] = np.clip(
                points[:, 1],
                0,
                height - 1,
            )

            cv2.fillPoly(
                mask,
                [points],
                255,
            )

    return mask


def mask_to_lines(mask, class_id):
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

        polygon = cv2.approxPolyDP(
            contour,
            epsilon=0.8,
            closed=True,
        ).reshape(-1, 2)

        if len(polygon) < 3:
            continue

        fields = [str(class_id)]

        for x, y in polygon:
            fields.extend([
                f"{float(x) / width:.8f}",
                f"{float(y) / height:.8f}",
            ])

        lines.append(
            " ".join(fields)
        )

    return lines


# ==================== 检查旧数据 ====================

def check_old_dataset():
    old_yaml = OLD / "data.yaml"

    if not old_yaml.is_file():
        raise FileNotFoundError(
            "找不到旧数据集配置："
            + str(old_yaml)
        )

    config = yaml.safe_load(
        old_yaml.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(config, dict):
        raise ValueError(
            "旧data.yaml格式错误"
        )

    names = normalized_names(
        config.get("names")
    )

    if names != ["fruit", "invalid"]:
        raise ValueError(
            "旧数据类别必须依次为fruit、invalid"
        )

    images_dir = OLD / "images" / "train"
    labels_dir = OLD / "labels" / "train"

    if not images_dir.is_dir():
        raise FileNotFoundError(
            str(images_dir)
        )

    if not labels_dir.is_dir():
        raise FileNotFoundError(
            str(labels_dir)
        )

    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            "旧训练集没有图片"
        )

    known = {}

    for image_path in image_paths:
        relative = image_path.relative_to(
            images_dir
        )

        label_path = (
            labels_dir / relative
        ).with_suffix(".txt")

        validate_label(label_path)

        image = read_image(
            image_path,
            cv2.IMREAD_COLOR,
        )

        digest = image_digest(image)

        if digest in known:
            raise RuntimeError(
                "旧数据存在重复图片，请先检查：\n"
                + str(known[digest])
                + "\n"
                + str(image_path)
            )

        known[digest] = image_path

    return image_paths, known


# ==================== 检查新增标注 ====================

def prepare_new_samples(known):
    if not JSON_PATH.is_file():
        raise FileNotFoundError(
            "找不到新增标注JSON："
            + str(JSON_PATH)
        )

    data = json.loads(
        JSON_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    categories_list = data.get(
        "categories", []
    )

    categories = {
        item["name"]: item["id"]
        for item in categories_list
    }

    if (
        len(categories_list) != 2
        or set(categories) != {
            "fruit", "invalid"
        }
        or len(set(categories.values())) != 2
    ):
        raise ValueError(
            "新增标签必须只有fruit和invalid"
        )

    images = data.get("images", [])
    annotations = data.get(
        "annotations", []
    )

    if not images:
        raise ValueError(
            "新增JSON没有图片"
        )

    image_ids = [
        item["id"] for item in images
    ]

    if len(set(image_ids)) != len(image_ids):
        raise ValueError(
            "JSON存在重复图片编号"
        )

    valid_image_ids = set(image_ids)
    valid_category_ids = set(
        categories.values()
    )

    for annotation in annotations:
        if (
            annotation.get("image_id")
            not in valid_image_ids
        ):
            raise ValueError(
                "标注引用了不存在的图片"
            )

        if (
            annotation.get("category_id")
            not in valid_category_ids
        ):
            raise ValueError(
                "标注引用了未知类别"
            )

    prepared = []
    skipped = []

    for index, item in enumerate(
        images,
        start=1,
    ):
        filename = item["file_name"]

        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or "\\" in filename
        ):
            raise ValueError(
                "JSON图片文件名无效"
            )

        source = SAMPLES / filename

        if not source.is_file():
            raise FileNotFoundError(
                "找不到对应原图："
                + str(source)
                + "\n请保持原图文件名"
                "与JSON中一致。"
            )

        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                "不支持的图片格式："
                + filename
            )

        image = read_image(
            source,
            cv2.IMREAD_COLOR,
        )

        height, width = image.shape[:2]

        if (
            width != item["width"]
            or height != item["height"]
        ):
            raise ValueError(
                "原图尺寸与标注不一致："
                + filename
            )

        digest = image_digest(image)

        if digest in known:
            skipped.append({
                "filename": filename,
                "duplicate_of": str(
                    known[digest]
                ),
            })
            continue

        image_annotations = [
            annotation
            for annotation in annotations
            if annotation["image_id"]
            == item["id"]
        ]

        fruit = make_mask(
            width,
            height,
            image_annotations,
            categories["fruit"],
        )

        invalid = make_mask(
            width,
            height,
            image_annotations,
            categories["invalid"],
        )

        # 无效区允许多圈，
        # 最终只保留水果内部的部分。
        invalid = cv2.bitwise_and(
            fruit,
            invalid,
        )

        fruit_lines = mask_to_lines(
            fruit,
            class_id=0,
        )

        invalid_lines = mask_to_lines(
            invalid,
            class_id=1,
        )

        if not fruit_lines:
            raise ValueError(
                "水果区域为空："
                + filename
            )

        if (
            np.any(invalid)
            and not invalid_lines
        ):
            raise ValueError(
                "无效区太小，无法转换："
                + filename
            )

        key = (
            f"new_batch3_{index:03d}_"
            + digest[:8]
        )

        prepared.append({
            "source": source,
            "key": key,
            "lines": (
                fruit_lines
                + invalid_lines
            ),
            "sha256": digest,
        })

        known[digest] = source

    return prepared, skipped


# ==================== 合并数据 ====================

def main():
    if NEW.exists():
        raise FileExistsError(
            "新版目录已经存在，未覆盖："
            + str(NEW)
            + "\n如果之前已成功合并，"
            "不需要再次运行此脚本。"
        )

    old_images, known = (
        check_old_dataset()
    )

    prepared, skipped = (
        prepare_new_samples(known)
    )

    if not prepared:
        raise RuntimeError(
            "没有可加入的新图片，"
            "可能全部已经存在。"
        )

    NEW.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix="ir_mask_v3_building_",
            dir=str(NEW.parent),
        )
    )

    try:
        # 复制旧数据，原目录不修改。
        shutil.copytree(
            OLD,
            temporary,
            dirs_exist_ok=True,
        )

        # 删除副本中的旧标签缓存。
        for cache in temporary.rglob(
            "*.cache"
        ):
            cache.unlink()

        images_dir = (
            temporary
            / "images"
            / "train"
        )

        labels_dir = (
            temporary
            / "labels"
            / "train"
        )

        report = []

        for item in prepared:
            image_name = (
                item["key"]
                + item["source"].suffix.lower()
            )

            label_name = (
                item["key"] + ".txt"
            )

            image_output = (
                images_dir / image_name
            )

            label_output = (
                labels_dir / label_name
            )

            if (
                image_output.exists()
                or label_output.exists()
            ):
                raise FileExistsError(
                    "文件名冲突："
                    + item["key"]
                )

            shutil.copy2(
                item["source"],
                image_output,
            )

            label_output.write_text(
                "\n".join(
                    item["lines"]
                ) + "\n",
                encoding="utf-8",
            )

            report.append({
                "original_filename": (
                    item["source"].name
                ),
                "new_filename": image_name,
                "label_filename": label_name,
                "sha256": item["sha256"],
            })

        config = {
            "path": NEW.as_posix(),
            "train": "images/train",

            # 当前仍是小样本流程试训。
            # 评估与训练使用相同图片，
            # 不构成独立验证。
            "val": "images/train",

            "names": CLASS_NAMES,
        }

        (
            temporary / "data.yaml"
        ).write_text(
            yaml.safe_dump(
                config,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        shutil.copy2(
            JSON_PATH,
            temporary
            / "batch3_annotations.json",
        )

        merge_report = {
            "old_dataset": str(OLD),
            "new_dataset": str(NEW),
            "old_image_count": len(
                old_images
            ),
            "new_images_added": len(
                prepared
            ),
            "duplicates_skipped": skipped,
            "added_files": report,
        }

        (
            temporary
            / "batch3_merge_report.json"
        ).write_text(
            json.dumps(
                merge_report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            temporary / "说明.txt"
        ).write_text(
            "水果与无效区分割试训数据。\n"
            "类别0：fruit；类别1：invalid。\n"
            "实际无效区取fruit与invalid交集。\n"
            "训练和评估使用相同图片，"
            "不能作为独立测试成绩。\n"
            "原v2数据集保持不变。\n",
            encoding="utf-8",
        )

        if NEW.exists():
            raise FileExistsError(
                "输出目录已被创建，"
                "停止写入。"
            )

        temporary.rename(NEW)

    except BaseException:
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )
        raise

    total = len(old_images) + len(
        prepared
    )

    print(
        "\n========== 合并完成 =========="
    )
    print("新版数据集：", NEW)
    print("原有图片：", len(old_images))
    print("新增图片：", len(prepared))
    print("跳过重复：", len(skipped))
    print("图片总数：", total)
    print("原v2数据集保持不变。")
    print("本程序尚未开始训练。")


if __name__ == "__main__":
    main()