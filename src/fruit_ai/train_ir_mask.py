from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# 已经解压的遮罩效果包
SOURCE = Path(
    r"D:\fruit_ai\ir_valid_area_package\results"
)

# 自动生成新的分割数据集
DATASET = Path(
    r"D:\fruit_ai\ir_mask_dataset_demo"
)

# 训练输出位置
PROJECT = Path(r"D:\fruit_ai\runs")

# 和种类识别模型分开
PRETRAINED = "yolo11n-seg.pt"

CLASS_NAMES = {
    0: "fruit",
    1: "invalid",
}


def read_image(path, flags):
    """支持中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)

    if image is None:
        raise RuntimeError(
            "图片读取失败：" + str(path)
        )

    return image


def mask_to_labels(mask, class_id):
    """
    将二值遮罩转成YOLO分割多边形。
    每个连通区域单独生成一行。
    """
    height, width = mask.shape

    binary = (
        (mask > 127).astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        binary,
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

        values = [str(class_id)]

        for x, y in polygon:
            values.append(
                "{:.8f}".format(
                    float(x) / width
                )
            )
            values.append(
                "{:.8f}".format(
                    float(y) / height
                )
            )

        lines.append(" ".join(values))

    return lines


def prepare_dataset():
    if DATASET.exists():
        raise FileExistsError(
            "试训数据集已经存在，未覆盖："
            + str(DATASET)
            + "\n如果之前已成功生成，"
            "再次训练时可使用下面说明中的命令。"
        )

    for name in (
        "images",
        "fruit_masks",
        "invalid_masks",
    ):
        if not (SOURCE / name).is_dir():
            raise FileNotFoundError(
                "找不到文件夹："
                + str(SOURCE / name)
            )

    image_paths = sorted(
        (SOURCE / "images").glob("*.png")
    )

    if not image_paths:
        raise RuntimeError(
            "images文件夹中没有PNG原图"
        )

    prepared = []

    for image_path in image_paths:
        key = image_path.stem

        fruit_path = (
            SOURCE / "fruit_masks" / (key + ".png")
        )
        invalid_path = (
            SOURCE / "invalid_masks" / (key + ".png")
        )

        if not fruit_path.is_file():
            raise FileNotFoundError(str(fruit_path))

        if not invalid_path.is_file():
            raise FileNotFoundError(str(invalid_path))

        image = read_image(
            image_path,
            cv2.IMREAD_COLOR,
        )

        fruit = read_image(
            fruit_path,
            cv2.IMREAD_GRAYSCALE,
        )

        invalid = read_image(
            invalid_path,
            cv2.IMREAD_GRAYSCALE,
        )

        if (
            image.shape[:2] != fruit.shape
            or fruit.shape != invalid.shape
        ):
            raise RuntimeError(
                "原图与遮罩尺寸不一致：" + key
            )

        fruit = (
            (fruit > 127).astype(np.uint8) * 255
        )
        invalid = (
            (invalid > 127).astype(np.uint8) * 255
        )

        # 再次限制无效区只能位于水果内部。
        invalid = cv2.bitwise_and(
            fruit,
            invalid,
        )

        fruit_lines = mask_to_labels(
            fruit,
            class_id=0,
        )

        invalid_lines = mask_to_labels(
            invalid,
            class_id=1,
        )

        if not fruit_lines:
            raise RuntimeError(
                "水果遮罩没有有效区域：" + key
            )

        if np.any(invalid) and not invalid_lines:
            raise RuntimeError(
                "无效区太小，无法生成多边形：" + key
            )

        prepared.append(
            (
                image_path,
                fruit_lines + invalid_lines,
            )
        )

    DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix="ir_mask_building_",
            dir=str(DATASET.parent),
        )
    )

    try:
        images_dir = temporary / "images" / "train"
        labels_dir = temporary / "labels" / "train"

        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        for image_path, lines in prepared:
            shutil.copy2(
                image_path,
                images_dir / image_path.name,
            )

            label_path = (
                labels_dir
                / (image_path.stem + ".txt")
            )

            label_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

        config = {
            "path": DATASET.as_posix(),
            "train": "images/train",

            # 仅为兼容训练器的数据检查与最终评估。
            # 没有独立验证数据，不能当成泛化成绩。
            "val": "images/train",

            "names": CLASS_NAMES,
        }

        (temporary / "data.yaml").write_text(
            yaml.safe_dump(
                config,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        (temporary / "说明.txt").write_text(
            "本数据集仅用于5张图片的流程试训。\n"
            "train与val指向相同图片，"
            "不构成独立验证。\n"
            "不包含damage标签。\n"
            "原始图片未覆盖。\n",
            encoding="utf-8",
        )

        if DATASET.exists():
            raise FileExistsError(
                "输出目录已经存在，停止写入"
            )

        temporary.rename(DATASET)

    except BaseException:
        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )
        raise

    print(
        "数据集生成完成：",
        DATASET,
    )
    print(
        "训练图片：",
        len(prepared),
    )

    return DATASET / "data.yaml"


def train(data_yaml):
    print("\n开始水果与无效区域分割试训。")
    print("不包含损伤识别。")
    print(
        "注意：没有独立验证集，"
        "不能用本次指标判断实际准确率。\n"
    )

    model = YOLO(PRETRAINED)

    model.train(
        data=str(data_yaml),

        epochs=60,
        patience=0,

        imgsz=640,
        batch=2,
        device=0,
        workers=0,

        project=str(PROJECT),
        name="ir_mask_demo",

        exist_ok=False,
        resume=False,

        # fruit和invalid需要保持独立，
        # 允许它们在空间上重叠。
        overlap_mask=False,
        mask_ratio=2,

        # 当前是固定顶光场景的小样本试训。
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        degrees=0.0,
        translate=0.05,
        scale=0.1,
        fliplr=0.5,
        flipud=0.0,

        # 关闭逐轮验证。
        # 最终评估如仍输出指标，
        # 使用的也是这些训练图片。
        val=False,
        plots=True,
        save=True,
    )

    save_dir = Path(model.trainer.save_dir)

    print("\n训练结束。")
    print(
        "模型位置：",
        save_dir / "weights" / "best.pt",
    )
    print(
        "这是流程试训模型，"
        "后续需要增加不同水果实物样本。"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-existing",
        action="store_true",
        help="使用已经生成的数据集重新训练",
    )

    args = parser.parse_args()

    if args.train_existing:
        data_yaml = DATASET / "data.yaml"

        if not data_yaml.is_file():
            raise FileNotFoundError(
                "请先首次运行生成数据集"
            )

    else:
        data_yaml = prepare_dataset()

    train(data_yaml)