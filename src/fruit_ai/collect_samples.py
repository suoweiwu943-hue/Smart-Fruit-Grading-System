from pathlib import Path
from datetime import datetime
import csv
import os
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ==================== 配置 ====================

SAVE_ROOT = Path(r"D:\fruit_ai\new_samples")
CAMERA_INDEX = 0

# 编号与旧数据集保持一致
CATEGORIES = {
    "0": ("apple", "苹果"),
    "1": ("banana", "香蕉"),
    "2": ("cherry_tomato", "圣女果"),
    "3": ("kiwi", "猕猴桃"),
    "4": ("mandarin", "橘子"),
    "5": ("xiang_pear", "香梨"),
    "b": ("background", "无水果背景"),
}

WINDOW_NAME = "Collect Fruit Samples"
SAVE_COOLDOWN = 0.8


# ==================== 工具函数 ====================

def load_font(size):
    for path in [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]:
        if path.is_file():
            return ImageFont.truetype(str(path), size)

    raise FileNotFoundError(
        "未找到微软雅黑或黑体字体"
    )


def new_group():
    """同一个水果使用同一个组号，方便之后划分数据集。"""
    return (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )


def save_photo(path, frame):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )

    if not ok:
        raise RuntimeError("照片编码失败")

    with path.open("xb") as output:
        output.write(encoded.tobytes())
        output.flush()
        os.fsync(output.fileno())


def append_record(filename, key, category, group_id):
    path = SAVE_ROOT / "采集记录.csv"
    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    with path.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as output:
        writer = csv.writer(output)

        if write_header:
            writer.writerow([
                "文件路径",
                "类别编号",
                "类别名称",
                "水果组号",
                "拍摄时间",
            ])

        writer.writerow([
            filename,
            "" if key == "b" else key,
            category,
            group_id,
            datetime.now().isoformat(
                timespec="seconds"
            ),
        ])


def make_preview(frame, lines, saved=False):
    photo = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    if photo.width < 640:
        height = round(
            photo.height * 640 / photo.width
        )
        photo = photo.resize(
            (640, height),
            Image.Resampling.LANCZOS
        )

    panel_height = 205
    top = photo.height

    canvas = Image.new(
        "RGB",
        (photo.width, top + panel_height),
        (245, 247, 250)
    )
    canvas.paste(photo, (0, 0))

    draw = ImageDraw.Draw(canvas)

    draw.line(
        (0, top, canvas.width, top),
        fill=(220, 225, 232),
        width=1
    )

    for index, line in enumerate(lines):
        draw.text(
            (18, top + 12 + index * 30),
            line,
            font=FONT_TITLE if index == 0 else FONT_BODY,
            fill=(
                (35, 90, 165)
                if index == 0
                else (75, 85, 100)
            )
        )

    if saved:
        draw.rectangle(
            (1, 1, photo.width - 2, top - 2),
            outline=(60, 170, 100),
            width=4
        )

    return cv2.cvtColor(
        np.array(canvas),
        cv2.COLOR_RGB2BGR
    )


# ==================== 初始化 ====================

SAVE_ROOT.mkdir(parents=True, exist_ok=True)

for category, _ in CATEGORIES.values():
    (SAVE_ROOT / category).mkdir(exist_ok=True)

FONT_TITLE = load_font(22)
FONT_BODY = load_font(18)

selected_key = None
group_id = None

group_count = 0
total_count = 0

last_save_time = -100.0
flash_until = 0.0

message = "先按数字选择水果类别，或按B选择无水果背景"

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

try:
    if not cap.isOpened():
        raise RuntimeError(
            "摄像头打不开，请先关闭capture_six和其他摄像头程序"
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_AUTOSIZE
    )

    print("照片目录：", SAVE_ROOT, flush=True)
    print("0苹果 1香蕉 2圣女果 3猕猴桃 4橘子 5香梨", flush=True)
    print("B：无水果背景；空格：拍照；N：换一个水果", flush=True)
    print("按键前请先点击摄像头窗口。", flush=True)

    read_failures = 0

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            read_failures += 1

            if read_failures >= 30:
                raise RuntimeError("摄像头连续读取失败")

            time.sleep(0.05)
            continue

        read_failures = 0

        if selected_key is None:
            title = "当前类别：尚未选择"
            group_text = "组号：尚未创建"
        else:
            category, chinese_name = CATEGORIES[selected_key]
            title = "当前类别：" + chinese_name
            group_text = "组号：" + group_id

        lines = [
            title,
            "0苹果  1香蕉  2圣女果  3猕猴桃  4橘子  5香梨",
            "B无水果背景    空格拍照    N换一个水果    Q退出",
            group_text,
            "本组 {} 张    本次共 {} 张".format(
                group_count,
                total_count
            ),
            message,
        ]

        preview = make_preview(
            frame,
            lines,
            saved=time.monotonic() < flash_until
        )

        cv2.imshow(WINDOW_NAME, preview)
        pressed = cv2.waitKey(1) & 0xFF

        if (
            pressed in (ord("q"), ord("Q"), 27)
            or cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE
            ) < 1
        ):
            break

        key = (
            chr(pressed).lower()
            if pressed != 255
            else ""
        )

        # 选择类别；切换类别时自动建立新组
        if key in CATEGORIES:
            if selected_key != key:
                selected_key = key
                group_id = new_group()
                group_count = 0

                message = (
                    "类别已选择，调整好角度后按空格拍照"
                )

                print(
                    "\n选择类别：",
                    CATEGORIES[key][1],
                    "组号：",
                    group_id,
                    flush=True
                )

        # 换另一个水果实物时按N，建立新组
        elif key == "n":
            if selected_key is None:
                message = "请先选择类别"
            else:
                group_id = new_group()
                group_count = 0
                message = "已建立新组，请拍摄另一个水果"

                print(
                    "新水果组号：",
                    group_id,
                    flush=True
                )

        # 空格拍照
        elif pressed == 32:
            now = time.monotonic()

            if selected_key is None:
                message = "请先按0～5选水果，或按B选无水果背景"
                continue

            if now - last_save_time < SAVE_COOLDOWN:
                continue

            category, chinese_name = CATEGORIES[selected_key]

            filename = "{}_{}_{:03d}.jpg".format(
                category,
                group_id,
                group_count + 1
            )

            photo_path = SAVE_ROOT / category / filename

            # 保存摄像头原图，不包含提示文字
            save_photo(photo_path, frame)

            append_record(
                str(photo_path.relative_to(SAVE_ROOT)),
                selected_key,
                category,
                group_id
            )

            group_count += 1
            total_count += 1

            last_save_time = now
            flash_until = now + 0.25

            message = "已保存，请改变角度、距离或姿态后再拍"

            print(
                "已保存：",
                photo_path,
                flush=True
            )

except KeyboardInterrupt:
    print("采集已停止。", flush=True)

finally:
    cap.release()
    cv2.destroyAllWindows()

    print(
        "本次保存 {} 张，目录：{}".format(
            total_count,
            SAVE_ROOT
        ),
        flush=True
    )