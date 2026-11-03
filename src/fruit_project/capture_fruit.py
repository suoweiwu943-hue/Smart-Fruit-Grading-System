from pathlib import Path
import io
import os
import subprocess
import sys
import time
import uuid
import webbrowser

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


MODEL_PATH = Path(
    r"D:\fruit_ai\runs\fruit_detect_v4\weights\best.pt"
)

RECEIVER_SCRIPT = Path(
    r"D:\fruit_project\receiver_fruit_flow.py"
)

ADDRESS = "http://127.0.0.1:5000"
VERSION = "paired-six-auto-v2"

CAMERA_INDEX = 0
CONFIDENCE = 0.75
STABLE_SECONDS = 3.0

COLOR_SAVE_DIR = Path(
    r"D:\fruit_project\color_photos"
)

NAMES = {
    "apple": "苹果",
    "banana": "香蕉",
    "cherry_tomato": "圣女果",
    "kiwi": "猕猴桃",
    "mandarin": "橘子",
    "orange": "橘子",
    "xiang_pear": "香梨",
    "pear": "小香梨",
}

WINDOW = "Fruit Recognition"

session = requests.Session()
session.trust_env = False

ignored_complete = set()


def api(method, endpoint, **kwargs):
    response = session.request(
        method,
        ADDRESS + endpoint,
        timeout=(1, 3),
        **kwargs,
    )

    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise RuntimeError("接收服务没有返回JSON")

    if not response.ok:
        raise RuntimeError(
            data.get(
                "message",
                "HTTP " + str(response.status_code),
            )
        )

    return data


def receiver_ready():
    try:
        response = session.get(
            ADDRESS + "/health",
            timeout=(1, 2),
        )
    except (
        requests.ConnectionError,
        requests.Timeout,
    ):
        return False

    response.raise_for_status()
    data = response.json()

    if (
        data.get("version") != VERSION
        or data.get("mode") != "fruit-entry-ir-six-v1"
        or data.get("next_fruit") is not True
    ):
        raise RuntimeError(
            "端口5000是旧接收程序，"
            "请先停止它，再运行receiver_fruit_flow.py。"
        )

    return True


def ensure_receiver():
    if receiver_ready():
        return

    if not RECEIVER_SCRIPT.is_file():
        raise FileNotFoundError(
            str(RECEIVER_SCRIPT)
        )

    log_path = (
        RECEIVER_SCRIPT.parent
        / "receiver_fruit_flow.log"
    )

    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(RECEIVER_SCRIPT),
            ],
            cwd=str(RECEIVER_SCRIPT.parent),
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if os.name == "nt"
                else 0
            ),
        )

    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        if receiver_ready():
            return

        if process.poll() is not None:
            raise RuntimeError(
                "receiver启动失败，请查看："
                + str(log_path)
            )

        time.sleep(0.5)

    raise RuntimeError("receiver启动超时")


def get_font(size):
    for filename in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if Path(filename).is_file():
            return ImageFont.truetype(
                filename,
                size,
            )

    raise FileNotFoundError("找不到中文字体")


def preview(
    frame,
    title,
    subtitle,
    hint,
    fonts,
):
    rgb = cv2.cvtColor(
        cv2.resize(frame, (800, 600)),
        cv2.COLOR_BGR2RGB,
    )

    canvas = Image.new(
        "RGB",
        (800, 750),
        "#f5f5f1",
    )

    canvas.paste(
        Image.fromarray(rgb),
        (0, 0),
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (20, 614),
        title,
        font=fonts[0],
        fill="#293029",
    )

    draw.text(
        (20, 661),
        subtitle,
        font=fonts[1],
        fill="#4e6048",
    )

    draw.text(
        (20, 709),
        hint,
        font=fonts[1],
        fill="#7d8974",
    )

    return cv2.cvtColor(
        np.array(canvas),
        cv2.COLOR_RGB2BGR,
    )


def jpeg_bytes(frame):
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not ok:
        raise RuntimeError("彩色照片编码失败")

    return encoded.tobytes()


def save_color_photo(recognition_id, content):
    """稳定识别完成时只保存一次，重传复用相同照片。"""
    COLOR_SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = COLOR_SAVE_DIR / (
        recognition_id + "_color.jpg"
    )

    temporary = path.with_suffix(".jpg.tmp")

    with temporary.open("wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())

    os.replace(temporary, path)

    print(
        "彩色照片已保存：",
        path,
        flush=True,
    )


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            str(MODEL_PATH)
        )

    ensure_receiver()
    webbrowser.open(ADDRESS)

    fonts = (
        get_font(28),
        get_font(20),
    )

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
        if os.name == "nt"
        else cv2.CAP_ANY,
    )

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "电脑彩色摄像头打开失败"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480,
    )

    candidate = None
    stable_since = None
    stable_count = 0

    pending = None
    color = None

    current_id = None
    current_record = None

    stage = "recognizing"
    message = ""

    next_poll = 0
    next_register = 0

    recognition_epoch = None
    switch_request = None
    next_switch = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError(
                    "摄像头读取失败"
                )

            clock = time.monotonic()

            title = "请放入一个水果"
            subtitle = (
                "稳定识别3秒，登记一次彩色照片"
            )
            hint = (
                "N换下一个水果；Q退出；"
                "也可点网页“下一个水果”"
            )

            # 读取服务端任务及识别版本编号。
            if clock >= next_poll:
                next_poll = clock + 1

                try:
                    task = api(
                        "GET",
                        "/entry-task",
                    )

                    remote_epoch = int(
                        task.get(
                            "recognition_epoch",
                            0,
                        )
                    )

                    if recognition_epoch is None:
                        recognition_epoch = remote_epoch

                    elif remote_epoch != recognition_epoch:
                        # 网页或其他客户端切换了水果，
                        # 丢弃旧登记请求，重新稳定识别。
                        recognition_epoch = remote_epoch

                        current_id = None
                        current_record = None

                        pending = None
                        color = None

                        candidate = None
                        stable_since = None
                        stable_count = 0

                        ignored_complete.clear()

                        stage = (
                            "switching"
                            if switch_request is not None
                            else "recognizing"
                        )

                    remote_id = task.get(
                        "session_id"
                    )
                    remote_status = task.get(
                        "status"
                    )

                    if switch_request is not None:
                        stage = "switching"

                    elif (
                        remote_id
                        and remote_id not in ignored_complete
                        and remote_status in (
                            "waiting_entry",
                            "collecting",
                            "complete",
                            "complete_with_errors",
                        )
                    ):
                        if current_id != remote_id:
                            current_id = remote_id

                            pending = None
                            color = None

                            candidate = None
                            stable_since = None
                            stable_count = 0

                        current_record = api(
                            "GET",
                            "/api/session/" + current_id,
                        )

                        stage = "waiting"

                    elif stage == "waiting":
                        current_id = None
                        current_record = None

                        stage = "recognizing"

                        candidate = None
                        stable_since = None
                        stable_count = 0

                    message = ""

                except Exception as error:
                    message = str(error)
                    title = "接收服务连接失败"
                    subtitle = message

            # N键切换水果。
            if stage == "switching":
                title = "正在切换下一水果"
                subtitle = (
                    message
                    or "上一任务的照片和记录会保留"
                )

                if clock >= next_switch:
                    next_switch = clock + 2

                    try:
                        result = api(
                            "POST",
                            "/next-fruit",
                            json={
                                "request_id": switch_request,
                            },
                        )

                        if result.get("status") != "ok":
                            raise RuntimeError(
                                "电脑没有确认切换"
                            )

                        recognition_epoch = int(
                            result["recognition_epoch"]
                        )

                        switch_request = None
                        ignored_complete.clear()

                        current_id = None
                        current_record = None

                        pending = None
                        color = None

                        candidate = None
                        stable_since = None
                        stable_count = 0

                        stage = "recognizing"
                        message = ""
                        next_poll = 0

                    except Exception as error:
                        message = str(error)

            # 彩色稳定识别。
            if (
                stage == "recognizing"
                and recognition_epoch is not None
            ):
                result = model.predict(
                    frame,
                    conf=CONFIDENCE,
                    imgsz=640,
                    verbose=False,
                )[0]

                recognition_clock = time.monotonic()

                name = None
                confidence = 0

                if (
                    result.boxes is not None
                    and len(result.boxes) == 1
                ):
                    box = result.boxes[0]

                    detected = result.names[
                        int(box.cls[0].item())
                    ]

                    if detected in NAMES:
                        name = detected

                        confidence = float(
                            box.conf[0].item()
                        )

                if name is None:
                    candidate = None
                    stable_since = None
                    stable_count = 0

                else:
                    if name != candidate:
                        candidate = name
                        stable_since = recognition_clock
                        stable_count = 1
                    else:
                        stable_count += 1

                    elapsed = (
                        recognition_clock - stable_since
                    )

                    title = (
                        "正在确认：" + NAMES[name]
                    )

                    subtitle = (
                        "持续识别 {:.1f} / 3 秒"
                    ).format(
                        min(
                            elapsed,
                            STABLE_SECONDS,
                        )
                    )

                    if (
                        elapsed >= STABLE_SECONDS
                        and stable_count >= 3
                    ):
                        pending = {
                            "recognition_id": uuid.uuid4().hex,
                            "species": name,
                            "confidence": str(confidence),
                            "recognition_epoch": str(
                                recognition_epoch
                            ),
                        }

                        color = jpeg_bytes(
                            frame.copy()
                        )

                        save_color_photo(
                            pending["recognition_id"],
                            color,
                        )

                        stage = "registering"
                        next_register = 0

            # 提交类别和一张彩色原图。
            if stage == "registering":
                title = "正在登记水果"
                subtitle = (
                    message
                    or "只保存这一张彩色识别照片"
                )

                if clock >= next_register:
                    next_register = clock + 2

                    try:
                        response = api(
                            "POST",
                            "/register-fruit",
                            data=pending,
                            files={
                                "image": (
                                    "color.jpg",
                                    io.BytesIO(color),
                                    "image/jpeg",
                                ),
                            },
                        )

                        current_id = response[
                            "session_id"
                        ]

                        stage = "waiting"
                        message = ""
                        current_record = None

                    except Exception as error:
                        message = str(error)

            # 显示任务进展。
            if stage == "waiting":
                record = current_record or {}
                status = record.get("status")

                title = (
                    record.get("species_cn")
                    or "水果"
                ) + " · 已登记"

                if status == "waiting_entry":
                    subtitle = (
                        "等待对射通光确认，"
                        "再遮挡光路入料"
                    )
                else:
                    subtitle = (
                        "请按网页提示每5秒换面，"
                        "自动拍摄与分析"
                    )

                if status in (
                    "complete",
                    "complete_with_errors",
                ):
                    title = (
                        record.get(
                            "species_cn",
                            "水果",
                        )
                        + " · "
                        + record["summary"]["grade"]
                    )

                    mean = record["summary"][
                        "mean_percent"
                    ]

                    subtitle = (
                        (
                            "平均损伤 {}%；".format(mean)
                            if mean is not None
                            else "数据不足，无法定级；"
                        )
                        + "移走本水果，按N识别下一个"
                    )

                    hint = (
                        "N识别下一个；Q退出"
                    )

                if message:
                    subtitle = (
                        "连接异常：" + message
                    )

            cv2.imshow(
                WINDOW,
                preview(
                    frame,
                    title,
                    subtitle,
                    hint,
                    fonts,
                ),
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            if key in (
                ord("n"),
                ord("N"),
            ):
                # 任意阶段都可以切换。
                # 网络失败时复用同一个请求编号重试。
                if switch_request is None:
                    switch_request = uuid.uuid4().hex

                stage = "switching"
                message = ""
                next_switch = 0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()