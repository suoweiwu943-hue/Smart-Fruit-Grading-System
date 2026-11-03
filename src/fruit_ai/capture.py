import argparse
import time
from pathlib import Path

import cv2


# 原始照片保存位置
SAVE_ROOT = Path(__file__).resolve().parent / "data_raw"

CLASSES = {
    "1": ("banana", "香蕉"),
    "2": ("xiang_pear", "香梨"),
    "3": ("cherry_tomato", "圣女果"),
    "4": ("mandarin", "橘子"),
    "5": ("kiwi", "猕猴桃"),
    "6": ("apple", "苹果"),
}


def choose_sample():
    """在终端选择类别和实物编号。"""
    print("\n请选择水果类别：")
    for key, (name, chinese) in CLASSES.items():
        print(f"  {key}. {chinese} ({name})")

    while True:
        choice = input("输入类别数字，或输入 q 退出：").strip()
        if choice.lower() == "q":
            return None
        if choice in CLASSES:
            break
        print("请输入1～6。")

    category, chinese = CLASSES[choice]

    while True:
        number = input(
            f"输入这一个{chinese}的实物编号，例如001："
        ).strip()

        if number.isascii() and number.isdigit() and 1 <= len(number) <= 6:
            number = number.zfill(3)
            break

        print("请输入1～6位数字。")

    sample_id = f"{category}_{number}"
    folder = SAVE_ROOT / category / sample_id
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n当前物品：{chinese}，编号：{sample_id}")
    print(f"保存目录：{folder}")
    print("同一个水果换角度拍，继续使用这个编号。")
    print("换另一个水果时，请更换编号。")

    return category, sample_id, folder


def next_photo_number(folder, sample_id):
    """接着已有编号保存，避免覆盖之前的照片。"""
    maximum = 0

    for path in folder.glob(f"{sample_id}_*.jpg"):
        suffix = path.stem[len(sample_id) + 1:]
        if suffix.isdigit():
            maximum = max(maximum, int(suffix))

    return maximum + 1


def save_photo(folder, sample_id, frame):
    """保存原始图像，不把预览文字写入训练照片。"""
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG编码失败")

    number = next_photo_number(folder, sample_id)

    while True:
        path = folder / f"{sample_id}_{number:03d}.jpg"

        try:
            # x模式禁止覆盖同名文件
            with path.open("xb") as output:
                output.write(encoded.tobytes())
            return path

        except FileExistsError:
            number += 1

        except Exception:
            # 清理写入失败的不完整照片
            if path.exists():
                path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头编号，默认0"
    )
    args = parser.parse_args()

    print("========== 果蔬照片采集 ==========")
    print("先关闭Windows相机、会议等占用摄像头的软件。")

    selection = choose_sample()
    if selection is None:
        return

    category, sample_id, folder = selection

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        print("\n摄像头打开失败。可以尝试：")
        print(r".\.venv\Scripts\python.exe capture.py --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "Fruit Capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    print("\n点击摄像头画面后使用按键：")
    print("  空格 / S：保存一张照片")
    print("  N：切换水果类别或实物编号")
    print("  Q / Esc：退出")
    print("每张照片只放一个水果，停稳后再拍。\n")

    last_saved_time = -10.0
    message = ""
    message_until = 0.0
    failed_frames = 0
    warmup_until = time.monotonic() + 2.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                failed_frames += 1
                if failed_frames >= 20:
                    print("连续读取图像失败，请检查摄像头。")
                    break
                time.sleep(0.05)
                continue

            failed_frames = 0
            now = time.monotonic()

            # 文字只画在预览副本上
            preview = frame.copy()

            lines = [
                f"Class: {category}",
                f"Sample: {sample_id}",
                "SPACE/S: save | N: change sample | Q: quit",
            ]

            if now < warmup_until:
                lines.append("Please wait: camera warming up...")

            if now < message_until:
                lines.append(message)

            for i, text in enumerate(lines):
                y = 32 + i * 32

                # 黑色描边，便于在明亮背景上看清
                cv2.putText(
                    preview, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4
                )
                cv2.putText(
                    preview, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 1
                )

            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if cv2.getWindowProperty(
                window, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

            if key in (32, ord("s"), ord("S")):
                # 限制保存频率，减少长按造成的重复照片
                if now < warmup_until or now - last_saved_time < 0.8:
                    continue

                try:
                    path = save_photo(folder, sample_id, frame)
                    print("已保存：", path)
                    message = f"Saved: {path.name}"
                    last_saved_time = now
                except Exception as error:
                    print("保存失败：", error)
                    message = "Save failed - see terminal"

                message_until = time.monotonic() + 2.0

            elif key in (ord("n"), ord("N")):
                cv2.destroyAllWindows()
                selection = choose_sample()

                if selection is None:
                    break

                category, sample_id, folder = selection
                message = ""

                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 960, 540)

                # 返回预览后先刷新图像，暂不允许立刻保存
                warmup_until = time.monotonic() + 2.0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n采集结束。照片保存在：", SAVE_ROOT)


if __name__ == "__main__":
    main()