from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps


VERSION = "paired-six-auto-v2"

MODEL_PATH = Path(
    r"D:\fruit_ai\runs\ir_damage_v5_trainonly\weights\best.pt"
)

SAVE_ROOT = Path(
    os.environ.get(
        "FRUIT_FLOW_ROOT",
        r"D:\fruit_project\fruit_flow_results"
    )
)

DEVICE = "0"
CONFIDENCE = 0.5
COUNTDOWN_SECONDS = 5
MIN_VALID_FRACTION = 0.50

VIEWS = [
    "正面", "背面", "左面",
    "右面", "顶部", "底部"
]

SPECIES = {
    "apple": "苹果",
    "banana": "香蕉",
    "cherry_tomato": "圣女果",
    "kiwi": "猕猴桃",
    "mandarin": "橘子",
    "xiang_pear": "香梨"
}

SAVE_ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE = SAVE_ROOT / "active.json"
CLEANUP_FILE = SAVE_ROOT / "photo_cleanup.json"
RECOGNITION_CONTROL = SAVE_ROOT / "recognition_control.json"
COLOR_SAVE_DIR = Path(os.environ.get("FRUIT_COLOR_ROOT", r"D:\fruit_project\color_photos"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

lock = threading.RLock()
model_lock = threading.Lock()
model = None


def now():
    return datetime.now().isoformat(
        timespec="milliseconds"
    )


def write_bytes(path, data):
    temporary = path.with_name(
        path.name + "." + uuid.uuid4().hex + ".tmp"
    )

    try:
        with temporary.open("wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def valid_id(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[a-f0-9]{32}", value)
    ):
        raise ValueError("任务编号无效")

    return value


def read_record(session_id):
    valid_id(session_id)

    try:
        return json.loads(
            (
                SAVE_ROOT / session_id / "record.json"
            ).read_text(encoding="utf-8")
        )

    except (OSError, ValueError):
        raise ValueError("找不到采集记录")


def all_records():
    records = []

    for path in SAVE_ROOT.glob("*/record.json"):
        try:
            record = json.loads(
                path.read_text(encoding="utf-8")
            )

            valid_id(record["session_id"])
            records.append(record)

        except (OSError, ValueError, KeyError):
            continue

    return sorted(
        records,
        key=lambda record: record["created_at"],
        reverse=True
    )


def active_id():
    if ACTIVE.is_file():
        data = json.loads(ACTIVE.read_text(encoding="utf-8"))
        sid = data.get("session_id")
        if isinstance(sid, str) and re.fullmatch(r"[a-f0-9]{32}", sid):
            if (SAVE_ROOT / sid / "record.json").is_file():
                return sid
    return None


def face_for(record, index):
    if (
        isinstance(index, bool)
        or str(index) not in (
            "1", "2", "3", "4", "5", "6"
        )
    ):
        raise ValueError("拍摄序号应为1到6")

    return record["faces"][int(index) - 1]


def payload():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        raise ValueError("请求格式错误")

    return data


def advance_capture(record):
    """只排拍摄，不等待模型分析；重复轮询不会重置五秒倒计时。"""
    if record["status"] != "collecting":
        return
    faces = record["faces"]
    if any(face["state"] in ("countdown", "capturing") for face in faces):
        return
    # 兼容升级前已经拍好、等待人工确认的有效结果。
    for face in faces:
        if face["state"] == "review":
            face["confirmed"] = bool(face.get("analysis", {}).get("usable"))
            face["state"] = "confirmed" if face["confirmed"] else "analysis_error"
    for face in faces:
        if face["state"] == "idle":
            face.update(
                capture_id=uuid.uuid4().hex,
                state="countdown",
                confirmed=False,
                error="",
                due_at=time.time() + COUNTDOWN_SECONDS,
                analysis={"status": "waiting", "usable": False, "message": "等待自动拍摄"}
            )
            return


def recompute(record):
    ratios = []
    faces = record["faces"]
    for face in faces:
        analysis = face.get("analysis", {})
        valid_pixels = analysis.get("valid_pixels", 0)
        damage_pixels = analysis.get("damage_pixels", 0)
        if (face.get("confirmed") and analysis.get("usable")
                and valid_pixels > 0 and 0 <= damage_pixels <= valid_pixels):
            ratios.append(100.0 * damage_pixels / valid_pixels)

    terminal = {"confirmed", "analysis_error", "capture_error"}
    captured = sum(bool(face.get("infrared")) for face in faces)
    finished = sum(face["state"] in terminal for face in faces)
    summary = {
        "confirmed_faces": len(ratios),
        "captured_faces": captured,
        "finished_faces": finished,
        "mean_percent": None,
        "max_percent": None,
        "grade": "未完成",
        "rule_version": "demo-5-15-v1",
        "thresholds": {"light_below": 5, "moderate_below": 15}
    }
    if len(ratios) == 6:
        average = sum(ratios) / 6
        grade = ("未检出损伤" if average == 0 else "轻度" if average < 5
                 else "中度" if average < 15 else "重度")
        summary.update(mean_percent=round(average, 2), max_percent=round(max(ratios), 2),
                       mean_percent_unrounded=average, grade=grade)
        if record["status"] != "cancelled":
            record["status"] = "complete"
    elif finished == 6:
        summary["grade"] = "数据不足，无法定级"
        if record["status"] != "cancelled":
            record["status"] = "complete_with_errors"
    elif record["status"] in ("complete", "complete_with_errors"):
        record["status"] = "collecting"
    record["summary"] = summary


def save_record(record):
    recompute(record)

    write_bytes(
        SAVE_ROOT / record["session_id"] / "record.json",
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")
    )


def capture_lookup(capture_id):
    valid_id(capture_id)

    for record in all_records():
        for face in record["faces"]:
            if face.get("capture_id") == capture_id:
                return record, face

    raise ValueError("拍摄任务已失效，请领取新任务")


def get_model():
    global model

    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "找不到模型：" + str(MODEL_PATH)
            )

        from ultralytics import YOLO

        loaded = YOLO(str(MODEL_PATH))

        names = {
            int(key): value
            for key, value in loaded.names.items()
        }

        if names != {
            0: "fruit",
            1: "invalid",
            2: "damage"
        }:
            raise ValueError(
                "模型类别不正确：" + str(names)
            )

        model = loaded

    return model


def save_png(folder, name, array):
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255

    Image.fromarray(array).save(
        folder / name,
        format="PNG"
    )


def analyze(rgb):
    height, width = rgb.shape[:2]

    with model_lock:
        result = get_model().predict(
            source=Image.fromarray(rgb),
            device=DEVICE,
            conf=CONFIDENCE,
            retina_masks=True,
            verbose=False
        )[0]

    fruits = []
    invalid = np.zeros((height, width), dtype=bool)
    damage_raw = np.zeros((height, width), dtype=bool)

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        categories = (
            result.boxes.cls.cpu().numpy().astype(int)
        )
        confidences = (
            result.boxes.conf.cpu().numpy()
        )

        for mask, category, confidence in zip(
            masks, categories, confidences
        ):
            binary = mask > 0.5

            if binary.shape != (height, width):
                binary = np.asarray(
                    Image.fromarray(
                        binary.astype(np.uint8) * 255
                    ).resize(
                        (width, height),
                        Image.Resampling.NEAREST
                    )
                ) > 0

            if category == 0 and binary.any():
                fruits.append(
                    (
                        int(binary.sum()),
                        binary,
                        float(confidence)
                    )
                )

            elif category == 1:
                invalid |= binary

            elif category == 2:
                damage_raw |= binary

    if not fruits:
        return {
            "status": "no_fruit",
            "message": "未检测到水果，不能统计损伤面积"
        }, None

    _, fruit, confidence = max(
        fruits,
        key=lambda item: item[0]
    )

    invalid &= fruit
    valid = fruit & ~invalid
    damage = damage_raw & valid

    fruit_pixels = int(fruit.sum())
    valid_pixels = int(valid.sum())
    damage_pixels = int(damage.sum())

    warnings = []

    if len(fruits) > 1:
        warnings.append(
            "检测到多个水果区域，已选面积最大区域，请检查轮廓"
        )

    if not invalid.any():
        warnings.append(
            "未检出水果内部过曝区域，请检查是否漏检"
        )

    if not valid_pixels:
        message = "有效区域为空，不能计算损伤占比"
        percent = None

    else:
        percent = round(
            100 * damage_pixels / valid_pixels, 2
        )

        message = (
            "有效区域内检出损伤"
            if damage_pixels
            else "有效区域内未检出损伤，不等于确认无损伤"
        )

    if warnings:
        message += "；" + "；".join(warnings)

    overlay = rgb.astype(np.float32)

    overlay[fruit] = (
        rgb[fruit] * 0.55
        + np.array([40, 80, 240]) * 0.45
    )

    overlay[invalid] = (
        rgb[invalid] * 0.45
        + np.array([0, 220, 220]) * 0.55
    )

    overlay[damage] = (
        rgb[damage] * 0.35
        + np.array([255, 40, 40]) * 0.65
    )

    processed = rgb.copy()
    processed[~valid] = (128, 128, 128)

    processed[damage] = (
        rgb[damage] * 0.35
        + np.array([255, 40, 40]) * 0.65
    ).astype(np.uint8)

    outcome = {
        "status": "done",
        "message": message,
        "damage_percent": percent,
        "valid_percent": round(
            100 * valid_pixels / fruit_pixels, 2
        ),
        "fruit_confidence": round(confidence, 4),
        "fruit_pixels": fruit_pixels,
        "invalid_pixels": int(invalid.sum()),
        "valid_pixels": valid_pixels,
        "damage_pixels": damage_pixels,
        "damage_outside_valid_pixels": int(
            (damage_raw & ~valid).sum()
        ),
        "model": str(MODEL_PATH),
        "confidence_threshold": CONFIDENCE
    }

    return outcome, (
        overlay.astype(np.uint8),
        processed,
        fruit,
        invalid,
        valid,
        damage,
        damage_raw
    )


def process_one(session_id, index, capture_id):
    with lock:
        record = read_record(session_id)
        face = face_for(record, index)

        if (
            face.get("capture_id") != capture_id
            or not face.get("infrared")
        ):
            return

        source = (
            SAVE_ROOT / session_id / face["infrared"]
        )

    folder = SAVE_ROOT / session_id

    try:
        with Image.open(source) as opened:
            rgb = np.asarray(
                ImageOps.exif_transpose(opened).convert("RGB")
            ).copy()

        result, arrays = analyze(rgb)
        files = []

        if arrays is not None:
            token = uuid.uuid4().hex[:10]

            labels = (
                "overlay",
                "valid_area",
                "fruit_mask",
                "invalid_mask",
                "valid_mask",
                "damage_mask",
                "damage_raw_mask"
            )

            for label, array in zip(labels, arrays):
                name = (
                    f"{index:02d}_{capture_id[:8]}_"
                    f"{label}_{token}.png"
                )

                save_png(folder, name, array)
                files.append(name)

            result.update(
                overlay=files[0],
                processed=files[1]
            )

        valid_pixels = result.get("valid_pixels", 0)
        fruit_pixels = result.get("fruit_pixels", 0)

        fraction = (
            valid_pixels / fruit_pixels
            if fruit_pixels
            else 0
        )

        result["usable"] = (
            result.get("status") == "done"
            and valid_pixels > 0
            and fraction >= MIN_VALID_FRACTION
        )

        if (
            result.get("status") == "done"
            and not result["usable"]
        ):
            result["message"] += (
                "；有效区域不足，本面不参与定级"
            )

        result.update(
            files=files,
            updated_at=now(),
            model=str(MODEL_PATH)
        )

    except Exception as error:
        logging.exception("红外分析失败")

        result = {
            "status": "error",
            "usable": False,
            "message": str(error)[:500],
            "files": []
        }

    with lock:
        record = read_record(session_id)
        face = face_for(record, index)

        if face.get("capture_id") != capture_id:
            return

        face["analysis"] = result

        face["confirmed"] = bool(result.get("usable"))
        face["state"] = "confirmed" if face["confirmed"] else "analysis_error"

        record["saved_files"].extend(
            result.get("files", [])
        )

        save_record(record)


def processing_loop():
    pause = threading.Event()

    while True:
        pending = None

        try:
            with lock:
                for record in all_records():
                    for face in record["faces"]:
                        status = face.get(
                            "analysis", {}
                        ).get("status")

                        if (
                            face.get("infrared")
                            and status in (
                                "queued", "processing"
                            )
                        ):
                            face["analysis"] = {
                                "status": "processing",
                                "usable": False,
                                "message": "正在分析"
                            }

                            save_record(record)

                            pending = (
                                record["session_id"],
                                face["index"],
                                face["capture_id"]
                            )
                            break

                    if pending:
                        break

            if pending:
                process_one(*pending)

        except Exception:
            logging.exception("后台处理异常")

        pause.wait(0.3)


@app.errorhandler(ValueError)
def bad_request(error):
    return jsonify(
        status="error",
        message=str(error)
    ), 409


@app.errorhandler(413)
def too_large(error):
    return jsonify(
        status="error",
        message="照片不能超过15MB"
    ), 413


@app.after_request
def no_cache(response):
    if not request.path.startswith("/images/"):
        response.headers["Cache-Control"] = "no-store"

    return response


@app.get("/")
def home():
    return PAGE


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        version=VERSION,
        mode="fruit-entry-ir-six-v1",
        model=str(MODEL_PATH),
        model_exists=MODEL_PATH.is_file(),
        next_fruit=True,
        photo_cleanup=True,
        capture_mode="automatic",
        countdown_seconds=COUNTDOWN_SECONDS
    )


def recognition_control():
    if not RECOGNITION_CONTROL.is_file():
        return {"epoch": 0, "requests": {}}
    data = json.loads(RECOGNITION_CONTROL.read_text(encoding="utf-8"))
    if not isinstance(data.get("epoch"), int) or data["epoch"] < 0:
        raise ValueError("识别切换状态文件异常")
    data.setdefault("requests", {})
    return data


@app.post("/next-fruit")
def next_fruit():
    data = payload()
    request_id = valid_id(data.get("request_id"))
    with lock:
        control = recognition_control()
        if request_id in control["requests"]:
            return jsonify(status="ok", recognition_epoch=control["epoch"], existing=True)
        sid = active_id()
        if sid:
            record = read_record(sid)
            if record["status"] in ("waiting_entry", "collecting"):
                record["status"] = "cancelled"
                record["stopped_reason"] = "next_fruit"
                record["stopped_at"] = now()
                save_record(record)
            if ACTIVE.exists():
                ACTIVE.unlink()
        control["epoch"] += 1
        control["requests"][request_id] = control["epoch"]
        control["requests"] = dict(list(control["requests"].items())[-32:])
        write_bytes(RECOGNITION_CONTROL, json.dumps(control).encode("utf-8"))
        return jsonify(status="ok", recognition_epoch=control["epoch"])


@app.post("/register-fruit")
def register_fruit():
    if not request.files:
        raise ValueError("请使用彩色摄像头稳定识别3秒后自动登记")
    data = (
        request.form
        if request.files
        else payload()
    )

    species = data.get("species")

    if species == "pear":
        species = "xiang_pear"

    if species == "orange":
        species = "mandarin"

    if species not in SPECIES:
        raise ValueError("请选择或识别六种水果之一")

    sid = (
        data.get("recognition_id")
        or uuid.uuid4().hex
    )

    valid_id(sid)

    confidence = float(
        data.get("confidence", 0)
    )

    if (
        not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("识别置信度无效")

    source = (
        "camera"
        if request.files
        else "manual"
    )

    content = None

    if request.files:
        image = request.files.get("image")

        if image is None:
            raise ValueError("缺少彩色识别照片")

        content = image.read()

        try:
            with Image.open(io.BytesIO(content)) as opened:
                if opened.format != "JPEG":
                    raise ValueError("需要JPEG")

                opened.verify()

        except Exception:
            raise ValueError("彩色JPEG照片无效")

    with lock:
        epoch = int(data.get("recognition_epoch", 0))
        if epoch != recognition_control()["epoch"]:
            raise ValueError("识别任务已切换，请重新稳定识别3秒")
        if (SAVE_ROOT / sid / "record.json").exists():
            record = read_record(sid)

            if (
                record["species"] != species
                or record.get("recognition_source") != source
            ):
                raise ValueError(
                    "同一登记编号不能更改种类或来源"
                )

            if (
                content is not None
                and hashlib.sha256(content).hexdigest()
                != record.get("color_sha256")
            ):
                raise ValueError(
                    "重试必须上传同一张彩色照片"
                )

            if (
                active_id() != sid
                or record["status"] == "cancelled"
            ):
                raise ValueError(
                    "原登记已结束或取消，请开始下一水果"
                )

            return jsonify(
                status="ok",
                session_id=sid,
                existing=True
            )

        previous = active_id()

        if (
            previous
            and read_record(previous)["status"]
            in ("waiting_entry", "collecting")
        ):
            raise ValueError(
                "请先完成或取消当前水果"
            )

        folder = SAVE_ROOT / sid
        folder.mkdir(exist_ok=True)

        record = {
            "session_id": sid,
            "species": species,
            "species_cn": SPECIES[species],
            "created_at": now(),
            "status": "waiting_entry",
            "saved_files": [],
            "folder": str(folder),
            "recognition_source": source,
            "recognition_confidence": confidence,
            "admission_armed": False,
            "color_image": "",
            "entry": None,
            "faces": [
                {
                    "index": index + 1,
                    "view": view,
                    "state": "idle",
                    "infrared": "",
                    "capture_id": "",
                    "confirmed": False,
                    "error": "",
                    "history": [],
                    "analysis": {
                        "status": "waiting",
                        "message": "等待拍摄",
                        "usable": False
                    }
                }
                for index, view in enumerate(VIEWS)
            ]
        }

        if content is not None:
            write_bytes(
                folder / "color_recognition.jpg",
                content
            )

            record["color_image"] = (
                "color_recognition.jpg"
            )

            record["color_sha256"] = (
                hashlib.sha256(content).hexdigest()
            )

            record["saved_files"].append(
                "color_recognition.jpg"
            )

        record["flow_version"] = VERSION
        save_record(record)

        write_bytes(
            ACTIVE,
            json.dumps(
                {"session_id": sid}
            ).encode()
        )

    return jsonify(
        status="ok",
        session_id=sid
    )


@app.get("/entry-task")
def entry_task():
    with lock:
        epoch = recognition_control()["epoch"]
        sid = active_id()

        if not sid:
            return jsonify(ready=False, recognition_epoch=epoch)

        record = read_record(sid)
        advance_capture(record)
        save_record(record)

        return jsonify(
            ready=record["status"] == "waiting_entry",
            recognition_epoch=epoch,
            session_id=sid,
            status=record["status"],
            species_cn=record["species_cn"]
        )


def accept_entry(record, source, event_id):
    record["entry"] = {
        "source": source,
        "event_id": event_id,
        "accepted_at": now()
    }

    record["status"] = "collecting"
    record["admission_armed"] = False
    advance_capture(record)
    save_record(record)


@app.post("/entry-signal")
def entry_signal():
    data = payload()
    event_id = valid_id(data.get("event_id"))

    if (
        not isinstance(data.get("blocked"), bool)
        or not isinstance(data.get("edge"), bool)
    ):
        raise ValueError(
            "blocked和edge必须为布尔值"
        )

    with lock:
        record = read_record(
            data.get("session_id")
        )

        if (
            active_id() != record["session_id"]
            or record["status"] == "cancelled"
        ):
            raise ValueError("入料登记已失效")

        if (
            record.get("entry") or {}
        ).get("event_id") == event_id:
            return jsonify(
                status="ok",
                accepted=True,
                existing=True
            )

        if record["status"] != "waiting_entry":
            return jsonify(
                status="ok",
                accepted=False,
                message="当前不等待入料"
            )

        record["last_sensor"] = {
            "blocked": data["blocked"],
            "received_at": now()
        }

        if not data["blocked"]:
            record["admission_armed"] = True

        elif (
            data["edge"]
            and record.get("admission_armed")
        ):
            accept_entry(
                record,
                "beam",
                event_id
            )

            return jsonify(
                status="ok",
                accepted=True
            )

        else:
            record["admission_armed"] = False

        save_record(record)

    return jsonify(
        status="ok",
        accepted=False
    )


@app.get("/camera-task")
def camera_task():
    with lock:
        sid = active_id()

        if not sid:
            return jsonify(ready=False)

        record = read_record(sid)

        if record["status"] != "collecting":
            return jsonify(ready=False)

        advance_capture(record)
        save_record(record)
        for face in record["faces"]:
            if face["state"] == "countdown":
                if time.time() < face["due_at"]:
                    return jsonify(ready=False)

                face["state"] = "capturing"
                save_record(record)

            if face["state"] == "capturing":
                return jsonify(
                    ready=True,
                    task_id=face["capture_id"],
                    index=face["index"],
                    view=face["view"],
                    species_cn=record["species_cn"]
                )

    return jsonify(ready=False)


@app.post("/ir-failed")
def ir_failed():
    data = payload()

    with lock:
        record, face = capture_lookup(
            data.get("task_id")
        )

        if str(data.get("index")) != str(face["index"]):
            raise ValueError("照片序号不一致")

        if (
            record["status"] == "cancelled"
            or active_id() != record["session_id"]
        ):
            raise ValueError("任务已失效")

        if (
            face["state"] == "capturing"
            and not face["infrared"]
        ):
            face.update(
                state="capture_error",
                error=str(
                    data.get("message", "拍摄失败")
                )[:300]
            )

            advance_capture(record)
            save_record(record)

    return jsonify(status="ok")


@app.post("/upload-ir")
def upload_ir():
    capture_id = request.form.get("task_id")
    uploaded = request.files.get("image")

    if uploaded is None:
        raise ValueError("缺少image照片")

    content = uploaded.read()

    try:
        with Image.open(io.BytesIO(content)) as opened:
            if opened.format != "JPEG":
                raise ValueError("需要JPEG照片")

            opened.verify()

    except Exception:
        raise ValueError("JPEG照片无效")

    with lock:
        record, face = capture_lookup(capture_id)

        if request.form.get("index") != str(face["index"]):
            raise ValueError("照片序号不一致")

        folder = SAVE_ROOT / record["session_id"]

        if face["infrared"]:
            old_hash = hashlib.sha256(
                (folder / face["infrared"]).read_bytes()
            ).digest()

            new_hash = hashlib.sha256(content).digest()

            if old_hash != new_hash:
                raise ValueError(
                    "照片内容不同，未覆盖原图"
                )

        else:
            if (
                record["status"] != "collecting"
                or active_id() != record["session_id"]
                or face["state"] != "capturing"
            ):
                raise ValueError(
                    "没有待上传的当前拍摄任务"
                )

            filename = (
                f'{face["index"]:02d}_{face["view"]}_'
                f'{capture_id}_红外原图.jpg'
            )

            write_bytes(folder / filename, content)

            face.update(
                infrared=filename,
                received_at=now(),
                state="analyzing",
                error="",
                analysis={
                    "status": "queued",
                    "usable": False,
                    "message": "照片已收到，等待分析"
                }
            )

            record["saved_files"].append(filename)
            advance_capture(record)
            save_record(record)

    return jsonify(
        status="ok",
        task_id=capture_id,
        index=face["index"]
    )


def face_actions(record, face):
    pending = face.get("analysis", {}).get("status") in ("queued", "processing")
    capture_busy = any(f["state"] in ("idle", "countdown", "capturing") for f in record["faces"])
    return {
        "can_retake": bool(active_id() == record["session_id"]
            and record["status"] not in ("waiting_entry", "cancelled")
            and not capture_busy and not pending),
        "can_reanalyze": bool(face.get("infrared") and not pending
            and face["state"] not in ("countdown", "capturing"))
    }


@app.post("/api/session/<session_id>/face/<int:index>/<action>")
def repeat_face(session_id, index, action):
    valid_id(session_id)
    data = payload()
    with lock:
        record = read_record(session_id)
        face = face_for(record, index)
        # A retry or a stale browser must never overwrite a newer capture/result.
        if (data.get("capture_id") != face.get("capture_id")
                or data.get("analysis_updated_at") != face.get("analysis", {}).get("updated_at")):
            raise ValueError("本面结果已更新，请刷新后重试")
        allowed = face_actions(record, face)
        if action == "retake":
            if not allowed["can_retake"]:
                raise ValueError("请等六面拍摄结束且本面分析完成后，在当前任务中重新拍照")
            previous = {key: value for key, value in face.items() if key != "history"}
            face.setdefault("history", []).append(previous)
            face.update(capture_id=uuid.uuid4().hex, state="countdown",
                        infrared="", confirmed=False, error="", received_at=None,
                        due_at=time.time() + COUNTDOWN_SECONDS,
                        analysis={"status": "waiting", "usable": False, "message": "等待重新拍摄"})
            record["status"] = "collecting"
        elif action == "reanalyze":
            if not allowed["can_reanalyze"]:
                raise ValueError("请先拍摄本面，或等待正在进行的分析结束")
            face.setdefault("analysis_history", []).append(face.get("analysis", {}))
            face.update(state="analyzing", confirmed=False, error="",
                        analysis={"status": "queued", "usable": False, "message": "等待重新分析"})
        else:
            raise ValueError("未知操作")
        save_record(record)
    return jsonify(status="ok")


@app.post("/cancel")
def cancel():
    with lock:
        sid = active_id()

        if sid:
            record = read_record(sid)

            if record["status"] in (
                "waiting_entry", "collecting"
            ):
                record["status"] = "cancelled"
                save_record(record)

    return jsonify(status="ok")


def read_cleanup():
    if not CLEANUP_FILE.is_file():
        return {"jobs": [], "deleted": {}}
    data = json.loads(CLEANUP_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise ValueError("照片清理队列异常，请保留文件并检查服务日志")
    data.setdefault("deleted", {})
    return data


def save_cleanup(data):
    write_bytes(CLEANUP_FILE, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def board_photo_names(record):
    names = set()
    for face in record["faces"]:
        for capture in [face] + face.get("history", []):
            capture_id = capture.get("capture_id", "")
            index = capture.get("index", face["index"])
            if re.fullmatch(r"[a-f0-9]{32}", str(capture_id)) and str(index) in ("1", "2", "3", "4", "5", "6"):
                stem = "{}_{:02d}".format(capture_id, int(index))
                names.update((stem + "_infrared.jpg", stem + ".json"))
    return sorted(names)


@app.post("/api/session/<session_id>/delete")
def delete_session(session_id):
    valid_id(session_id)
    data = payload()
    if data.get("confirm_delete") is not True:
        raise ValueError("删除任务及照片需要确认")
    with lock:
        cleanup = read_cleanup()
        existing = next((job for job in cleanup["jobs"] if job["session_id"] == session_id), None)
        folder = SAVE_ROOT / session_id
        if not (folder / "record.json").is_file():
            if session_id in cleanup["deleted"] or existing:
                return jsonify(status="ok", deleted=True, board_pending=existing is not None)
            raise ValueError("找不到历史任务")
        record = read_record(session_id)
        if active_id() == session_id and record["status"] in ("waiting_entry", "collecting"):
            raise ValueError("当前任务正在等待入料或采集中，不能删除")
        if any(face.get("analysis", {}).get("status") in ("queued", "processing") for face in record["faces"]):
            raise ValueError("任务仍在分析，请等待分析结束后删除")
        if folder.is_symlink() or folder.resolve().parent != SAVE_ROOT.resolve():
            raise ValueError("任务目录异常，未执行删除")
        names = board_photo_names(record)
        if names and existing is None:
            cleanup["jobs"].append({
                "job_id": uuid.uuid4().hex,
                "session_id": session_id,
                "created_at": now(),
                "files": names
            })
            # 先持久化SD卡清理任务；电脑或板端重启后仍可继续。
            save_cleanup(cleanup)
        try:
            color_copy = COLOR_SAVE_DIR / (session_id + "_color.jpg")
            if color_copy.exists() or color_copy.is_symlink():
                color_copy.unlink()
            # 删除本任务的所有原图、分割结果、掩膜和record.json。
            shutil.rmtree(folder)
            if ACTIVE.is_file():
                active = json.loads(ACTIVE.read_text(encoding="utf-8"))
                if active.get("session_id") == session_id:
                    ACTIVE.unlink()
        except OSError as error:
            logging.exception("任务照片删除失败")
            return jsonify(status="error", message="部分文件正在占用或无法删除，请稍后重试：" + str(error)[:200]), 500
        cleanup["deleted"][session_id] = now()
        save_cleanup(cleanup)
        return jsonify(status="ok", deleted=True, board_pending=bool(names))


@app.get("/photo-cleanup-task")
def photo_cleanup_task():
    with lock:
        cleanup = read_cleanup()
        if not cleanup["jobs"]:
            return jsonify(ready=False)
        return jsonify(ready=True, **cleanup["jobs"][0])


@app.post("/photo-cleanup-done")
def photo_cleanup_done():
    data = payload()
    job_id = valid_id(data.get("job_id"))
    with lock:
        cleanup = read_cleanup()
        cleanup["jobs"] = [job for job in cleanup["jobs"] if job["job_id"] != job_id]
        save_cleanup(cleanup)
    return jsonify(status="ok", job_id=job_id)


@app.get("/api/tasks")
def tasks():
    with lock:
        records = all_records()[:100]

        return jsonify(
            active_session=active_id(),
            records=[
                {
                    "session_id": record["session_id"],
                    "created_at": record["created_at"],
                    "status": record["status"],
                    "species_cn": record["species_cn"],
                    "summary": record["summary"],
                    "deletable": not (
                        (record["session_id"] == active_id() and record["status"] in ("waiting_entry", "collecting"))
                        or any(face.get("analysis", {}).get("status") in ("queued", "processing") for face in record["faces"])
                    )
                }
                for record in records
            ]
        )


@app.get("/api/session/<session_id>")
def session_detail(session_id):
    with lock:
        record = read_record(session_id)
        if active_id() == session_id:
            advance_capture(record)
            save_record(record)
        result = dict(record)
        result["faces"] = [dict(face, **face_actions(record, face)) for face in record["faces"]]

        result["deletable"] = not (
            (active_id() == session_id and record["status"] in ("waiting_entry", "collecting"))
            or any(face.get("analysis", {}).get("status") in ("queued", "processing") for face in record["faces"])
        )
        result["server_time"] = time.time()
        result["is_active"] = (
            active_id() == session_id
        )

        return jsonify(result)


@app.get("/api/task/<capture_id>")
def board_task_detail(capture_id):
    with lock:
        record, face = capture_lookup(capture_id)

        status = (
            "complete"
            if face["infrared"]
            else "cancelled"
            if record["status"] == "cancelled"
            else "capturing_pairs"
        )

        return jsonify(
            task_id=capture_id,
            status=status,
            pairs=[
                {
                    "index": item["index"],
                    "view": item["view"],
                    "color": "",
                    "infrared": (
                        item["infrared"]
                        if item["index"] == face["index"]
                        else ""
                    )
                }
                for item in record["faces"]
            ]
        )


@app.get("/images/<session_id>/<filename>")
def image_file(session_id, filename):
    with lock:
        record = read_record(session_id)
        folder = SAVE_ROOT / session_id

        if (
            filename not in record["saved_files"]
            or (folder / filename).resolve().parent
            != folder.resolve()
        ):
            return "照片不存在", 404

        return send_from_directory(
            str(folder), filename
        )


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>水果识别与红外六面检测</title>
<meta name="page-layout" content="face-images-repeat-v8">
<style>
*{box-sizing:border-box}
body{margin:0;background:#fafafa;color:#252525;font:14px "Microsoft YaHei",sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 30px 36px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;padding:0 0 16px;border-bottom:1px solid #d8d8d8}
h1{margin:0;font-size:23px;font-weight:500;letter-spacing:.3px}h2{margin:0;font-size:16px;font-weight:600}
p{line-height:1.8}.muted{font-size:12px;color:#777}
.panel{margin:0;padding:24px 0;border-bottom:1px solid #dedede}
.stage{display:flex;align-items:baseline;gap:14px}.count{font-size:25px;font-weight:500;min-width:32px;font-variant-numeric:tabular-nums;color:#333}
.prompt{font-size:18px;font-weight:500;margin-bottom:6px}.hint{font-size:13px;line-height:1.8;color:#777}
.heading{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:0 0 18px}
#species{font-size:12px;color:#777}
.color-row{display:grid;grid-template-columns:320px minmax(0,1fr);gap:28px;align-items:start;margin:0 0 28px}
.color-photo{min-width:0}.color-photo img,.color-photo .placeholder{width:100%;height:auto;aspect-ratio:4/3}
.color-caption{font-size:12px;color:#777;margin:8px 0 0}
.color-meta{padding:10px 0 0}.color-meta h3{font-size:14px;font-weight:400;color:#777;margin:0 0 12px}
#color-species{font-size:23px;font-weight:500;line-height:1.6;color:#252525}
.details{margin:18px 0 0;display:grid;grid-template-columns:68px minmax(0,1fr);gap:10px;font-size:13px;line-height:1.7}
.details dt{color:#888}.details dd{margin:0;overflow-wrap:anywhere}
.ir-heading{border-top:1px solid #e4e4e4;padding-top:20px;margin-top:0;margin-bottom:14px}
#progress{margin:0;font-size:12px;color:#777}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));column-gap:24px;row-gap:28px}
.card{min-width:0;background:transparent;border:0;border-radius:0;overflow:visible}
.card-head{display:flex;align-items:baseline;justify-content:space-between;gap:8px;padding:0 0 9px}
.card-head strong{font-size:14px;font-weight:500}.badge{font-size:11px;font-weight:400;color:#888;white-space:nowrap}
.card img,.placeholder{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#f0f0ed;border:1px solid #e4e4e1}
img{cursor:zoom-in}.placeholder{display:grid;place-items:center;font-size:12px;color:#999}
.metric{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin:9px 0 0;padding:0;font-size:12px;color:#888}
.percent{font-size:16px;font-weight:500;color:#333;font-variant-numeric:tabular-nums}
.active img,.active .placeholder{outline:1px solid #7d8c98;outline-offset:1px}
.error-note{font-size:12px;color:#995545;line-height:1.7;margin:8px 0 0;padding:0}.error .badge{color:#995545}
.result-top{display:flex;align-items:baseline;gap:24px;margin:0 0 18px;padding:0;font-size:14px}
.result-label{font-size:13px;color:#777;margin-right:10px}.mean{font-size:22px;font-weight:500;color:#252525;font-variant-numeric:tabular-nums}
.grade{font-size:14px;color:#555;background:none;border:0;border-radius:0;padding:0}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:12px 16px;text-align:left;border-bottom:1px solid #e4e4e4}
th{font-size:12px;font-weight:400;background:#f1f1ef;color:#777}
th:nth-child(2),td:nth-child(2){text-align:right;padding-right:42px}td:nth-child(2){font-weight:500;color:#333}
th:last-child,td:last-child{color:#777;font-size:12px}
#summary-note{font-size:12px;color:#888;margin:12px 0 0}
footer{border:0;padding-top:18px;font-size:11px;color:#888;line-height:1.9}footer p{margin:5px 0}#folder{font-size:11px;overflow-wrap:anywhere;margin-bottom:14px;color:#999}
@media(max-width:850px){main{padding:20px}.color-row{grid-template-columns:260px minmax(0,1fr);gap:22px}.grid{column-gap:16px;row-gap:22px}.card-head{align-items:flex-start;flex-direction:column;gap:4px}.card-head strong{font-size:13px}.badge{font-size:10px}.metric{font-size:11px}.percent{font-size:15px}}
@media(max-width:600px){main{padding:18px 14px}h1{font-size:20px}header{align-items:flex-start}.panel{padding:20px 0}.prompt{font-size:16px}.count{font-size:22px;min-width:22px}.stage{gap:10px}.color-row{grid-template-columns:1fr;gap:14px;margin-bottom:22px}.color-photo{max-width:360px}.color-meta{padding:0}.color-meta h3{margin-bottom:5px}.details{margin-top:12px;gap:6px}#color-species{font-size:20px}.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 14px}.heading{gap:10px}.ir-heading{align-items:flex-start;flex-direction:column;gap:7px}h2{font-size:15px}.result-top{flex-wrap:wrap;gap:10px 18px}.mean{font-size:20px}.grade{font-size:13px}th,td{padding:11px 8px}th:nth-child(2),td:nth-child(2){padding-right:12px}}

.history-bar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:16px 0;border-bottom:1px solid #dedede}
.history-bar label{font-size:13px;color:#555}
.history-bar select{font:inherit;font-size:13px;color:#333;background:#fff;border:1px solid #ccc;border-radius:2px;padding:7px 10px;max-width:100%;min-width:310px}
.history-bar button{font:inherit;font-size:12px;padding:7px 12px;color:#444;background:#fff;border:1px solid #ccc;border-radius:2px;cursor:pointer}
.history-bar button:disabled{color:#aaa;cursor:default;border-color:#ddd}
#history-note{font-size:12px;color:#888}
@media(max-width:600px){.history-bar{gap:9px}.history-bar select{min-width:0;flex:1;width:100%}.history-bar label{width:100%}#history-note{width:100%}}

.task-toolbar{display:flex;align-items:center;gap:20px;padding:14px 0;border-bottom:1px solid #dedede;min-height:58px}
.task-toolbar .history-bar{padding:0;border:0;flex:0 1 auto;flex-wrap:nowrap;gap:9px;min-width:0}
.task-toolbar .history-bar label{white-space:nowrap;width:auto}
.task-toolbar .history-bar select{width:290px;min-width:0;max-width:30vw;text-overflow:ellipsis}
.task-toolbar .history-bar button{white-space:nowrap;padding:7px 10px}
#delete-task{color:#8e5144}#delete-task:disabled{color:#aaa}
.inline-status{display:flex;align-items:baseline;gap:8px;min-width:0;flex:1;white-space:nowrap}
.inline-status .count{font-size:17px;min-width:0}
.inline-status .prompt{font-size:14px;margin:0;flex-shrink:0;white-space:nowrap}
.inline-status .hint{font-size:12px;line-height:1.6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.delete-message{font-size:12px;color:#777;margin:10px 0 0;line-height:1.7}
@media(max-width:1050px){.task-toolbar{gap:12px}.task-toolbar .history-bar select{width:230px;max-width:28vw}.inline-status .hint{display:none}}
@media(max-width:700px){.task-toolbar{flex-wrap:wrap;gap:10px}.task-toolbar .history-bar{width:100%;flex-wrap:wrap}.task-toolbar .history-bar select{width:auto;max-width:none;flex:1;min-width:100px}.inline-status{width:100%;flex-basis:100%}.inline-status .prompt{white-space:normal}.inline-status .hint{display:none}}
.face-images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.face-image{min-width:0;margin:0}.face-image figcaption{font-size:11px;color:#888;margin-top:6px}
.face-actions{display:flex;gap:8px;margin-top:10px}
.face-actions button{font:inherit;font-size:12px;border:1px solid #ccc;border-radius:2px;background:#fff;color:#444;padding:6px 10px;cursor:pointer}
.face-actions button:disabled{color:#aaa;border-color:#ddd;cursor:default}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<header><h1>水果识别与红外六面检测</h1><div><div id="connection" class="muted">正在连接</div></div></header>
<div class="task-toolbar">
<div class="history-bar">
<label for="history">历史任务</label>
<select id="history" aria-label="选择历史任务"><option value="current">当前任务（自动跟随）</option></select>
<button id="back-current" type="button" disabled>返回当前</button>
<button id="delete-task" type="button" disabled>删除任务</button>
<button id="next-fruit" type="button" title="保留已有记录并识别下一水果；未完成任务会停止采集">下一个水果</button>
<span id="history-note" hidden></span>
</div>
<div class="inline-status" aria-live="polite"><span id="countdown" class="count">—</span><span id="prompt" class="prompt">等待彩色识别</span><span id="hint" class="hint" title="同一种水果连续稳定识别3秒后自动保存照片">稳定识别3秒后保存照片</span></div>
</div>
<p id="delete-message" class="delete-message" role="status" hidden></p>
<section class="panel">
<div class="heading"><h2>采集照片</h2><span id="species">等待识别</span></div>
<div class="color-row">
<div class="color-photo"><div id="color"></div><p class="color-caption">彩色照片</p></div>
<div class="color-meta"><h3>识别结果</h3><div id="color-species">—</div><dl class="details"><dt>登记时间</dt><dd id="registered-time">—</dd><dt>拍摄方式</dt><dd>稳定识别3秒后保存</dd></dl></div>
</div>
<div class="heading ir-heading"><h2>红外六面</h2><p id="progress">尚未开始拍摄</p></div>
<div id="faces" class="grid"></div>
</section>
<section class="panel"><div class="heading"><h2>六面损伤比例汇总</h2><span class="muted">损伤面积占有效区域的比例</span></div><div class="result-top"><div><span class="result-label">平均损伤率</span><span id="mean" class="mean">—</span></div><div id="grade" class="grade">等待六面结果</div></div><div id="summary" class="table-wrap"></div><p id="summary-note" class="muted">六面有效结果齐全后计算平均值。</p></section>
<footer><p id="folder"></p><p>六面顺序：正面 → 背面 → 左面 → 右面 → 顶部 → 底部。按提示换面，每面有5秒摆放时间，到时自动拍摄，后台自动分析。</p><p>损伤率＝有效区域内的损伤像素 ÷ 有效区域像素 × 100%；有效区域＝水果区域扣除无效区域。六面有效结果齐全后计算算术平均。</p><p>演示分级：0%未检出损伤；大于0%且小于5%轻度；5%至小于15%中度；15%及以上重度。六面均值不是整果真实表面积损伤比例。</p></footer>
</main><script>
const $=id=>document.getElementById(id);
const views=['正面','背面','左面','右面','顶部','底部'];
const states={idle:'等待拍摄',countdown:'摆放倒计时',capturing:'拍摄或上传中',analyzing:'分析中',review:'分析完成',confirmed:'分析完成',capture_error:'拍摄失败',analysis_error:'结果无效'};
let current=null,busy=false,facesKey='',colorKey='',clockOffset=0;
let selectedTask='current',selectionRevision=0,historyKey='',deleting=false,switchingFruit=false,nextRequestId=null;
async function api(url,options={}){const c=new AbortController();const timer=setTimeout(()=>c.abort(),5000);try{const r=await fetch(url,{...options,cache:'no-store',signal:c.signal});const d=await r.json();if(!r.ok)throw Error(d.message||'请求失败');return d;}finally{clearTimeout(timer);}}
function photo(record,name,label){if(!name){const p=document.createElement('div');p.className='placeholder';p.textContent='等待照片';return p;}const im=document.createElement('img');const url='/images/'+encodeURIComponent(record.session_id)+'/'+encodeURIComponent(name);im.src=url;im.alt=label;im.onclick=()=>window.open(url,'_blank','noopener');return im;}
function percent(a){return a.usable&&a.damage_percent!=null?Number(a.damage_percent).toFixed(2)+'%':'—';}
function cell(row,text,tag='td'){const c=document.createElement(tag);c.textContent=text;row.append(c);}
function renderFaces(record){
 const key=record.session_id+JSON.stringify(record.faces);if(key===facesKey)return;
 facesKey=key;$('faces').replaceChildren();
 for(const face of record.faces){
  const a=face.analysis||{};const failed=['capture_error','analysis_error'].includes(face.state);
  const card=document.createElement('article');card.className='card'+(['countdown','capturing'].includes(face.state)?' active':'')+(failed?' error':'');
  const head=document.createElement('div');head.className='card-head';
  const title=document.createElement('strong');title.textContent=String(face.index).padStart(2,'0')+' '+face.view;
  const state=document.createElement('span');state.className='badge';state.textContent=states[face.state]||face.state;head.append(title,state);card.append(head);
  const images=document.createElement('div');images.className='face-images';
  for(const [name,label] of [[face.infrared,'红外原图'],[a.overlay,'分割结果'],[a.processed,'有效区域与损伤']]){
   const box=document.createElement('figure');box.className='face-image';box.append(photo(record,name,face.view+' · '+label));
   const caption=document.createElement('figcaption');caption.textContent=label;box.append(caption);images.append(box);
  }
  card.append(images);
  const metric=document.createElement('p');metric.className='metric';const label=document.createElement('span');label.textContent='损伤占有效区域';const value=document.createElement('strong');value.className='percent';value.textContent=percent(a);metric.append(label,value);card.append(metric);
  const actions=document.createElement('div');actions.className='face-actions';
  for(const [action,label,allowed] of [['retake','重新拍照',face.can_retake],['reanalyze','重新分析',face.can_reanalyze]]){
   const button=document.createElement('button');button.type='button';button.textContent=label;
   button.disabled=!allowed||faceActionBusy;button.title=action==='retake'?'六面拍摄结束后，可在当前任务中重拍；有5秒摆放时间':'使用本面已保存的原图重新分析';
   button.onclick=()=>repeatFace(record,face,action);actions.append(button);
  }
  card.append(actions);
  if(failed){const note=document.createElement('p');note.className='error-note';note.textContent=face.error||a.message||'本面结果无效';card.append(note);}
  $('faces').append(card);
 }
}
let faceActionBusy=false;
async function repeatFace(record,face,action){
 if(faceActionBusy||deleting||switchingFruit)return;
 if(action==='retake'&&!window.confirm('重新拍摄「'+face.view+'」？确认后有5秒摆放时间，原照片保留，六面平均值会重新计算。'))return;
 faceActionBusy=true;selectionRevision++;facesKey='';renderFaces(record);
 try{
  await api('/api/session/'+encodeURIComponent(record.session_id)+'/face/'+face.index+'/'+action,{
   method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({capture_id:face.capture_id,analysis_updated_at:(face.analysis||{}).updated_at??null})
  });
  $('delete-message').hidden=false;$('delete-message').textContent=action==='retake'?'请摆放「'+face.view+'」，5秒后自动重新拍摄。':'已提交「'+face.view+'」重新分析。';
 }catch(error){$('delete-message').hidden=false;$('delete-message').textContent='操作未确认：'+error.message+'。请查看最新状态后再操作。';}
 finally{faceActionBusy=false;facesKey='';if(current)renderFaces(current);refresh();}
}

function render(record){current=record;$('delete-task').disabled=deleting||switchingFruit||!record.deletable;clockOffset=record.server_time*1000-Date.now();const s=record.summary;$('species').textContent=record.species_cn;$('color-species').textContent=record.species_cn;$('registered-time').textContent=(record.created_at||'—').replace('T',' ').slice(0,19);const ckey=record.session_id+record.color_image;if(ckey!==colorKey){colorKey=ckey;$('color').replaceChildren(photo(record,record.color_image,'彩色识别照片'));}const n=s.captured_faces??record.faces.filter(f=>f.infrared).length;$('progress').textContent='已拍摄 '+n+' / 6 面 · 有效结果 '+s.confirmed_faces+' / 6 面';$('mean').textContent=s.mean_percent!=null?Number(s.mean_percent).toFixed(2)+'%':'—';$('grade').textContent=s.mean_percent!=null?s.grade:record.status==='complete_with_errors'?'数据不足，无法定级':'等待六面结果';$('summary-note').textContent=record.status==='complete_with_errors'?'有拍摄失败或无效结果，本次不计算六面平均值。':s.mean_percent!=null?'六面分析已完成。':'六面有效结果齐全后自动显示平均值与等级。';const table=document.createElement('table');const h=document.createElement('tr');for(const t of ['拍摄面','损伤占有效区域','状态'])cell(h,t,'th');table.append(h);for(const f of record.faces){const tr=document.createElement('tr');for(const v of [f.view,percent(f.analysis||{}),states[f.state]||f.state])cell(tr,v);table.append(tr);}$('summary').replaceChildren(table);$('folder').textContent='保存位置：'+record.folder;renderFaces(record);updateClock();}
function updateClock(){if(!current)return;const r=current;
if(selectedTask!=='current'){
 $('countdown').textContent='—';
 const labels={waiting_entry:'等待入料',collecting:'采集中',complete:'已完成',complete_with_errors:'数据不足',cancelled:'已取消'};
 $('prompt').textContent='历史 · '+r.species_cn+' · '+(labels[r.status]||r.status);
 $('hint').textContent='登记时间：'+(r.created_at||'').replace('T',' ').slice(0,19)+'。查看历史不会影响当前采集。';
 return;
}
$('countdown').textContent='—';if(r.status==='waiting_entry'){$('prompt').textContent=r.admission_armed?'通光已确认，请用水果遮挡光路':'识别成功，进入入料触发';$('hint').textContent=r.admission_armed?'遮挡成功后自动进入红外六面拍摄。':'请对准对射两端，先保持光路无遮挡。';return;}if(['complete','complete_with_errors'].includes(r.status)){$('countdown').textContent='✓';$('prompt').textContent=r.status==='complete'?'六面检测完成 · '+r.summary.grade:'六面流程结束 · 数据不足，无法定级';$('hint').textContent='移走当前水果，点击“下一个水果”，或在彩色窗口按 N。';return;}if(r.status==='cancelled'){$('prompt').textContent='采集已停止';$('hint').textContent='照片和已有结果已保留。';return;}const f=r.faces.find(x=>['countdown','capturing'].includes(x.state));if(!f){$('prompt').textContent='六面拍摄已结束，正在汇总分析';$('hint').textContent='分析自动进行，请稍候。';return;}$('prompt').textContent=(f.index===1?'入料成功，请摆放':'请换到')+'「'+f.view+'」 · '+f.index+'/6';if(f.state==='countdown'){const seconds=Math.max(0,Math.ceil((f.due_at*1000-Date.now()-clockOffset)/1000));$('countdown').textContent=seconds;$('hint').textContent=seconds?'请在倒计时内摆好该面，拍照时保持静止。':'倒计时结束，等待板端拍照。';}else{$('countdown').textContent='…';$('hint').textContent='正在拍摄或上传，请保持当前面，出现下一面提示后再换面。';}}
function updateHistory(data){
 const key=JSON.stringify(data.records)+data.active_session+selectedTask;
 if(key===historyKey)return;
 historyKey=key;
 const select=$('history');
 const previous=Array.from(select.options).find(o=>o.value===selectedTask);
 const retainedLabel=previous?previous.textContent:'已选历史任务';
 select.replaceChildren();
 const follow=document.createElement('option');
 follow.value='current';follow.textContent='当前任务（自动跟随）';select.append(follow);
 const labels={waiting_entry:'等待入料',collecting:'采集中',complete:'已完成',complete_with_errors:'数据不足',cancelled:'已取消'};
 let selectedListed=false;
 for(const item of data.records){
  const option=document.createElement('option');option.value=item.session_id;
  const s=item.summary||{};
  const date=(item.created_at||'').replace('T',' ').slice(0,19);
  const result=s.mean_percent!=null?Number(s.mean_percent).toFixed(2)+'% · '+s.grade:labels[item.status]||s.grade||item.status;
  option.textContent=date+' · '+item.species_cn+' · '+result+(item.session_id===data.active_session?'（当前）':'');
  if(item.session_id===selectedTask)selectedListed=true;
  select.append(option);
 }
 // 新任务超过列表上限时仍保留已经选中的历史项。
 if(selectedTask!=='current'&&!selectedListed){
  const option=document.createElement('option');option.value=selectedTask;option.textContent=retainedLabel;select.append(option);
 }
 select.value=selectedTask;
 $('back-current').disabled=selectedTask==='current';
 $('history-note').textContent=selectedTask==='current'?'正在查看当前任务':'正在查看历史任务';
}
function chooseTask(value){
 selectedTask=value;selectionRevision++;current=null;
 $('back-current').disabled=value==='current';$('delete-task').disabled=true;
 $('history-note').textContent=value==='current'?'正在查看当前任务':'正在查看历史任务';
 $('countdown').textContent='—';$('prompt').textContent='正在读取任务';$('hint').textContent='';
 refresh();
}
async function refresh(){
 if(busy||deleting||switchingFruit||faceActionBusy)return;
 busy=true;
 const revision=selectionRevision;
 try{
  const data=await api('/api/tasks');
  if(revision!==selectionRevision)return;
  updateHistory(data);
  const id=selectedTask==='current'?data.active_session:selectedTask;
  if(id){
   const record=await api('/api/session/'+encodeURIComponent(id));
   if(revision===selectionRevision)render(record);
  }else if(revision===selectionRevision){
   current=null;facesKey='';colorKey='';$('color').replaceChildren(photo(empty,'','彩色照片'));renderFaces(empty);$('species').textContent='等待识别';$('color-species').textContent='—';$('registered-time').textContent='—';$('progress').textContent='尚未开始拍摄';$('mean').textContent='—';$('grade').textContent='等待六面结果';$('summary').replaceChildren();$('summary-note').textContent='六面有效结果齐全后计算平均值。';$('folder').textContent='';$('delete-task').disabled=true;$('countdown').textContent='—';$('prompt').textContent='等待彩色摄像头识别水果';$('hint').textContent='同一种水果连续稳定识别3秒后，自动保存彩色照片。';
  }
  $('connection').textContent='接收服务已连接';
 }catch(error){
  if(revision===selectionRevision){$('connection').textContent='任务读取失败，正在重试';$('history-note').textContent=error.message||'读取失败';}
 }finally{
  busy=false;
  if(revision!==selectionRevision)refresh();
 }
}
async function deleteSelected(){
 if(deleting||switchingFruit||!current||!current.deletable)return;
 const target=current;
 const label=(target.created_at||'').replace('T',' ').slice(0,19)+' · '+target.species_cn;
 if(!window.confirm('删除任务：'+label+'？\n\n电脑上的记录、彩色及红外照片、分析图片将被永久删除。板端照片将在联网后清理。'))return;
 deleting=true;selectionRevision++;$('next-fruit').disabled=true;
 $('delete-task').disabled=true;$('history').disabled=true;$('back-current').disabled=true;
 try{
  const result=await api('/api/session/'+encodeURIComponent(target.session_id)+'/delete',{
   method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirm_delete:true})
  });
  if(result.status!=='ok')throw Error(result.message||'删除失败');
  selectedTask='current';current=null;facesKey='';colorKey='';historyKey='';selectionRevision++;
  $('delete-message').hidden=false;
  $('delete-message').textContent=result.board_pending?'任务和电脑照片已删除。板端SD卡副本将在联网后自动清理。':'任务及照片已删除。';
 }catch(error){
  $('delete-message').hidden=false;$('delete-message').textContent='删除失败：'+error.message;
 }finally{
  deleting=false;$('next-fruit').disabled=false;$('history').disabled=false;$('back-current').disabled=selectedTask==='current';
  if(current)$('delete-task').disabled=!current.deletable;
  refresh();
 }
}
async function nextFruit(){
 if(switchingFruit||deleting)return;
 switchingFruit=true;selectionRevision++;
 if(!nextRequestId){
  const bytes=new Uint8Array(16);window.crypto.getRandomValues(bytes);
  nextRequestId=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');
 }
 $('next-fruit').disabled=true;$('delete-task').disabled=true;
 $('history').disabled=true;$('back-current').disabled=true;
 try{
  const result=await api('/next-fruit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:nextRequestId})});
  if(result.status!=='ok')throw Error(result.message||'切换失败');
  nextRequestId=null;selectedTask='current';current=null;facesKey='';colorKey='';historyKey='';selectionRevision++;
  $('delete-message').hidden=false;$('delete-message').textContent='已切换。请移走上一水果，放入下一水果，保持稳定3秒；之前的记录和照片已保留。';
 }catch(error){
  $('delete-message').hidden=false;$('delete-message').textContent='切换未确认：'+error.message+'。可再次点击“下一个水果”重试。';
 }finally{
  switchingFruit=false;$('next-fruit').disabled=false;$('history').disabled=false;$('back-current').disabled=selectedTask==='current';
  if(current)$('delete-task').disabled=!current.deletable;
  refresh();
 }
}
$('next-fruit').addEventListener('click',nextFruit);
$('delete-task').addEventListener('click',deleteSelected);
const previousUpdateClock=updateClock;
updateClock=function(){previousUpdateClock();$('hint').title=$('hint').textContent;const box=$('hint').parentElement;if(box)box.title=$('prompt').textContent+'；'+$('hint').textContent;};
$('history').addEventListener('change',()=>chooseTask($('history').value));
$('back-current').addEventListener('click',()=>chooseTask('current'));

const empty={session_id:'',faces:views.map((view,i)=>({index:i+1,view,state:'idle',infrared:'',analysis:{}}))};$('color').replaceChildren(photo(empty,'','彩色照片'));renderFaces(empty);refresh();setInterval(refresh,400);setInterval(updateClock,100);
</script></body></html>
'''


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(
        logging.WARNING
    )

    threading.Thread(
        target=processing_loop,
        daemon=True,
        name="ir-analysis"
    ).start()

    print(
        "水果检测：http://127.0.0.1:5000",
        flush=True
    )

    print(
        "模型：",
        MODEL_PATH,
        "存在：",
        MODEL_PATH.is_file(),
        flush=True
    )

    print(
        "结果目录：",
        SAVE_ROOT,
        flush=True
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )