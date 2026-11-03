from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import logging
import os
import re
import threading
import uuid

import numpy as np
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
)
from PIL import Image, ImageOps


# ==================== 配置 ====================

# 与现有板端程序保持一致。
VERSION = "paired-six-v1"

MODEL_PATH = Path(
    r"D:\fruit_ai\runs\ir_mask_demo_v5\weights\best.pt"
)

SAVE_ROOT = Path(
    os.environ.get(
        "IR_BOARD_TEST_ROOT",
        r"D:\fruit_project\ir_board_test_results",
    )
)

DEVICE = "0"       # RTX 4050；使用 CPU 时改为 "cpu"
CONFIDENCE = 0.5

SAVE_ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE = SAVE_ROOT / "active.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

lock = threading.RLock()
model_lock = threading.Lock()
model = None

VIEWS = [
    "正面",
    "左面",
    "背面",
    "右面",
    "顶部",
    "底部",
]


# ==================== 文件与记录 ====================

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


def save_record(record):
    path = (
        SAVE_ROOT
        / record["task_id"]
        / "record.json"
    )

    write_bytes(
        path,
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
    )


def get_record(task_id):
    if (
        not isinstance(task_id, str)
        or not re.fullmatch(
            r"[a-f0-9]{32}",
            task_id,
        )
    ):
        raise ValueError("任务编号无效")

    try:
        path = (
            SAVE_ROOT
            / task_id
            / "record.json"
        )

        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError):
        raise ValueError("找不到任务记录")


def active_id():
    if ACTIVE.is_file():
        return json.loads(
            ACTIVE.read_text(encoding="utf-8")
        ).get("task_id")

    return None


def all_records():
    records = []

    for path in SAVE_ROOT.glob("*/record.json"):
        try:
            records.append(
                json.loads(
                    path.read_text(encoding="utf-8")
                )
            )

        except (OSError, ValueError):
            continue

    return sorted(
        records,
        key=lambda item: item.get(
            "created_at", ""
        ),
        reverse=True,
    )


def body():
    value = request.get_json(silent=True)

    if not isinstance(value, dict):
        raise ValueError("请求格式错误")

    return value


def first_pair(record):
    return record["pairs"][0]


# ==================== 红外模型 ====================

def get_model():
    global model

    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"找不到模型：{MODEL_PATH}"
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
        }:
            raise ValueError(
                f"模型类别不正确：{names}"
            )

        model = loaded

    return model


def save_png(folder, name, array):
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255

    Image.fromarray(array).save(
        folder / name,
        format="PNG",
    )


def analyze(rgb):
    height, width = rgb.shape[:2]

    with model_lock:
        result = get_model().predict(
            source=Image.fromarray(rgb),
            device=DEVICE,
            conf=CONFIDENCE,
            retina_masks=True,
            verbose=False,
        )[0]

    fruits = []

    invalid = np.zeros(
        (height, width),
        dtype=bool,
    )

    if (
        result.masks is not None
        and result.boxes is not None
    ):
        masks = (
            result.masks.data.cpu().numpy()
        )

        classes = (
            result.boxes.cls.cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf.cpu().numpy()
        )

        for mask, category, confidence in zip(
            masks,
            classes,
            confidences,
        ):
            binary = mask > 0.5

            if binary.shape != (height, width):
                binary = np.asarray(
                    Image.fromarray(
                        binary.astype(np.uint8) * 255
                    ).resize(
                        (width, height),
                        Image.Resampling.NEAREST,
                    )
                ) > 0

            if category == 0 and binary.any():
                fruits.append((
                    int(binary.sum()),
                    binary,
                    float(confidence),
                ))

            elif category == 1:
                invalid |= binary

    if not fruits:
        return {
            "status": "no_fruit",
            "message": (
                "未检测到水果，未生成有效区域图"
            ),
        }, None

    # 单果测试，选择面积最大的水果区域。
    _, fruit, confidence = max(
        fruits,
        key=lambda item: item[0],
    )

    # 实际过曝区域 = 水果与过曝掩膜的交集。
    invalid &= fruit

    # 有效区域 = 水果减去过曝区域。
    valid = fruit & ~invalid

    processed = rgb.copy()
    processed[~valid] = (128, 128, 128)

    # 蓝色：水果；青色：水果内部过曝区域。
    overlay = rgb.astype(np.float32)

    overlay[fruit] = (
        overlay[fruit] * 0.55
        + np.array([40, 80, 240]) * 0.45
    )

    overlay[invalid] = (
        rgb[invalid] * 0.45
        + np.array([0, 220, 220]) * 0.55
    )

    warnings = []

    if len(fruits) > 1:
        warnings.append(
            "检测到多个水果区域，已选择面积最大的区域"
        )

    if not invalid.any():
        warnings.append(
            "未检测到水果内部过曝区域，请检查是否漏检"
        )

    outcome = {
        "status": "done",
        "message": (
            "；".join(warnings) or "处理完成"
        ),
        "valid_percent": round(
            100
            * int(valid.sum())
            / int(fruit.sum()),
            2,
        ),
        "fruit_confidence": round(
            confidence, 4
        ),
        "fruit_pixels": int(fruit.sum()),
        "invalid_pixels": int(invalid.sum()),
        "valid_pixels": int(valid.sum()),
    }

    return outcome, (
        overlay.astype(np.uint8),
        processed,
        fruit,
        invalid,
        valid,
    )


# ==================== 后台处理 ====================

def processing_loop():
    pause = threading.Event()

    while True:
        try:
            with lock:
                pending = next(
                    (
                        record
                        for record in all_records()
                        if record.get(
                            "analysis", {}
                        ).get("status") in (
                            "queued",
                            "processing",
                        )
                    ),
                    None,
                )

                if pending:
                    pending["analysis"] = {
                        "status": "processing",
                        "message": "正在处理红外照片",
                    }

                    save_record(pending)

            if pending:
                process_one(
                    pending["task_id"]
                )

        except Exception:
            logging.exception("后台处理异常")

        pause.wait(0.5)


def process_one(task_id):
    with lock:
        record = get_record(task_id)
        original = first_pair(record)[
            "infrared"
        ]

    folder = SAVE_ROOT / task_id

    try:
        with Image.open(
            folder / original
        ) as opened:
            rgb = np.asarray(
                ImageOps.exif_transpose(
                    opened
                ).convert("RGB")
            ).copy()

        outcome, arrays = analyze(rgb)

        filenames = []

        if arrays is not None:
            token = uuid.uuid4().hex[:10]

            labels = (
                "overlay",
                "valid_area",
                "fruit_mask",
                "invalid_mask",
                "valid_mask",
            )

            for label, array in zip(
                labels,
                arrays,
            ):
                name = f"{label}_{token}.png"

                save_png(
                    folder,
                    name,
                    array,
                )

                filenames.append(name)

            outcome.update(
                overlay=filenames[0],
                processed=filenames[1],
            )

        outcome.update(
            files=filenames,
            updated_at=now(),
        )

    except Exception as error:
        logging.exception("红外分割失败")

        outcome = {
            "status": "error",
            "message": str(error)[:500],
            "files": [],
        }

    with lock:
        record = get_record(task_id)
        record["analysis"] = outcome
        save_record(record)


# ==================== 通用接口 ====================

@app.errorhandler(ValueError)
def invalid(error):
    return jsonify(
        status="error",
        message=str(error),
    ), 409


@app.errorhandler(413)
def too_large(error):
    return jsonify(
        status="error",
        message="单张照片不能超过15MB",
    ), 413


@app.after_request
def no_cache(response):
    if not request.path.startswith(
        "/images/"
    ):
        response.headers[
            "Cache-Control"
        ] = "no-store"

    return response


@app.get("/")
def home():
    return PAGE


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        version=VERSION,
        model_exists=MODEL_PATH.is_file(),
    )


# ==================== 网页发起拍摄 ====================

@app.post("/shoot")
def shoot():
    with lock:
        previous = active_id()

        if previous:
            old = get_record(previous)

            if (
                old["status"]
                == "capturing_pairs"
            ):
                return jsonify(
                    status="ok",
                    task_id=previous,
                    existing=True,
                )

        task_id = uuid.uuid4().hex

        (SAVE_ROOT / task_id).mkdir()

        record = {
            "task_id": task_id,
            "species": "fruit",
            "species_cn": "红外测试",
            "created_at": now(),
            "status": "capturing_pairs",
            "folder": str(
                SAVE_ROOT / task_id
            ),
            "photos": [],
            "infrared_photos": [],
            "analysis": {
                "status": "waiting",
                "message": "等待板子拍摄并上传",
            },
            # 保留六面记录结构以兼容现有协议。
            # 本测试程序只拍第一张。
            "pairs": [
                {
                    "index": index + 1,
                    "view": view,
                    "color": "",
                    "infrared": "",
                    "ir_state": (
                        "queued"
                        if index == 0
                        else "idle"
                    ),
                    "error": "",
                }
                for index, view in enumerate(
                    VIEWS
                )
            ],
        }

        save_record(record)

        write_bytes(
            ACTIVE,
            json.dumps({
                "task_id": task_id
            }).encode("utf-8"),
        )

    return jsonify(
        status="ok",
        task_id=task_id,
    )


# ==================== 板端领取拍摄任务 ====================

@app.get("/camera-task")
def camera_task():
    with lock:
        task_id = active_id()

        if not task_id:
            return jsonify(ready=False)

        record = get_record(task_id)
        pair = first_pair(record)

        if (
            record["status"]
            != "capturing_pairs"
            or pair["infrared"]
            or pair["ir_state"] not in (
                "queued",
                "capturing",
            )
        ):
            return jsonify(ready=False)

        if pair["ir_state"] == "queued":
            pair["ir_state"] = "capturing"
            save_record(record)

        return jsonify(
            ready=True,
            task_id=task_id,
            index=1,
            view=pair["view"],
            species_cn=record["species_cn"],
        )


# ==================== 板端报告失败 ====================

@app.post("/ir-failed")
def ir_failed():
    data = body()

    if str(data.get("index")) != "1":
        raise ValueError(
            "测试任务照片序号应为1"
        )

    with lock:
        record = get_record(
            data.get("task_id")
        )

        if (
            record["task_id"] != active_id()
            or record["status"]
            != "capturing_pairs"
        ):
            raise ValueError(
                "不是当前拍摄任务"
            )

        pair = first_pair(record)

        if not pair["infrared"]:
            pair["ir_state"] = "error"

            pair["error"] = str(
                data.get(
                    "message",
                    "板端拍摄失败",
                )
            )[:300]

            save_record(record)

    return jsonify(
        status="ok",
        task_id=record["task_id"],
        index=1,
    )


# ==================== 接收板端照片 ====================

@app.post("/upload-ir")
def upload_ir():
    task_id = request.form.get(
        "task_id"
    )

    if request.form.get("index") != "1":
        raise ValueError(
            "测试任务照片序号应为1"
        )

    uploaded = request.files.get(
        "image"
    )

    if uploaded is None:
        raise ValueError("缺少image照片")

    data = uploaded.read()

    try:
        with Image.open(
            io.BytesIO(data)
        ) as opened:
            if opened.format != "JPEG":
                raise ValueError(
                    "需要JPEG照片"
                )

            opened.verify()

    except Exception:
        raise ValueError("JPEG照片无效")

    with lock:
        record = get_record(task_id)
        pair = first_pair(record)

        # 已完成任务允许相同照片重复上传确认。
        if record["status"] == "complete":
            if not pair["infrared"]:
                raise ValueError(
                    "任务状态无效"
                )

        elif (
            task_id != active_id()
            or record["status"]
            != "capturing_pairs"
            or pair["ir_state"] not in (
                "queued",
                "capturing",
                "done",
            )
        ):
            raise ValueError(
                "不是待接收照片的当前任务"
            )

        path = (
            SAVE_ROOT
            / task_id
            / "infrared_original.jpg"
        )

        if path.exists():
            old_digest = hashlib.sha256(
                path.read_bytes()
            ).digest()

            new_digest = hashlib.sha256(
                data
            ).digest()

            if old_digest != new_digest:
                raise ValueError(
                    "同名照片内容不同，未覆盖原图"
                )

        else:
            write_bytes(path, data)

        if not pair["infrared"]:
            pair.update(
                infrared=path.name,
                infrared_received_at=now(),
                ir_state="done",
                error="",
            )

            record[
                "infrared_photos"
            ].append({
                "filename": path.name,
                "view": pair["view"],
                "index": 1,
            })

            record.update(
                status="complete",
                completed_at=now(),
            )

            record["analysis"] = {
                "status": "queued",
                "message": (
                    "照片已收到，等待分割"
                ),
            }

            save_record(record)

    return jsonify(
        status="ok",
        task_id=task_id,
        index=1,
    )


# ==================== 重试与取消 ====================

@app.post("/retry")
def retry():
    with lock:
        task_id = active_id()

        if not task_id:
            raise ValueError("没有当前任务")

        record = get_record(task_id)

        if (
            record["status"]
            != "capturing_pairs"
        ):
            raise ValueError(
                "当前没有待拍摄任务，"
                "请点击拍摄新照片"
            )

        first_pair(record).update(
            ir_state="queued",
            error="",
        )

        save_record(record)

    return jsonify(
        status="ok",
        task_id=task_id,
    )


@app.post("/cancel")
def cancel():
    data = (
        request.get_json(silent=True)
        or {}
    )

    with lock:
        task_id = (
            data.get("task_id")
            or active_id()
        )

        if not task_id:
            raise ValueError("没有当前任务")

        record = get_record(task_id)

        if (
            record["status"]
            == "capturing_pairs"
        ):
            record["status"] = "cancelled"
            save_record(record)

    return jsonify(
        status="ok",
        task_id=task_id,
    )


@app.post("/reprocess/<task_id>")
def reprocess(task_id):
    with lock:
        record = get_record(task_id)

        if not first_pair(record)[
            "infrared"
        ]:
            raise ValueError(
                "尚未收到红外照片"
            )

        if record.get(
            "analysis", {}
        ).get("status") not in (
            "queued",
            "processing",
        ):
            record["analysis"] = {
                "status": "queued",
                "message": "等待重新处理",
            }

            save_record(record)

    return jsonify(
        status="ok",
        task_id=task_id,
    )


# ==================== 查询与图片 ====================

@app.get("/api/tasks")
def tasks():
    with lock:
        return jsonify(
            all_records()[:100]
        )


@app.get("/api/task/<task_id>")
def task(task_id):
    with lock:
        return jsonify(
            get_record(task_id)
        )


@app.get("/images/<task_id>/<filename>")
def image_file(task_id, filename):
    with lock:
        record = get_record(task_id)

        allowed = [
            first_pair(record)[
                "infrared"
            ]
        ] + record.get(
            "analysis", {}
        ).get("files", [])

        folder = SAVE_ROOT / task_id

        if (
            filename not in allowed
            or (
                folder / filename
            ).resolve().parent
            != folder.resolve()
        ):
            return "不存在", 404

        return send_from_directory(
            str(folder),
            filename,
        )


# ==================== 网页 ====================

PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">
<title>板端红外拍摄测试</title>

<style>
*{
  box-sizing:border-box;
}

body{
  margin:0;
  background:#f4f5f0;
  color:#29352c;
  font:14px "Microsoft YaHei",sans-serif;
}

main{
  max-width:1250px;
  margin:auto;
  padding:24px;
}

h1{
  font-size:25px;
  font-weight:500;
}

p{
  line-height:1.7;
}

button{
  cursor:pointer;
  padding:9px 14px;
  border:1px solid #bac8b3;
  background:white;
  color:#3e5c38;
  border-radius:4px;
  margin:3px;
}

button:disabled{
  opacity:.5;
  cursor:default;
}

.toolbar,.card{
  background:white;
  padding:16px;
  border:1px solid #dce0d5;
  border-radius:5px;
  margin-top:16px;
}

.pictures{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
}

figure{
  margin:0;
  min-width:0;
}

img{
  width:100%;
  height:290px;
  object-fit:contain;
  background:#eee;
  cursor:zoom-in;
}

figcaption,.muted{
  font-size:12px;
  color:#788370;
  overflow-wrap:anywhere;
}

.error{
  color:#a84336;
}

@media(max-width:700px){
  main{
    padding:12px;
  }

  .pictures{
    grid-template-columns:1fr;
  }
}
</style>
</head>

<body>
<main>
  <h1>板端红外拍摄测试</h1>

  <p>
    将水果放在板端红外摄像头前，
    点击拍摄后保持静止，等待照片上传。
  </p>

  <div class="toolbar">
    <button id="shoot">板端拍摄一张</button>
    <button id="retry">重试拍摄</button>
    <button id="cancel">取消当前拍摄</button>

    <span id="connection">正在连接</span>
  </div>

  <p id="notice"></p>

  <p class="muted">
    蓝色为水果分割区域，青色为水果内部过曝区域。
    有效区域图将背景和过曝区域填灰。
    面积占比不代表合格率或损伤率。
    点击图片放大。
  </p>

  <div id="results"></div>
</main>

<script>
const $ = id => document.getElementById(id);

let polling = false;
let actionBusy = false;
let lastKey = "";

async function api(url,options={}){
  const controller = new AbortController();

  const timer = setTimeout(
    () => controller.abort(),
    5000
  );

  try{
    const response = await fetch(url,{
      ...options,
      cache:"no-store",
      signal:controller.signal
    });

    const data = await response.json();

    if(!response.ok){
      throw new Error(
        data.message || "请求失败"
      );
    }

    return data;

  }finally{
    clearTimeout(timer);
  }
}

async function act(url){
  if(actionBusy){
    return;
  }

  actionBusy = true;
  $("shoot").disabled = true;

  try{
    const data = await api(url,{
      method:"POST"
    });

    $("notice").textContent = data.existing
      ? "已有待拍摄任务，请等待或点击重试拍摄。"
      : "指令已提交。";

    await refresh();

  }catch(error){
    $("notice").textContent = error.message;

  }finally{
    actionBusy = false;
    $("shoot").disabled = false;
  }
}

function render(records){
  const key = JSON.stringify(records);

  if(key === lastKey){
    return;
  }

  lastKey = key;
  $("results").replaceChildren();

  for(const record of records){
    const pair = record.pairs[0];
    const analysis = record.analysis || {};

    const card = document.createElement("section");
    card.className = "card";

    const title = document.createElement("h3");

    title.textContent = (
      record.created_at.replace("T"," ")
      + " · "
      + record.task_id.slice(0,8)
    );

    card.append(title);

    const note = document.createElement("p");

    if(record.status === "cancelled"){
      note.textContent = "拍摄已取消";

    }else if(pair.error){
      note.textContent = (
        "板端拍摄失败：" + pair.error
      );

    }else if(pair.infrared){
      note.textContent = (
        analysis.status === "done"
          ? "有效区域占该面水果分割面积 "
            + analysis.valid_percent
            + "% · "
          : ""
      ) + (
        analysis.message || "等待处理"
      );

    }else if(pair.ir_state === "capturing"){
      note.textContent = (
        "板端正在拍摄或上传，请保持静止。"
      );

    }else{
      note.textContent = (
        "等待板端领取拍摄任务；"
        + "长时间无响应请检查网线和板端服务。"
      );
    }

    if(
      pair.error
      || ["error","no_fruit"].includes(
        analysis.status
      )
    ){
      note.className = "error";
    }

    card.append(note);

    const pictures = document.createElement("div");
    pictures.className = "pictures";

    const entries = [
      [pair.infrared,"红外原图"],
      [analysis.overlay,"分割叠加图"],
      [analysis.processed,"有效区域图"]
    ];

    for(const [name,label] of entries){
      if(!name){
        continue;
      }

      const figure = document.createElement("figure");
      const img = document.createElement("img");
      const caption = document.createElement("figcaption");

      const url = (
        "/images/"
        + record.task_id
        + "/"
        + encodeURIComponent(name)
      );

      img.src = url;
      img.alt = label;

      img.onclick = () => {
        window.open(
          url,
          "_blank",
          "noopener"
        );
      };

      caption.textContent = label;

      figure.append(img,caption);
      pictures.append(figure);
    }

    card.append(pictures);

    if(pair.infrared){
      const button = document.createElement("button");

      button.textContent = "重新处理";

      button.disabled = [
        "queued",
        "processing"
      ].includes(analysis.status);

      button.onclick = () => {
        act(
          "/reprocess/" + record.task_id
        );
      };

      card.append(button);
    }

    const folder = document.createElement("p");

    folder.className = "muted";
    folder.textContent = (
      "保存位置：" + record.folder
    );

    card.append(folder);
    $("results").append(card);
  }
}

async function refresh(){
  if(polling){
    return;
  }

  polling = true;

  try{
    const records = await api("/api/tasks");
    render(records);

    $("connection").textContent = (
      "接收服务已连接"
    );

  }catch(error){
    $("connection").textContent = (
      "连接中断，正在重试"
    );

  }finally{
    polling = false;
  }
}

$("shoot").onclick = () => act("/shoot");
$("retry").onclick = () => act("/retry");
$("cancel").onclick = () => act("/cancel");

refresh();
setInterval(refresh,1000);
</script>
</body>
</html>
"""


# ==================== 启动 ====================

if __name__ == "__main__":
    logging.getLogger(
        "werkzeug"
    ).setLevel(logging.WARNING)

    threading.Thread(
        target=processing_loop,
        daemon=True,
    ).start()

    print(
        "板端红外测试：http://127.0.0.1:5000",
        flush=True,
    )

    print(
        "模型：",
        MODEL_PATH,
        flush=True,
    )

    print(
        "模型存在：",
        MODEL_PATH.is_file(),
        flush=True,
    )

    print(
        "测试结果：",
        SAVE_ROOT,
        flush=True,
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )