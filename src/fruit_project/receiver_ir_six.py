from pathlib import Path
from datetime import datetime
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

VERSION = "paired-six-v1"  # 兼容现有板端服务
MODEL_PATH = Path(r"D:\fruit_ai\runs\ir_damage_v4\weights\best.pt")
SAVE_ROOT = Path(os.environ.get("IR_SIX_ROOT", r"D:\fruit_project\ir_six_results"))
DEVICE = "0"
CONFIDENCE = 0.5
COUNTDOWN_SECONDS = 5
MIN_VALID_FRACTION = 0.50  # 超过一半水果区域无效时，要求重拍
VIEWS = ["正面", "背面", "左面", "右面", "顶部", "底部"]
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
ACTIVE = SAVE_ROOT / "active.json"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
lock = threading.RLock()
model_lock = threading.Lock()
model = None


def now():
    return datetime.now().isoformat(timespec="milliseconds")


def write_bytes(path, data):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("任务编号无效")
    return value


def read_record(session_id):
    valid_id(session_id)
    try:
        return json.loads((SAVE_ROOT / session_id / "record.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("找不到采集记录")


def all_records():
    result = []
    for p in SAVE_ROOT.glob("*/record.json"):
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
            valid_id(value["session_id"])
            result.append(value)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(result, key=lambda r: r["created_at"], reverse=True)


def active_id():
    if ACTIVE.is_file():
        return json.loads(ACTIVE.read_text(encoding="utf-8")).get("session_id")
    return None


def face_for(record, index):
    if isinstance(index, bool) or str(index) not in ("1", "2", "3", "4", "5", "6"):
        raise ValueError("拍摄序号应为1到6")
    return record["faces"][int(index) - 1]


def payload():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("请求格式错误")
    return data


def recompute(record):
    usable = []
    for face in record["faces"]:
        analysis = face.get("analysis", {})
        if not face.get("confirmed") or not analysis.get("usable"):
            continue
        valid_pixels = analysis.get("valid_pixels", 0)
        damage_pixels = analysis.get("damage_pixels", 0)
        if not valid_pixels or not 0 <= damage_pixels <= valid_pixels:
            continue
        usable.append(100.0 * damage_pixels / valid_pixels)
    summary = {"confirmed_faces": len(usable), "mean_percent": None,
               "max_percent": None, "grade": "未完成", "rule_version": "demo-5-15-v1",
               "thresholds": {"light_below": 5, "moderate_below": 15}}
    if len(usable) == 6:
        average = sum(usable) / 6
        grade = "未检出损伤" if average == 0 else "轻度" if average < 5 else "中度" if average < 15 else "重度"
        summary.update(mean_percent=round(average, 2), max_percent=round(max(usable), 2),
                       mean_percent_unrounded=average, grade=grade)
        if record["status"] != "cancelled":
            record["status"] = "complete"
    elif record["status"] == "complete":
        record["status"] = "collecting"
    record["summary"] = summary


def save_record(record):
    recompute(record)
    write_bytes(SAVE_ROOT / record["session_id"] / "record.json",
                json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"))


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
            raise FileNotFoundError(f'找不到模型：{MODEL_PATH}')
        from ultralytics import YOLO
        loaded = YOLO(str(MODEL_PATH))
        names = {int(k): v for k, v in loaded.names.items()}
        if names != {0: 'fruit', 1: 'invalid', 2: 'damage'}:
            raise ValueError(f'模型类别不正确：{names}')
        model = loaded
    return model

def save_png(folder, name, array):
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    Image.fromarray(array).save(folder / name, format='PNG')

def analyze(rgb):
    height, width = rgb.shape[:2]
    with model_lock:
        result = get_model().predict(source=Image.fromarray(rgb), device=DEVICE, conf=CONFIDENCE, retina_masks=True, verbose=False)[0]
    fruits = []
    invalid = np.zeros((height, width), dtype=bool)
    damage_raw = np.zeros((height, width), dtype=bool)
    if result.masks is not None and result.boxes is not None:
        for mask, category, confidence in zip(result.masks.data.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
            binary = mask > 0.5
            if binary.shape != (height, width):
                binary = np.asarray(Image.fromarray(binary.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)) > 0
            if category == 0 and binary.any():
                fruits.append((int(binary.sum()), binary, float(confidence)))
            elif category == 1:
                invalid |= binary
            elif category == 2:
                damage_raw |= binary
    if not fruits:
        return ({'status': 'no_fruit', 'message': '未检测到水果，不能统计损伤面积'}, None)
    _, fruit, confidence = max(fruits, key=lambda item: item[0])
    invalid &= fruit
    valid = fruit & ~invalid
    damage = damage_raw & valid
    valid_pixels = int(valid.sum())
    damage_pixels = int(damage.sum())
    warnings = []
    if len(fruits) > 1:
        warnings.append('检测到多个水果区域，已选面积最大区域，请检查轮廓')
    if not invalid.any():
        warnings.append('未检出水果内部过曝区域，请检查是否漏检')
    if not valid_pixels:
        message = '有效区域为空，不能计算损伤占比'
        percent = None
    else:
        percent = round(100 * damage_pixels / valid_pixels, 2)
        message = '有效区域内检出损伤' if damage_pixels else '有效区域内未检出损伤，不等于确认无损伤'
    if warnings:
        message += '；' + '；'.join(warnings)
    overlay = rgb.astype(np.float32)
    overlay[fruit] = rgb[fruit] * 0.55 + np.array([40, 80, 240]) * 0.45
    overlay[invalid] = rgb[invalid] * 0.45 + np.array([0, 220, 220]) * 0.55
    overlay[damage] = rgb[damage] * 0.35 + np.array([255, 40, 40]) * 0.65
    processed = rgb.copy()
    processed[~valid] = (128, 128, 128)
    processed[damage] = (rgb[damage] * 0.35 + np.array([255, 40, 40]) * 0.65).astype(np.uint8)
    outcome = {'status': 'done', 'message': message, 'damage_percent': percent, 'valid_percent': round(100 * valid_pixels / int(fruit.sum()), 2), 'fruit_confidence': round(confidence, 4), 'fruit_pixels': int(fruit.sum()), 'invalid_pixels': int(invalid.sum()), 'valid_pixels': valid_pixels, 'damage_pixels': damage_pixels, 'damage_outside_valid_pixels': int((damage_raw & ~valid).sum()), 'model': str(MODEL_PATH), 'confidence_threshold': CONFIDENCE}
    return (outcome, (overlay.astype(np.uint8), processed, fruit, invalid, valid, damage, damage_raw))

def process_one(session_id, index, capture_id):
    with lock:
        record = read_record(session_id)
        face = face_for(record, index)
        if face.get("capture_id") != capture_id or not face.get("infrared"):
            return
        source = SAVE_ROOT / session_id / face["infrared"]
    folder = SAVE_ROOT / session_id
    try:
        with Image.open(source) as opened:
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert("RGB")).copy()
        result, arrays = analyze(rgb)
        files = []
        if arrays is not None:
            token = uuid.uuid4().hex[:10]
            labels = ("overlay", "valid_area", "fruit_mask", "invalid_mask", "valid_mask", "damage_mask", "damage_raw_mask")
            for label, array in zip(labels, arrays):
                name = f"{index:02d}_{capture_id[:8]}_{label}_{token}.png"
                save_png(folder, name, array)
                files.append(name)
            result.update(overlay=files[0], processed=files[1])
        valid_pixels = result.get("valid_pixels", 0)
        fruit_pixels = result.get("fruit_pixels", 0)
        fraction = valid_pixels / fruit_pixels if fruit_pixels else 0
        result["usable"] = result.get("status") == "done" and valid_pixels > 0 and fraction >= MIN_VALID_FRACTION
        if result.get("status") == "done" and not result["usable"]:
            result["message"] += "；有效区域不足，请调整后重拍本面"
        result.update(files=files, updated_at=now(), model=str(MODEL_PATH))
    except Exception as error:
        logging.exception("红外分析失败")
        result = {"status": "error", "usable": False, "message": str(error)[:500], "files": []}
    with lock:
        record = read_record(session_id)
        face = face_for(record, index)
        if face.get("capture_id") != capture_id:
            return  # 重拍后的过期处理结果不能覆盖新照片。
        face["analysis"] = result
        face["state"] = "review" if result.get("usable") else "analysis_error"
        record["saved_files"].extend(result.get("files", []))
        save_record(record)


def processing_loop():
    pause = threading.Event()
    while True:
        pending = None
        try:
            with lock:
                for record in all_records():
                    for face in record["faces"]:
                        if face.get("infrared") and face.get("analysis", {}).get("status") in ("queued", "processing"):
                            face["analysis"] = {"status": "processing", "usable": False, "message": "正在分析"}
                            save_record(record)
                            pending = (record["session_id"], face["index"], face["capture_id"])
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
    return jsonify(status="error", message=str(error)), 409


@app.errorhandler(413)
def too_large(error):
    return jsonify(status="error", message="照片不能超过15MB"), 413


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
    return jsonify(status="ok", version=VERSION, mode="infrared-six-v1",
                   model=str(MODEL_PATH), model_exists=MODEL_PATH.is_file())


@app.post("/new-six")
def new_six():
    with lock:
        previous = active_id()
        if previous:
            record = read_record(previous)
            if record["status"] == "collecting":
                return jsonify(status="ok", session_id=previous, existing=True)
        session_id = uuid.uuid4().hex
        (SAVE_ROOT / session_id).mkdir()
        record = {
            "session_id": session_id, "species": "fruit", "species_cn": "水果",
            "created_at": now(), "status": "collecting", "saved_files": [],
            "folder": str(SAVE_ROOT / session_id),
            "faces": [{"index": i + 1, "view": view, "state": "idle", "infrared": "",
                       "capture_id": "", "confirmed": False, "error": "", "history": [],
                       "analysis": {"status": "waiting", "message": "等待拍摄", "usable": False}}
                      for i, view in enumerate(VIEWS)],
        }
        save_record(record)
        write_bytes(ACTIVE, json.dumps({"session_id": session_id}).encode())
    return jsonify(status="ok", session_id=session_id)


@app.post("/prepare")
def prepare():
    data = payload()
    with lock:
        record = read_record(data.get("session_id"))
        if active_id() != record["session_id"] or record["status"] == "cancelled":
            raise ValueError("不是当前可拍摄任务")
        face = face_for(record, data.get("index"))
        for other in record["faces"]:
            if other["state"] in ("countdown", "capturing", "analyzing"):
                if other["index"] == face["index"]:
                    return jsonify(status="ok", session_id=record["session_id"], existing=True)
                raise ValueError("请等待当前面的拍摄和分析完成")
        if not face.get("infrared") and any(
            not previous["confirmed"] for previous in record["faces"][:face["index"] - 1]
        ):
            raise ValueError("请先完成并确认前面的拍摄面")
        # 重拍已完成的一面，不改变其他面；每次使用新拍摄编号避免板端复用缓存照片。
        if face.get("capture_id"):
            snapshot = {k: v for k, v in face.items() if k != "history"}
            face["history"].append(snapshot)
        face.update(capture_id=uuid.uuid4().hex, confirmed=False, infrared="", error="",
                    state="countdown", due_at=time.time() + COUNTDOWN_SECONDS,
                    analysis={"status": "waiting", "usable": False, "message": "等待新照片"})
        record["status"] = "collecting"
        save_record(record)
    return jsonify(status="ok", session_id=record["session_id"], index=face["index"])


@app.get("/camera-task")
def camera_task():
    with lock:
        sid = active_id()
        if not sid:
            return jsonify(ready=False)
        record = read_record(sid)
        if record["status"] != "collecting":
            return jsonify(ready=False)
        for face in record["faces"]:
            if face["state"] == "countdown":
                if time.time() < face["due_at"]:
                    return jsonify(ready=False)
                face["state"] = "capturing"
                save_record(record)
            if face["state"] == "capturing":
                return jsonify(ready=True, task_id=face["capture_id"], index=face["index"],
                               view=face["view"], species_cn=record["species_cn"])
    return jsonify(ready=False)


@app.post("/ir-failed")
def ir_failed():
    data = payload()
    with lock:
        record, face = capture_lookup(data.get("task_id"))
        if str(data.get("index")) != str(face["index"]):
            raise ValueError("照片序号不一致")
        if record["status"] == "cancelled" or active_id() != record["session_id"]:
            raise ValueError("任务已失效")
        if face["state"] == "capturing" and not face["infrared"]:
            face.update(state="capture_error", error=str(data.get("message", "拍摄失败"))[:300])
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
        # 相同照片的重复上传允许确认，即使已开始下一面。
        if face["infrared"]:
            if hashlib.sha256((folder / face["infrared"]).read_bytes()).digest() != hashlib.sha256(content).digest():
                raise ValueError("照片内容不同，未覆盖原图")
        else:
            if (record["status"] != "collecting" or active_id() != record["session_id"]
                    or face["state"] != "capturing"):
                raise ValueError("没有待上传的当前拍摄任务")
            filename = f'{face["index"]:02d}_{face["view"]}_{capture_id}_红外原图.jpg'
            write_bytes(folder / filename, content)
            face.update(infrared=filename, received_at=now(), state="analyzing", error="",
                        analysis={"status": "queued", "usable": False, "message": "照片已收到，等待分析"})
            record["saved_files"].append(filename)
            save_record(record)
    return jsonify(status="ok", task_id=capture_id, index=face["index"])


@app.post("/confirm")
def confirm():
    data = payload()
    with lock:
        record = read_record(data.get("session_id"))
        if record["session_id"] != active_id() or record["status"] == "cancelled":
            raise ValueError("任务已经失效")
        face = face_for(record, data.get("index"))
        if not face.get("infrared") or not face.get("analysis", {}).get("usable"):
            raise ValueError("本面结果无效，请先重拍或重新分析")
        if face["state"] not in ("review", "confirmed"):
            raise ValueError("本面尚未分析完成")
        face.update(confirmed=True, state="confirmed")
        save_record(record)
    return jsonify(status="ok")


@app.post("/reprocess")
def reprocess():
    data = payload()
    with lock:
        record = read_record(data.get("session_id"))
        if record["status"] == "cancelled" or record["session_id"] != active_id():
            raise ValueError("只能重新分析当前任务")
        face = face_for(record, data.get("index"))
        if not face["infrared"]:
            raise ValueError("本面尚未收到照片")
        if face.get("analysis", {}).get("status") not in ("queued", "processing"):
            face.update(confirmed=False, state="analyzing",
                        analysis={"status": "queued", "usable": False, "message": "等待重新分析"})
            save_record(record)
    return jsonify(status="ok")


@app.post("/cancel")
def cancel():
    with lock:
        sid = active_id()
        if sid:
            record = read_record(sid)
            if record["status"] == "collecting":
                record["status"] = "cancelled"
                save_record(record)
    return jsonify(status="ok")


@app.get("/api/tasks")
def tasks():
    with lock:
        records = all_records()[:100]
        return jsonify(active_session=active_id(), records=[{
            "session_id": r["session_id"], "created_at": r["created_at"],
            "status": r["status"], "summary": r["summary"]} for r in records])


@app.get("/api/session/<session_id>")
def session_detail(session_id):
    with lock:
        record = read_record(session_id)
        result = dict(record)
        result["server_time"] = time.time()
        result["is_active"] = active_id() == session_id
        return jsonify(result)


@app.get("/api/task/<capture_id>")
def board_task_detail(capture_id):
    with lock:
        record, face = capture_lookup(capture_id)
        status = "complete" if face["infrared"] else "cancelled" if record["status"] == "cancelled" else "capturing_pairs"
        return jsonify(task_id=capture_id, status=status,
                       pairs=[{"index": f["index"], "view": f["view"], "color": "",
                               "infrared": f["infrared"] if f["index"] == face["index"] else ""}
                              for f in record["faces"]])


@app.get("/images/<session_id>/<filename>")
def image_file(session_id, filename):
    with lock:
        record = read_record(session_id)
        folder = SAVE_ROOT / session_id
        if filename not in record["saved_files"] or (folder / filename).resolve().parent != folder.resolve():
            return "照片不存在", 404
        return send_from_directory(str(folder), filename)


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>红外六面损伤检测</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f5f0;color:#29352c;font:14px "Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:25px;font-weight:500}h2{font-size:18px}p{line-height:1.7}button,select{font:inherit;padding:8px 12px;margin:3px;border:1px solid #becbb7;border-radius:4px;background:white;color:#365637}button{cursor:pointer}button:disabled{opacity:.5;cursor:default}.bar,.summary,.face{background:white;border:1px solid #dce0d5;border-radius:5px;padding:16px;margin:12px 0}.bar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.muted{font-size:12px;color:#75816e;overflow-wrap:anywhere}.notice{color:#9a552d}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.face{margin:0;min-width:0}.pictures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}figure{margin:0;min-width:0}img{width:100%;height:180px;object-fit:contain;background:#eee;cursor:zoom-in}figcaption{font-size:11px;color:#75816e;padding:5px 0}.placeholder{height:180px;display:grid;place-items:center;background:#eee;color:#89947f}.grade{font-size:27px;color:#365c3a}.error{color:#aa4637}.face.active{border-color:#799b6a}.face.confirmed{border-color:#aac298}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #e3e8dc;text-align:left;padding:7px}.rules{display:flex;flex-wrap:wrap;gap:16px;font-size:12px;color:#75816e}#prompt{font-size:20px;color:#3a6638}#countdown{font-size:24px;font-weight:bold;color:#a05e20}
@media(max-width:850px){main{padding:12px}.grid{grid-template-columns:1fr}img,.placeholder{height:180px}}
</style></head><body><main>
<h1>红外六面损伤检测</h1>
<div class="bar"><button id="new">开始新水果</button><button id="cancel">取消当前采集</button><button id="latest">查看当前任务</button><select id="history" aria-label="历史记录"></select><span id="connection">正在连接</span></div>
<p id="notice" class="notice"></p><p id="prompt">点击“开始新水果”，先摆好正面。</p><p id="countdown"></p>
<div class="summary" id="summary">尚未开始采集。</div>
<div class="rules"><span>初始演示规则：0% 未检出损伤；大于0%且小于5% 轻度；5%至小于15% 中度；15%及以上 重度。</span><span>平均值是六个视角损伤占比的算术平均，不是整果真实表面积比例。</span></div>
<p class="muted">顺序：正面、背面、左面、右面、顶部、底部。每面摆好后点拍摄，倒计时5秒，保持静止直到上传完成。检查蓝色水果轮廓、青色过曝和红色损伤，正确再确认本面。有效区域不足50%时要求重拍；其他分割错误也请重拍或补充标注。</p>
<div id="faces" class="grid"></div><p id="folder" class="muted"></p>
</main><script>
const $=id=>document.getElementById(id);let selected=null,current=null,follow=true,busy=false,actionBusy=false,facesKey='',historyKey='',clockOffset=0;
const states={idle:'等待摆放',countdown:'倒计时',capturing:'板端拍摄或上传中',analyzing:'正在分析',review:'等待确认',confirmed:'本面已确认',capture_error:'拍摄失败',analysis_error:'请检查并重拍'};
async function api(url,options={}){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),5000);try{const r=await fetch(url,{...options,cache:'no-store',signal:controller.signal});const d=await r.json();if(!r.ok)throw new Error(d.message||'请求失败');return d;}finally{clearTimeout(timer);}}
async function action(url,data={}){if(actionBusy)return;actionBusy=true;try{const r=await api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(r.session_id){selected=r.session_id;follow=true;}facesKey='';$('notice').textContent='操作已提交。';await refresh();}catch(e){$('notice').textContent=e.message;}finally{actionBusy=false;}}
function image(record,name,label){const f=document.createElement('figure');if(name){const im=document.createElement('img');const url='/images/'+record.session_id+'/'+encodeURIComponent(name);im.src=url;im.alt=label;im.onclick=()=>window.open(url,'_blank','noopener');f.append(im);}else{const p=document.createElement('div');p.className='placeholder';p.textContent='等待照片';f.append(p);}const c=document.createElement('figcaption');c.textContent=label;f.append(c);return f;}
function makeButton(label,disabled,handler){const b=document.createElement('button');b.textContent=label;b.disabled=disabled;b.onclick=handler;return b;}
function render(record){current=record;clockOffset=record.server_time*1000-Date.now();const summary=record.summary;
 $('summary').replaceChildren();const grade=document.createElement('div');grade.className='grade';grade.textContent=record.status==='cancelled'?'本次已取消':summary.grade;$('summary').append(grade);
 const p=document.createElement('p');p.textContent='已确认 '+summary.confirmed_faces+' / 6 面'+(summary.mean_percent===null?' · 六面有效且确认后才计算平均值':' · 六面平均损伤 '+summary.mean_percent+'% · 最大单面损伤 '+summary.max_percent+'%');$('summary').append(p);
 const t=document.createElement('table');const head=document.createElement('tr');for(const label of ['面','损伤占比','有效区域占比','状态']){const th=document.createElement('th');th.textContent=label;head.append(th);}t.append(head);
 for(const f of record.faces){const a=f.analysis||{},tr=document.createElement('tr');for(const value of [f.view,a.usable&&a.damage_percent!==null?a.damage_percent+'%':'—',a.valid_percent===undefined?'—':a.valid_percent+'%',states[f.state]||f.state]){const td=document.createElement('td');td.textContent=value;tr.append(td);}t.append(tr);}$('summary').append(t);
 $('folder').textContent='保存位置：'+record.folder;
 const key=record.session_id+JSON.stringify(record.faces)+record.status+record.is_active;
 if(key!==facesKey){facesKey=key;$('faces').replaceChildren();const pending=record.faces.find(f=>!f.confirmed);const occupied=record.faces.some(f=>['countdown','capturing','analyzing'].includes(f.state));
 for(const f of record.faces){const a=f.analysis||{},section=document.createElement('section');section.className='face'+(f.confirmed?' confirmed':'')+(pending&&pending.index===f.index?' active':'');const h=document.createElement('h2');h.textContent=String(f.index).padStart(2,'0')+' '+f.view+' · '+(states[f.state]||f.state);section.append(h);
 const pics=document.createElement('div');pics.className='pictures';pics.append(image(record,f.infrared,'红外原图'),image(record,a.overlay,'三类分割图'),image(record,a.processed,'有效区域与损伤'));section.append(pics);
 const note=document.createElement('p');note.className='muted';note.textContent=f.error||a.message||'等待拍摄';if(f.error||f.state==='analysis_error')note.className+=' error';section.append(note);
 const disabled=!record.is_active||record.status==='cancelled';const data={session_id:record.session_id,index:f.index};const sequential=Boolean(pending&&pending.index===f.index);
 section.append(makeButton(f.infrared?'重拍本面':'摆好了，拍本面',disabled||occupied||(!sequential&&!f.confirmed&&!f.infrared),()=>action('/prepare',data)));
 if(f.infrared){section.append(makeButton(f.confirmed?'已确认':'确认本面',disabled||!a.usable||!['review','confirmed'].includes(f.state)||f.confirmed,()=>action('/confirm',data)));section.append(makeButton('重新分析',disabled||occupied,()=>action('/reprocess',data)));}
 $('faces').append(section);}}
 updateClock();}
function updateClock(){if(!current)return;const record=current;const f=record.faces.find(f=>['countdown','capturing','analyzing'].includes(f.state))||record.faces.find(f=>!f.confirmed);$('countdown').textContent='';
 if(record.status==='cancelled'){$('prompt').textContent='采集已取消，照片保留。';return;}if(record.status==='complete'){$('prompt').textContent='六面采集完成，可查看等级或开始下一个水果。';return;}
 if(!f)return;$('prompt').textContent=(current.is_active?'当前':'历史任务')+'：'+f.view+' · '+(states[f.state]||f.state);
 if(f.state==='countdown'){const seconds=Math.max(0,Math.ceil((f.due_at*1000-Date.now()-clockOffset)/1000));$('countdown').textContent=seconds?seconds+' 秒后拍摄，请保持静止':'倒计时结束，等待板端拍摄';}
 else if(f.state==='idle')$('prompt').textContent='请摆好 '+f.view+'，点击该面“摆好了，拍本面”。';else if(f.state==='review')$('prompt').textContent='检查 '+f.view+' 的分割结果，正确后点击“确认本面”，再翻到下一面。';}
async function refresh(){if(busy)return;busy=true;try{const data=await api('/api/tasks');if(follow){selected=data.active_session||(data.records[0]||{}).session_id;}
 const hk=JSON.stringify(data.records)+selected;if(hk!==historyKey){historyKey=hk;$('history').replaceChildren();for(const r of data.records){const o=document.createElement('option');o.value=r.session_id;o.textContent=r.created_at.replace('T',' ').slice(0,19)+' · '+r.summary.confirmed_faces+'/6 · '+r.summary.grade;o.selected=r.session_id===selected;$('history').append(o);}}
 if(selected){const requested=selected;const r=await api('/api/session/'+requested);if(selected===requested)render(r);}$('connection').textContent='接收服务已连接';}catch(e){$('connection').textContent='连接中断，正在重试';}finally{busy=false;}}
$('new').onclick=()=>action('/new-six');$('cancel').onclick=()=>action('/cancel');$('latest').onclick=()=>{follow=true;refresh();};$('history').onchange=()=>{follow=false;selected=$('history').value;facesKey='';refresh();};refresh();setInterval(refresh,1000);setInterval(updateClock,200);
</script></body></html>'''


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    threading.Thread(target=processing_loop, daemon=True, name="ir-analysis").start()
    print("红外六面检测：http://127.0.0.1:5000", flush=True)
    print("模型：", MODEL_PATH, "存在：", MODEL_PATH.is_file(), flush=True)
    print("结果目录：", SAVE_ROOT, flush=True)
    print("人工翻面，每面拍后检查并确认；未完成六面不输出等级。", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
