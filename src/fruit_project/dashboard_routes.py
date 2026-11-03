from pathlib import Path
import json
import re

from flask import jsonify


STATUS_NAMES = {
    "capturing_color": "正在拍摄六面",
    "waiting_infrared_confirmation": "等待按Enter确认",
    "waiting_infrared": "等待红外入料",
    "infrared_capturing": "正在拍摄或上传红外",
    "complete": "全部完成",
    "cancelled": "已取消",
    "error": "发生异常",
}


def register_dashboard(app, capture_root):
    root = Path(capture_root)

    def read_tasks():
        tasks = []

        for path in root.glob("*/record.json"):
            try:
                record = json.loads(
                    path.read_text(encoding="utf-8")
                )

                task_id = record.get("task_id", "")

                if not re.fullmatch(
                    r"[a-f0-9]{32}", task_id
                ):
                    continue

                tasks.append((path, record))

            except (OSError, ValueError, TypeError):
                continue

        tasks.sort(
            key=lambda item: str(
                item[1].get("created_at", "")
            ),
            reverse=True
        )
        return tasks

    def summarize(path, record):
        folder = path.parent
        task_id = record["task_id"]

        images = []

        for photo in record.get("photos", []):
            filename = photo.get("filename", "")

            if (
                filename
                and Path(filename).name == filename
                and (folder / filename).is_file()
            ):
                images.append({
                    "filename": filename,
                    "label": photo.get("view", "彩色照片"),
                    "kind": "color",
                })

        infrared = record.get(
            "infrared_filename", ""
        )

        has_infrared = (
            bool(infrared)
            and Path(infrared).name == infrared
            and (folder / infrared).is_file()
        )

        if has_infrared:
            images.append({
                "filename": infrared,
                "label": "红外照片",
                "kind": "infrared",
            })

        state = record.get("status", "unknown")
        color_count = sum(
            item["kind"] == "color"
            for item in images
        )

        return {
            "task_id": task_id,
            "species": record.get(
                "species_cn",
                record.get("species", "未识别")
            ),
            "confidence": record.get("confidence"),
            "created_at": record.get("created_at", ""),
            "status": state,
            "status_name": STATUS_NAMES.get(
                state, state
            ),
            "color_count": color_count,
            "infrared_received": has_infrared,
            "images": images,
            "folder": str(folder),
            "error": record.get("error", ""),
        }

    @app.get("/dashboard/tasks")
    def dashboard_tasks():
        tasks = [
            summarize(path, record)
            for path, record in read_tasks()[:100]
        ]

        # 列表不需要携带图片明细
        for task in tasks:
            task.pop("images", None)

        response = jsonify(tasks)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/dashboard/task/<task_id>")
    def dashboard_task(task_id):
        for path, record in read_tasks():
            if record["task_id"] == task_id:
                response = jsonify(
                    summarize(path, record)
                )
                response.headers["Cache-Control"] = (
                    "no-store"
                )
                return response

        return jsonify(
            message="未找到该任务"
        ), 404

    @app.get("/dashboard")
    def dashboard_page():
        return PAGE


PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>水果采集管理</title>
<style>
*{box-sizing:border-box}
body{
 margin:0;background:#f3f5f8;color:#233047;
 font-family:"Microsoft YaHei",sans-serif
}
header{
 background:#fff;border-bottom:1px solid #e3e8ef;
 padding:23px 28px
}
header h1{margin:0;font-size:24px}
header p{margin:8px 0 0;color:#788395;font-size:13px}
main{max-width:1400px;margin:auto;padding:24px}
.layout{display:grid;grid-template-columns:280px 1fr;gap:20px}
.card{
 background:#fff;border:1px solid #e5e9f0;
 border-radius:14px;padding:20px;margin-bottom:18px
}
h2{font-size:18px;margin:0 0 16px}
button{
 font-family:inherit;cursor:pointer;border:0;
 border-radius:8px;padding:9px 13px;
 background:#edf3ff;color:#315ba2
}
.task{
 display:block;width:100%;text-align:left;
 background:#f7f9fc;color:#334155;
 margin-bottom:10px;border:1px solid transparent
}
.task.active{background:#edf3ff;border-color:#8aaee9}
.task strong{display:block;font-size:15px}
.task small{
 display:block;color:#7c8797;font-size:12px;
 margin-top:6px;line-height:1.5
}
.history{max-height:680px;overflow:auto}
.muted{color:#7c8797;font-size:13px}
.topline{
 display:flex;justify-content:space-between;
 align-items:center;gap:12px;flex-wrap:wrap
}
.badge{
 border-radius:20px;background:#edf3ff;
 color:#3e68ac;padding:7px 13px;font-size:13px
}
.badge.done{background:#e8f5ee;color:#278052}
.stats{
 display:grid;grid-template-columns:repeat(3,1fr);
 gap:12px;margin-top:18px
}
.stat{background:#f7f9fc;padding:14px;border-radius:10px}
.stat small{display:block;color:#7c8797;margin-bottom:8px}
.stat strong{font-size:21px}
.steps{
 display:grid;grid-template-columns:repeat(4,1fr);
 gap:10px;margin-top:18px
}
.step{
 padding:12px 8px;border-radius:9px;
 background:#f0f2f6;color:#8a94a3;
 text-align:center;font-size:13px
}
.step.done{background:#e8f5ee;color:#278052}
.step.current{background:#edf3ff;color:#3765ac}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.photo{
 border:1px solid #e5e9f0;border-radius:10px;
 overflow:hidden;background:#f7f9fc
}
.photo img{
 display:block;width:100%;height:180px;
 object-fit:contain;background:#eef1f5
}
.photo p{margin:0;padding:10px 12px;font-size:13px}
.photo a{display:block}
.empty{
 padding:35px 15px;color:#8a94a3;
 text-align:center;background:#f7f9fc;border-radius:10px
}
.ir img{width:100%;max-height:480px;object-fit:contain}
.path{overflow-wrap:anywhere;line-height:1.7}
#error{color:#ba4850}
@media(max-width:900px){
 .layout{grid-template-columns:1fr}
 .history{max-height:220px}
}
@media(max-width:600px){
 main{padding:12px}
 .grid{grid-template-columns:repeat(2,1fr)}
 .stats{gap:7px}
 .stat{padding:10px}
 .stat strong{font-size:17px}
 .steps{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>
<header>
 <h1>水果采集管理</h1>
 <p>种类识别 · 六面拍摄 · 人工确认 · 红外入料拍摄</p>
</header>

<main>
 <div class="layout">
  <aside class="card">
   <div class="topline">
    <h2 style="margin:0">历史任务</h2>
    <button id="latest">跟随最新</button>
   </div>
   <p class="muted" id="mode">当前跟随最新任务</p>
   <div id="history" class="history"></div>
  </aside>

  <section>
   <div class="card">
    <div class="topline">
     <h2 id="fruit" style="margin:0">等待拍摄任务</h2>
     <span id="status" class="badge">暂无任务</span>
    </div>

    <div class="stats">
     <div class="stat">
      <small>种类识别置信度</small>
      <strong id="confidence">—</strong>
     </div>
     <div class="stat">
      <small>彩色照片</small>
      <strong id="colorCount">0 / 6</strong>
     </div>
     <div class="stat">
      <small>红外照片</small>
      <strong id="irCount">未收到</strong>
     </div>
    </div>

    <div class="steps">
     <div id="step0" class="step">1 种类已识别</div>
     <div id="step1" class="step">2 六面拍摄</div>
     <div id="step2" class="step">3 人工确认</div>
     <div id="step3" class="step">4 红外拍摄</div>
    </div>

    <p id="message" class="muted">启动电脑拍摄程序后自动显示。</p>
    <p id="error"></p>
   </div>

   <div class="card">
    <h2>六面彩色照片</h2>
    <div id="colors" class="grid"></div>
   </div>

   <div class="card">
    <h2>红外照片</h2>
    <div id="infrared"></div>
   </div>

   <div class="card">
    <h2>任务信息</h2>
    <p id="taskId" class="muted path">—</p>
    <p id="created" class="muted"></p>
    <p id="folder" class="muted path"></p>
    <p class="muted">点击照片可以打开原图。未接入损伤判断。</p>
   </div>
  </section>
 </div>
</main>

<script>
let followLatest = true;
let selectedId = null;
let previousPhotos = "";
let previousHistory = "";
let busy = false;
let doneTask = null;

const text = (id, value) => {
 document.getElementById(id).textContent = value;
};

function photoUrl(taskId, filename) {
 return "/photos/" + encodeURIComponent(taskId)
   + "/" + encodeURIComponent(filename);
}

function makePhoto(taskId, item, infrared=false) {
 const box = document.createElement("div");
 box.className = infrared ? "photo ir" : "photo";

 const link = document.createElement("a");
 link.href = photoUrl(taskId, item.filename);
 link.target = "_blank";
 link.rel = "noopener";

 const image = document.createElement("img");
 image.src = link.href;
 image.alt = item.label;
 link.appendChild(image);

 const label = document.createElement("p");
 label.textContent = item.label;

 box.append(link, label);
 return box;
}

function showSteps(task) {
 const s = task.status;
 const confirmed = [
   "waiting_infrared", "infrared_capturing", "complete"
 ].includes(s);

 const classes = [
   "done",
   task.color_count >= 6 ? "done" :
      (s === "capturing_color" ? "current" : ""),
   confirmed ? "done" :
      (s === "waiting_infrared_confirmation" ? "current" : ""),
   task.infrared_received ? "done" :
      (confirmed ? "current" : "")
 ];

 classes.forEach((value, index) => {
   document.getElementById("step" + index).className =
     "step " + value;
 });
}

function showDetail(task) {
 text("fruit", task.species);
 text("status", task.status_name);
 text("colorCount", task.color_count + " / 6");
 text("irCount", task.infrared_received ? "已保存" : "未收到");

 const confidence = Number(task.confidence);
 text("confidence",
   task.confidence != null && Number.isFinite(confidence)
     ? (confidence * 100).toFixed(1) + "%" : "—"
 );

 document.getElementById("status").className =
   "badge" + (task.status === "complete" ? " done" : "");

 const messages = {
   capturing_color: "按摄像头窗口提示转动同一个水果。",
   waiting_infrared_confirmation:
     "请点击摄像头窗口，按 Enter 确认进入红外阶段。",
   waiting_infrared:
     "先保持对射光路无遮挡，再将同一个水果入料遮挡。",
   infrared_capturing:
     "等待延时拍摄和上传，请保持板子供电及网络连接。",
   complete: "本次采集完成，可以准备下一个水果。",
   cancelled: "本次已取消，已保存的照片仍保留。",
   error: "采集出现异常，请查看拍摄程序或板端日志。"
 };

 text("message", messages[task.status] || task.status_name);
 text("error", task.error || "");
 text("taskId", "任务编号：" + task.task_id);
 text("created", "创建时间：" + task.created_at.replace("T", " "));
 text("folder", "保存位置：" + task.folder);
 showSteps(task);

 const signature = task.task_id + JSON.stringify(task.images);

 if (signature !== previousPhotos) {
   const colors = document.getElementById("colors");
   const infrared = document.getElementById("infrared");

   colors.replaceChildren();
   infrared.replaceChildren();

   const colorImages = task.images.filter(i => i.kind === "color");
   const irImages = task.images.filter(i => i.kind === "infrared");

   if (!colorImages.length) {
     const empty = document.createElement("div");
     empty.className = "empty";
     empty.textContent = "等待彩色照片";
     colors.appendChild(empty);
   }

   colorImages.forEach(item => {
     colors.appendChild(makePhoto(task.task_id, item));
   });

   if (!irImages.length) {
     const empty = document.createElement("div");
     empty.className = "empty";
     empty.textContent = "等待红外照片上传";
     infrared.appendChild(empty);
   }

   irImages.forEach(item => {
     infrared.appendChild(makePhoto(task.task_id, item, true));
   });

   previousPhotos = signature;
 }

 if (task.status === "complete" && doneTask !== task.task_id) {
   doneTask = task.task_id;
   document.title = "采集完成 · " + task.species;
 } else if (task.status !== "complete") {
   document.title = task.species + " · " + task.status_name;
 }
}

async function refresh() {
 if (busy) return;
 busy = true;

 try {
   const response = await fetch("/dashboard/tasks", {cache:"no-store"});
   if (!response.ok) throw new Error("读取任务列表失败");

   const tasks = await response.json();

   if (followLatest && tasks.length) {
     selectedId = tasks[0].task_id;
   }

   const historySignature =
     JSON.stringify(tasks) + selectedId;

   if (historySignature !== previousHistory) {
     const history = document.getElementById("history");
     history.replaceChildren();

     if (!tasks.length) {
       const empty = document.createElement("p");
       empty.className = "muted";
       empty.textContent = "暂无拍摄记录";
       history.appendChild(empty);
     }

     tasks.forEach(task => {
       const button = document.createElement("button");
       button.className =
         "task" + (task.task_id === selectedId ? " active" : "");

       const name = document.createElement("strong");
       name.textContent = task.species;

       const detail = document.createElement("small");
       detail.textContent =
         task.created_at.replace("T", " ")
         + "\n" + task.status_name
         + " · 彩色 " + task.color_count + "/6"
         + " · 红外 " + (task.infrared_received ? "1/1" : "0/1");

       button.append(name, detail);
       button.onclick = () => {
         followLatest = false;
         selectedId = task.task_id;
         previousHistory = "";
         text("mode", "正在查看历史任务");
         refresh();
       };

       history.appendChild(button);
     });

     previousHistory = historySignature;
   }

   if (selectedId) {
     const detailResponse = await fetch(
       "/dashboard/task/" + encodeURIComponent(selectedId),
       {cache:"no-store"}
     );
     if (!detailResponse.ok) throw new Error("读取任务详情失败");

     showDetail(await detailResponse.json());
   }

 } catch (error) {
   text("error", "连接异常：" + error.message + "，正在重试。");
 } finally {
   busy = false;
 }
}

document.getElementById("latest").onclick = () => {
 followLatest = true;
 previousHistory = "";
 text("mode", "当前跟随最新任务");
 refresh();
};

setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>
"""