from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from pathlib import Path
from datetime import datetime
import hashlib
import io
import os
import re
import threading
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024
SAVE_DIR = Path(__file__).resolve().parent / 'received'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
latest = {}
count = 0

@app.get('/health')
def health():
    return jsonify(status='ok')

@app.get('/')
def index():
    return '''<!doctype html><meta charset="utf-8"><title>照片接收</title>
<body style="font-family:sans-serif;max-width:900px;margin:30px auto">
<h2>板子照片接收</h2><p id="status">等待照片……</p><img id="photo" style="max-width:100%">
<script>
let previous='';
async function refresh(){try{
const r=await fetch('/latest',{cache:'no-store'});const d=await r.json();
if(d.filename){document.getElementById('status').textContent='任务：'+d.task_id+' ｜接收时间：'+d.received_at+' ｜本次启动新接收：'+d.count+' 张';
if(previous!==d.filename){document.getElementById('photo').src='/photos/'+encodeURIComponent(d.filename);previous=d.filename;}}
}catch(e){document.getElementById('status').textContent='接收服务连接中断';}}
setInterval(refresh,1000);refresh();</script></body>'''

@app.get('/latest')
def get_latest():
    with lock:
        return jsonify(latest)

@app.get('/photos/<filename>')
def photo(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.post('/upload')
def upload():
    global latest, count
    task_id = request.form.get('task_id', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', task_id):
        return jsonify(status='error', message='Invalid task_id'), 400
    uploaded = request.files.get('image')
    if uploaded is None:
        return jsonify(status='error', message='Missing image'), 400
    data = uploaded.read()
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != 'JPEG':
                raise ValueError('Expected JPEG')
            img.verify()
    except Exception:
        return jsonify(status='error', message='Invalid JPEG'), 400
    filename = task_id + '.jpg'
    destination = SAVE_DIR / filename
    digest = hashlib.sha256(data).digest()
    with lock:
        duplicate = destination.exists()
        if duplicate:
            if hashlib.sha256(destination.read_bytes()).digest() != digest:
                return jsonify(status='error', message='Task ID conflict'), 409
        else:
            temporary = SAVE_DIR / (uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('xb') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            count += 1
        latest = dict(task_id=task_id, filename=filename, count=count,
                      received_at=datetime.now().astimezone().isoformat(timespec='seconds'))
        print('照片已保存：', destination, '重复确认：', duplicate, flush=True)
    return jsonify(status='ok', task_id=task_id, duplicate=duplicate, filename=filename)

if __name__ == '__main__':
    print('照片保存位置：', SAVE_DIR)
    print('浏览器打开：http://127.0.0.1:5000/')
    app.run(host='0.0.0.0', port=5000, debug=False)
