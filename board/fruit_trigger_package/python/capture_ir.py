# Run from a Jupyter cell in the directory containing fruit_trigger.bit/.hwh.
from pynq import Overlay
from pathlib import Path
from datetime import datetime
import cv2
import requests
import json
import os
import time
import uuid

PC_ADDRESS = 'http://192.168.2.10:5000'
BLOCKED_LEVEL = 0  # Set from test_ir.py BEFORE enabling.
SAVE_DIR = Path.home() / 'fruit_project' / 'photos'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def stamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')

def record_save(path, data):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def capture(path):
    cam = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    frame = None
    try:
        if not cam.isOpened():
            raise RuntimeError('Cannot open /dev/video0')
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            ok, image = cam.read()
            if ok and image is not None:
                frame = image
    finally:
        cam.release()
    if frame is None:
        raise RuntimeError('No camera frame')
    ok, data = cv2.imencode('.jpg', frame)
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    with path.open('xb') as f:
        f.write(data.tobytes())
        f.flush()
        os.fsync(f.fileno())
    return list(frame.shape)

def send(path, task_id):
    with path.open('rb') as f:
        r = session.post(PC_ADDRESS + '/upload', data={'task_id': task_id},
                         files={'image': (path.name, f, 'image/jpeg')}, timeout=(5, 20))
    r.raise_for_status()
    answer = r.json()
    if answer.get('status') != 'ok' or answer.get('task_id') != task_id:
        raise RuntimeError('Receiver returned an unexpected acknowledgement')

def capture_and_send(event_count):
    task_id = uuid.uuid4().hex
    path = SAVE_DIR / (task_id + '.jpg')
    recpath = SAVE_DIR / (task_id + '.json')
    rec = dict(task_id=task_id, pl_event_count=event_count, created_at=stamp(),
               filename=path.name, status='capturing')
    record_save(recpath, rec)
    try:
        rec['shape'] = capture(path)
    except Exception as e:
        rec.update(status='capture_failed', error=str(e))
        record_save(recpath, rec)
        print('Capture failed:', e, flush=True)
        return
    rec.update(status='pending_upload', captured_at=stamp())
    record_save(recpath, rec)
    print('SD saved:', path, flush=True)
    for attempt in range(1, 4):
        rec['attempts'] = attempt
        try:
            send(path, task_id)
        except Exception as e:
            rec['error'] = str(e)
            record_save(recpath, rec)
            print('Upload failed:', attempt, str(e), flush=True)
            if attempt < 3:
                time.sleep(1)
        else:
            rec.update(status='uploaded', uploaded_at=stamp())
            rec.pop('error', None)
            record_save(recpath, rec)
            print('PC saved:', task_id, flush=True)
            return
    print('Photo retained on SD for later retry.', flush=True)

def retry_pending():
    for recpath in sorted(SAVE_DIR.glob('*.json')):
        try:
            rec = json.loads(recpath.read_text(encoding='utf-8'))
            if rec.get('status') != 'pending_upload':
                continue
            send(SAVE_DIR / rec['filename'], rec['task_id'])
            rec.update(status='uploaded', uploaded_at=stamp())
            rec.pop('error', None)
            record_save(recpath, rec)
            print('Retry uploaded:', rec['task_id'], flush=True)
        except Exception as e:
            print('Retry failed:', recpath.name, e, flush=True)

session = requests.Session()
session.trust_env = False
gpio = None
try:
    response = session.get(PC_ADDRESS + '/health', timeout=(3, 5))
    response.raise_for_status()
    if response.json().get('status') != 'ok':
        raise RuntimeError('Receiver health response invalid')
    ol = Overlay('fruit_trigger.bit')
    gpio = ol.axi_gpio_0
    gpio.write(4, 0xFFFFFFFF)
    gpio.write(12, 0)
    gpio.write(8, 0)
    retry_pending()  # Retry SD records before accepting new objects.
    ctrl = 1 | (BLOCKED_LEVEL << 3)
    gpio.write(8, ctrl)
    print('Ready: clear beam first, then block. Ctrl+C / notebook stop ends.', flush=True)
    while True:
        s = gpio.read(0)
        if s & 4:
            gpio.write(8, ctrl | 4 | 2)
            deadline = time.monotonic() + 1
            while gpio.read(0) & 4:
                if time.monotonic() > deadline:
                    raise RuntimeError('PL acknowledge timeout')
                time.sleep(.001)
            gpio.write(8, ctrl | 4)
            try:
                capture_and_send(s >> 8)
            finally:
                gpio.write(8, ctrl)
            print('Waiting for beam clear and next object.', flush=True)
        time.sleep(.01)
except KeyboardInterrupt:
    print('Stopped.')
finally:
    if gpio is not None:
        gpio.write(8, 0)
    session.close()
