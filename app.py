import os, json, csv, time, threading, wave
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
DATA_DIR.mkdir(exist_ok=True)
FACES_DIR.mkdir(exist_ok=True)
ROSTER_FILE = DATA_DIR / "students.json"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"
EVENTS_FILE = DATA_DIR / "session_events.csv"
MODEL_FILE = DATA_DIR / "face_model.yml"
LABELS_FILE = DATA_DIR / "labels.json"

app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", "change-this-local-demo-secret")

state_lock = threading.Lock()
camera = None
camera_lock = threading.Lock()
latest_jpeg = None
running = False
session_started = None
session_id = None
current_session_attendance = set()
metrics = {"frames": 0, "faces": 0, "motion": 0.0, "face_ratio": 0.0, "audio_level": 0.0}
recent_face_counts = deque(maxlen=180)
prev_gray = None
last_marked = {}
last_event_second = 0
unknown_count = 0

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("OpenCV face detector could not be loaded.")

def load_roster():
    if ROSTER_FILE.exists():
        try:
            return json.loads(ROSTER_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def save_roster(roster):
    ROSTER_FILE.write_text(json.dumps(roster, indent=2), encoding="utf-8")

def init_csv(path, headers):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

init_csv(ATTENDANCE_FILE, ["timestamp", "session_id", "student_id", "student_name", "method"])
init_csv(EVENTS_FILE, ["timestamp", "session_id", "face_count", "motion_score", "face_ratio", "audio_level", "note"])

def load_model():
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        return None, {}
    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(str(MODEL_FILE))
        labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        return model, {int(k): v for k, v in labels.items()}
    except Exception:
        return None, {}

recognizer, label_to_student = load_model()

def train_model():
    global recognizer, label_to_student
    if not hasattr(cv2, "face"):
        return False, "Install opencv-contrib-python (included in requirements.txt), then restart."
    images, labels = [], []
    roster = load_roster()
    id_to_label = {}
    label = 0
    for student in roster:
        folder = FACES_DIR / str(student["id"])
        if not folder.exists():
            continue
        count_before = len(images)
        for p in folder.glob("*.jpg"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(img)
                labels.append(label)
        if len(images) > count_before:
            id_to_label[label] = student
            label += 1
    if not images:
        return False, "No face samples found. Add a student and capture face samples first."
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, np.array(labels, dtype=np.int32))
    model.write(str(MODEL_FILE))
    LABELS_FILE.write_text(json.dumps({str(k): v for k, v in id_to_label.items()}, indent=2), encoding="utf-8")
    recognizer, label_to_student = model, id_to_label
    return True, f"Model trained with {len(images)} face samples for {len(id_to_label)} student(s)."

def mark_attendance(student, method="face-recognition"):
    global current_session_attendance
    if not session_id:
        return
    sid = str(student["id"])
    if sid in current_session_attendance:
        return
    stamp = datetime.now().isoformat(timespec="seconds")
    with ATTENDANCE_FILE.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([stamp, session_id, sid, student["name"], method])
    current_session_attendance.add(sid)
    last_marked[sid] = stamp

def log_event(face_count, motion, face_ratio, audio_level=0.0, note=""):
    global last_event_second
    now = int(time.time())
    if now == last_event_second:
        return
    last_event_second = now
    with EVENTS_FILE.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), session_id or "", face_count,
                                round(motion, 4), round(face_ratio, 4), round(audio_level, 4), note])

def identify_face(face_gray):
    if recognizer is None:
        return None, None
    try:
        label, confidence = recognizer.predict(cv2.resize(face_gray, (200, 200)))
        # LBPH confidence is a distance; lower is better. Threshold is a demo default.
        if confidence < 75 and label in label_to_student:
            return label_to_student[label], float(confidence)
    except Exception:
        pass
    return None, None

def process_frame(frame):
    global prev_gray, metrics, unknown_count
    frame = cv2.resize(frame, (960, 540)) if frame.shape[1] > 960 else frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(45, 45))
    motion = 0.0
    if prev_gray is not None and prev_gray.shape == gray.shape:
        diff = cv2.absdiff(prev_gray, gray)
        motion = float(np.mean(diff) / 255.0)
    prev_gray = gray
    h, w = gray.shape
    face_area = 0
    identified = []
    for (x, y, fw, fh) in faces:
        face_area += fw * fh
        roi = gray[y:y+fh, x:x+fw]
        student, confidence = identify_face(roi)
        if student:
            color = (30, 190, 80)
            label = f'{student["name"]} ({confidence:.0f})'
            mark_attendance(student)
            identified.append(student["id"])
        else:
            color = (30, 165, 240)
            label = "Unknown face"
            unknown_count += 1
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
        cv2.putText(frame, label, (x, max(24, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    count = len(faces)
    ratio = min(1.0, face_area / max(1, w*h))
    with state_lock:
        metrics.update({"frames": metrics["frames"]+1, "faces": count, "motion": motion, "face_ratio": ratio})
    recent_face_counts.append(count)
    if running:
        log_event(count, motion, ratio)
    cv2.putText(frame, f"Faces: {count} | Recognized this session: {len(current_session_attendance)}",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame

def camera_worker():
    global camera, latest_jpeg, running
    while running:
        with camera_lock:
            if camera is None:
                camera = cv2.VideoCapture(0)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap = camera
        if cap is None or not cap.isOpened():
            time.sleep(0.5)
            continue
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        try:
            processed = process_frame(frame)
            ok, buf = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                with state_lock:
                    latest_jpeg = buf.tobytes()
        except Exception as exc:
            print("Frame processing error:", exc)
        time.sleep(0.03)
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

def mjpeg_stream():
    while True:
        with state_lock:
            frame = latest_jpeg
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)

def csv_rows(path):
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

@app.route("/")
def index():
    roster = load_roster()
    records = csv_rows(ATTENDANCE_FILE)
    session_records = [r for r in records if session_id and r.get("session_id") == session_id]
    events = csv_rows(EVENTS_FILE)
    current_events = [e for e in events if session_id and e.get("session_id") == session_id]
    latest = current_events[-20:]
    avg_faces = round(sum(float(e.get("face_count") or 0) for e in current_events) / max(1, len(current_events)), 1)
    avg_motion = round(sum(float(e.get("motion_score") or 0) for e in current_events) / max(1, len(current_events)), 3)
    suggestions = make_suggestions(current_events, len(session_records), len(roster))
    return render_template("index.html", roster=roster, attendance=session_records,
        session_id=session_id, session_started=session_started, running=running,
        total_students=len(roster), attendance_count=len(session_records), avg_faces=avg_faces,
        avg_motion=avg_motion, suggestions=suggestions, events=latest,
        model_ready=recognizer is not None)

def make_suggestions(events, present, total):
    if not events:
        return ["Start a session to collect classroom signals. Suggestions are based on simple observable indicators, not a definitive measure of attention."]
    faces = [float(e.get("face_count") or 0) for e in events]
    motions = [float(e.get("motion_score") or 0) for e in events]
    notes = []
    if total and present / total < 0.5:
        notes.append("Attendance is below half of the registered roster; verify the camera view and roster before drawing conclusions.")
    if len(faces) >= 5 and max(faces) - min(faces) >= 2:
        notes.append("Detected face counts varied during the sampled interval. Check camera framing and whether learners are entering or leaving.")
    if motions and sum(motions)/len(motions) < 0.015:
        notes.append("Low image motion was observed. Consider asking a quick question or running a short interactive check-in; low motion does not necessarily mean low engagement.")
    elif motions:
        notes.append("Movement varied in the sampled video. A short poll or pair discussion may help sustain participation.")
    if len(events) >= 30:
        notes.append("A session has accumulated enough samples for a basic trend review. Compare these signals with actual participation and instructor observations.")
    if not notes:
        notes.append("Continue teaching and use direct questions or a short quiz to validate whether learners are following.")
    notes.append("Privacy reminder: get consent, provide a non-camera alternative, and avoid using automated scores for grading or disciplinary decisions.")
    return notes

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/metrics")
def api_metrics():
    with state_lock:
        return jsonify({**metrics, "attendance_count": len(current_session_attendance), "running": running,
                        "session_id": session_id, "model_ready": recognizer is not None})

@app.route("/students/add", methods=["POST"])
def add_student():
    name = request.form.get("name", "").strip()
    sid = request.form.get("student_id", "").strip()
    if not name or not sid:
        flash("Enter both a student ID and name.", "error")
        return redirect(url_for("index"))
    roster = load_roster()
    if any(str(s["id"]).lower() == sid.lower() for s in roster):
        flash("That student ID already exists.", "error")
        return redirect(url_for("index"))
    roster.append({"id": sid, "name": name})
    save_roster(roster)
    (FACES_DIR / sid).mkdir(exist_ok=True)
    flash(f"Added {name}. Capture face samples, then train the model.", "success")
    return redirect(url_for("index"))

@app.route("/students/delete/<sid>", methods=["POST"])
def delete_student(sid):
    roster = [s for s in load_roster() if str(s["id"]) != sid]
    save_roster(roster)
    import shutil
    shutil.rmtree(FACES_DIR / sid, ignore_errors=True)
    flash("Student and stored samples deleted.", "success")
    return redirect(url_for("index"))

@app.route("/students/capture/<sid>", methods=["POST"])
def capture_samples(sid):
    roster = load_roster()
    student = next((s for s in roster if str(s["id"]) == sid), None)
    if not student:
        flash("Student not found.", "error")
        return redirect(url_for("index"))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        flash("Could not open webcam. Close other apps using it and retry.", "error")
        return redirect(url_for("index"))
    folder = FACES_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)
    saved, tries = 0, 0
    while saved < 20 and tries < 180:
        tries += 1
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(70,70))
        if len(faces):
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            face = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            cv2.imwrite(str(folder / f"sample_{int(time.time()*1000)}_{saved}.jpg"), face)
            saved += 1
            time.sleep(0.12)
    cap.release()
    flash(f"Captured {saved} face samples for {student['name']}. Keep the face centered and well lit.", "success" if saved else "error")
    return redirect(url_for("index"))

@app.route("/model/train", methods=["POST"])
def train_route():
    ok, message = train_model()
    flash(message, "success" if ok else "error")
    return redirect(url_for("index"))

@app.route("/session/start", methods=["POST"])
def start_session():
    global running, session_started, session_id, current_session_attendance, metrics
    if running:
        flash("A session is already running.", "error")
        return redirect(url_for("index"))
    session_id = datetime.now().strftime("S%Y%m%d_%H%M%S")
    session_started = datetime.now().isoformat(timespec="seconds")
    current_session_attendance = set()
    metrics = {"frames": 0, "faces": 0, "motion": 0.0, "face_ratio": 0.0, "audio_level": 0.0}
    running = True
    threading.Thread(target=camera_worker, daemon=True).start()
    flash("Session started. Allow camera permission if prompted.", "success")
    return redirect(url_for("index"))

@app.route("/session/stop", methods=["POST"])
def stop_session():
    global running
    running = False
    flash("Session stopped. Attendance and session events have been saved locally.", "success")
    return redirect(url_for("index"))

@app.route("/session/export")
def export_session():
    import io
    from flask import send_file
    if not session_id:
        flash("Start a session first.", "error")
        return redirect(url_for("index"))
    rows = [r for r in csv_rows(ATTENDANCE_FILE) if r.get("session_id") == session_id]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp","session_id","student_id","student_name","method"])
    writer.writeheader()
    writer.writerows(rows)
    from io import BytesIO
    return send_file(BytesIO(output.getvalue().encode("utf-8")), mimetype="text/csv",
        as_attachment=True, download_name=f"attendance_{session_id}.csv")

@app.route("/video/analyze", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    if not file or not file.filename:
        flash("Choose a video file first.", "error")
        return redirect(url_for("index"))
    temp = DATA_DIR / "uploaded_video"
    file.save(str(temp))
    cap = cv2.VideoCapture(str(temp))
    if not cap.isOpened():
        temp.unlink(missing_ok=True)
        flash("Could not read this video format.", "error")
        return redirect(url_for("index"))
    samples = 0
    total_faces = 0
    motions = []
    prev = None
    while samples < 300:
        ok, frame = cap.read()
        if not ok: break
        if samples % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.15, 5, minSize=(45,45))
            total_faces += len(faces)
            if prev is not None and prev.shape == gray.shape:
                motions.append(float(np.mean(cv2.absdiff(prev, gray))/255.0))
            prev = gray
        samples += 1
    cap.release()
    temp.unlink(missing_ok=True)
    analyzed = max(1, len(motions))
    report = {"sampled_frames": len(motions), "average_faces_detected": round(total_faces/max(1,len(motions)),2),
              "average_motion_score": round(sum(motions)/analyzed,4) if motions else 0,
              "note": "Face counts and motion are coarse visual indicators, not individual engagement or attention scores."}
    return render_template("video_report.html", report=report)

@app.route("/audio/analyze", methods=["POST"])
def analyze_audio():
    file = request.files.get("audio")
    if not file or not file.filename:
        flash("Choose a WAV audio file first.", "error")
        return redirect(url_for("index"))
    temp = DATA_DIR / "uploaded_audio.wav"
    file.save(str(temp))
    try:
        with wave.open(str(temp), "rb") as wf:
            channels, width, rate, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
            raw = wf.readframes(frames)
        if width not in (1, 2, 4):
            raise ValueError("Unsupported WAV sample width")
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if width == 1:
            audio = audio - 128
        if channels > 1:
            audio = audio[:(len(audio)//channels)*channels].reshape(-1, channels).mean(axis=1)
        duration = len(audio) / max(1, rate)
        scale = float(2 ** (8*width-1))
        rms = float(np.sqrt(np.mean((audio / scale) ** 2))) if len(audio) else 0.0
        peak = float(np.max(np.abs(audio / scale))) if len(audio) else 0.0
        # Windowed RMS provides a coarse measure of how audio activity varies over time.
        window = max(1, int(rate * 1.0))
        levels = []
        for i in range(0, len(audio), window):
            chunk = audio[i:i+window]
            if len(chunk): levels.append(float(np.sqrt(np.mean((chunk / scale) ** 2))))
        active = sum(1 for level in levels if level > 0.02)
        report = {"duration": round(duration, 1), "sample_rate": rate, "rms": round(rms, 4),
                  "peak": round(peak, 4), "active_seconds": active, "sampled_seconds": len(levels),
                  "note": "Audio level is only a signal-energy proxy. It cannot identify speakers, determine who is speaking, or measure engagement."}
    except Exception as exc:
        temp.unlink(missing_ok=True)
        flash("Could not analyze this audio. Please use a standard PCM WAV file. " + str(exc), "error")
        return redirect(url_for("index"))
    temp.unlink(missing_ok=True)
    return render_template("audio_report.html", report=report)

@app.route("/health")
def health():
    return jsonify({"status":"ok", "camera_running":running, "face_model_ready":recognizer is not None})

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
