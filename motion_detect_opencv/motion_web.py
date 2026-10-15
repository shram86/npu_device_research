#!/usr/bin/env python3
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, render_template, request, send_from_directory

app = Flask(__name__)

frame_lock = threading.Lock()
latest_jpeg = None

status_lock = threading.Lock()
camera_connected = False
motion_detected = False

BASE = Path(
    os.environ.get("MOTION_DATA_DIR", Path(__file__).resolve().parent / "monitor")
).expanduser().resolve()
BASE.mkdir(parents=True, exist_ok=True)

MAX_STORAGE = 5 * 1024**3  # 5 GiB
STORAGE_RESERVE = 100 * 1024**2  # 100 MiB


def directory_size(directory):
    total = 0
    for file in directory.rglob("*"):
        if file.is_file():
            try:
                total += file.stat().st_size
            except FileNotFoundError:
                pass
    return total


def enforce_storage_limit(exclude=None):
    """Remove oldest completed events until below the reserve threshold."""
    target = MAX_STORAGE - STORAGE_RESERVE
    total = directory_size(BASE)

    if total < target:
        return total

    events = sorted(
        (p for p in BASE.iterdir()
         if p.is_dir() and p != exclude),
        key=lambda p: p.name
    )

    for event in events:
        if total < target:
            break

        size = directory_size(event)
        shutil.rmtree(event)
        total -= size
        print(
            f"Storage cleanup: removed {event.name} "
            f"({size / 1024**2:.1f} MiB)",
            flush=True
        )

    return total

CAMERA = int(os.environ.get("MOTION_CAMERA", "0"))
WIDTH = 640
HEIGHT = 480
FPS = 20.0

WARMUP_SECONDS = 5
MIN_AREA = 1000
START_FRAMES = 2
END_SECONDS = 1.0
RECONNECT_SECONDS = 2

running = True



def open_camera():
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


mog = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=False
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

cap = None
writer = None
event_dir = None
event_active = False
best_area = 0
best_frame = None
last_motion = 0
motion_streak = 0
warmup_start = None
event_count = 0
last_storage_check = [0]
last_stream_frame = 0.0
STREAM_INTERVAL = 1.0 / 15.0


def finish_event():
    global writer, event_active, best_frame, best_area

    if writer is not None:
        writer.release()
        writer = None

    if event_active and best_frame is not None:
        path = event_dir / "peak.jpg"
        cv2.imwrite(str(path), best_frame)
        print(
            f"[{datetime.now():%H:%M:%S}] "
            f"Event finished; peak area={best_area:.0f}; "
            f"saved={event_dir}",
            flush=True
        )

    event_active = False
    best_frame = None
    best_area = 0



@app.route("/")
def dashboard():
    events = []
    for folder in sorted(BASE.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        if not (folder / "peak.jpg").is_file():
            continue

        events.append({
            "id": folder.name,
            "timestamp": folder.name[:15],
            "has_video": (folder / "motion.avi").is_file(),
        })

        if len(events) >= 12:
            break

    return render_template("dashboard.html", events=events)


@app.route("/api/events")
def api_events():
    try:
        offset = max(0, int(request.args.get("offset", 0)))
        limit = min(48, max(1, int(request.args.get("limit", 12))))
    except ValueError:
        return {"error": "Invalid pagination parameters"}, 400

    folders = sorted(
        (
            folder for folder in BASE.iterdir()
            if folder.is_dir() and (folder / "peak.jpg").is_file()
        ),
        key=lambda folder: folder.name,
        reverse=True
    )

    batch = folders[offset:offset + limit]

    return {
        "events": [
            {
                "id": folder.name,
                "timestamp": folder.name[:15],
                "has_video": (folder / "motion.avi").is_file(),
            }
            for folder in batch
        ],
        "total": len(folders),
        "has_more": offset + limit < len(folders),
    }


@app.route("/events/<event_id>/<filename>")
def event_file(event_id, filename):
    if not event_id.isdigit() and not all(
        c.isdigit() or c == "_" for c in event_id
    ):
        return "Invalid event ID", 400

    if filename not in ("peak.jpg", "motion.avi"):
        return "Invalid filename", 400

    return send_from_directory(
        str(BASE / event_id),
        filename,
        as_attachment=(filename == "motion.avi")
    )


@app.route("/api/status")
def api_status():
    with status_lock:
        connected = camera_connected
        motion = motion_detected

    # Read recording state separately from the camera thread.
    recording = event_active and writer is not None

    # Count only completed events with a saved snapshot.
    completed_events = sum(
        1 for folder in BASE.iterdir()
        if folder.is_dir() and (folder / "peak.jpg").is_file()
    )

    return {
        "camera_connected": connected,
        "motion_detected": motion,
        "recording": recording,
        "storage_gib": round(directory_size(BASE) / 1024**3, 3),
        "storage_limit_gib": 5,
        "events": completed_events,
    }


def stream_frames():
    while running:
        with frame_lock:
            frame = latest_jpeg

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(STREAM_INTERVAL)


@app.route("/stream")
def stream():
    return Response(
        stream_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def camera_worker():
    global latest_jpeg, last_stream_frame
    global cap, writer, event_dir, event_active
    global best_area, best_frame, last_motion, motion_streak
    global warmup_start, event_count, mog
    global camera_connected, motion_detected
    print("Motion monitor starting...", flush=True)
    print(f"Events directory: {BASE}", flush=True)

    # Enforce retention before opening the camera.
    initial_usage = enforce_storage_limit()
    print(
        f"Recording storage: {initial_usage / 1024**3:.3f} GiB "
        f"/ {MAX_STORAGE / 1024**3:.1f} GiB",
        flush=True
    )

    try:
        while running:
            if cap is None:
                cap = open_camera()

                if cap is None:
                    print("Camera unavailable; retrying...", flush=True)
                    time.sleep(RECONNECT_SECONDS)
                    continue

                print("Camera connected", flush=True)
                with status_lock:
                    camera_connected = True

                mog = cv2.createBackgroundSubtractorMOG2(
                    history=300,
                    varThreshold=25,
                    detectShadows=False
                )

                warmup_start = time.monotonic()
                motion_streak = 0

            ok, frame = cap.read()

            if not ok:
                print("Camera read failed; reconnecting...", flush=True)
                with status_lock:
                    camera_connected = False
                    motion_detected = False

                if event_active:
                    finish_event()

                cap.release()
                cap = None
                time.sleep(RECONNECT_SECONDS)
                continue

            now = time.monotonic()
            mask = mog.apply(frame)

            if now - warmup_start < WARMUP_SECONDS:
                continue

            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, kernel
            )
            mask = cv2.dilate(mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            significant = [
                c for c in contours
                if cv2.contourArea(c) >= MIN_AREA
            ]

            motion_area = sum(
                cv2.contourArea(c) for c in significant
            )

            has_motion = bool(significant)
            with status_lock:
                motion_detected = has_motion

            if has_motion:
                motion_streak += 1
            else:
                motion_streak = 0

            annotated = frame.copy()

            for contour in significant:
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

            if not event_active and motion_streak >= START_FRAMES:
                event_count += 1
                event_active = True
                last_motion = now
                best_area = 0
                best_frame = None

                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                used = enforce_storage_limit()
                if used >= MAX_STORAGE - STORAGE_RESERVE:
                    print(
                        "Storage reserve reached; cannot start new recording",
                        flush=True
                    )
                    event_active = False
                    motion_streak = 0
                    continue

                event_dir = BASE / timestamp
                event_dir.mkdir(parents=True, exist_ok=True)

                height, width = frame.shape[:2]

                writer = cv2.VideoWriter(
                    str(event_dir / "motion.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    FPS,
                    (width, height)
                )

                if not writer.isOpened():
                    print("Warning: video recording unavailable", flush=True)
                    writer.release()
                    writer = None

                print(
                    f"[{datetime.now():%H:%M:%S}] "
                    f"Motion event {event_count} started",
                    flush=True
                )

            if now - last_stream_frame >= STREAM_INTERVAL:
                # The final annotated image is published below,
                # after any motion-event overlay has been added.
                publish_stream_frame = True
                last_stream_frame = now
            else:
                publish_stream_frame = False

            if event_active:
                if has_motion:
                    last_motion = now

                cv2.putText(
                    annotated,
                    "MOTION DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

                if motion_area > best_area:
                    best_area = motion_area
                    best_frame = annotated.copy()

                if writer is not None:
                    writer.write(annotated)

                    # Check approximately once per second during recording.
                    if int(now) != int(last_storage_check[0]):
                        last_storage_check[0] = now
                        used = enforce_storage_limit(exclude=event_dir)

                        if used >= MAX_STORAGE - STORAGE_RESERVE:
                            print(
                                "Storage reserve reached; closing active recording",
                                flush=True
                            )
                            writer.release()
                            writer = None

                if now - last_motion >= END_SECONDS:
                    finish_event()

            if publish_stream_frame:
                encoded, jpeg = cv2.imencode(
                    ".jpg", annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                if encoded:
                    with frame_lock:
                        latest_jpeg = jpeg.tobytes()

    finally:
        if event_active:
            finish_event()

        if cap is not None:
            cap.release()

        with status_lock:
            camera_connected = False
            motion_detected = False
        print("Motion monitor stopped", flush=True)


if __name__ == "__main__":
    thread = threading.Thread(target=camera_worker, daemon=True)
    thread.start()
    try:
        app.run(host="0.0.0.0", port=int(os.environ.get("MOTION_PORT", "5000")), threaded=True, use_reloader=False)
    finally:
        running = False
        thread.join(timeout=5)
        print("Web monitor stopped", flush=True)
