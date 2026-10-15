import cv2
import os
import time
from pathlib import Path

OUT = Path(os.environ.get("MOTION_DATA_DIR", Path(__file__).resolve().parent / "events")).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

mog = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=25,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

MIN_AREA = 1000
START_FRAMES = 2
END_SECONDS = 1.0
WARMUP_SECONDS = 5
RUN_SECONDS = 30

start = time.monotonic()
event_active = False
event_id = 0
motion_streak = 0
last_motion = 0
best_area = 0
best_frame = None
writer = None

print("Learning background for 5 seconds.")
print("Then walk through the camera view.")
print("Running for 30 seconds...")

def finish_event():
    global event_active, writer, best_frame, best_area

    if writer is not None:
        writer.release()
        writer = None

    if best_frame is not None:
        filename = OUT / f"event_{event_id:03d}_peak.jpg"
        cv2.imwrite(str(filename), best_frame)
        print(f"Event {event_id}: peak area={best_area:.0f}, saved {filename}")

    event_active = False
    best_frame = None
    best_area = 0

try:
    while time.monotonic() - start < RUN_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.monotonic()
        mask = mog.apply(frame)

        if now - start < WARMUP_SECONDS:
            continue

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        significant = [
            c for c in contours if cv2.contourArea(c) >= MIN_AREA
        ]
        motion_area = sum(cv2.contourArea(c) for c in significant)
        has_motion = bool(significant)

        annotated = frame.copy()

        for c in significant:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(
                annotated, (x, y), (x + w, y + h),
                (0, 0, 255), 2
            )

        if has_motion:
            motion_streak += 1
        else:
            motion_streak = 0

        if not event_active and motion_streak >= START_FRAMES:
            event_id += 1
            event_active = True
            best_area = 0
            best_frame = None

            h, w = frame.shape[:2]
            filename = OUT / f"event_{event_id:03d}.avi"

            writer = cv2.VideoWriter(
                str(filename),
                cv2.VideoWriter_fourcc(*"MJPG"),
                20.0,
                (w, h)
            )

            if not writer.isOpened():
                print("WARNING: Video writer unavailable")
                writer = None

            print(f"Event {event_id} started")

        if event_active:
            if has_motion:
                last_motion = now

            cv2.putText(
                annotated, "MOTION EVENT",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 255), 2
            )

            if motion_area > best_area:
                best_area = motion_area
                best_frame = annotated.copy()

            if writer is not None:
                writer.write(annotated)

            if now - last_motion >= END_SECONDS:
                finish_event()

finally:
    if event_active:
        finish_event()
    cap.release()

print(f"Finished. Events detected: {event_id}")
print(f"Files saved in: {OUT}")
