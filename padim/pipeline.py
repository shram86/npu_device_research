"""Camera, inference, rendering and event workers; independent of Flask."""
from collections import deque
import ctypes
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np


LOG = logging.getLogger(__name__)
MODES = ("original", "heatmap", "side-by-side")



if __package__:
    from .model import NBBackend, ONNXBackend, Preprocessor, view_geometry
else:
    from model import NBBackend, ONNXBackend, Preprocessor, view_geometry


def load_config(filename):
    filename = Path(filename).resolve()
    config = json.loads(filename.read_text())
    config.setdefault("score_model","../models/network_binary_score.nb")
    config.setdefault("score_metadata","../models/score_meta.json")
    config.setdefault("scoring","cpu")
    config.setdefault("heatmap_visible_fraction",0.6)
    for key in ("score_model", "score_metadata", "model", "library", "onnx_model", "metadata", "statistics", "events_dir"):
        config[key] = str((filename.parent / config[key]).resolve())
    for key in ("width", "height", "camera_fps", "stream_fps", "camera_buffers", "persistence_frames"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["backend"] not in ("nb", "onnx"):
        raise ValueError("backend must be nb or onnx")
    for key in ("threshold", "heatmap_min", "heatmap_max", "heatmap_alpha", "stale_seconds"):
        if not np.isfinite(config[key]):
            raise ValueError(f"{key} must be finite")
    if config["stale_seconds"] <= 0 or not 0 <= config["heatmap_alpha"] <= 1:
        raise ValueError("Invalid stale_seconds or heatmap_alpha")
    if not np.isfinite(config["heatmap_visible_fraction"]) or not 0 <= config["heatmap_visible_fraction"] < 1:
        raise ValueError("heatmap_visible_fraction must be in [0,1)")
    if config["heatmap_max"] <= config["heatmap_min"] or not 1 <= config["jpeg_quality"] <= 100:
        raise ValueError("Invalid heatmap range or JPEG quality")
    if (config['map_width'],config['map_height']) != (64,64):
        raise ValueError('PaDiM statistics require 64x64 anomaly maps')
    if config['scoring'] not in ('cpu','npu'):raise ValueError('scoring must be cpu or npu')
    if config['backend']=='onnx' and config['scoring']=='npu':
        config['scoring']='cpu'
    config.setdefault('preprocessing','pil')
    if config['preprocessing'] not in ('pil','neon'):
        raise ValueError('preprocessing must be pil or neon')
    config.setdefault('crop_position',{'x':.5,'y':.5})
    config['roi'] = view_geometry(config['width'],config['height'],config['crop_position'])
    return config


class Alarm:
    def __init__(self, threshold, persistence):
        self.threshold = threshold
        self.persistence = persistence
        self.hits = 0
        self.latched = False

    def update(self, score):
        self.hits = min(self.persistence, self.hits + 1) if score >= self.threshold else 0
        triggered = not self.latched and self.hits >= self.persistence
        self.latched |= triggered
        return triggered

    def reset(self):
        self.hits = 0
        self.latched = False


def render_views(frame, anomaly_map, config):
    """Fixed heatmap range; never normalize by each frame's min/max."""
    roi = config["roi"]
    x, y, w, h = (roi[key] for key in ("x", "y", "width", "height"))
    original = frame.copy()
    cv2.rectangle(original, (x, y), (x+w-1, y+h-1), (80, 220, 160), 2)
    overlay = frame.copy()
    if anomaly_map is not None:
        scaled = np.clip((anomaly_map-config["heatmap_min"]) /
                         (config["heatmap_max"]-config["heatmap_min"]), 0, 1)
        scaled = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        cutoff = config.get("heatmap_visible_fraction", 0.6)
        visible = scaled >= cutoff
        # Only strong responses are painted; weaker areas retain original pixels.
        warm = np.clip((scaled-cutoff)/(1-cutoff), 0, 1)
        colors = np.empty((*scaled.shape, 3), dtype=np.uint8)
        colors[..., 0] = 0
        colors[..., 1] = np.uint8((1-warm)*255)
        colors[..., 2] = 255
        blended = cv2.addWeighted(frame[y:y+h, x:x+w],
                                 1-config["heatmap_alpha"], colors,
                                 config["heatmap_alpha"], 0)
        region = overlay[y:y+h, x:x+w]
        region[visible] = blended[visible]
    cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (80, 220, 160), 2)
    height, width = frame.shape[:2]
    left_width = width * 3 // 4
    side = np.zeros_like(frame)
    left_height = round(height * left_width / width)
    side[:left_height, :left_width] = cv2.resize(original, (left_width, left_height))
    crop_width = width - left_width
    crop_height = min(height, round(h * crop_width / w))
    side[:crop_height, left_width:] = cv2.resize(overlay[y:y+h, x:x+w], (crop_width, crop_height))
    return {"original": original, "heatmap": overlay, "side-by-side": side}


class Monitor:
    def __init__(self, config, backend_factory=None):
        self.config = config
        self.backend_factory = backend_factory or (NBBackend if config["backend"] == "nb" else ONNXBackend)
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.threads = []
        self.raw = None
        self.raw_time = 0.0
        self.raw_sequence = 0
        self.capture_generation = 0
        self.connected = False
        self.result = None
        self.result_sequence = 0
        self.jpegs = {}
        self.jpeg_sequence = 0
        self.camera_times = deque(maxlen=120)
        self.inference_times = deque(maxlen=60)
        self.alarm = Alarm(config["threshold"], config["persistence_frames"])
        self.model_error = None
        self.inference_error = None
        self.camera_error = "Waiting for camera"
        self.render_error = None
        self.inference_ms = None
        self.latency_ms = {}
        self.capture_to_result_ms = None
        self.event_queue = queue.Queue(maxsize=2)
        self.event_error = None
        self.dropped_events = 0
        self.last_event = None
        self.preprocessor = None

    def start(self):
        # Load and use vendor runtime on the same thread; never in an HTTP handler.
        worker = threading.Thread(target=self._infer, name="padim-inference", daemon=True)
        self.threads.append(worker)
        worker.start()
        if not self.ready.wait(60) or self.model_error:
            self.stop()
            raise RuntimeError(self.model_error or "Model loading timed out")
        for target, name in ((self._capture, "camera"), (self._render, "render"), (self._record, "recorder")):
            worker = threading.Thread(target=target, name=f"padim-{name}", daemon=True)
            self.threads.append(worker)
            worker.start()

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=5)

    def publish_camera(self, frame):
        if frame.shape != (self.config["height"], self.config["width"], 3):
            raise ValueError(f"Camera must provide {self.config['width']}x{self.config['height']} color frames")
        now = time.monotonic()
        with self.condition:
            self.raw = frame.copy()
            self.raw_time = now
            self.raw_sequence += 1
            self.connected = True
            self.camera_error = None
            self.camera_times.append(now)
            self.condition.notify_all()

    def _disconnect(self, message):
        with self.condition:
            self.connected = False
            self.raw = None
            self.capture_generation += 1
            self.result = None
            self.result_sequence += 1
            self.camera_error = message
            self.alarm.hits = 0
            self.condition.notify_all()

    def _capture(self):
        cfg = self.config
        while not self.stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(cfg["camera"], cv2.CAP_V4L2)
                if not cap.isOpened():
                    raise OSError(f"Cannot open {cfg['camera']}")
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
                cap.set(cv2.CAP_PROP_FPS, cfg["camera_fps"])
                if not cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.get("camera_buffers", 4)):
                    LOG.warning("Camera backend rejected requested capture buffer count")
                LOG.info("Camera buffers: requested=%d reported=%.0f",
                         cfg.get("camera_buffers", 4), cap.get(cv2.CAP_PROP_BUFFERSIZE))
                code = int(cap.get(cv2.CAP_PROP_FOURCC))
                LOG.info("Camera negotiated %dx%d @ %.1f FPS (%s)",
                         cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                         cap.get(cv2.CAP_PROP_FPS), ''.join(chr((code >> (8*i)) & 255) for i in range(4)))
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok:
                        raise OSError("Camera read failed")
                    self.publish_camera(frame)
            except (OSError, ValueError, cv2.error) as error:
                LOG.warning("%s; reconnecting in two seconds", error)
                self._disconnect(str(error))
            finally:
                if cap is not None:
                    cap.release()
            self.stop_event.wait(2)
        self._disconnect("Stopped")

    def _infer(self):
        backend = None
        previous = 0
        try:
            backend = self.backend_factory(self.config)
            mode=self.config.get('preprocessing','pil')
            library=backend.lib if isinstance(backend,NBBackend) else self.config['library']
            self.preprocessor=Preprocessor(mode,library if mode=='neon' else None,self.config.get('crop_position'))
            LOG.info("PaDiM preprocessing: %s shortest-side resize + configured crop, RGB ImageNet normalization",mode)
            LOG.info("PaDiM distance scoring: %s",self.config.get("scoring","cpu"))
            self.ready.set()
            while not self.stop_event.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.stop_event.is_set() or
                                            (self.raw is not None and self.raw_sequence != previous), timeout=1)
                    if self.stop_event.is_set():
                        break
                    if self.raw is None or self.raw_sequence == previous:
                        continue
                    frame, captured, previous = self.raw, self.raw_time, self.raw_sequence
                    generation = self.capture_generation
                if time.monotonic()-captured > self.config["stale_seconds"]:
                    continue
                try:
                    started = time.monotonic()
                    image = self.preprocessor(frame, self.config["roi"])
                    preprocessed = time.monotonic()
                    anomaly_map, score = backend.run(image)
                    backend_done = time.monotonic()
                    if anomaly_map.shape != (self.config["map_height"], self.config["map_width"]) or not np.isfinite(anomaly_map).all() or not np.isfinite(score):
                        raise ValueError("Invalid model output")
                    if not np.isclose(score, float(anomaly_map.max()), rtol=1e-4, atol=1e-5):
                        raise ValueError("Backend score differs from maximum calibrated anomaly map")
                    now = time.monotonic()
                    latency = dict(preprocess_ms=(preprocessed-started)*1000,
                                   backend_ms=(backend_done-preprocessed)*1000,
                                   validation_ms=(now-backend_done)*1000)
                    native = getattr(backend, "last_timings", {})
                    if all(key in native for key in ("input_ms", "network_ms", "output_ms")):
                        latency.update(native)
                        latency["bridge_ms"] = max(0.0, latency["backend_ms"]-sum(native.values()))
                    with self.condition:
                        # Never let a delayed result trigger an alarm after camera loss.
                        if not self.connected or generation != self.capture_generation:
                            self.alarm.hits = 0
                            continue
                        self.latency_ms = latency
                        if now-captured > self.config["stale_seconds"]:
                            self.inference_ms = (now-started)*1000
                            self.capture_to_result_ms = (now-captured)*1000
                            self.inference_error = (
                                f"Inference result too old: {now-captured:.2f}s exceeds "
                                f"stale_seconds={self.config['stale_seconds']:g}. "
                                "For CPU diagnostics increase --stale-seconds.")
                            LOG.warning("%s", self.inference_error)
                            self.alarm.hits = 0
                            continue
                        if self.result is not None and captured-self.result[3] > self.config["stale_seconds"]:
                            self.alarm.hits = 0
                        self.result = (frame, anomaly_map.copy(), float(score), captured, now)
                        self.result_sequence += 1
                        self.inference_ms = (now-started)*1000
                        self.capture_to_result_ms = (now-captured)*1000
                        self.inference_times.append(now)
                        self.inference_error = None
                        triggered = self.alarm.update(score)
                        self.condition.notify_all()
                    if triggered and self.config["record_events"]:
                        try:
                            self.event_queue.put_nowait((frame, anomaly_map.copy(), float(score)))
                        except queue.Full:
                            with self.condition:
                                self.dropped_events += 1
                except (RuntimeError, ValueError, OSError, cv2.error) as error:
                    LOG.exception("Inference failed")
                    with self.condition:
                        self.inference_error = str(error)
                        self.alarm.hits = 0
                    self.stop_event.wait(.5)
        except Exception as error:
            LOG.exception("Model worker failed")
            with self.condition:
                self.model_error = str(error)
            self.ready.set()
        finally:
            if backend is not None:
                backend.close()

    def _render(self):
        previous = None
        while not self.stop_event.is_set():
            started = time.monotonic()
            with self.condition:
                result = self.result
                healthy = self._healthy(time.monotonic())
                # Healthy overlays use the exact inference frame, never a newer frame.
                if healthy and result is not None:
                    frame, anomaly_map = result[:2]
                    key = (True, self.result_sequence)
                else:
                    frame, anomaly_map = self.raw, None
                    key = (False, self.raw_sequence, self.connected)
                    if frame is None:
                        frame = np.zeros((self.config["height"], self.config["width"], 3), dtype=np.uint8)
            if frame is not None and key != previous:
                try:
                    views = render_views(frame, anomaly_map, self.config)
                    if not healthy:
                        for view in views.values():
                            cv2.putText(view, "INFERENCE UNAVAILABLE", (24, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, .8, (60, 140, 255), 2)
                    encoded = {}
                    for mode, view in views.items():
                        ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, self.config["jpeg_quality"]])
                        if not ok:
                            raise ValueError("JPEG encoding failed")
                        encoded[mode] = jpeg.tobytes()
                    with self.condition:
                        self.jpegs = encoded
                        self.jpeg_sequence += 1
                        self.render_error = None
                        self.condition.notify_all()
                    previous = key
                except (ValueError, cv2.error) as error:
                    with self.condition:
                        self.render_error = str(error)
                    LOG.exception("Render failed")
            self.stop_event.wait(max(.001, 1/self.config["stream_fps"] - (time.monotonic()-started)))

    def _record(self):
        while not self.stop_event.is_set() or not self.event_queue.empty():
            try:
                frame, anomaly_map, score = self.event_queue.get(timeout=.5)
            except queue.Empty:
                continue
            try:
                event = Path(self.config["events_dir"]) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                event.mkdir(parents=True, exist_ok=False)
                views = render_views(frame, anomaly_map, self.config)
                for name, image in (("original.jpg", frame), ("heatmap.jpg", views["heatmap"])):
                    if not cv2.imwrite(str(event/name), image):
                        raise OSError(f"Could not write {name}")
                np.save(event/"anomaly_map.npy", anomaly_map)
                (event/"event.json.tmp").write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                    "score": score, "threshold": self.config["threshold"],
                    "threshold_calibrated": self.config["threshold_calibrated"],
                    "roi": self.config["roi"], "crop_position": self.config.get("crop_position"), "backend": self.config["backend"],
                    "model": self.config["model"] if self.config["backend"] == "nb" else self.config["onnx_model"],
                    "statistics": self.config["statistics"], "preprocessing": self.config.get("preprocessing","pil"), "scoring": self.config.get("scoring","cpu"), "score_model": self.config.get("score_model"), "smoothing_sigma": 2}, indent=2))
                (event/"event.json.tmp").replace(event/"event.json")
                with self.condition:
                    self.last_event, self.event_error = event.name, None
            except (OSError, ValueError, cv2.error) as error:
                LOG.exception("Event recording failed")
                with self.condition:
                    self.event_error = str(error)
            finally:
                self.event_queue.task_done()

    def _healthy(self, now):
        return bool(self.connected and now-self.raw_time <= self.config["stale_seconds"] and
                    self.result is not None and now-self.result[3] <= self.config["stale_seconds"] and
                    not self.model_error and not self.inference_error)

    @staticmethod
    def _fps(times, now, stale):
        recent = [t for t in times if now-t <= stale]
        return (len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 and recent[-1]>recent[0] else 0.0

    def status(self):
        now = time.monotonic()
        with self.condition:
            healthy = self._healthy(now)
            connected = self.connected and now-self.raw_time <= self.config["stale_seconds"]
            state = "UNKNOWN" if not healthy else ("ANOMALY" if self.alarm.latched else
                                                   "PENDING" if self.alarm.hits else "NORMAL")
            return {"status": state, "camera_connected": connected, "inference_ready": healthy,
                    "scoring": self.config.get("scoring","cpu"),
                    "score": self.result[2] if self.result else None,
                    "score_age_seconds": now-self.result[3] if self.result else None,
                    "threshold": self.config["threshold"],
                    "threshold_calibrated": self.config["threshold_calibrated"],
                    "alarm": self.alarm.latched, "persistence_hits": self.alarm.hits,
                    "persistence_frames": self.config["persistence_frames"],
                    "camera_fps": self._fps(self.camera_times,now,self.config["stale_seconds"]),
                    "inference_fps": self._fps(self.inference_times,now,self.config["stale_seconds"]),
                    "latency_ms": dict(self.latency_ms),
                    "inference_ms": self.inference_ms, "capture_to_result_ms": self.capture_to_result_ms,
                    "preprocessing": self.config.get("preprocessing","pil"),
                    "backend": self.config["backend"], "width": self.config["width"], "height": self.config["height"],
                    "error": self.model_error or self.inference_error or (self.camera_error if not connected else None),
                    "render_error": self.render_error, "event_error": self.event_error,
                    "last_event": self.last_event, "dropped_events": self.dropped_events}

    def reset_alarm(self):
        with self.condition:
            self.alarm.reset()
        return self.status()

    def stream(self, mode):
        previous = -1
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.stop_event.is_set() or
                                        (mode in self.jpegs and previous != self.jpeg_sequence), timeout=1)
                if self.stop_event.is_set():
                    return
                if mode not in self.jpegs or previous == self.jpeg_sequence:
                    continue
                jpeg, previous = self.jpegs[mode], self.jpeg_sequence
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
